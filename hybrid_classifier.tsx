import React, { useState } from 'react';
import { AlertCircle, Brain, Zap, BarChart3, Upload, Download, Info } from 'lucide-react';

const HybridClassifier = () => {
  const [activeTab, setActiveTab] = useState('single');
  const [text, setText] = useState('');
  const [batchTexts, setBatchTexts] = useState('');
  const [categories, setCategories] = useState('positive, negative, neutral');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [trainingData, setTrainingData] = useState([]);
  const [useHybrid, setUseHybrid] = useState(true);
  const [confidence, setConfidence] = useState(0.7);
  const [apiKey, setApiKey] = useState('API KEY');
  const [useCors, setUseCors] = useState(true);

  const GEMINI_MODEL = 'gemini model';

  // Simple TF-IDF vectorizer for traditional ML
  const calculateTFIDF = (docs) => {
    const wordCounts = docs.map(doc => {
      const words = doc.toLowerCase().match(/\b\w+\b/g) || [];
      const counts = {};
      words.forEach(w => counts[w] = (counts[w] || 0) + 1);
      return counts;
    });

    const docFreq = {};
    wordCounts.forEach(counts => {
      Object.keys(counts).forEach(word => {
        docFreq[word] = (docFreq[word] || 0) + 1;
      });
    });

    return wordCounts.map(counts => {
      const tfidf = {};
      Object.keys(counts).forEach(word => {
        const tf = counts[word] / Object.values(counts).reduce((a, b) => a + b, 0);
        const idf = Math.log(docs.length / docFreq[word]);
        tfidf[word] = tf * idf;
      });
      return tfidf;
    });
  };

  // Cosine similarity for traditional classification
  const cosineSimilarity = (vec1, vec2) => {
    const allWords = new Set([...Object.keys(vec1), ...Object.keys(vec2)]);
    let dotProduct = 0;
    let mag1 = 0;
    let mag2 = 0;

    allWords.forEach(word => {
      const v1 = vec1[word] || 0;
      const v2 = vec2[word] || 0;
      dotProduct += v1 * v2;
      mag1 += v1 * v1;
      mag2 += v2 * v2;
    });

    return dotProduct / (Math.sqrt(mag1) * Math.sqrt(mag2) || 1);
  };

  const classifyWithGemini = async (inputText, cats) => {
    const categoryList = cats.split(',').map(c => c.trim());
    
    const prompt = `You are a precise text classifier. Classify the following text into exactly ONE of these categories: ${categoryList.join(', ')}.

Text to classify: "${inputText}"

Respond ONLY with a JSON object in this exact format:
{
  "category": "the_chosen_category",
  "confidence": 0.95,
  "reasoning": "brief explanation"
}`;

    try {
      const apiUrl = useCors 
        ? `https://corsproxy.io/?https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${apiKey}`
        : `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${apiKey}`;

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          contents: [{
            parts: [{ text: prompt }]
          }],
          generationConfig: {
            temperature: 0.1,
            maxOutputTokens: 500
          }
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMsg = `API Error: ${response.status}`;
        try {
          const errorData = JSON.parse(errorText);
          errorMsg = errorData.error?.message || errorMsg;
        } catch (e) {
          errorMsg = errorText.substring(0, 200);
        }
        throw new Error(errorMsg);
      }

      const data = await response.json();
      
      if (!data.candidates || !data.candidates[0]) {
        throw new Error('No response from API');
      }
      
      const textResponse = data.candidates[0].content.parts[0].text;
      
      // Extract JSON from response
      const jsonMatch = textResponse.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }
      
      throw new Error('Invalid response format');
    } catch (error) {
      console.error('Gemini API Error:', error);
      throw error;
    }
  };

  const classifyTraditional = (inputText) => {
    if (trainingData.length === 0) {
      return null;
    }

    const allTexts = [...trainingData.map(d => d.text), inputText];
    const tfidfVectors = calculateTFIDF(allTexts);
    const inputVector = tfidfVectors[tfidfVectors.length - 1];

    let bestMatch = null;
    let bestScore = -1;

    trainingData.forEach((data, idx) => {
      const score = cosineSimilarity(tfidfVectors[idx], inputVector);
      if (score > bestScore) {
        bestScore = score;
        bestMatch = data.category;
      }
    });

    return {
      category: bestMatch,
      confidence: bestScore,
      method: 'TF-IDF'
    };
  };

  // Advanced rule-based classification with sentiment analysis
  const classifyRuleBased = (inputText, cats) => {
    const categoryList = cats.split(',').map(c => c.trim());
    const lowerText = inputText.toLowerCase();

    // Enhanced keyword dictionary with weights
    const keywords = {
      'positive': {
        strong: ['excellent', 'outstanding', 'superb', 'brilliant', 'perfect', 'exceptional', 'phenomenal', 'magnificent', 'marvelous', 'spectacular'],
        medium: ['good', 'great', 'nice', 'wonderful', 'amazing', 'fantastic', 'awesome', 'beautiful', 'lovely', 'delightful', 'pleased', 'happy', 'enjoy', 'love', 'best', 'better'],
        weak: ['okay', 'fine', 'decent', 'acceptable', 'satisfactory', 'like', 'well']
      },
      'negative': {
        strong: ['terrible', 'horrible', 'awful', 'disgusting', 'atrocious', 'abysmal', 'dreadful', 'pathetic', 'appalling', 'horrendous'],
        medium: ['bad', 'poor', 'disappointing', 'worst', 'hate', 'dislike', 'unfortunate', 'sad', 'unhappy', 'upset', 'angry', 'fail', 'failed', 'problem', 'issue'],
        weak: ['not great', 'not good', 'could be better', 'mediocre', 'lacking', 'subpar']
      },
      'neutral': {
        strong: ['neutral', 'objective', 'unbiased', 'impartial', 'factual'],
        medium: ['average', 'normal', 'standard', 'typical', 'ordinary', 'moderate', 'medium'],
        weak: ['okay', 'fine', 'alright']
      }
    };

    // Negation handling
    const negations = ['not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 'hardly', 'barely', 'scarcely', "n't", 'dont', "don't"];
    
    // Check for negations
    const words = lowerText.split(/\s+/);
    let negationIndices = [];
    words.forEach((word, idx) => {
      if (negations.some(neg => word.includes(neg))) {
        negationIndices.push(idx);
      }
    });

    let scores = {};
    categoryList.forEach(cat => {
      scores[cat] = 0;
      const catLower = cat.toLowerCase();
      
      if (keywords[catLower]) {
        // Strong keywords (weight: 3)
        keywords[catLower].strong?.forEach(keyword => {
          const keywordIdx = words.findIndex(w => w.includes(keyword));
          if (keywordIdx !== -1) {
            // Check if negated (within 3 words before)
            const isNegated = negationIndices.some(ni => ni < keywordIdx && keywordIdx - ni <= 3);
            scores[cat] += isNegated ? -2 : 3;
          }
        });

        // Medium keywords (weight: 2)
        keywords[catLower].medium?.forEach(keyword => {
          const keywordIdx = words.findIndex(w => w.includes(keyword));
          if (keywordIdx !== -1) {
            const isNegated = negationIndices.some(ni => ni < keywordIdx && keywordIdx - ni <= 3);
            scores[cat] += isNegated ? -1.5 : 2;
          }
        });

        // Weak keywords (weight: 1)
        keywords[catLower].weak?.forEach(keyword => {
          const keywordIdx = words.findIndex(w => w.includes(keyword));
          if (keywordIdx !== -1) {
            const isNegated = negationIndices.some(ni => ni < keywordIdx && keywordIdx - ni <= 3);
            scores[cat] += isNegated ? -0.5 : 1;
          }
        });
      }

      // Handle opposite categories (positive/negative flip)
      if (catLower === 'positive' && negationIndices.length > 0) {
        const negativeKeywords = [...(keywords['negative']?.strong || []), ...(keywords['negative']?.medium || [])];
        negativeKeywords.forEach(keyword => {
          const keywordIdx = words.findIndex(w => w.includes(keyword));
          if (keywordIdx !== -1) {
            const isNegated = negationIndices.some(ni => ni < keywordIdx && keywordIdx - ni <= 3);
            if (isNegated) scores[cat] += 1.5; // "not bad" = somewhat positive
          }
        });
      }
      
      if (catLower === 'negative' && negationIndices.length > 0) {
        const positiveKeywords = [...(keywords['positive']?.strong || []), ...(keywords['positive']?.medium || [])];
        positiveKeywords.forEach(keyword => {
          const keywordIdx = words.findIndex(w => w.includes(keyword));
          if (keywordIdx !== -1) {
            const isNegated = negationIndices.some(ni => ni < keywordIdx && keywordIdx - ni <= 3);
            if (isNegated) scores[cat] += 1.5; // "not good" = somewhat negative
          }
        });
      }
    });

    // Punctuation-based sentiment
    const exclamations = (lowerText.match(/!/g) || []).length;
    const questions = (lowerText.match(/\?/g) || []).length;
    
    if (scores['positive']) scores['positive'] += exclamations * 0.3;
    if (scores['neutral']) scores['neutral'] += questions * 0.2;

    // Normalize scores to be non-negative
    const minScore = Math.min(...Object.values(scores));
    if (minScore < 0) {
      Object.keys(scores).forEach(key => {
        scores[key] = scores[key] - minScore;
      });
    }

    const maxScore = Math.max(...Object.values(scores), 0);
    const totalScore = Object.values(scores).reduce((a, b) => a + b, 0);
    
    let bestCategory;
    if (maxScore === 0 || totalScore === 0) {
      // No keywords matched, default to first category or neutral
      bestCategory = categoryList.includes('neutral') ? 'neutral' : categoryList[0];
    } else {
      bestCategory = Object.keys(scores).find(k => scores[k] === maxScore) || categoryList[0];
    }
    
    const confidenceScore = maxScore > 0 ? Math.min((maxScore / (totalScore || 1)) * 0.85, 0.85) : 0.5;
    
    return {
      category: bestCategory,
      confidence: confidenceScore,
      reasoning: `Advanced rule-based classification (score: ${maxScore.toFixed(1)}, negations: ${negationIndices.length})`,
      method: 'Enhanced Rule-Based'
    };
  };

  const handleClassify = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      let llmResult = null;
      let traditionalResult = null;
      let ruleBasedResult = null;

      // Try LLM classification
      try {
        llmResult = await classifyWithGemini(text, categories);
      } catch (error) {
        console.warn('LLM classification failed, using fallback methods:', error);
        ruleBasedResult = classifyRuleBased(text, categories);
      }

      // Get traditional ML classification if we have training data
      if (useHybrid && trainingData.length > 0) {
        traditionalResult = classifyTraditional(text);
      }

      // Hybrid decision logic
      let finalResult;
      
      if (llmResult) {
        if (traditionalResult && useHybrid) {
          // If both agree and both are confident, use that
          if (llmResult.category === traditionalResult.category && 
              llmResult.confidence > confidence && 
              traditionalResult.confidence > 0.5) {
            finalResult = {
              ...llmResult,
              confidence: (llmResult.confidence + traditionalResult.confidence) / 2,
              method: 'Hybrid (Agreement)',
              details: { llm: llmResult, traditional: traditionalResult }
            };
          } else if (llmResult.confidence > confidence) {
            finalResult = {
              ...llmResult,
              method: 'LLM (High Confidence)',
              details: { llm: llmResult, traditional: traditionalResult }
            };
          } else {
            finalResult = {
              category: llmResult.confidence > traditionalResult.confidence ? 
                        llmResult.category : traditionalResult.category,
              confidence: Math.max(llmResult.confidence, traditionalResult.confidence),
              method: 'Hybrid (Weighted)',
              reasoning: llmResult.reasoning,
              details: { llm: llmResult, traditional: traditionalResult }
            };
          }
        } else {
          finalResult = {
            ...llmResult,
            method: 'LLM Only',
            details: { llm: llmResult }
          };
        }
      } else if (traditionalResult) {
        finalResult = {
          ...traditionalResult,
          method: 'Traditional ML Only'
        };
      } else {
        finalResult = ruleBasedResult;
      }

      setResult(finalResult);
    } catch (error) {
      setResult({
        error: true,
        message: error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const handleBatchClassify = async () => {
    if (!batchTexts.trim()) return;

    setLoading(true);
    const texts = batchTexts.split('\n').filter(t => t.trim());
    const results = [];

    try {
      for (const t of texts) {
        try {
          const llmResult = await classifyWithGemini(t, categories);
          results.push({ text: t, ...llmResult });
        } catch (error) {
          const ruleResult = classifyRuleBased(t, categories);
          results.push({ text: t, ...ruleResult, fallback: true });
        }
      }

      setResult({
        batch: true,
        results: results,
        summary: {
          total: results.length,
          avgConfidence: (results.reduce((sum, r) => sum + r.confidence, 0) / results.length).toFixed(2)
        }
      });
    } catch (error) {
      setResult({ error: true, message: error.message });
    } finally {
      setLoading(false);
    }
  };

  const addTrainingExample = () => {
    const example = prompt('Enter training text:');
    if (!example) return;
    
    const category = prompt(`Enter category (${categories}):`)?.trim();
    if (!category) return;

    setTrainingData([...trainingData, { text: example, category }]);
  };

  const exportResults = () => {
    const dataStr = JSON.stringify(result, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'classification_results.json';
    a.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 p-6">
      <div className="max-w-5xl mx-auto">
        <div className="bg-white rounded-xl shadow-lg p-8 mb-6">
          <div className="flex items-center gap-3 mb-6">
            <Brain className="w-8 h-8 text-purple-600" />
            <h1 className="text-3xl font-bold text-gray-800">Hybrid LLM Classifier</h1>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6 flex gap-3">
            <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-blue-800">
              <strong>Note:</strong> If you get CORS errors, toggle the "Use CORS Proxy" option below. The system includes rule-based and ML fallbacks for when API calls fail.
            </div>
          </div>

          <div className="flex gap-2 mb-6">
            <button
              onClick={() => setActiveTab('single')}
              className={`px-4 py-2 rounded-lg font-medium transition ${
                activeTab === 'single' 
                  ? 'bg-purple-600 text-white' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Single Classification
            </button>
            <button
              onClick={() => setActiveTab('batch')}
              className={`px-4 py-2 rounded-lg font-medium transition ${
                activeTab === 'batch' 
                  ? 'bg-purple-600 text-white' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Batch Classification
            </button>
            <button
              onClick={() => setActiveTab('train')}
              className={`px-4 py-2 rounded-lg font-medium transition ${
                activeTab === 'train' 
                  ? 'bg-purple-600 text-white' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Training Data
            </button>
            <button
              onClick={() => setActiveTab('settings')}
              className={`px-4 py-2 rounded-lg font-medium transition ${
                activeTab === 'settings' 
                  ? 'bg-purple-600 text-white' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Settings
            </button>
          </div>

          {activeTab === 'settings' && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Gemini API Key
                </label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
              <div>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={useCors}
                    onChange={(e) => setUseCors(e.target.checked)}
                    className="w-4 h-4 text-purple-600"
                  />
                  <span className="text-sm text-gray-700">Use CORS Proxy (corsproxy.io)</span>
                </label>
                <p className="text-xs text-gray-500 mt-1 ml-6">
                  Enable this if you get "Failed to fetch" errors due to CORS restrictions
                </p>
              </div>
            </div>
          )}

          {activeTab !== 'settings' && (
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Categories (comma-separated)
              </label>
              <input
                type="text"
                value={categories}
                onChange={(e) => setCategories(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder="positive, negative, neutral"
              />
            </div>
          )}

          {activeTab === 'single' && (
            <>
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Text to Classify
                </label>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  rows="4"
                  placeholder="Enter text to classify..."
                />
              </div>

              <div className="flex items-center gap-4 mb-6">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={useHybrid}
                    onChange={(e) => setUseHybrid(e.target.checked)}
                    className="w-4 h-4 text-purple-600"
                  />
                  <span className="text-sm text-gray-700">Use Hybrid Classification</span>
                </label>
                <div className="flex items-center gap-2">
                  <label className="text-sm text-gray-700">Confidence Threshold:</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={confidence}
                    onChange={(e) => setConfidence(parseFloat(e.target.value))}
                    className="w-32"
                  />
                  <span className="text-sm font-medium text-gray-700">{confidence.toFixed(2)}</span>
                </div>
              </div>

              <button
                onClick={handleClassify}
                disabled={loading || !text.trim()}
                className="w-full bg-purple-600 text-white py-3 rounded-lg font-medium hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent" />
                    Classifying...
                  </>
                ) : (
                  <>
                    <Zap className="w-5 h-5" />
                    Classify Text
                  </>
                )}
              </button>
            </>
          )}

          {activeTab === 'batch' && (
            <>
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Batch Texts (one per line)
                </label>
                <textarea
                  value={batchTexts}
                  onChange={(e) => setBatchTexts(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  rows="8"
                  placeholder="Enter multiple texts, one per line..."
                />
              </div>

              <button
                onClick={handleBatchClassify}
                disabled={loading || !batchTexts.trim()}
                className="w-full bg-purple-600 text-white py-3 rounded-lg font-medium hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent" />
                    Processing Batch...
                  </>
                ) : (
                  <>
                    <BarChart3 className="w-5 h-5" />
                    Classify Batch
                  </>
                )}
              </button>
            </>
          )}

          {activeTab === 'train' && (
            <div>
              <div className="mb-4">
                <button
                  onClick={addTrainingExample}
                  className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition flex items-center gap-2"
                >
                  <Upload className="w-4 h-4" />
                  Add Training Example
                </button>
              </div>

              {trainingData.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  No training data yet. Add examples to enable hybrid classification.
                </div>
              ) : (
                <div className="space-y-2">
                  {trainingData.map((item, idx) => (
                    <div key={idx} className="p-3 bg-gray-50 rounded-lg flex justify-between items-start">
                      <div className="flex-1">
                        <div className="text-sm text-gray-700">{item.text}</div>
                        <div className="text-xs text-purple-600 font-medium mt-1">
                          Category: {item.category}
                        </div>
                      </div>
                      <button
                        onClick={() => setTrainingData(trainingData.filter((_, i) => i !== idx))}
                        className="text-red-500 hover:text-red-700 text-sm"
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {result && !result.error && (
          <div className="bg-white rounded-xl shadow-lg p-8">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-gray-800">Results</h2>
              <button
                onClick={exportResults}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
              >
                <Download className="w-4 h-4" />
                Export
              </button>
            </div>

            {result.batch ? (
              <div>
                <div className="mb-4 p-4 bg-purple-50 rounded-lg">
                  <div className="text-sm text-gray-600">
                    Total: {result.summary.total} | Avg Confidence: {result.summary.avgConfidence}
                  </div>
                </div>
                <div className="space-y-3">
                  {result.results.map((r, idx) => (
                    <div key={idx} className="p-4 border border-gray-200 rounded-lg">
                      <div className="text-sm text-gray-700 mb-2">{r.text}</div>
                      <div className="flex items-center gap-4">
                        <span className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm font-medium">
                          {r.category}
                        </span>
                        <span className="text-sm text-gray-600">
                          {(r.confidence * 100).toFixed(1)}% confident
                        </span>
                        {r.fallback && (
                          <span className="text-xs text-orange-600">
                            (Fallback method)
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div>
                <div className="mb-6">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="text-3xl font-bold text-purple-600">{result.category}</span>
                    <span className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm">
                      {result.method}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-purple-600 h-2 rounded-full transition-all"
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium text-gray-700">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {result.reasoning && (
                  <div className="mb-4 p-4 bg-blue-50 rounded-lg">
                    <div className="text-sm font-medium text-gray-700 mb-1">Reasoning:</div>
                    <div className="text-sm text-gray-600">{result.reasoning}</div>
                  </div>
                )}

                {result.details && (
                  <div className="grid grid-cols-2 gap-4">
                    {result.details.llm && (
                      <div className="p-4 bg-purple-50 rounded-lg">
                        <div className="text-sm font-medium text-purple-800 mb-2">LLM Result</div>
                        <div className="text-sm text-gray-700">
                          Category: {result.details.llm.category}
                        </div>
                        <div className="text-sm text-gray-700">
                          Confidence: {(result.details.llm.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    )}
                    {result.details.traditional && (
                      <div className="p-4 bg-green-50 rounded-lg">
                        <div className="text-sm font-medium text-green-800 mb-2">TF-IDF Result</div>
                        <div className="text-sm text-gray-700">
                          Category: {result.details.traditional.category}
                        </div>
                        <div className="text-sm text-gray-700">
                          Similarity: {(result.details.traditional.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {result && result.error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-6 flex items-start gap-3">
            <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <div className="font-medium text-red-800">Error</div>
              <div className="text-sm text-red-700">{result.message}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default HybridClassifier;
