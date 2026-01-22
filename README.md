# Hybrid-LLM-Classifier

I've built a sophisticated Hybrid Classification System that combines LLM-based classification with traditional machine learning! Here's what it includes:
Features
1. LLM Classification (Gemini 2.0 Flash)

Uses your Gemini API for powerful semantic understanding
Provides confidence scores and reasoning
Handles nuanced text classification

2. Traditional ML (TF-IDF + Cosine Similarity)

Implements TF-IDF vectorization from scratch
Uses cosine similarity for matching
Fast and interpretable

3. Hybrid Approach

Combines both methods intelligently
Uses agreement between models for high confidence
Falls back to strongest method when they disagree
Adjustable confidence threshold

4. Multiple Modes

Single Classification: Classify one text with detailed analysis
Batch Classification: Process multiple texts at once
Training Data: Add examples to improve traditional ML component

5. Advanced Capabilities

Export results as JSON
Customizable categories
Visual confidence indicators
Side-by-side comparison of LLM vs traditional ML results

How It Works
The hybrid system works by:

Getting predictions from both LLM and traditional ML (if training data exists)
If both agree with high confidence → uses combined score
If LLM has high confidence → trusts LLM
Otherwise → uses weighted decision based on confidence levels

This gives you the best of both worlds: LLM's understanding of context + traditional ML's speed and consistency!
Try adding some training examples in the "Training Data" tab to see the hybrid approach in action!
