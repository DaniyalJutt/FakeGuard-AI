"""
PHASE 4: TRANSFORMER FEATURES
Generates transformer-based sentiment features (simplified version)
"""

import pandas as pd
import numpy as np


def simple_transformer_score(text):
    """Simple sentiment scoring based on text features"""
    if not text or pd.isna(text):
        return 0.0, 0.5, 0.0, 0.0
    
    text = str(text).lower()
    
    # Positive/negative word counts (simple heuristic)
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'acha', 'achha']
    negative_words = ['bad', 'terrible', 'worst', 'hate', 'poor', 'bekar', 'ganda']
    
    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)
    
    # Calculate score
    total = pos_count + neg_count + 1
    score = (pos_count - neg_count) / total
    confidence = min(total / 10, 1.0)
    
    # Simple embeddings (mean and std of word lengths)
    words = text.split()
    if words:
        emb_mean = np.mean([len(w) for w in words])
        emb_std = np.std([len(w) for w in words])
    else:
        emb_mean = 0.0
        emb_std = 0.0
    
    return score, confidence, emb_mean, emb_std


def apply_transformer_features(df):
    """
    Apply transformer feature extraction
    
    Args:
        df: DataFrame with 'text_cleaned' column
        
    Returns:
        df: DataFrame with transformer feature columns added
    """
    print("\n" + "="*80)
    print("PHASE 4: TRANSFORMER FEATURES")
    print("="*80)
    
    # Apply
    transformer_results = df['text_cleaned'].apply(simple_transformer_score)
    df['transformer_sentiment_score'] = [r[0] for r in transformer_results]
    df['transformer_confidence'] = [r[1] for r in transformer_results]
    df['transformer_emb_mean'] = [r[2] for r in transformer_results]
    df['transformer_emb_std'] = [r[3] for r in transformer_results]
    
    print(f"✅ Transformer features generated (simplified)")
    print(f"\n📊 Sentiment score range: [{df['transformer_sentiment_score'].min():.2f}, {df['transformer_sentiment_score'].max():.2f}]")
    
    return df

