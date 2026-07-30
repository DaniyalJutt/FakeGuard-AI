"""
PHASE 7: FEATURE ASSEMBLY
Combines all extracted features into final feature matrix
"""

import pandas as pd
import numpy as np


def assemble_features(df):
    """
    Assemble all features into final feature matrix
    
    Args:
        df: DataFrame with all extracted features
        
    Returns:
        X_exp2: Feature matrix (numpy array)
        feature_columns: List of feature column names
    """
    # Only print if not in Streamlit (to avoid repeated printing)
    import sys
    is_streamlit = 'streamlit' in sys.modules
    
    if not is_streamlit:
        print("\n" + "="*80)
        print("PHASE 7: FEATURE ASSEMBLY")
        print("="*80)
    
    # Add emotion confidence (simplified)
    df['emotion_confidence'] = df['transformer_confidence'] * 0.8
    
    # Ensure rating column exists
    if 'rating' not in df.columns:
        df['rating'] = 3.0  # Default neutral rating
    
    # Convert boolean to int
    df['sarcasm_detected'] = df['sarcasm_detected'].astype(int)
    
    # Define final feature set (19 features)
    feature_columns = [
        'transformer_sentiment_score',
        'transformer_confidence',
        'transformer_emb_mean',
        'transformer_emb_std',
        'lexicon_sentiment_polarity',
        'lexicon_sentiment_adjusted',
        'lexicon_sarcasm_confidence',
        'emotion_confidence',
        'sarcasm_detected',
        'sarcasm_confidence',
        'meta_text_length',
        'meta_word_count',
        'meta_emoji_count',
        'meta_uppercase_ratio',
        'rule_score_final',
        'score_repetition',
        'stat_unique_word_ratio',
        'stat_punctuation_density',
        'rating'
    ]
    
    # Extract features
    X_exp2 = df[feature_columns].copy()
    
    # Handle any missing values
    X_exp2 = X_exp2.fillna(0).astype(np.float64)
    
    if not is_streamlit:
        print(f"✅ Feature assembly complete")
        print(f"✅ Feature matrix shape: {X_exp2.shape}")
        print(f"✅ Features: {len(feature_columns)}")
    
    return X_exp2, feature_columns, df

