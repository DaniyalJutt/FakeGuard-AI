"""
PHASE 5: LEXICON ENGINE
Calculates lexicon-based sentiment and sarcasm detection
"""

import pandas as pd


# Lexicon definitions
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'awesome', 'wonderful',
    'fantastic', 'best', 'love', 'perfect', 'acha', 'achha', 'zabardast'
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'horrible', 'worst', 'poor', 'hate',
    'disappointing', 'useless', 'bekar', 'ganda', 'kharab'
}


def lexicon_sentiment(text):
    """Calculate lexicon-based sentiment"""
    if not text or pd.isna(text):
        return 0.0, 0.0, False, 0.0
    
    text = str(text).lower()
    words = set(text.split())
    
    pos_count = len(words.intersection(POSITIVE_WORDS))
    neg_count = len(words.intersection(NEGATIVE_WORDS))
    
    # Calculate polarity
    total = pos_count + neg_count
    if total > 0:
        polarity = (pos_count - neg_count) / total
    else:
        polarity = 0.0
    
    # Adjusted (with boost)
    adjusted = polarity * 1.2 if abs(polarity) > 0.5 else polarity
    
    # Sarcasm detection (simple: positive words + negative context)
    sarcasm_detected = (pos_count > 0 and neg_count > 0)
    sarcasm_confidence = min(pos_count * neg_count / 10, 1.0)
    
    return polarity, adjusted, sarcasm_detected, sarcasm_confidence


def apply_lexicon_engine(df):
    """
    Apply lexicon-based sentiment analysis
    
    Args:
        df: DataFrame with 'text_cleaned' column
        
    Returns:
        df: DataFrame with lexicon sentiment columns added
    """
    print("\n" + "="*80)
    print("PHASE 5: LEXICON ENGINE")
    print("="*80)
    
    # Apply
    lexicon_results = df['text_cleaned'].apply(lexicon_sentiment)
    df['lexicon_sentiment_polarity'] = [r[0] for r in lexicon_results]
    df['lexicon_sentiment_adjusted'] = [r[1] for r in lexicon_results]
    df['sarcasm_detected'] = [r[2] for r in lexicon_results]
    df['lexicon_sarcasm_confidence'] = [r[3] for r in lexicon_results]
    df['sarcasm_confidence'] = df['lexicon_sarcasm_confidence']
    
    print(f"✅ Lexicon sentiment calculated")
    print(f"✅ Sarcasm detected in {df['sarcasm_detected'].sum()} reviews ({df['sarcasm_detected'].mean()*100:.1f}%)")
    
    return df

