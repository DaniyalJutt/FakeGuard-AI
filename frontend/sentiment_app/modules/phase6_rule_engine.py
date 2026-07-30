"""
PHASE 6: RULE-BASED SCORES
Calculates rule-based quality scores: repetition, unique word ratio, punctuation density
"""

import pandas as pd
import re


def calculate_rule_scores(text):
    """Calculate rule-based quality scores"""
    if not text or pd.isna(text):
        return 0.0, 0.0, 0.0
    
    text = str(text)
    words = text.split()
    
    if not words:
        return 0.0, 0.0, 0.0
    
    # Repetition score
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    max_repeat = max(word_counts.values())
    repetition_score = max_repeat / len(words) if len(words) > 0 else 0
    
    # Unique word ratio
    unique_ratio = len(set(words)) / len(words) if len(words) > 0 else 0
    
    # Punctuation density
    punct_density = len(re.findall(r'[!?.,;:]', text)) / (len(text) + 1)
    
    return repetition_score, unique_ratio, punct_density


def apply_rule_engine(df):
    """
    Apply rule-based scoring
    
    Args:
        df: DataFrame with 'text_cleaned' column
        
    Returns:
        df: DataFrame with rule-based score columns added
    """
    print("\n" + "="*80)
    print("PHASE 6: RULE-BASED SCORES")
    print("="*80)
    
    # Apply
    rule_results = df['text_cleaned'].apply(calculate_rule_scores)
    df['score_repetition'] = [r[0] for r in rule_results]
    df['stat_unique_word_ratio'] = [r[1] for r in rule_results]
    df['stat_punctuation_density'] = [r[2] for r in rule_results]
    
    # Final rule score (weighted average)
    df['rule_score_final'] = (
        0.4 * df['score_repetition'] +
        0.3 * (1 - df['stat_unique_word_ratio']) +
        0.3 * df['stat_punctuation_density']
    )
    
    print(f"✅ Rule-based scores calculated")
    print(f"\n📊 Rule score statistics:")
    print(df['rule_score_final'].describe())
    
    return df

