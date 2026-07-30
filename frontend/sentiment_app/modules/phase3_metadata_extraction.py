"""
PHASE 3: TOKEN-LEVEL FEATURES
Extracts token-level metadata: text length, word count, punctuation density, etc.
"""

import pandas as pd
import re
import numpy as np


def extract_metadata(text):
    """Extract token-level metadata"""
    if not text or pd.isna(text):
        return {
            'text_length': 0,
            'word_count': 0,
            'punct_density': 0,
            'uppercase_ratio': 0
        }
    
    text = str(text)
    words = text.split()
    
    return {
        'text_length': len(text),
        'word_count': len(words),
        'punct_density': len(re.findall(r'[!?.,;:]', text)) / (len(text) + 1),
        'uppercase_ratio': sum(c.isupper() for c in text) / (len(text) + 1)
    }


def apply_metadata_extraction(df):
    """
    Apply metadata extraction to cleaned text
    
    Args:
        df: DataFrame with 'text_cleaned' and 'emoji_count' columns
        
    Returns:
        df: DataFrame with metadata columns added
    """
    print("\n" + "="*80)
    print("PHASE 3: TOKEN-LEVEL FEATURES")
    print("="*80)
    
    # Apply metadata extraction
    metadata = df['text_cleaned'].apply(extract_metadata)
    df['meta_text_length'] = [m['text_length'] for m in metadata]
    df['meta_word_count'] = [m['word_count'] for m in metadata]
    df['meta_punct_density'] = [m['punct_density'] for m in metadata]
    df['meta_uppercase_ratio'] = [m['uppercase_ratio'] for m in metadata]
    df['meta_emoji_count'] = df['emoji_count']
    
    print(f"✅ Metadata extracted")
    print(f"\n📊 Statistics:")
    print(df[['meta_text_length', 'meta_word_count', 'meta_emoji_count']].describe())
    
    return df

