"""
PHASE 1: LANGUAGE DETECTION
Detects language of input text: en, ur, hi, roman_urdu, or unknown
"""

import pandas as pd
import re
from langdetect import detect, LangDetectException


def detect_language(text):
    """Detect language: en, ur, hi, roman_urdu, or unknown"""
    if not text or pd.isna(text) or len(str(text).strip()) < 3:
        return 'unknown'
    
    try:
        text = str(text)
        # Check for Urdu script
        urdu_pattern = r'[\u0600-\u06FF]'
        has_urdu_script = bool(re.search(urdu_pattern, text))
        
        # Check for Roman Urdu markers
        roman_urdu_words = ['hai', 'nahi', 'kya', 'acha', 'bahut', 'bohat']
        has_roman_urdu = any(word in text.lower() for word in roman_urdu_words)
        
        detected = detect(text)
        
        if has_urdu_script:
            return 'ur'
        elif has_roman_urdu:
            return 'roman_urdu'
        elif detected in ['en', 'ur', 'hi']:
            return detected
        else:
            return 'other'
    except:
        return 'unknown'


def apply_language_detection(df, verbose=True):
    """
    Apply language detection to dataframe and filter supported languages
    
    Args:
        df: DataFrame with 'text' column
        verbose: Whether to print progress (for Streamlit compatibility)
        
    Returns:
        df_filtered: DataFrame with only supported languages
    """
    if verbose:
        print("="*80)
        print("PHASE 1: LANGUAGE DETECTION")
        print("="*80)
    
    # Apply language detection
    df['language'] = df['text'].apply(detect_language)
    
    # Filter: Keep only supported languages
    supported_langs = ['en', 'ur', 'hi', 'roman_urdu']
    df_filtered = df[df['language'].isin(supported_langs)].copy()
    
    if verbose:
        print(f"✅ Original reviews: {len(df)}")
        print(f"✅ After filtering: {len(df_filtered)}")
        print(f"\n📊 Language Distribution:")
        print(df_filtered['language'].value_counts())
    
    return df_filtered.reset_index(drop=True)

