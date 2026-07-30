"""
PHASE 2: DEEP CLEANING
Cleans text: removes HTML, URLs, normalizes text, extracts emoji count
"""

import pandas as pd
import re
import emoji


def deep_clean_text(text):
    """Clean text: remove HTML, URLs, normalize"""
    if pd.isna(text):
        return "", 0
    
    text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Extract emoji count before removal
    emoji_count = emoji.emoji_count(text)
    
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    
    # Normalize repeated characters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Lowercase
    text = text.lower()
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text, emoji_count


def apply_text_preprocessing(df):
    """
    Apply deep cleaning to text column
    
    Args:
        df: DataFrame with 'text' column
        
    Returns:
        df: DataFrame with 'text_cleaned' and 'emoji_count' columns added
    """
    print("\n" + "="*80)
    print("PHASE 2: DEEP CLEANING")
    print("="*80)
    
    # Apply cleaning
    results = df['text'].apply(lambda x: deep_clean_text(x))
    df['text_cleaned'] = [r[0] for r in results]
    df['emoji_count'] = [r[1] for r in results]
    
    print(f"✅ Text cleaned for {len(df)} reviews")
    print(f"✅ Average emojis per review: {df['emoji_count'].mean():.2f}")
    print(f"\n📝 Sample cleaned text:")
    print(df[['text', 'text_cleaned']].head(3))
    
    return df

