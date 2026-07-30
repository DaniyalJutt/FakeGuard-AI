"""
Scorer Module - Adapted from Phase 2
Computes rule-based fake review scores
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========================
# CONFIGURATION
# ========================

SCORING_WEIGHTS = {
    # Text Style & Content
    'grammatically_perfect': 0.5,
    'short_review': 0.5,
    'promo_tone': 0.5,
    'high_emoji_ratio': 0.5,
    
    # Duplicate / Cross-Posting
    'exact_duplicate_across_users': 1.25,
    'near_duplicate_cluster': 0.50,
    
    # Behavior / Profile
    'generic_username': 0.75,
    'burst_posting': 1.0,
    'always_extreme_ratings': 0.5,
    
    # Rating vs Text
    'rating_text_mismatch': 1.25,
    
    # Cross-App
    'same_text_on_multiple_apps': 1.25,
    'suspicious_time_pattern': 0.5
}

# ========================
# HELPER FUNCTIONS
# ========================

def check_grammatically_perfect(text, token_count):
    """Heuristic: Long fluent sentences with proper punctuation"""
    if not text or token_count < 5:
        return 0
    
    has_capital_start = text[0].isupper() if text else False
    has_end_punctuation = text[-1] in ['.', '!', '?'] if text else False
    
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    
    if has_capital_start and has_end_punctuation and avg_sentence_length >= 8:
        return 1
    return 0

def detect_near_duplicates(df, threshold=0.85):
    """Detect near-duplicate reviews using TF-IDF + cosine similarity"""
    texts = df['text_cleaned'].fillna('').tolist()
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Find clusters of similar reviews
    cluster_sizes = []
    for i in range(len(df)):
        similar_indices = np.where(similarity_matrix[i] > threshold)[0]
        similar_indices = similar_indices[similar_indices != i]
        cluster_sizes.append(len(similar_indices) + 1)
    
    df['near_duplicate_cluster_size'] = cluster_sizes
    return df

def detect_suspicious_time_patterns(df):
    """Detect users posting at unusual hours"""
    unusual_hours = [2, 3, 4, 5]  # 2 AM to 5 AM
    
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    
    user_unusual_counts = df[df['hour'].isin(unusual_hours)].groupby('username').size()
    user_total_counts = df.groupby('username').size()
    
    suspicious_users = (user_unusual_counts / user_total_counts > 0.5)
    suspicious_users = suspicious_users[suspicious_users].index.tolist()
    
    df['suspicious_time_pattern'] = df['username'].isin(suspicious_users).astype(int)
    return df

def detect_cross_app_duplicates(df):
    """Detect same text across multiple apps"""
    text_app_counts = df.groupby('text_cleaned')['app_name'].nunique()
    text_multi_app = text_app_counts[text_app_counts >= 2].index.tolist()
    
    df['same_text_on_multiple_apps'] = df['text_cleaned'].isin(text_multi_app).astype(int)
    return df

# ========================
# SCORING FUNCTIONS
# ========================

def compute_scores(df):
    """
    Compute rule-based fake review scores
    
    Args:
        df: DataFrame with preprocessed reviews
    
    Returns:
        DataFrame with fake_score column added
    """
    
    # ============================================
    # TEXT STYLE & CONTENT SCORES
    # ============================================
    
    # Grammatically perfect
    df['score_grammatically_perfect'] = df.apply(
        lambda row: check_grammatically_perfect(row['text_cleaned'], row['token_count']) * SCORING_WEIGHTS['grammatically_perfect'],
        axis=1
    )
    
    # Short review
    df['score_short_review'] = (df['token_count'] <= 3).astype(float) * SCORING_WEIGHTS['short_review']
    
    # Promo tone
    df['score_promo_tone'] = df['contains_promo_words'] * SCORING_WEIGHTS['promo_tone']
    
    # High emoji ratio
    df['emoji_ratio'] = df['num_emojis'] / df['token_count'].replace(0, 1)
    df['score_high_emoji_ratio'] = (df['emoji_ratio'] > 0.2).astype(float) * SCORING_WEIGHTS['high_emoji_ratio']
    
    # ============================================
    # DUPLICATE / CROSS-POSTING SCORES
    # ============================================
    
    # Exact duplicate across users
    df['score_exact_duplicate'] = (df['same_text_count'] >= 3).astype(float) * SCORING_WEIGHTS['exact_duplicate_across_users']
    
    # Near duplicates
    df = detect_near_duplicates(df, threshold=0.85)
    df['score_near_duplicate'] = (df['near_duplicate_cluster_size'] >= 5).astype(float) * SCORING_WEIGHTS['near_duplicate_cluster']
    
    # ============================================
    # BEHAVIOR / PROFILE SCORES
    # ============================================
    
    # Generic username
    df['score_generic_username'] = df['generic_username'] * SCORING_WEIGHTS['generic_username']
    
    # Burst posting
    df['score_burst_posting'] = (
        (df['is_burst'] == 1) & (df['user_total_reviews'] >= 3)
    ).astype(float) * SCORING_WEIGHTS['burst_posting']
    
    # Always extreme ratings
    df['score_extreme_ratings'] = (
        ((df['user_avg_rating'] >= 4.8) | (df['user_avg_rating'] <= 1.2)) & 
        (df['user_rating_std'] < 0.5)
    ).astype(float) * SCORING_WEIGHTS['always_extreme_ratings']
    
    # ============================================
    # RATING VS TEXT SCORES
    # ============================================
    
    df['score_rating_mismatch'] = df['rating_text_mismatch'] * SCORING_WEIGHTS['rating_text_mismatch']
    
    # ============================================
    # CROSS-APP / TIME PATTERN SCORES
    # ============================================
    
    # Cross-app duplicates
    df = detect_cross_app_duplicates(df)
    df['score_cross_app'] = df['same_text_on_multiple_apps'] * SCORING_WEIGHTS['same_text_on_multiple_apps']
    
    # Suspicious time patterns
    df = detect_suspicious_time_patterns(df)
    df['score_suspicious_time'] = df['suspicious_time_pattern'] * SCORING_WEIGHTS['suspicious_time_pattern']
    
    # ============================================
    # COMPUTE FINAL FAKE SCORE
    # ============================================
    
    score_columns = [col for col in df.columns if col.startswith('score_')]
    df['fake_score'] = df[score_columns].sum(axis=1)
    
    return df

