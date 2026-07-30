"""
CONFIGURATION
System configuration and constants
"""

# Supported languages
SUPPORTED_LANGUAGES = ['en', 'ur', 'hi', 'roman_urdu']

# Model paths
MODEL_PATHS = {
    'best_model': 'models/best_model_fused.pkl',
    'preprocessing': 'models/preprocessing_objects.pkl',
    'tfidf_vectorizer': 'models/tfidf_vectorizer.pkl',
    'bow_vectorizer': 'models/bow_vectorizer.pkl',
    'transformer_info': 'models/transformer_model_info.pkl',
    'sentiment_summary': 'models/sentiment_summary_balanced.pkl'
}

# Feature weights for fusion
FEATURE_FUSION_WEIGHTS = {
    'tfidf': 0.2,
    'exp2': 0.8
}

# Output directories
OUTPUT_DIRS = {
    'uploads': 'data/uploads',
    'processed': 'data/processed',
    'exports': 'data/exports'
}

# Sentiment labels
SENTIMENT_LABELS = {
    0: 'negative',
    1: 'neutral',
    2: 'positive'
}

# Lexicon words (can be expanded)
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'awesome', 'wonderful',
    'fantastic', 'best', 'love', 'perfect', 'acha', 'achha', 'zabardast'
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'horrible', 'worst', 'poor', 'hate',
    'disappointing', 'useless', 'bekar', 'ganda', 'kharab'
}

# Roman Urdu markers
ROMAN_URDU_MARKERS = ['hai', 'nahi', 'kya', 'acha', 'bahut', 'bohat']

