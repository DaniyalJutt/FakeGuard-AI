"""
Sentiment Analysis System - Modules Package
Phase-wise processing modules for sentiment analysis
"""

from .phase1_language_detection import apply_language_detection, detect_language
from .phase2_text_preprocessing import apply_text_preprocessing, deep_clean_text
from .phase3_metadata_extraction import apply_metadata_extraction, extract_metadata
from .phase4_transformer_features import apply_transformer_features, simple_transformer_score
from .phase5_lexicon_engine import apply_lexicon_engine, lexicon_sentiment
from .phase6_rule_engine import apply_rule_engine, calculate_rule_scores
from .phase7_feature_assembly import assemble_features
from .phase8_prediction import load_models, make_predictions
from .phase9_visualization import (
    generate_all_visualizations,
    plot_sentiment_distribution,
    plot_rating_sentiment_heatmap,
    plot_wordclouds,
    plot_confidence_distribution,
    plot_sarcasm_analytics
)

# Streamlit visualization functions
from .phase9_visualization_streamlit import (
    create_sentiment_pie_chart,
    create_sentiment_bar_chart,
    create_rating_sentiment_heatmap,
    create_confidence_distribution,
    create_sarcasm_analytics,
    create_language_distribution,
    create_tsne_visualization,
    create_emotion_distribution,
    create_metadata_analysis,
    create_wordcloud_image,
)

__all__ = [
    'apply_language_detection',
    'detect_language',
    'apply_text_preprocessing',
    'deep_clean_text',
    'apply_metadata_extraction',
    'extract_metadata',
    'apply_transformer_features',
    'simple_transformer_score',
    'apply_lexicon_engine',
    'lexicon_sentiment',
    'apply_rule_engine',
    'calculate_rule_scores',
    'assemble_features',
    'load_models',
    'make_predictions',
    'generate_all_visualizations',
    'plot_sentiment_distribution',
    'plot_rating_sentiment_heatmap',
    'plot_wordclouds',
    'plot_confidence_distribution',
    'plot_sarcasm_analytics',
    # Streamlit visualizations
    'create_sentiment_pie_chart',
    'create_sentiment_bar_chart',
    'create_rating_sentiment_heatmap',
    'create_confidence_distribution',
    'create_sarcasm_analytics',
    'create_language_distribution',
    'create_tsne_visualization',
    'create_emotion_distribution',
    'create_metadata_analysis',
    'create_wordcloud_image',
]

