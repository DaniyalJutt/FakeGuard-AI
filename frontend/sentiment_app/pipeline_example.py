"""
MAIN PIPELINE EXAMPLE
Demonstrates how to use all phase modules together
"""

import warnings
warnings.filterwarnings('ignore')

# Import modules
from utils.file_handler import load_csv, export_results
from modules import (
    apply_language_detection,
    apply_text_preprocessing,
    apply_metadata_extraction,
    apply_transformer_features,
    apply_lexicon_engine,
    apply_rule_engine,
    assemble_features,
    load_models,
    make_predictions,
    generate_all_visualizations
)


def run_full_pipeline(file_path):
    """
    Run the complete sentiment analysis pipeline
    
    Args:
        file_path: Path to input CSV file
    """
    # Phase 0: Load data
    df = load_csv(file_path)
    
    # Phase 1: Language detection
    df = apply_language_detection(df)
    
    # Phase 2: Text preprocessing
    df = apply_text_preprocessing(df)
    
    # Phase 3: Metadata extraction
    df = apply_metadata_extraction(df)
    
    # Phase 4: Transformer features
    df = apply_transformer_features(df)
    
    # Phase 5: Lexicon engine
    df = apply_lexicon_engine(df)
    
    # Phase 6: Rule engine
    df = apply_rule_engine(df)
    
    # Phase 7: Feature assembly
    X_exp2, feature_columns, df = assemble_features(df)
    
    # Phase 8: Load models and make predictions
    model, tfidf_vectorizer, svd, scaler_exp1, scaler_exp2, le = load_models()
    df = make_predictions(df, X_exp2, model, tfidf_vectorizer, svd, scaler_exp1, scaler_exp2, le)
    
    # Phase 9: Visualization
    generate_all_visualizations(df)
    
    # Phase 10: Export results
    summary = export_results(df)
    
    return df, summary


if __name__ == "__main__":
    # Example usage
    file_path = input("Enter CSV file path: ")
    df, summary = run_full_pipeline(file_path)
    print("\n✅ Pipeline execution complete!")

