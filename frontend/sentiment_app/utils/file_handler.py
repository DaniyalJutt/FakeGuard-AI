"""
FILE HANDLER
Handles CSV file upload, download, and export operations
"""

import pandas as pd
import pickle
import os


def load_csv(file_path):
    """
    Load CSV file and validate required columns
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        df: DataFrame with validated columns
    """
    print("="*80)
    print("PHASE 0: INPUT LAYER")
    print("="*80)
    
    df = pd.read_csv(file_path)
    
    print(f"✅ File loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"✅ Columns: {list(df.columns)}")
    
    # Validate required columns
    if 'text' not in df.columns and 'text_original' not in df.columns:
        raise ValueError("❌ Error: 'text' or 'text_original' column required!")
    
    # Standardize column name
    if 'text_original' in df.columns:
        df['text'] = df['text_original']
    
    print("✅ Text column validated")
    
    # Preview
    print("\n📝 Preview:")
    print(df.head())
    
    return df


def export_results(df, output_dir='data/exports'):
    """
    Export processed results to CSV and summary statistics
    
    Args:
        df: DataFrame with predictions
        output_dir: Output directory path
        
    Returns:
        dict: Summary statistics
    """
    print("\n" + "="*80)
    print("PHASE 10: EXPORT LAYER")
    print("="*80)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export 1: Full processed dataset
    output_full = os.path.join(output_dir, 'processed_dataset_full.csv')
    df.to_csv(output_full, index=False)
    print(f"✅ Full dataset exported: {output_full}")
    
    # Export 2: Predictions only
    output_predictions = os.path.join(output_dir, 'predictions_only.csv')
    if 'sentiment_final' in df.columns and 'prediction_confidence' in df.columns:
        df[['text', 'sentiment_final', 'prediction_confidence']].to_csv(
            output_predictions, index=False
        )
        print(f"✅ Predictions exported: {output_predictions}")
    
    # Export 3: Summary statistics
    summary = {
        'total_reviews': len(df),
        'sentiment_distribution': df['sentiment_final'].value_counts().to_dict() if 'sentiment_final' in df.columns else {},
        'avg_confidence': df['prediction_confidence'].mean() if 'prediction_confidence' in df.columns else 0.0,
        'sarcasm_rate': df['sarcasm_detected'].mean() if 'sarcasm_detected' in df.columns else 0.0,
        'language_distribution': df['language'].value_counts().to_dict() if 'language' in df.columns else {}
    }
    
    summary_path = os.path.join(output_dir, 'summary_stats.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"✅ Summary statistics saved: {summary_path}")
    
    # Print final summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    print(f"Total Reviews Processed: {summary['total_reviews']}")
    print(f"\nSentiment Distribution:")
    for sentiment, count in summary['sentiment_distribution'].items():
        pct = (count / summary['total_reviews']) * 100 if summary['total_reviews'] > 0 else 0
        print(f"  {sentiment}: {count} ({pct:.1f}%)")
    print(f"\nAverage Confidence: {summary['avg_confidence']:.2%}")
    print(f"Sarcasm Rate: {summary['sarcasm_rate']*100:.1f}%")
    print("\n" + "="*80)
    print("✅ ALL PHASES COMPLETE!")
    print("="*80)
    
    return summary

