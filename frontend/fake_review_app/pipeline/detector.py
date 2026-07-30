"""
Detector Module - Adapted from Phase 7
Main fake review detection pipeline using ensemble (Rule-based + LightGBM)
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ========================
# CONFIGURATION
# ========================

# Decision thresholds
THRESHOLD_AUTO_REMOVE = 3.5      # High precision
THRESHOLD_HUMAN_REVIEW = 2.5     # Balanced
ENSEMBLE_WEIGHT_RULE = 0.6       # Rule-based weight
ENSEMBLE_WEIGHT_LGBM = 0.4       # LightGBM weight

# ========================
# FAKE REVIEW DETECTOR CLASS
# ========================

class FakeReviewDetector:
    """Production-ready fake review detection system"""
    
    def __init__(self, models_dir='models'):
        """
        Initialize and load all models
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        self.lgbm_model = None
        self.tfidf = None
        self.feature_names = None
        self._load_models()
    
    def _load_models(self):
        """Load all trained models"""
        try:
            # Get absolute path
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_path = os.path.join(base_path, self.models_dir)
            
            # Load LightGBM model
            lgbm_path = os.path.join(models_path, 'lgbm_model.txt')
            if not os.path.exists(lgbm_path):
                raise FileNotFoundError(f"LightGBM model not found at {lgbm_path}")
            self.lgbm_model = lgb.Booster(model_file=lgbm_path)
            
            # Load TF-IDF vectorizer
            tfidf_path = os.path.join(models_path, 'tfidf_vectorizer.pkl')
            if not os.path.exists(tfidf_path):
                raise FileNotFoundError(f"TF-IDF vectorizer not found at {tfidf_path}")
            self.tfidf = joblib.load(tfidf_path)
            
            # Load feature names
            features_path = os.path.join(models_path, 'feature_names.pkl')
            if not os.path.exists(features_path):
                raise FileNotFoundError(f"Feature names not found at {features_path}")
            self.feature_names = joblib.load(features_path)
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found! {e}")
        except Exception as e:
            raise Exception(f"Error loading models: {e}")
    
    def extract_features(self, df):
        """Extract features for prediction"""
        
        # Metadata features
        metadata_features = [
            'token_count', 'char_count', 'num_emojis', 'num_urls',
            'num_uppercase_words', 'unique_word_ratio', 'sentiment_score',
            'user_total_reviews', 'time_diff_seconds', 'same_text_count',
            'user_avg_rating', 'user_rating_std', 'rating'
        ]
        
        available_features = [f for f in metadata_features if f in df.columns]
        X_meta = df[available_features].fillna(0)
        
        # TF-IDF features
        X_tfidf = self.tfidf.transform(df['text_cleaned'].fillna(''))
        X_tfidf_df = pd.DataFrame(
            X_tfidf.toarray(),
            columns=[f'tfidf_{i}' for i in range(X_tfidf.shape[1])]
        )
        
        # Combine
        X = pd.concat([X_meta.reset_index(drop=True), 
                       X_tfidf_df.reset_index(drop=True)], axis=1)
        
        # Ensure column order matches training
        X = X[self.feature_names]
        
        return X
    
    def predict_single(self, review_data):
        """Predict single review"""
        
        # Extract features
        X = self.extract_features(pd.DataFrame([review_data]))
        
        # Get predictions
        rule_score = review_data.get('fake_score', 0)
        lgbm_proba = self.lgbm_model.predict(X)[0]
        
        # Ensemble score
        ensemble_score = (ENSEMBLE_WEIGHT_RULE * rule_score + 
                         ENSEMBLE_WEIGHT_LGBM * lgbm_proba * 10)  # Scale LGBM to 0-10
        
        # Decision
        if ensemble_score >= THRESHOLD_AUTO_REMOVE:
            decision = "REMOVE"
            confidence = "high"
        elif ensemble_score >= THRESHOLD_HUMAN_REVIEW:
            decision = "REVIEW"
            confidence = "medium"
        else:
            decision = "CLEAN"
            confidence = "high"
        
        return {
            'rule_score': rule_score,
            'lgbm_proba': lgbm_proba,
            'ensemble_score': ensemble_score,
            'decision': decision,
            'confidence': confidence
        }
    
    def predict_batch(self, df):
        """Predict batch of reviews"""
        
        # Extract features
        X = self.extract_features(df)
        
        # Get predictions
        rule_scores = df['fake_score'].values
        lgbm_probas = self.lgbm_model.predict(X)
        
        # Ensemble scores
        ensemble_scores = (ENSEMBLE_WEIGHT_RULE * rule_scores + 
                          ENSEMBLE_WEIGHT_LGBM * lgbm_probas * 10)
        
        # Decisions
        decisions = np.where(
            ensemble_scores >= THRESHOLD_AUTO_REMOVE, 'REMOVE',
            np.where(ensemble_scores >= THRESHOLD_HUMAN_REVIEW, 'REVIEW', 'CLEAN')
        )
        
        confidences = np.where(
            (ensemble_scores >= THRESHOLD_AUTO_REMOVE) | (ensemble_scores < THRESHOLD_HUMAN_REVIEW),
            'high', 'medium'
        )
        
        # Add to dataframe
        df['rule_score'] = rule_scores
        df['lgbm_proba'] = lgbm_probas
        df['ensemble_score'] = ensemble_scores
        df['decision'] = decisions
        df['confidence'] = confidences
        df['processed_at'] = datetime.now()
        
        return df
    
    def process_reviews(self, df):
        """
        Main processing function: takes scored reviews and returns categorized results
        
        Args:
            df: DataFrame with fake_score column (from scorer)
        
        Returns:
            dict with 'clean', 'removed', 'review' DataFrames
        """
        # Run batch inference
        df_processed = self.predict_batch(df.copy())
        
        # Split into categories
        df_remove = df_processed[df_processed['decision'] == 'REMOVE'].copy()
        df_review = df_processed[df_processed['decision'] == 'REVIEW'].copy()
        df_clean = df_processed[df_processed['decision'] == 'CLEAN'].copy()
        
        return {
            'clean': df_clean,
            'removed': df_remove,
            'review': df_review,
            'all': df_processed
        }

# ========================
# HELPER FUNCTIONS
# ========================

def explain_decision(review_data, prediction):
    """Explain why a review was flagged/cleaned"""
    
    reasons = []
    
    # Rule-based reasons
    if review_data.get('same_text_count', 0) >= 3:
        reasons.append(f"Duplicate text ({review_data['same_text_count']} users)")
    
    if review_data.get('token_count', 0) <= 3:
        reasons.append("Very short review")
    
    if review_data.get('rating_text_mismatch', 0) == 1:
        reasons.append("Rating-sentiment mismatch")
    
    if review_data.get('user_avg_rating', 0) >= 4.8 and review_data.get('user_rating_std', 1) < 0.5:
        reasons.append("User always gives extreme ratings")
    
    if review_data.get('is_burst', 0) == 1:
        reasons.append("Burst posting detected")
    
    # Scores
    reasons.append(f"Rule score: {prediction['rule_score']:.2f}")
    reasons.append(f"ML probability: {prediction['lgbm_proba']:.3f}")
    reasons.append(f"Ensemble score: {prediction['ensemble_score']:.2f}")
    
    return " | ".join(reasons) if reasons else "No specific flags"

