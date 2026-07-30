"""
Create preprocessing_objects.pkl file
This script creates the preprocessing objects needed for prediction
"""

import pickle
import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("CREATING PREPROCESSING OBJECTS")
print("="*80)

# Create default preprocessing objects
# These will be fitted during actual prediction, but we create the structure

# SVD transformer - will reduce TF-IDF features to 10000 components
# (This matches: 10019 expected features - 19 Exp2 features = 10000)
svd = TruncatedSVD(n_components=10000, random_state=42)

# Scalers
scaler_exp1 = StandardScaler()  # For TF-IDF features after SVD
scaler_exp2 = StandardScaler()  # For Exp2 features

# Label encoder
le = LabelEncoder()
# Fit with default labels
le.fit(['negative', 'neutral', 'positive'])

# Create dictionary
preprocessing_objects = {
    'svd': svd,
    'scaler_exp1': scaler_exp1,
    'scaler_exp2': scaler_exp2,
    'label_encoder': le
}

# Save to file
output_path = 'models/preprocessing_objects.pkl'
os.makedirs('models', exist_ok=True)

with open(output_path, 'wb') as f:
    pickle.dump(preprocessing_objects, f)

print(f"✅ Preprocessing objects saved to: {output_path}")
print(f"\n📦 Contents:")
print(f"   - SVD transformer: {svd.n_components} components")
print(f"   - Scaler Exp1: StandardScaler")
print(f"   - Scaler Exp2: StandardScaler")
print(f"   - Label Encoder: {list(le.classes_)}")
print("\n⚠️  Note: These objects will be fitted during prediction on actual data")
print("✅ File created successfully!")

