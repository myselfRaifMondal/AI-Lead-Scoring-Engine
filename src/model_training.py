import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score
from dotenv import load_dotenv

load_dotenv()

# Load Data
DATA_PATH = os.getenv("DATA_PATH", "./data/leads.csv")
print(f"Loading data from: {DATA_PATH}")
data = pd.read_csv(DATA_PATH)
print(f"Data shape: {data.shape}")
print(f"Data columns: {list(data.columns)}")

# Define target first (before feature engineering)
TARGET = 'is_high_intent'
target = data[TARGET].astype(int)  # Ensure target is 0/1

# Remove target from features before feature engineering
feature_data = data.drop(columns=[TARGET])

# Feature Engineering
from feature_engineering import FeatureEngineer
feature_engineer = FeatureEngineer()
features = feature_engineer.create_all_features(feature_data)

# Ensure target is binary 0/1
target = (target > 0).astype(int)

print(f"Target distribution: {target.value_counts().to_dict()}")
print(f"Target unique values: {sorted(target.unique())}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train model
print(f"Training data shape: {X_train.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Training target unique values: {sorted(y_train.unique())}")

try:
    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, objective='binary:logistic')
    model.fit(X_train, y_train)
    print("Model training completed successfully!")
except Exception as e:
    print(f"Error during model training: {e}")
    raise

# Predict and evaluate
preds = model.predict(X_test)
preds_proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, preds_proba)
precision = precision_score(y_test, preds)

print(f"ROC AUC: {roc_auc:.2f}")
print(f"Precision: {precision:.2f}")

# Save model
model_path = 'models/xgboost_lead_scoring.model'
model.save_model(model_path)
print(f"Model saved to {model_path}")

# Feature importances
feature_importances = model.feature_importances_
feature_names = features.columns
feature_importance_explanation = feature_engineer.get_feature_importance_explanation(feature_names, feature_importances)

for feature, info in feature_importance_explanation.items():
    print(f"Feature: {feature}, Importance: {info['importance']}, Explanation: {info['explanation']}")
