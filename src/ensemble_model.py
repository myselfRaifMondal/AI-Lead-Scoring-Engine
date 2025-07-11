import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleLeadScorer:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.feature_engineer = None
        
    def initialize_models(self):
        """Initialize all models in the ensemble"""
        logger.info("Initializing ensemble models")
        
        # XGBoost model
        self.models['xgboost'] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # LightGBM model
        self.models['lightgbm'] = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Neural Network model
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42
        )
        
        # Initialize equal weights
        self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Train all models in the ensemble"""
        logger.info("Training ensemble models")
        
        performance_metrics = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Validate
            val_preds = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_preds)
            
            performance_metrics[name] = val_auc
            logger.info(f"{name} validation AUC: {val_auc:.4f}")
        
        # Update weights based on performance
        self._update_weights(performance_metrics)
        
        return performance_metrics
    
    def _update_weights(self, performance_metrics: Dict[str, float]):
        """Update model weights based on validation performance"""
        logger.info("Updating model weights based on performance")
        
        # Softmax weighting based on AUC scores
        scores = np.array(list(performance_metrics.values()))
        exp_scores = np.exp(scores * 5)  # Scale for more distinction
        weights = exp_scores / np.sum(exp_scores)
        
        for i, name in enumerate(self.models.keys()):
            self.weights[name] = weights[i]
            logger.info(f"{name} weight: {self.weights[name]:.4f}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        ensemble_probs = np.zeros((X.shape[0], 2))
        
        for name, model in self.models.items():
            model_probs = model.predict_proba(X)
            ensemble_probs += self.weights[name] * model_probs
        
        return ensemble_probs
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions (binary)"""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance from tree-based models"""
        feature_importance = {}
        
        # Get feature names
        feature_names = None
        if hasattr(self.models['xgboost'], 'feature_names_in_'):
            feature_names = self.models['xgboost'].feature_names_in_
        
        if feature_names is not None:
            # XGBoost importance
            xgb_importance = dict(zip(feature_names, self.models['xgboost'].feature_importances_))
            
            # LightGBM importance
            lgb_importance = dict(zip(feature_names, self.models['lightgbm'].feature_importances_))
            
            # Weighted average
            for feature in feature_names:
                feature_importance[feature] = (
                    self.weights['xgboost'] * xgb_importance.get(feature, 0) +
                    self.weights['lightgbm'] * lgb_importance.get(feature, 0)
                )
        
        return feature_importance
    
    def save_model(self, filepath: str):
        """Save the ensemble model"""
        logger.info(f"Saving ensemble model to {filepath}")
        
        model_data = {
            'models': self.models,
            'weights': self.weights,
            'feature_engineer': self.feature_engineer
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load the ensemble model"""
        logger.info(f"Loading ensemble model from {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.weights = model_data['weights']
        self.feature_engineer = model_data['feature_engineer']
    
    def explain_prediction(self, X: pd.DataFrame, lead_id: str) -> Dict[str, Any]:
        """Explain prediction for a specific lead"""
        if X.empty:
            return {"error": "No data provided"}
        
        # Get prediction
        proba = self.predict_proba(X)[0]
        score = proba[1]
        
        # Get feature importance
        feature_importance = self.get_feature_importance()
        
        # Get top contributing features
        top_features = sorted(feature_importance.items(), 
                            key=lambda x: x[1], 
                            reverse=True)[:10]
        
        # Individual model predictions
        model_predictions = {}
        for name, model in self.models.items():
            model_proba = model.predict_proba(X)[0][1]
            model_predictions[name] = {
                'probability': model_proba,
                'weight': self.weights[name],
                'weighted_contribution': model_proba * self.weights[name]
            }
        
        return {
            'lead_id': lead_id,
            'final_score': score,
            'confidence': max(proba[0], proba[1]),
            'top_features': top_features,
            'model_contributions': model_predictions,
            'explanation': self._generate_explanation(score, top_features)
        }
    
    def _generate_explanation(self, score: float, top_features: List[tuple]) -> str:
        """Generate human-readable explanation"""
        if score > 0.8:
            intent_level = "Very High"
        elif score > 0.6:
            intent_level = "High"
        elif score > 0.4:
            intent_level = "Medium"
        else:
            intent_level = "Low"
        
        explanation = f"Lead shows {intent_level} intent (score: {score:.2f}). "
        
        if top_features:
            explanation += f"Key factors: {top_features[0][0]} (importance: {top_features[0][1]:.3f})"
            if len(top_features) > 1:
                explanation += f", {top_features[1][0]} (importance: {top_features[1][1]:.3f})"
        
        return explanation


# Training script
if __name__ == "__main__":
    from feature_engineering import FeatureEngineer
    
    # Load data (placeholder)
    # data = pd.read_csv("data/leads.csv")
    # 
    # # Feature engineering
    # feature_engineer = FeatureEngineer()
    # features = feature_engineer.create_all_features(data)
    # target = data['is_high_intent']
    # 
    # # Split data
    # X_train, X_temp, y_train, y_temp = train_test_split(
    #     features, target, test_size=0.4, random_state=42
    # )
    # X_val, X_test, y_val, y_test = train_test_split(
    #     X_temp, y_temp, test_size=0.5, random_state=42
    # )
    # 
    # # Train ensemble
    # ensemble = EnsembleLeadScorer()
    # ensemble.feature_engineer = feature_engineer
    # ensemble.initialize_models()
    # 
    # performance = ensemble.train_ensemble(X_train, y_train, X_val, y_val)
    # 
    # # Test ensemble
    # test_preds = ensemble.predict_proba(X_test)[:, 1]
    # test_auc = roc_auc_score(y_test, test_preds)
    # 
    # print(f"Ensemble Test AUC: {test_auc:.4f}")
    # 
    # # Save model
    # ensemble.save_model("models/ensemble_lead_scorer.pkl")
    
    pass
