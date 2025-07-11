import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ensemble_model import EnsembleLeadScorer
from feature_engineering import FeatureEngineer

class TestEnsembleLeadScorer:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'lead_id': ['lead_1', 'lead_2', 'lead_3'],
            'property_listing_views': [10, 5, 15],
            'emails_opened': [3, 1, 5],
            'emails_sent': [5, 2, 8],
            'annual_income': [75000, 50000, 120000],
            'age': [32, 28, 45],
            'credit_score': [720, 650, 780],
            'employment_type': ['Full-time', 'Part-time', 'Full-time'],
            'marital_status': ['Married', 'Single', 'Married'],
            'is_high_intent': [1, 0, 1]
        })
    
    @pytest.fixture
    def ensemble_scorer(self):
        """Create ensemble scorer instance"""
        scorer = EnsembleLeadScorer()
        scorer.initialize_models()
        return scorer
    
    def test_initialize_models(self, ensemble_scorer):
        """Test model initialization"""
        assert 'xgboost' in ensemble_scorer.models
        assert 'lightgbm' in ensemble_scorer.models
        assert 'neural_network' in ensemble_scorer.models
        assert len(ensemble_scorer.weights) == 3
        assert all(w > 0 for w in ensemble_scorer.weights.values())
        assert abs(sum(ensemble_scorer.weights.values()) - 1.0) < 1e-6
    
    def test_train_ensemble(self, ensemble_scorer, sample_data):
        """Test ensemble training"""
        # Prepare features
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_all_features(sample_data.drop('is_high_intent', axis=1))
        target = sample_data['is_high_intent']
        
        # Split data
        X_train, X_val = features[:2], features[2:]
        y_train, y_val = target[:2], target[2:]
        
        # Train ensemble
        performance = ensemble_scorer.train_ensemble(X_train, y_train, X_val, y_val)
        
        # Check results
        assert isinstance(performance, dict)
        assert len(performance) == 3
        assert all(0 <= score <= 1 for score in performance.values())
    
    def test_predict_proba(self, ensemble_scorer, sample_data):
        """Test probability prediction"""
        # Prepare features
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_all_features(sample_data.drop('is_high_intent', axis=1))
        target = sample_data['is_high_intent']
        
        # Train with minimal data
        ensemble_scorer.train_ensemble(features, target, features, target)
        
        # Test prediction
        probs = ensemble_scorer.predict_proba(features)
        
        assert probs.shape == (3, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs >= 0) and np.all(probs <= 1)
    
    def test_predict(self, ensemble_scorer, sample_data):
        """Test binary prediction"""
        # Prepare features
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_all_features(sample_data.drop('is_high_intent', axis=1))
        target = sample_data['is_high_intent']
        
        # Train with minimal data
        ensemble_scorer.train_ensemble(features, target, features, target)
        
        # Test prediction
        preds = ensemble_scorer.predict(features)
        
        assert len(preds) == 3
        assert all(pred in [0, 1] for pred in preds)
    
    def test_explain_prediction(self, ensemble_scorer, sample_data):
        """Test prediction explanation"""
        # Prepare features
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_all_features(sample_data.drop('is_high_intent', axis=1))
        target = sample_data['is_high_intent']
        
        # Train with minimal data
        ensemble_scorer.train_ensemble(features, target, features, target)
        
        # Test explanation
        explanation = ensemble_scorer.explain_prediction(features[:1], 'test_lead')
        
        assert 'lead_id' in explanation
        assert 'final_score' in explanation
        assert 'confidence' in explanation
        assert 'explanation' in explanation
        assert 'model_contributions' in explanation
        assert explanation['lead_id'] == 'test_lead'
        assert 0 <= explanation['final_score'] <= 1
    
    def test_save_load_model(self, ensemble_scorer, sample_data, tmp_path):
        """Test model save and load"""
        # Prepare features
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_all_features(sample_data.drop('is_high_intent', axis=1))
        target = sample_data['is_high_intent']
        
        # Train ensemble
        ensemble_scorer.feature_engineer = feature_engineer
        ensemble_scorer.train_ensemble(features, target, features, target)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        ensemble_scorer.save_model(str(model_path))
        
        # Load model
        new_scorer = EnsembleLeadScorer()
        new_scorer.load_model(str(model_path))
        
        # Test that loaded model works
        original_pred = ensemble_scorer.predict_proba(features)
        loaded_pred = new_scorer.predict_proba(features)
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)
    
    def test_update_weights(self, ensemble_scorer):
        """Test weight update based on performance"""
        # Mock performance metrics
        performance = {
            'xgboost': 0.85,
            'lightgbm': 0.80,
            'neural_network': 0.75
        }
        
        ensemble_scorer._update_weights(performance)
        
        # Check that weights sum to 1
        assert abs(sum(ensemble_scorer.weights.values()) - 1.0) < 1e-6
        
        # Check that better performing model has higher weight
        assert ensemble_scorer.weights['xgboost'] > ensemble_scorer.weights['neural_network']
    
    def test_generate_explanation(self, ensemble_scorer):
        """Test explanation generation"""
        # Test different score ranges
        explanations = []
        for score in [0.9, 0.7, 0.5, 0.2]:
            explanation = ensemble_scorer._generate_explanation(score, [('feature1', 0.5), ('feature2', 0.3)])
            explanations.append(explanation)
            assert isinstance(explanation, str)
            assert str(score) in explanation
        
        # Check that different scores produce different intent levels
        assert "Very High" in explanations[0] or "High" in explanations[0]
        assert "Low" in explanations[3]
    
    def test_empty_data_handling(self, ensemble_scorer):
        """Test handling of empty data"""
        empty_df = pd.DataFrame()
        
        explanation = ensemble_scorer.explain_prediction(empty_df, 'test_lead')
        assert 'error' in explanation
        assert explanation['error'] == 'No data provided'


class TestFeatureEngineer:
    
    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineer instance"""
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'property_listing_views': [10, 5, 15],
            'emails_opened': [3, 1, 5],
            'emails_sent': [5, 2, 8],
            'annual_income': [75000, 50000, 120000],
            'age': [32, 28, 45],
            'credit_score': [720, 650, 780],
            'employment_type': ['Full-time', 'Part-time', 'Full-time'],
            'marital_status': ['Married', 'Single', 'Married']
        })
    
    def test_create_behavioral_features(self, feature_engineer, sample_data):
        """Test behavioral feature creation"""
        result = feature_engineer.create_behavioral_features(sample_data)
        
        assert 'page_views_property_listings' in result.columns
        assert 'email_open_rate' in result.columns
        assert 'whatsapp_response_time' in result.columns
        
        # Check email open rate calculation
        expected_open_rate = result['emails_opened'] / (result['emails_sent'] + 1e-8)
        np.testing.assert_array_almost_equal(result['email_open_rate'], expected_open_rate)
    
    def test_create_demographic_features(self, feature_engineer, sample_data):
        """Test demographic feature creation"""
        result = feature_engineer.create_demographic_features(sample_data)
        
        assert 'income_level' in result.columns
        assert 'age_group' in result.columns
        assert 'credit_score_band' in result.columns
        
        # Check age group categorization
        assert result['age_group'].iloc[0] == 'Millennial'  # age 32
        assert result['age_group'].iloc[2] == 'GenX'  # age 45
    
    def test_create_interaction_features(self, feature_engineer, sample_data):
        """Test interaction feature creation"""
        # First create required features
        sample_data = feature_engineer.create_behavioral_features(sample_data)
        sample_data = feature_engineer.create_demographic_features(sample_data)
        
        result = feature_engineer.create_interaction_features(sample_data)
        
        assert 'income_x_search_frequency' in result.columns
        assert 'age_x_property_type' in result.columns
        assert 'employment_x_loan_amount' in result.columns
    
    def test_create_time_features(self, feature_engineer, sample_data):
        """Test time-based feature creation"""
        result = feature_engineer.create_time_features(sample_data)
        
        assert 'recency_last_activity' in result.columns
        assert 'frequency_monthly_visits' in result.columns
        assert 'trend_engagement_score' in result.columns
        
        # Check that recency is non-negative
        assert all(result['recency_last_activity'] >= 0)
    
    def test_handle_missing_values(self, feature_engineer):
        """Test missing value handling"""
        # Create data with missing values
        data_with_missing = pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4],
            'categorical_col': ['A', 'B', None, 'A']
        })
        
        result = feature_engineer.handle_missing_values(data_with_missing)
        
        # Check no missing values remain
        assert not result.isnull().any().any()
        assert len(result) == len(data_with_missing)
    
    def test_encode_categorical_features(self, feature_engineer, sample_data):
        """Test categorical feature encoding"""
        result = feature_engineer.encode_categorical_features(sample_data)
        
        # Check that categorical columns are now numeric
        assert pd.api.types.is_numeric_dtype(result['employment_type'])
        assert pd.api.types.is_numeric_dtype(result['marital_status'])
        
        # Check that encoding is consistent
        assert result['employment_type'].iloc[0] == result['employment_type'].iloc[2]  # Both Full-time
    
    def test_scale_features(self, feature_engineer, sample_data):
        """Test feature scaling"""
        # First handle missing values and encode categoricals
        sample_data = feature_engineer.handle_missing_values(sample_data)
        sample_data = feature_engineer.encode_categorical_features(sample_data)
        
        result = feature_engineer.scale_features(sample_data)
        
        # Check that numerical features are scaled (approximately mean=0, std=1)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert abs(result[col].mean()) < 1e-10  # Close to 0
            assert abs(result[col].std() - 1) < 1e-10  # Close to 1
    
    def test_create_all_features(self, feature_engineer, sample_data):
        """Test complete feature engineering pipeline"""
        result = feature_engineer.create_all_features(sample_data)
        
        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        
        # Check that no missing values remain
        assert not result.isnull().any().any()
        
        # Check that features were created
        assert 'email_open_rate' in result.columns
        assert 'age_group' in result.columns
        assert 'recency_last_activity' in result.columns
    
    def test_feature_importance_explanation(self, feature_engineer):
        """Test feature importance explanation"""
        feature_names = ['email_open_rate', 'income_level', 'unknown_feature']
        importances = [0.3, 0.2, 0.1]
        
        explanations = feature_engineer.get_feature_importance_explanation(feature_names, importances)
        
        assert len(explanations) == 3
        assert explanations['email_open_rate']['importance'] == 0.3
        assert 'explanation' in explanations['email_open_rate']
        assert 'email engagement' in explanations['email_open_rate']['explanation']
        
        # Check unknown feature gets default explanation
        assert 'contributes to lead scoring' in explanations['unknown_feature']['explanation']


if __name__ == "__main__":
    pytest.main([__file__])
