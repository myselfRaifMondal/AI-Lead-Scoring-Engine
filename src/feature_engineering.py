import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        
    def create_behavioral_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral features from user interaction data"""
        logger.info("Creating behavioral features")
        
        # Website engagement features
        data['page_views_property_listings'] = data.get('property_listing_views', 0)
        data['time_spent_mortgage_calculator'] = data.get('mortgage_calc_time', 0)
        data['contact_form_submissions'] = data.get('contact_forms', 0)
        
        # Communication patterns
        data['email_open_rate'] = data.get('emails_opened', 0) / (data.get('emails_sent', 1) + 1e-8)
        data['whatsapp_response_time'] = data.get('avg_whatsapp_response_hours', 24)
        data['broker_message_frequency'] = data.get('broker_messages_per_week', 0)
        
        # Search behavior
        data['property_search_frequency'] = data.get('searches_per_week', 0)
        data['saved_searches_count'] = data.get('saved_searches', 0)
        data['favorited_properties'] = data.get('favorited_count', 0)
        data['price_range_searches'] = data.get('price_searches', 0)
        
        return data
    
    def create_demographic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create demographic features"""
        logger.info("Creating demographic features")
        
        # Financial profile
        data['income_level'] = data.get('annual_income', 0)
        data['employment_status'] = data.get('employment_type', 'Unknown')
        data['credit_score_band'] = pd.cut(data.get('credit_score', 650), 
                                         bins=[0, 580, 670, 740, 850], 
                                         labels=['Poor', 'Fair', 'Good', 'Excellent'])
        
        # Life stage indicators
        data['age_group'] = pd.cut(data.get('age', 35), 
                                 bins=[0, 25, 35, 45, 55, 100], 
                                 labels=['Young', 'Millennial', 'GenX', 'Boomer', 'Senior'])
        data['family_size'] = data.get('household_size', 1)
        data['marital_status'] = data.get('marital_status', 'Unknown')
        
        # Professional background
        data['job_title_seniority'] = data.get('job_level', 'Individual')
        data['industry'] = data.get('industry', 'Unknown')
        data['company_size'] = data.get('company_employees', 0)
        
        return data
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different data types"""
        logger.info("Creating interaction features")
        
        # Income-behavior interactions
        data['income_x_search_frequency'] = data['income_level'] * data['property_search_frequency']
        data['age_x_property_type'] = data['age_group'].astype(str) + '_' + data.get('preferred_property_type', 'Unknown')
        data['employment_x_loan_amount'] = data['employment_status'] + '_' + pd.cut(data.get('loan_inquiry_amount', 0), 
                                                                                   bins=[0, 100000, 300000, 500000, float('inf')], 
                                                                                   labels=['Low', 'Medium', 'High', 'Premium']).astype(str)
        
        return data
    
    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        logger.info("Creating time-based features")
        
        current_time = datetime.now()
        
        # Recency features - convert to numerical only
        if 'last_activity' in data.columns:
            last_activity_date = pd.to_datetime(data['last_activity'], errors='coerce')
            data['recency_last_activity'] = (current_time - last_activity_date).dt.days
            # Fill any NaN values with a default (e.g., 30 days)
            data['recency_last_activity'] = data['recency_last_activity'].fillna(30)
        else:
            data['recency_last_activity'] = 30  # Default value
        
        # Frequency features
        data['frequency_monthly_visits'] = data.get('monthly_visits', 0)
        data['frequency_weekly_searches'] = data.get('weekly_searches', 0)
        
        # Trend features
        data['engagement_trend_7d'] = data.get('engagement_last_7d', 0) - data.get('engagement_prev_7d', 0)
        data['trend_engagement_score'] = np.where(data['engagement_trend_7d'] > 0, 1, 0)
        
        return data
    
    def create_market_context_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create market context features"""
        logger.info("Creating market context features")
        
        # Market indicators
        data['local_price_trend'] = data.get('area_price_change_3m', 0)
        data['inventory_level'] = data.get('local_inventory_months', 6)
        data['interest_rate_environment'] = data.get('current_interest_rate', 7.0)
        
        # Market urgency indicators
        data['market_urgency_score'] = (
            (data['local_price_trend'] > 0.05).astype(int) +
            (data['inventory_level'] < 3).astype(int) +
            (data['interest_rate_environment'] > 7.5).astype(int)
        )
        
        return data
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies"""
        logger.info("Handling missing values")
        
        # Numeric columns - use median imputation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in self.imputers:
                self.imputers[col] = SimpleImputer(strategy='median')
                data[col] = self.imputers[col].fit_transform(data[[col]]).ravel()
            else:
                data[col] = self.imputers[col].transform(data[[col]]).ravel()
        
        # Categorical columns - use mode imputation
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in self.imputers:
                self.imputers[col] = SimpleImputer(strategy='most_frequent')
                data[col] = self.imputers[col].fit_transform(data[[col]]).ravel()
            else:
                data[col] = self.imputers[col].transform(data[[col]]).ravel()
        
        return data
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        logger.info("Encoding categorical features")
        
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
            else:
                # Handle unseen categories
                unique_values = set(data[col].unique())
                known_values = set(self.label_encoders[col].classes_)
                unseen_values = unique_values - known_values
                
                if unseen_values:
                    # Add unseen values to the encoder
                    self.label_encoders[col].classes_ = np.append(
                        self.label_encoders[col].classes_, 
                        list(unseen_values)
                    )
                
                data[col] = self.label_encoders[col].transform(data[col].astype(str))
        
        return data
    
    def scale_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        logger.info("Scaling features")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if fit:
            data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
        else:
            data[numeric_cols] = self.scaler.transform(data[numeric_cols])
        
        return data
    
    def cleanup_datatypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean up data types to ensure compatibility with ML models"""
        logger.info("Cleaning up data types")
        
        # Drop datetime columns that shouldn't be in final features
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            logger.info(f"Dropping datetime columns: {list(datetime_cols)}")
            data = data.drop(columns=datetime_cols)
        
        # Convert any remaining object columns to category for better handling
        object_cols = data.select_dtypes(include=['object']).columns
        for col in object_cols:
            if col not in ['lead_id']:  # Keep lead_id as object if present
                data[col] = data[col].astype('category')
        
        # Ensure all numerical columns are float64
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col] = data[col].astype('float64')
        
        # Remove any columns that are all NaN
        data = data.dropna(axis=1, how='all')
        
        logger.info(f"Final data shape: {data.shape}")
        logger.info(f"Final data types: {data.dtypes.value_counts().to_dict()}")
        
        return data
    
    def create_all_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Create all features in the pipeline"""
        logger.info("Starting feature engineering pipeline")
        
        # Create feature groups
        data = self.create_behavioral_features(data)
        data = self.create_demographic_features(data)
        data = self.create_interaction_features(data)
        data = self.create_time_features(data)
        data = self.create_market_context_features(data)
        
        # Handle missing values
        data = self.handle_missing_values(data)
        
        # Encode categorical features
        data = self.encode_categorical_features(data)
        
        # Clean up data types
        data = self.cleanup_datatypes(data)
        
        # Scale features
        data = self.scale_features(data, fit=fit)
        
        logger.info("Feature engineering pipeline completed")
        return data
    
    def get_feature_importance_explanation(self, feature_names: List[str], 
                                         importances: List[float]) -> Dict[str, str]:
        """Provide explanations for feature importance"""
        explanations = {
            'page_views_property_listings': 'Higher property listing views indicate stronger purchase intent',
            'email_open_rate': 'Higher email engagement shows active interest in communications',
            'income_level': 'Higher income correlates with qualification and purchasing power',
            'recency_last_activity': 'Recent activity indicates current interest and urgency',
            'property_search_frequency': 'Frequent searches show active house hunting behavior',
            'credit_score_band': 'Better credit scores indicate higher loan approval likelihood',
            'market_urgency_score': 'Market conditions influence buying urgency and timing'
        }
        
        feature_explanations = {}
        for name, importance in zip(feature_names, importances):
            feature_explanations[name] = {
                'importance': importance,
                'explanation': explanations.get(name, 'Feature contributes to lead scoring prediction')
            }
        
        return feature_explanations
