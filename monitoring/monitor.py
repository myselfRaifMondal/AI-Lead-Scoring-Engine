import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import json
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import redis
import psycopg2
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

# Configure logging
logger = structlog.get_logger()

# Prometheus metrics
PREDICTION_COUNTER = Counter('lead_scoring_predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('lead_scoring_prediction_latency_seconds', 'Prediction latency')
DATA_DRIFT_GAUGE = Gauge('lead_scoring_data_drift_score', 'Data drift score')
MODEL_PERFORMANCE_GAUGE = Gauge('lead_scoring_model_performance', 'Model performance metric')

class DataDriftMonitor:
    def __init__(self, redis_client, postgres_conn):
        self.redis_client = redis_client
        self.postgres_conn = postgres_conn
        self.reference_stats = {}
        self.drift_threshold = 0.1
        
    def calculate_psi(self, reference_dist: np.ndarray, current_dist: np.ndarray) -> float:
        """Calculate Population Stability Index"""
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        reference_dist = np.clip(reference_dist, eps, 1)
        current_dist = np.clip(current_dist, eps, 1)
        
        psi = np.sum((current_dist - reference_dist) * np.log(current_dist / reference_dist))
        return psi
    
    def detect_feature_drift(self, reference_data: pd.DataFrame, 
                           current_data: pd.DataFrame) -> Dict[str, float]:
        """Detect drift in individual features"""
        drift_scores = {}
        
        for column in reference_data.columns:
            if column in current_data.columns:
                # For numerical features
                if pd.api.types.is_numeric_dtype(reference_data[column]):
                    # Use KS test for numerical features
                    ks_stat, p_value = stats.ks_2samp(reference_data[column], current_data[column])
                    drift_scores[column] = ks_stat
                else:
                    # For categorical features, use PSI
                    ref_counts = reference_data[column].value_counts(normalize=True)
                    curr_counts = current_data[column].value_counts(normalize=True)
                    
                    # Align categories
                    all_categories = set(ref_counts.index) | set(curr_counts.index)
                    ref_aligned = np.array([ref_counts.get(cat, 0) for cat in all_categories])
                    curr_aligned = np.array([curr_counts.get(cat, 0) for cat in all_categories])
                    
                    psi = self.calculate_psi(ref_aligned, curr_aligned)
                    drift_scores[column] = psi
        
        return drift_scores
    
    def monitor_drift(self, current_data: pd.DataFrame) -> Dict[str, any]:
        """Monitor data drift and return alerts"""
        # Get reference data from Redis cache
        reference_data_json = self.redis_client.get('reference_data')
        if not reference_data_json:
            logger.warning("No reference data found for drift monitoring")
            return {"status": "no_reference_data"}
        
        reference_data = pd.read_json(reference_data_json)
        
        # Calculate drift scores
        drift_scores = self.detect_feature_drift(reference_data, current_data)
        
        # Identify features with significant drift
        drifted_features = {k: v for k, v in drift_scores.items() if v > self.drift_threshold}
        
        # Calculate overall drift score
        overall_drift = np.mean(list(drift_scores.values()))
        
        # Update Prometheus metrics
        DATA_DRIFT_GAUGE.set(overall_drift)
        
        # Log drift information
        logger.info("Data drift monitoring completed", 
                   overall_drift=overall_drift,
                   drifted_features=len(drifted_features))
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_drift_score": overall_drift,
            "drift_threshold": self.drift_threshold,
            "drifted_features": drifted_features,
            "all_drift_scores": drift_scores,
            "alert": overall_drift > self.drift_threshold
        }

class ModelPerformanceMonitor:
    def __init__(self, postgres_conn):
        self.postgres_conn = postgres_conn
        self.performance_threshold = 0.05  # 5% performance drop threshold
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate model performance metrics"""
        return {
            "auc": roc_auc_score(y_true, y_proba),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": 2 * precision_score(y_true, y_pred) * recall_score(y_true, y_pred) / 
                  (precision_score(y_true, y_pred) + recall_score(y_true, y_pred))
        }
    
    def monitor_performance(self, window_days: int = 7) -> Dict[str, any]:
        """Monitor model performance over recent window"""
        # Get recent predictions and outcomes from database
        query = """
        SELECT prediction_score, actual_outcome, created_at
        FROM lead_predictions 
        WHERE created_at >= %s AND actual_outcome IS NOT NULL
        ORDER BY created_at DESC
        """
        
        cursor = self.postgres_conn.cursor()
        cursor.execute(query, (datetime.now() - timedelta(days=window_days),))
        
        results = cursor.fetchall()
        cursor.close()
        
        if len(results) < 100:  # Minimum sample size
            logger.warning("Insufficient data for performance monitoring")
            return {"status": "insufficient_data"}
        
        # Convert to arrays
        y_proba = np.array([r[0] for r in results])
        y_true = np.array([r[1] for r in results])
        y_pred = (y_proba > 0.5).astype(int)
        
        # Calculate current metrics
        current_metrics = self.calculate_metrics(y_true, y_pred, y_proba)
        
        # Get baseline metrics from Redis
        baseline_metrics_json = self.redis_client.get('baseline_metrics')
        if baseline_metrics_json:
            baseline_metrics = json.loads(baseline_metrics_json)
            
            # Calculate performance degradation
            performance_changes = {}
            for metric, current_value in current_metrics.items():
                baseline_value = baseline_metrics.get(metric, current_value)
                performance_changes[metric] = current_value - baseline_value
        else:
            performance_changes = {}
        
        # Update Prometheus metrics
        MODEL_PERFORMANCE_GAUGE.set(current_metrics.get('auc', 0))
        
        # Check for significant performance drop
        auc_drop = performance_changes.get('auc', 0)
        alert = auc_drop < -self.performance_threshold
        
        logger.info("Model performance monitoring completed",
                   current_auc=current_metrics.get('auc'),
                   auc_change=auc_drop,
                   alert=alert)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": current_metrics,
            "performance_changes": performance_changes,
            "alert": alert,
            "sample_size": len(results)
        }

class ContinuousLearningManager:
    def __init__(self, postgres_conn, redis_client):
        self.postgres_conn = postgres_conn
        self.redis_client = redis_client
        self.retrain_threshold = 0.02  # 2% performance drop triggers retrain
        
    def should_retrain(self, performance_metrics: Dict[str, float]) -> bool:
        """Determine if model should be retrained"""
        # Get last retrain timestamp
        last_retrain = self.redis_client.get('last_retrain_timestamp')
        if last_retrain:
            last_retrain_time = datetime.fromisoformat(last_retrain.decode())
            time_since_retrain = datetime.now() - last_retrain_time
            
            # Don't retrain too frequently (minimum 1 day)
            if time_since_retrain.days < 1:
                return False
        
        # Check performance degradation
        auc_change = performance_metrics.get('auc', 0)
        if auc_change < -self.retrain_threshold:
            return True
        
        # Check drift score
        drift_score = float(self.redis_client.get('overall_drift_score') or 0)
        if drift_score > 0.1:  # High drift threshold
            return True
        
        return False
    
    def trigger_retrain(self):
        """Trigger model retraining pipeline"""
        logger.info("Triggering model retraining")
        
        # Update last retrain timestamp
        self.redis_client.set('last_retrain_timestamp', datetime.now().isoformat())
        
        # Here you would typically trigger your ML pipeline
        # For example, send a message to a queue, call an API, etc.
        # This is a placeholder for the actual implementation
        
        return {"status": "retrain_triggered", "timestamp": datetime.now().isoformat()}

class LeadScoringMonitor:
    def __init__(self, redis_host='localhost', redis_port=6379,
                 postgres_host='localhost', postgres_port=5432, 
                 postgres_db='lead_scoring', postgres_user='postgres',
                 postgres_password='password'):
        
        # Initialize connections
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.postgres_conn = psycopg2.connect(
            host=postgres_host,
            port=postgres_port,
            database=postgres_db,
            user=postgres_user,
            password=postgres_password
        )
        
        # Initialize monitors
        self.drift_monitor = DataDriftMonitor(self.redis_client, self.postgres_conn)
        self.performance_monitor = ModelPerformanceMonitor(self.postgres_conn)
        self.learning_manager = ContinuousLearningManager(self.postgres_conn, self.redis_client)
        
    def run_monitoring_cycle(self):
        """Run complete monitoring cycle"""
        logger.info("Starting monitoring cycle")
        
        # Get recent data for monitoring
        query = """
        SELECT * FROM lead_features 
        WHERE created_at >= %s
        ORDER BY created_at DESC
        LIMIT 10000
        """
        
        cursor = self.postgres_conn.cursor()
        cursor.execute(query, (datetime.now() - timedelta(days=1),))
        
        # Convert to DataFrame (simplified)
        # In real implementation, you'd properly handle column names and types
        recent_data = pd.DataFrame(cursor.fetchall())
        cursor.close()
        
        if recent_data.empty:
            logger.warning("No recent data available for monitoring")
            return
        
        # Monitor data drift
        drift_results = self.drift_monitor.monitor_drift(recent_data)
        
        # Monitor model performance
        performance_results = self.performance_monitor.monitor_performance()
        
        # Check if retraining is needed
        if (drift_results.get('alert', False) or 
            performance_results.get('alert', False)):
            
            should_retrain = self.learning_manager.should_retrain(
                performance_results.get('performance_changes', {})
            )
            
            if should_retrain:
                retrain_result = self.learning_manager.trigger_retrain()
                logger.info("Retraining triggered", result=retrain_result)
        
        # Store results
        monitoring_results = {
            "timestamp": datetime.now().isoformat(),
            "drift_monitoring": drift_results,
            "performance_monitoring": performance_results
        }
        
        self.redis_client.setex('latest_monitoring_results', 3600, json.dumps(monitoring_results))
        
        logger.info("Monitoring cycle completed", results=monitoring_results)
        
        return monitoring_results

# API endpoint for monitoring dashboard
def get_monitoring_dashboard(monitor: LeadScoringMonitor) -> Dict:
    """Get current monitoring status for dashboard"""
    latest_results = monitor.redis_client.get('latest_monitoring_results')
    if latest_results:
        return json.loads(latest_results)
    else:
        return {"status": "no_data"}

if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8001)
    
    # Initialize monitoring
    monitor = LeadScoringMonitor()
    
    # Run monitoring cycle (in production, this would be scheduled)
    results = monitor.run_monitoring_cycle()
    print(json.dumps(results, indent=2))
