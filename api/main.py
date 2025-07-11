from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import datetime
import redis
import psycopg2
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from contextlib import asynccontextmanager

# Add src to path
import sys
sys.path.append('/app/src')
sys.path.append('/app')

try:
    from ensemble_model import EnsembleLeadScorer
    from feature_engineering import FeatureEngineer
    from llm_reranker import LLMReranker
except ImportError:
    try:
        from src.ensemble_model import EnsembleLeadScorer
        from src.feature_engineering import FeatureEngineer
        from src.llm_reranker import LLMReranker
    except ImportError:
        # For development
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        from ensemble_model import EnsembleLeadScorer
        from feature_engineering import FeatureEngineer
        from llm_reranker import LLMReranker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('lead_scoring_predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('lead_scoring_prediction_latency_seconds', 'Prediction latency')
ERROR_COUNTER = Counter('lead_scoring_errors_total', 'Total errors', ['error_type'])

# Global variables for models
ensemble_scorer = None
feature_engineer = None
llm_reranker = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global ensemble_scorer, feature_engineer, llm_reranker, redis_client
    
    logger.info("Starting AI Lead Scoring Engine...")
    
    # Initialize Redis
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    redis_client = redis.from_url(redis_url)
    
    # Initialize models
    try:
        ensemble_scorer = EnsembleLeadScorer()
        feature_engineer = FeatureEngineer()
        
        # Try to load existing model
        model_path = os.getenv('MODEL_PATH', 'models/ensemble_lead_scorer.pkl')
        if os.path.exists(model_path):
            ensemble_scorer.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            # Initialize with default models if no trained model exists
            ensemble_scorer.initialize_models()
            logger.warning("No trained model found, using default initialization")
        
        # Initialize LLM reranker
        openai_key = os.getenv('OPENAI_API_KEY')
        llm_reranker = LLMReranker(openai_api_key=openai_key)
        
        logger.info("All models initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Lead Scoring Engine...")
    if redis_client:
        redis_client.close()

app = FastAPI(
    title="AI Lead Scoring Engine",
    description="Real-time lead scoring for brokers with sub-300ms response times",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class LeadData(BaseModel):
    data: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    models_loaded: bool

# Dependency for database connection
def get_db_connection():
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise HTTPException(status_code=500, detail="Database URL not configured")
    
    try:
        conn = psycopg2.connect(database_url)
        yield conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")
    finally:
        conn.close()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = all([
        ensemble_scorer is not None,
        feature_engineer is not None,
        llm_reranker is not None
    ])
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        models_loaded=models_loaded
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/score-lead")
async def score_lead(request: LeadData, db_conn = Depends(get_db_connection)):
    """Score a lead and return prediction with explanation"""
    start_time = time.time()
    
    try:
        PREDICTION_COUNTER.inc()
        
        # Validate input
        if not request.data:
            ERROR_COUNTER.labels(error_type="validation_error").inc()
            raise HTTPException(status_code=400, detail="No data provided")
        
        lead_id = request.data.get('lead_id', 'unknown')
        logger.info(f"Scoring lead: {lead_id}")
        
        # Convert to DataFrame
        lead_df = pd.DataFrame([request.data])
        
        # Feature engineering
        try:
            features = feature_engineer.create_all_features(lead_df, fit=False)
        except Exception as e:
            ERROR_COUNTER.labels(error_type="feature_engineering_error").inc()
            logger.error(f"Feature engineering failed: {e}")
            raise HTTPException(status_code=500, detail="Feature engineering failed")
        
        # Get base prediction from ensemble model
        try:
            base_scores = ensemble_scorer.predict_proba(features)[:, 1]
        except Exception as e:
            ERROR_COUNTER.labels(error_type="model_prediction_error").inc()
            logger.error(f"Model prediction failed: {e}")
            raise HTTPException(status_code=500, detail="Model prediction failed")
        
        # Apply LLM re-ranking if text data available
        text_fields = ['email_content', 'chat_messages', 'property_inquiries', 'sales_notes']
        has_text_data = any(field in request.data for field in text_fields)
        
        if has_text_data:
            try:
                final_scores = llm_reranker.rerank_leads(lead_df, base_scores)
            except Exception as e:
                logger.warning(f"LLM re-ranking failed, using base scores: {e}")
                final_scores = base_scores
        else:
            final_scores = base_scores
        
        # Generate explanation
        try:
            explanation = ensemble_scorer.explain_prediction(features, lead_id)
            
            # Add LLM adjustment if applied
            if has_text_data:
                llm_adjustment = final_scores[0] - base_scores[0]
                explanation['llm_adjustment'] = llm_adjustment
                
                # Get text explanation
                text_data = ' '.join([
                    str(request.data.get(field, '')) 
                    for field in text_fields 
                    if field in request.data
                ])
                explanation['text_analysis'] = llm_reranker.explain_text_adjustment(
                    text_data, llm_adjustment
                )
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            explanation = {
                'lead_id': lead_id,
                'final_score': final_scores[0],
                'explanation': 'Explanation generation failed'
            }
        
        # Store prediction in database
        try:
            cursor = db_conn.cursor()
            cursor.execute("""
                INSERT INTO model_predictions (
                    lead_id, prediction_score, confidence_score, 
                    explanation_text, response_time_ms
                ) VALUES (%s, %s, %s, %s, %s)
            """, (
                lead_id,
                float(final_scores[0]),
                explanation.get('confidence', 0.0),
                explanation.get('explanation', ''),
                int((time.time() - start_time) * 1000)
            ))
            db_conn.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
            # Don't fail the request if storage fails
        
        # Cache result in Redis
        try:
            cache_key = f"lead_score:{lead_id}"
            cache_data = {
                'score': float(final_scores[0]),
                'timestamp': datetime.now().isoformat(),
                'explanation': explanation.get('explanation', '')
            }
            redis_client.setex(cache_key, 900, str(cache_data))  # 15 minutes
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")
        
        # Record latency
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        
        # Check SLA compliance
        if latency > 0.3:  # 300ms SLA
            logger.warning(f"SLA violation: {latency:.3f}s for lead {lead_id}")
        
        logger.info(f"Lead {lead_id} scored in {latency:.3f}s: {final_scores[0]:.3f}")
        
        return explanation
        
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNTER.labels(error_type="unexpected_error").inc()
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/lead/{lead_id}/score")
async def get_lead_score(lead_id: str):
    """Get cached lead score"""
    try:
        cache_key = f"lead_score:{lead_id}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            return JSONResponse(content=eval(cached_result))
        else:
            raise HTTPException(status_code=404, detail="Lead score not found")
            
    except Exception as e:
        logger.error(f"Failed to retrieve cached score: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve score")

@app.get("/high-intent-leads")
async def get_high_intent_leads(limit: int = 100, db_conn = Depends(get_db_connection)):
    """Get list of high-intent leads"""
    try:
        cursor = db_conn.cursor()
        cursor.execute("""
            SELECT lead_id, prediction_score, confidence_score, 
                   explanation_text, prediction_timestamp
            FROM model_predictions mp
            WHERE prediction_score > 0.7
            AND id IN (
                SELECT MAX(id) FROM model_predictions GROUP BY lead_id
            )
            ORDER BY prediction_score DESC
            LIMIT %s
        """, (limit,))
        
        results = cursor.fetchall()
        cursor.close()
        
        leads = []
        for result in results:
            leads.append({
                'lead_id': result[0],
                'score': float(result[1]),
                'confidence': float(result[2]),
                'explanation': result[3],
                'timestamp': result[4].isoformat()
            })
        
        return {'high_intent_leads': leads, 'count': len(leads)}
        
    except Exception as e:
        logger.error(f"Failed to retrieve high-intent leads: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve leads")

@app.get("/model/status")
async def get_model_status():
    """Get model status and performance metrics"""
    try:
        status = {
            'model_loaded': ensemble_scorer is not None,
            'feature_engineer_loaded': feature_engineer is not None,
            'llm_reranker_loaded': llm_reranker is not None,
            'redis_connected': redis_client is not None
        }
        
        if ensemble_scorer:
            status['model_weights'] = ensemble_scorer.weights
            status['model_count'] = len(ensemble_scorer.models)
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model status")

@app.post("/model/retrain")
async def trigger_model_retrain():
    """Trigger model retraining (admin endpoint)"""
    try:
        # This would typically trigger a background job or message queue
        logger.info("Model retraining triggered")
        
        # For now, just return acknowledgment
        return {
            'status': 'retrain_triggered',
            'timestamp': datetime.now().isoformat(),
            'message': 'Model retraining has been queued'
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger retraining: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger retraining")

@app.get("/monitoring/dashboard")
async def get_monitoring_dashboard():
    """Get monitoring dashboard data"""
    try:
        # Get recent monitoring results from Redis
        monitoring_data = redis_client.get('latest_monitoring_results')
        
        if monitoring_data:
            return JSONResponse(content=eval(monitoring_data))
        else:
            return {
                'status': 'no_data',
                'message': 'No monitoring data available'
            }
            
    except Exception as e:
        logger.error(f"Failed to get monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get monitoring data")

# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    ERROR_COUNTER.labels(error_type="global_error").inc()
    logger.error(f"Global exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            'error': 'Internal server error',
            'timestamp': datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    
    uvicorn.run(app, host=host, port=port, log_level="info")
