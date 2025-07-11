from typing import Dict
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import sys
import uvicorn
from dotenv import load_dotenv

# Add paths for imports
sys.path.append('/app/src')
sys.path.append('/app')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from ensemble_model import EnsembleLeadScorer
    from feature_engineering import FeatureEngineer
except ImportError:
    try:
        from src.ensemble_model import EnsembleLeadScorer
        from src.feature_engineering import FeatureEngineer
    except ImportError:
        print("Warning: Could not import models. Using minimal API.")
        EnsembleLeadScorer = None
        FeatureEngineer = None

load_dotenv()

app = FastAPI(title="AI Lead Scoring API", version="1.0")

# Load model
lead_scorer = None
if EnsembleLeadScorer:
    try:
        directory_path = os.path.dirname(__file__)
        model_path = os.path.join(directory_path, '..', 'models', 'ensemble_lead_scorer.pkl')
        lead_scorer = EnsembleLeadScorer()
        if os.path.exists(model_path):
            lead_scorer.load_model(model_path)
        else:
            lead_scorer.initialize_models()
            print("Warning: No trained model found, using default initialization")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        lead_scorer = None

# Request model
class LeadData(BaseModel):
    data: Dict

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if lead_scorer is not None else "degraded",
        "models_loaded": lead_scorer is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/score-lead")
async def score_lead(request: LeadData):
    if not lead_scorer:
        raise HTTPException(status_code=503, detail="Model not available")
    
    if not FeatureEngineer:
        raise HTTPException(status_code=503, detail="Feature engineer not available")
    
    try:
        data = pd.DataFrame([request.data])
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_all_features(data)
        
        # Get prediction
        result = lead_scorer.explain_prediction(features, lead_id=request.data.get("lead_id", "unknown"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", 8000)))
