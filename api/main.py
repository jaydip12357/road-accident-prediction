from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Dict
from api.model_loader import ModelLoader
import os

# Initialize FastAPI
app = FastAPI(
    title="Road Accident Severity Prediction API",
    description="Predict accident severity based on road conditions",
    version="1.0.0"
)

# Load models on startup
model_loader = None

@app.on_event("startup")
def load_models():
    global model_loader
    model_loader = ModelLoader(model_dir='models2')

# Input schema
class AccidentInput(BaseModel):
    speed_limit: int
    number_of_vehicles: int
    number_of_casualties: int
    hour: int
    light_conditions: str
    weather_conditions: str
    road_surface_conditions: str
    urban_or_rural_area: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "speed_limit": 30,
                "number_of_vehicles": 2,
                "number_of_casualties": 1,
                "hour": 17,
                "light_conditions": "Daylight",
                "weather_conditions": "Fine no high winds",
                "road_surface_conditions": "Dry",
                "urban_or_rural_area": "Urban"
            }
        }

# Output schema
class PredictionOutput(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    confidence: float

@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Road Accident Severity Prediction API",
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model_loader is not None}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: AccidentInput):
    """Predict accident severity."""
    try:
        # Convert input to dictionary
        input_dict = input_data.dict()
        
        # Create DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Encode categorical features
        for col in model_loader.categorical_cols:
            if col in input_df.columns and col in model_loader.label_encoders:
                try:
                    input_df[col] = model_loader.label_encoders[col].transform(input_df[col].astype(str))
                except:
                    input_df[col] = 0
        
        # Ensure all features present
        for col in model_loader.feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns
        input_df = input_df[model_loader.feature_names]
        
        # Convert to numeric
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Scale
        input_scaled = model_loader.scaler.transform(input_df)
        
        # Predict
        prediction = model_loader.model.predict(input_scaled)[0]
        probabilities = model_loader.model.predict_proba(input_scaled)[0]
        
        # Map prediction
        severity_map = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
        predicted_class = severity_map[prediction]
        
        # Format output
        prob_dict = {
            'Slight': float(probabilities[0]),
            'Serious': float(probabilities[1]),
            'Fatal': float(probabilities[2])
        }
        
        confidence = float(max(probabilities))
        
        return PredictionOutput(
            prediction=predicted_class,
            probabilities=prob_dict,
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
