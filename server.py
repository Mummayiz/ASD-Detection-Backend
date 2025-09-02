from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Dict, Optional, Any
import joblib
import numpy as np
import pandas as pd
import os
import motor.motor_asyncio
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import base64
import cv2
import asyncio
import logging
from bson import ObjectId
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_mongo_doc(doc):
    """Clean MongoDB document by removing ObjectId and converting datetime"""
    if doc is None:
        return None
    
    cleaned = {}
    for key, value in doc.items():
        if key == '_id':
            continue  # Skip ObjectId
        elif isinstance(value, ObjectId):
            cleaned[key] = str(value)
        elif isinstance(value, datetime):
            cleaned[key] = value.isoformat()
        elif isinstance(value, dict):
            cleaned[key] = clean_mongo_doc(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_mongo_doc(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value
    return cleaned

def json_encoder(obj):
    """Custom JSON encoder for MongoDB ObjectId and datetime objects"""
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

app = FastAPI(
    title="ASD Detection API",
    description="Machine Learning API for Autism Spectrum Disorder Detection with Multi-Stage Assessment",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client.asd_detection

# Global variables for models
models = {}
scalers = {}
encoders = {}

@app.on_event("startup")
async def load_models():
    """Load trained ML models on startup"""
    global models, scalers, encoders
    
    try:
        # Use relative path (works both locally & on Render)
        base_path = os.path.join(os.path.dirname(__file__), "models")
        
        # Load behavioral models
        models['behavioral_rf'] = joblib.load(os.path.join(base_path, 'behavioral_rf_model.joblib'))
        models['behavioral_svm'] = joblib.load(os.path.join(base_path, 'behavioral_svm_model.joblib'))
        scalers['behavioral'] = joblib.load(os.path.join(base_path, 'behavioral_scaler.joblib'))
        encoders['behavioral'] = joblib.load(os.path.join(base_path, 'behavioral_label_encoder.joblib'))
        
        logger.info("Behavioral models loaded successfully")
        
        # Load eye tracking models if available
        if os.path.exists(os.path.join(base_path, 'eye_tracking_rf_model.joblib')):
            models['eye_tracking_rf'] = joblib.load(os.path.join(base_path, 'eye_tracking_rf_model.joblib'))
            models['eye_tracking_svm'] = joblib.load(os.path.join(base_path, 'eye_tracking_svm_model.joblib'))
            scalers['eye_tracking'] = joblib.load(os.path.join(base_path, 'eye_tracking_scaler.joblib'))
            logger.info("Eye tracking models loaded successfully")
        
        logger.info("All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise e


# 🔽 Keep the rest of your code exactly the same (no other changes needed) 🔽

# ... (all your assessment classes, endpoints, helper functions, etc.)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
