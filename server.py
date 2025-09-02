from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Dict, Optional, Any
import joblib
import numpy as np
import os
import motor.motor_asyncio
from datetime import datetime
import logging
from bson import ObjectId
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Utility Functions
# ---------------------------
def clean_mongo_doc(doc):
    if doc is None:
        return None
    cleaned = {}
    for key, value in doc.items():
        if key == "_id":
            continue
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

# ---------------------------
# FastAPI setup
# ---------------------------
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

# ---------------------------
# Database connection
# ---------------------------
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client.asd_detection

# ---------------------------
# Global models
# ---------------------------
models = {}
scalers = {}
encoders = {}

# ---------------------------
# PSO Class
# ---------------------------
class PSO:
    def __init__(self, n_particles=20, n_iterations=50, w=0.5, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize_prediction(self, predictions, weights=None):
        if weights is None:
            weights = np.random.random(len(predictions))
            weights = weights / np.sum(weights)

        n_models = len(predictions)
        particles = np.random.random((self.n_particles, n_models))
        particles = particles / particles.sum(axis=1, keepdims=True)
        velocities = np.random.uniform(-0.1, 0.1, (self.n_particles, n_models))

        personal_best = particles.copy()
        personal_best_scores = np.full(self.n_particles, -np.inf)
        global_best = particles[0].copy()
        global_best_score = -np.inf

        for _ in range(min(self.n_iterations, 20)):
            for i in range(self.n_particles):
                ensemble_pred = np.average(predictions, weights=particles[i])
                score = abs(ensemble_pred - 0.5)

                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best[i] = particles[i].copy()

                if score > global_best_score:
                    global_best_score = score
                    global_best = particles[i].copy()

            for i in range(self.n_particles):
                r1, r2 = np.random.random(n_models), np.random.random(n_models)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - particles[i]) +
                                 self.c2 * r2 * (global_best - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.abs(particles[i])
                particles[i] = particles[i] / np.sum(particles[i])

        return global_best, global_best_score

# ---------------------------
# Request Models
# ---------------------------
class BehavioralAssessment(BaseModel):
    A1_Score: float
    A2_Score: float
    A3_Score: float
    A4_Score: float
    A5_Score: float
    A6_Score: float
    A7_Score: float
    A8_Score: float
    A9_Score: float
    A10_Score: float
    age: float
    gender: str

    @validator("A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
               "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score")
    def validate_scores(cls, v):
        if v not in [0, 0.5, 1]:
            raise ValueError("Scores must be 0, 0.5, or 1")
        return v

    @validator("age")
    def validate_age(cls, v):
        if v < 0 or v > 100:
            raise ValueError("Age must be between 0 and 100")
        return v

    @validator("gender")
    def validate_gender(cls, v):
        if v not in ["f", "m"]:
            raise ValueError("Gender must be f or m")
        return v

class EyeTrackingData(BaseModel):
    fixation_count: float
    mean_saccade: float
    max_saccade: float
    std_saccade: float
    mean_x: float
    mean_y: float
    std_x: float
    std_y: float
    mean_pupil: float

class FacialAnalysisData(BaseModel):
    facial_features: List[float]
    emotion_scores: Dict[str, float]
    attention_patterns: Dict[str, float]

class CompleteAssessmentRequest(BaseModel):
    session_id: str

# ---------------------------
# Load Models
# ---------------------------
@app.on_event("startup")
async def load_models():
    global models, scalers, encoders
    try:
        base_path = "/app/models"
        models["behavioral_rf"] = joblib.load(os.path.join(base_path, "behavioral_rf_model.joblib"))
        models["behavioral_svm"] = joblib.load(os.path.join(base_path, "behavioral_svm_model.joblib"))
        scalers["behavioral"] = joblib.load(os.path.join(base_path, "behavioral_scaler.joblib"))
        encoders["behavioral"] = joblib.load(os.path.join(base_path, "behavioral_label_encoder.joblib"))
        logger.info("Behavioral models loaded ✅")

        if os.path.exists(os.path.join(base_path, "eye_tracking_rf_model.joblib")):
            models["eye_tracking_rf"] = joblib.load(os.path.join(base_path, "eye_tracking_rf_model.joblib"))
            models["eye_tracking_svm"] = joblib.load(os.path.join(base_path, "eye_tracking_svm_model.joblib"))
            scalers["eye_tracking"] = joblib.load(os.path.join(base_path, "eye_tracking_scaler.joblib"))
            logger.info("Eye tracking models loaded ✅")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise e

# ---------------------------
# Root & Health
# ---------------------------
@app.get("/")
async def root():
    return {"message": "ASD Detection API", "version": "1.0.0", "status": "active"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": list(models.keys())
    }

# ---------------------------
# Behavioral Assessment
# ---------------------------
@app.post("/api/assessment/behavioral")
async def assess_behavioral(data: BehavioralAssessment):
    try:
        features = np.array([[
            data.A1_Score, data.A2_Score, data.A3_Score, data.A4_Score, data.A5_Score,
            data.A6_Score, data.A7_Score, data.A8_Score, data.A9_Score, data.A10_Score,
            data.age, 1 if data.gender == "m" else 0
        ]])
        features_scaled = scalers["behavioral"].transform(features)
        rf_pred = models["behavioral_rf"].predict_proba(features_scaled)[0]
        svm_pred = models["behavioral_svm"].predict_proba(features_scaled)[0]
        pso = PSO()
        base_predictions = [rf_pred[1], svm_pred[1]]
        optimal_weights, pso_score = pso.optimize_prediction(base_predictions)
        pso_prob = np.average(base_predictions, weights=optimal_weights)
        pso_pred = 1 if pso_prob > 0.5 else 0

        result = {
            "prediction": int(pso_pred),
            "probability": float(pso_prob),
            "confidence": float(pso_score),
            "model_results": {
                "random_forest": {"probability": float(rf_pred[1]), "prediction": int(rf_pred[1] > 0.5)},
                "svm": {"probability": float(svm_pred[1]), "prediction": int(svm_pred[1] > 0.5)},
                "pso": {"probability": float(pso_prob), "prediction": int(pso_pred), "weights": optimal_weights.tolist()}
            },
            "stage": "behavioral",
            "timestamp": datetime.now().isoformat()
        }
        await db.assessments.insert_one({"stage": "behavioral", "data": data.dict(), "result": result, "timestamp": datetime.now()})
        return result
    except Exception as e:
        logger.error(f"Behavioral assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

# ---------------------------
# Eye Tracking Assessment
# ---------------------------
@app.post("/api/assessment/eye_tracking")
async def assess_eye_tracking(data: EyeTrackingData):
    try:
        if "eye_tracking_rf" not in models:
            raise HTTPException(status_code=501, detail="Eye tracking models not available")
        features = np.array([[data.fixation_count, data.mean_saccade, data.max_saccade, data.std_saccade,
                              data.mean_x, data.mean_y, data.std_x, data.std_y, data.mean_pupil]])
        features_scaled = scalers["eye_tracking"].transform(features)
        rf_pred = models["eye_tracking_rf"].predict_proba(features_scaled)[0]
        svm_pred = models["eye_tracking_svm"].predict_proba(features_scaled)[0]
        pso = PSO()
        base_predictions = [rf_pred[1], svm_pred[1]]
        optimal_weights, pso_score = pso.optimize_prediction(base_predictions)
        pso_prob = np.average(base_predictions, weights=optimal_weights)
        pso_pred = 1 if pso_prob > 0.5 else 0
        result = {
            "prediction": int(pso_pred),
            "probability": float(pso_prob),
            "confidence": float(pso_score),
            "stage": "eye_tracking",
            "timestamp": datetime.now().isoformat()
        }
        await db.assessments.insert_one({"stage": "eye_tracking", "data": data.dict(), "result": result, "timestamp": datetime.now()})
        return result
    except Exception as e:
        logger.error(f"Eye tracking assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

# ---------------------------
# Facial Analysis (Mock)
# ---------------------------
@app.post("/api/assessment/facial_analysis")
async def assess_facial_analysis(data: FacialAnalysisData):
    try:
        feature_mean = np.mean(data.facial_features) if data.facial_features else 0.5
        attention_score = data.attention_patterns.get("attention_to_faces", 0.5)
        emotion_variability = np.std(list(data.emotion_scores.values())) if data.emotion_scores else 0.5
        combined_score = (feature_mean * 0.4 + attention_score * 0.4 + emotion_variability * 0.2)
        prediction = 1 if combined_score > 0.6 else 0
        result = {
            "prediction": int(prediction),
            "probability": float(combined_score),
            "confidence": float(abs(combined_score - 0.5) * 2),
            "stage": "facial_analysis",
            "timestamp": datetime.now().isoformat()
        }
        await db.assessments.insert_one({"stage": "facial_analysis", "data": data.dict(), "result": result, "timestamp": datetime.now()})
        return result
    except Exception as e:
        logger.error(f"Facial analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

# ---------------------------
# Complete Assessment
# ---------------------------
@app.post("/api/assessment/complete")
async def complete_assessment(request: CompleteAssessmentRequest):
    try:
        session_id = request.session_id
        behavioral = await db.assessments.find_one({"stage": "behavioral"}, sort=[("timestamp", -1)])
        eye_tracking = await db.assessments.find_one({"stage": "eye_tracking"}, sort=[("timestamp", -1)])
        facial = await db.assessments.find_one({"stage": "facial_analysis"}, sort=[("timestamp", -1)])
        stage_results = {}
        if behavioral: stage_results["behavioral"] = behavioral["result"]
        if eye_tracking: stage_results["eye_tracking"] = eye_tracking["result"]
        if facial: stage_results["facial_analysis"] = facial["result"]

        stage_weights = {"behavioral": 0.6, "eye_tracking": 0.25, "facial_analysis": 0.15}
        weighted_score, total_weight = 0, 0
        for stage, result in stage_results.items():
            if stage in stage_weights:
                weighted_score += result["probability"] * stage_weights[stage]
                total_weight += stage_weights[stage]
        final_probability = weighted_score / total_weight if total_weight > 0 else 0.5
        final_prediction = 1 if final_probability > 0.5 else 0
        confidence = abs(final_probability - 0.5) * 2

        final_result = {
            "session_id": session_id,
            "final_prediction": int(final_prediction),
            "final_probability": float(final_probability),
            "confidence_score": float(confidence),
            "stage_results": stage_results,
            "assessment_date": datetime.now().isoformat(),
            "stages_completed": len(stage_results)
        }
        return final_result
    except Exception as e:
        logger.error(f"Complete assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
