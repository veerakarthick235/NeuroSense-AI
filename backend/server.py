from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal
from datetime import datetime, timezone, timedelta
from pathlib import Path
import google.generativeai as genai
import numpy as np
from sklearn.linear_model import LinearRegression
import statistics
import bcrypt
import jwt
import uuid
import random
import logging
import os
import uvicorn

# -------------------------------------------------------
# Load environment variables
# -------------------------------------------------------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# Ensure environment variables are loaded
MONGO_URL = os.environ.get("MONGO_URL")
DB_NAME = os.environ.get("DB_NAME")
JWT_SECRET = os.environ.get("JWT_SECRET")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Safety checks and logging setup
if not all([MONGO_URL, DB_NAME, JWT_SECRET, GEMINI_API_KEY]):
    logging.warning("Missing one or more environment variables!")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logging.warning("GEMINI_API_KEY not found. AI insights will be disabled.")


# -------------------------------------------------------
# FastAPI App
# -------------------------------------------------------
app = FastAPI()
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

# -------------------------------------------------------
# Database
# -------------------------------------------------------
# Initialize client and db using environment variables
client = AsyncIOMotorClient(MONGO_URL) if MONGO_URL else None
db = client[DB_NAME] if client and DB_NAME else None

# The problematic line 'if not db:' has been removed to fix the Render deployment error.
# We rely on the checks inside the route handlers instead.


# -------------------------------------------------------
# Models
# -------------------------------------------------------
class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    password_hash: Optional[str] = None
    name: str
    role: Literal["patient", "researcher"] = "patient"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserRegister(BaseModel):
    email: str
    password: str
    name: str
    role: Literal["patient", "researcher"] = "patient"


class UserLogin(BaseModel):
    email: str
    password: str


class SensorData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    data_type: Literal["vocal", "movement", "social"]
    metrics: dict
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthMetrics(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    vocal_score: float
    movement_score: float
    social_score: float
    overall_score: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TBIAlert(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    severity: Literal["low", "medium", "high"]
    message: str
    metrics: dict
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AIInsight(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    insight_type: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GenerateInsightRequest(BaseModel):
    user_id: str


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def hash_password(password: str):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str):
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_token(user_id: str, email: str, role: str):
    payload = {
        "user_id": user_id,
        "email": email,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(days=7),
    }
    # Check if JWT_SECRET is set before encoding
    if not JWT_SECRET:
         raise RuntimeError("JWT secret not configured.")
         
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # Check if JWT_SECRET is set before decoding
        if not JWT_SECRET:
             raise HTTPException(status_code=500, detail="Server misconfiguration: JWT secret not set.")
        
        return jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication")


def generate_simulated_sensor_data(data_type: str):
    if data_type == "vocal":
        return {
            "pitch_mean": round(random.uniform(80, 250), 2),
            "pitch_variance": round(random.uniform(10, 50), 2),
            "speech_rate": round(random.uniform(100, 180), 2),
            "pause_frequency": round(random.uniform(0.1, 0.5), 2),
            "voice_quality": round(random.uniform(0.6, 1.0), 2),
        }
    elif data_type == "movement":
        return {
            "acceleration_x": round(random.uniform(-2, 2), 3),
            "acceleration_y": round(random.uniform(-2, 2), 3),
            "acceleration_z": round(random.uniform(-2, 2), 3),
            "gyro_x": round(random.uniform(-1, 1), 3),
            "gyro_y": round(random.uniform(-1, 1), 3),
            "gyro_z": round(random.uniform(-1, 1), 3),
            "gait_stability": round(random.uniform(0.5, 1.0), 2),
        }
    else:
        return {
            "interaction_count": random.randint(5, 30),
            "response_time_avg": round(random.uniform(1, 5), 2),
            "sentiment_score": round(random.uniform(-0.5, 1.0), 2),
            "engagement_level": round(random.uniform(0.4, 1.0), 2),
        }

# -------------------------------------------------------
# Auth
# -------------------------------------------------------
@api_router.post("/auth/register")
async def register(data: UserRegister):
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection error")

    existing = await db.users.find_one({"email": data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=data.email,
        password_hash=hash_password(data.password),
        name=data.name,
        role=data.role,
    )

    d = user.model_dump()
    d["created_at"] = d["created_at"].isoformat()
    await db.users.insert_one(d)

    return {
        "token": create_token(user.id, user.email, user.role),
        "user": {"id": user.id, "email": user.email, "name": user.name, "role": user.role},
    }


@api_router.post("/auth/login")
async def login(data: UserLogin):
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection error")

    user = await db.users.find_one({"email": data.email})
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "token": create_token(user["id"], user["email"], user["role"]),
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
        },
    }


# -------------------------------------------------------
# Sensor simulation
# -------------------------------------------------------
@api_router.post("/data/sensors/simulate")
async def simulate(current_user=Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection error")
        
    uid = current_user["user_id"]
    out = []

    for typ in ["vocal", "movement", "social"]:
        metrics = generate_simulated_sensor_data(typ)
        obj = SensorData(user_id=uid, data_type=typ, metrics=metrics)
        d = obj.model_dump()
        d["timestamp"] = d["timestamp"].isoformat()
        await db.sensor_data.insert_one(d)
        out.append(d)

    return {"message": "Simulated data generated", "data": out}


# -------------------------------------------------------
# Metrics
# -------------------------------------------------------
@api_router.get("/metrics/latest")
async def latest_metrics(current_user=Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection error")
        
    uid = current_user["user_id"]

    async def get_latest(type):
        return await db.sensor_data.find_one(
            {"user_id": uid, "data_type": type}, {"_id": 0}, sort=[("timestamp", -1)]
        )

    vocal = await get_latest("vocal")
    movement = await get_latest("movement")
    social = await get_latest("social")

    # Use 0 if data is missing to allow score calculation to proceed
    v = vocal["metrics"]["voice_quality"] * 100 if vocal and "voice_quality" in vocal.get("metrics", {}) else 0
    m = movement["metrics"]["gait_stability"] * 100 if movement and "gait_stability" in movement.get("metrics", {}) else 0
    s = social["metrics"]["engagement_level"] * 100 if social and "engagement_level" in social.get("metrics", {}) else 0
    
    # Calculate overall score, avoiding division by zero if all are 0/missing
    count = sum(1 for score in [v, m, s] if score != 0)
    overall = (v + m + s) / count if count > 0 else 0


    obj = HealthMetrics(
        user_id=uid,
        vocal_score=v,
        movement_score=m,
        social_score=s,
        overall_score=overall,
    )

    d = obj.model_dump()
    d["timestamp"] = d["timestamp"].isoformat()
    await db.health_metrics.insert_one(d)

    return d


# -------------------------------------------------------
# Alerts
# -------------------------------------------------------
@api_router.post("/alerts/check")
async def check_alerts(current_user=Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection error")
        
    uid = current_user["user_id"]
    latest = await db.health_metrics.find_one(
        {"user_id": uid}, {"_id": 0}, sort=[("timestamp", -1)]
    )

    if not latest:
        return {"alerts_created": 0, "alerts": []}

    score = latest["overall_score"]

    if score < 60:
        sev = "high"
        msg = "Significant cognitive decline detected. Immediate review recommended."
    elif score < 75:
        sev = "medium"
        msg = "Moderate cognitive deviation detected. Trend monitoring is advised."
    else:
        return {"alerts_created": 0, "alerts": []}

    alert = TBIAlert(
        user_id=uid,
        severity=sev,
        message=msg,
        metrics=latest,
    )

    d = alert.model_dump()
    d["timestamp"] = d["timestamp"].isoformat()
    await db.tbi_alerts.insert_one(d)

    return {"alerts_created": 1, "alerts": [d]}


# -------------------------------------------------------
# AI Insight (Gemini)
# -------------------------------------------------------
@api_router.post("/insights/generate")
async def advanced_ai_insight(request: GenerateInsightRequest, current_user=Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection error")
    if not GEMINI_API_KEY:
         raise HTTPException(status_code=503, detail="AI service is unavailable due to missing API key.")
        
    uid = current_user["user_id"]

    records = await db.health_metrics.find(
        {"user_id": uid}, {"_id": 0}
    ).sort("timestamp", -1).limit(7).to_list(7)

    if not records:
        return {"message": "No data available"}

    records = list(reversed(records))

    overall = [r.get("overall_score", 0) for r in records]
    vocal = [r.get("vocal_score", 0) for r in records]
    movement = [r.get("movement_score", 0) for r in records]
    social = [r.get("social_score", 0) for r in records]

    # Time series analysis for trend and volatility
    X = np.arange(len(overall)).reshape(-1, 1)
    
    # Check if there are enough points for regression and ensure non-zero values
    if len(overall) > 1 and sum(overall) != 0:
        slope = round(LinearRegression().fit(X, overall).coef_[0], 3)
    else:
        slope = 0
        
    vol = round(np.std(overall), 3)

    # Anomaly detection (last score significantly below average)
    mean_vocal = statistics.mean(vocal) if vocal else 0
    mean_movement = statistics.mean(movement) if movement else 0
    mean_social = statistics.mean(social) if social else 0

    anomalies = {
        "vocal_anomaly": vocal[-1] < (mean_vocal - 10) if mean_vocal > 0 else False,
        "movement_anomaly": movement[-1] < (mean_movement - 10) if mean_movement > 0 else False,
        "social_anomaly": social[-1] < (mean_social - 10) if mean_social > 0 else False,
    }

    # Custom Risk Score calculation
    current_overall_score = overall[-1] if overall else 100
    risk_score = (
        (100 - current_overall_score) * 0.5 +  # Weighting current status
        (vol * 2) +                           # Weighting volatility
        (10 if anomalies["movement_anomaly"] else 0) +
        (10 if anomalies["vocal_anomaly"] else 0)
    )

    risk = (
        "Low" if risk_score < 40 else
        "Medium" if risk_score < 70 else
        "High"
    )

    prompt = f"""
    Provide a clinical-style cognitive analysis for a patient based on the following key metrics.
    
    The analysis should focus on **trend, stability, and potential risks**.
    
    * **Current Overall Score:** {current_overall_score:.1f}
    * **Trend (Slope over last 7 days):** {slope} (Higher means improving)
    * **Volatility (Standard Deviation):** {vol} (Higher means less stable)
    * **Anomalies Detected (Last reading significantly low):**
        * Vocal: {anomalies["vocal_anomaly"]}
        * Movement: {anomalies["movement_anomaly"]}
    * **Calculated Risk Category:** {risk}
    
    Based on these data points, summarize the patient's current cognitive stability and provide one actionable recommendation for their care team.
    """

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        text = response.text.strip()

        obj = AIInsight(
            user_id=uid,
            insight_type="advanced_analysis",
            content=text,
        )

        d = obj.model_dump()
        d["timestamp"] = d["timestamp"].isoformat()
        await db.ai_insights.insert_one(d)

        return d

    except Exception as e:
        logging.error(f"Insight generation failed: {e}")
        raise HTTPException(status_code=500, detail="Insight generation failed")


# -------------------------------------------------------
# Researcher
# -------------------------------------------------------
@api_router.get("/research/patients")
async def patients(current_user=Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection error")
        
    if current_user["role"] != "researcher":
        raise HTTPException(status_code=403, detail="Permission denied")

    # Fetch all patients
    pts = await db.users.find(
        {"role": "patient"}, {"_id": 0, "password_hash": 0}
    ).to_list(500)

    # Attach latest metrics for each patient
    for p in pts:
        p["latest_metrics"] = await db.health_metrics.find_one(
            {"user_id": p["id"]}, {"_id": 0}, sort=[("timestamp", -1)]
        )

    return pts


@api_router.get("/research/statistics")
async def stats(current_user=Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection error")
        
    if current_user["role"] != "researcher":
        raise HTTPException(status_code=403, detail="Permission denied")

    total = await db.users.count_documents({"role": "patient"})
    sensor = await db.sensor_data.count_documents({})
    alerts = await db.tbi_alerts.count_documents({})

    recents = await db.health_metrics.find(
        {}, {"_id": 0}
    ).sort("timestamp", -1).limit(100).to_list(100)

    avg = {"overall": 0, "vocal": 0, "movement": 0, "social": 0}
    if recents:
        count = len(recents)
        avg = {
            "overall": sum(m.get("overall_score", 0) for m in recents) / count,
            "vocal": sum(m.get("vocal_score", 0) for m in recents) / count,
            "movement": sum(m.get("movement_score", 0) for m in recents) / count,
            "social": sum(m.get("social_score", 0) for m in recents) / count,
        }

    return {
        "total_patients": total,
        "total_sensor_readings": sensor,
        "total_alerts": alerts,
        "average_scores": avg,
    }


# -------------------------------------------------------
# Root
# -------------------------------------------------------
@api_router.get("/")
async def root():
    return {"message": "NeuroSense AI Backend Running"}


app.include_router(api_router)

# -------------------------------------------------------
# 🔥 FINAL CORS CONFIGURATION — FIXES THE ORIGINAL NETLIFY/RENDER ERROR
# -------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    # Allow the Netlify frontend origin (and common localhost ports for testing)
    allow_origins=[
        "https://neuro-sense-ai.netlify.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Render needs this for OPTIONS (preflight) requests
@app.options("/{full_path:path}")
async def preflight(full_path: str):
    return {"message": "OK"}


# -------------------------------------------------------
# Shutdown
# -------------------------------------------------------
@app.on_event("shutdown")
async def shutdown():
    if client:
        client.close()
