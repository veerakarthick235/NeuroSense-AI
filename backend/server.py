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

# -----------------------------------------------
# Load environment variables
# -----------------------------------------------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ["DB_NAME"]
JWT_SECRET = os.environ["JWT_SECRET"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# Gemini setup
genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------------------------
# FastAPI Setup
# -----------------------------------------------
app = FastAPI()
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

# -----------------------------------------------
# MongoDB Client
# -----------------------------------------------
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]


# -----------------------------------------------
# MODELS
# -----------------------------------------------
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


# -----------------------------------------------
# HELPERS
# -----------------------------------------------
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
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        return jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
    except:
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


# -----------------------------------------------
# AUTH ENDPOINTS
# -----------------------------------------------
@api_router.post("/auth/register")
async def register(data: UserRegister):
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

    token = create_token(user.id, user.email, user.role)
    return {"token": token, "user": {"id": user.id, "email": user.email, "name": user.name, "role": user.role}}


@api_router.post("/auth/login")
async def login(data: UserLogin):
    user = await db.users.find_one({"email": data.email})
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(user["id"], user["email"], user["role"])
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
        },
    }


@api_router.get("/auth/me")
async def me(current_user=Depends(get_current_user)):
    user = await db.users.find_one({"id": current_user["user_id"]}, {"_id": 0, "password_hash": 0})
    return user


# -----------------------------------------------
# SENSOR SIMULATION
# -----------------------------------------------
@api_router.post("/data/sensors/simulate")
async def simulate(current_user=Depends(get_current_user)):
    uid = current_user["user_id"]
    output = []

    for typ in ["vocal", "movement", "social"]:
        metrics = generate_simulated_sensor_data(typ)
        obj = SensorData(user_id=uid, data_type=typ, metrics=metrics)
        d = obj.model_dump()
        d["timestamp"] = d["timestamp"].isoformat()
        await db.sensor_data.insert_one(d)
        output.append(d)

    return {"message": "Simulated data generated", "data": output}


# -----------------------------------------------
# METRICS
# -----------------------------------------------
@api_router.get("/metrics/latest")
async def latest_metrics(current_user=Depends(get_current_user)):
    uid = current_user["user_id"]

    async def latest(type):
        return await db.sensor_data.find_one(
            {"user_id": uid, "data_type": type}, {"_id": 0}, sort=[("timestamp", -1)]
        )

    vocal = await latest("vocal")
    movement = await latest("movement")
    social = await latest("social")

    v = vocal["metrics"]["voice_quality"] * 100 if vocal else 0
    m = movement["metrics"]["gait_stability"] * 100 if movement else 0
    s = social["metrics"]["engagement_level"] * 100 if social else 0

    overall = (v + m + s) / 3

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


@api_router.get("/metrics/history")
async def history(days: int = 7, current_user=Depends(get_current_user)):
    uid = current_user["user_id"]
    start = datetime.now(timezone.utc) - timedelta(days=days)

    hist = await db.health_metrics.find(
        {"user_id": uid, "timestamp": {"$gte": start.isoformat()}}, {"_id": 0}
    ).sort("timestamp", 1).to_list(500)

    return hist


# -----------------------------------------------
# ALERTS
# -----------------------------------------------
@api_router.post("/alerts/check")
async def check_alerts(current_user=Depends(get_current_user)):
    uid = current_user["user_id"]
    latest = await db.health_metrics.find_one(
        {"user_id": uid}, {"_id": 0}, sort=[("timestamp", -1)]
    )

    if not latest:
        return {"alerts_created": 0, "alerts": []}

    score = latest["overall_score"]

    if score < 60:
        sev = "high"
        msg = "Significant cognitive decline detected."
    elif score < 75:
        sev = "medium"
        msg = "Moderate decline detected — monitor closely."
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


# -----------------------------------------------
# ADVANCED AI INSIGHT (Gemini)
# -----------------------------------------------
@api_router.get("/cors-test")
async def cors_test():
    return {"message": "CORS OK"}


@api_router.post("/insights/generate")
async def advanced_ai_insight(request: GenerateInsightRequest, current_user=Depends(get_current_user)):
    uid = current_user["user_id"]

    records = await db.health_metrics.find(
        {"user_id": uid}, {"_id": 0}
    ).sort("timestamp", -1).limit(7).to_list(7)

    if not records:
        return {"message": "No data available"}

    records = list(reversed(records))

    overall = [r["overall_score"] for r in records]
    vocal = [r["vocal_score"] for r in records]
    movement = [r["movement_score"] for r in records]
    social = [r["social_score"] for r in records]

    X = np.arange(len(overall)).reshape(-1, 1)
    slope = round(LinearRegression().fit(X, overall).coef_[0], 3)
    vol = round(np.std(overall), 3)

    anomalies = {
        "vocal_anomaly": vocal[-1] < (statistics.mean(vocal) - 10),
        "movement_anomaly": movement[-1] < (statistics.mean(movement) - 10),
        "social_anomaly": social[-1] < (statistics.mean(social) - 10),
    }

    risk_score = (
        (100 - overall[-1]) * 0.5 +
        (vol * 2) +
        (10 if anomalies["movement_anomaly"] else 0) +
        (10 if anomalies["vocal_anomaly"] else 0)
    )

    risk = (
        "Low" if risk_score < 40 else
        "Medium" if risk_score < 70 else
        "High"
    )

    prompt = f"""
    Provide a clinical-style cognitive analysis.

    Trend slope: {slope}
    Volatility: {vol}
    Anomalies: {anomalies}
    Risk Score: {risk_score}
    Risk Category: {risk}
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
        logging.error(str(e))
        raise HTTPException(status_code=500, detail="Insight generation failed")


# -----------------------------------------------
# RESEARCHER ENDPOINTS
# -----------------------------------------------
@api_router.get("/research/patients")
async def patients(current_user=Depends(get_current_user)):
    if current_user["role"] != "researcher":
        raise HTTPException(status_code=403)

    pts = await db.users.find(
        {"role": "patient"}, {"_id": 0, "password_hash": 0}
    ).to_list(500)

    for p in pts:
        m = await db.health_metrics.find_one(
            {"user_id": p["id"]}, {"_id": 0}, sort=[("timestamp", -1)]
        )
        p["latest_metrics"] = m

    return pts


@api_router.get("/research/statistics")
async def stats(current_user=Depends(get_current_user)):
    if current_user["role"] != "researcher":
        raise HTTPException(status_code=403)

    total = await db.users.count_documents({"role": "patient"})
    sensor = await db.sensor_data.count_documents({})
    alerts = await db.tbi_alerts.count_documents({})

    recents = await db.health_metrics.find({}, {"_id": 0}).sort("timestamp", -1).limit(100).to_list(100)

    if recents:
        avg = {
            "overall": sum(m["overall_score"] for m in recents) / len(recents),
            "vocal": sum(m["vocal_score"] for m in recents) / len(recents),
            "movement": sum(m["movement_score"] for m in recents) / len(recents),
            "social": sum(m["social_score"] for m in recents) / len(recents),
        }
    else:
        avg = {"overall": 0, "vocal": 0, "movement": 0, "social": 0}

    return {
        "total_patients": total,
        "total_sensor_readings": sensor,
        "total_alerts": alerts,
        "average_scores": avg,
    }


# -----------------------------------------------
# ROOT ENDPOINT
# -----------------------------------------------
@api_router.get("/")
async def root():
    return {"message": "NeuroSense AI Backend Running"}


# Attach router
app.include_router(api_router)


# -----------------------------------------------
# CORS — FIXED FOR NETLIFY + RENDER
# -----------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://neuro-sense-ai.netlify.app",
        "https://*.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------
# SHUTDOWN
# -----------------------------------------------
@app.on_event("shutdown")
async def shutdown():
    client.close()
