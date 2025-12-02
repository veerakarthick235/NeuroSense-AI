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
# Models
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
# Helpers
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
        "exp": datetime.now(timezone.utc) + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
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
# Auth Endpoints
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
# Sensor Simulation
# -----------------------------------------------
@api_router.post("/data/sensors/simulate")
async def simulate(current_user=Depends(get_current_user)):
    user_id = current_user["user_id"]
    output = []

    for typ in ["vocal", "movement", "social"]:
        metrics = generate_simulated_sensor_data(typ)
        obj = SensorData(user_id=user_id, data_type=typ, metrics=metrics)

        d = obj.model_dump()
        d["timestamp"] = d["timestamp"].isoformat()
        await db.sensor_data.insert_one(d)
        d.pop("_id", None)
        output.append(d)

    return {"message": "Simulated data generated", "data": output}


@api_router.get("/data/sensors")
async def sensor_data(data_type: Optional[str] = None, limit: int = 50, current_user=Depends(get_current_user)):
    q = {"user_id": current_user["user_id"]}
    if data_type:
        q["data_type"] = data_type

    data = await db.sensor_data.find(q, {"_id": 0}).sort("timestamp", -1).limit(limit).to_list(limit)
    return data


# -----------------------------------------------
# Metrics
# -----------------------------------------------
@api_router.get("/metrics/latest")
async def latest_metrics(current_user=Depends(get_current_user)):
    uid = current_user["user_id"]

    def get_latest(typ):
        return db.sensor_data.find_one(
            {"user_id": uid, "data_type": typ}, {"_id": 0}, sort=[("timestamp", -1)]
        )

    vocal = await get_latest("vocal")
    movement = await get_latest("movement")
    social = await get_latest("social")

    v = vocal["metrics"]["voice_quality"] * 100 if vocal else 0
    m = movement["metrics"]["gait_stability"] * 100 if movement else 0
    s = social["metrics"]["engagement_level"] * 100 if social else 0

    overall = (v + m + s) / 3

    obj = HealthMetrics(
        user_id=uid,
        vocal_score=round(v, 2),
        movement_score=round(m, 2),
        social_score=round(s, 2),
        overall_score=round(overall, 2),
    )

    d = obj.model_dump()
    d["timestamp"] = d["timestamp"].isoformat()

    await db.health_metrics.insert_one(d)
    d.pop("_id", None)

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
# Alerts
# -----------------------------------------------
@api_router.post("/alerts/check")
async def check_alerts(current_user=Depends(get_current_user)):
    uid = current_user["user_id"]
    latest = await db.health_metrics.find_one(
        {"user_id": uid}, {"_id": 0}, sort=[("timestamp", -1)]
    )

    if not latest:
        return {"message": "No metrics available"}

    score = latest["overall_score"]
    alerts = []

    if score < 60:
        sev = "high"
        msg = "Significant decline in health metrics — immediate care recommended."
    elif score < 75:
        sev = "medium"
        msg = "Moderate changes detected — monitor health closely."
    else:
        return {"alerts_created": 0, "alerts": []}

    alert = TBIAlert(
        user_id=uid,
        severity=sev,
        message=msg,
        metrics=latest
    )

    d = alert.model_dump()
    d["timestamp"] = d["timestamp"].isoformat()

    await db.tbi_alerts.insert_one(d)
    d.pop("_id", None)
    alerts.append(d)

    return {"alerts_created": len(alerts), "alerts": alerts}


@api_router.get("/alerts")
async def get_alerts(current_user=Depends(get_current_user)):
    uid = current_user["user_id"]
    data = await db.tbi_alerts.find({"user_id": uid}, {"_id": 0}).sort("timestamp", -1).to_list(20)
    return data


# -----------------------------------------------
# ADVANCED AI INSIGHT ENGINE (Gemini 2.5 Flash)
# -----------------------------------------------
@api_router.post("/insights/generate")
async def advanced_ai_insight(request: GenerateInsightRequest, current_user=Depends(get_current_user)):

    uid = current_user["user_id"]

    records = await db.health_metrics.find(
        {"user_id": uid}, {"_id": 0}
    ).sort("timestamp", -1).limit(7).to_list(7)

    if not records:
        return {"message": "No data available"}

    # Reverse to oldest → latest
    records = list(reversed(records))

    overall = [m["overall_score"] for m in records]
    vocal = [m["vocal_score"] for m in records]
    movement = [m["movement_score"] for m in records]
    social = [m["social_score"] for m in records]

    # Trend slope
    X = np.arange(len(overall)).reshape(-1, 1)
    lr = LinearRegression().fit(X, np.array(overall))
    slope = round(lr.coef_[0], 3)

    # Volatility
    vol = round(np.std(overall), 3)

    # Anomaly detection
    anomalies = {
        "vocal_anomaly": vocal[-1] < (statistics.mean(vocal) - 10),
        "movement_anomaly": movement[-1] < (statistics.mean(movement) - 10),
        "social_anomaly": social[-1] < (statistics.mean(social) - 10),
    }

    # TBI Risk Score
    risk_score = (
        (100 - overall[-1]) * 0.5 +
        (vol * 2) +
        (10 if anomalies["movement_anomaly"] else 0) +
        (10 if anomalies["vocal_anomaly"] else 0)
    )

    if risk_score < 40:
        risk = "Low"
    elif risk_score < 70:
        risk = "Medium"
    else:
        risk = "High"

    analysis_summary = f"""
    Trend slope: {slope}
    Volatility: {vol}
    Vocal anomaly: {anomalies['vocal_anomaly']}
    Movement anomaly: {anomalies['movement_anomaly']}
    Social anomaly: {anomalies['social_anomaly']}
    TBI Risk Score: {risk_score:.1f}
    Risk Category: {risk}
    """

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"""
        You are a clinical cognitive health AI.

        Analyze the following patient data and generate:

        1. Cognitive health summary
        2. Key observations and trend analysis
        3. Explanation of anomalies
        4. TBI risk interpretation
        5. 3–5 medical recommendations
        6. If risk is HIGH, include a warning section

        Patient Analytics:
        {analysis_summary}
        """

        response = model.generate_content(prompt)
        text = response.text.strip()

        obj = AIInsight(
            user_id=uid,
            insight_type="advanced_analysis",
            content=text
        )

        d = obj.model_dump()
        d["timestamp"] = d["timestamp"].isoformat()

        await db.ai_insights.insert_one(d)
        d.pop("_id", None)

        return d

    except Exception as e:
        logging.error(f"Gemini Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Insight generation failed")


# -----------------------------------------------
# Researcher Endpoints
# -----------------------------------------------
@api_router.get("/research/patients")
async def patients(current_user=Depends(get_current_user)):
    if current_user["role"] != "researcher":
        raise HTTPException(status_code=403, detail="Forbidden")

    pts = await db.users.find(
        {"role": "patient"}, {"_id": 0, "password_hash": 0}
    ).to_list(1000)

    for p in pts:
        m = await db.health_metrics.find_one(
            {"user_id": p["id"]}, {"_id": 0}, sort=[("timestamp", -1)]
        )
        p["latest_metrics"] = m

    return pts


@api_router.get("/research/statistics")
async def stats(current_user=Depends(get_current_user)):
    if current_user["role"] != "researcher":
        raise HTTPException(status_code=403, detail="Forbidden")

    total_patients = await db.users.count_documents({"role": "patient"})
    sensor_count = await db.sensor_data.count_documents({})
    alert_count = await db.tbi_alerts.count_documents({})

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
        "total_patients": total_patients,
        "total_sensor_readings": sensor_count,
        "total_alerts": alert_count,
        "average_scores": avg,
    }


# -----------------------------------------------
# ROOT + CORS
# -----------------------------------------------
@api_router.get("/")
async def root():
    return {"message": "NeuroSense AI Backend Running"}


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------
# Shutdown Cleanup
# -----------------------------------------------
@app.on_event("shutdown")
async def shutdown():
    client.close()
