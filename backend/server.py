from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal, List
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
import json
from bson.objectid import ObjectId 

# -------------------------------------------------------
# Configuration & Environment Variables
# -------------------------------------------------------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "neuro_sense_db")
JWT_SECRET = os.environ.get("JWT_SECRET", "super-secret-key")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not set. AI features will be disabled.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# -------------------------------------------------------
# FastAPI App Setup
# -------------------------------------------------------
app = FastAPI()
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

# -------------------------------------------------------
# Database Connection
# -------------------------------------------------------
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# -------------------------------------------------------
# Models
# -------------------------------------------------------
class User(BaseModel):
    model_config = ConfigDict(extra="allow")
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
    model_config = ConfigDict(extra="allow")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    data_type: Literal["vocal", "movement", "social"]
    metrics: dict
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class HealthMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    vocal_score: float
    movement_score: float
    social_score: float
    overall_score: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TBIAlert(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    severity: Literal["low", "medium", "high"]
    message: str
    metrics: dict
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AIInsight(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    insight_type: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class GenerateInsightRequest(BaseModel):
    user_id: str

# -------------------------------------------------------
# Utility Functions 
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
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        return jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
    except:
        raise HTTPException(status_code=401, detail="Invalid authentication")

def serialize_doc(doc):
    """Recursively converts MongoDB ObjectIds to strings and datetimes to ISO format."""
    if doc is None: return None
    if isinstance(doc, list):
        return [serialize_doc(x) for x in doc]
    if isinstance(doc, dict):
        new_doc = {}
        for k, v in doc.items():
            if k == "_id":
                new_doc["mongo_id"] = str(v)
            elif isinstance(v, ObjectId):
                new_doc[k] = str(v)
            elif isinstance(v, datetime):
                new_doc[k] = v.isoformat()
            elif isinstance(v, (dict, list)):
                new_doc[k] = serialize_doc(v)
            else:
                new_doc[k] = v
        return new_doc
    return doc

def to_datetime(val):
    if isinstance(val, datetime): return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace('Z', '+00:00'))
        except: return None
    return None

def generate_simulated_sensor_data(data_type: str):
    if data_type == "vocal":
        return {"pitch_mean": round(random.uniform(80, 250), 2), "pitch_variance": round(random.uniform(10, 50), 2), "speech_rate": round(random.uniform(100, 180), 2), "voice_quality": round(random.uniform(0.5, 1.0), 2)}
    elif data_type == "movement":
        return {"acceleration_x": round(random.uniform(-2, 2), 3), "gait_stability": round(random.uniform(0.4, 1.0), 2)}
    else:
        return {"engagement_level": round(random.uniform(0.4, 1.0), 2)}

# -------------------------------------------------------
# API Routes
# -------------------------------------------------------

@api_router.post("/auth/register")
async def register(data: UserRegister):
    if await db.users.find_one({"email": data.email}):
        raise HTTPException(status_code=400, detail="Email exists")
    user = User(email=data.email, password_hash=hash_password(data.password), name=data.name, role=data.role)
    await db.users.insert_one(user.model_dump())
    return {"token": create_token(user.id, user.email, user.role), "user": serialize_doc(user.model_dump())}

@api_router.post("/auth/login")
async def login(data: UserLogin):
    u = await db.users.find_one({"email": data.email})
    if not u or not verify_password(data.password, u["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"token": create_token(u["id"], u["email"], u["role"]), "user": serialize_doc(u)}

@api_router.get("/auth/me")
async def me(curr=Depends(get_current_user)):
    u = await db.users.find_one({"id": curr["user_id"]}, {"_id": 0, "password_hash": 0})
    return serialize_doc(u)

@api_router.post("/data/sensors/simulate")
async def simulate(curr=Depends(get_current_user)):
    uid = curr["user_id"]
    # 30s limit
    if await db.health_metrics.find_one({"user_id": uid, "timestamp": {"$gte": (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()}}):
        raise HTTPException(status_code=429, detail="Wait 30s")

    now = datetime.now(timezone.utc)
    v_m = generate_simulated_sensor_data("vocal")
    m_m = generate_simulated_sensor_data("movement")
    s_m = generate_simulated_sensor_data("social")

    metrics = HealthMetrics(
        user_id=uid, vocal_score=v_m["voice_quality"]*100, 
        movement_score=m_m["gait_stability"]*100, 
        social_score=s_m["engagement_level"]*100,
        overall_score=(v_m["voice_quality"] + m_m["gait_stability"] + s_m["engagement_level"])*100/3,
        timestamp=now
    )
    await db.health_metrics.insert_one(metrics.model_dump())
    return serialize_doc(metrics.model_dump())

@api_router.get("/metrics/latest")
async def get_latest(curr=Depends(get_current_user)):
    m = await db.health_metrics.find_one({"user_id": curr["user_id"]}, sort=[("timestamp", -1)])
    return serialize_doc(m) if m else {}

@api_router.get("/metrics/history")
async def get_history(days: int = 7, curr=Depends(get_current_user)):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cursor = db.health_metrics.find({"user_id": curr["user_id"], "timestamp": {"$gte": cutoff}})
    return [serialize_doc(doc) for doc in await cursor.to_list(None)]

@api_router.get("/alerts")
async def alerts(curr=Depends(get_current_user)):
    cursor = db.tbi_alerts.find({"user_id": curr["user_id"]}).sort("timestamp", -1)
    return [serialize_doc(doc) for doc in await cursor.to_list(None)]

@api_router.post("/insights/generate")
async def generate_insight_endpoint(req: GenerateInsightRequest, curr=Depends(get_current_user)):
    # Fetch data
    records = await db.health_metrics.find({"user_id": req.user_id}).sort("timestamp", 1).to_list(10)
    if len(records) < 3:
        raise HTTPException(status_code=404, detail="Need at least 3 data points")

    scores = [r["overall_score"] for r in records]
    X = np.arange(len(scores)).reshape(-1, 1)
    slope = LinearRegression().fit(X, scores).coef_[0]
    
    # Check last alert safely
    last_alert = await db.tbi_alerts.find_one({"user_id": req.user_id}, sort=[("timestamp", -1)])
    alert_time = to_datetime(last_alert["timestamp"]).strftime('%Y-%m-%d') if last_alert else "None"

    prompt = f"Analyze TBI recovery. Latest Score: {scores[-1]}. Trend Slope: {slope:.2f}. Last Alert Date: {alert_time}. Provide a clinical summary."
    
    try:
        # UPDATED TO GEMINI 2.0 FLASH
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        insight = AIInsight(user_id=req.user_id, insight_type="Advanced AI Analysis", content=response.text)
        await db.ai_insights.insert_one(insight.model_dump())
        return serialize_doc(insight.model_dump())
    except Exception as e:
        logging.error(f"Gemini Error: {e}")
        raise HTTPException(status_code=500, detail="AI generation failed")

@api_router.get("/insights")
async def insights(curr=Depends(get_current_user)):
    cursor = db.ai_insights.find({"user_id": curr["user_id"]}).sort("timestamp", -1)
    return [serialize_doc(doc) for doc in await cursor.to_list(None)]

# -------------------------------------------------------
# Researcher Routes
# -------------------------------------------------------
@api_router.get("/research/patients")
async def get_patients(curr=Depends(get_current_user)):
    if curr["role"] != "researcher": raise HTTPException(status_code=403)
    pts = await db.users.find({"role": "patient"}, {"password_hash": 0}).to_list(None)
    for p in pts:
        p["latest_metrics"] = await db.health_metrics.find_one({"user_id": p["id"]}, sort=[("timestamp", -1)])
    return [serialize_doc(p) for p in pts]

@api_router.get("/research/statistics")
async def get_stats(curr=Depends(get_current_user)):
    if curr["role"] != "researcher": raise HTTPException(status_code=403)
    count = await db.users.count_documents({"role": "patient"})
    metrics = await db.health_metrics.find().to_list(100)
    avg = sum(m["overall_score"] for m in metrics)/len(metrics) if metrics else 0
    return {"total_patients": count, "average_scores": {"overall": avg, "vocal": avg, "movement": avg, "social": avg}, "total_sensor_readings": len(metrics) * 3, "total_alerts": await db.tbi_alerts.count_documents({})}

@api_router.get("/export/data")
async def export_data(curr=Depends(get_current_user)):
    if curr["role"] != "researcher": raise HTTPException(status_code=403)
    data = {
        "users": await db.users.find({}, {"password_hash": 0}).to_list(None),
        "metrics": await db.health_metrics.find().to_list(None),
        "alerts": await db.tbi_alerts.find().to_list(None)
    }
    return JSONResponse(content=serialize_doc(data))

app.include_router(api_router)

# -------------------------------------------------------
# CORS & Options
# -------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://neuro-sense-ai.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{path:path}")
async def preflight(path: str): return {"status": "ok"}

@app.on_event("shutdown")
async def shutdown(): client.close()
