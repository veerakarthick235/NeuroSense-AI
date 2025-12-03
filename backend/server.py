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

def generate_simulated_sensor_data(data_type: str):
    # Base simulation logic as provided
    if data_type == "vocal":
        # Introduce randomness with a bias towards 'normal' (e.g., score 75-100)
        base_quality = random.uniform(0.75, 1.0)
        
        # Randomly introduce a drop for variation/alerts ~20% of the time
        if random.random() < 0.2:
            base_quality = random.uniform(0.5, 0.75)
            
        return {
            "pitch_mean": round(random.uniform(80, 250), 2),
            "pitch_variance": round(random.uniform(10, 50), 2),
            "speech_rate": round(random.uniform(100, 180), 2),
            "pause_frequency": round(random.uniform(0.1, 0.5), 2),
            "voice_quality": round(base_quality, 2), # Score generator will use this
        }
    elif data_type == "movement":
        base_stability = random.uniform(0.75, 1.0)
        if random.random() < 0.2:
            base_stability = random.uniform(0.5, 0.75)

        return {
            "acceleration_x": round(random.uniform(-2, 2), 3),
            "acceleration_y": round(random.uniform(-2, 2), 3),
            "acceleration_z": round(random.uniform(-2, 2), 3),
            "gyro_x": round(random.uniform(-1, 1), 3),
            "gyro_y": round(random.uniform(-1, 1), 3),
            "gyro_z": round(random.uniform(-1, 1), 3),
            "gait_stability": round(base_stability, 2), # Score generator will use this
        }
    else:
        base_engagement = random.uniform(0.75, 1.0)
        if random.random() < 0.2:
            base_engagement = random.uniform(0.5, 0.75)

        return {
            "interaction_count": random.randint(5, 30),
            "response_time_avg": round(random.uniform(1, 5), 2),
            "sentiment_score": round(random.uniform(-0.5, 1.0), 2),
            "engagement_level": round(base_engagement, 2), # Score generator will use this
        }

# Helper to convert MongoDB documents to serializable format
def serialize_doc(doc):
    if doc:
        doc.pop('_id', None)
        if 'timestamp' in doc and isinstance(doc['timestamp'], datetime):
            doc['timestamp'] = doc['timestamp'].isoformat()
        if 'created_at' in doc and isinstance(doc['created_at'], datetime):
            doc['created_at'] = doc['created_at'].isoformat()
        
        # Recursively handle nested dictionaries like 'metrics' in alerts
        if 'metrics' in doc and isinstance(doc['metrics'], dict):
            if 'timestamp' in doc['metrics'] and isinstance(doc['metrics']['timestamp'], datetime):
                 doc['metrics']['timestamp'] = doc['metrics']['timestamp'].isoformat()
        
    return doc

# -------------------------------------------------------
# Auth Endpoints
# -------------------------------------------------------
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

    d = user.model_dump(by_alias=True, exclude_none=True)
    d["created_at"] = d["created_at"].isoformat()
    await db.users.insert_one(d)

    token = create_token(user.id, user.email, user.role)
    return {
        "token": token,
        "user": {"id": user.id, "email": user.email, "name": user.name, "role": user.role},
    }

@api_router.post("/auth/login")
async def login(data: UserLogin):
    user = await db.users.find_one({"email": data.email})
    if not user or not verify_password(data.password, user.get("password_hash", "")):
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
    user_doc = await db.users.find_one(
        {"id": current_user["user_id"]}, {"_id": 0, "password_hash": 0}
    )
    return serialize_doc(user_doc)


# -------------------------------------------------------
# Sensor & Metric Generation
# -------------------------------------------------------
@api_router.post("/data/sensors/simulate")
async def simulate(current_user=Depends(get_current_user)):
    """Generates 3 new sensor data points, calculates a new HealthMetric, and checks for Alerts."""
    uid = current_user["user_id"]

    # Rate limiting check: prevent creation if a metric already exists in the last 30 seconds
    thirty_seconds_ago = datetime.now(timezone.utc) - timedelta(seconds=30)
    recent_metric = await db.health_metrics.find_one({
        "user_id": uid,
        "timestamp": {"$gte": thirty_seconds_ago.isoformat()}
    })

    if recent_metric:
        raise HTTPException(status_code=429, detail="Data simulation is rate-limited. Please wait a moment.")

    # 1. Generate and insert new sensor data
    sensor_data_items = []
    current_time = datetime.now(timezone.utc)
    for data_type in ["vocal", "movement", "social"]:
        metrics = generate_simulated_sensor_data(data_type)
        obj = SensorData(user_id=uid, data_type=data_type, metrics=metrics, timestamp=current_time)
        d = obj.model_dump(by_alias=True, exclude_none=True)
        d["timestamp"] = d["timestamp"].isoformat()
        await db.sensor_data.insert_one(d)
        sensor_data_items.append(d)

    # 2. Calculate and store new health metrics
    vocal_m = [d["metrics"] for d in sensor_data_items if d["data_type"] == "vocal"][0]
    movement_m = [d["metrics"] for d in sensor_data_items if d["data_type"] == "movement"][0]
    social_m = [d["metrics"] for d in sensor_data_items if d["data_type"] == "social"][0]

    v_score = vocal_m["voice_quality"] * 100
    m_score = movement_m["gait_stability"] * 100
    s_score = social_m["engagement_level"] * 100
    
    overall = (v_score + m_score + s_score) / 3

    metrics_obj = HealthMetrics(
        user_id=uid,
        vocal_score=v_score,
        movement_score=m_score,
        social_score=s_score,
        overall_score=overall,
        timestamp=current_time
    )

    metrics_dict = metrics_obj.model_dump(by_alias=True, exclude_none=True)
    metrics_dict["timestamp"] = metrics_dict["timestamp"].isoformat()
    await db.health_metrics.insert_one(metrics_dict)
    
    # 3. Check and insert alerts for the new metrics
    alerts_list = []
    
    if overall < 60:
        severity = "high"
        message = "Critical decline detected across multiple metrics. Immediate medical consultation recommended."
    elif overall < 75:
        severity = "medium"
        message = "Moderate cognitive deviation detected. Monitor closely and consider a check-up."
    elif overall < 85:
        severity = "low"
        message = "Slight fluctuations detected. Continue monitoring daily activity."
    else:
        severity = None

    if severity:
        alert = TBIAlert(
            user_id=uid,
            severity=severity,
            message=message,
            metrics=metrics_dict,
            timestamp=current_time
        )
        d = alert.model_dump(by_alias=True, exclude_none=True)
        d["timestamp"] = d["timestamp"].isoformat()
        await db.tbi_alerts.insert_one(d)
        alerts_list.append(d)


    return {"message": "Simulated data and metrics generated", "latest_metrics": metrics_dict, "alerts": alerts_list}


# -------------------------------------------------------
# Metrics Endpoints
# -------------------------------------------------------
@api_router.get("/metrics/latest")
async def get_latest_metrics(current_user=Depends(get_current_user)):
    """Fetches the latest calculated HealthMetrics for the current user."""
    uid = current_user["user_id"]

    latest = await db.health_metrics.find_one(
        {"user_id": uid}, {"_id": 0}, sort=[("timestamp", -1)]
    )
    
    # Default return structure for a user with no data
    if not latest:
        return HealthMetrics(
            user_id=uid,
            vocal_score=0,
            movement_score=0,
            social_score=0,
            overall_score=0,
            timestamp=datetime.now(timezone.utc)
        ).model_dump(by_alias=True, exclude_none=True)

    return serialize_doc(latest)


@api_router.get("/metrics/history")
async def history(days: int = 7, current_user=Depends(get_current_user)):
    """Fetches the history of HealthMetrics for the last 'days'."""
    uid = current_user["user_id"]
    time_cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    cursor = db.health_metrics.find(
        {"user_id": uid, "timestamp": {"$gte": time_cutoff.isoformat()}},
        {"_id": 0}
    ).sort("timestamp", 1)

    return [serialize_doc(doc) for doc in await cursor.to_list(length=None)]


# -------------------------------------------------------
# Alerts Endpoints
# -------------------------------------------------------
@api_router.get("/alerts")
async def get_alerts(current_user=Depends(get_current_user)):
    """Fetches all TBI Alerts for the current user."""
    uid = current_user["user_id"]
    
    cursor = db.tbi_alerts.find(
        {"user_id": uid}, {"_id": 0}
    ).sort("timestamp", -1)

    return [serialize_doc(doc) for doc in await cursor.to_list(length=None)]

@api_router.post("/alerts/check")
async def check_alerts_manual(current_user=Depends(get_current_user)):
    """Manually checks the latest metric against alert thresholds and creates an alert if needed."""
    uid = current_user["user_id"]
    latest = await db.health_metrics.find_one(
        {"user_id": uid}, {"_id": 0}, sort=[("timestamp", -1)]
    )

    if not latest:
        raise HTTPException(status_code=404, detail="No latest metrics found to check alerts.")

    score = latest["overall_score"]
    alerts_list = []
    current_time = datetime.now(timezone.utc)
    
    # Check alert conditions
    severity = None
    message = None
    
    if score < 60:
        severity = "high"
        message = "Significant cognitive decline detected. High risk profile."
    elif score < 75:
        severity = "medium"
        message = "Moderate cognitive deviation detected. Elevated risk."
    
    if severity:
        # Check if an identical alert exists recently to avoid spamming the DB
        recent_alert = await db.tbi_alerts.find_one({
            "user_id": uid,
            "severity": severity,
            "timestamp": {"$gte": (current_time - timedelta(minutes=5)).isoformat()}
        })
        
        if not recent_alert:
            alert = TBIAlert(
                user_id=uid,
                severity=severity,
                message=message,
                metrics=latest,
                timestamp=current_time
            )

            d = alert.model_dump(by_alias=True, exclude_none=True)
            d["timestamp"] = d["timestamp"].isoformat()
            await db.tbi_alerts.insert_one(d)
            alerts_list.append(d)
        
    return {"alerts_created": len(alerts_list), "alerts": alerts_list}

# -------------------------------------------------------
# Gemini Insights Endpoints
# -------------------------------------------------------
@api_router.get("/insights")
async def get_insights(current_user=Depends(get_current_user)):
    """Fetches all AI Insights for the current user."""
    uid = current_user["user_id"]
    
    cursor = db.ai_insights.find(
        {"user_id": uid}, {"_id": 0}
    ).sort("timestamp", -1)

    return [serialize_doc(doc) for doc in await cursor.to_list(length=None)]

@api_router.post("/insights/generate")
async def advanced_ai_insight(
    request: GenerateInsightRequest, current_user=Depends(get_current_user)
):
    """Generates a new, advanced AI insight based on the patient's recent history."""
    if current_user["role"] != "patient":
        raise HTTPException(status_code=403, detail="Access denied. Patient role required for insight generation.")

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="AI Service is currently unavailable.")

    uid = request.user_id

    # Fetch last 7 days of health metrics
    time_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    
    # Fetch records and ensure timestamps are parsed as datetime for correct sorting/slicing
    records = (
        await db.health_metrics.find(
            {"user_id": uid, "timestamp": {"$gte": time_cutoff.isoformat()}},
            {"_id": 0}
        )
        .sort("timestamp", 1)
        .to_list(7)
    )

    if not records or len(records) < 3: # Require at least 3 points for a meaningful trend
        raise HTTPException(
            status_code=404, 
            detail="Insufficient data. Require at least 3 data points from the last 7 days to generate an insight."
        )

    # Prepare numerical data for linear regression and statistics
    overall_scores = [r["overall_score"] for r in records]
    vocal_scores = [r["vocal_score"] for r in records]
    movement_scores = [r["movement_score"] for r in records]
    social_scores = [r["social_score"] for r in records]
    
    X = np.arange(len(overall_scores)).reshape(-1, 1)

    # Calculate slope (trend)
    try:
        reg = LinearRegression().fit(X, np.array(overall_scores))
        slope = round(reg.coef_[0], 3)
    except Exception:
        slope = 0.0

    # Calculate volatility (standard deviation)
    vol = round(np.std(overall_scores), 3)

    # Anomalies detection (last score much lower than average)
    anomalies = {
        "vocal_anomaly": vocal_scores[-1] < (statistics.mean(vocal_scores) - 10),
        "movement_anomaly": movement_scores[-1] < (statistics.mean(movement_scores) - 10),
        "social_anomaly": social_scores[-1] < (statistics.mean(social_scores) - 10),
    }

    # Simplified Risk Calculation
    risk_score = (
        (100 - overall_scores[-1]) * 0.5
        + (vol * 2)
        + (10 if anomalies["movement_anomaly"] else 0)
        + (10 if anomalies["vocal_anomaly"] else 0)
    )

    risk = "Low" if risk_score < 40 else "Medium" if risk_score < 70 else "High"
    
    # Prepare the prompt for Gemini
    prompt = f"""
    You are an AI assistant providing a detailed clinical-style cognitive analysis for a patient based on their last {len(records)} health metrics. The goal is to identify trends related to Traumatic Brain Injury (TBI) recovery or decline.

    **Patient Data Metrics Summary:**
    - Latest Overall Score (0-100): {overall_scores[-1]:.1f}
    - Latest Vocal Score: {vocal_scores[-1]:.1f}
    - Latest Movement Score: {movement_scores[-1]:.1f}
    - Latest Social Score: {social_scores[-1]:.1f}
    - Overall Trend (Slope of the last {len(records)} days): {slope} (Positive means improvement, negative means decline)
    - Volatility (Standard Deviation of scores): {vol:.3f} (Higher means more fluctuation)
    - Recent Anomalies (Last score significantly lower than mean): {anomalies}

    **Calculated Risk Assessment:**
    - Risk Score: {risk_score:.2f}
    - Risk Category: {risk}

    **Instructions:**
    1. Analyze the provided data, commenting on the trend (improving/declining), stability (volatility), and specific metric areas with anomalies.
    2. Provide a concise summary of the patient's current cognitive status and risk level based on the metrics.
    3. Suggest clear, actionable recommendations for the patient and/or their caregiver for the next 7 days.
    4. Format the output as a single, clean block of text suitable for display, using paragraphs for readability. Do not include any titles, headers, or markdown formatting like lists or bullets.
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        obj = AIInsight(
            user_id=uid,
            insight_type="Advanced Analysis", # Changed to simple string
            content=response.text.strip(),
            timestamp=datetime.now(timezone.utc)
        )

        d = obj.model_dump(by_alias=True, exclude_none=True)
        d["timestamp"] = d["timestamp"].isoformat()
        await db.ai_insights.insert_one(d)

        return d

    except Exception as e:
        # Log the error for server-side debugging
        logging.error(f"Insight generation failed: {e}")
        # Raise HTTP exception for client feedback (this is the 500 error seen in console)
        raise HTTPException(status_code=500, detail=f"Insight generation failed due to an external error.")


# -------------------------------------------------------
# Researcher Endpoints
# -------------------------------------------------------
@api_router.get("/research/patients")
async def patients(current_user=Depends(get_current_user)):
    """Fetches a list of all patient users and their latest metrics for researchers."""
    if current_user["role"] != "researcher":
        raise HTTPException(status_code=403, detail="Access denied. Researcher role required.")

    pts = await db.users.find(
        {"role": "patient"}, {"_id": 0, "password_hash": 0}
    ).to_list(500)

    for p in pts:
        p_doc = serialize_doc(p)
        latest_metric = await db.health_metrics.find_one(
            {"user_id": p_doc["id"]}, {"_id": 0}, sort=[("timestamp", -1)]
        )
        p_doc["latest_metrics"] = serialize_doc(latest_metric)
        p.update(p_doc) # Update the original list item with serialized data

    return pts


@api_router.get("/research/statistics")
async def stats(current_user=Depends(get_current_user)):
    """Provides aggregated statistics over the patient population for researchers."""
    if current_user["role"] != "researcher":
        raise HTTPException(status_code=403, detail="Access denied. Researcher role required.")

    total = await db.users.count_documents({"role": "patient"})
    sensor_count = await db.sensor_data.count_documents({})
    alert_count = await db.tbi_alerts.count_documents({})

    recents = (
        await db.health_metrics.find({}, {"_id": 0, "overall_score": 1, "vocal_score": 1, "movement_score": 1, "social_score": 1})
        .sort("timestamp", -1)
        .limit(100)
        .to_list(100)
    )

    if recents:
        avg = {
            "overall": round(sum(r["overall_score"] for r in recents) / len(recents), 2),
            "vocal": round(sum(r["vocal_score"] for r in recents) / len(recents), 2),
            "movement": round(sum(r["movement_score"] for r in recents) / len(recents), 2),
            "social": round(sum(r["social_score"] for r in recents) / len(recents), 2),
        }
    else:
        avg = {"overall": 0.0, "vocal": 0.0, "movement": 0.0, "social": 0.0}

    return {
        "total_patients": total,
        "total_sensor_readings": sensor_count,
        "total_alerts": alert_count,
        "average_scores": avg,
    }

@api_router.get("/export/data")
async def export_data(current_user=Depends(get_current_user)):
    """Exports all database data related to patients, metrics, alerts, and insights as a single JSON file."""
    if current_user["role"] != "researcher":
        raise HTTPException(status_code=403, detail="Access denied. Researcher role required.")

    all_data = {}
    
    # Fetch all data from relevant collections
    users_cursor = db.users.find({}, {"_id": 0, "password_hash": 0})
    all_data['users'] = await users_cursor.to_list(length=None)

    metrics_cursor = db.health_metrics.find({}, {"_id": 0})
    all_data['health_metrics'] = await metrics_cursor.to_list(length=None)

    alerts_cursor = db.tbi_alerts.find({}, {"_id": 0})
    all_data['tbi_alerts'] = await alerts_cursor.to_list(length=None)
    
    insights_cursor = db.ai_insights.find({}, {"_id": 0})
    all_data['ai_insights'] = await insights_cursor.to_list(length=None)

    # Convert all documents to serializable format (especially datetimes)
    def deep_serialize(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, list):
            return [deep_serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {k: deep_serialize(v) for k, v in obj.items() if k != '_id'}
        return obj

    serialized_data = deep_serialize(all_data)

    # Return as JSON file response
    return JSONResponse(
        content=json.dumps(serialized_data, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=neuro-sense-data.json"}
    )


# -------------------------------------------------------
# Root Endpoint
# -------------------------------------------------------
@api_router.get("/")
async def root():
    return {"message": "NeuroSense AI Backend Running"}


app.include_router(api_router)

# -------------------------------------------------------
# CORS Middleware
# -------------------------------------------------------
# Configured to handle Netlify dynamic subdomains and localhost during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://neuro-sense-ai.netlify.app",
        # Explicitly adding the Render URL for safety in case of self-requests/redirects
        os.environ.get("BACKEND_URL", "https://neuro-sense-ai.onrender.com")
    ],
    # This regex is meant to cover all Netlify previews and the main domain
    allow_origin_regex=r"https?:\/\/(localhost(:[0-9]+)?|([a-zA-Z0-9\-]+\.netlify\.app))", 
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers (including Authorization for JWT)
)

# Render needs explicit OPTIONS preflight handling for some environments
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return {"status": "ok"}


# -------------------------------------------------------
# Shutdown Event
# -------------------------------------------------------
@app.on_event("shutdown")
async def shutdown():
    print("Closing MongoDB connection...")
    client.close()
    print("MongoDB connection closed.")
