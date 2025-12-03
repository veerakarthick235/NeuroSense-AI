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
# Load environment variables
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
# FastAPI App
# -------------------------------------------------------
app = FastAPI()
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

# -------------------------------------------------------
# Database
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

# -------------------------------------------------------
# Auth
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

    d = user.model_dump()
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
    return await db.users.find_one(
        {"id": current_user["user_id"]}, {"_id": 0, "password_hash": 0}
    )

# -------------------------------------------------------
# Sensor Simulation
# -------------------------------------------------------
@api_router.post("/data/sensors/simulate")
async def simulate(current_user=Depends(get_current_user)):
    uid = current_user["user_id"]

    # Check if a health metric entry already exists for the last minute to prevent spamming
    one_minute_ago = datetime.now(timezone.utc) - timedelta(minutes=1)
    recent_metric = await db.health_metrics.find_one({
        "user_id": uid,
        "timestamp": {"$gte": one_minute_ago.isoformat()}
    })

    if recent_metric:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")

    # 1. Generate and insert new sensor data
    sensor_data_items = []
    for data_type in ["vocal", "movement", "social"]:
        metrics = generate_simulated_sensor_data(data_type)
        obj = SensorData(user_id=uid, data_type=data_type, metrics=metrics)
        d = obj.model_dump()
        d["timestamp"] = d["timestamp"].isoformat()
        await db.sensor_data.insert_one(d)
        sensor_data_items.append(d)

    # 2. Calculate and store new health metrics
    vocal = [d for d in sensor_data_items if d["data_type"] == "vocal"][0]
    movement = [d for d in sensor_data_items if d["data_type"] == "movement"][0]
    social = [d for d in sensor_data_items if d["data_type"] == "social"][0]

    v = vocal["metrics"]["voice_quality"] * 100
    m = movement["metrics"]["gait_stability"] * 100
    s = social["metrics"]["engagement_level"] * 100
    
    # Randomly adjust scores to ensure variety and potential dips for alerts
    v = max(0, min(100, v + random.uniform(-10, 10)))
    m = max(0, min(100, m + random.uniform(-10, 10)))
    s = max(0, min(100, s + random.uniform(-10, 10)))

    overall = (v + m + s) / 3

    metrics_obj = HealthMetrics(
        user_id=uid,
        vocal_score=v,
        movement_score=m,
        social_score=s,
        overall_score=overall,
    )

    metrics_dict = metrics_obj.model_dump()
    metrics_dict["timestamp"] = metrics_dict["timestamp"].isoformat()
    await db.health_metrics.insert_one(metrics_dict)
    
    # 3. Check and insert alerts for the new metrics
    alert_count = 0
    alerts_list = []
    
    # Very basic alert logic (can be expanded)
    if overall < 60:
        sev = "high"
        msg = "Critical decline detected across multiple metrics. Immediate medical consultation recommended."
        alert_count += 1
    elif overall < 75:
        sev = "medium"
        msg = "Noticeable deviation in recent activity. Monitor closely and consider a check-up."
        alert_count += 1
    elif overall < 85:
        sev = "low"
        msg = "Slight fluctuations detected. Continue monitoring daily activity."
        alert_count += 1
    
    if alert_count > 0:
        alert = TBIAlert(
            user_id=uid,
            severity=sev,
            message=msg,
            metrics=metrics_dict,
        )
        d = alert.model_dump()
        d["timestamp"] = d["timestamp"].isoformat()
        await db.tbi_alerts.insert_one(d)
        alerts_list.append(d)


    return {"message": "Simulated data and metrics generated", "latest_metrics": metrics_dict, "alerts": alerts_list}


# -------------------------------------------------------
# Metrics Endpoints
# -------------------------------------------------------
@api_router.get("/metrics/latest")
async def get_latest_metrics(current_user=Depends(get_current_user)):
    uid = current_user["user_id"]

    latest = await db.health_metrics.find_one(
        {"user_id": uid}, {"_id": 0}, sort=[("timestamp", -1)]
    )
    
    if not latest:
        # Return default structure if no data exists
        return {
            "id": "none",
            "user_id": uid,
            "vocal_score": 0,
            "movement_score": 0,
            "social_score": 0,
            "overall_score": 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    return latest


@api_router.get("/metrics/history")
async def history(days: int = 7, current_user=Depends(get_current_user)):
    uid = current_user["user_id"]
    time_cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    cursor = db.health_metrics.find(
        {"user_id": uid, "timestamp": {"$gte": time_cutoff.isoformat()}},
        {"_id": 0}
    ).sort("timestamp", 1)

    # Convert cursor to list and return
    return await cursor.to_list(length=None)

# -------------------------------------------------------
# Alerts Endpoints
# -------------------------------------------------------
@api_router.get("/alerts")
async def get_alerts(current_user=Depends(get_current_user)):
    uid = current_user["user_id"]
    
    # Fetch all alerts for the user, newest first
    cursor = db.tbi_alerts.find(
        {"user_id": uid}, {"_id": 0}
    ).sort("timestamp", -1)

    return await cursor.to_list(length=None)

@api_router.post("/alerts/check")
async def check_alerts(current_user=Depends(get_current_user)):
    uid = current_user["user_id"]
    latest = await db.health_metrics.find_one(
        {"user_id": uid}, {"_id": 0}, sort=[("timestamp", -1)]
    )

    if not latest:
        return {"alerts_created": 0, "alerts": []}

    score = latest["overall_score"]
    alerts_list = []
    
    # Simplified alert logic, could be more complex with trend analysis etc.
    severity = None
    message = None
    
    if score < 60:
        severity = "high"
        message = "Critical decline detected across multiple metrics. Immediate medical consultation recommended."
    elif score < 75:
        severity = "medium"
        message = "Moderate cognitive deviation detected. Monitor closely and consider a check-up."
    
    if severity:
        alert = TBIAlert(
            user_id=uid,
            severity=severity,
            message=message,
            metrics=latest,
        )

        d = alert.model_dump()
        d["timestamp"] = d["timestamp"].isoformat()
        await db.tbi_alerts.insert_one(d)
        alerts_list.append(d)
        
    return {"alerts_created": len(alerts_list), "alerts": alerts_list}

# -------------------------------------------------------
# Insights Endpoints
# -------------------------------------------------------
@api_router.get("/insights")
async def get_insights(current_user=Depends(get_current_user)):
    uid = current_user["user_id"]
    
    # Fetch all insights for the user, newest first
    cursor = db.ai_insights.find(
        {"user_id": uid}, {"_id": 0}
    ).sort("timestamp", -1)

    return await cursor.to_list(length=None)

@api_router.post("/insights/generate")
async def advanced_ai_insight(
    request: GenerateInsightRequest, current_user=Depends(get_current_user)
):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="AI Service is currently unavailable.")

    uid = request.user_id

    # Fetch last 7 days of health metrics
    time_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    
    records = (
        await db.health_metrics.find(
            {"user_id": uid, "timestamp": {"$gte": time_cutoff.isoformat()}},
            {"_id": 0}
        )
        .sort("timestamp", 1)
        .to_list(7)
    )

    if not records:
        raise HTTPException(status_code=404, detail="Not enough data to generate insight.")

    # Filter out records where 'timestamp' isn't a string (shouldn't happen with the model_dump, but for safety)
    parsed_records = []
    for r in records:
        if isinstance(r.get("timestamp"), str):
            try:
                r["timestamp"] = datetime.fromisoformat(r["timestamp"].replace('Z', '+00:00'))
                parsed_records.append(r)
            except ValueError:
                # Handle cases where timestamp might be malformed, skip
                pass
        
    records = sorted(parsed_records, key=lambda x: x["timestamp"])

    # If there are fewer than 2 data points, we can't calculate a trend/volatility meaningfully
    if len(records) < 2:
        return AIInsight(
            user_id=uid,
            insight_type="data_insufficiency",
            content="Insufficient data points to perform meaningful trend analysis. Please ensure there are at least two days of data.",
        ).model_dump(by_alias=True, exclude_none=True)

    # Extracting data for analysis
    overall = [r["overall_score"] for r in records]
    vocal = [r["vocal_score"] for r in records]
    movement = [r["movement_score"] for r in records]
    social = [r["social_score"] for r in records]

    # Time series analysis (Simple Linear Regression for trend)
    X = np.arange(len(overall)).reshape(-1, 1)
    
    # Using float values for consistency with MongoDB storage
    overall_float = np.array([float(s) for s in overall])

    # Calculate slope (trend)
    try:
        reg = LinearRegression().fit(X, overall_float)
        slope = round(reg.coef_[0], 3)
    except Exception:
        slope = 0.0

    # Calculate volatility (standard deviation)
    vol = round(np.std(overall_float), 3)

    # Anomalies detection (example: score significantly lower than average)
    avg_vocal = statistics.mean(vocal)
    avg_movement = statistics.mean(movement)
    avg_social = statistics.mean(social)

    anomalies = {
        "vocal_anomaly": vocal[-1] < (avg_vocal - 10),
        "movement_anomaly": movement[-1] < (avg_movement - 10),
        "social_anomaly": social[-1] < (avg_social - 10),
    }

    # Simplified Risk Calculation based on current score, volatility, and anomalies
    risk_score = (
        (100 - overall[-1]) * 0.5
        + (vol * 2)
        + (10 if anomalies["movement_anomaly"] else 0)
        + (10 if anomalies["vocal_anomaly"] else 0)
    )

    risk = "Low" if risk_score < 40 else "Medium" if risk_score < 70 else "High"
    
    # Find last alert for context
    last_alert = await db.tbi_alerts.find_one(
        {"user_id": uid}, {"_id": 0, "message": 1, "severity": 1, "timestamp": 1}, sort=[("timestamp", -1)]
    )

    prompt = f"""
    You are an AI assistant providing a detailed clinical-style cognitive analysis for a patient based on their last {len(records)} health metrics.

    **Patient Data Metrics Summary:**
    - Latest Overall Score (0-100): {overall[-1]:.1f}
    - Latest Vocal Score: {vocal[-1]:.1f}
    - Latest Movement Score: {m:.1f}
    - Latest Social Score: {s:.1f}
    - Overall Trend (Slope of last {len(records)} days): {slope} (Positive means improvement, negative means decline)
    - Volatility (Standard Deviation): {vol:.3f} (Higher means more fluctuation)
    - Recent Anomalies: {anomalies}

    **Calculated Risk Assessment:**
    - Risk Score: {risk_score:.2f}
    - Risk Category: {risk}
    
    {f"**Previous Alert:** A {last_alert['severity']} alert was issued on {last_alert['timestamp'].strftime('%Y-%m-%d')} for: {last_alert['message']}" if last_alert else ""}

    **Instructions:**
    1. Analyze the provided data, focusing on the trend, volatility, and any observed anomalies.
    2. Provide a concise summary of the patient's current cognitive status and risk level based on the metrics.
    3. Suggest clear, actionable recommendations for the patient and/or their caregiver.
    4. Format the output as a single, clean block of text suitable for display, using paragraphs for readability. Do not include any titles, headers, or markdown formatting like lists or bullets.
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        obj = AIInsight(
            user_id=uid,
            insight_type="advanced_analysis",
            content=response.text.strip(),
        )

        d = obj.model_dump()
        d["timestamp"] = d["timestamp"].isoformat()
        await db.ai_insights.insert_one(d)

        return d

    except Exception as e:
        logging.error(f"Error generating insight for user {uid}: {e}")
        raise HTTPException(status_code=500, detail="Insight generation failed.")


# -------------------------------------------------------
# Researcher Endpoints
# -------------------------------------------------------
@api_router.get("/research/patients")
async def patients(current_user=Depends(get_current_user)):
    if current_user["role"] != "researcher":
        raise HTTPException(status_code=403, detail="Access denied. Researcher role required.")

    pts = await db.users.find(
        {"role": "patient"}, {"_id": 0, "password_hash": 0}
    ).to_list(500)

    # Fetch latest metrics for each patient
    for p in pts:
        latest_metric = await db.health_metrics.find_one(
            {"user_id": p["id"]}, {"_id": 0}, sort=[("timestamp", -1)]
        )
        # Convert datetime objects to ISO format strings for JSON serialization
        if latest_metric:
             latest_metric["timestamp"] = latest_metric["timestamp"].isoformat()
        p["latest_metrics"] = latest_metric

    return pts


@api_router.get("/research/statistics")
async def stats(current_user=Depends(get_current_user)):
    if current_user["role"] != "researcher":
        raise HTTPException(status_code=403, detail="Access denied. Researcher role required.")

    total = await db.users.count_documents({"role": "patient"})
    sensor_count = await db.sensor_data.count_documents({})
    alert_count = await db.tbi_alerts.count_documents({})

    # Fetch up to the last 100 health metrics across all users for aggregation
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
    if current_user["role"] != "researcher":
        raise HTTPException(status_code=403, detail="Access denied. Researcher role required.")

    all_data = {}
    
    # Fetch all user data
    users_cursor = db.users.find({}, {"_id": 0, "password_hash": 0})
    users = await users_cursor.to_list(length=None)
    all_data['users'] = users

    # Fetch all health metrics
    metrics_cursor = db.health_metrics.find({}, {"_id": 0})
    metrics = await metrics_cursor.to_list(length=None)
    all_data['health_metrics'] = metrics

    # Fetch all alerts
    alerts_cursor = db.tbi_alerts.find({}, {"_id": 0})
    alerts = await alerts_cursor.to_list(length=None)
    all_data['tbi_alerts'] = alerts
    
    # Fetch all insights
    insights_cursor = db.ai_insights.find({}, {"_id": 0})
    insights = await insights_cursor.to_list(length=None)
    all_data['ai_insights'] = insights

    # Convert all datetime objects to ISO strings for export
    def format_document(doc):
        for key, value in doc.items():
            if isinstance(value, datetime):
                doc[key] = value.isoformat()
            elif isinstance(value, dict):
                doc[key] = format_document(value)
        return doc

    all_data['users'] = [format_document(user) for user in all_data['users']]
    all_data['health_metrics'] = [format_document(metric) for metric in all_data['health_metrics']]
    all_data['tbi_alerts'] = [format_document(alert) for alert in all_data['tbi_alerts']]
    all_data['ai_insights'] = [format_document(insight) for insight in all_data['ai_insights']]

    # Return as JSON file
    return JSONResponse(
        content=json.dumps(all_data, default=str),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=neuro-sense-data.json"}
    )


# -------------------------------------------------------
# Root
# -------------------------------------------------------
@api_router.get("/")
async def root():
    return {"message": "NeuroSense AI Backend Running"}


app.include_router(api_router)

# -------------------------------------------------------
# PRODUCTION CORS — FINAL FIX FOR NETLIFY + RENDER
# -------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Added localhost for development
        "https://neuro-sense-ai.netlify.app",
    ],
    allow_origin_regex=r"https?:\/\/(localhost(:[0-9]+)?|([a-zA-Z0-9\-]+\.netlify\.app))",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Render needs OPTIONS for everything
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return {"status": "ok"}


# -------------------------------------------------------
# Shutdown
# -------------------------------------------------------
@app.on_event("shutdown")
async def shutdown():
    print("Closing MongoDB connection...")
    client.close()
    print("MongoDB connection closed.")
