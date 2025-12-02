# NeuroSense AI – Full Stack Application

This is the complete README for the NeuroSense AI project including **backend (FastAPI)** and **frontend (React + CRACO)**.

## 📌 Project Overview
NeuroSense AI is a multimodal cognitive‑health monitoring system that uses:
- Simulated sensor inputs (vocal, movement, social)
- Health scoring algorithms
- Advanced AI analysis using **Google Gemini 2.5 Flash**
- Researcher analytics dashboard
- Patient dashboard with charts, insights, alerts, and AI chat assistant

---

## 🧠 Features

### ✅ Patient Dashboard
- Simulate sensor data
- View health score (overall, vocal, movement, social)
- 7-day trend chart
- Multimodal radar graph
- TBI alerts
- AI-generated insights (Gemini)
- AI chat assistant

### 🧪 Researcher Dashboard
- View all patients
- Download dataset
- Population-level statistics
- Score distribution charts

---

# 📂 Project Structure
```
NeuroSense AI/
│
├── backend/
│   ├── server.py
│   ├── requirements.txt
│   └── .env
│
└── frontend/
    ├── src/
    ├── public/
    ├── package.json
    └── .env
```

---

# ⚙️ Backend Setup (FastAPI)

### 1️⃣ Install dependencies  
```
pip install -r requirements.txt
```

### 2️⃣ Create `.env`
```
MONGO_URL="your-mongo-url"
DB_NAME="neurosense_db"
JWT_SECRET="your-secret"
GEMINI_API_KEY="your-gemini-key"
CORS_ORIGINS="*"
```

### 3️⃣ Run backend  
```
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Backend will start at:  
👉 **http://localhost:8000**

---

# 🌐 Frontend Setup (React + CRACO)

### 1️⃣ Install dependencies  
```
npm install
```

### 2️⃣ Create `.env`  
```
REACT_APP_BACKEND_URL=http://localhost:8000
REACT_APP_ENABLE_VISUAL_EDITS=false
```

### 3️⃣ Start frontend  
```
npm start
```

Frontend will run at:  
👉 **http://localhost:3000**

---

# 🔌 API Endpoints

### Authentication
```
POST /api/auth/register
POST /api/auth/login
GET  /api/auth/me
```

### Sensor Data
```
POST /api/data/sensors/simulate
GET  /api/data/sensors
```

### Health Metrics
```
GET /api/metrics/latest
GET /api/metrics/history
```

### Alerts
```
POST /api/alerts/check
GET  /api/alerts
```

### Insights (Gemini AI)
```
POST /api/insights/generate
GET  /api/insights
```

### Researcher Tools
```
GET /api/research/patients
GET /api/research/statistics
GET /api/export/data
```

---

# 🤖 AI Processing – Gemini 2.5 Flash
The backend uses the following code to generate insights:

```python
model = genai.GenerativeModel("models/gemini-2.5-flash")
response = model.generate_content(prompt)
```

AI generates:
- Cognitive summary  
- Anomaly explanations  
- Trend analysis  
- TBI risk category  
- Medical recommendations  
- Warning section (for high risk)  

---
