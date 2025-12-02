import axios from "axios";

/*
  ---------------------------------------------------------
  AUTO-DETECT BACKEND URL
  ---------------------------------------------------------
  Priority:
  1. REACT_APP_BACKEND_URL  (Netlify environment variable)
  2. Render backend direct URL (fallback)
  ---------------------------------------------------------
*/

const BASE_URL =
  process.env.REACT_APP_BACKEND_URL ||
  "https://neuro-sense-ai.onrender.com"; // fallback for localhost builds

console.log("🔗 API Base URL:", BASE_URL);

// Create axios instance
const api = axios.create({
  baseURL: `${BASE_URL}/api`,
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: false, // Netlify + Render CORS safe
});

/*
  ---------------------------------------------------------
  AUTO-ATTACH JWT TOKEN
  ---------------------------------------------------------
*/
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

/*
  ---------------------------------------------------------
  FIXED, SAFE API ENDPOINT WRAPPERS
  ---------------------------------------------------------
  These functions ensure the frontend ALWAYS uses the
  correct HTTP method and correct backend routes.
  ---------------------------------------------------------
*/

// AUTH
export const registerUser = (data) => api.post("/auth/register", data);
export const loginUser = (data) => api.post("/auth/login", data);
export const fetchMe = () => api.get("/auth/me");

// METRICS
export const fetchLatestMetrics = () => api.get("/metrics/latest");
export const fetchMetricHistory = (days = 7) =>
  api.get(`/metrics/history?days=${days}`);

// SENSOR SIMULATION
export const simulateSensorData = () =>
  api.post("/data/sensors/simulate");

// ALERTS
export const checkAlerts = () => api.post("/alerts/check");

// AI INSIGHTS (POST, not GET)
export const generateInsight = (userId) =>
  api.post("/insights/generate", { user_id: userId });

// RESEARCHER
export const fetchPatients = () => api.get("/research/patients");
export const fetchStatistics = () => api.get("/research/statistics");

export default api;
