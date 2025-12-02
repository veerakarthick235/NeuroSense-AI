import axios from "axios";

/*
  AUTO-DETECT BACKEND URL
*/
const BASE_URL =
  process.env.REACT_APP_BACKEND_URL ||
  "https://neuro-sense-ai.onrender.com";

console.log("🔗 API Base URL:", BASE_URL);

const api = axios.create({
  baseURL: `${BASE_URL}/api`,
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: false,
});

// Automatically attach JWT token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

/*
  -------- NAMED EXPORTS (Fix for Netlify) --------
*/

// AUTH
export const registerUser = (data) => api.post("/auth/register", data);
export const loginUser = (data) => api.post("/auth/login", data);
export const fetchMe = () => api.get("/auth/me");

// METRICS
export const fetchLatestMetrics = () => api.get("/metrics/latest");
export const fetchMetricHistory = (days = 7) =>
  api.get(`/metrics/history?days=${days}`);

// SENSOR
export const simulateSensorData = () => api.post("/data/sensors/simulate");

// ALERTS
export const checkAlerts = () => api.post("/alerts/check");

// INSIGHTS
export const generateInsight = (userId) =>
  api.post("/insights/generate", { user_id: userId });

// RESEARCHER
export const fetchPatients = () => api.get("/research/patients");
export const fetchStatistics = () => api.get("/research/statistics");

// DEFAULT EXPORT (for imports like `import api from '../api'`)
export default api;
