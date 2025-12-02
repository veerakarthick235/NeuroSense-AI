import axios from "axios";

// Auto-detect environment
const BASE_URL =
  process.env.REACT_APP_BACKEND_URL ||
  "https://neuro-sense-ai.onrender.com"; // fallback

console.log("🔗 API Base URL:", BASE_URL);

const api = axios.create({
  baseURL: `${BASE_URL}/api`,
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: false, // CORS-safe for Netlify → Render
});

// Automatically attach token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;
