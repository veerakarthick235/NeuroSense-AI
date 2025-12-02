import axios from "axios";

/*
  AUTO-DETECT BACKEND
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

// Auto attach token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export default api;
