import axios from "axios";

// Auto-detect environment and consolidate the URL construction
// This ensures that whether you are developing locally or running on Netlify, 
// the correct backend URL is used.
const BASE_URL = process.env.REACT_APP_BACKEND_URL 
  ? `${process.env.REACT_APP_BACKEND_URL}/api` 
  : "https://neuro-sense-ai.onrender.com/api"; // Production fallback

console.log("🔗 API Base URL:", BASE_URL);

const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  // Recommended to set to true if you are using session cookies, 
  // but acceptable as false if strictly relying on localStorage JWTs.
  withCredentials: true, 
});

// 1. Request Interceptor: Automatically attach the Bearer token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// 2. Response Interceptor: Automatically handle 401 Unauthorized (Token Expiration)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Check if the response exists, has a status, and the status is 401 (Unauthorized)
    if (error.response && error.response.status === 401) {
      console.error("Authentication Error: Token expired or invalid. Forcing logout.");
      
      // Clear the local token
      localStorage.removeItem("token");
      
      // Ensure this runs only in the browser and the user is not already on the login page
      if (typeof window !== 'undefined' && window.location.pathname !== "/login") {
        // Redirect the user. You might need to change "/login" to your actual path.
        window.location.assign("/login"); 
      }
    }
    
    // Reject promise for the calling component to handle generic errors
    return Promise.reject(error);
  }
);

export default api;
