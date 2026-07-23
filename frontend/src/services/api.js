import axios from "axios";
import { logout } from "./auth";

// ==============================
// AXIOS INSTANCE
// ==============================
const api = axios.create({
  baseURL: "http://127.0.0.1:8000",
});

// ==============================
// REQUEST INTERCEPTOR
// Automatically attach JWT token
// ==============================
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("access_token");

    // Ensure headers object exists
    config.headers = config.headers || {};

    // Attach token if available
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    } else {
      delete config.headers.Authorization;
    }

    return config;
  },
  (error) => Promise.reject(error)
);

// ==============================
// RESPONSE INTERCEPTOR
// Auto logout on invalid token (not business rule violations)
// ==============================
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error?.response?.status;
    const detail = error?.response?.data?.detail || '';

    // Only auto-logout on actual authentication failures
    // Do NOT logout on business rule violations like prediction limits
    if (status === 401) {
      console.warn("[API] Unauthorized (401) → Logging out user");
      logout();
    } else if (status === 403) {
      // Check if this is a business rule violation (like prediction limit)
      // If the detail mentions 'limit' or 'free', it's not an auth failure
      if (detail.includes('limit') || detail.includes('free') || detail.includes('prediction')) {
        console.warn("[API] Business rule violation (403) → NOT logging out");
        // Do NOT logout - let the component handle the error message
      } else {
        console.warn("[API] Unauthorized (403) → Logging out user");
        logout();
      }
    }

    return Promise.reject(error);
  }
);

export default api;