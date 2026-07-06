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
// Auto logout on invalid token
// ==============================
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error?.response?.status;

    if (status === 401 || status === 403) {
      console.warn("[API] Unauthorized → Logging out user");
      logout();
    }

    return Promise.reject(error);
  }
);

export default api;