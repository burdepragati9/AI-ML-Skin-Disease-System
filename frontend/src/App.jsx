import { BrowserRouter, Routes, Route, Navigate, useLocation } from "react-router-dom";
import { useEffect } from "react";

import Login from "./pages/Login";
import Signup from "./pages/Signup";
import ForgotPassword from "./pages/ForgotPassword";
import ResetPassword from "./pages/ResetPassword";

import DoctorDashboardProtected from "./pages/DoctorDashboardProtected";
import AdminDashboardProtected from "./pages/AdminDashboardProtected";

import ImagePrediction from "./pages/prediction/ImagePrediction";
import PredictionHistory from "./pages/prediction/PredictionHistory";
import ProfileManagement from "./pages/ProfileManagement";
import Reports from "./pages/Reports";
import ProtectedRoute from "./components/auth/ProtectedRoute";
import ManageDoctors from "./pages/admin/ManageDoctors";
import SystemMonitoring from "./pages/admin/SystemMonitoring";


import api from "./services/api";
import { getToken, logout } from "./services/auth";


function SessionKeeper() {
  const location = useLocation();

  useEffect(() => {
    // Validate token on app load / route changes (keeps session alive)
    const token = getToken();
    if (!token) return;

    api
      .get("/auth/me")
      .catch(() => {
        logout();
      });
  }, [location.pathname]);

  return null;
}

function RoleRedirect() {
  const token = getToken();

  if (!token) return <Navigate to="/login" replace />;

  // If role is present in localStorage, redirect quickly.
  const role = localStorage.getItem("role");
  if (role === "doctor") return <Navigate to="/doctor-dashboard" replace />;
  if (role === "admin") return <Navigate to="/admin-dashboard" replace />;

  return <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <BrowserRouter>
      <SessionKeeper />
      <Routes>
        <Route path="/" element={<RoleRedirect />} />

        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route path="/reset-password" element={<ResetPassword />} />

        <Route path="/doctor-dashboard" element={<DoctorDashboardProtected />} />
        <Route path="/admin-dashboard" element={<AdminDashboardProtected />} />

        {/* legacy */}
        <Route path="/dashboard" element={<RoleRedirect />} />

        <Route
          path="/image-prediction"
          element={
            <ProtectedRoute allowedRoles={["doctor", "admin"]}>
              <ImagePrediction />
            </ProtectedRoute>
          }
        />

        <Route
          path="/prediction-history"
          element={
            <ProtectedRoute allowedRoles={["doctor"]}>
              <PredictionHistory />
            </ProtectedRoute>
          }
        />


        <Route
          path="/profile-management"
          element={
            <ProtectedRoute allowedRoles={["doctor"]}>
              <ProfileManagement />
            </ProtectedRoute>
          }
        />

        <Route
          path="/reports"
          element={
            <ProtectedRoute allowedRoles={["doctor"]}>
              <Reports />
            </ProtectedRoute>
          }
        />

        <Route
          path="/manage-doctors"
          element={
            <ProtectedRoute allowedRoles={["admin"]}>
              <ManageDoctors />
            </ProtectedRoute>
          }
        />

        <Route
          path="/system-monitoring"
          element={
            <ProtectedRoute allowedRoles={["admin"]}>
              <SystemMonitoring />
            </ProtectedRoute>
          }
        />

        <Route path="*" element={<RoleRedirect />} />
      </Routes>
    </BrowserRouter>
  );
}

