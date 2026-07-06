import ProtectedRoute from "../components/auth/ProtectedRoute";
import Dashboard from "./admin/Dashboard";

export default function AdminDashboardProtected() {
  return (
    <ProtectedRoute allowedRoles={["admin"]}>
      <Dashboard />
    </ProtectedRoute>
  );
}