import ProtectedRoute from "../components/auth/ProtectedRoute";
import DoctorDashboard from "./DoctorDashboard";

export default function DoctorDashboardProtected() {
  return (
    <ProtectedRoute allowedRoles={["doctor"]}>
      <DoctorDashboard />
    </ProtectedRoute>
  );
}

