import { NavLink } from "react-router-dom";
import "./Sidebar.css";

const doctorNavItems = [
  { to: "/doctor-dashboard", label: "Doctor Dashboard" },
  { to: "/profile-management", label: "Profile Management" },

  { to: "/image-prediction", label: "Image Prediction" },
  { to: "/prediction-history", label: "Prediction History" },
  { to: "/reports", label: "Reports" },

  { to: "/login", label: "Logout" },
];

const adminNavItems = [
  { to: "/admin-dashboard", label: "Admin Analytics" },
  { to: "/manage-doctors", label: "Doctor Management" },
  { to: "/system-monitoring", label: "System Monitoring" },
  { to: "/login", label: "Logout" },
];

export default function Sidebar({ role = "admin" }) {
  const navItems = role === "doctor" ? doctorNavItems : adminNavItems;

  return (
    <aside className="sidebar">
      <div className="sidebarLogo">
        <span className="sidebarLogoMark">🩺</span>
        <span className="sidebarLogoText">Skin Disease System</span>
      </div>

      <nav className="sidebarNav">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              isActive ? "sidebarLink isActive" : "sidebarLink"
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
