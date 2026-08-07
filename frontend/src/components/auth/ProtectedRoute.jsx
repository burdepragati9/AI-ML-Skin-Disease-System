import { Navigate } from "react-router-dom";
import { useEffect, useState } from "react";

import { getToken, getUserRole, logout } from "../../services/auth";
import { validateSessionOnce } from "../../services/session";

export default function ProtectedRoute({ allowedRoles, children }) {

  const [verified, setVerified] = useState(false);
  const [allowed, setAllowed] = useState(false);
  const allowedRolesKey = (allowedRoles || []).join("|");

  useEffect(() => {
    let cancelled = false;

    async function verify() {
      const token = getToken();
      if (!token) {
        logout();
        if (!cancelled) {
          setAllowed(false);
          setVerified(true);
        }
        return;
      }

      try {
        // Validate token + session
        await validateSessionOnce();

        const role = getUserRole();
        if (!allowedRoles || allowedRoles.length === 0) {
          if (!cancelled) {
            setAllowed(true);
            setVerified(true);
          }
          return;
        }

        const ok = allowedRoles.includes(role);
        if (!cancelled) {
          setAllowed(ok);
          setVerified(true);
        }
      } catch {
        logout();
        if (!cancelled) {
          setAllowed(false);
          setVerified(true);
        }
      }
    }

    verify();
    return () => {
      cancelled = true;
    };
  }, [allowedRolesKey]);

  if (!verified) return null;

  if (!allowed) return <Navigate to="/login" replace />;
  return children ? children : <Outlet />;
}



