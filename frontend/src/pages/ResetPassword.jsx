import { useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import api from "../services/api";

export default function ResetPassword() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const initialToken = searchParams.get("token") || "";

  const [token, setToken] = useState(initialToken);
  const [newPassword, setNewPassword] = useState("");
  const [status, setStatus] = useState({
    variant: "idle",
    message: "",
  });

  const isValid = useMemo(() => {
    return (token || "").trim().length > 0 && (newPassword || "").length >= 6;
  }, [token, newPassword]);

  async function onSubmit(e) {
    e.preventDefault();

    setStatus({ variant: "idle", message: "" });

    try {
      const res = await api.post("/auth/reset-password", {
        token: (token || "").trim(),
        new_password: newPassword,
      });

      if (res?.data?.success) {
        setStatus({ variant: "success", message: "Password reset successful." });
        setTimeout(() => navigate("/login"), 700);
        return;
      }

      setStatus({
        variant: "error",
        message: res?.data?.detail || "Reset failed.",
      });
    } catch (err) {
      const maybe = err?.response?.data;
      setStatus({
        variant: "error",
        message: maybe?.detail || "Reset failed.",
      });
    }
  }

  return (
    <div className="authPage" role="main" aria-label="Reset Password">
      <div className="authShell">
        <div className="authCard">
          <div className="authCardHeader">
            <div className="authCardIcon" aria-hidden="true">🩺</div>
            <div>
              <h2 className="authTitle">Reset Password</h2>
              <p className="authSubtitle">Enter the token and set your new password.</p>
            </div>
          </div>

          <form className="authForm" onSubmit={onSubmit}>
            <label className="authField">
              <span className="authLabel">Reset token</span>
              <input
                type="text"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="Paste token"
                className="authInput"
              />
            </label>

            <label className="authField">
              <span className="authLabel">New password</span>
              <input
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                placeholder="At least 6 characters"
                className="authInput"
              />
            </label>

            {status.message ? (
              <div
                role="status"
                aria-live="polite"
                className={
                  status.variant === "success" ? "authNotice authNoticeSuccess" : "authNotice authNoticeError"
                }
              >
                {status.message}
              </div>
            ) : null}

            <button type="submit" disabled={!isValid} className="authBtn authBtnPrimary">
              Reset Password
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

