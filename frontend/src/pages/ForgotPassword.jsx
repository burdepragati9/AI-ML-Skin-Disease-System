import { useMemo, useState } from "react";

import "./ForgotPassword.css";

import api from "../services/api";

export default function ForgotPassword() {
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState({ variant: "idle", message: "" });
  const [resetToken, setResetToken] = useState("");

  const isValid = useMemo(() => {
    return (email || "").trim().length > 0;
  }, [email]);

  async function onSubmit(e) {
    e.preventDefault();
    setStatus({ variant: "idle", message: "" });
    setResetToken("");

    try {
      const res = await api.post("/auth/forgot-password", {
        email: (email || "").trim(),
      });

      if (res?.data?.success) {
        // required: show token in UI for now
        const token = res?.data?.reset_token || res?.data?.token || "";
        setResetToken(token);
        setStatus({ variant: "success", message: "Reset token generated." });
        return;
      }

      setStatus({ variant: "error", message: res?.data?.detail || "Request failed." });
    } catch (err) {
      const maybe = err?.response?.data;
      setStatus({ variant: "error", message: maybe?.detail || "Request failed." });
    }
  }

  return (
    <div className="authPage" role="main" aria-label="Forgot Password">
      <div className="authShell">
        <div className="authCard">
          <div className="authCardHeader">
            <div className="authCardIcon" aria-hidden="true">🔐</div>
            <div>
              <h2 className="authTitle">Forgot Password</h2>
              <p className="authSubtitle">Request a reset token to regain access.</p>
            </div>
          </div>

          <form className="authForm" onSubmit={onSubmit}>
            <label className="authField">
              <span className="authLabel">Email</span>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
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
              Generate Reset Token
            </button>
          </form>

          {resetToken ? (
            <div className="authToken">
              <h3 className="authTokenTitle">Reset token (demo)</h3>
              <pre className="authTokenPre">{resetToken}</pre>
              <p className="authTokenHint">
                Paste it into Reset Password form:{" "}
                <span className="authTokenLink">/reset-password</span>
              </p>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

