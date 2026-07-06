import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import api from "../services/api";

import "./Signup.css";

export default function Signup() {
  const navigate = useNavigate();

  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const [status, setStatus] = useState({ variant: "idle", message: "" });

  const isValid = useMemo(() => {
    const nameOk = (fullName || "").trim().length > 1;
    const emailOk = (email || "").trim().length > 0;
    const passOk = (password || "").length >= 6;
    const confirmOk = (confirmPassword || "").length >= 6;
    const matchOk = password === confirmPassword;
    return nameOk && emailOk && passOk && confirmOk && matchOk;
  }, [fullName, email, password, confirmPassword]);

  async function onSignup(e) {
    e.preventDefault();

    const trimmedName = (fullName || "").trim();
    const trimmedEmail = (email || "").trim();

    if (!trimmedName) {
      setStatus({ variant: "error", message: "Please enter your full name." });
      return;
    }
    if (!trimmedEmail) {
      setStatus({ variant: "error", message: "Please enter your email." });
      return;
    }
    if (!password) {
      setStatus({ variant: "error", message: "Please enter your password." });
      return;
    }
    if (password !== confirmPassword) {
      setStatus({ variant: "error", message: "Passwords do not match." });
      return;
    }

    setStatus({ variant: "idle", message: "" });

    try {
      // Standard signup endpoint (per requirement). If backend differs, show server detail.
      const res = await api.post("/auth/signup", {
        full_name: trimmedName,
        specialization: "General Physician",
        clinic_name: "",
        email: trimmedEmail,
        phone: "",
        experience: 0,
        location: "",
        password,
      });

      if (res?.data?.success) {
        setStatus({ variant: "success", message: res?.data?.detail || "Signup successful." });
        setTimeout(() => navigate("/login"), 900);
        return;
      }

      setStatus({
        variant: "error",
        message: res?.data?.detail || "Signup failed.",
      });
    } catch (err) {
      const maybe = err?.response?.data;
      setStatus({ variant: "error", message: maybe?.detail || "Signup failed." });
    }
  }

  function onLogin() {
    navigate("/login");
  }

  return (
    <div className="signupPage" role="main" aria-label="Signup">
      <div className="signupShell">
        <div className="signupBanner" aria-hidden="true">
          <div className="signupBannerInner">
            <div className="signupBannerTop">
              <div className="signupBrandMarkLarge">🩺</div>
              <div>
                <div className="signupBannerTitle">Skin Disease Detection System</div>
                <div className="signupBannerSubtitle">
                  Secure access for doctors and administrators.
                </div>
              </div>
            </div>

            <div className="signupIllustration">
              <div className="signupIllustrationInner">
                Medical illustration placeholder
              </div>
            </div>

            <div className="signupPills">
              <span className="pill">AI-assisted</span>
              <span className="pill">Clinical workflow</span>
              <span className="pill">Private data</span>
            </div>
          </div>
        </div>

        <div className="signupPanel">
          <div className="signupCard">
            <div className="signupHeader">
              <div className="signupBrandMark" aria-hidden="true">🩺</div>
              <div>
                <h1 className="signupTitle">Create your account</h1>
                <p className="signupSubtitle">Join to access secure disease detection tools.</p>
              </div>
            </div>

            <form className="signupForm" onSubmit={onSignup}>
              <div className="signupFields">
                <div className="signupField">
                  <label className="signupLabel" htmlFor="fullName">Full Name</label>
                  <input
                    id="fullName"
                    type="text"
                    className="signupInput"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    placeholder="Dr. Jane Doe"
                    autoComplete="name"
                  />
                </div>

                <div className="signupField">
                  <label className="signupLabel" htmlFor="signupEmail">Email</label>
                  <div className="fieldWithIcon">
                    <span className="icon" aria-hidden="true">✉️</span>
                    <input
                      id="signupEmail"
                      type="email"
                      className="signupInput"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="you@example.com"
                      autoComplete="email"
                    />
                  </div>
                </div>

                <div className="signupField">
                  <label className="signupLabel" htmlFor="signupPassword">Password</label>
                  <div className="fieldWithIcon">
                    <span className="icon" aria-hidden="true">🔒</span>
                    <input
                      id="signupPassword"
                      type={showPassword ? "text" : "password"}
                      className="signupInput"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="At least 6 characters"
                      autoComplete="new-password"
                    />
                  </div>
                  <div className="passwordToggleRow">
                    <button
                      type="button"
                      className="toggleBtn"
                      onClick={() => setShowPassword((v) => !v)}
                    >
                      {showPassword ? "Hide" : "Show"}
                    </button>
                  </div>
                </div>

                <div className="signupField">
                  <label className="signupLabel" htmlFor="confirmPassword">Confirm Password</label>
                  <div className="fieldWithIcon">
                    <span className="icon" aria-hidden="true">🔒</span>
                    <input
                      id="confirmPassword"
                      type={showConfirmPassword ? "text" : "password"}
                      className="signupInput"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      placeholder="Repeat your password"
                      autoComplete="new-password"
                    />
                  </div>
                  <div className="passwordToggleRow">
                    <button
                      type="button"
                      className="toggleBtn"
                      onClick={() => setShowConfirmPassword((v) => !v)}
                    >
                      {showConfirmPassword ? "Hide" : "Show"}
                    </button>
                  </div>
                </div>
              </div>

              {status.message ? (
                <div
                  className={
                    status.variant === "success"
                      ? "signupNotice signupNoticeSuccess"
                      : status.variant === "error"
                      ? "signupNotice signupNoticeError"
                      : "signupNotice"
                  }
                  role="status"
                  aria-live="polite"
                  style={{ marginTop: 12 }}
                >
                  {status.message}
                </div>
              ) : null}

              <div className="signupActions">
                <button type="submit" className="signupBtn signupBtnPrimary" disabled={!isValid}>
                  Create Account
                </button>

                <div className="signupSecondaryRow">
                  <button type="button" className="signupBtnLink" onClick={onLogin}>
                    Back to Login
                  </button>
                </div>
              </div>

              <p className="signupFinePrint">This is a secure authentication setup.</p>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

