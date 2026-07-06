import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import api from "../services/api";
import { saveRoleAndUser, saveToken } from "../services/auth";
import "./Login.css";

export default function Login() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [status, setStatus] = useState({ variant: "idle", message: "" });

  const isValid = useMemo(() => {
    const userEmail = (email || "").trim();
    return userEmail.length > 0 && (password || "").length > 0;
  }, [email, password]);

  async function onLogin(event) {
    event.preventDefault();

    const userEmail = (email || "").trim();

    if (!userEmail) {
      setStatus({ variant: "error", message: "Please enter your email." });
      return;
    }

    if (!password) {
      setStatus({ variant: "error", message: "Please enter your password." });
      return;
    }

    setStatus({ variant: "idle", message: "" });

    try {
      const res = await api.post("/auth/login", {
        email: userEmail,
        password,
      });

      const data = res?.data;

      // expected from backend: returns JWT access_token
      const token = data?.access_token || data?.accessToken;
      const role = data?.role;
      const user = data?.user;

      if (!token) {
        setStatus({ variant: "error", message: data?.detail || "Login failed." });
        return;
      }

      saveToken(token);
      saveRoleAndUser({ role, user });

      if (role === "doctor") {
        navigate("/doctor-dashboard");
        return;
      }

      if (role === "admin") {
        navigate("/admin-dashboard");
        return;
      }

      // fallback
      navigate("/dashboard");
    } catch (e) {
      const maybe = e?.response?.data;
      if (e?.response?.status === 401) {
        setStatus({ variant: "error", message: "Invalid email or password." });
        return;
      }
      setStatus({ variant: "error", message: maybe?.detail || "Unable to connect to server." });
    }
  }

  function onForgotPassword() {
    navigate("/forgot-password");
  }

  function onSignup() {
    navigate("/signup");
  }

  return (
    <div className="loginPage" role="main" aria-label="Skin Disease Detection Login">
      <div className="loginShell">
        <div className="loginBanner" aria-hidden="true">
          <div className="loginBannerInner">
            <div className="loginBannerTop">
              <div className="loginBrandMarkLarge">🩺</div>
              <div>
                <div className="loginBannerTitle">Skin Disease Detection System</div>
                <div className="loginBannerSubtitle">
                  Secure access for doctors and administrators.
                </div>
              </div>
            </div>

            <div className="loginIllustration">
              <div className="loginIllustrationInner">
                Medical illustration placeholder
              </div>
            </div>

            <div className="loginPills">
              <span className="pill">AI-assisted</span>
              <span className="pill">Clinical workflow</span>
              <span className="pill">Private data</span>
            </div>
          </div>
        </div>

        <div className="loginPanel">
          <div className="loginCard">
            <div className="loginHeader">
              <div className="loginBrandMark" aria-hidden="true">🩺</div>
              <div>
                <h1 className="loginTitle">Skin Disease Detection System</h1>
                <p className="loginSubtitle">Secure access for doctors and administrators.</p>
              </div>
            </div>

            <form className="loginForm" onSubmit={onLogin}>
              <div className="loginFields">
                <div className="loginField">
                  <label className="loginLabel" htmlFor="email">Email</label>
                  <div className="fieldWithIcon">
                    <span className="icon" aria-hidden="true">✉️</span>
                    <input
                      id="email"
                      name="email"
                      type="email"
                      className="loginInput"
                      placeholder="you@example.com"
                      value={email}
                      onChange={(event) => setEmail(event.target.value)}
                      autoComplete="email"
                    />
                  </div>
                </div>

                <div className="loginField">
                  <label className="loginLabel" htmlFor="password">Password</label>
                  <div className="fieldWithIcon">
                    <span className="icon" aria-hidden="true">🔒</span>
                    <input
                      id="password"
                      name="password"
                      type="password"
                      className="loginInput"
                      placeholder="••••••••"
                      value={password}
                      onChange={(event) => setPassword(event.target.value)}
                      autoComplete="current-password"
                    />
                  </div>
                </div>
              </div>

              {status.message ? (
                <div
                  className={
                    status.variant === "success"
                      ? "loginNotice loginNoticeSuccess"
                      : status.variant === "error"
                      ? "loginNotice loginNoticeError"
                      : "loginNotice"
                  }
                  role="status"
                  aria-live="polite"
                  style={{ marginTop: 12 }}
                >
                  {status.message}
                </div>
              ) : null}

              <div className="loginActions">
                <button className="loginBtn loginBtnPrimary" type="submit" disabled={!isValid}>Login</button>

                <div className="loginSecondaryRow">
                  <button type="button" className="loginBtnLink" onClick={onForgotPassword}>
                    Forgot Password
                  </button>
                  <button type="button" className="loginBtnLink" onClick={onSignup}>
                    Signup
                  </button>
                </div>
              </div>

              <p className="loginFinePrint">This is a secure authentication setup.</p>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}


