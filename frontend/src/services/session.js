import api from "./api";
import { logout } from "./auth";

let sessionValidationPromise = null;

export function validateSessionOnce() {
  if (!sessionValidationPromise) {
    sessionValidationPromise = api.get("/auth/me").catch((error) => {
      logout();
      throw error;
    }).finally(() => {
      sessionValidationPromise = null;
    });
  }

  return sessionValidationPromise;
}
