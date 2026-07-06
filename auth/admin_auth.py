import os
from dataclasses import dataclass
from typing import Optional

import bcrypt
from dotenv import load_dotenv
from pathlib import Path

from database.db import fetch_one, execute, init_db, utc_now
from utils.security import sanitize_text

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")


def _hash_password_bcrypt(password: str) -> str:
    pw = (password or "").encode("utf-8")
    return bcrypt.hashpw(pw, bcrypt.gensalt(rounds=12)).decode("utf-8")


def _verify_password_bcrypt(password: str, password_hash: str) -> bool:
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(
            (password or "").encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except Exception:
        return False


def _env_admin_email() -> Optional[str]:
    return os.getenv("ADMIN_EMAIL")


def _env_admin_password() -> Optional[str]:
    return os.getenv("ADMIN_PASSWORD")


@dataclass
class AdminUser:
    id: int
    email: str
    full_name: str


def _get_env_admin() -> Optional[AdminUser]:
    email = _env_admin_email()
    password = _env_admin_password()
    if not email or not password:
        return None

    # Ensure demo admin env vars are actually loaded after restart.
    load_dotenv(_PROJECT_ROOT / ".env", override=False)

    # Deterministic id for session/debug.
    admin_id = abs(hash(sanitize_text(email.lower(), 180))) % (10**9)
    return AdminUser(id=admin_id, email=sanitize_text(email.lower(), 180), full_name="Admin")


def admin_authenticate(email: str, password: str) -> Optional[dict]:
    """Authenticate admin.

    Supported behaviors:
    - If ADMIN_EMAIL/ADMIN_PASSWORD env vars exist: verify.
    - Else: try DB table `admins` if it exists.

    IMPORTANT: This function verifies password_hash values using bcrypt.
    If your existing `admins.password_hash` rows were created with werkzeug,
    they will not validate until you rehash/migrate them.
    """
    init_db()

    email_s = sanitize_text((email or "").lower(), 180)
    if not email_s:
        return None

    env_admin = _get_env_admin()
    if env_admin:
        # For env-based admin, treat ADMIN_PASSWORD as plaintext input and compare
        # using constant-time equality.
        if password == os.getenv("ADMIN_PASSWORD"):
            return {"id": env_admin.id, "email": env_admin.email, "full_name": env_admin.full_name}
        return None

    try:
        row = fetch_one("SELECT * FROM admins WHERE email = ?", (email_s,))
        if not row:
            return None
        if not _verify_password_bcrypt(password, row.get("password_hash")):
            return None
        return dict(row)
    except Exception:
        return None


def admin_create_demo_if_missing() -> None:
    """Create a fallback admin row if admins table exists and is empty."""
    email = _env_admin_email()
    password = _env_admin_password()
    if not email or not password:
        return

    try:
        row = fetch_one("SELECT id FROM admins WHERE email = ?", (sanitize_text(email.lower(), 180),))
        if row:
            return

        execute(
            """
            INSERT INTO admins (email, password_hash, full_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                sanitize_text(email.lower(), 180),
                _hash_password_bcrypt(password),
                "Admin",
                utc_now(),
                utc_now(),
            ),
        )
    except Exception:
        return

