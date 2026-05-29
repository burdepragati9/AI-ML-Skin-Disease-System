import os
import sqlite3
from dataclasses import dataclass
from typing import Optional

# Load environment variables for Streamlit/runtime.
# This uses python-dotenv without changing existing auth flows.
from dotenv import load_dotenv
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

from werkzeug.security import check_password_hash, generate_password_hash


from database.db import fetch_one, execute, init_db, utc_now

from utils.security import sanitize_text


# Admin credentials: separated from doctor credentials.
# This project currently uses SQLite directly in `database/db.py` migrations.
# To avoid breaking DB schema, we support two modes:
#   1) Single admin account configured via environment variables.
#   2) Admins table (optional) if already created.
#
# Mode 1 is controlled by ADMIN_EMAIL and ADMIN_PASSWORD in environment.
# This keeps changes limited and avoids requiring DB migration right now.


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
        return

    # Ensure demo admin env vars are actually loaded after restart.
    # (Useful if the process environment was updated without a full restart.)
    # No side effects if values are already present.
    load_dotenv(_PROJECT_ROOT / ".env", override=False)


    # Deterministic id for session/debug.
    # Not stored in DB; just a stable marker.
    admin_id = abs(hash(sanitize_text(email.lower(), 180))) % (10**9)
    return AdminUser(id=admin_id, email=sanitize_text(email.lower(), 180), full_name="Admin")


def admin_authenticate(email: str, password: str) -> Optional[dict]:

    """Authenticate admin.

    Supported behaviors:
    - If ADMIN_EMAIL/ADMIN_PASSWORD env vars exist: verify against them.
    - Else: try DB table `admins` if it exists.

    Returns a dict user object if authenticated, else None.
    """

    init_db()
    email_s = sanitize_text((email or "").lower(), 180)
    if not email_s:
        return None

    env_admin = _get_env_admin()
    if env_admin:
        if email_s == env_admin.email and (password or "") == os.getenv("ADMIN_PASSWORD"):
            return {"id": env_admin.id, "email": env_admin.email, "full_name": env_admin.full_name}
        return None

    # Fallback: optional admins table.
    try:
        row = fetch_one("SELECT * FROM admins WHERE email = ?", (email_s,))
        if not row:
            return None
        if not check_password_hash(row["password_hash"], password or ""):
            return None
        return dict(row)
    except Exception:
        return None


def admin_create_demo_if_missing() -> None:
    """Create a fallback admin row if admins table exists and is empty.

    Not creating schema; only helps when schema is already present.
    """
    email = _env_admin_email()
    password = _env_admin_password()
    if not email or not password:
        return

    try:
        row = fetch_one("SELECT id FROM admins WHERE email = ?", (sanitize_text(email.lower(), 180),))
        if row:
            return
        # If schema doesn't exist, this will fail harmlessly.
        execute(
            """
            INSERT INTO admins (email, password_hash, full_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                sanitize_text(email.lower(), 180),
                generate_password_hash(password),
                "Admin",
                utc_now(),
                utc_now(),
            ),
        )
    except Exception:
        return

