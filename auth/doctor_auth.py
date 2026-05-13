import secrets
from datetime import datetime, timedelta
from typing import Optional

from werkzeug.security import check_password_hash, generate_password_hash

from database.db import ensure_usage_row, execute, fetch_one, utc_now
from utils.config import FREE_SEARCH_LIMIT
from utils.security import sanitize_text


def create_doctor(profile: dict, password: str) -> int:
    if len(password or "") < 8:
        raise ValueError("Password must be at least 8 characters.")

    required = ["full_name", "doctor_id", "specialization", "email"]
    for field in required:
        if not sanitize_text(profile.get(field, "")):
            raise ValueError(f"{field.replace('_', ' ').title()} is required.")

    now = utc_now()
    doctor_pk = execute(
        """
        INSERT INTO doctors (
            full_name, doctor_id, specialization, clinic_name, email, phone,
            profile_photo, experience, location, password_hash, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sanitize_text(profile["full_name"]),
            sanitize_text(profile["doctor_id"], 80),
            sanitize_text(profile["specialization"]),
            sanitize_text(profile.get("clinic_name", "")),
            sanitize_text(profile["email"].lower(), 180),
            sanitize_text(profile.get("phone", ""), 40),
            sanitize_text(profile.get("profile_photo", ""), 500),
            int(profile.get("experience") or 0),
            sanitize_text(profile.get("location", "")),
            generate_password_hash(password),
            now,
            now,
        ),
    )
    ensure_usage_row(doctor_pk)
    return doctor_pk


def authenticate_doctor(email: str, password: str) -> Optional[dict]:
    row = fetch_one("SELECT * FROM doctors WHERE email = ?", (sanitize_text(email.lower(), 180),))
    if not row or not check_password_hash(row["password_hash"], password or ""):
        return None
    ensure_usage_row(int(row["id"]))
    return dict(row)


def get_doctor(doctor_pk: int) -> Optional[dict]:
    row = fetch_one("SELECT * FROM doctors WHERE id = ?", (doctor_pk,))
    return dict(row) if row else None


def update_doctor_profile(doctor_pk: int, profile: dict) -> None:
    execute(
        """
        UPDATE doctors
        SET full_name = ?, specialization = ?, clinic_name = ?, phone = ?,
            profile_photo = COALESCE(NULLIF(?, ''), profile_photo),
            experience = ?, location = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            sanitize_text(profile.get("full_name", "")),
            sanitize_text(profile.get("specialization", "")),
            sanitize_text(profile.get("clinic_name", "")),
            sanitize_text(profile.get("phone", ""), 40),
            sanitize_text(profile.get("profile_photo", ""), 500),
            int(profile.get("experience") or 0),
            sanitize_text(profile.get("location", "")),
            utc_now(),
            doctor_pk,
        ),
    )


def create_reset_token(email: str) -> Optional[str]:
    row = fetch_one("SELECT id FROM doctors WHERE email = ?", (sanitize_text(email.lower(), 180),))
    if not row:
        return None
    token = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(hours=1)).isoformat(timespec="seconds")
    execute(
        "UPDATE doctors SET reset_token = ?, reset_expires_at = ?, updated_at = ? WHERE id = ?",
        (token, expires, utc_now(), row["id"]),
    )
    return token


def reset_password(token: str, new_password: str) -> bool:
    if len(new_password or "") < 8:
        raise ValueError("Password must be at least 8 characters.")

    row = fetch_one("SELECT * FROM doctors WHERE reset_token = ?", (sanitize_text(token, 255),))
    if not row or not row["reset_expires_at"]:
        return False
    if datetime.fromisoformat(row["reset_expires_at"]) < datetime.utcnow():
        return False

    execute(
        """
        UPDATE doctors
        SET password_hash = ?, reset_token = NULL, reset_expires_at = NULL, updated_at = ?
        WHERE id = ?
        """,
        (generate_password_hash(new_password), utc_now(), row["id"]),
    )
    return True


def usage_for_doctor(doctor_pk: int) -> dict:
    ensure_usage_row(doctor_pk)
    row = fetch_one("SELECT * FROM free_search_usage WHERE doctor_id = ?", (doctor_pk,))
    used = int(row["used_count"])
    limit = int(row["free_limit"] or FREE_SEARCH_LIMIT)
    return {"used": used, "limit": limit, "remaining": max(limit - used, 0)}


def assert_can_search(doctor_pk: int) -> None:
    usage = usage_for_doctor(doctor_pk)
    if usage["remaining"] <= 0:
        raise PermissionError("Free search limit reached. Please upgrade to continue.")


def consume_search(doctor_pk: int) -> None:
    assert_can_search(doctor_pk)
    execute(
        """
        UPDATE free_search_usage
        SET used_count = used_count + 1, updated_at = ?
        WHERE doctor_id = ?
        """,
        (utc_now(), doctor_pk),
    )
