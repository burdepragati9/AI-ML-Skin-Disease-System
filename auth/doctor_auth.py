import os
import secrets
import re
from datetime import datetime, timedelta
from typing import Optional

import bcrypt

from database.db import ensure_usage_row, execute, fetch_one, utc_now
from utils.config import FREE_SEARCH_LIMIT
from utils.security import sanitize_text
from utils.queries import (
    GET_LAST_DOCTOR_ID,
    INSERT_DOCTOR,
    GET_DOCTOR_BY_EMAIL,
    UPDATE_DOCTOR_PROFILE,
    GET_DOCTOR_ID_BY_EMAIL,
    UPDATE_DOCTOR_RESET_TOKEN,
    GET_DOCTOR_BY_RESET_TOKEN,
    RESET_DOCTOR_PASSWORD,
    GET_FREE_SEARCH_USAGE,
    CONSUME_FREE_SEARCH,
)

def _generate_next_doctor_id() -> str:
    """Create the next doctor_id from the most recently stored doctor_id."""
    row = fetch_one(GET_LAST_DOCTOR_ID)
    if not row or not sanitize_text(row["doctor_id"]):
        return "DOC001"

    current_id = sanitize_text(row["doctor_id"], 80)
    match = re.match(r"^(.*?)(\d+)$", current_id)
    if not match:
        return "DOC001"

    prefix, number = match.groups()
    next_number = str(int(number) + 1).zfill(len(number))
    return f"{prefix}{next_number}"


def _hash_password_bcrypt(password: str) -> str:
    pw = (password or "").encode("utf-8")
    hashed = bcrypt.hashpw(pw, bcrypt.gensalt(rounds=12))
    return hashed.decode("utf-8")


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


def create_doctor(profile: dict, password: str) -> int:
    if len(password or "") < 8:
        raise ValueError("Password must be at least 8 characters.")

    required = ["full_name", "specialization", "email"]
    for field in required:
        if not sanitize_text(profile.get(field, "")):
            raise ValueError(f"{field.replace('_', ' ').title()} is required.")

    now = utc_now()
    doctor_id = _generate_next_doctor_id()

    doctor_pk = execute(
        INSERT_DOCTOR,
        (
            sanitize_text(profile["full_name"]),
            doctor_id,
            sanitize_text(profile["specialization"]),
            sanitize_text(profile.get("clinic_name", "")),
            sanitize_text(profile["email"].lower(), 180),
            sanitize_text(profile.get("phone", ""), 40),
            sanitize_text(profile.get("profile_photo", ""), 500),
            int(profile.get("experience") or 0),
            sanitize_text(profile.get("location", "")),
            _hash_password_bcrypt(password),
            now,
            now,
        ),
    )

    ensure_usage_row(doctor_pk)
    return doctor_pk


def authenticate_doctor(email: str, password: str) -> Optional[dict]:
    row = fetch_one(
        GET_DOCTOR_BY_EMAIL,
        (sanitize_text(email.lower(), 180),),
    )
    if not row:
        return None

    if not _verify_password_bcrypt(password, row["password_hash"]):
        return None

    ensure_usage_row(int(row["id"]))
    return dict(row)


def update_doctor_profile(doctor_pk: int, profile: dict) -> None:
    execute(
        UPDATE_DOCTOR_PROFILE,
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
    row = fetch_one(
        GET_DOCTOR_ID_BY_EMAIL,
        (sanitize_text(email.lower(), 180),),
    )
    if not row:
        return None

    token = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(hours=1)).isoformat(timespec="seconds")

    execute(
        UPDATE_DOCTOR_RESET_TOKEN,
        (token, expires, utc_now(), row["id"]),
    )
    return token


def reset_password(token: str, new_password: str) -> bool:
    if len(new_password or "") < 8:
        raise ValueError("Password must be at least 8 characters.")

    row = fetch_one(
        GET_DOCTOR_BY_RESET_TOKEN,
        (sanitize_text(token, 255),),
    )
    if not row or not row["reset_expires_at"]:
        return False

    if datetime.fromisoformat(row["reset_expires_at"]) < datetime.utcnow():
        return False

    execute(
        RESET_DOCTOR_PASSWORD,
        (
            _hash_password_bcrypt(new_password),
            utc_now(),
            row["id"],
        ),
    )
    return True


def usage_for_doctor(doctor_pk: int) -> dict:
    ensure_usage_row(doctor_pk)
    row = fetch_one(
        GET_FREE_SEARCH_USAGE,
        (doctor_pk,),
    )
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
        CONSUME_FREE_SEARCH,
        (utc_now(), doctor_pk),
    )

