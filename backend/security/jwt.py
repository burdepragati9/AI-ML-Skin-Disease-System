from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt as pyjwt


def _get_secret() -> str:
    secret = os.getenv("JWT_SECRET")
    if not secret:
        # Development fallback. Replace with a real secret in production.
        secret = "dev-only-change-me"
    return secret


def _get_algorithm() -> str:
    return os.getenv("JWT_ALGORITHM", "HS256")


def create_access_token(*, user_id: int, email: str, role: str, expires_minutes: int = 60 * 24) -> str:
    now = datetime.now(tz=timezone.utc)
    exp = now + timedelta(minutes=expires_minutes)

    payload: Dict[str, Any] = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }

    return pyjwt.encode(payload, _get_secret(), algorithm=_get_algorithm())


def decode_token(token: str) -> Dict[str, Any]:
    return pyjwt.decode(
        token,
        _get_secret(),
        algorithms=[_get_algorithm()],
        options={"require": ["sub", "exp"]},
    )

