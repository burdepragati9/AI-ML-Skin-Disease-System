"""utils.privacy

Healthcare privacy protection helpers.

Goal:
- Prevent any personally identifiable or sensitive non-medical fields from being
  sent to external AI services (e.g., Gemini/OpenAI).
- Only allow explicitly whitelisted payload parts/fields.

This is a defense-in-depth layer on top of prompt design and AI workflow.
"""

from __future__ import annotations

from typing import Any, Iterable


# Whitelist of top-level keys allowed to be sent to external AI.
# Keep this strict: only include the fixed medical prompt and image bytes.
ALLOWED_TOP_LEVEL_KEYS: set[str] = {
    "prompt",
    "parts",
}

# Allowed keys inside each "part".
ALLOWED_PART_KEYS: set[str] = {
    "mime_type",
    "data",
    "text",
}


# Explicitly block keys that commonly leak PII/PHI metadata.
# (This list is intentionally conservative.)
BLOCKED_KEYS: set[str] = {
    "patient_name",
    "doctor_name",
    "email",
    "phone",
    "address",
    "username",
    "user_id",
    "doctor_id",
    "patient_id",
    "session_id",
    "hospital_name",
    "clinic_name",
    "profile_photo",
    "created_at",
    "uploaded_by",
    "metadata",
    "db_row",
    "raw_db",
    "request",
}


def _sanitize_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Return a sanitized dict containing only whitelisted keys."""
    out: dict[str, Any] = {}
    for k, v in data.items():
        if k in BLOCKED_KEYS:
            continue
        if k in ALLOWED_TOP_LEVEL_KEYS:
            out[k] = v
    return out


def sanitize_payload(payload: Any) -> Any:
    """Sanitize an outbound AI payload.

    This function is intentionally strict:
    - If payload is a dict: keep only allowed top-level keys.
    - If payload is list/tuple: recursively sanitize dict elements.
    - If payload is a primitive/bytes: return as-is.

    For Gemini, we mostly sanitize "parts" elements to ensure they contain only
    image bytes (data) and the required mime_type.
    """

    # Dict path (most common)
    if isinstance(payload, dict):
        # First, remove blocked/non-whitelisted top-level keys.
        cleaned = _sanitize_dict(payload)

        # Special handling for Gemini "parts": ensure part objects contain only
        # allowed keys.
        parts = cleaned.get("parts")
        if isinstance(parts, list):
            safe_parts: list[Any] = []
            for part in parts:
                if not isinstance(part, dict):
                    # If something unexpected exists, drop it.
                    continue
                safe_part = {k: v for k, v in part.items() if k in ALLOWED_PART_KEYS and k not in BLOCKED_KEYS}
                safe_parts.append(safe_part)
            cleaned["parts"] = safe_parts

        return cleaned

    # List path
    if isinstance(payload, (list, tuple)):
        return [sanitize_payload(x) for x in payload]

    # Primitive/bytes path
    return payload


def ensure_gemini_safe_parts(parts: Iterable[Any]) -> list[dict[str, Any]]:
    """Ensure Gemini request parts contain ONLY allowed mime/data/text.

    Gemini expects the "parts" list items to be well-formed part objects.
    Defense-in-depth note:
    - We sanitize keys, but we intentionally KEEP the canonical keys required
      by Gemini (mime_type + data for images).
    - We DO NOT modify or transform the binary image bytes.

    This function should never change the request structure in a way that
    would make Gemini reject the payload.
    """
    safe_parts: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue

        # Keep only allowed keys.
        safe_part = {
            k: v
            for k, v in part.items()
            if k in ALLOWED_PART_KEYS and k not in BLOCKED_KEYS
        }

        # Do not drop the part if it contains the required image keys.
        has_required_image_keys = (
            safe_part.get("mime_type") == "image/jpeg" and "data" in safe_part
        )
        if not safe_part:
            continue

        if has_required_image_keys:
            safe_parts.append(safe_part)
        else:
            # For non-image parts, only append if it still contains at least one
            # allowed key (Gemini will determine validity).
            allowed_any = any(k in safe_part for k in ("mime_type", "data", "text"))
            if allowed_any:
                safe_parts.append(safe_part)

    return safe_parts


