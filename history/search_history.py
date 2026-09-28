import logging
from pathlib import Path
from typing import Any

from database.db import execute, fetch_all, fetch_one, utc_now
from utils.config import UPLOAD_HISTORY_PATH
from utils.security import image_hash, safe_disease_slug, save_optimized_image
from utils.queries import (
    INSERT_SEARCH_HISTORY,
    GET_RECENT_SEARCHES,
    COUNT_DOCTOR_SEARCHES,
    GET_DOCTOR_DISEASE_STATS,
    GET_DOCTOR_SOURCE_STATS,
    GET_DOCTOR_TOP_IMAGES,
)

# The only disease classes supported by the ML system.
SUPPORTED_CLASSES = ["Acne", "Psoriasis", "Tinea", "Vitiligo"]

LOGGER = logging.getLogger(__name__)

# Ensure the uploads directory exists at import time so StaticFiles can serve it.
UPLOAD_HISTORY_PATH.mkdir(parents=True, exist_ok=True)
LOGGER.info("[Search History] Upload directory: %s", UPLOAD_HISTORY_PATH.resolve())


def record_search(
    doctor_pk: int | None,

    image,
    disease: str,
    confidence: float,
    prediction_source: str,
    image_name: str | None = None,
    ai_fallback_status: str = "not_used",
    retraining_status: str = "not_required",
    ensemble_metadata: dict[str, Any] | None = None,
    consent_for_training: bool = False,
) -> int:
    # Safety validation: ensure the stored disease is within supported scope.
    # This is a defense-in-depth check; validated predictions should already
    # be within scope by the time they reach this layer.
    if disease not in SUPPORTED_CLASSES:
        LOGGER.warning(
            "[Search History] Attempted to store unsupported disease '%s'. "
            "This should not happen — prediction should have been validated upstream. "
            "Falling back to 'Unknown' to avoid corrupting history.",
            disease,
        )
        disease = "Unknown"

    disease_name = safe_disease_slug(disease)
    img_hash = image_hash(image)

    # Always save the uploaded image so it can be displayed in Reports/History.
    # consent_for_training only controls whether the image is used for
    # self-learning/training — it must NOT prevent the image from being
    # persisted for display. Never delete this image after prediction.
    image_path = save_optimized_image(
        image,
        UPLOAD_HISTORY_PATH / disease_name,
        f"search_{disease_name}",
    )

    LOGGER.info(
        "[Search History] Saved image: original=%s saved=%s abs=%s exists=%s",
        image_name,
        image_path.name if image_path else None,
        str(image_path.resolve()) if image_path else None,
        image_path.exists() if image_path else False,
    )

    # Requirement: store image_name (uploaded filename).
    image_name = (image_name or "").strip() or None


    ensemble_metadata = ensemble_metadata or {}
    per_model = ensemble_metadata.get("per_model", {})
    return execute(
        INSERT_SEARCH_HISTORY,
        (
            doctor_pk,
            disease_name,
            str(image_path) if image_path else None,
            img_hash,
            image_name,
            float(confidence),
            prediction_source,
            ai_fallback_status,
            retraining_status,
            per_model.get("mobilenetv2", {}).get("prediction"),
            per_model.get("mobilenetv2", {}).get("confidence"),
            per_model.get("efficientnetb0", {}).get("prediction"),
            per_model.get("efficientnetb0", {}).get("confidence"),
            per_model.get("densenet121", {}).get("prediction"),
            per_model.get("densenet121", {}).get("confidence"),
            ensemble_metadata.get("ensemble_prediction"),
            ensemble_metadata.get("ensemble_confidence"),
            ensemble_metadata.get("ai_verification_summary"),
            ensemble_metadata.get("model_agreement"),
            ensemble_metadata.get("model_predictions_json"),
            1 if consent_for_training else 0,
            utc_now(),
        ),
    )



def recent_searches(doctor_pk: int, limit: int = 10, offset: int = 0, disease: str = ""):
    params: list = [doctor_pk]
    where = "WHERE doctor_id = ?"
    if disease:
        where += " AND disease LIKE ?"
        params.append(f"%{disease}%")
    params.extend([limit, offset])
    return fetch_all(
    GET_RECENT_SEARCHES.format(
        where_clause=where
    ),
    params,
)


def doctor_search_stats(doctor_pk: int) -> dict:
    total = fetch_one(COUNT_DOCTOR_SEARCHES, (doctor_pk,))["c"]
    diseases = fetch_all(
        GET_DOCTOR_DISEASE_STATS,
        (doctor_pk,),
    )
    sources = fetch_all(
        GET_DOCTOR_SOURCE_STATS,
        (doctor_pk,),
    )
    images = fetch_all(
        GET_DOCTOR_TOP_IMAGES,
        (doctor_pk,),
    )
    return {"total": total, "diseases": diseases, "sources": sources, "images": images}
