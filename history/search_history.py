import logging
from pathlib import Path
from typing import Any

from database.db import execute, fetch_all, fetch_one, utc_now
from utils.config import UPLOAD_HISTORY_PATH
from utils.security import image_hash, safe_disease_slug, save_optimized_image

# The only disease classes supported by the ML system.
SUPPORTED_CLASSES = ["Acne", "Psoriasis", "Tinea", "Vitiligo"]

LOGGER = logging.getLogger(__name__)


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
    
    # Only save image if consent is given
    if consent_for_training:
        image_path = save_optimized_image(
            image,
            UPLOAD_HISTORY_PATH / disease_name,
            f"search_{disease_name}",
        )
    else:
        image_path = None

    # Requirement: store image_name (uploaded filename).
    image_name = (image_name or "").strip() or None


    ensemble_metadata = ensemble_metadata or {}
    per_model = ensemble_metadata.get("per_model", {})
    return execute(
        """
        INSERT INTO searches (
            doctor_id, disease, image_path, image_hash, image_name,
            confidence, prediction_source,
            ai_fallback_status, retraining_status,
            mobilenet_prediction, mobilenet_confidence,

            efficientnet_prediction, efficientnet_confidence,
            densenet_prediction, densenet_confidence,
            ensemble_prediction, ensemble_confidence,
            ai_verification_summary, model_agreement, model_predictions_json,
            consent_for_training,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
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
        f"""
        SELECT * FROM searches
        {where}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        params,
    )


def doctor_search_stats(doctor_pk: int) -> dict:
    total = fetch_one("SELECT COUNT(*) AS c FROM searches WHERE doctor_id = ?", (doctor_pk,))["c"]
    diseases = fetch_all(
        """
        SELECT disease, COUNT(*) AS count
        FROM searches
        WHERE doctor_id = ?
        GROUP BY disease
        ORDER BY count DESC
        LIMIT 10
        """,
        (doctor_pk,),
    )
    sources = fetch_all(
        """
        SELECT prediction_source, COUNT(*) AS count
        FROM searches
        WHERE doctor_id = ?
        GROUP BY prediction_source
        """,
        (doctor_pk,),
    )
    images = fetch_all(
        """
        SELECT disease, image_path, COUNT(*) AS count
        FROM searches
        WHERE doctor_id = ?
        GROUP BY image_hash
        ORDER BY count DESC
        LIMIT 6
        """,
        (doctor_pk,),
    )
    return {"total": total, "diseases": diseases, "sources": sources, "images": images}
