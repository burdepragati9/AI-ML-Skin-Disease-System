from datetime import datetime, timedelta

from database.db import fetch_all, fetch_one
from utils.queries import (
    GET_LATEST_TRAINING_LOG,
    COUNT_QUEUED_TRAINING,
    COUNT_PROCESSING_TRAINING,
    COUNT_COMPLETED_TRAINING,
    COUNT_FAILED_TRAINING,
    COUNT_NEWLY_LEARNED_IMAGES,
    GET_AI_RECOGNIZED_IMAGES,
    COUNT_TOTAL_AI_PREDICTIONS,
    COUNT_RETRAINED_IMAGES,
    GET_MOST_COMMON_AI_DISEASE,
    GET_LATEST_ACCURACY,
    GET_PREDICTION_SOURCE_COUNTS,
    GET_DISEASE_COUNTS,
    GET_PREDICTION_SOURCE_COUNTS_BY_TIME,
    GET_DISEASE_FREQUENCY_BY_TIME,
)


def training_status() -> dict:
    last_log = fetch_one(GET_LATEST_TRAINING_LOG)
    queued = fetch_one(COUNT_QUEUED_TRAINING)["c"]
    processing = fetch_one(COUNT_PROCESSING_TRAINING)["c"]
    completed = fetch_one(COUNT_COMPLETED_TRAINING)["c"]
    failed = fetch_one(COUNT_FAILED_TRAINING)["c"]

    return {
        "last_log": dict(last_log) if last_log else None,
        "queued": queued,
        "processing": processing,
        "completed": completed,
        "failed": failed,
    }


def newly_learned_images_count() -> int:
    return int(fetch_one(COUNT_NEWLY_LEARNED_IMAGES)["c"])

def ai_recognized_images(limit: int = 100):
    return fetch_all(
        GET_AI_RECOGNIZED_IMAGES,
        (limit,),
    )


def admin_summary() -> dict:
    """Backwards-compatible admin summary (all-time)."""
    total_ai = fetch_one(COUNT_TOTAL_AI_PREDICTIONS)["c"]

    retrained = fetch_one(COUNT_RETRAINED_IMAGES)["c"]

    common = fetch_one(GET_MOST_COMMON_AI_DISEASE)

    latest_accuracy = fetch_one(GET_LATEST_ACCURACY)

    source_counts = fetch_all(GET_PREDICTION_SOURCE_COUNTS)

    disease_counts = fetch_all(GET_DISEASE_COUNTS)

    improvement = 0.0
    if latest_accuracy:
        improvement = (
            float(latest_accuracy["accuracy_after"] or 0)
            - float(latest_accuracy["accuracy_before"] or 0)
        )

    return {
        "total_ai": total_ai,
        "retrained": retrained,
        "most_common": dict(common) if common else None,
        "accuracy_improvement": improvement,
        "source_counts": source_counts,
        "disease_counts": disease_counts,
    }
    

def _time_window_start(period: str) -> str | None:
    """Return ISO timestamp lower bound for SQLite comparisons."""
    now = datetime.utcnow()
    period_norm = (period or "").strip().lower()

    if period_norm in {"weekly", "week"}:
        start = now - timedelta(days=7)
    elif period_norm in {"monthly", "month"}:
        start = now - timedelta(days=30)
    elif period_norm in {"yearly", "year"}:
        start = now - timedelta(days=365)
    else:
        return None

    return start.isoformat(timespec="seconds")


def admin_prediction_source_counts_by_time(period: str):
    start_ts = _time_window_start(period)

    where = ""
    params: list = []

    if start_ts:
        where = "WHERE created_at >= ?"
        params.append(start_ts)

    return fetch_all(
        GET_PREDICTION_SOURCE_COUNTS_BY_TIME.format(
            where_clause=where
        ),
        tuple(params),
    )
    
def admin_disease_frequency_by_time(period: str):
    start_ts = _time_window_start(period)

    where = ""
    params: list = []

    if start_ts:
        where = "WHERE created_at >= ?"
        params.append(start_ts)

    return fetch_all(
        GET_DISEASE_FREQUENCY_BY_TIME.format(
            where_clause=where
        ),
        tuple(params),
    )


def admin_summary_by_time(period: str) -> dict:
    """Admin-only time-filtered analytics snapshot."""
    disease_counts = admin_disease_frequency_by_time(period)
    source_counts = admin_prediction_source_counts_by_time(period)

    total_ai = fetch_one(COUNT_TOTAL_AI_PREDICTIONS)["c"]

    return {
        "total_ai": total_ai,
        "disease_counts": disease_counts,
        "source_counts": source_counts,
    }
    