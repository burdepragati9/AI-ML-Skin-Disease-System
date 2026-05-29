from datetime import datetime, timedelta

from database.db import fetch_all, fetch_one


def training_status() -> dict:
    last_log = fetch_one("SELECT * FROM training_logs ORDER BY created_at DESC LIMIT 1")
    queued = fetch_one("SELECT COUNT(*) AS c FROM training_queue WHERE status = 'queued'")["c"]
    processing = fetch_one("SELECT COUNT(*) AS c FROM training_queue WHERE status = 'processing'")["c"]
    completed = fetch_one("SELECT COUNT(*) AS c FROM training_queue WHERE status = 'completed'")["c"]
    failed = fetch_one("SELECT COUNT(*) AS c FROM training_queue WHERE status = 'failed'")["c"]
    return {
        "last_log": dict(last_log) if last_log else None,
        "queued": queued,
        "processing": processing,
        "completed": completed,
        "failed": failed,
    }


def newly_learned_images_count() -> int:
    return int(fetch_one("SELECT COUNT(*) AS c FROM ai_predictions WHERE duplicate_of IS NULL")["c"])


def ai_recognized_images(limit: int = 100):
    return fetch_all(
        """
        SELECT image_name, image_path, predicted_disease, confidence, source, created_at
        FROM ai_predictions
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )


def admin_summary() -> dict:
    """Backwards-compatible admin summary (all-time)."""
    total_ai = fetch_one("SELECT COUNT(*) AS c FROM ai_predictions")["c"]

    retrained = fetch_one("SELECT COUNT(*) AS c FROM training_queue WHERE status = 'completed'")["c"]
    common = fetch_one(
        """
        SELECT predicted_disease, COUNT(*) AS count
        FROM ai_predictions
        WHERE duplicate_of IS NULL
        GROUP BY predicted_disease
        ORDER BY count DESC
        LIMIT 1
        """
    )
    latest_accuracy = fetch_one(
        """
        SELECT accuracy_before, accuracy_after
        FROM training_logs
        WHERE accuracy_after IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 1
        """
    )
    source_counts = fetch_all(
        """
        SELECT prediction_source, COUNT(*) AS count
        FROM searches
        GROUP BY prediction_source
        """
    )
    disease_counts = fetch_all(
        """
        SELECT disease, COUNT(*) AS count
        FROM searches
        GROUP BY disease
        ORDER BY count DESC
        LIMIT 10
        """
    )
    improvement = 0.0
    if latest_accuracy:
        improvement = float(latest_accuracy["accuracy_after"] or 0) - float(latest_accuracy["accuracy_before"] or 0)
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
        f"""
        SELECT prediction_source, COUNT(*) AS count
        FROM searches
        {where}
        GROUP BY prediction_source
        ORDER BY count DESC
        """,
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
        f"""
        SELECT disease, COUNT(*) AS count
        FROM searches
        {where}
        GROUP BY disease
        ORDER BY count DESC
        LIMIT 10
        """,
        tuple(params),
    )


def admin_summary_by_time(period: str) -> dict:
    """Admin-only time-filtered analytics snapshot."""
    disease_counts = admin_disease_frequency_by_time(period)
    source_counts = admin_prediction_source_counts_by_time(period)

    total_ai = fetch_one("SELECT COUNT(*) AS c FROM ai_predictions")["c"]
    return {
        "total_ai": total_ai,
        "disease_counts": disease_counts,
        "source_counts": source_counts,
    }

