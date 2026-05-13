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
