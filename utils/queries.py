# utils/queries.py


# =========================
# Admin Analytics Queries
# =========================

GET_LATEST_TRAINING_LOG = """
SELECT *
FROM training_logs
ORDER BY created_at DESC
LIMIT 1
"""

COUNT_QUEUED_TRAINING = """
SELECT COUNT(*) AS c
FROM training_queue
WHERE status = 'queued'
"""

COUNT_PROCESSING_TRAINING = """
SELECT COUNT(*) AS c
FROM training_queue
WHERE status = 'processing'
"""

COUNT_COMPLETED_TRAINING = """
SELECT COUNT(*) AS c
FROM training_queue
WHERE status = 'completed'
"""

COUNT_FAILED_TRAINING = """
SELECT COUNT(*) AS c
FROM training_queue
WHERE status = 'failed'
"""

COUNT_NEWLY_LEARNED_IMAGES = """
SELECT COUNT(*) AS c
FROM ai_predictions
WHERE duplicate_of IS NULL
"""

GET_AI_RECOGNIZED_IMAGES = """
SELECT image_name, image_path, predicted_disease, confidence, source, created_at
FROM ai_predictions
ORDER BY created_at DESC
LIMIT ?
"""

COUNT_TOTAL_AI_PREDICTIONS = """
SELECT COUNT(*) AS c
FROM ai_predictions
"""

COUNT_RETRAINED_IMAGES = """
SELECT COUNT(*) AS c
FROM training_queue
WHERE status = 'completed'
"""

GET_MOST_COMMON_AI_DISEASE = """
SELECT predicted_disease, COUNT(*) AS count
FROM ai_predictions
WHERE duplicate_of IS NULL
GROUP BY predicted_disease
ORDER BY count DESC
LIMIT 1
"""

GET_LATEST_ACCURACY = """
SELECT accuracy_before, accuracy_after
FROM training_logs
WHERE accuracy_after IS NOT NULL
ORDER BY created_at DESC
LIMIT 1
"""

GET_PREDICTION_SOURCE_COUNTS = """
SELECT prediction_source, COUNT(*) AS count
FROM searches
GROUP BY prediction_source
"""

GET_DISEASE_COUNTS = """
SELECT disease, COUNT(*) AS count
FROM searches
GROUP BY disease
ORDER BY count DESC
LIMIT 10
"""

GET_PREDICTION_SOURCE_COUNTS_BY_TIME = """
SELECT prediction_source, COUNT(*) AS count
FROM searches
{where_clause}
GROUP BY prediction_source
ORDER BY count DESC
"""

GET_DISEASE_FREQUENCY_BY_TIME = """
SELECT disease, COUNT(*) AS count
FROM searches
{where_clause}
GROUP BY disease
ORDER BY count DESC
LIMIT 10
"""

# =========================
# Admin Route Queries
# =========================

GET_ALL_DOCTORS = """
SELECT 
    id,
    full_name AS doctor_name,
    email,
    specialization,
    created_at
FROM doctors
ORDER BY created_at DESC
"""

COUNT_DOCTOR_SEARCHES = """
SELECT COUNT(*) AS count
FROM searches
WHERE doctor_id = ?
"""

GET_RECENT_TRAINING_LOGS = """
SELECT 
    id,
    event_type,
    status,
    disease,
    accuracy_before,
    accuracy_after,
    message,
    image_path,
    created_at
FROM training_logs
ORDER BY created_at DESC
LIMIT 20
"""

# =========================
# Admin Authentication Queries
# =========================

GET_ADMIN_BY_EMAIL = """
SELECT *
FROM admins
WHERE email = ?
"""

GET_ADMIN_ID_BY_EMAIL = """
SELECT id
FROM admins
WHERE email = ?
"""

INSERT_ADMIN = """
INSERT INTO admins (
    email,
    password_hash,
    full_name,
    created_at,
    updated_at
)
VALUES (?, ?, ?, ?, ?)
"""

# =========================
# Doctor Authentication Queries
# =========================

GET_LAST_DOCTOR_ID = """
SELECT doctor_id
FROM doctors
ORDER BY id DESC
LIMIT 1
"""

INSERT_DOCTOR = """
INSERT INTO doctors (
    full_name, doctor_id, specialization, clinic_name, email, phone,
    profile_photo, experience, location, password_hash, created_at, updated_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

GET_DOCTOR_BY_EMAIL = """
SELECT *
FROM doctors
WHERE email = ?
"""

UPDATE_DOCTOR_PROFILE = """
UPDATE doctors
SET full_name = ?, specialization = ?, clinic_name = ?, phone = ?,
    profile_photo = COALESCE(NULLIF(?, ''), profile_photo),
    experience = ?, location = ?, updated_at = ?
WHERE id = ?
"""

GET_DOCTOR_ID_BY_EMAIL = """
SELECT id
FROM doctors
WHERE email = ?
"""

UPDATE_DOCTOR_RESET_TOKEN = """
UPDATE doctors
SET reset_token = ?, reset_expires_at = ?, updated_at = ?
WHERE id = ?
"""

GET_DOCTOR_BY_RESET_TOKEN = """
SELECT *
FROM doctors
WHERE reset_token = ?
"""

RESET_DOCTOR_PASSWORD = """
UPDATE doctors
SET password_hash = ?, reset_token = NULL, reset_expires_at = NULL, updated_at = ?
WHERE id = ?
"""

GET_FREE_SEARCH_USAGE = """
SELECT *
FROM free_search_usage
WHERE doctor_id = ?
"""

CONSUME_FREE_SEARCH = """
UPDATE free_search_usage
SET used_count = used_count + 1, updated_at = ?
WHERE doctor_id = ?
"""

# =========================
# Auth Route Queries
# =========================

GET_DOCTOR_PROFILE_BY_ID = """
SELECT
    full_name,
    specialization,
    clinic_name,
    email,
    phone,
    experience,
    location
FROM doctors
WHERE id = ?
"""

# =========================
# Prediction Route Queries
# =========================

COUNT_DOCTOR_SEARCHES_FOR_LIMIT = """
SELECT COUNT(*) AS c
FROM searches
WHERE doctor_id = ?
"""

# =========================
# Profile Route Queries
# =========================

GET_DOCTOR_PROFILE = """
SELECT
    id,
    full_name,
    doctor_id,
    specialization,
    clinic_name,
    email,
    phone,
    profile_photo,
    experience,
    location,
    created_at,
    updated_at
FROM doctors
WHERE id = ?
"""

CHECK_DOCTOR_EXISTS = """
SELECT id
FROM doctors
WHERE id = ?
"""

# =========================
# Search History Queries
# =========================

INSERT_SEARCH_HISTORY = """
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
"""

GET_RECENT_SEARCHES = """
SELECT *
FROM searches
{where_clause}
ORDER BY created_at DESC
LIMIT ? OFFSET ?
"""

COUNT_DOCTOR_SEARCHES = """
SELECT COUNT(*) AS c
FROM searches
WHERE doctor_id = ?
"""

GET_DOCTOR_DISEASE_STATS = """
SELECT disease, COUNT(*) AS count
FROM searches
WHERE doctor_id = ?
GROUP BY disease
ORDER BY count DESC
LIMIT 10
"""

GET_DOCTOR_SOURCE_STATS = """
SELECT prediction_source, COUNT(*) AS count
FROM searches
WHERE doctor_id = ?
GROUP BY prediction_source
"""

GET_DOCTOR_TOP_IMAGES = """
SELECT disease, image_path, COUNT(*) AS count
FROM searches
WHERE doctor_id = ?
GROUP BY image_hash
ORDER BY count DESC
LIMIT 6
"""