CREATE TABLE IF NOT EXISTS doctors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    doctor_id TEXT NOT NULL UNIQUE,
    specialization TEXT NOT NULL,
    clinic_name TEXT,
    email TEXT NOT NULL UNIQUE,
    phone TEXT,
    profile_photo TEXT,
    experience INTEGER DEFAULT 0,
    location TEXT,
    password_hash TEXT NOT NULL,
    reset_token TEXT,
    reset_expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS free_search_usage (
    doctor_id INTEGER PRIMARY KEY,
    free_limit INTEGER NOT NULL DEFAULT 3,
    used_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (doctor_id) REFERENCES doctors(id)
);

CREATE TABLE IF NOT EXISTS searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id INTEGER,
    disease TEXT NOT NULL,
    image_path TEXT,
    image_hash TEXT,
    confidence REAL NOT NULL,
    prediction_source TEXT NOT NULL,
    ai_fallback_status TEXT NOT NULL DEFAULT 'not_used',
    retraining_status TEXT NOT NULL DEFAULT 'not_required',
    created_at TEXT NOT NULL,
    FOREIGN KEY (doctor_id) REFERENCES doctors(id)
);

CREATE TABLE IF NOT EXISTS ai_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT NOT NULL,
    image_path TEXT NOT NULL,
    image_hash TEXT NOT NULL UNIQUE,
    predicted_disease TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'AI',
    doctor_id INTEGER,
    fallback_status TEXT NOT NULL DEFAULT 'AI_FALLBACK',
    retraining_status TEXT NOT NULL DEFAULT 'queued',
    duplicate_of INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (doctor_id) REFERENCES doctors(id)
);

CREATE TABLE IF NOT EXISTS image_hashes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_hash TEXT NOT NULL UNIQUE,
    image_path TEXT NOT NULL,
    disease TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    image_path TEXT,
    disease TEXT,
    accuracy_before REAL,
    accuracy_after REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    disease TEXT NOT NULL,
    doctor_id INTEGER,
    source TEXT NOT NULL DEFAULT 'AI_FALLBACK',
    status TEXT NOT NULL DEFAULT 'queued',
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    trained_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (doctor_id) REFERENCES doctors(id)
);

CREATE INDEX IF NOT EXISTS idx_searches_doctor_created ON searches(doctor_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_searches_disease ON searches(disease);
CREATE INDEX IF NOT EXISTS idx_ai_predictions_disease ON ai_predictions(predicted_disease);
CREATE INDEX IF NOT EXISTS idx_training_queue_status ON training_queue(status, created_at);
CREATE INDEX IF NOT EXISTS idx_training_logs_created ON training_logs(created_at DESC);
