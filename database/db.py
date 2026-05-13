import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable

from utils.config import DB_PATH, FREE_SEARCH_LIMIT, PROJECT_ROOT


MIGRATIONS_DIR = PROJECT_ROOT / "database" / "migrations"


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(column[1] == column_name for column in columns)


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_sql: str,
) -> None:
    if not _column_exists(conn, table_name, column_name):
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")


def _upgrade_existing_schema(conn: sqlite3.Connection) -> None:
    """Apply additive schema upgrades for databases created before migrations grew."""
    _add_column_if_missing(
        conn,
        "searches",
        "ai_fallback_status",
        "ai_fallback_status TEXT NOT NULL DEFAULT 'not_used'",
    )
    _add_column_if_missing(
        conn,
        "searches",
        "retraining_status",
        "retraining_status TEXT NOT NULL DEFAULT 'not_required'",
    )
    _add_column_if_missing(conn, "ai_predictions", "doctor_id", "doctor_id INTEGER")
    _add_column_if_missing(
        conn,
        "ai_predictions",
        "fallback_status",
        "fallback_status TEXT NOT NULL DEFAULT 'AI_FALLBACK'",
    )
    _add_column_if_missing(
        conn,
        "ai_predictions",
        "retraining_status",
        "retraining_status TEXT NOT NULL DEFAULT 'queued'",
    )
    _add_column_if_missing(conn, "training_queue", "doctor_id", "doctor_id INTEGER")
    _add_column_if_missing(
        conn,
        "training_queue",
        "source",
        "source TEXT NOT NULL DEFAULT 'AI_FALLBACK'",
    )
    _add_column_if_missing(conn, "training_queue", "trained_at", "trained_at TEXT")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        for migration_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            conn.executescript(migration_path.read_text(encoding="utf-8"))
        _upgrade_existing_schema(conn)
        conn.commit()


@contextmanager
def get_conn():
    init_db()
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def fetch_one(query: str, params: Iterable = ()):
    with get_conn() as conn:
        return conn.execute(query, tuple(params)).fetchone()


def fetch_all(query: str, params: Iterable = ()):
    with get_conn() as conn:
        return conn.execute(query, tuple(params)).fetchall()


def execute(query: str, params: Iterable = ()) -> int:
    with get_conn() as conn:
        cursor = conn.execute(query, tuple(params))
        return int(cursor.lastrowid or 0)


def ensure_usage_row(doctor_pk: int) -> None:
    execute(
        """
        INSERT OR IGNORE INTO free_search_usage (doctor_id, free_limit, used_count, updated_at)
        VALUES (?, ?, 0, ?)
        """,
        (doctor_pk, FREE_SEARCH_LIMIT, utc_now()),
    )
