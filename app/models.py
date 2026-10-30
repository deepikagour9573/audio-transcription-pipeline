"""SQLite persistence and data access for transcription jobs."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.settings import DB_PATH

# Store timestamps in one consistent UTC format, even if the app runs in different time zones.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# Each operation gets its own SQLite connection; rows come back with column names for readability.
def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Create the table on first run and mark unfinished jobs after a restart so they are not left stuck.
def init_db() -> None:
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                audio_path TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                transcript TEXT,
                segments_json TEXT,
                language TEXT,
                duration_seconds REAL,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            UPDATE jobs SET status='failed',
                error='Service restarted before this job completed',
                updated_at=?
            WHERE status IN ('processing', 'queued')
        """, (now_iso(),))

# A new upload starts as queued because the transcription worker has not picked it up yet.
def create_job(job_id: str, filename: str, audio_path: str) -> None:
    created = now_iso()
    with connect_db() as conn:
        conn.execute("""
            INSERT INTO jobs (job_id, original_filename, audio_path, status,
                              attempts, created_at, updated_at)
            VALUES (?, ?, ?, 'queued', 0, ?, ?)
        """, (job_id, filename, audio_path, created, created))

# Return a plain dictionary so the API and service layer can work with the job easily.
def get_job(job_id: str) -> dict[str, Any] | None:
    with connect_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    return dict(row) if row else None

# Only allow known job fields to be updated; this helps catch accidental typos early.
def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    allowed = {
        "status", "attempts", "transcript", "segments_json", "language",
        "duration_seconds", "error",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Unsupported job fields: {', '.join(sorted(unknown))}")
    fields["updated_at"] = now_iso()
    assignments = ", ".join(f"{key}=?" for key in fields)
    values = list(fields.values()) + [job_id]
    with connect_db() as conn:
        conn.execute(f"UPDATE jobs SET {assignments} WHERE job_id=?", values)
