"""Audit log: one row per question, kept for a limited time.

Questions can reveal personal information, so the log stores only what is needed:
the time, the role (not the person), the question, the answer and the source ids.
Rows older than AUDIT_RETENTION_DAYS (default 90) are deleted on every write.
"""
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent / "data" / "audit.db"
RETENTION_DAYS = int(os.getenv("AUDIT_RETENTION_DAYS", "90"))


def _connect():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        role TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        source_ids TEXT NOT NULL,
        refused INTEGER NOT NULL)""")
    return conn


def record(role, question, answer, sources, refused):
    """Store one interaction and delete the rows past the retention limit."""
    now = datetime.now(timezone.utc)
    conn = _connect()
    try:
        with conn:  # commits on success
            conn.execute(
                "INSERT INTO interactions (created_at, role, question, answer, source_ids, refused) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (now.isoformat(timespec="seconds"), role, question, answer,
                 json.dumps([chunk["id"] for _, chunk in sources]), int(refused)))
            cutoff = (now - timedelta(days=RETENTION_DAYS)).isoformat(timespec="seconds")
            conn.execute("DELETE FROM interactions WHERE created_at < ?", (cutoff,))
    finally:
        conn.close()


def count():
    conn = _connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    finally:
        conn.close()
