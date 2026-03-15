"""
Initialize the database: enable pgvector and create video_summaries if missing.
Requires DATABASE_URL (e.g. postgresql://user:pass@localhost:5432/cosmos_videos).
The database must already exist; this does not create it.
Run once: python -m db.init  or  from db.init import run_init; run_init()
"""

from __future__ import annotations

from db.connection import get_connection


def run_init() -> None:
    conn = None
    try:
        conn = get_connection()
        if conn is None:
            raise RuntimeError("DATABASE_URL is not set")
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS video_summaries (
                    id            BIGSERIAL PRIMARY KEY,
                    created_at    TIMESTAMPTZ DEFAULT NOW(),
                    filename      TEXT,
                    duration_sec  NUMERIC(10,2),
                    summary_style TEXT,
                    summary_text  TEXT NOT NULL,
                    embedding     vector(384)
                );
            """)
        conn.commit()
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    run_init()