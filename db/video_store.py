"""
Store video summaries in PostgreSQL with pgvector.
Search is in db.search_video. Caller is responsible for computing embeddings
(e.g. via embeddings.embedder).
"""
# write
from __future__ import annotations

from typing import Any

from db.connection import get_connection


def _ensure_vector_registered(conn: Any) -> None:
    """Register pgvector type on the connection so vector columns work."""
    try:
        from pgvector.psycopg2 import register_vector

        register_vector(conn)
    except ImportError:
        pass  # pgvector not installed; raw SQL may still work with cast


def insert_summary(
    filename: str | None,
    duration_sec: float | None,
    summary_style: str,
    summary_text: str,
    embedding: list[float],
    storage_bucket: str | None = None,
    storage_path: str | None = None,
) -> int | None:
    """
    Insert a video summary row. Returns the new row id, or None on failure.

    Args:
        filename: Original video filename (optional).
        duration_sec: Video duration in seconds (optional).
        summary_style: e.g. "detailed", "concise", "bullet points".
        summary_text: Full summary text (required).
        embedding: List of 384 floats (must match table vector(384)).
        storage_bucket: Supabase Storage bucket name (optional; DB may default).
        storage_path: Object key inside the bucket after upload (optional).
    """
    if not summary_text.strip():
        return None

    bucket = storage_bucket or "video"

    conn = None
    try:
        conn = get_connection()
        if conn is None:
            return None
        _ensure_vector_registered(conn)

        from pgvector import Vector

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO video_summaries
                  (filename, duration_sec, summary_style, summary_text, embedding,
                   storage_bucket, storage_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    filename,
                    duration_sec,
                    summary_style,
                    summary_text,
                    Vector(embedding),
                    bucket,
                    storage_path,
                ),
            )
            row = cur.fetchone()
            conn.commit()
            return row[0] if row else None
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
