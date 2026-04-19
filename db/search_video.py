"""
Search video summaries in PostgreSQL with pgvector (cosine similarity).
Caller is responsible for computing query embeddings (e.g. via embeddings.embedder).
"""
# query Supabase database for video_summaries

from __future__ import annotations

from typing import Any

from db.connection import get_connection
from db.video_store import _table_columns


def _ensure_vector_registered(conn: Any) -> None:
    """Register pgvector type on the connection so vector columns work."""
    try:
        from pgvector.psycopg2 import register_vector

        register_vector(conn)
    except ImportError:
        pass  # pgvector not installed; raw SQL may still work with cast


def search_similar_by_text(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Embed a natural-language query with the same model as stored rows, then
    return the closest summaries by cosine distance (see search_similar).
    """
    from embeddings.embedder import embed_text

    q = (query or "").strip()
    if not q:
        return []
    return search_similar(embed_text(q), limit=limit)


def search_similar(
    query_embedding: list[float],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Return video summaries most similar to the query embedding (cosine distance).

    Args:
        query_embedding: List of 384 floats from the same model as stored embeddings.
        limit: Max number of results (default 10).

    Returns:
        List of dicts with keys: id, created_at, filename, duration_sec,
        summary_style, summary_text, distance (cosine distance; lower = more similar),
        and storage_object_path when that column exists on the table.
    """
    conn = None
    try:
        conn = get_connection()
        if conn is None:
            return []
        _ensure_vector_registered(conn)

        from pgvector import Vector

        supported = _table_columns(conn, "video_summaries")
        select_cols = [
            "id",
            "created_at",
            "filename",
            "duration_sec",
            "summary_style",
            "summary_text",
        ]
        if "storage_object_path" in supported:
            select_cols.append("storage_object_path")
        col_sql = ", ".join(select_cols)

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {col_sql}, embedding <=> %s AS distance
                FROM video_summaries
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (Vector(query_embedding), Vector(query_embedding), limit),
            )
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
    except Exception:
        raise
    finally:
        if conn:
            conn.close()
