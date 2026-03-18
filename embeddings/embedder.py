"""
Text embedding for video summaries using sentence-transformers.
Output dimension is 384 (all-MiniLM-L6-v2), matching db schema vector(384).
"""

from __future__ import annotations

from functools import lru_cache

#_get_model() will only run once
@lru_cache(maxsize=1)
def _get_model():
    """Lazy-load the embedding model so import is fast."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str) -> list[float]:
    """
    Embed a single text string into a 384-dimensional vector.

    Args:
        text: Input text (e.g. summary or search query).

    Returns:
        List of 384 floats. Empty or whitespace-only text returns the
        embedding of a single space to avoid errors.
    """
    if not text or not text.strip():
        text = " "
    model = _get_model()
    vec = model.encode(text, convert_to_numpy=True)
    return vec.tolist()
