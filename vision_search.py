"""
Build searchable text from frame-level captions + summary so users can find videos by
objects / scenes (e.g. "red car", "cocina") via the same embedding model as summaries.

True pixel-level detection would need a separate detector; here we search what Cosmos wrote.
"""
from __future__ import annotations

import re
from typing import Dict, List

# Sentence-transformers context limit - stay safely under token cap
_MAX_EMBED_CHARS = 10_000


def build_search_text(
    summary: str,
    frame_descriptions: List[Dict[str, str]],
) -> str:
    """Concatenate summary + every frame caption so embeddings catch object/scene queries."""
    parts: List[str] = [summary.strip(), "", "=== Per-frame captions ===", ""]
    for fd in frame_descriptions:
        desc = (fd.get("description") or "").strip()
        if desc:
            parts.append(desc)
    return "\n".join(parts).strip()[:_MAX_EMBED_CHARS]


def suggest_search_terms(
    frame_descriptions: List[Dict[str, str]],
    max_terms: int = 15,
) -> List[str]:
    """Lightweight keyword hints from captions (nouns-ish tokens + 2-word phrases)."""
    from summarys.summary_templates import extract_keywords_from_frames

    topics = extract_keywords_from_frames(frame_descriptions)
    text = " ".join((fd.get("description") or "") for fd in frame_descriptions).lower()
    # simple bigrams on words length > 3
    words = re.findall(r"[a-záéíóúñ]{4,}", text)
    bigrams: Dict[str, int] = {}
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i + 1]}"
        bigrams[bg] = bigrams.get(bg, 0) + 1
    top_bigrams = sorted(bigrams.items(), key=lambda x: -x[1])[:8]
    out = list(topics[: max_terms // 2])
    for bg, _ in top_bigrams:
        if bg not in out and len(out) < max_terms:
            out.append(bg)
    return out[:max_terms]
