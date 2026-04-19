"""
Turn Cosmos per-frame captions into a structured summary via Ollama.
"""
from __future__ import annotations

import os
from typing import Dict, List

from summarys.summary_templates import (
    DEFAULT_VISION_MODEL_LABEL,
    metadata_line,
    ollama_user_prompt,
)


def summarize_frames_with_ollama(
    frame_descriptions: List[Dict[str, str]],
    timestamps: List[float],
    style: str = "formal",
    model: str | None = None,
    host: str | None = None,
    vision_model: str | None = None,
) -> str:
    import ollama

    resolved_model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
    resolved_host = host or os.getenv("OLLAMA_HOST")
    vision = vision_model or os.getenv("COSMOS_MODEL_LABEL", DEFAULT_VISION_MODEL_LABEL)

    client = ollama.Client(host=resolved_host) if resolved_host else ollama.Client()

    lines: List[str] = []
    for fd, ts in zip(frame_descriptions, timestamps):
        desc = (fd.get("description") or "").strip()
        lines.append(f"[{ts:.2f}s] {desc}")
    transcript = "\n".join(lines).strip()
    if not transcript:
        return metadata_line(style, "ollama", vision) + "\n\n_No content to summarize._"

    user_prompt = ollama_user_prompt(transcript, style, vision_model=vision)
    response = client.chat(
        model=resolved_model,
        messages=[{"role": "user", "content": user_prompt}],
        options={"num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "4096"))},
    )
    message = response.get("message") or {}
    body = (message.get("content") or "").strip()
    if not body:
        return metadata_line(style, "ollama", vision) + "\n\n_Ollama returned an empty response._"

    return f"{metadata_line(style, 'ollama', vision)}\n\n{body}"

