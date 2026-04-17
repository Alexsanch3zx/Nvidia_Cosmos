"""
Canonical summarization layout for video summaries (deterministic / heuristic).

Every stored summary uses the same markdown skeleton so DB rows stay consistent.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

# Bump when you change section names or semantics
TEMPLATE_ID = "cosmos_summary_v1"
DEFAULT_VISION_MODEL_LABEL = "Cosmos-Reason2-8B"

# Sidebar labels -> internal keys (stored in DB / metadata)
ANALYSIS_STYLES: list[tuple[str, str]] = [
    ("Bullet points", "bullet_points"),
    ("Concise", "concise"),
    ("Formal", "formal"),
    ("Municipal report (detailed)", "municipal_report"),
]


def style_key_from_label(label: str) -> str:
    for display, key in ANALYSIS_STYLES:
        if display == label:
            return key
    return "formal"


def metadata_line(
    style: str,
    engine: str,
    vision_model: str = DEFAULT_VISION_MODEL_LABEL,
) -> str:
    """Machine-readable first line (safe for embeddings; strip if you need plain prose only)."""
    return (
        f"<!-- summary_template:{TEMPLATE_ID} | style:{style} | engine:{engine} | "
        f"vision:{vision_model} -->"
    )


def _ollama_municipal_report_prompt(transcript: str, vision_model: str) -> str:
    return f"""You are drafting an official field observation record from timestamped visual descriptions produced by {vision_model}.

Use only facts supported by the notes below.

Frame notes:
{transcript}

Write markdown with these sections:
## Record identification
## Executive summary
## Detailed chronological account
## Persons, vehicles, objects, and environment
## Actions and sequence of events
## Uncertainties, occlusions, and limitations
## Administrative closing
"""


def ollama_user_prompt(
    transcript: str,
    style: str,
    vision_model: str = DEFAULT_VISION_MODEL_LABEL,
) -> str:
    style_key = (style or "formal").lower().strip().replace(" ", "_")
    if style_key == "municipal_report":
        return _ollama_municipal_report_prompt(transcript, vision_model)

    return f"""You turn timestamped frame descriptions into a structured video summary.

Vision captions came from {vision_model}. Use ONLY information supported by the frame notes below.

Frame notes:
{transcript}

Output MUST use this markdown shape:
## Overview
## Chronological highlights
## Takeaways

Keep wording concise and factual. No speculation.
"""


def _format_heuristic_municipal_report(
    frame_descriptions: List[Dict[str, str]],
    timestamps: List[float],
    format_timestamp,
    vision_model: str,
    style_key: str,
) -> str:
    """English municipal-style layout without an external LLM."""
    lines: List[str] = [metadata_line(style_key, "heuristic", vision_model), ""]
    first_ts = format_timestamp(timestamps[0]) if timestamps else "unknown"
    last_ts = format_timestamp(timestamps[-1]) if timestamps else "unknown"
    first = (frame_descriptions[0].get("description") or "").strip()
    last = (frame_descriptions[-1].get("description") or "").strip()

    lines.append("## Record identification")
    lines.append(
        f"This record was produced from sampled video frames analyzed by **{vision_model}**. "
        f"Sampled interval on the timeline spans approximately **{first_ts}** through **{last_ts}**."
    )
    lines.append("")
    lines.append("## Executive summary")
    lines.append(
        f"The footage documents the following initial observation: {first} "
        f"The sequence concludes with: {last} "
        "This is a condensed municipal-style layout from frame captions only."
    )
    lines.append("")
    lines.append("## Detailed chronological account")
    for i, fd in enumerate(frame_descriptions):
        ts = format_timestamp(timestamps[i]) if i < len(timestamps) else "?"
        desc = (fd.get("description") or "").strip()
        lines.append(f"1. **{ts}** - {desc}")
    lines.append("")
    lines.append("## Persons, vehicles, objects, and environment")
    topics = extract_keywords_from_frames(frame_descriptions)
    if topics:
        for t in topics:
            lines.append(f"- Term recurring in captions: **{t}**")
    else:
        lines.append("- No additional keyword clusters extracted from captions.")
    lines.append("")
    lines.append("## Uncertainties, occlusions, and limitations")
    lines.append(
        "- This document was generated without a large language model; phrasing follows raw frame text only. "
        "- Identity, intent, and off-camera events are not established."
    )
    lines.append("")
    lines.append(f"_Template `{TEMPLATE_ID}` - heuristic municipal layout._")
    return "\n".join(lines)


def format_heuristic_summary(
    frame_descriptions: List[Dict[str, str]],
    timestamps: List[float],
    style: str,
    format_timestamp,
    vision_model: str = DEFAULT_VISION_MODEL_LABEL,
) -> str:
    """Deterministic summary; municipal report uses an English incident-record style."""
    if not frame_descriptions:
        return metadata_line(style, "heuristic", vision_model) + "\n\n_No content to summarize._"

    style_key = (style or "formal").lower().strip().replace(" ", "_")
    if style_key in ("detailed", "deep"):
        style_key = "formal"
    if style_key == "bulletpoints":
        style_key = "bullet_points"

    if style_key == "municipal_report":
        return _format_heuristic_municipal_report(
            frame_descriptions, timestamps, format_timestamp, vision_model, style_key
        )

    lines: List[str] = [metadata_line(style_key, "heuristic", vision_model), ""]

    # Overview: first + last + count
    first = (frame_descriptions[0].get("description") or "").strip()
    last = (frame_descriptions[-1].get("description") or "").strip()
    dur = format_timestamp(timestamps[-1]) if timestamps else "unknown"
    lines.append("## Overview")
    if style_key == "concise":
        lines.append(
            f"The clip runs through **{dur}** (sampled frames). It begins with: {first[:280]}{'...' if len(first) > 280 else ''} "
            f"It ends with: {last[:280]}{'...' if len(last) > 280 else ''}"
        )
    elif style_key == "formal":
        lines.append(
            f"The audiovisual material (approximately **{dur}** of sampled content) initially presents the following. "
            f"{first} "
            f"The sequence concludes as follows. {last}"
        )
    else:
        # bullet_points: short context paragraph
        lines.append(
            f"**Duration (sampled):** {dur}. **Start:** {first[:350]}{'...' if len(first) > 350 else ''} "
            f"**End:** {last[:350]}{'...' if len(last) > 350 else ''}"
        )
    lines.append("")

    lines.append("## Chronological highlights")
    if style_key == "bullet_points":
        max_bullets = min(len(frame_descriptions), 24)
    elif style_key == "concise":
        max_bullets = 6
    else:
        max_bullets = 10
    step = max(1, len(frame_descriptions) // max_bullets) if max_bullets < len(frame_descriptions) else 1
    for i in range(0, len(frame_descriptions), step):
        ts = format_timestamp(timestamps[i]) if i < len(timestamps) else "?"
        desc = (frame_descriptions[i].get("description") or "").strip()
        short = desc if len(desc) <= 400 else desc[:397] + "..."
        lines.append(f"- **[{ts}]** {short}")
    lines.append("")

    lines.append("## Takeaways")
    topics = extract_keywords_from_frames(frame_descriptions)
    if topics:
        for t in topics[:5]:
            lines.append(f"- Recurring theme: **{t}**")
    else:
        lines.append("- No strong recurring keywords detected from sampled frames.")
    lines.append("")
    lines.append(f"_Generated with template `{TEMPLATE_ID}` (heuristic)._")

    return "\n".join(lines)


def extract_keywords_from_frames(frame_descriptions: List[Dict[str, str]]) -> List[str]:
    """Public alias for topic-like tokens from frame captions (search hints)."""
    return _rough_topics(frame_descriptions)


def _rough_topics(frame_descriptions: List[Dict[str, str]]) -> List[str]:
    common = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "is", "are", "was", "were", "this", "that", "these", "those", "there", "here",
    }
    freq: Dict[str, int] = {}
    for fd in frame_descriptions:
        for w in (fd.get("description") or "").lower().split():
            w = re.sub(r"[^a-z0-9]", "", w)
            if len(w) > 5 and w not in common:
                freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: -x[1])
    return [w for w, _ in ranked[:8]]


def parse_template_id_from_summary(text: str) -> str | None:
    m = re.search(r"summary_template:([a-zA-Z0-9_\-]+)", text[:500])
    return m.group(1) if m else None


def record_for_storage(
    *,
    summary_text: str,
    filename: str | None,
    duration_sec: float | None,
    style: str,
    engine: str,
    vision_model: str = DEFAULT_VISION_MODEL_LABEL,
    search_text: str | None = None,
) -> dict[str, Any]:
    """Row shape for JSONL local store (and optional future API)."""
    row: dict[str, Any] = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "template_id": TEMPLATE_ID,
        "vision_model": vision_model,
        "summary_style": style,
        "summary_engine": engine,
        "filename": filename,
        "duration_sec": duration_sec,
        "summary_text": summary_text,
    }
    if search_text:
        row["search_text"] = search_text
    return row


def jsonl_dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False) + "\n"

