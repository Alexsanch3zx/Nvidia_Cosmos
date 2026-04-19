"""
Upload video files to Supabase Storage (REST) and build public object URLs.

Requires:
  SUPABASE_URL           e.g. https://YOUR_PROJECT_REF.supabase.co
  SUPABASE_SERVICE_ROLE_KEY   (server-side only; never expose in the browser)

Optional:
  SUPABASE_VIDEO_BUCKET  defaults to "video"

Add a column to Postgres (run in Supabase SQL editor):

  ALTER TABLE video_summaries
    ADD COLUMN IF NOT EXISTS storage_object_path TEXT;
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import requests


def _config() -> tuple[str, str, str]:
    base = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    bucket = (os.getenv("SUPABASE_VIDEO_BUCKET") or "video").strip()
    return base, key, bucket


def is_storage_configured() -> bool:
    base, key, _ = _config()
    return bool(base and key)


def safe_video_filename(original: str | None) -> str:
    name = (original or "clip").split("/")[-1].split("\\")[-1]
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name).strip("._") or "clip"
    if not Path(name).suffix:
        name = f"{name}.mp4"
    return name[:180]


def content_type_for_filename(filename: str | None) -> str:
    ext = Path(filename or "").suffix.lower()
    return {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }.get(ext, "application/octet-stream")


def build_object_key(username: str, original_filename: str | None) -> str:
    """Path inside the bucket: <username>/<uuid>_<safe_name>."""
    user = re.sub(r"[^a-zA-Z0-9._-]", "_", (username or "user").strip())[:64] or "user"
    safe = safe_video_filename(original_filename)
    return f"{user}/{uuid.uuid4().hex}_{safe}"


def upload_local_file_to_video_bucket(
    local_path: str,
    object_key: str,
    content_type: str = "video/mp4",
) -> None:
    """Upload a file from disk into the configured bucket at object_key. Raises on HTTP error."""
    base, service_key, bucket = _config()
    if not base or not service_key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to upload videos to Storage."
        )

    from urllib.parse import quote

    encoded = quote(object_key, safe="/")
    endpoint = f"{base}/storage/v1/object/{bucket}/{encoded}"
    headers = {
        "Authorization": f"Bearer {service_key}",
        "apikey": service_key,
        "Content-Type": content_type,
    }
    with open(local_path, "rb") as f:
        data = f.read()
    r = requests.post(endpoint, headers=headers, data=data, params={"upsert": "true"}, timeout=600)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Storage upload failed ({r.status_code}): {r.text[:800]}")


def public_video_url(object_key: str) -> str:
    """Public URL for a path inside the public bucket (read)."""
    base, _, bucket = _config()
    if not base:
        raise RuntimeError("SUPABASE_URL is not set")
    from urllib.parse import quote

    encoded = quote(object_key, safe="/")
    return f"{base}/storage/v1/object/public/{bucket}/{encoded}"


def try_public_video_url(object_key: str | None) -> str | None:
    """Same as public_video_url but returns None if SUPABASE_URL is missing or key is empty."""
    if not object_key:
        return None
    base = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    if not base:
        return None
    bucket = (os.getenv("SUPABASE_VIDEO_BUCKET") or "video").strip()
    from urllib.parse import quote

    encoded = quote(object_key, safe="/")
    return f"{base}/storage/v1/object/public/{bucket}/{encoded}"
