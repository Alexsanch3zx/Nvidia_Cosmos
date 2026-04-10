"""
Upload videos to Supabase Storage from the Streamlit server.

Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the environment.
Optional: SUPABASE_VIDEO_BUCKET (default: video).
"""
from __future__ import annotations

import os
from urllib.parse import quote


def storage_configured() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


def video_bucket() -> str:
    return os.getenv("SUPABASE_VIDEO_BUCKET", "video").strip() or "video"


def upload_local_file(local_path: str, object_key: str, content_type: str = "video/mp4") -> None:
    """Upload a local file to the configured bucket at object_key. Raises on failure."""
    from supabase import create_client

    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY for Storage uploads.")

    with open(local_path, "rb") as f:
        data = f.read()

    client = create_client(url, key)
    client.storage.from_(video_bucket()).upload(
        object_key,
        data,
        file_options={"content-type": content_type},
    )


def public_object_url(object_key: str, bucket: str | None = None) -> str | None:
    """HTTPS URL for a public bucket object. Returns None if SUPABASE_URL is unset."""
    base = os.getenv("SUPABASE_URL", "").rstrip("/")
    if not base or not object_key:
        return None
    b = (bucket or video_bucket()).strip()
    encoded = "/".join(quote(part, safe="") for part in object_key.split("/"))
    return f"{base}/storage/v1/object/public/{b}/{encoded}"
