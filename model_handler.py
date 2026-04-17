import base64
import io
import json
import os
import tempfile

import cv2
import numpy as np
import requests
from PIL import Image
from typing import List, Dict

VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://10.20.1.116:8000/v1")
MODEL_ID = os.getenv("COSMOS_MODEL_ID", "nvidia/Cosmos-Reason2-8B")


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _frames_to_video_data_url(frames: List[Image.Image], fps: int = 4) -> str:
    first = np.array(frames[0].convert("RGB"))
    h, w = first.shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    writer = cv2.VideoWriter(
        tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for frame in frames:
        writer.write(cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR))
    writer.release()
    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmp_path)
    return f"data:video/mp4;base64,{b64}"


def _call_api(messages: list, stream: bool = True) -> str:
    response = requests.post(
        f"{VLLM_API_BASE}/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7,
            "stream": stream,
            "media_io_kwargs": {"video": {"fps": 4}},
        },
        stream=stream,
        timeout=120,
    )
    response.raise_for_status()

    if not stream:
        return response.json()["choices"][0]["message"]["content"]

    parts = []
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8") if isinstance(line, bytes) else line
        if not decoded.startswith("data: "):
            continue
        payload = decoded[6:]
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                parts.append(delta)
        except (json.JSONDecodeError, KeyError):
            continue
    return "".join(parts)


class CosmosModelHandler:
    """Handles interaction with Nvidia's Cosmos-Reason2-8B model via vLLM API."""

    def __init__(
        self,
        api_base: str = VLLM_API_BASE,
        model_id: str = MODEL_ID,
    ):
        self.api_base = api_base
        self.model_id = model_id
        print(f"CosmosModelHandler using vLLM API at {self.api_base} ({self.model_id})")

    def analyze_single_frame(
        self,
        image: Image.Image,
        prompt: str = "Describe what is happening in this image in detail.",
    ) -> str:
        messages = [
            {"role": "system", "content": "You are a video surveillance analyst."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_url(image)},
                    },
                ],
            },
        ]
        try:
            return _call_api(messages)
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            return f"Error: Could not analyze frame - {str(e)}"

    def analyze_frames(
        self,
        frames: List[Image.Image],
        batch_size: int = 1,
    ) -> List[Dict[str, str]]:
        descriptions = []
        for idx, frame in enumerate(frames):
            print(f"Analyzing frame {idx + 1}/{len(frames)}...")
            prompt = (
                f"You are analyzing frame {idx + 1} of a video. "
                "Describe what is happening, including: "
                "1) The main subjects or objects, "
                "2) The action or activity taking place, "
                "3) The setting or environment, "
                "4) Any notable details. "
                "Be concise but informative."
            )
            description = self.analyze_single_frame(frame, prompt)
            descriptions.append({"frame_index": idx, "description": description})
        return descriptions

    def analyze_video_frames(
        self,
        frames: List[Image.Image],
        prompt: str = "Analyze this footage and describe what is happening.",
        fps: int = 4,
    ) -> str:
        """Send all frames as a single video to the model."""
        video_data_url = _frames_to_video_data_url(frames, fps=fps)
        messages = [
            {"role": "system", "content": "You are a video surveillance analyst."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "video_url",
                        "video_url": {"url": video_data_url},
                    },
                ],
            },
        ]
        try:
            return _call_api(messages)
        except Exception as e:
            print(f"Error analyzing video: {e}")
            return f"Error: Could not analyze video - {str(e)}"

    def analyze_with_context(
        self,
        frames: List[Image.Image],
        previous_context: str = "",
    ) -> List[Dict[str, str]]:
        descriptions = []
        context = previous_context

        for idx, frame in enumerate(frames):
            print(f"Analyzing frame {idx + 1}/{len(frames)} with context...")
            if context:
                prompt = (
                    f"Previously in this video: {context}\n\n"
                    "Now, describe what is happening in this new frame. "
                    "Focus on what has changed or what is new."
                )
            else:
                prompt = "This is the first frame of a video. Describe what you see in detail."

            description = self.analyze_single_frame(frame, prompt)
            descriptions.append({"frame_index": idx, "description": description})
            context = description

        return descriptions

    def cleanup(self):
        """No-op — resources are managed by the remote vLLM container."""
        pass
