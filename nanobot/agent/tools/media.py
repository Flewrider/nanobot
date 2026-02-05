"""Media analysis tools: analyze video, audio, and image files."""

import base64
import mimetypes
import json
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


# Supported media types for Gemini 3 Flash
SUPPORTED_VIDEO = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".mpeg", ".mpg", ".3gp"}
SUPPORTED_AUDIO = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".wma", ".opus"}
SUPPORTED_IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".gif", ".bmp"}

# Max file sizes (in bytes) - conservative limits for inline encoding
MAX_VIDEO_SIZE = 20 * 1024 * 1024  # 20MB
MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25MB
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


class AnalyzeMediaTool(Tool):
    """
    Tool to analyze video, audio, or image files using the LLM's vision/audio capabilities.

    The tool encodes the media and returns a special JSON marker that the agent loop
    will detect and inject into the next LLM call as multimodal content.
    """

    @property
    def name(self) -> str:
        return "analyze_media"

    @property
    def description(self) -> str:
        return (
            "Analyze a video, audio, or image file. The file will be sent to the AI for analysis. "
            "Use this to review videos you've created, inspect images, or transcribe/analyze audio. "
            "Supports: video (mp4, webm, mov, etc.), audio (mp3, wav, flac, etc.), image (png, jpg, webp, etc.). "
            "Optionally provide a prompt to guide the analysis (e.g., 'Check if the text is readable')."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the media file to analyze"},
                "prompt": {
                    "type": "string",
                    "description": "Optional prompt to guide the analysis (e.g., 'Describe what happens in this video', 'Is the audio clear?', 'What text is visible?')",
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        path: str = kwargs["path"]
        prompt: str = kwargs.get(
            "prompt", "Please analyze this media and describe what you see/hear."
        )

        try:
            file_path = Path(path).expanduser().resolve()

            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Not a file: {path}"

            suffix = file_path.suffix.lower()
            file_size = file_path.stat().st_size

            # Determine media type
            if suffix in SUPPORTED_VIDEO:
                media_type = "video"
                max_size = MAX_VIDEO_SIZE
            elif suffix in SUPPORTED_AUDIO:
                media_type = "audio"
                max_size = MAX_AUDIO_SIZE
            elif suffix in SUPPORTED_IMAGE:
                media_type = "image"
                max_size = MAX_IMAGE_SIZE
            else:
                return f"Error: Unsupported file type: {suffix}. Supported: video ({', '.join(SUPPORTED_VIDEO)}), audio ({', '.join(SUPPORTED_AUDIO)}), image ({', '.join(SUPPORTED_IMAGE)})"

            # Check file size
            if file_size > max_size:
                size_mb = file_size / (1024 * 1024)
                max_mb = max_size / (1024 * 1024)
                return f"Error: File too large ({size_mb:.1f}MB). Maximum for {media_type}: {max_mb:.0f}MB"

            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type:
                # Fallback MIME types
                mime_map = {
                    ".mp4": "video/mp4",
                    ".webm": "video/webm",
                    ".mov": "video/quicktime",
                    ".avi": "video/x-msvideo",
                    ".mkv": "video/x-matroska",
                    ".mp3": "audio/mpeg",
                    ".wav": "audio/wav",
                    ".flac": "audio/flac",
                    ".ogg": "audio/ogg",
                    ".m4a": "audio/mp4",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".webp": "image/webp",
                    ".gif": "image/gif",
                }
                mime_type = mime_map.get(suffix, f"{media_type}/{suffix[1:]}")

            # Encode file as base64
            b64_data = base64.b64encode(file_path.read_bytes()).decode("utf-8")

            # Return special marker that the loop will detect
            # This JSON structure tells the loop to inject the media into the next LLM call
            marker = {
                "__media_injection__": True,
                "media_type": media_type,
                "mime_type": mime_type,
                "data": b64_data,
                "path": str(file_path),
                "prompt": prompt,
                "size_bytes": file_size,
            }

            return json.dumps(marker)

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error analyzing media: {str(e)}"


def parse_media_injection(result: str) -> dict[str, Any] | None:
    """
    Check if a tool result contains a media injection marker.

    Returns the parsed marker dict if found, None otherwise.
    """
    if not result.startswith('{"__media_injection__":'):
        return None

    try:
        data = json.loads(result)
        if data.get("__media_injection__"):
            return data
    except (json.JSONDecodeError, KeyError):
        pass

    return None


def build_media_content(media_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build multimodal content block from media injection data.

    Returns a list suitable for a user message content array.
    """
    media_type = media_data["media_type"]
    mime_type = media_data["mime_type"]
    b64_data = media_data["data"]
    prompt = media_data.get("prompt", "Please analyze this media.")

    content = []

    # Add the prompt as text
    content.append({"type": "text", "text": prompt})

    # Add the media
    if media_type == "image":
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}}
        )
    elif media_type == "video":
        # Gemini format for video
        content.append(
            {
                "type": "image_url",  # Gemini uses image_url for video too via litellm
                "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
            }
        )
    elif media_type == "audio":
        # Gemini format for audio
        content.append(
            {
                "type": "input_audio",
                "input_audio": {
                    "data": b64_data,
                    "format": mime_type.split("/")[-1],  # e.g., "mp3", "wav"
                },
            }
        )

    return content
