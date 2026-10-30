from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Resolve paths from the project folder, not from whichever directory starts the command.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = Path(os.getenv("TRANSCRIPTION_DATA_DIR", BASE_DIR / "data"))
AUDIO_DIR = DATA_DIR / "audio"
DB_PATH = DATA_DIR / "transcriptions.db"

# Keep upload limits and model choices in one place so they are easy to adjust.
SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
MAX_CONCURRENT_TRANSCRIPTIONS = int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "2"))
MAX_BULK_FILES = 10

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"

# Prefer an explicit FFmpeg path from .env; otherwise look for ffmpeg on PATH.
def resolve_ffmpeg_executable() -> Optional[Path]:
    configured = os.getenv("FFMPEG_PATH")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_absolute() or candidate.parent != Path("."):
            return candidate
        resolved = shutil.which(configured)
        return Path(resolved) if resolved else None
    resolved = shutil.which("ffmpeg")
    return Path(resolved) if resolved else None

FFMPEG_EXECUTABLE = resolve_ffmpeg_executable()

# These folders are created automatically on a fresh checkout.
DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
