"""Business logic: audio storage, model lifecycle and background transcription."""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile

from app.models import get_job, update_job
from app.settings import (
    AUDIO_DIR, FFMPEG_EXECUTABLE, MAX_CONCURRENT_TRANSCRIPTIONS,
    MAX_UPLOAD_BYTES, WHISPER_BEAM_SIZE, WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE, WHISPER_MODEL, WHISPER_VAD_FILTER,
)

# Keep the model and background-task state in this service module, away from HTTP details.
logger = logging.getLogger(__name__)
transcription_slots = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)
active_tasks: set[asyncio.Task] = set()
model = None
ffmpeg_version: Optional[str] = None
_ffmpeg_dll_directory_handle = None

# Check FFmpeg at startup so audio decoding problems are caught before a user uploads a file.
def verify_ffmpeg() -> str:
    if FFMPEG_EXECUTABLE is None:
        raise RuntimeError(
            "FFmpeg was not found. Set FFMPEG_PATH to the full path of ffmpeg.exe "
            "or add FFmpeg to PATH."
        )
    if not FFMPEG_EXECUTABLE.is_file():
        raise RuntimeError(f"FFmpeg executable does not exist: {FFMPEG_EXECUTABLE}")
    try:
        result = subprocess.run(
            [str(FFMPEG_EXECUTABLE), "-version"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not run FFmpeg at {FFMPEG_EXECUTABLE}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"FFmpeg version check failed: {detail}")
    return result.stdout.splitlines()[0] if result.stdout else "FFmpeg is available"

def configure_ffmpeg_environment() -> None:
    global _ffmpeg_dll_directory_handle
    if FFMPEG_EXECUTABLE is None:
        return
    bin_dir = str(FFMPEG_EXECUTABLE.parent)
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir.casefold() not in {entry.casefold() for entry in entries}:
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    if os.name == "nt":
        _ffmpeg_dll_directory_handle = os.add_dll_directory(bin_dir)

# Load the speech model once at startup instead of reloading it for every upload.
async def initialize_services() -> None:
    global model, ffmpeg_version
    ffmpeg_version = await asyncio.to_thread(verify_ffmpeg)
    configure_ffmpeg_environment()
    try:
        from faster_whisper import WhisperModel
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Could not import Faster-Whisper/PyAV native dependencies. "
            "Check the installed package files and dependent DLLs."
        ) from exc
    model = await asyncio.to_thread(
        WhisperModel, WHISPER_MODEL, device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )

# Read in small pieces so a large upload does not need to sit in memory all at once.
async def save_upload(file: UploadFile, destination: Path) -> int:
    total = 0
    try:
        with destination.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413, detail="Audio file exceeds the upload limit"
                    )
                out.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return total

# This is the background journey for one job: queued, processing, then completed or failed.
async def run_transcription(job_id: str) -> None:
    job = get_job(job_id)
    if not job:
        return
    async with transcription_slots:
        update_job(job_id, status="processing", error=None,
                   attempts=job["attempts"] + 1)
        try:
            if model is None:
                raise RuntimeError("Transcription model is not ready")

            # Faster-Whisper is synchronous, so we move its work to a thread below.
            def transcribe_sync():
                segments, info = model.transcribe(
                    job["audio_path"], beam_size=WHISPER_BEAM_SIZE,
                    vad_filter=WHISPER_VAD_FILTER,
                )
                rows, text_parts = [], []
                for segment in segments:
                    segment_text = segment.text.strip()
                    text_parts.append(segment_text)
                    rows.append({
                        "start": round(segment.start, 2),
                        "end": round(segment.end, 2),
                        "text": segment_text,
                    })
                return {
                    "transcript": " ".join(text_parts).strip(),
                    "segments": rows,
                    "language": info.language,
                    "duration_seconds": round(info.duration, 2),
                }

            # Let the event loop keep serving requests while the model works in a worker thread.
            result = await asyncio.to_thread(transcribe_sync)
            import json
            update_job(
                job_id, status="completed", transcript=result["transcript"],
                segments_json=json.dumps(result["segments"]),
                language=result["language"],
                duration_seconds=result["duration_seconds"], error=None,
            )
        except asyncio.CancelledError:
            update_job(job_id, status="failed",
                       error="Job cancelled during service shutdown")
            raise
        except Exception:
            logger.exception("Transcription failed for job %s", job_id)
            update_job(
                job_id, status="failed",
                error="Transcription failed. Check server logs for details.",
            )

# Keep a reference to each task so it is not lost while it is running.
def schedule_job(job_id: str) -> None:
    task = asyncio.create_task(run_transcription(job_id))
    active_tasks.add(task)
    task.add_done_callback(active_tasks.discard)

# Stop active tasks cleanly when the server is shutting down.
async def shutdown_services() -> None:
    for task in list(active_tasks):
        task.cancel()
    if active_tasks:
        await asyncio.gather(*active_tasks, return_exceptions=True)
