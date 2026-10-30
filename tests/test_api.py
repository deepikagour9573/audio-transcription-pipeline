
import pytest
from fastapi.testclient import TestClient

from app.main import app


# Reuse one test client fixture so each test can make requests against the FastAPI app.
@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


# The health endpoint should tell us whether the main runtime pieces are ready.
def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "ffmpeg_available" in data


# A text file should be rejected before it reaches the transcription service.
def test_unsupported_audio_format(client):
    response = client.post(
        "/transcriptions",
        files={"file": ("notes.txt", b"not an audio file", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported audio format"


# Unknown IDs should return a clear 404 instead of an empty response.
def test_missing_transcription_job(client):
    response = client.get(
        "/transcriptions/00000000-0000-0000-0000-000000000000"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Transcription job not found"


def test_retry_missing_job(client):
    response = client.post(
        "/transcriptions/00000000-0000-0000-0000-000000000000/retry"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Transcription job not found"

def test_retry_nonexistent_job_returns_404(client):
    response = client.post(
        "/transcriptions/11111111-1111-1111-1111-111111111111/retry"
    )

    assert response.status_code == 404


def test_retry_only_allows_failed_jobs(client):
    import uuid
    from app.settings import AUDIO_DIR
    from app.models import connect_db, now_iso

    job_id = str(uuid.uuid4())
    audio_path = AUDIO_DIR / f"{job_id}.mp3"
    audio_path.write_bytes(b"test audio placeholder")

    timestamp = now_iso()

    try:
        with connect_db() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, original_filename, audio_path, status,
                    attempts, created_at, updated_at
                )
                VALUES (?, ?, ?, 'completed', 1, ?, ?)
                """,
                (
                    job_id,
                    "test.mp3",
                    str(audio_path),
                    timestamp,
                    timestamp,
                ),
            )

        response = client.post(f"/transcriptions/{job_id}/retry")

        assert response.status_code == 409
        assert response.json()["detail"] == "Only failed jobs can be retried"

    finally:
        with connect_db() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        audio_path.unlink(missing_ok=True)


def test_bulk_upload_rejects_unsupported_format(client):
    response = client.post(
        "/transcriptions/bulk",
        files=[
            ("files", ("notes.txt", b"not audio", "text/plain")),
        ],
    )

    assert response.status_code == 202
    data = response.json()
    assert data["total_files"] == 1
    assert data["accepted"] == 0
    assert data["rejected"] == 1
    assert data["results"][0]["status"] == "rejected"
    assert data["results"][0]["error"] == "Unsupported audio format"


def test_bulk_upload_rejects_more_than_10_files(client):
    files = [
        ("files", (f"audio{i}.mp3", b"fake audio", "audio/mpeg"))
        for i in range(11)
    ]

    response = client.post("/transcriptions/bulk", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "A maximum of 10 files can be uploaded at once"
    )


# Keep this test focused on job creation; real model inference belongs in an integration test.
def test_bulk_upload_accepts_multiple_valid_files(client, monkeypatch):
    from app.settings import AUDIO_DIR
    from app.models import connect_db

    # Keep this test focused on upload/job creation, not Whisper inference.
    monkeypatch.setattr("app.services.schedule_job", lambda job_id: None)

    files = [
        ("files", ("sample1.mp3", b"fake audio content 1", "audio/mpeg")),
        ("files", ("sample2.wav", b"fake audio content 2", "audio/wav")),
    ]
    job_ids = []

    try:
        response = client.post("/transcriptions/bulk", files=files)

        assert response.status_code == 202
        data = response.json()
        assert data["total_files"] == 2
        assert data["accepted"] == 2
        assert data["rejected"] == 0
        assert len(data["results"]) == 2

        for result in data["results"]:
            assert result["status"] == "queued"
            assert result["job_id"]
            job_ids.append(result["job_id"])

        with connect_db() as conn:
            rows = conn.execute(
                "SELECT job_id, status FROM jobs WHERE job_id IN (?, ?)",
                tuple(job_ids),
            ).fetchall()

        assert len(rows) == 2
        assert all(row["status"] == "queued" for row in rows)

    finally:
        if job_ids:
            with connect_db() as conn:
                conn.executemany(
                    "DELETE FROM jobs WHERE job_id = ?",
                    [(job_id,) for job_id in job_ids],
                )

            for job_id in job_ids:
                for suffix in (".mp3", ".wav"):
                    (AUDIO_DIR / f"{job_id}{suffix}").unlink(missing_ok=True)
