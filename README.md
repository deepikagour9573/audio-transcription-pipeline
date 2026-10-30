# Audio Transcription Pipeline

A Python-based audio transcription service built with **FastAPI** and
**Faster-Whisper**. It accepts audio uploads, creates trackable
transcription jobs, processes audio in the background, and provides
transcript results with segment-level timestamps.

## Features

-   Upload a single audio file.
-   Upload multiple audio files in one request (up to 10 files).
-   Validate supported file extensions and reject empty or oversized
    uploads.
-   Save audio files locally using unique UUID-based filenames.
-   Track each transcription request with a job ID.
-   Process transcription jobs in the background using Python `asyncio`.
-   Limit concurrent transcription operations with an
    `asyncio.Semaphore`.
-   Run synchronous Faster-Whisper inference through
    `asyncio.to_thread()`.
-   Return transcript text, segment timestamps, detected language, and
    audio duration.
-   Check job status and retrieve completed results.
-   Manually retry failed jobs when the original audio file is still
    available.
-   Check application, model, and FFmpeg readiness through a health
    endpoint.
-   Run API tests with Pytest.

## Technology Stack

  Component                            Technology
  ------------------------------------ -------------------------------------------------
  Language                             Python 3.12
  API framework                        FastAPI
  ASGI server                          Uvicorn
  Speech recognition                   Faster-Whisper
  Inference engine                     CTranslate2
  Audio decoding                       PyAV; FFmpeg availability is checked at startup
  Database                             SQLite
  Audio storage                        Local filesystem
  Background processing                `asyncio.create_task()`
  Non-blocking synchronous inference   `asyncio.to_thread()`
  Concurrency control                  `asyncio.Semaphore`
  Testing                              Pytest and FastAPI TestClient

The application uses a pretrained Faster-Whisper **base** model by
default. It performs inference; it does not train or fine-tune a speech
recognition model.

## Project Structure

``` text
transcription_pipeline/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api.py
│   ├── models.py
│   ├── services.py
│   └── settings.py
│
├── data/
│   ├── audio/
│   └── transcriptions.db
│
├── tests/
│   ├── __init__.py
│   └── test_api.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

### Responsibilities of the application modules

-   **`main.py`** -- Creates the FastAPI application, registers the API
    router, and manages startup and shutdown.
-   **`api.py`** -- Defines the HTTP endpoints for health checks, single
    uploads, bulk uploads, job status/result retrieval, and retries.
-   **`models.py`** -- Handles SQLite connections and transcription job
    records, including creating, retrieving, and updating jobs.
-   **`services.py`** -- Contains the application services for FFmpeg
    checks, Faster-Whisper initialization, upload saving, background job
    scheduling, and transcription processing.
-   **`settings.py`** -- Centralizes application paths,
    environment-based settings, upload limits, concurrency settings,
    model configuration, and FFmpeg path resolution.
-   **`tests/`** -- Contains automated API tests.

## How the Processing Flow Works

1.  The client uploads an audio file to `POST /transcriptions`.
2.  The API validates the extension and saves the upload in chunks of 1
    MB, enforcing the configured maximum upload size.
3.  The application generates a UUID job ID and creates a SQLite job
    record with status `queued`.
4.  The API schedules the transcription coroutine using
    `asyncio.create_task()` and immediately returns **HTTP 202
    Accepted** with the job ID. It does not wait for Whisper to finish.
5.  The background service moves the job to `processing` and runs
    Faster-Whisper. Synchronous model inference is run using
    `asyncio.to_thread()` so it does not block the event loop.
6.  The service saves the full transcript, segment-level timestamps,
    detected language, and audio duration in SQLite, then marks the job
    `completed`.
7.  If processing fails, the job is marked `failed`. The client can
    check the current status through the job status endpoint.

The client can poll `GET /transcriptions/{job_id}` to check whether a
job is `queued`, `processing`, `completed`, or `failed`.

> **Current implementation note:** Background tasks run inside the
> FastAPI application process. They are not distributed or persisted in
> an external queue. If the application stops while a job is running,
> the task can be lost. On startup, jobs left in `queued` or
> `processing` are marked as failed; they are not automatically resumed.

## Supported Audio Formats and Limits

The application accepts files with these extensions:

-   `.wav`
-   `.mp3`
-   `.m4a`
-   `.flac`
-   `.ogg`

Default limits:

-   Maximum upload size: **100 MB per file**
-   Maximum files in one bulk request: **10**
-   Maximum concurrent transcription operations: **2**

The upload is written to disk in 1 MB pieces to avoid reading the entire
file into memory at once. This is **chunked file writing**, not audio
segmentation. The current implementation passes the saved audio file to
Faster-Whisper as a whole; explicit long-audio chunking and timestamp
offset merging are not implemented.

## Database and File Storage

SQLite stores job metadata and transcription results in
`data/transcriptions.db`. The `jobs` table includes:

-   `job_id`
-   `original_filename`
-   `audio_path`
-   `status`
-   `attempts`
-   `transcript`
-   `segments_json`
-   `language`
-   `duration_seconds`
-   `error`
-   `created_at`
-   `updated_at`

Audio files are stored separately in `data/audio/` with UUID-based
filenames. The database stores the audio file path; audio binary content
is not stored in SQLite.

The data directory and audio directory are created automatically when
the application settings module is imported.

## API Endpoints

  ----------------------------------------------------------------------------------
  Method                  Endpoint                           Description
  ----------------------- ---------------------------------- -----------------------
  `GET`                   `/health`                          Reports application,
                                                             model, and FFmpeg
                                                             readiness information.

  `POST`                  `/transcriptions`                  Uploads one audio file
                                                             and returns a job ID.

  `POST`                  `/transcriptions/bulk`             Accepts up to 10 files
                                                             and reports
                                                             accepted/rejected
                                                             files.

  `GET`                   `/transcriptions/{job_id}`         Returns job status and,
                                                             when completed,
                                                             transcription details.

  `POST`                  `/transcriptions/{job_id}/retry`   Schedules a retry for a
                                                             failed job if its audio
                                                             file still exists.
  ----------------------------------------------------------------------------------

### Common HTTP status codes

  Status code   Meaning
  ------------- -------------------------------------------------------------
  `200`         Request completed successfully, such as a status lookup.
  `202`         Upload or retry request accepted for background processing.
  `400`         Unsupported format, empty upload, or invalid request.
  `404`         Transcription job was not found.
  `409`         Retry requested for a job that is not failed.
  `410`         Original audio file is no longer available for retry.
  `413`         Uploaded file exceeds the configured size limit.

## Setup on Windows

### 1. Prerequisites

-   Python 3.12
-   Git (if cloning the repository)
-   FFmpeg installed and available on `PATH`, or configured through
    `FFMPEG_PATH`

Check Python and FFmpeg:

``` bat
python --version
ffmpeg -version
```

### 2. Create and activate a virtual environment

Run these commands from the project root (the folder containing
`requirements.txt`):

``` bat
python -m venv .venv
.venv\Scripts\activate
```

If you already have a virtual environment such as `.venv312`, you can
activate that instead.

### 3. Install dependencies

``` bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example configuration:

``` bat
copy .env.example .env
```

Edit `.env` if you need to change the defaults. Do not commit `.env` to
GitHub.

### 5. Start the API

From the project root:

``` bat
python -m uvicorn app.main:app --reload
```

If port 8000 is unavailable on your machine, use another port:

``` bat
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 9000
```

The first startup may take longer because Faster-Whisper may need to
download the configured model.

## Try the API

Open Swagger UI in your browser:

-   `http://127.0.0.1:8000/docs`

If you started Uvicorn on port 9000, use `http://127.0.0.1:9000/docs`.

Suggested manual test:

1.  Open `POST /transcriptions` in Swagger.
2.  Click **Try it out**, select an audio file, and click **Execute**.
3.  Copy the returned `job_id`.
4.  Call `GET /transcriptions/{job_id}` with that ID.
5.  Check the status until it becomes `completed` or `failed`.
6.  For a completed job, inspect the full transcript and timestamped
    segments.

You can also check readiness at `/health`.

## Run Tests

From the project root, with the virtual environment activated:

``` bat
python -m pytest -v
```

The API tests use FastAPI TestClient. Some upload tests replace
background scheduling so they can verify upload validation and job
creation without running actual Whisper inference.

The test suite does not replace a manual or automated end-to-end test
using a real audio sample. A real-audio test is useful for checking
model inference, transcript content, and timestamps.

## Configuration Reference

The application reads optional settings from `.env`. The defaults are:

  ---------------------------------------------------------------------------------
  Variable                          Default                 Purpose
  --------------------------------- ----------------------- -----------------------
  `TRANSCRIPTION_DATA_DIR`          `data`                  Location for audio
                                                            storage and SQLite
                                                            database

  `MAX_UPLOAD_BYTES`                `104857600`             Maximum upload size in
                                                            bytes (100 MB)

  `MAX_CONCURRENT_TRANSCRIPTIONS`   `2`                     Concurrent
                                                            transcription limit

  `WHISPER_MODEL`                   `base`                  Faster-Whisper model
                                                            name

  `WHISPER_DEVICE`                  `cpu`                   Inference device

  `WHISPER_COMPUTE_TYPE`            `int8`                  Model computation type

  `WHISPER_BEAM_SIZE`               `5`                     Beam search size

  `WHISPER_VAD_FILTER`              `true`                  Enable voice activity
                                                            detection filter

  `FFMPEG_PATH`                     Auto-detected           Optional path to the
                                                            FFmpeg executable
  ---------------------------------------------------------------------------------

## Current Scope and Limitations

The current project provides a working foundation for an audio
transcription workflow. The following are not implemented in the current
version:

-   PostgreSQL or other server-based database
-   Redis, Celery, or a durable distributed task queue
-   Automatic retry with exponential backoff or dead-letter queue
-   Explicit audio segmentation for long recordings
-   Cloud object storage such as Amazon S3 or Azure Blob Storage
-   User authentication, JWT, or role-based access control
-   Administration dashboard
-   Speaker diarization, summaries, action-item extraction, or
    transcript question answering
-   Production monitoring, CI/CD, and deployment automation

## Possible Future Improvements

For a production-oriented version, I would consider:

1.  Moving job metadata to PostgreSQL and adding schema migrations.
2.  Using Redis and Celery for durable background processing and
    independently scalable workers.
3.  Adding automatic retries, backoff, and dead-letter handling.
4.  Moving audio files to object storage such as Amazon S3 or Azure Blob
    Storage.
5.  Implementing long-audio chunking with overlap handling and timestamp
    offset correction.
6.  Adding authentication, authorization, rate limits, quotas, and audit
    logging.
7.  Building an administration dashboard for job monitoring and
    transcript management.
8.  Adding diarization, summaries, action items, and multilingual
    capabilities.
9.  Containerizing the application and adding CI/CD, monitoring,
    logging, and backup policies.

These are planned improvements, not features of the current
implementation.

## Security and Repository Notes

-   Do not commit `.env`, credentials, API keys, or passwords.
-   Do not commit your local virtual environment, generated Python
    cache, test cache, uploaded audio files, or local SQLite database.
-   Keep `.env.example` in the repository with safe placeholder/default
    values.
-   Review `git status` before committing to make sure only intended
    source and documentation files are staged.

------------------------------------------------------------------------

**Project:** Audio Transcription Pipeline\
**Backend:** FastAPI\
**Speech Recognition:** Faster-Whisper\
**Database:** SQLite\
**Language:** Python 3.12
