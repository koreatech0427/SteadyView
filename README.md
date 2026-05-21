# STEADYVIEW

FastAPI video restoration service with a browser UI for previewing, processing,
comparing, and downloading restored videos.

## Features

- Superresolution with BasicVSR++
- Stabilization-only correction
- Upright correction-only processing
- Combined processing for Superresolution, Stabilization, and Upright Correction
- Browser-playable MP4 preview/output conversion
- Background job processing with progress polling and cancellation
- Chunked uploads for large video files
- Original audio muxing back into processed output when available

## Structure

- `app.py`: FastAPI entry point, static UI serving, upload, job, preview, and result API routes
- `backend/video_processor.py`: Processing orchestration for selected restoration options
- `backend/pipelines/superresolution`: BasicVSR++ runner
- `backend/pipelines/stabilization`: Stabilization-only pipeline
- `backend/pipelines/upright_stabilization`: Upright and stabilization + upright pipelines
- `static/index.html`: Browser UI
- `static/styles.css`: UI styling
- `static/app.js`: Upload, preview, job polling, cancellation, comparison, and download behavior

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open:

```text
http://localhost:8000
```

## Team

| 역할 | 이름 | 소속 |
| --- | --- | --- |
| 팀장 | 강혁 | 한국기술교육대학교 컴퓨터공학부 21학번 |
| 팀원 | 이은호 | 한국기술교육대학교 컴퓨터공학부 21학번 |
| 팀원 | 박은준 | 한국기술교육대학교 컴퓨터공학부 21학번 |
| 팀원 | 김재인 | 한국기술교육대학교 컴퓨터공학부 23학번 |

## API

- `GET /api/health`: Health check
- `GET /api/options`: Available restoration options
- `GET /api/runtime`: Runtime device information
- `POST /api/preview`: Return browser-playable MP4 preview bytes
- `POST /api/process`: Process a video synchronously and return MP4 bytes
- `POST /api/jobs`: Create an async processing job
- `GET /api/jobs/{job_id}`: Get job status and progress
- `POST /api/jobs/{job_id}/cancel`: Request job cancellation
- `GET /api/jobs/{job_id}/result`: Download completed result
- `HEAD /api/jobs/{job_id}/result`: Check completed result metadata
- `POST /api/uploads`: Create a chunked upload session
- `POST /api/uploads/{upload_id}/chunks`: Upload one file chunk
- `POST /api/uploads/{upload_id}/complete`: Assemble chunks and start a job

## Restoration Options

The UI and API support these options:

- `Superresolution`
- `Stabilization`
- `Upright Correction`
- `Superresolution + Stabilization`
- `Superresolution + Upright Correction`
- `Stabilization + Upright Correction`
- `Superresolution + Stabilization + Upright Correction`

`backend/video_processor.py` runs motion correction stages first, then runs
Superresolution when selected. Final output is prepared as a browser-playable
MP4, and original audio is copied back when the source video has audio.

## Upright Model

Put the upright model checkpoint here:

```text
backend/pipelines/upright_stabilization/models/best_stage2_true_hybrid_real_best.pth
```

Or set a custom path before running the server:

```powershell
$env:STEADYVIEW_UPRIGHT_MODEL_PATH="C:\path\to\model.pth"
```

## BasicVSR++ Superresolution

Superresolution expects a BasicVSR++ checkout that contains:

```text
run_basicvsrpp_chunk.py
```

Default local path on Windows:

```text
C:\Users\korea\Desktop\BasicVSR++
```

Default Docker path:

```text
/opt/basicvsrpp
```

To use a custom location:

```powershell
$env:STEADYVIEW_BASICVSRPP_DIR="C:\path\to\BasicVSR++"
```

Optional settings:

```powershell
$env:STEADYVIEW_BASICVSRPP_PYTHON="C:\path\to\python.exe"
$env:STEADYVIEW_BASICVSRPP_CHUNK_SIZE="8"
$env:STEADYVIEW_BASICVSRPP_MODEL="real_basicvsr"
```

## Run With Docker

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8000
```

For `Superresolution` or combined options that include it, Docker mounts
BasicVSR++ from:

```text
C:\Users\korea\Desktop\BasicVSR++
```

If it lives elsewhere, set this before `docker compose up`:

```powershell
$env:STEADYVIEW_BASICVSRPP_HOST_DIR="C:\path\to\BasicVSR++"
```
