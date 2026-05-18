# STEADYVIEW

FastAPI version of the STEADYVIEW video restoration service.

## Structure

- `app.py`: FastAPI entry point, upload/preview/process API routes, and static UI serving
- `backend/video_processor.py`: Backend boundary for video restoration logic
- `static/index.html`: Browser UI
- `static/styles.css`: UI styling
- `static/app.js`: Upload, preview, process, and download behavior

## Run

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

- `GET /api/health`: health check
- `GET /api/options`: available restoration options
- `POST /api/preview`: returns browser-playable MP4 preview bytes
- `POST /api/process`: returns processed MP4 bytes

`process_video()` currently returns a browser-playable version of the uploaded
video for options that are not wired yet. `Stabilization + Upright Correction`
is wired to `backend/pipelines/upright_stabilization`.

## Upright Model

Put the upright model checkpoint here:

```text
backend/pipelines/upright_stabilization/models/best_stage2_true_hybrid_real_best.pth
```

Or set a custom path in PowerShell before running the server:

```powershell
$env:STEADYVIEW_UPRIGHT_MODEL_PATH="C:\path\to\model.pth"
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
Real-ESRGAN from:

```text
C:\Users\korea\Desktop\Real-ESRGAN\Real-ESRGAN_V2
```

If it lives elsewhere, set this before `docker compose up`:

```powershell
$env:STEADYVIEW_REAL_ESRGAN_HOST_DIR="C:\path\to\Real-ESRGAN_V2"
```
