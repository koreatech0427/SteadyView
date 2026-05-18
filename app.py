import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.video_processor import (
    VideoConversionError,
    VideoProcessingError,
    make_browser_playable,
    process_video_file,
    process_video,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
JOBS_DIR = BASE_DIR / "jobs"
UPLOAD_CHUNK_SIZE = 1024 * 1024

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}
RESTORATION_OPTIONS = {
    "Superresolution",
    "Stabilization",
    "Upright Correction",
    "Superresolution + Stabilization",
    "Superresolution + Upright Correction",
    "Stabilization + Upright Correction",
    "Superresolution + Stabilization + Upright Correction",
}

app = FastAPI(title="STEADYVIEW API", version="2.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

jobs_lock = threading.Lock()
jobs: dict[str, dict[str, object]] = {}
JOBS_DIR.mkdir(exist_ok=True)


@app.middleware("http")
async def add_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers.setdefault("Cache-Control", "public, max-age=3600")
    elif request.url.path == "/":
        response.headers.setdefault("Cache-Control", "no-cache")
    elif request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/options")
def options() -> dict[str, list[str]]:
    return {"options": sorted(RESTORATION_OPTIONS)}


@app.get("/api/runtime")
def runtime() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {
            "device_type": "cpu",
            "device_name": "CPU",
            "cuda_available": False,
            "torch_available": False,
        }

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    return {
        "device_type": "gpu" if cuda_available else "cpu",
        "device_name": device_name,
        "cuda_available": cuda_available,
        "torch_available": True,
        "torch_cuda": torch.version.cuda,
    }


@app.post("/api/preview")
async def preview_video(file: UploadFile = File(...)) -> Response:
    file_name = _validate_video_file(file.filename)
    video_bytes = await file.read()

    try:
        preview_bytes = make_browser_playable(video_bytes, file_name)
    except VideoConversionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(content=preview_bytes, media_type="video/mp4")


@app.post("/api/jobs", status_code=202)
async def create_job(
    option: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, object]:
    file_name = _validate_video_file(file.filename)
    if option not in RESTORATION_OPTIONS:
        raise HTTPException(status_code=400, detail="지원하지 않는 복원 옵션입니다.")

    job_id = uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    input_path = job_dir / f"input{Path(file_name).suffix.lower()}"
    await _save_upload_file(file, input_path)

    job = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "작업을 준비하고 있습니다.",
        "option": option,
        "source_name": file_name,
        "result_name": f"steadyview_{Path(file_name).with_suffix('.mp4').name}",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    _save_job(job_id, job)

    worker = threading.Thread(
        target=_run_job,
        args=(job_id, input_path, option, file_name),
        daemon=True,
    )
    worker.start()

    return _public_job(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    return _public_job(_get_job_or_404(job_id))


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str) -> FileResponse:
    job = _get_job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="아직 작업이 완료되지 않았습니다.")

    result_path = JOBS_DIR / job_id / "output.mp4"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="결과 영상 파일을 찾을 수 없습니다.")

    return FileResponse(
        result_path,
        media_type="video/mp4",
        filename=str(job["result_name"]),
    )


@app.head("/api/jobs/{job_id}/result")
def head_job_result(job_id: str) -> Response:
    job = _get_job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="?꾩쭅 ?묒뾽???꾨즺?섏? ?딆븯?듬땲??")

    result_path = JOBS_DIR / job_id / "output.mp4"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="寃곌낵 ?곸긽 ?뚯씪??李얠쓣 ???놁뒿?덈떎.")

    return Response(
        media_type="video/mp4",
        headers={
            "Content-Length": str(result_path.stat().st_size),
            "Accept-Ranges": "bytes",
        },
    )


@app.post("/api/process")
async def process_video_endpoint(
    option: str = Form(...),
    file: UploadFile = File(...),
) -> Response:
    file_name = _validate_video_file(file.filename)
    if option not in RESTORATION_OPTIONS:
        raise HTTPException(status_code=400, detail="지원하지 않는 복원 옵션입니다.")

    video_bytes = await file.read()

    try:
        result = process_video(video_bytes, option, file_name)
    except VideoConversionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except VideoProcessingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    headers = {
        "Content-Disposition": f'attachment; filename="steadyview_{result.file_name}"',
        "X-Steadyview-Option": result.option,
    }
    return Response(content=result.video_bytes, media_type="video/mp4", headers=headers)


def _run_job(job_id: str, input_path: Path, option: str, file_name: str) -> None:
    _update_job(job_id, status="running", progress=5, message="영상 처리를 시작했습니다.")
    try:
        def report(progress: int, message: str) -> None:
            _update_job(job_id, status="running", progress=progress, message=message)

        result_path = JOBS_DIR / job_id / "output.mp4"
        result = process_video_file(input_path, option, file_name, result_path, progress_callback=report)
        _update_job(
            job_id,
            status="done",
            progress=100,
            message="영상 처리가 완료되었습니다.",
            result_name=f"steadyview_{result.file_name}",
        )
    except (VideoConversionError, VideoProcessingError) as exc:
        _update_job(job_id, status="failed", progress=100, message=_user_error_message(exc))
    except Exception as exc:
        _update_job(job_id, status="failed", progress=100, message=_user_error_message(exc))


async def _save_upload_file(file: UploadFile, target_path: Path) -> None:
    with target_path.open("wb") as output:
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            output.write(chunk)


def _update_job(job_id: str, **changes: object) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return
        job.update(changes)
        job["updated_at"] = _utc_now()
        status_path = JOBS_DIR / job_id / "status.json"
        status_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_job(job_id: str, job: dict[str, object]) -> None:
    with jobs_lock:
        jobs[job_id] = job
        status_path = JOBS_DIR / job_id / "status.json"
        status_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_job_or_404(job_id: str) -> dict[str, object]:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is not None:
            return dict(job)

    status_path = JOBS_DIR / job_id / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")

    try:
        job = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="작업 상태 파일을 읽을 수 없습니다.") from exc

    with jobs_lock:
        jobs[job_id] = job
    return dict(job)


def _public_job(job: dict[str, object]) -> dict[str, object]:
    return {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "option": job["option"],
        "source_name": job["source_name"],
        "result_name": job["result_name"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_error_message(exc: Exception) -> str:
    message = str(exc)
    if "Superresolution processing failed" in message or "Real-ESRGAN" in message:
        return f"초해상도 처리 중 오류가 발생했습니다. {message}"
    if "Stabilization + upright processing failed" in message:
        return f"흔들림 보정 + 수평 보정 처리 중 오류가 발생했습니다. {message}"
    if "Stabilization processing failed" in message:
        return f"흔들림 보정 처리 중 오류가 발생했습니다. {message}"
    if isinstance(exc, VideoConversionError):
        return f"영상 변환 중 오류가 발생했습니다. {message}"
    return f"영상 처리 중 예상치 못한 오류가 발생했습니다. {message}"


def _validate_video_file(file_name: str | None) -> str:
    if not file_name:
        raise HTTPException(status_code=400, detail="파일 이름이 없습니다.")

    suffix = Path(file_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"{allowed} 파일만 지원합니다.")

    return Path(file_name).name
