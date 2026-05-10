from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.video_processor import (
    VideoConversionError,
    VideoProcessingError,
    make_browser_playable,
    process_video,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

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


def _validate_video_file(file_name: str | None) -> str:
    if not file_name:
        raise HTTPException(status_code=400, detail="파일 이름이 없습니다.")

    suffix = Path(file_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"{allowed} 파일만 지원합니다.")

    return Path(file_name).name
