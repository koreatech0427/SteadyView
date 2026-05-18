import os
import subprocess
import sys
from pathlib import Path
from shutil import which


DEFAULT_REAL_ESRGAN_DIRS = (
    Path("/opt/real-esrgan"),
    Path(r"C:\Users\korea\Desktop\Real-ESRGAN\Real-ESRGAN_V2"),
)
FFMPEG_FALLBACK = Path(r"C:\ffmpeg_SR\bin\ffmpeg.exe")
FFPROBE_FALLBACK = Path(r"C:\ffmpeg_SR\bin\ffprobe.exe")


def _default_real_esrgan_dir() -> Path:
    for candidate in DEFAULT_REAL_ESRGAN_DIRS:
        if (candidate / "inference_realesrgan_video.py").exists():
            return candidate
    return DEFAULT_REAL_ESRGAN_DIRS[-1]


def run_superresolution(input_path: str, output_path: str) -> None:
    """Run Real-ESRGAN video superresolution on a video file."""
    real_esrgan_dir = Path(os.environ.get("STEADYVIEW_REAL_ESRGAN_DIR", _default_real_esrgan_dir()))
    script_path = real_esrgan_dir / "inference_realesrgan_video.py"
    if not script_path.exists():
        raise FileNotFoundError(
            "Real-ESRGAN video script was not found. Mount it at /opt/real-esrgan, put it at "
            f"{DEFAULT_REAL_ESRGAN_DIRS[-1]}, or set STEADYVIEW_REAL_ESRGAN_DIR."
        )

    target_path = Path(output_path)
    output_dir = target_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    input_stem = Path(input_path).stem
    suffix = "steadyview_sr"
    generated_path = output_dir / f"{input_stem}_{suffix}.mp4"

    command = [
        sys.executable,
        str(script_path),
        "-i",
        str(input_path),
        "-o",
        str(output_dir),
        "-n",
        os.environ.get("STEADYVIEW_SR_MODEL", "realesr-general-x4v3"),
        "-s",
        os.environ.get("STEADYVIEW_SR_OUTSCALE", "4"),
        "-t",
        os.environ.get("STEADYVIEW_SR_TILE", "0"),
        "--suffix",
        suffix,
        "--crf",
        os.environ.get("STEADYVIEW_SR_CRF", "16"),
        "--preset",
        os.environ.get("STEADYVIEW_SR_PRESET", "slow"),
        "--no_compare",
    ]

    try:
        subprocess.run(command, cwd=real_esrgan_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        message = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            details = (exc.stderr or exc.stdout or "").strip()
            if details:
                message = details[-2000:]
        raise RuntimeError(f"Real-ESRGAN superresolution failed: {message}") from exc

    if not generated_path.exists():
        raise FileNotFoundError(f"Real-ESRGAN output was not created: {generated_path}")

    original_width, original_height = _probe_video_size(Path(input_path))
    _resize_to_original_resolution(generated_path, target_path, original_width, original_height)


def _ffmpeg_executable() -> Path:
    executable = which("ffmpeg")
    if executable:
        return Path(executable)
    if FFMPEG_FALLBACK.exists():
        return FFMPEG_FALLBACK
    return Path("ffmpeg")


def _ffprobe_executable() -> Path:
    executable = which("ffprobe")
    if executable:
        return Path(executable)
    if FFPROBE_FALLBACK.exists():
        return FFPROBE_FALLBACK
    return Path("ffprobe")


def _probe_video_size(path: Path) -> tuple[int, int]:
    try:
        probe = subprocess.run(
            [
                str(_ffprobe_executable()),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not read original video resolution: {exc}") from exc

    size = probe.stdout.strip()
    try:
        width_text, height_text = size.split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise RuntimeError(f"Unexpected ffprobe resolution output: {size}") from exc

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid original video resolution: {width}x{height}")

    return width, height


def _resize_to_original_resolution(
    generated_path: Path,
    target_path: Path,
    width: int,
    height: int,
) -> None:
    try:
        subprocess.run(
            [
                str(_ffmpeg_executable()),
                "-y",
                "-i",
                str(generated_path),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-vf",
                f"scale={width}:{height}:flags=lanczos",
                "-c:v",
                "libx264",
                "-crf",
                os.environ.get("STEADYVIEW_SR_FINAL_CRF", os.environ.get("STEADYVIEW_SR_CRF", "16")),
                "-preset",
                os.environ.get("STEADYVIEW_SR_FINAL_PRESET", os.environ.get("STEADYVIEW_SR_PRESET", "slow")),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(target_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        message = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            details = (exc.stderr or exc.stdout or "").strip()
            if details:
                message = details[-2000:]
        raise RuntimeError(f"Could not resize superresolution output to original resolution: {message}") from exc

    if not target_path.exists():
        raise FileNotFoundError(f"Final superresolution output was not created: {target_path}")

    generated_path.unlink(missing_ok=True)
