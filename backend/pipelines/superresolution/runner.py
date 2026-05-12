import os
import subprocess
import sys
from pathlib import Path


DEFAULT_REAL_ESRGAN_DIR = Path(r"C:\Users\korea\Desktop\Real-ESRGAN\Real-ESRGAN_V2")


def run_superresolution(input_path: str, output_path: str) -> None:
    """Run Real-ESRGAN video superresolution on a video file."""
    real_esrgan_dir = Path(os.environ.get("STEADYVIEW_REAL_ESRGAN_DIR", DEFAULT_REAL_ESRGAN_DIR))
    script_path = real_esrgan_dir / "inference_realesrgan_video.py"
    if not script_path.exists():
        raise FileNotFoundError(
            "Real-ESRGAN video script was not found. Put Real-ESRGAN at "
            f"{DEFAULT_REAL_ESRGAN_DIR} or set STEADYVIEW_REAL_ESRGAN_DIR."
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

    generated_path.replace(target_path)
