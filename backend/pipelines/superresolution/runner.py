import os
import subprocess
import tempfile
import time
from pathlib import Path
from shutil import which
from typing import Callable


DEFAULT_BASICVSRPP_DIRS = (
    Path(r"C:\Users\korea\Desktop\BasicVSR++"),
    Path("/opt/basicvsrpp"),
)
DEFAULT_BASICVSRPP_VENV_PYTHON = Path(r"C:\Users\korea\Desktop\BasicVSR++\.venv310\Scripts\python.exe")
FFMPEG_FALLBACK = Path(r"C:\ffmpeg_SR\bin\ffmpeg.exe")
FFPROBE_FALLBACK = Path(r"C:\ffmpeg_SR\bin\ffprobe.exe")


def _default_basicvsrpp_dir() -> Path:
    for candidate in DEFAULT_BASICVSRPP_DIRS:
        if (candidate / "run_basicvsrpp_chunk.py").exists():
            return candidate
    return DEFAULT_BASICVSRPP_DIRS[0]


def run_superresolution(
    input_path: str,
    output_path: str,
    cancel_callback: Callable[[], bool] | None = None,
) -> None:
    """Run BasicVSR++ video restoration on a video file."""
    basicvsrpp_dir = Path(os.environ.get("STEADYVIEW_BASICVSRPP_DIR", _default_basicvsrpp_dir()))
    script_path = basicvsrpp_dir / "run_basicvsrpp_chunk.py"
    if not script_path.exists():
        raise FileNotFoundError(
            "BasicVSR++ script was not found. Put it at "
            f"{DEFAULT_BASICVSRPP_DIRS[0]}, mount it at /opt/basicvsrpp, or set STEADYVIEW_BASICVSRPP_DIR."
        )

    python_path = _basicvsrpp_python(basicvsrpp_dir)
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(python_path),
        str(script_path),
        "--input",
        str(input_path),
        "--output",
        str(target_path),
        "--chunk-size",
        os.environ.get("STEADYVIEW_BASICVSRPP_CHUNK_SIZE", "8"),
        "--model-name",
        os.environ.get("STEADYVIEW_BASICVSRPP_MODEL", "real_basicvsr"),
    ]

    _run_cancellable_command(command, basicvsrpp_dir, cancel_callback)

    if not target_path.exists():
        raise FileNotFoundError(f"BasicVSR++ output was not created: {target_path}")


def _run_cancellable_command(
    command: list[str],
    cwd: Path,
    cancel_callback: Callable[[], bool] | None,
) -> None:
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as output:
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise RuntimeError(str(exc)) from exc

        try:
            while process.poll() is None:
                if cancel_callback is not None and cancel_callback():
                    _terminate_process_tree(process)
                    raise RuntimeError("JobCancelled")
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                _terminate_process_tree(process)

        if process.returncode != 0:
            output.seek(0)
            details = output.read().strip()
            message = details[-2000:] if details else f"exit code {process.returncode}"
            raise RuntimeError(message)


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def _basicvsrpp_python(basicvsrpp_dir: Path) -> Path:
    configured = os.environ.get("STEADYVIEW_BASICVSRPP_PYTHON")
    if configured:
        return Path(configured)

    local_venv = basicvsrpp_dir / ".venv310" / "Scripts" / "python.exe"
    if local_venv.exists():
        return local_venv

    if DEFAULT_BASICVSRPP_VENV_PYTHON.exists():
        return DEFAULT_BASICVSRPP_VENV_PYTHON

    executable = which("python")
    if executable:
        return Path(executable)

    return Path("python")


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
