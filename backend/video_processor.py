import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile, which
from typing import Callable

from backend.pipelines.stabilization import run_stabilization
from backend.pipelines.superresolution import run_superresolution
from backend.pipelines.upright_stabilization import run_upright_adjustment, run_upright_stabilization


@dataclass(frozen=True)
class ProcessingResult:
    video_bytes: bytes
    option: str
    file_name: str


class VideoConversionError(RuntimeError):
    pass


class VideoProcessingError(RuntimeError):
    pass


ProgressCallback = Callable[[int, str], None]
CancelCallback = Callable[[], bool]
FFMPEG_FALLBACK = Path(r"C:\ffmpeg_SR\bin\ffmpeg.exe")
FFPROBE_FALLBACK = Path(r"C:\ffmpeg_SR\bin\ffprobe.exe")


def make_browser_playable(video_bytes: bytes, file_name: str) -> bytes:
    """Return bytes that browser video players can preview.

    Browsers generally do not preview AVI reliably, so AVI input is converted to
    MP4 with ffmpeg. MP4/MOV bytes are returned unchanged.
    """
    def report(_progress: int, _message: str) -> None:
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        suffix = Path(file_name).suffix.lower() or ".mp4"
        input_path = temp_path / f"input{suffix}"
        output_path = temp_path / "preview.mp4"
        input_path.write_bytes(video_bytes)

        if _is_browser_playable_mp4(input_path):
            return video_bytes
        report(96, "브라우저에서 재생할 수 있는 영상으로 정리하고 있습니다.")

        try:
            subprocess.run(
                [
                    str(_ffmpeg_executable()),
                    "-y",
                    "-i",
                    str(input_path),
                    "-vcodec",
                    "libx264",
                    "-acodec",
                    "aac",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise VideoConversionError(
                "AVI preview conversion failed. Upload MP4/MOV, or allow ffmpeg in Windows security policy."
            ) from exc
        return output_path.read_bytes()


def _is_browser_playable_mp4(path: Path) -> bool:
    if path.suffix.lower() != ".mp4":
        return False

    try:
        probe = subprocess.run(
            [
                str(_ffprobe_executable()),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,codec_tag_string,pix_fmt",
                "-of",
                "default=noprint_wrappers=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False

    stream_info = probe.stdout.lower()
    return (
        "codec_name=h264" in stream_info
        and "codec_tag_string=avc1" in stream_info
        and "pix_fmt=yuv420p" in stream_info
    )


def process_video(
    video_bytes: bytes,
    option: str,
    file_name: str,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> ProcessingResult:
    """Process a video and return the restored bytes.

    This is the backend boundary for the app. Replace the body with the real
    stabilization/restoration pipeline when the model code is ready.
    """
    output_name = str(Path(file_name).with_suffix(".mp4"))
    features = set(option.split(" + "))
    last_reported_progress = -1

    def report(progress: int, message: str) -> None:
        nonlocal last_reported_progress
        _raise_if_cancelled(cancel_callback)
        progress = max(0, min(int(progress), 100))
        if progress < last_reported_progress:
            progress = last_reported_progress
        if progress == last_reported_progress:
            return
        last_reported_progress = progress
        if progress_callback is not None:
            progress_callback(progress, message)

    def stage_reporter(start: int, end: int) -> ProgressCallback:
        def mapped(stage_progress: int, message: str) -> None:
            bounded = max(0, min(int(stage_progress), 100))
            progress = start + round((end - start) * (bounded / 100))
            report(progress, message)

        return mapped

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        suffix = Path(file_name).suffix.lower() or ".mp4"
        current_path = temp_path / f"input{suffix}"
        current_path.write_bytes(video_bytes)

        has_motion_stage = "Upright Correction" in features or "Stabilization" in features
        has_sr_stage = "Superresolution" in features
        if has_motion_stage and has_sr_stage:
            motion_range = (8, 58)
            sr_range = (60, 92)
        elif has_motion_stage:
            motion_range = (8, 92)
            sr_range = (0, 0)
        else:
            motion_range = (0, 0)
            sr_range = (8, 92)

        if "Upright Correction" in features:
            upright_output_path = temp_path / "upright_output.mp4"
            try:
                report(motion_range[0], "수평/흔들림 보정을 시작했습니다.")
                if "Stabilization" in features:
                    run_upright_stabilization(
                        str(current_path),
                        str(upright_output_path),
                        progress_callback=stage_reporter(*motion_range),
                        cancel_callback=cancel_callback,
                    )
                else:
                    run_upright_adjustment(
                        str(current_path),
                        str(upright_output_path),
                        progress_callback=stage_reporter(*motion_range),
                        cancel_callback=cancel_callback,
                    )
            except Exception as exc:
                raise VideoProcessingError(f"Stabilization + upright processing failed: {exc}") from exc
            current_path = upright_output_path
            report(motion_range[1], "수평/흔들림 보정이 완료되었습니다.")
            features.discard("Upright Correction")
            features.discard("Stabilization")
        elif "Stabilization" in features:
            stabilized_output_path = temp_path / "stabilized_output.mp4"
            try:
                report(motion_range[0], "흔들림 보정을 시작했습니다.")
                run_stabilization(
                    str(current_path),
                    str(stabilized_output_path),
                    progress_callback=stage_reporter(*motion_range),
                    cancel_callback=cancel_callback,
                )
            except Exception as exc:
                raise VideoProcessingError(f"Stabilization processing failed: {exc}") from exc
            current_path = stabilized_output_path
            report(motion_range[1], "흔들림 보정이 완료되었습니다.")
            features.discard("Stabilization")

        if "Superresolution" in features:
            superresolution_output_path = temp_path / "superresolution_output.mp4"
            try:
                report(sr_range[0], "초해상도 처리를 시작했습니다.")
                run_superresolution(
                    str(current_path),
                    str(superresolution_output_path),
                    cancel_callback=cancel_callback,
                )
            except Exception as exc:
                raise VideoProcessingError(f"Superresolution processing failed: {exc}") from exc
            current_path = superresolution_output_path
            report(sr_range[1], "초해상도 처리가 완료되었습니다.")
            features.discard("Superresolution")

        try:
            report(96, "브라우저 재생용 영상으로 정리하고 있습니다.")
            current_path = _mux_original_audio(current_path, temp_path / f"input{suffix}", temp_path / "with_audio.mp4")
            playable_bytes = make_browser_playable(current_path.read_bytes(), current_path.name)
        except VideoConversionError:
            playable_bytes = current_path.read_bytes()

    return ProcessingResult(video_bytes=playable_bytes, option=option, file_name=output_name)


def process_video_file(
    input_path: str | Path,
    option: str,
    file_name: str,
    output_path: str | Path,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> ProcessingResult:
    """Process a video from disk and write the restored video to disk."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_name = str(Path(file_name).with_suffix(".mp4"))
    features = set(option.split(" + "))
    last_reported_progress = -1

    def report(progress: int, message: str) -> None:
        nonlocal last_reported_progress
        _raise_if_cancelled(cancel_callback)
        progress = max(0, min(int(progress), 100))
        if progress < last_reported_progress:
            progress = last_reported_progress
        if progress == last_reported_progress:
            return
        last_reported_progress = progress
        if progress_callback is not None:
            progress_callback(progress, message)

    def stage_reporter(start: int, end: int) -> ProgressCallback:
        def mapped(stage_progress: int, message: str) -> None:
            bounded = max(0, min(int(stage_progress), 100))
            progress = start + round((end - start) * (bounded / 100))
            report(progress, message)

        return mapped

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        current_path = input_path
        _raise_if_cancelled(cancel_callback)

        has_motion_stage = "Upright Correction" in features or "Stabilization" in features
        has_sr_stage = "Superresolution" in features
        if has_motion_stage and has_sr_stage:
            motion_range = (8, 58)
            sr_range = (60, 92)
        elif has_motion_stage:
            motion_range = (8, 92)
            sr_range = (0, 0)
        else:
            motion_range = (0, 0)
            sr_range = (8, 92)

        if "Upright Correction" in features:
            upright_output_path = temp_path / "upright_output.mp4"
            try:
                report(motion_range[0], "Starting upright/stabilization correction.")
                if "Stabilization" in features:
                    run_upright_stabilization(
                        str(current_path),
                        str(upright_output_path),
                        progress_callback=stage_reporter(*motion_range),
                    )
                else:
                    run_upright_adjustment(
                        str(current_path),
                        str(upright_output_path),
                        progress_callback=stage_reporter(*motion_range),
                    )
            except Exception as exc:
                raise VideoProcessingError(f"Stabilization + upright processing failed: {exc}") from exc
            current_path = upright_output_path
            report(motion_range[1], "Upright/stabilization correction completed.")
            features.discard("Upright Correction")
            features.discard("Stabilization")
        elif "Stabilization" in features:
            stabilized_output_path = temp_path / "stabilized_output.mp4"
            try:
                report(motion_range[0], "Starting stabilization correction.")
                run_stabilization(
                    str(current_path),
                    str(stabilized_output_path),
                    progress_callback=stage_reporter(*motion_range),
                )
            except Exception as exc:
                raise VideoProcessingError(f"Stabilization processing failed: {exc}") from exc
            current_path = stabilized_output_path
            report(motion_range[1], "Stabilization correction completed.")
            features.discard("Stabilization")

        if "Superresolution" in features:
            superresolution_output_path = temp_path / "superresolution_output.mp4"
            try:
                report(sr_range[0], "Starting superresolution processing.")
                run_superresolution(
                    str(current_path),
                    str(superresolution_output_path),
                    cancel_callback=cancel_callback,
                )
            except Exception as exc:
                raise VideoProcessingError(f"Superresolution processing failed: {exc}") from exc
            current_path = superresolution_output_path
            report(sr_range[1], "Superresolution processing completed.")
            features.discard("Superresolution")

        try:
            report(96, "Preparing browser-playable video.")
            current_path = _mux_original_audio(
                current_path,
                input_path,
                temp_path / "with_audio.mp4",
                cancel_callback=cancel_callback,
            )
            _write_browser_playable_file(current_path, output_path, cancel_callback=cancel_callback)
        except VideoConversionError:
            _raise_if_cancelled(cancel_callback)
            copyfile(current_path, output_path)

    return ProcessingResult(video_bytes=b"", option=option, file_name=output_name)


def _write_browser_playable_file(
    input_path: Path,
    output_path: Path,
    cancel_callback: CancelCallback | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _raise_if_cancelled(cancel_callback)
    if _is_browser_playable_mp4(input_path):
        copyfile(input_path, output_path)
        return

    try:
        _run_cancellable_subprocess(
            [
                str(_ffmpeg_executable()),
                "-y",
                "-i",
                str(input_path),
                "-vcodec",
                "libx264",
                "-acodec",
                "aac",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            cancel_callback=cancel_callback,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise VideoConversionError("Browser-playable conversion failed.") from exc


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


def _has_audio_stream(path: Path) -> bool:
    try:
        probe = subprocess.run(
            [
                str(_ffprobe_executable()),
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False

    return "audio" in probe.stdout.lower()


def _mux_original_audio(
    processed_path: Path,
    original_path: Path,
    output_path: Path,
    cancel_callback: CancelCallback | None = None,
) -> Path:
    _raise_if_cancelled(cancel_callback)
    if not _has_audio_stream(original_path):
        return processed_path

    try:
        _run_cancellable_subprocess(
            [
                str(_ffmpeg_executable()),
                "-y",
                "-i",
                str(processed_path),
                "-i",
                str(original_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            cancel_callback=cancel_callback,
        )
    except (OSError, subprocess.CalledProcessError):
        return processed_path

    return output_path if output_path.exists() else processed_path


def _run_cancellable_subprocess(
    command: list[str],
    cancel_callback: CancelCallback | None = None,
) -> None:
    if cancel_callback is None:
        subprocess.run(command, check=True, capture_output=True)
        return

    import time

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    try:
        while process.poll() is None:
            if cancel_callback():
                _terminate_process_tree(process)
                raise RuntimeError("JobCancelled")
            time.sleep(0.5)
    finally:
        if process.poll() is None:
            _terminate_process_tree(process)

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


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


def _raise_if_cancelled(cancel_callback: CancelCallback | None) -> None:
    if cancel_callback is not None and cancel_callback():
        raise RuntimeError("JobCancelled")
