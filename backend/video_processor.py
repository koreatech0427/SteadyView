import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from backend.pipelines.stabilization import run_stabilization
from backend.pipelines.upright_stabilization import run_upright_stabilization


@dataclass(frozen=True)
class ProcessingResult:
    video_bytes: bytes
    option: str
    file_name: str


class VideoConversionError(RuntimeError):
    pass


class VideoProcessingError(RuntimeError):
    pass


def make_browser_playable(video_bytes: bytes, file_name: str) -> bytes:
    """Return bytes that browser video players can preview.

    Browsers generally do not preview AVI reliably, so AVI input is converted to
    MP4 with ffmpeg. MP4/MOV bytes are returned unchanged.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        suffix = Path(file_name).suffix.lower() or ".mp4"
        input_path = temp_path / f"input{suffix}"
        output_path = temp_path / "preview.mp4"
        input_path.write_bytes(video_bytes)

        if _is_browser_playable_mp4(input_path):
            return video_bytes

        try:
            subprocess.run(
                [
                    "ffmpeg",
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
                "ffprobe",
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


def process_video(video_bytes: bytes, option: str, file_name: str) -> ProcessingResult:
    """Process a video and return the restored bytes.

    This is the backend boundary for the app. Replace the body with the real
    stabilization/restoration pipeline when the model code is ready.
    """
    output_name = str(Path(file_name).with_suffix(".mp4"))
    features = set(option.split(" + "))

    if "Upright Correction" in features:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            suffix = Path(file_name).suffix.lower() or ".mp4"
            input_path = temp_path / f"input{suffix}"
            raw_output_path = temp_path / "upright_output.mp4"
            input_path.write_bytes(video_bytes)

            try:
                run_upright_stabilization(str(input_path), str(raw_output_path))
            except Exception as exc:
                raise VideoProcessingError(f"Stabilization + upright processing failed: {exc}") from exc

            try:
                playable_bytes = make_browser_playable(raw_output_path.read_bytes(), raw_output_path.name)
            except VideoConversionError:
                playable_bytes = raw_output_path.read_bytes()

        return ProcessingResult(video_bytes=playable_bytes, option=option, file_name=output_name)

    if "Stabilization" in features:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            suffix = Path(file_name).suffix.lower() or ".mp4"
            input_path = temp_path / f"input{suffix}"
            raw_output_path = temp_path / "stabilized_output.mp4"
            input_path.write_bytes(video_bytes)

            try:
                run_stabilization(str(input_path), str(raw_output_path))
            except Exception as exc:
                raise VideoProcessingError(f"Stabilization processing failed: {exc}") from exc

            try:
                playable_bytes = make_browser_playable(raw_output_path.read_bytes(), raw_output_path.name)
            except VideoConversionError:
                playable_bytes = raw_output_path.read_bytes()

        return ProcessingResult(video_bytes=playable_bytes, option=option, file_name=output_name)

    playable_bytes = make_browser_playable(video_bytes, file_name)
    return ProcessingResult(video_bytes=playable_bytes, option=option, file_name=output_name)
