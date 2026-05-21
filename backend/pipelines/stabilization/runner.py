import sys
from pathlib import Path
from typing import Callable


PIPELINE_DIR = Path(__file__).resolve().parent
LOCAL_MODULES = {
    "Asap",
    "getPath",
    "new_warping",
    "smoothPath",
    "stabilizer",
    "steady_view_main",
    "tracker",
}


def _drop_foreign_modules() -> None:
    pipeline_path = PIPELINE_DIR.resolve()
    for module_name in LOCAL_MODULES:
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            sys.modules.pop(module_name, None)
            continue

        try:
            Path(module_file).resolve().relative_to(pipeline_path)
        except ValueError:
            sys.modules.pop(module_name, None)


def run_stabilization(
    input_path: str,
    output_path: str,
    progress_callback=None,
    cancel_callback: Callable[[], bool] | None = None,
) -> None:
    """Run the stabilization-only pipeline on a video file."""
    def check_cancel() -> None:
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("JobCancelled")

    def report(progress: int, message: str) -> None:
        check_cancel()
        if progress_callback is not None:
            progress_callback(progress, message)

    def map_progress(start: int, end: int):
        def mapped(percent: int, message: str) -> None:
            bounded = max(0, min(int(percent), 100))
            report(start + round((end - start) * (bounded / 100)), message)

        return mapped

    _drop_foreign_modules()
    check_cancel()

    pipeline_path = str(PIPELINE_DIR)
    if pipeline_path in sys.path:
        sys.path.remove(pipeline_path)
    sys.path.insert(0, pipeline_path)

    from getPath import get_path
    from smoothPath import smooth_path
    from stabilizer import render_combined_video
    from steady_view_main import get_actual_frame_count
    from tracker import get_tracks

    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = "cpu"

    print(f"Steady View pipeline started. (Stabilization Only, device={device})")
    report(1, "흔들림 보정을 준비하고 있습니다.")

    n_frames = get_actual_frame_count(input_path)
    mesh_size = 16
    demand = 1024

    matched_pairs, info = get_tracks(
        input_path,
        mesh_size,
        demand,
        n_frames,
        progress_callback=map_progress(5, 55),
        cancel_callback=cancel_callback,
    )
    if not matched_pairs:
        raise RuntimeError("Feature tracking failed.")

    actual_n = len(matched_pairs)
    if actual_n != n_frames - 1:
        n_frames = actual_n + 1

    scale_inv = 1.0 / info["scale"]
    rescaled_pairs = [
        (pa.astype("float32") * scale_inv, pb.astype("float32") * scale_inv)
        for pa, pb in matched_pairs
    ]

    sigma = min(16, n_frames // 5)
    report(62, "카메라 경로를 계산하고 있습니다.")
    camera_path = get_path(mesh_size, rescaled_pairs, info)
    check_cancel()
    report(70, "흔들림 경로를 부드럽게 보정하고 있습니다.")
    smoothed = smooth_path(camera_path, sigma=sigma)
    check_cancel()

    render_combined_video(
        input_path,
        output_path,
        camera_path,
        smoothed,
        info,
        None,
        device,
        progress_callback=map_progress(75, 100),
        cancel_callback=cancel_callback,
    )

    report(100, "흔들림 보정이 완료되었습니다.")
    print("Steady View pipeline finished. (Stabilization Only)")
