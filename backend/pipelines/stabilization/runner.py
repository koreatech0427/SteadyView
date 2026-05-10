import sys
from pathlib import Path


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


def run_stabilization(input_path: str, output_path: str) -> None:
    """Run the stabilization-only pipeline on a video file."""
    for module_name in LOCAL_MODULES:
        sys.modules.pop(module_name, None)

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

    n_frames = get_actual_frame_count(input_path)
    mesh_size = 16
    demand = 1024

    matched_pairs, info = get_tracks(input_path, mesh_size, demand, n_frames)
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
    camera_path = get_path(mesh_size, rescaled_pairs, info)
    smoothed = smooth_path(camera_path, sigma=sigma)

    render_combined_video(
        input_path,
        output_path,
        camera_path,
        smoothed,
        info,
        None,
        device,
    )

    print("Steady View pipeline finished. (Stabilization Only)")
