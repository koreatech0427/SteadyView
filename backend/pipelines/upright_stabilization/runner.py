import os
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PIPELINE_DIR / "models" / "best_model_eff_b0_bright_global7_residual_fusion_real_best.pth"
LOCAL_MODULES = {
    "Asap",
    "analysis",
    "config",
    "getPath",
    "joint_path",
    "main",
    "new_warping",
    "render",
    "smoothPath",
    "upright_model",
}


def run_upright_stabilization(input_path: str, output_path: str) -> None:
    """Run the Stabilization + Upright pipeline on a video file."""
    model_path = Path(os.environ.get("STEADYVIEW_UPRIGHT_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.exists():
        raise FileNotFoundError(
            "Upright model file was not found. Put the .pth file at "
            f"{DEFAULT_MODEL_PATH} or set STEADYVIEW_UPRIGHT_MODEL_PATH."
        )

    for module_name in LOCAL_MODULES:
        sys.modules.pop(module_name, None)

    pipeline_path = str(PIPELINE_DIR)
    if pipeline_path in sys.path:
        sys.path.remove(pipeline_path)
    sys.path.insert(0, pipeline_path)

    from main import run_pipeline

    run_pipeline(video_path=input_path, model_path=str(model_path), output_path=output_path)
