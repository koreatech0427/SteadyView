import os
import sys
from pathlib import Path
from typing import Callable


PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PIPELINE_DIR / "models" / "best_stage2_true_hybrid_real_best.pth"
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
    "upright_adjustment",
    "upright_model",
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


def run_upright_stabilization(
    input_path: str,
    output_path: str,
    progress_callback=None,
    cancel_callback: Callable[[], bool] | None = None,
) -> None:
    """Run the Stabilization + Upright pipeline on a video file."""
    model_path = Path(os.environ.get("STEADYVIEW_UPRIGHT_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.exists():
        raise FileNotFoundError(
            "Upright model file was not found. Put the .pth file at "
            f"{DEFAULT_MODEL_PATH} or set STEADYVIEW_UPRIGHT_MODEL_PATH."
        )

    _drop_foreign_modules()

    pipeline_path = str(PIPELINE_DIR)
    if pipeline_path in sys.path:
        sys.path.remove(pipeline_path)
    sys.path.insert(0, pipeline_path)

    from main import run_pipeline

    run_pipeline(
        video_path=input_path,
        model_path=str(model_path),
        output_path=output_path,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )


def run_upright_adjustment(
    input_path: str,
    output_path: str,
    progress_callback=None,
    cancel_callback: Callable[[], bool] | None = None,
) -> None:
    """Run upright-only correction using optical-flow temporal smoothing."""
    model_path = Path(os.environ.get("STEADYVIEW_UPRIGHT_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.exists():
        raise FileNotFoundError(
            "Upright model file was not found. Put the .pth file at "
            f"{DEFAULT_MODEL_PATH} or set STEADYVIEW_UPRIGHT_MODEL_PATH."
        )

    _drop_foreign_modules()

    pipeline_path = str(PIPELINE_DIR)
    if pipeline_path in sys.path:
        sys.path.remove(pipeline_path)
    sys.path.insert(0, pipeline_path)

    from upright_adjustment import run_upright_adjustment as run_pipeline

    run_pipeline(
        video_path=input_path,
        model_path=str(model_path),
        output_path=output_path,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )
