"""Type checking + runtime smoke tests for gamma module."""

import numpy as np
import numpy.typing as npt
from pathlib import Path

import gamma


MODELS_DIR = Path(__file__).resolve().parents[3] / "tests" / "models"
SIMPLE_MLP = str(MODELS_DIR / "simple_mlp.onnx")


def test_diff_basic() -> None:
    """Test diff function type hints."""
    result: gamma.DiffResult = gamma.diff(SIMPLE_MLP, SIMPLE_MLP)
    _: bool = result.is_equivalent
    _2: float = result.max_divergence
    _3: list[gamma.LayerComparison] = result.layers


def test_diff_with_options() -> None:
    """Test diff with optional parameters."""
    input_data: npt.NDArray[np.float32] = np.zeros((1, 2), dtype=np.float32)
    mapping: dict[str, str] = {"output": "output"}

    result: gamma.DiffResult = gamma.diff(
        SIMPLE_MLP,
        SIMPLE_MLP,
        tolerance=1e-4,
        input=input_data,
        continue_after_divergence=False,
        layer_mapping=mapping,
    )

    _: str = result.summary()
    _2: list[gamma.DiffStatus] = result.statuses()


def test_layer_comparison() -> None:
    """Test LayerComparison type hints."""
    result = gamma.diff(SIMPLE_MLP, SIMPLE_MLP)
    if result.layers:
        layer: gamma.LayerComparison = result.layers[0]
        _: str = layer.name
        _2: float = layer.max_diff
        _3: bool = layer.exceeds_tolerance
        _4: list[int] = layer.shape_a


def test_run_with_intermediates() -> None:
    """Test run_with_intermediates type hints."""
    input_data: npt.NDArray[np.float32] = np.zeros((1, 2), dtype=np.float32)
    outputs: dict[str, npt.NDArray[np.float32]] = gamma.run_with_intermediates(
        SIMPLE_MLP, input_data
    )
    for name, arr in outputs.items():
        _: str = name
        _2: npt.NDArray[np.float32] = arr


def test_load_model_info() -> None:
    """Test load_model_info type hints."""
    info: dict[str, object] = gamma.load_model_info(SIMPLE_MLP)
    _ = info["layer_count"]
    _ = info["layer_names"]


def test_load_npy(tmp_path: Path) -> None:
    """Test load_npy type hints."""
    npy_path = tmp_path / "data.npy"
    np.save(str(npy_path), np.zeros((2, 3), dtype=np.float32))
    data: npt.NDArray[np.float32] = gamma.load_npy(str(npy_path))
    _ = data.shape


def test_diff_status_enum() -> None:
    """Test DiffStatus enum type hints."""
    _: gamma.DiffStatus = gamma.DiffStatus.Ok
    _2: gamma.DiffStatus = gamma.DiffStatus.DriftStarts
    _3: gamma.DiffStatus = gamma.DiffStatus.ExceedsTolerance
    _4: gamma.DiffStatus = gamma.DiffStatus.ShapeMismatch


def test_version() -> None:
    """Test __version__ type hint."""
    version: str = gamma.__version__
    _ = version
