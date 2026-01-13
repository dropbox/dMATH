"""
Type stubs for γ-CROWN Python bindings.

This module provides neural network verification and testing tools,
specifically designed for pytest integration and model porting workflows.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt

__version__: str

class DiffStatus(Enum):
    """Status of a layer comparison in a model diff."""
    Ok = ...
    DriftStarts = ...
    ExceedsTolerance = ...
    ShapeMismatch = ...

class LayerComparison:
    """Result of comparing a single layer between two models.

    Attributes:
        name: Layer name from model A
        name_b: Layer name from model B (if different mapping)
        max_diff: Maximum absolute difference between outputs
        mean_diff: Mean absolute difference between outputs
        exceeds_tolerance: Whether max_diff exceeds the tolerance threshold
        shape_a: Output shape from model A
        shape_b: Output shape from model B
    """
    name: str
    name_b: Optional[str]
    max_diff: float
    mean_diff: float
    exceeds_tolerance: bool
    shape_a: List[int]
    shape_b: List[int]

class DiffResult:
    """Result of a full model diff operation.

    Attributes:
        layers: List of per-layer comparison results
        first_bad_layer: Index of first layer exceeding tolerance (None if all OK)
        drift_start_layer: Index where numerical drift begins (None if none)
        max_divergence: Maximum difference found across all layers
        tolerance: Tolerance threshold used for comparison
        suggestion: Suggested fix if divergence found

    Properties:
        is_equivalent: True if models match within tolerance
        first_bad_layer_name: Name of first bad layer (None if all OK)
    """
    layers: List[LayerComparison]
    first_bad_layer: Optional[int]
    drift_start_layer: Optional[int]
    max_divergence: float
    tolerance: float
    suggestion: Optional[str]

    @property
    def is_equivalent(self) -> bool:
        """Check if models are equivalent within tolerance."""
        ...

    @property
    def first_bad_layer_name(self) -> Optional[str]:
        """Get the name of the first bad layer, if any."""
        ...

    def statuses(self) -> List[DiffStatus]:
        """Get status for each layer."""
        ...

    def summary(self) -> str:
        """Get a formatted summary table (like CLI output)."""
        ...

def diff(
    model_a: str,
    model_b: str,
    tolerance: float = 1e-5,
    input: Optional[npt.NDArray[np.float32]] = None,
    continue_after_divergence: bool = True,
    layer_mapping: Optional[Dict[str, str]] = None,
) -> DiffResult:
    """Compare two ONNX models layer-by-layer to find divergence.

    This is the main entry point for model comparison. It runs inference on both
    models with the same input and compares intermediate outputs at each layer.

    Args:
        model_a: Path to first ONNX model
        model_b: Path to second ONNX model
        tolerance: Maximum allowed difference (default: 1e-5)
        input: Optional numpy array for input (default: zeros)
        continue_after_divergence: Whether to continue after finding divergence
        layer_mapping: Optional dict mapping layer names from A to B

    Returns:
        DiffResult with comparison results

    Raises:
        ValueError: If models cannot be loaded or compared

    Example:
        >>> diff_result = gamma.diff("model_a.onnx", "model_b.onnx")
        >>> assert diff_result.is_equivalent, f"Diverges at {diff_result.first_bad_layer_name}"

        >>> # With custom tolerance
        >>> diff_result = gamma.diff("model_a.onnx", "model_b.onnx", tolerance=1e-4)

        >>> # With custom input
        >>> import numpy as np
        >>> input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        >>> diff_result = gamma.diff("model_a.onnx", "model_b.onnx", input=input_data)
    """
    ...

def run_with_intermediates(
    model_path: str,
    input: npt.NDArray[np.float32],
) -> Dict[str, npt.NDArray[np.float32]]:
    """Run inference on an ONNX model and return all intermediate outputs.

    This is useful for inspecting what's happening inside a model.

    Args:
        model_path: Path to ONNX model
        input: Numpy array input

    Returns:
        Dict mapping layer names to numpy arrays of their outputs

    Raises:
        ValueError: If model cannot be loaded or inference fails

    Example:
        >>> import numpy as np
        >>> input_data = np.zeros((1, 10), dtype=np.float32)
        >>> outputs = gamma.run_with_intermediates("model.onnx", input_data)
        >>> for layer_name, output in outputs.items():
        ...     print(f"{layer_name}: {output.shape}")
    """
    ...

def load_model_info(model_path: str) -> Dict[str, Any]:
    """Load model info (inputs, outputs, layers).

    Args:
        model_path: Path to ONNX model

    Returns:
        Dict with model information:
        - inputs: List of dicts with 'name' and 'shape' keys
        - outputs: List of dicts with 'name' and 'shape' keys
        - layer_count: Number of layers
        - layer_names: List of layer names

    Raises:
        ValueError: If model cannot be loaded

    Example:
        >>> info = gamma.load_model_info("model.onnx")
        >>> print(f"Model has {info['layer_count']} layers")
        >>> for inp in info['inputs']:
        ...     print(f"Input: {inp['name']} shape={inp['shape']}")
    """
    ...

def load_npy(path: str) -> npt.NDArray[np.float32]:
    """Load a numpy file (.npy).

    Args:
        path: Path to .npy file

    Returns:
        Numpy array with float32 dtype

    Raises:
        ValueError: If file cannot be loaded

    Example:
        >>> data = gamma.load_npy("test_input.npy")
        >>> print(f"Loaded array with shape {data.shape}")
    """
    ...

# ==============================================================================
# P2: Sensitivity Analysis
# ==============================================================================

class LayerSensitivity:
    """Result of analyzing a single layer's sensitivity.

    Sensitivity measures how much a layer amplifies input uncertainty:
    - sensitivity < 1.0: Layer contracts bounds (stable)
    - sensitivity = 1.0: Layer preserves bounds (neutral)
    - sensitivity > 1.0: Layer amplifies bounds (potentially unstable)

    Attributes:
        name: Layer name from the model
        layer_type: Layer type string (e.g., "Linear", "ReLU")
        input_width: Input bound width (max across all elements)
        output_width: Output bound width (max across all elements)
        sensitivity: Amplification factor (output_width / input_width)
        mean_output_width: Mean bound width at output
        output_shape: Shape of layer output
        has_overflow: True if output bounds contain infinity/NaN
    """
    name: str
    layer_type: str
    input_width: float
    output_width: float
    sensitivity: float
    mean_output_width: float
    output_shape: List[int]
    has_overflow: bool

    def is_high_sensitivity(self, threshold: float) -> bool:
        """Check if this layer amplifies significantly (sensitivity > threshold)."""
        ...

    def is_contractive(self) -> bool:
        """Check if this layer contracts bounds (sensitivity < 1.0)."""
        ...

class SensitivityResult:
    """Result of a full sensitivity analysis.

    Attributes:
        layers: Per-layer sensitivity analysis
        total_sensitivity: Product of all layer sensitivities
        max_sensitivity: Maximum single-layer sensitivity
        max_sensitivity_layer: Index of layer with max sensitivity
        input_epsilon: Input perturbation size used
        final_width: Final output bound width
        overflow_at_layer: Index of first overflow layer (None if none)

    Properties:
        max_sensitivity_layer_name: Name of layer with max sensitivity
        has_overflow: True if any layer overflowed
    """
    layers: List[LayerSensitivity]
    total_sensitivity: float
    max_sensitivity: float
    max_sensitivity_layer: Optional[int]
    input_epsilon: float
    final_width: float
    overflow_at_layer: Optional[int]

    @property
    def max_sensitivity_layer_name(self) -> Optional[str]:
        """Get name of the layer with maximum sensitivity."""
        ...

    @property
    def has_overflow(self) -> bool:
        """Check if overflow occurred."""
        ...

    def summary(self) -> str:
        """Get a formatted summary table."""
        ...

    def hot_spots(self, threshold: float) -> List[LayerSensitivity]:
        """Get high-sensitivity layers (above threshold)."""
        ...

def sensitivity_analysis(
    model_path: str,
    epsilon: float = 0.01,
    continue_after_overflow: bool = False,
) -> SensitivityResult:
    """Analyze layer-by-layer sensitivity (noise amplification).

    Computes how each layer amplifies input uncertainty. High sensitivity
    layers are "choke points" where verification becomes difficult.

    Args:
        model_path: Path to ONNX model
        epsilon: Input perturbation size (default: 0.01)
        continue_after_overflow: Keep going after overflow (default: False)

    Returns:
        SensitivityResult with per-layer analysis

    Raises:
        ValueError: If model cannot be loaded or analysis fails

    Example:
        >>> result = gamma.sensitivity_analysis("model.onnx")
        >>> print(f"Max sensitivity: {result.max_sensitivity:.2f}")
        >>> for layer in result.hot_spots(10.0):
        ...     print(f"  {layer.name}: {layer.sensitivity:.2f}x")
    """
    ...

# ==============================================================================
# P2: Quantization Safety Analysis
# ==============================================================================

class QuantSafety(Enum):
    """Quantization safety status for a layer.

    Values:
        Safe: Values within representable range
        Denormal: Values in denormal range (precision loss)
        ScalingRequired: Values require careful scaling (int8)
        Overflow: Values may overflow the format
        Unknown: Bounds are infinite or NaN
    """
    Safe = ...
    Denormal = ...
    ScalingRequired = ...
    Overflow = ...
    Unknown = ...

class LayerQuantization:
    """Result of analyzing a single layer's quantization safety.

    Attributes:
        name: Layer name from the model
        layer_type: Layer type string
        min_bound: Minimum output bound across all elements
        max_bound: Maximum output bound across all elements
        max_abs: Maximum absolute value in bounds
        output_shape: Shape of layer output
        float16_safety: Safety status for float16 quantization
        int8_safety: Safety status for int8 quantization
        int8_scale: Suggested int8 scale factor (if applicable)
        has_overflow: True if bounds are infinite/NaN
    """
    name: str
    layer_type: str
    min_bound: float
    max_bound: float
    max_abs: float
    output_shape: List[int]
    float16_safety: QuantSafety
    int8_safety: QuantSafety
    int8_scale: Optional[float]
    has_overflow: bool

    def is_float16_safe(self) -> bool:
        """Check if safe for float16."""
        ...

    def is_int8_safe(self) -> bool:
        """Check if safe for int8 (with or without scaling)."""
        ...

class QuantizationResult:
    """Result of a full quantization safety analysis.

    Attributes:
        layers: Per-layer quantization analysis
        float16_safe: True if all layers safe for float16
        int8_safe: True if all layers safe for int8
        float16_overflow_count: Number of layers with float16 overflow risk
        int8_overflow_count: Number of layers with int8 overflow risk
        denormal_count: Number of layers in float16 denormal range
        input_epsilon: Input perturbation size used
    """
    layers: List[LayerQuantization]
    float16_safe: bool
    int8_safe: bool
    float16_overflow_count: int
    int8_overflow_count: int
    denormal_count: int
    input_epsilon: float

    def summary(self) -> str:
        """Get a formatted summary table."""
        ...

    def float16_unsafe_layers(self) -> List[LayerQuantization]:
        """Get layers that are unsafe for float16."""
        ...

    def int8_unsafe_layers(self) -> List[LayerQuantization]:
        """Get layers that are unsafe for int8."""
        ...

def quantize_check(
    model_path: str,
    epsilon: float = 0.01,
    check_float16: bool = True,
    check_int8: bool = True,
) -> QuantizationResult:
    """Check if model layers can safely be quantized to float16/int8.

    Uses bound propagation to determine the output range of each layer,
    then checks if those ranges fit within the target format.

    Args:
        model_path: Path to ONNX model
        epsilon: Input perturbation size (default: 0.01)
        check_float16: Check float16 safety (default: True)
        check_int8: Check int8 safety (default: True)

    Returns:
        QuantizationResult with per-layer safety analysis

    Raises:
        ValueError: If model cannot be loaded or analysis fails

    Example:
        >>> result = gamma.quantize_check("model.onnx")
        >>> assert result.float16_safe, "Model has float16 overflow risk"
        >>> for layer in result.float16_unsafe_layers():
        ...     print(f"  Unsafe: {layer.name}")
    """
    ...

# ==============================================================================
# P2: Bound Width Profiling
# ==============================================================================

class BoundStatus(Enum):
    """Bound width status indicator.

    Values:
        Tight: Bounds are tight and stable
        Moderate: Bounds are moderate
        Wide: Bounds are wide - verification getting harder
        VeryWide: Bounds are very wide - verification difficult
        Overflow: Bounds have overflowed (infinity/NaN)
    """
    Tight = ...
    Moderate = ...
    Wide = ...
    VeryWide = ...
    Overflow = ...

class LayerProfile:
    """Result of profiling a single layer's bounds.

    Attributes:
        name: Layer name from the model
        layer_type: Layer type string
        input_width: Input bound width (max across all elements)
        output_width: Output bound width (max across all elements)
        mean_output_width: Mean output bound width
        median_output_width: Median output bound width
        growth_ratio: Width growth ratio (output/input)
        cumulative_expansion: Total expansion from network input
        output_shape: Shape of layer output
        num_elements: Number of elements in output
        status: Bound status indicator
    """
    name: str
    layer_type: str
    input_width: float
    output_width: float
    mean_output_width: float
    median_output_width: float
    growth_ratio: float
    cumulative_expansion: float
    output_shape: List[int]
    num_elements: int
    status: BoundStatus

    def is_choke_point(self, threshold: float) -> bool:
        """Check if this layer is a choke point (high growth)."""
        ...

class ProfileResult:
    """Result of a full bound profiling analysis.

    Attributes:
        layers: Per-layer bound profiles
        input_epsilon: Input perturbation size used
        initial_width: Initial input bound width (2 * epsilon)
        final_width: Final output bound width
        total_expansion: Total expansion (final_width / initial_width)
        max_growth_layer: Index of layer with highest growth ratio
        max_growth_ratio: Maximum single-layer growth ratio
        overflow_at_layer: Index of first overflow layer (None if none)
        difficulty_score: Verification difficulty score (0-100)

    Properties:
        max_growth_layer_name: Name of layer with max growth
        has_overflow: True if any layer overflowed
    """
    layers: List[LayerProfile]
    input_epsilon: float
    initial_width: float
    final_width: float
    total_expansion: float
    max_growth_layer: Optional[int]
    max_growth_ratio: float
    overflow_at_layer: Optional[int]
    difficulty_score: float

    @property
    def max_growth_layer_name(self) -> Optional[str]:
        """Get name of layer with maximum growth."""
        ...

    @property
    def has_overflow(self) -> bool:
        """Check if overflow occurred."""
        ...

    def summary(self) -> str:
        """Get a formatted summary table."""
        ...

    def choke_points(self, threshold: float) -> List[LayerProfile]:
        """Get choke points (layers with growth above threshold)."""
        ...

    def problematic_layers(self) -> List[LayerProfile]:
        """Get problematic layers (wide or worse bounds)."""
        ...

def profile_bounds(
    model_path: str,
    epsilon: float = 0.01,
) -> ProfileResult:
    """Profile bound widths through the network.

    Tracks how bound widths grow layer-by-layer, helping identify where
    verification becomes difficult. Also computes a verification difficulty score.

    Args:
        model_path: Path to ONNX model
        epsilon: Input perturbation size (default: 0.01)

    Returns:
        ProfileResult with per-layer bound analysis

    Raises:
        ValueError: If model cannot be loaded or analysis fails

    Example:
        >>> result = gamma.profile_bounds("model.onnx")
        >>> print(f"Difficulty: {result.difficulty_score:.0f}/100")
        >>> for layer in result.choke_points(5.0):
        ...     print(f"  {layer.name}: {layer.growth_ratio:.2f}x growth")
    """
    ...

# ==============================================================================
# Verification API
# ==============================================================================

class OutputBound:
    """A single output bound (lower, upper).

    Attributes:
        lower: Lower bound value
        upper: Upper bound value

    Properties:
        width: Width of the bound interval (upper - lower)
        midpoint: Midpoint of the bound ((lower + upper) / 2)
    """
    lower: float
    upper: float

    @property
    def width(self) -> float:
        """Width of the bound interval."""
        ...

    @property
    def midpoint(self) -> float:
        """Midpoint of the bound."""
        ...

class VerifyStatus(Enum):
    """Status of a verification result.

    Values:
        Verified: Property verified for all inputs in the region
        Violated: Counterexample found that violates the property
        Unknown: Verification inconclusive (bounds too loose)
        Timeout: Verification timed out
    """
    Verified = ...
    Violated = ...
    Unknown = ...
    Timeout = ...

class VerifyResult:
    """Result of neural network verification.

    Attributes:
        status: Verification status (Verified, Violated, Unknown, Timeout)
        output_bounds: Certified output bounds (if available)
        counterexample: Input that violates the property (if Violated)
        counterexample_output: Network output at counterexample (if Violated)
        reason: Explanation for Unknown/Timeout status
        method: Verification method used
        epsilon: Input perturbation radius used

    Properties:
        is_verified: True if property was verified
        is_violated: True if violation was found
    """
    status: VerifyStatus
    output_bounds: Optional[List[OutputBound]]
    counterexample: Optional[List[float]]
    counterexample_output: Optional[List[float]]
    reason: Optional[str]
    method: str
    epsilon: float

    @property
    def is_verified(self) -> bool:
        """Check if the property was verified."""
        ...

    @property
    def is_violated(self) -> bool:
        """Check if a violation was found."""
        ...

    def max_output_width(self) -> Optional[float]:
        """Get max output bound width (for diagnostics)."""
        ...

    def summary(self) -> str:
        """Get formatted summary."""
        ...

def verify(
    model_path: str,
    epsilon: float = 0.01,
    method: str = "alpha",
    timeout: int = 60,
) -> VerifyResult:
    """Verify a neural network property using bound propagation.

    Uses bound propagation (IBP, CROWN, α-CROWN, or β-CROWN) to compute
    certified output bounds for all inputs within an epsilon ball.

    Args:
        model_path: Path to ONNX model
        epsilon: Input perturbation radius (default: 0.01)
        method: Verification method - 'ibp', 'crown', 'alpha', or 'beta' (default: 'alpha')
        timeout: Timeout in seconds (default: 60)

    Returns:
        VerifyResult with verification status and output bounds

    Raises:
        ValueError: If model cannot be loaded, method is invalid, or verification fails

    Example:
        >>> result = gamma.verify("model.onnx", epsilon=0.01)
        >>> assert result.is_verified, f"Verification failed: {result.reason}"
        >>> print(f"Output bounds certified with max width: {result.max_output_width():.2e}")

        >>> # With different method
        >>> result = gamma.verify("model.onnx", method="ibp")  # Fastest, loosest bounds

        >>> # With timeout
        >>> result = gamma.verify("model.onnx", timeout=120)  # 2 minutes
    """
    ...

# ==============================================================================
# Model Comparison API
# ==============================================================================

class BoundViolation:
    """A single bound violation between two models.

    Attributes:
        index: Element index of the violation
        ref_lower: Reference model lower bound
        ref_upper: Reference model upper bound
        target_lower: Target model lower bound
        target_upper: Target model upper bound
        lower_diff: Absolute difference in lower bounds
        upper_diff: Absolute difference in upper bounds
    """
    index: int
    ref_lower: float
    ref_upper: float
    target_lower: float
    target_upper: float
    lower_diff: float
    upper_diff: float

class CompareResult:
    """Result of comparing two models using bound propagation.

    Attributes:
        is_equivalent: True if bounds match within tolerance
        max_lower_diff: Maximum difference in lower bounds
        max_upper_diff: Maximum difference in upper bounds
        tolerance: Tolerance threshold used
        overlap_pct: Percentage of bounds that overlap
        ref_max_width: Max bound width from reference model
        target_max_width: Max bound width from target model
        method: Verification method used
        epsilon: Input perturbation radius used
        violations: List of bound violations
    """
    is_equivalent: bool
    max_lower_diff: float
    max_upper_diff: float
    tolerance: float
    overlap_pct: float
    ref_max_width: float
    target_max_width: float
    method: str
    epsilon: float
    violations: List[BoundViolation]

    def summary(self) -> str:
        """Get a formatted summary."""
        ...

def compare(
    reference: str,
    target: str,
    tolerance: float = 0.001,
    epsilon: float = 0.01,
    method: str = "crown",
) -> CompareResult:
    """Compare two models using bound propagation.

    Runs bound propagation on both models with the same input perturbation
    and compares the resulting output bounds element-wise.

    Args:
        reference: Path to reference ONNX model
        target: Path to target ONNX model
        tolerance: Maximum allowed difference in bounds (default: 0.001)
        epsilon: Input perturbation radius (default: 0.01)
        method: Verification method - 'ibp', 'crown', 'alpha' (default: 'crown')

    Returns:
        CompareResult with comparison results

    Raises:
        ValueError: If models cannot be loaded or method is invalid

    Example:
        >>> result = gamma.compare("model_pytorch.onnx", "model_coreml.onnx")
        >>> assert result.is_equivalent, f"Bounds differ: max diff = {result.max_lower_diff:.2e}"

        >>> # With different method
        >>> result = gamma.compare("model_a.onnx", "model_b.onnx", method="ibp")
    """
    ...

# ==============================================================================
# Weights API
# ==============================================================================

class TensorInfo:
    """Information about a tensor in a weight file.

    Attributes:
        name: Tensor name
        shape: Tensor shape
        elements: Total number of elements
    """
    name: str
    shape: List[int]
    elements: int

class WeightsInfo:
    """Result of weight file inspection.

    Attributes:
        format: File format (e.g., "ONNX", "SafeTensors")
        tensor_count: Number of tensors
        total_params: Total number of parameters
        tensors: List of tensor information
    """
    format: str
    tensor_count: int
    total_params: int
    tensors: List[TensorInfo]

    def summary(self) -> str:
        """Get a formatted summary."""
        ...

def weights_info(path: str) -> WeightsInfo:
    """Get information about weights in a file or directory.

    Supports ONNX (.onnx), SafeTensors (.safetensors), PyTorch (.pt, .pth, .bin),
    GGUF (.gguf), and CoreML (.mlmodel, .mlpackage) formats.

    Also supports directories containing:
    - Sharded SafeTensors files (*.safetensors)
    - MLX models (config.json + *.safetensors)
    - HuggingFace PyTorch checkpoints (pytorch_model.bin, model.pt)
    - HuggingFace sharded PyTorch checkpoints (pytorch_model-*.bin + pytorch_model.bin.index.json)

    Args:
        path: Path to weights file or directory

    Returns:
        WeightsInfo with tensor information

    Raises:
        ValueError: If file/directory cannot be loaded or format is unsupported

    Example:
        >>> info = gamma.weights_info("model.safetensors")
        >>> print(f"Total params: {info.total_params:,}")
        >>> for t in info.tensors[:5]:
        ...     print(f"  {t.name}: {t.shape}")

        >>> # GGUF (llama.cpp) format
        >>> info = gamma.weights_info("model.gguf")
        >>> print(f"GGUF model: {info.tensor_count} tensors")

        >>> # CoreML format
        >>> info = gamma.weights_info("model.mlmodel")  # or model.mlpackage
        >>> print(f"CoreML model: {info.total_params:,} params")

        >>> # Sharded SafeTensors directory (HuggingFace models)
        >>> info = gamma.weights_info("path/to/llama-7b/")
        >>> print(f"Sharded model: {info.total_params:,} params across {info.tensor_count} tensors")
    """
    ...

class TensorComparison:
    """Result of comparing a single tensor between two weight files.

    Attributes:
        name: Tensor name
        status: Comparison status ("match", "differs", "shape_mismatch", "missing_in_a", "missing_in_b")
        max_diff: Maximum absolute difference (if both present and shapes match)
        shape_a: Shape in file A (if present)
        shape_b: Shape in file B (if present)
    """
    name: str
    status: str
    max_diff: Optional[float]
    shape_a: Optional[List[int]]
    shape_b: Optional[List[int]]

class WeightsDiffResult:
    """Result of comparing two weight files.

    Attributes:
        is_match: True if all tensors match within tolerance
        max_diff: Maximum difference across all tensors
        tolerance: Tolerance threshold used
        differing_count: Number of differing tensors
        total_tensors_a: Number of tensors in file A
        total_tensors_b: Number of tensors in file B
        comparisons: Per-tensor comparison results
    """
    is_match: bool
    max_diff: float
    tolerance: float
    differing_count: int
    total_tensors_a: int
    total_tensors_b: int
    comparisons: List[TensorComparison]

    def summary(self) -> str:
        """Get a formatted summary."""
        ...

    def matching_tensors(self) -> List[TensorComparison]:
        """Get matching tensors."""
        ...

    def differing_tensors(self) -> List[TensorComparison]:
        """Get differing tensors."""
        ...

def weights_diff(
    file_a: str,
    file_b: str,
    tolerance: float = 1e-6,
) -> WeightsDiffResult:
    """Compare weights between two files or directories.

    Supports ONNX (.onnx), SafeTensors (.safetensors), PyTorch (.pt, .pth, .bin),
    GGUF (.gguf), and CoreML (.mlmodel, .mlpackage) formats.
    Also supports directories containing sharded SafeTensors files, MLX models, or PyTorch checkpoints.

    Files/directories can be different formats (e.g., compare .pt to .safetensors,
    or .gguf to a sharded SafeTensors directory).

    Args:
        file_a: Path to first weights file or directory
        file_b: Path to second weights file or directory
        tolerance: Maximum allowed absolute difference (default: 1e-6)

    Returns:
        WeightsDiffResult with comparison results

    Raises:
        ValueError: If files/directories cannot be loaded or formats are unsupported

    Example:
        >>> result = gamma.weights_diff("model_a.safetensors", "model_b.safetensors")
        >>> assert result.is_match, f"Max diff: {result.max_diff:.2e}"
        >>> for diff in result.differing_tensors():
        ...     print(f"  {diff.name}: {diff.status}")

        >>> # Cross-format comparison
        >>> result = gamma.weights_diff("model.gguf", "model.safetensors")
        >>> print(f"Cross-format match: {result.is_match}")

        >>> # Compare sharded directory to single file
        >>> result = gamma.weights_diff("path/to/model/", "model.safetensors")
        >>> print(f"Sharded vs single: {result.is_match}")
    """
    ...

# ==============================================================================
# Benchmark API
# ==============================================================================

class BenchResultItem:
    """Single benchmark result item.

    Attributes:
        name: Name of the benchmark
        iterations: Number of iterations run
        per_iter_ns: Time per iteration in nanoseconds
        per_iter_us: Time per iteration in microseconds
        per_iter_ms: Time per iteration in milliseconds
        total_ns: Total time in nanoseconds
        total_ms: Total time in milliseconds
    """
    name: str
    iterations: int
    per_iter_ns: int
    per_iter_us: float
    per_iter_ms: float
    total_ns: int
    total_ms: float

class BenchDimensions:
    """Dimensions used for benchmarks.

    Attributes:
        batch: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        intermediate_dim: Intermediate (feedforward) dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        epsilon: Epsilon perturbation used
    """
    batch: int
    seq_len: int
    hidden_dim: int
    intermediate_dim: int
    num_heads: int
    head_dim: int
    epsilon: float

class BenchResult:
    """Full benchmark result.

    Attributes:
        benchmark_type: Type of benchmark (layer, attention, full)
        valid_type: Whether the benchmark type was valid
        dimensions: Dimensions used for the benchmark
        results: Individual benchmark results
    """
    benchmark_type: str
    valid_type: bool
    dimensions: BenchDimensions
    results: List[BenchResultItem]

    def summary(self) -> str:
        """Get a summary of all benchmark results."""
        ...

def bench(
    benchmark_type: str = "layer",
) -> BenchResult:
    """Run γ-CROWN benchmarks.

    Runs performance benchmarks for neural network verification operations.

    Args:
        benchmark_type: Type of benchmark to run. Options:
            - "layer" (default): Individual layer IBP performance
            - "attention": Attention component (MatMul, Softmax) performance
            - "full": Full pipeline scaling tests

    Returns:
        BenchResult with timing information for each benchmark

    Raises:
        ValueError: If benchmark fails

    Example:
        >>> result = gamma.bench()  # Run layer benchmarks
        >>> print(result.summary())
        >>> for r in result.results:
        ...     print(f"{r.name}: {r.per_iter_ms:.3f}ms")

        >>> result = gamma.bench("attention")  # Run attention benchmarks
        >>> result = gamma.bench("full")  # Run full pipeline benchmarks
    """
    ...
