# γ-CROWN Python Bindings

Python bindings for γ-CROWN neural network verification library.

## Installation

```bash
pip install gamma
```

## Quick Start

```python
import gamma

# Compare two ONNX models layer-by-layer
diff = gamma.diff("model_a.onnx", "model_b.onnx")

# Check if models are equivalent
assert diff.is_equivalent, f"Diverges at {diff.first_bad_layer_name}"

# Get detailed summary
print(diff.summary())
```

## Usage with pytest

```python
import gamma
import numpy as np

def test_model_port_equivalent():
    """Verify PyTorch→CoreML port produces identical outputs."""
    diff = gamma.diff(
        "model_pytorch.onnx",
        "model_coreml.onnx",
        tolerance=1e-5
    )
    assert diff.is_equivalent, f"Diverges at {diff.first_bad_layer_name}"

def test_custom_input():
    """Test with specific input values."""
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    diff = gamma.diff(
        "model_a.onnx",
        "model_b.onnx",
        input=input_data
    )
    assert diff.max_divergence < 1e-4
```

## Type Hints

γ-CROWN includes full type stub support (PEP 561). IDEs like VS Code and PyCharm will provide:
- Autocompletion for all functions and classes
- Inline documentation
- Type checking with mypy/pyright

```python
# Type checking works out of the box
import gamma
import numpy as np
import numpy.typing as npt

def analyze_model(path: str) -> gamma.DiffResult:
    """Fully typed function."""
    return gamma.diff(path, path)

# mypy will catch type errors:
# result: int = gamma.diff("a", "b")  # Error: expected DiffResult
```

Run type checking:
```bash
mypy your_tests.py
```

## API Reference

### `gamma.diff(model_a, model_b, tolerance=1e-5, input=None, continue_after_divergence=True, layer_mapping=None)`

Compare two ONNX models layer-by-layer.

**Parameters:**
- `model_a`: Path to first ONNX model
- `model_b`: Path to second ONNX model
- `tolerance`: Maximum allowed difference (default: 1e-5)
- `input`: Optional numpy array for input (default: zeros)
- `continue_after_divergence`: Whether to continue after finding divergence (default: True)
- `layer_mapping`: Optional dict mapping layer names from A to B

**Returns:** `DiffResult` object

### `DiffResult`

- `is_equivalent`: bool - Whether models match within tolerance
- `max_divergence`: float - Maximum difference across all layers
- `first_bad_layer`: int or None - Index of first diverging layer
- `first_bad_layer_name`: str or None - Name of first diverging layer
- `layers`: list[LayerComparison] - Per-layer comparison results
- `suggestion`: str or None - Suggested root cause
- `summary()`: str - Formatted comparison table

### `gamma.run_with_intermediates(model_path, input)`

Run inference and get all intermediate layer outputs.

**Parameters:**
- `model_path`: Path to ONNX model
- `input`: Numpy array input

**Returns:** Dict mapping layer names to numpy arrays

### `gamma.load_model_info(model_path)`

Get model metadata (inputs, outputs, layers).

**Returns:** Dict with model information

---

## P2: Advanced Analysis Commands

### Sensitivity Analysis

Analyze how each layer amplifies input uncertainty. High sensitivity layers are "choke points" where verification becomes difficult.

```python
import gamma

# Analyze sensitivity
result = gamma.sensitivity_analysis("model.onnx", epsilon=0.01)

# Check results
print(f"Max sensitivity: {result.max_sensitivity:.2f}x at {result.max_sensitivity_layer_name}")
print(f"Total sensitivity: {result.total_sensitivity:.2e}")

# Find problematic layers (>10x amplification)
for layer in result.hot_spots(10.0):
    print(f"  {layer.name}: {layer.sensitivity:.2f}x")

# Use in tests
def test_no_extreme_amplification():
    result = gamma.sensitivity_analysis("model.onnx")
    hot_spots = result.hot_spots(100.0)  # >100x amplification
    assert len(hot_spots) == 0, f"Found {len(hot_spots)} unstable layers"
```

### `gamma.sensitivity_analysis(model_path, epsilon=0.01, continue_after_overflow=False)`

**Parameters:**
- `model_path`: Path to ONNX model
- `epsilon`: Input perturbation size (default: 0.01)
- `continue_after_overflow`: Keep analyzing after overflow (default: False)

**Returns:** `SensitivityResult` object with:
- `layers`: list[LayerSensitivity] - Per-layer analysis
- `max_sensitivity`: float - Highest single-layer sensitivity
- `max_sensitivity_layer_name`: str or None - Name of most sensitive layer
- `total_sensitivity`: float - Product of all sensitivities
- `hot_spots(threshold)`: Get layers with sensitivity > threshold
- `summary()`: Formatted table

---

### Quantization Safety Check

Check if model layers can safely be quantized to float16/int8 without overflow.

```python
import gamma

# Check quantization safety
result = gamma.quantize_check("model.onnx")

# Overall safety
print(f"Float16 safe: {result.float16_safe}")
print(f"Int8 safe: {result.int8_safe}")

# Find unsafe layers
for layer in result.float16_unsafe_layers():
    print(f"  {layer.name}: bounds [{layer.min_bound:.2e}, {layer.max_bound:.2e}]")

# Use in tests
def test_quantization_safe():
    result = gamma.quantize_check("model.onnx")
    assert result.float16_safe, f"Model has float16 overflow risk:\n{result.summary()}"
```

### `gamma.quantize_check(model_path, epsilon=0.01, check_float16=True, check_int8=True)`

**Parameters:**
- `model_path`: Path to ONNX model
- `epsilon`: Input perturbation size (default: 0.01)
- `check_float16`: Check float16 safety (default: True)
- `check_int8`: Check int8 safety (default: True)

**Returns:** `QuantizationResult` object with:
- `layers`: list[LayerQuantization] - Per-layer analysis
- `float16_safe`: bool - True if all layers safe for float16
- `int8_safe`: bool - True if all layers safe for int8
- `float16_unsafe_layers()`: Get layers unsafe for float16
- `int8_unsafe_layers()`: Get layers unsafe for int8
- `summary()`: Formatted table

**QuantSafety enum values:**
- `Safe`: Values within representable range
- `Denormal`: Values in denormal range (precision loss)
- `ScalingRequired`: Values need scaling (int8)
- `Overflow`: Values may overflow
- `Unknown`: Bounds are infinite/NaN

---

### Bound Width Profiling

Track how bound widths grow through the network. Helps identify where verification becomes difficult.

```python
import gamma

# Profile bounds
result = gamma.profile_bounds("model.onnx")

# Check verification difficulty (0-100)
print(f"Verification difficulty: {result.difficulty_score:.0f}/100")
print(f"Total expansion: {result.total_expansion:.2f}x")

# Find choke points (>5x growth)
for layer in result.choke_points(5.0):
    print(f"  {layer.name}: {layer.growth_ratio:.2f}x growth")

# Use in tests
def test_verification_feasible():
    result = gamma.profile_bounds("model.onnx")
    assert result.difficulty_score < 50, (
        f"Difficulty {result.difficulty_score}/100 too high"
    )
```

### `gamma.profile_bounds(model_path, epsilon=0.01)`

**Parameters:**
- `model_path`: Path to ONNX model
- `epsilon`: Input perturbation size (default: 0.01)

**Returns:** `ProfileResult` object with:
- `layers`: list[LayerProfile] - Per-layer analysis
- `difficulty_score`: float - Verification difficulty (0-100)
- `total_expansion`: float - Final width / initial width
- `max_growth_layer_name`: str or None - Layer with highest growth
- `choke_points(threshold)`: Get layers with growth > threshold
- `problematic_layers()`: Get layers with wide/overflow bounds
- `summary()`: Formatted table

**BoundStatus enum values:**
- `Tight`: Bounds are tight and stable
- `Moderate`: Bounds are moderate
- `Wide`: Bounds are wide - verification harder
- `VeryWide`: Bounds very wide - verification difficult
- `Overflow`: Bounds overflowed (infinity/NaN)
