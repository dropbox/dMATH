//! Python bindings for γ-CROWN neural network verification.
//!
//! Provides a `pytest`-friendly API for neural network testing and debugging.
//!
//! ## Example Usage
//!
//! ```python
//! import gamma
//!
//! def test_port_equivalent():
//!     diff = gamma.diff("model_torch.onnx", "model_coreml.onnx")
//!     assert diff.max_divergence < 1e-5, f"Diverges at {diff.first_bad_layer}"
//!
//! def test_specific_tolerance():
//!     diff = gamma.diff("model_a.onnx", "model_b.onnx", tolerance=1e-4)
//!     assert diff.is_equivalent
//! ```

use ndarray::IxDyn;
use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

// P2 command imports
use gamma_onnx::profile::{self, BoundStatus as RustBoundStatus};
use gamma_onnx::quantize::{self, QuantFormat, QuantSafety as RustQuantSafety};
use gamma_onnx::sensitivity;

// Verification imports
use gamma_core::{
    Bound as RustBound, VerificationResult as RustVerificationResult, VerificationSpec,
};
#[cfg(feature = "coreml")]
use gamma_onnx::coreml::load_coreml;
#[cfg(feature = "gguf")]
use gamma_onnx::gguf::load_gguf;
use gamma_onnx::load_onnx;
use gamma_onnx::native::load_weights;
#[cfg(feature = "pytorch")]
use gamma_onnx::pytorch::load_pytorch;
use gamma_onnx::safetensors::load_safetensors;
use gamma_onnx::WeightStore;
use gamma_propagate::{
    BoundPropagation, GELULayer, Layer, LayerNormLayer, LinearLayer, MatMulLayer, Network,
    PropagationConfig, PropagationMethod, SoftmaxLayer, Verifier,
};
use gamma_tensor::BoundedTensor;
use ndarray::{Array1, Array2, IxDyn as NdIxDyn};
use std::time::Instant;

/// Result of comparing a single layer between two models.
#[pyclass]
#[derive(Clone)]
pub struct LayerComparison {
    #[pyo3(get)]
    pub name: String,

    #[pyo3(get)]
    pub name_b: Option<String>,

    #[pyo3(get)]
    pub max_diff: f32,

    #[pyo3(get)]
    pub mean_diff: f32,

    #[pyo3(get)]
    pub exceeds_tolerance: bool,

    #[pyo3(get)]
    pub shape_a: Vec<usize>,

    #[pyo3(get)]
    pub shape_b: Vec<usize>,
}

#[pymethods]
impl LayerComparison {
    fn __repr__(&self) -> String {
        format!(
            "LayerComparison(name='{}', max_diff={:.2e}, exceeds={})",
            self.name, self.max_diff, self.exceeds_tolerance
        )
    }
}

/// Status of a layer comparison.
#[pyclass]
#[derive(Clone)]
pub enum DiffStatus {
    Ok,
    DriftStarts,
    ExceedsTolerance,
    ShapeMismatch,
}

#[pymethods]
impl DiffStatus {
    fn __repr__(&self) -> String {
        match self {
            DiffStatus::Ok => "DiffStatus.Ok".to_string(),
            DiffStatus::DriftStarts => "DiffStatus.DriftStarts".to_string(),
            DiffStatus::ExceedsTolerance => "DiffStatus.ExceedsTolerance".to_string(),
            DiffStatus::ShapeMismatch => "DiffStatus.ShapeMismatch".to_string(),
        }
    }
}

/// Result of a full model diff operation.
#[pyclass]
#[derive(Clone)]
pub struct DiffResult {
    #[pyo3(get)]
    pub layers: Vec<LayerComparison>,

    #[pyo3(get)]
    pub first_bad_layer: Option<usize>,

    #[pyo3(get)]
    pub drift_start_layer: Option<usize>,

    #[pyo3(get)]
    pub max_divergence: f32,

    #[pyo3(get)]
    pub tolerance: f32,

    #[pyo3(get)]
    pub suggestion: Option<String>,
}

#[pymethods]
impl DiffResult {
    /// Check if models are equivalent within tolerance.
    #[getter]
    fn is_equivalent(&self) -> bool {
        self.first_bad_layer.is_none()
    }

    /// Get the name of the first bad layer, if any.
    #[getter]
    fn first_bad_layer_name(&self) -> Option<String> {
        self.first_bad_layer
            .and_then(|i| self.layers.get(i))
            .map(|l| l.name.clone())
    }

    /// Get status for each layer.
    fn statuses(&self) -> Vec<DiffStatus> {
        self.layers
            .iter()
            .enumerate()
            .map(|(i, l)| {
                if l.shape_a != l.shape_b {
                    DiffStatus::ShapeMismatch
                } else if l.exceeds_tolerance {
                    DiffStatus::ExceedsTolerance
                } else if self.drift_start_layer == Some(i) {
                    DiffStatus::DriftStarts
                } else {
                    DiffStatus::Ok
                }
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "DiffResult(layers={}, max_divergence={:.2e}, is_equivalent={})",
            self.layers.len(),
            self.max_divergence,
            self.is_equivalent()
        )
    }

    /// Get a formatted summary table (like CLI output).
    fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Layer-by-Layer Comparison".to_string());
        lines.push("==========================".to_string());
        lines.push(format!(
            "{:<40} | {:>12} | {}",
            "Layer", "Max Diff", "Status"
        ));
        lines.push(format!("{:-<40}-+-{:-<12}-+--------", "", ""));

        let statuses = self.statuses();
        for (layer, status) in self.layers.iter().zip(statuses.iter()) {
            let status_str = match status {
                DiffStatus::Ok => "OK",
                DiffStatus::DriftStarts => "DRIFT STARTS",
                DiffStatus::ExceedsTolerance => "EXCEEDS",
                DiffStatus::ShapeMismatch => "SHAPE MISMATCH",
            };
            lines.push(format!(
                "{:<40} | {:>12.2e} | {}",
                truncate_name(&layer.name, 40),
                layer.max_diff,
                status_str
            ));
        }

        if let Some(ref suggestion) = self.suggestion {
            lines.push(String::new());
            lines.push(format!("Suggestion: {}", suggestion));
        }

        lines.join("\n")
    }
}

/// Truncate a name to fit in a given width.
fn truncate_name(name: &str, width: usize) -> String {
    if name.len() <= width {
        name.to_string()
    } else {
        format!("...{}", &name[name.len() - width + 3..])
    }
}

/// Compare two ONNX models layer-by-layer to find divergence.
///
/// This is the main entry point for model comparison. It runs inference on both
/// models with the same input and compares intermediate outputs at each layer.
///
/// Args:
///     model_a: Path to first ONNX model
///     model_b: Path to second ONNX model
///     tolerance: Maximum allowed difference (default: 1e-5)
///     input: Optional numpy array for input (default: zeros)
///     continue_after_divergence: Whether to continue after finding divergence (default: True)
///     layer_mapping: Optional dict mapping layer names from A to B
///     diagnose: Enable root cause diagnosis (default: False)
///
/// Returns:
///     DiffResult with comparison results
///
/// Example:
///     >>> diff = gamma.diff("model_a.onnx", "model_b.onnx")
///     >>> assert diff.is_equivalent, f"Diverges at {diff.first_bad_layer_name}"
#[pyfunction]
#[pyo3(signature = (model_a, model_b, tolerance=1e-5, input=None, continue_after_divergence=true, layer_mapping=None, diagnose=false))]
#[allow(clippy::too_many_arguments)]
fn diff(
    py: Python<'_>,
    model_a: &str,
    model_b: &str,
    tolerance: f32,
    input: Option<&Bound<'_, PyArrayDyn<f32>>>,
    continue_after_divergence: bool,
    layer_mapping: Option<HashMap<String, String>>,
    diagnose: bool,
) -> PyResult<DiffResult> {
    // Convert numpy input if provided
    let input_array = input.map(|arr| {
        let readonly = arr.readonly();
        let shape: Vec<usize> = readonly.shape().to_vec();
        let data: Vec<f32> = readonly.as_slice().unwrap().to_vec();
        ndarray::ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap()
    });

    // Build config
    let config = gamma_onnx::diff::DiffConfig {
        tolerance,
        continue_after_divergence,
        input: input_array,
        layer_mapping: layer_mapping.unwrap_or_default(),
        diagnose,
    };

    // Run diff (release GIL during computation)
    let result = Python::detach(py, || {
        gamma_onnx::diff::diff_models(model_a, model_b, &config)
    })
    .map_err(|e| PyValueError::new_err(format!("Diff error: {}", e)))?;

    // Convert to Python types
    let layers: Vec<LayerComparison> = result
        .layers
        .into_iter()
        .map(|l| LayerComparison {
            name: l.name,
            name_b: l.name_b,
            max_diff: l.max_diff,
            mean_diff: l.mean_diff,
            exceeds_tolerance: l.exceeds_tolerance,
            shape_a: l.shape_a,
            shape_b: l.shape_b,
        })
        .collect();

    Ok(DiffResult {
        layers,
        first_bad_layer: result.first_bad_layer,
        drift_start_layer: result.drift_start_layer,
        max_divergence: result.max_divergence,
        tolerance: result.tolerance,
        suggestion: result.suggestion,
    })
}

/// Run inference on an ONNX model and return all intermediate outputs.
///
/// This is useful for inspecting what's happening inside a model.
///
/// Args:
///     model_path: Path to ONNX model
///     input: Numpy array input
///
/// Returns:
///     Dict mapping layer names to numpy arrays
#[pyfunction]
fn run_with_intermediates<'py>(
    py: Python<'py>,
    model_path: &str,
    input: &Bound<'py, PyArrayDyn<f32>>,
) -> PyResult<HashMap<String, Py<PyArrayDyn<f32>>>> {
    // Convert input
    let readonly = input.readonly();
    let shape: Vec<usize> = readonly.shape().to_vec();
    let data: Vec<f32> = readonly.as_slice().unwrap().to_vec();
    let input_array = ndarray::ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap();

    // Run inference (release GIL)
    let outputs = Python::detach(py, || {
        gamma_onnx::diff::run_inference_with_intermediates(model_path, &input_array)
    })
    .map_err(|e| PyValueError::new_err(format!("Inference error: {}", e)))?;

    // Convert outputs to numpy arrays
    let mut result = HashMap::new();
    for (name, arr) in outputs {
        let py_arr = arr.to_pyarray(py).unbind();
        result.insert(name, py_arr);
    }

    Ok(result)
}

/// Load model info (inputs, outputs, layers).
///
/// Args:
///     model_path: Path to ONNX model
///
/// Returns:
///     Dict with model information
#[pyfunction]
fn load_model_info(py: Python<'_>, model_path: &str) -> PyResult<Py<pyo3::types::PyDict>> {
    let info = gamma_onnx::diff::load_model_info(model_path)
        .map_err(|e| PyValueError::new_err(format!("Load error: {}", e)))?;

    let result = pyo3::types::PyDict::new(py);

    // Convert inputs
    let inputs_list = pyo3::types::PyList::empty(py);
    for i in &info.inputs {
        let input_dict = pyo3::types::PyDict::new(py);
        input_dict.set_item("name", &i.name)?;
        input_dict.set_item("shape", &i.shape)?;
        inputs_list.append(input_dict)?;
    }
    result.set_item("inputs", inputs_list)?;

    // Convert outputs
    let outputs_list = pyo3::types::PyList::empty(py);
    for o in &info.outputs {
        let output_dict = pyo3::types::PyDict::new(py);
        output_dict.set_item("name", &o.name)?;
        output_dict.set_item("shape", &o.shape)?;
        outputs_list.append(output_dict)?;
    }
    result.set_item("outputs", outputs_list)?;

    // Layer count and names
    result.set_item("layer_count", info.layers.len())?;
    let layer_names: Vec<String> = info.layers.iter().map(|l| l.name.clone()).collect();
    result.set_item("layer_names", layer_names)?;

    Ok(result.unbind())
}

/// Load a numpy file (.npy).
#[pyfunction]
fn load_npy(py: Python<'_>, path: &str) -> PyResult<Py<PyArrayDyn<f32>>> {
    let arr = gamma_onnx::diff::load_npy(path)
        .map_err(|e| PyValueError::new_err(format!("NPY load error: {}", e)))?;
    Ok(arr.to_pyarray(py).unbind())
}

// ============================================================================
// P2 Commands: Sensitivity Analysis
// ============================================================================

/// Result of analyzing a single layer's sensitivity.
#[pyclass]
#[derive(Clone)]
pub struct LayerSensitivity {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub layer_type: String,
    #[pyo3(get)]
    pub input_width: f32,
    #[pyo3(get)]
    pub output_width: f32,
    #[pyo3(get)]
    pub sensitivity: f32,
    #[pyo3(get)]
    pub mean_output_width: f32,
    #[pyo3(get)]
    pub output_shape: Vec<usize>,
    #[pyo3(get)]
    pub has_overflow: bool,
}

#[pymethods]
impl LayerSensitivity {
    fn __repr__(&self) -> String {
        format!(
            "LayerSensitivity(name='{}', sensitivity={:.2})",
            self.name, self.sensitivity
        )
    }

    /// Check if this layer amplifies significantly (sensitivity > threshold).
    fn is_high_sensitivity(&self, threshold: f32) -> bool {
        self.sensitivity > threshold
    }

    /// Check if this layer contracts bounds (sensitivity < 1.0).
    fn is_contractive(&self) -> bool {
        self.sensitivity < 1.0
    }
}

/// Result of a full sensitivity analysis.
#[pyclass]
#[derive(Clone)]
pub struct SensitivityResult {
    #[pyo3(get)]
    pub layers: Vec<LayerSensitivity>,
    #[pyo3(get)]
    pub total_sensitivity: f32,
    #[pyo3(get)]
    pub max_sensitivity: f32,
    #[pyo3(get)]
    pub max_sensitivity_layer: Option<usize>,
    #[pyo3(get)]
    pub input_epsilon: f32,
    #[pyo3(get)]
    pub final_width: f32,
    #[pyo3(get)]
    pub overflow_at_layer: Option<usize>,
}

#[pymethods]
impl SensitivityResult {
    fn __repr__(&self) -> String {
        format!(
            "SensitivityResult(layers={}, max_sensitivity={:.2}, total_sensitivity={:.2e})",
            self.layers.len(),
            self.max_sensitivity,
            self.total_sensitivity
        )
    }

    /// Get a formatted summary table.
    fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Sensitivity Analysis".to_string());
        lines.push("====================".to_string());
        lines.push(format!(
            "{:<40} | {:>10} | {:>10} | {:>10} | Status",
            "Layer", "In Width", "Out Width", "Sens."
        ));
        lines.push(format!(
            "{:-<40}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+--------",
            "", "", "", ""
        ));

        for (i, layer) in self.layers.iter().enumerate() {
            let status = if layer.has_overflow {
                "OVERFLOW"
            } else if layer.sensitivity > 10.0 {
                "HIGH"
            } else if layer.sensitivity > 2.0 {
                "MODERATE"
            } else if layer.sensitivity < 1.0 {
                "STABLE"
            } else {
                "OK"
            };

            let is_max = self.max_sensitivity_layer == Some(i);
            let marker = if is_max { " <<<" } else { "" };

            lines.push(format!(
                "{:<40} | {:>10.3e} | {:>10.3e} | {:>10.2} | {}{}",
                truncate_name(&layer.name, 40),
                layer.input_width,
                layer.output_width,
                layer.sensitivity,
                status,
                marker
            ));
        }

        lines.push(String::new());
        lines.push(format!("Total sensitivity: {:.2e}", self.total_sensitivity));
        lines.push(format!(
            "Max single-layer sensitivity: {:.2} at layer {}",
            self.max_sensitivity,
            self.max_sensitivity_layer
                .and_then(|i| self.layers.get(i))
                .map(|l| l.name.as_str())
                .unwrap_or("N/A")
        ));

        lines.join("\n")
    }

    /// Get name of the layer with maximum sensitivity.
    #[getter]
    fn max_sensitivity_layer_name(&self) -> Option<String> {
        self.max_sensitivity_layer
            .and_then(|i| self.layers.get(i))
            .map(|l| l.name.clone())
    }

    /// Get high-sensitivity layers (above threshold).
    fn hot_spots(&self, threshold: f32) -> Vec<LayerSensitivity> {
        self.layers
            .iter()
            .filter(|l| l.sensitivity > threshold)
            .cloned()
            .collect()
    }

    /// Check if overflow occurred.
    #[getter]
    fn has_overflow(&self) -> bool {
        self.overflow_at_layer.is_some()
    }
}

/// Analyze layer-by-layer sensitivity (noise amplification).
///
/// Computes how each layer amplifies input uncertainty. High sensitivity
/// layers are "choke points" where verification becomes difficult.
///
/// Args:
///     model_path: Path to ONNX model
///     epsilon: Input perturbation size (default: 0.01)
///     continue_after_overflow: Keep going after overflow (default: False)
///
/// Returns:
///     SensitivityResult with per-layer analysis
///
/// Example:
///     >>> result = gamma.sensitivity("model.onnx")
///     >>> print(f"Max sensitivity: {result.max_sensitivity:.2f}")
///     >>> for layer in result.hot_spots(10.0):
///     ...     print(f"  {layer.name}: {layer.sensitivity:.2f}x")
#[pyfunction]
#[pyo3(signature = (model_path, epsilon=0.01, continue_after_overflow=false))]
fn sensitivity_analysis(
    py: Python<'_>,
    model_path: &str,
    epsilon: f32,
    continue_after_overflow: bool,
) -> PyResult<SensitivityResult> {
    let config = sensitivity::SensitivityConfig {
        epsilon,
        continue_after_overflow,
        input: None,
    };

    let result = Python::detach(py, || sensitivity::analyze_sensitivity(model_path, &config))
        .map_err(|e| PyValueError::new_err(format!("Sensitivity error: {}", e)))?;

    // Convert to Python types
    let layers: Vec<LayerSensitivity> = result
        .layers
        .into_iter()
        .map(|l| LayerSensitivity {
            name: l.name,
            layer_type: l.layer_type,
            input_width: l.input_width,
            output_width: l.output_width,
            sensitivity: l.sensitivity,
            mean_output_width: l.mean_output_width,
            output_shape: l.output_shape,
            has_overflow: l.has_overflow,
        })
        .collect();

    Ok(SensitivityResult {
        layers,
        total_sensitivity: result.total_sensitivity,
        max_sensitivity: result.max_sensitivity,
        max_sensitivity_layer: result.max_sensitivity_layer,
        input_epsilon: result.input_epsilon,
        final_width: result.final_width,
        overflow_at_layer: result.overflow_at_layer,
    })
}

// ============================================================================
// P2 Commands: Quantization Safety Analysis
// ============================================================================

/// Quantization safety status.
#[pyclass]
#[derive(Clone)]
pub enum QuantSafety {
    Safe,
    Denormal,
    ScalingRequired,
    Overflow,
    Unknown,
}

#[pymethods]
impl QuantSafety {
    fn __repr__(&self) -> String {
        match self {
            QuantSafety::Safe => "QuantSafety.Safe".to_string(),
            QuantSafety::Denormal => "QuantSafety.Denormal".to_string(),
            QuantSafety::ScalingRequired => "QuantSafety.ScalingRequired".to_string(),
            QuantSafety::Overflow => "QuantSafety.Overflow".to_string(),
            QuantSafety::Unknown => "QuantSafety.Unknown".to_string(),
        }
    }

    fn __str__(&self) -> String {
        match self {
            QuantSafety::Safe => "SAFE".to_string(),
            QuantSafety::Denormal => "DENORMAL".to_string(),
            QuantSafety::ScalingRequired => "SCALE".to_string(),
            QuantSafety::Overflow => "OVERFLOW".to_string(),
            QuantSafety::Unknown => "UNKNOWN".to_string(),
        }
    }
}

impl From<RustQuantSafety> for QuantSafety {
    fn from(s: RustQuantSafety) -> Self {
        match s {
            RustQuantSafety::Safe => QuantSafety::Safe,
            RustQuantSafety::Denormal => QuantSafety::Denormal,
            RustQuantSafety::ScalingRequired => QuantSafety::ScalingRequired,
            RustQuantSafety::Overflow => QuantSafety::Overflow,
            RustQuantSafety::Unknown => QuantSafety::Unknown,
        }
    }
}

/// Result of analyzing a single layer's quantization safety.
#[pyclass]
#[derive(Clone)]
pub struct LayerQuantization {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub layer_type: String,
    #[pyo3(get)]
    pub min_bound: f32,
    #[pyo3(get)]
    pub max_bound: f32,
    #[pyo3(get)]
    pub max_abs: f32,
    #[pyo3(get)]
    pub output_shape: Vec<usize>,
    #[pyo3(get)]
    pub float16_safety: QuantSafety,
    #[pyo3(get)]
    pub int8_safety: QuantSafety,
    #[pyo3(get)]
    pub int8_scale: Option<f32>,
    #[pyo3(get)]
    pub has_overflow: bool,
}

#[pymethods]
impl LayerQuantization {
    fn __repr__(&self) -> String {
        format!(
            "LayerQuantization(name='{}', f16={}, i8={})",
            self.name,
            self.float16_safety.__str__(),
            self.int8_safety.__str__()
        )
    }

    /// Check if safe for float16.
    fn is_float16_safe(&self) -> bool {
        matches!(self.float16_safety, QuantSafety::Safe)
    }

    /// Check if safe for int8 (with or without scaling).
    fn is_int8_safe(&self) -> bool {
        matches!(
            self.int8_safety,
            QuantSafety::Safe | QuantSafety::ScalingRequired
        )
    }
}

/// Result of a full quantization analysis.
#[pyclass]
#[derive(Clone)]
pub struct QuantizationResult {
    #[pyo3(get)]
    pub layers: Vec<LayerQuantization>,
    #[pyo3(get)]
    pub float16_safe: bool,
    #[pyo3(get)]
    pub int8_safe: bool,
    #[pyo3(get)]
    pub float16_overflow_count: usize,
    #[pyo3(get)]
    pub int8_overflow_count: usize,
    #[pyo3(get)]
    pub denormal_count: usize,
    #[pyo3(get)]
    pub input_epsilon: f32,
}

#[pymethods]
impl QuantizationResult {
    fn __repr__(&self) -> String {
        format!(
            "QuantizationResult(layers={}, float16_safe={}, int8_safe={})",
            self.layers.len(),
            self.float16_safe,
            self.int8_safe
        )
    }

    /// Get a formatted summary table.
    fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Quantization Safety Analysis".to_string());
        lines.push("============================".to_string());
        lines.push(format!(
            "{:<40} | {:>12} | {:>12} | {:>8} | {:>8}",
            "Layer", "Min", "Max", "F16", "I8"
        ));
        lines.push(format!(
            "{:-<40}-+-{:-<12}-+-{:-<12}-+-{:-<8}-+-{:-<8}",
            "", "", "", "", ""
        ));

        for layer in &self.layers {
            lines.push(format!(
                "{:<40} | {:>12.3e} | {:>12.3e} | {:>8} | {:>8}",
                truncate_name(&layer.name, 40),
                layer.min_bound,
                layer.max_bound,
                layer.float16_safety.__str__(),
                layer.int8_safety.__str__()
            ));
        }

        lines.push(String::new());
        lines.push(format!(
            "Float16: {} ({} overflow, {} denormal)",
            if self.float16_safe { "SAFE" } else { "UNSAFE" },
            self.float16_overflow_count,
            self.denormal_count
        ));
        lines.push(format!(
            "Int8:    {} ({} overflow)",
            if self.int8_safe { "SAFE" } else { "UNSAFE" },
            self.int8_overflow_count
        ));

        lines.join("\n")
    }

    /// Get layers that are unsafe for float16.
    fn float16_unsafe_layers(&self) -> Vec<LayerQuantization> {
        self.layers
            .iter()
            .filter(|l| {
                matches!(
                    l.float16_safety,
                    QuantSafety::Overflow | QuantSafety::Unknown
                )
            })
            .cloned()
            .collect()
    }

    /// Get layers that are unsafe for int8.
    fn int8_unsafe_layers(&self) -> Vec<LayerQuantization> {
        self.layers
            .iter()
            .filter(|l| matches!(l.int8_safety, QuantSafety::Overflow | QuantSafety::Unknown))
            .cloned()
            .collect()
    }
}

/// Check if model layers can safely be quantized to float16/int8.
///
/// Uses bound propagation to determine the output range of each layer,
/// then checks if those ranges fit within the target format.
///
/// Args:
///     model_path: Path to ONNX model
///     epsilon: Input perturbation size (default: 0.01)
///     check_float16: Check float16 safety (default: True)
///     check_int8: Check int8 safety (default: True)
///
/// Returns:
///     QuantizationResult with per-layer safety analysis
///
/// Example:
///     >>> result = gamma.quantize_check("model.onnx")
///     >>> assert result.float16_safe, "Model has float16 overflow risk"
///     >>> for layer in result.float16_unsafe_layers():
///     ...     print(f"  Unsafe: {layer.name}")
#[pyfunction]
#[pyo3(signature = (model_path, epsilon=0.01, check_float16=true, check_int8=true))]
fn quantize_check(
    py: Python<'_>,
    model_path: &str,
    epsilon: f32,
    check_float16: bool,
    check_int8: bool,
) -> PyResult<QuantizationResult> {
    let mut formats = Vec::new();
    if check_float16 {
        formats.push(QuantFormat::Float16);
    }
    if check_int8 {
        formats.push(QuantFormat::Int8);
    }

    let config = quantize::QuantizeConfig {
        epsilon,
        continue_after_overflow: true,
        input: None,
        formats,
    };

    let result = Python::detach(py, || quantize::analyze_quantization(model_path, &config))
        .map_err(|e| PyValueError::new_err(format!("Quantization error: {}", e)))?;

    // Convert to Python types
    let layers: Vec<LayerQuantization> = result
        .layers
        .into_iter()
        .map(|l| LayerQuantization {
            name: l.name,
            layer_type: l.layer_type,
            min_bound: l.min_bound,
            max_bound: l.max_bound,
            max_abs: l.max_abs,
            output_shape: l.output_shape,
            float16_safety: l.float16_safety.into(),
            int8_safety: l.int8_safety.into(),
            int8_scale: l.int8_scale,
            has_overflow: l.has_overflow,
        })
        .collect();

    Ok(QuantizationResult {
        layers,
        float16_safe: result.float16_safe,
        int8_safe: result.int8_safe,
        float16_overflow_count: result.float16_overflow_count,
        int8_overflow_count: result.int8_overflow_count,
        denormal_count: result.denormal_count,
        input_epsilon: result.input_epsilon,
    })
}

// ============================================================================
// P2 Commands: Bound Width Profiling
// ============================================================================

/// Bound width status indicator.
#[pyclass]
#[derive(Clone)]
pub enum BoundStatus {
    Tight,
    Moderate,
    Wide,
    VeryWide,
    Overflow,
}

#[pymethods]
impl BoundStatus {
    fn __repr__(&self) -> String {
        match self {
            BoundStatus::Tight => "BoundStatus.Tight".to_string(),
            BoundStatus::Moderate => "BoundStatus.Moderate".to_string(),
            BoundStatus::Wide => "BoundStatus.Wide".to_string(),
            BoundStatus::VeryWide => "BoundStatus.VeryWide".to_string(),
            BoundStatus::Overflow => "BoundStatus.Overflow".to_string(),
        }
    }

    fn __str__(&self) -> String {
        match self {
            BoundStatus::Tight => "TIGHT".to_string(),
            BoundStatus::Moderate => "MODERATE".to_string(),
            BoundStatus::Wide => "WIDE".to_string(),
            BoundStatus::VeryWide => "VERY WIDE".to_string(),
            BoundStatus::Overflow => "OVERFLOW".to_string(),
        }
    }
}

impl From<RustBoundStatus> for BoundStatus {
    fn from(s: RustBoundStatus) -> Self {
        match s {
            RustBoundStatus::Tight => BoundStatus::Tight,
            RustBoundStatus::Moderate => BoundStatus::Moderate,
            RustBoundStatus::Wide => BoundStatus::Wide,
            RustBoundStatus::VeryWide => BoundStatus::VeryWide,
            RustBoundStatus::Overflow => BoundStatus::Overflow,
        }
    }
}

/// Result of profiling a single layer's bounds.
#[pyclass]
#[derive(Clone)]
pub struct LayerProfile {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub layer_type: String,
    #[pyo3(get)]
    pub input_width: f32,
    #[pyo3(get)]
    pub output_width: f32,
    #[pyo3(get)]
    pub mean_output_width: f32,
    #[pyo3(get)]
    pub median_output_width: f32,
    #[pyo3(get)]
    pub growth_ratio: f32,
    #[pyo3(get)]
    pub cumulative_expansion: f32,
    #[pyo3(get)]
    pub output_shape: Vec<usize>,
    #[pyo3(get)]
    pub num_elements: usize,
    #[pyo3(get)]
    pub status: BoundStatus,
}

#[pymethods]
impl LayerProfile {
    fn __repr__(&self) -> String {
        format!(
            "LayerProfile(name='{}', growth={:.2}x, status={})",
            self.name,
            self.growth_ratio,
            self.status.__str__()
        )
    }

    /// Check if this layer is a choke point (high growth).
    fn is_choke_point(&self, threshold: f32) -> bool {
        self.growth_ratio > threshold
    }
}

/// Result of a full bound profiling analysis.
#[pyclass]
#[derive(Clone)]
pub struct ProfileResult {
    #[pyo3(get)]
    pub layers: Vec<LayerProfile>,
    #[pyo3(get)]
    pub input_epsilon: f32,
    #[pyo3(get)]
    pub initial_width: f32,
    #[pyo3(get)]
    pub final_width: f32,
    #[pyo3(get)]
    pub total_expansion: f32,
    #[pyo3(get)]
    pub max_growth_layer: Option<usize>,
    #[pyo3(get)]
    pub max_growth_ratio: f32,
    #[pyo3(get)]
    pub overflow_at_layer: Option<usize>,
    #[pyo3(get)]
    pub difficulty_score: f32,
}

#[pymethods]
impl ProfileResult {
    fn __repr__(&self) -> String {
        format!(
            "ProfileResult(layers={}, expansion={:.2}x, difficulty={:.0}/100)",
            self.layers.len(),
            self.total_expansion,
            self.difficulty_score
        )
    }

    /// Get a formatted summary table.
    fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Bound Width Profile".to_string());
        lines.push("===================".to_string());
        lines.push(format!(
            "{:<40} | {:>10} | {:>10} | {:>8} | {:>10} | Status",
            "Layer", "In Width", "Out Width", "Growth", "Cumul."
        ));
        lines.push(format!(
            "{:-<40}-+-{:-<10}-+-{:-<10}-+-{:-<8}-+-{:-<10}-+--------",
            "", "", "", "", ""
        ));

        for (i, layer) in self.layers.iter().enumerate() {
            let is_max = self.max_growth_layer == Some(i);
            let marker = if is_max { " <<<" } else { "" };

            lines.push(format!(
                "{:<40} | {:>10.3e} | {:>10.3e} | {:>8.2}x | {:>10.2}x | {}{}",
                truncate_name(&layer.name, 40),
                layer.input_width,
                layer.output_width,
                layer.growth_ratio,
                layer.cumulative_expansion,
                layer.status.__str__(),
                marker
            ));
        }

        lines.push(String::new());
        lines.push(format!(
            "Initial width: {:.2e} (epsilon = {:.2e})",
            self.initial_width, self.input_epsilon
        ));
        lines.push(format!("Final width: {:.2e}", self.final_width));
        lines.push(format!("Total expansion: {:.2}x", self.total_expansion));
        lines.push(format!(
            "Verification difficulty: {:.0}/100",
            self.difficulty_score
        ));

        lines.join("\n")
    }

    /// Get name of layer with maximum growth.
    #[getter]
    fn max_growth_layer_name(&self) -> Option<String> {
        self.max_growth_layer
            .and_then(|i| self.layers.get(i))
            .map(|l| l.name.clone())
    }

    /// Get choke points (layers with growth above threshold).
    fn choke_points(&self, threshold: f32) -> Vec<LayerProfile> {
        self.layers
            .iter()
            .filter(|l| l.growth_ratio > threshold)
            .cloned()
            .collect()
    }

    /// Get problematic layers (wide or worse bounds).
    fn problematic_layers(&self) -> Vec<LayerProfile> {
        self.layers
            .iter()
            .filter(|l| {
                matches!(
                    l.status,
                    BoundStatus::Wide | BoundStatus::VeryWide | BoundStatus::Overflow
                )
            })
            .cloned()
            .collect()
    }

    /// Check if overflow occurred.
    #[getter]
    fn has_overflow(&self) -> bool {
        self.overflow_at_layer.is_some()
    }
}

/// Profile bound widths through the network.
///
/// Tracks how bound widths grow layer-by-layer, helping identify where
/// verification becomes difficult. Also computes a verification difficulty score.
///
/// Args:
///     model_path: Path to ONNX model
///     epsilon: Input perturbation size (default: 0.01)
///
/// Returns:
///     ProfileResult with per-layer bound analysis
///
/// Example:
///     >>> result = gamma.profile_bounds("model.onnx")
///     >>> print(f"Difficulty: {result.difficulty_score:.0f}/100")
///     >>> for layer in result.choke_points(5.0):
///     ...     print(f"  {layer.name}: {layer.growth_ratio:.2f}x growth")
#[pyfunction]
#[pyo3(signature = (model_path, epsilon=0.01))]
fn profile_bounds(py: Python<'_>, model_path: &str, epsilon: f32) -> PyResult<ProfileResult> {
    let config = profile::ProfileConfig {
        epsilon,
        continue_after_overflow: true,
        input: None,
    };

    let result = Python::detach(py, || profile::profile_bounds(model_path, &config))
        .map_err(|e| PyValueError::new_err(format!("Profile error: {}", e)))?;

    // Convert to Python types
    let layers: Vec<LayerProfile> = result
        .layers
        .into_iter()
        .map(|l| LayerProfile {
            name: l.name,
            layer_type: l.layer_type,
            input_width: l.input_width,
            output_width: l.output_width,
            mean_output_width: l.mean_output_width,
            median_output_width: l.median_output_width,
            growth_ratio: l.growth_ratio,
            cumulative_expansion: l.cumulative_expansion,
            output_shape: l.output_shape,
            num_elements: l.num_elements,
            status: l.status.into(),
        })
        .collect();

    Ok(ProfileResult {
        layers,
        input_epsilon: result.input_epsilon,
        initial_width: result.initial_width,
        final_width: result.final_width,
        total_expansion: result.total_expansion,
        max_growth_layer: result.max_growth_layer,
        max_growth_ratio: result.max_growth_ratio,
        overflow_at_layer: result.overflow_at_layer,
        difficulty_score: result.difficulty_score,
    })
}

// ============================================================================
// Verification API
// ============================================================================

/// A single output bound (lower, upper).
#[pyclass]
#[derive(Clone)]
pub struct OutputBound {
    #[pyo3(get)]
    pub lower: f32,
    #[pyo3(get)]
    pub upper: f32,
}

#[pymethods]
impl OutputBound {
    fn __repr__(&self) -> String {
        format!(
            "OutputBound(lower={:.6}, upper={:.6})",
            self.lower, self.upper
        )
    }

    /// Width of the bound interval.
    #[getter]
    fn width(&self) -> f32 {
        self.upper - self.lower
    }

    /// Midpoint of the bound.
    #[getter]
    fn midpoint(&self) -> f32 {
        (self.lower + self.upper) / 2.0
    }
}

impl From<RustBound> for OutputBound {
    fn from(b: RustBound) -> Self {
        OutputBound {
            lower: b.lower,
            upper: b.upper,
        }
    }
}

/// Status of a verification result.
#[pyclass]
#[derive(Clone)]
pub enum VerifyStatus {
    /// Property verified: all outputs within bounds for all inputs in region.
    Verified,
    /// Property violated: counterexample found.
    Violated,
    /// Verification inconclusive: bounds too loose.
    Unknown,
    /// Verification timed out.
    Timeout,
}

#[pymethods]
impl VerifyStatus {
    fn __repr__(&self) -> String {
        match self {
            VerifyStatus::Verified => "VerifyStatus.Verified".to_string(),
            VerifyStatus::Violated => "VerifyStatus.Violated".to_string(),
            VerifyStatus::Unknown => "VerifyStatus.Unknown".to_string(),
            VerifyStatus::Timeout => "VerifyStatus.Timeout".to_string(),
        }
    }

    fn __str__(&self) -> String {
        match self {
            VerifyStatus::Verified => "VERIFIED".to_string(),
            VerifyStatus::Violated => "VIOLATED".to_string(),
            VerifyStatus::Unknown => "UNKNOWN".to_string(),
            VerifyStatus::Timeout => "TIMEOUT".to_string(),
        }
    }
}

/// Result of neural network verification.
#[pyclass]
#[derive(Clone)]
pub struct VerifyResult {
    #[pyo3(get)]
    pub status: VerifyStatus,

    #[pyo3(get)]
    pub output_bounds: Option<Vec<OutputBound>>,

    #[pyo3(get)]
    pub counterexample: Option<Vec<f32>>,

    #[pyo3(get)]
    pub counterexample_output: Option<Vec<f32>>,

    #[pyo3(get)]
    pub reason: Option<String>,

    #[pyo3(get)]
    pub method: String,

    #[pyo3(get)]
    pub epsilon: f32,
}

#[pymethods]
impl VerifyResult {
    fn __repr__(&self) -> String {
        format!(
            "VerifyResult(status={}, method='{}', epsilon={:.2e})",
            self.status.__str__(),
            self.method,
            self.epsilon
        )
    }

    /// Check if the property was verified.
    #[getter]
    fn is_verified(&self) -> bool {
        matches!(self.status, VerifyStatus::Verified)
    }

    /// Check if a violation was found.
    #[getter]
    fn is_violated(&self) -> bool {
        matches!(self.status, VerifyStatus::Violated)
    }

    /// Get max output bound width (for diagnostics).
    fn max_output_width(&self) -> Option<f32> {
        self.output_bounds.as_ref().map(|bounds| {
            bounds
                .iter()
                .map(|b| b.width())
                .fold(f32::NEG_INFINITY, f32::max)
        })
    }

    /// Get formatted summary.
    fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Verification Result".to_string());
        lines.push("===================".to_string());
        lines.push(format!("Status:  {}", self.status.__str__()));
        lines.push(format!("Method:  {}", self.method));
        lines.push(format!("Epsilon: {:.2e}", self.epsilon));

        if let Some(ref bounds) = self.output_bounds {
            let max_width = bounds.iter().map(|b| b.width()).fold(0.0_f32, f32::max);
            let mean_width: f32 =
                bounds.iter().map(|b| b.width()).sum::<f32>() / bounds.len() as f32;
            lines.push(format!("Outputs: {} bounds", bounds.len()));
            lines.push(format!("Max width:  {:.2e}", max_width));
            lines.push(format!("Mean width: {:.2e}", mean_width));
        }

        if let Some(ref reason) = self.reason {
            lines.push(format!("Reason: {}", reason));
        }

        lines.join("\n")
    }
}

/// Verify a neural network property using bound propagation.
///
/// Uses bound propagation (IBP, CROWN, α-CROWN, or β-CROWN) to compute
/// certified output bounds for all inputs within an epsilon ball.
///
/// Args:
///     model_path: Path to ONNX model
///     epsilon: Input perturbation radius (default: 0.01)
///     method: Verification method - 'ibp', 'crown', 'alpha', or 'beta' (default: 'alpha')
///     timeout: Timeout in seconds (default: 60)
///
/// Returns:
///     VerifyResult with verification status and output bounds
///
/// Example:
///     >>> result = gamma.verify("model.onnx", epsilon=0.01)
///     >>> assert result.is_verified, f"Verification failed: {result.reason}"
///     >>> print(f"Output bounds certified with max width: {result.max_output_width():.2e}")
#[pyfunction]
#[pyo3(signature = (model_path, epsilon=0.01, method="alpha", timeout=60))]
fn verify(
    py: Python<'_>,
    model_path: &str,
    epsilon: f32,
    method: &str,
    timeout: u64,
) -> PyResult<VerifyResult> {
    // Parse method
    let prop_method = match method {
        "ibp" => PropagationMethod::Ibp,
        "crown" => PropagationMethod::Crown,
        "alpha" => PropagationMethod::AlphaCrown,
        "sdp" | "sdp-crown" => PropagationMethod::SdpCrown,
        "beta" => PropagationMethod::BetaCrown,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown method: {}. Use 'ibp', 'crown', 'alpha', 'sdp-crown', or 'beta'",
                method
            )));
        }
    };

    let config = PropagationConfig {
        method: prop_method,
        max_iterations: 100,
        tolerance: 1e-4,
        use_gpu: false,
    };

    // Load and verify (release GIL during computation)
    let result = Python::detach(py, || {
        // Load ONNX model
        let onnx_model = load_onnx(model_path)?;
        let onnx_network = &onnx_model.network;

        // Convert to propagate network
        let prop_network = onnx_model.to_propagate_network()?;

        // Create verifier
        let verifier = Verifier::new(config);

        // Create input shape, handling dynamic dimensions
        let mut input_shape: Vec<usize> = onnx_network
            .inputs
            .first()
            .map(|i| {
                i.shape
                    .iter()
                    .map(|&d| if d < 0 { 16 } else { d as usize })
                    .collect()
            })
            .unwrap_or_else(|| vec![100]);

        // Squeeze leading batch dimension of 1
        if input_shape.len() >= 2 && input_shape[0] == 1 {
            input_shape.remove(0);
        }

        let input_dim: usize = input_shape.iter().product();

        // Get output dimension
        let output_dim = onnx_network
            .outputs
            .first()
            .map(|o| {
                o.shape
                    .iter()
                    .map(|&d| if d < 0 { 16 } else { d })
                    .product::<i64>() as usize
            })
            .unwrap_or(10);

        // Create specification
        let spec = VerificationSpec {
            input_bounds: vec![RustBound::new(-epsilon, epsilon); input_dim],
            output_bounds: vec![RustBound::new(f32::NEG_INFINITY, f32::INFINITY); output_dim],
            timeout_ms: Some(timeout * 1000),
            input_shape: Some(input_shape),
        };

        // Run verification
        verifier.verify(&prop_network, &spec)
    })
    .map_err(|e| PyValueError::new_err(format!("Verification error: {}", e)))?;

    // Convert result to Python types
    let method_str = method.to_string();

    let (status, output_bounds, counterexample, counterexample_output, reason) = match result {
        RustVerificationResult::Verified { output_bounds, .. } => (
            VerifyStatus::Verified,
            Some(output_bounds.into_iter().map(|b| b.into()).collect()),
            None,
            None,
            None,
        ),
        RustVerificationResult::Violated {
            counterexample,
            output,
            details,
        } => {
            // Include violation explanation if available
            let reason = details.as_ref().map(|d| d.explanation.clone());
            (
                VerifyStatus::Violated,
                None,
                Some(counterexample),
                Some(output),
                reason,
            )
        }
        RustVerificationResult::Unknown { bounds, reason } => (
            VerifyStatus::Unknown,
            Some(bounds.into_iter().map(|b| b.into()).collect()),
            None,
            None,
            Some(reason),
        ),
        RustVerificationResult::Timeout { partial_bounds } => (
            VerifyStatus::Timeout,
            partial_bounds.map(|b| b.into_iter().map(|bound| bound.into()).collect()),
            None,
            None,
            Some("Verification timed out".to_string()),
        ),
    };

    Ok(VerifyResult {
        status,
        output_bounds,
        counterexample,
        counterexample_output,
        reason,
        method: method_str,
        epsilon,
    })
}

// ============================================================================
// Model Comparison API
// ============================================================================

/// Result of comparing two models using bound propagation.
#[pyclass]
#[derive(Clone)]
pub struct CompareResult {
    #[pyo3(get)]
    pub is_equivalent: bool,

    #[pyo3(get)]
    pub max_lower_diff: f32,

    #[pyo3(get)]
    pub max_upper_diff: f32,

    #[pyo3(get)]
    pub tolerance: f32,

    #[pyo3(get)]
    pub overlap_pct: f32,

    #[pyo3(get)]
    pub ref_max_width: f32,

    #[pyo3(get)]
    pub target_max_width: f32,

    #[pyo3(get)]
    pub method: String,

    #[pyo3(get)]
    pub epsilon: f32,

    #[pyo3(get)]
    pub violations: Vec<BoundViolation>,
}

#[pymethods]
impl CompareResult {
    fn __repr__(&self) -> String {
        format!(
            "CompareResult(is_equivalent={}, max_lower_diff={:.2e}, max_upper_diff={:.2e})",
            self.is_equivalent, self.max_lower_diff, self.max_upper_diff
        )
    }

    /// Get a formatted summary.
    fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Model Comparison Result".to_string());
        lines.push("=======================".to_string());
        lines.push(format!(
            "Equivalent: {}",
            if self.is_equivalent { "YES" } else { "NO" }
        ));
        lines.push(format!("Method: {}", self.method));
        lines.push(format!("Epsilon: {:.2e}", self.epsilon));
        lines.push(format!("Tolerance: {:.2e}", self.tolerance));
        lines.push(format!("Max lower bound diff: {:.2e}", self.max_lower_diff));
        lines.push(format!("Max upper bound diff: {:.2e}", self.max_upper_diff));
        lines.push(format!("Bound overlap: {:.2}%", self.overlap_pct));
        lines.push(format!("Reference max width: {:.2e}", self.ref_max_width));
        lines.push(format!("Target max width: {:.2e}", self.target_max_width));

        if !self.violations.is_empty() {
            lines.push(format!("\nViolations ({}):", self.violations.len()));
            for v in self.violations.iter().take(10) {
                lines.push(format!(
                    "  [{}] ref=[{:.6}, {:.6}] target=[{:.6}, {:.6}]",
                    v.index, v.ref_lower, v.ref_upper, v.target_lower, v.target_upper
                ));
            }
            if self.violations.len() > 10 {
                lines.push(format!("  ... and {} more", self.violations.len() - 10));
            }
        }

        lines.join("\n")
    }
}

/// A single bound violation between two models.
#[pyclass]
#[derive(Clone)]
pub struct BoundViolation {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub ref_lower: f32,
    #[pyo3(get)]
    pub ref_upper: f32,
    #[pyo3(get)]
    pub target_lower: f32,
    #[pyo3(get)]
    pub target_upper: f32,
    #[pyo3(get)]
    pub lower_diff: f32,
    #[pyo3(get)]
    pub upper_diff: f32,
}

#[pymethods]
impl BoundViolation {
    fn __repr__(&self) -> String {
        format!(
            "BoundViolation(idx={}, lower_diff={:.2e}, upper_diff={:.2e})",
            self.index, self.lower_diff, self.upper_diff
        )
    }
}

/// Compare two models using bound propagation.
///
/// Runs bound propagation on both models with the same input perturbation
/// and compares the resulting output bounds element-wise.
///
/// Args:
///     reference: Path to reference ONNX model
///     target: Path to target ONNX model
///     tolerance: Maximum allowed difference in bounds (default: 0.001)
///     epsilon: Input perturbation radius (default: 0.01)
///     method: Verification method - 'ibp', 'crown', 'alpha' (default: 'crown')
///
/// Returns:
///     CompareResult with comparison results
///
/// Example:
///     >>> result = gamma.compare("model_pytorch.onnx", "model_coreml.onnx")
///     >>> assert result.is_equivalent, f"Bounds differ: max diff = {result.max_lower_diff:.2e}"
#[pyfunction]
#[pyo3(signature = (reference, target, tolerance=0.001, epsilon=0.01, method="crown"))]
fn compare(
    py: Python<'_>,
    reference: &str,
    target: &str,
    tolerance: f32,
    epsilon: f32,
    method: &str,
) -> PyResult<CompareResult> {
    // Validate method
    if !["ibp", "crown", "alpha"].contains(&method) {
        return Err(PyValueError::new_err(format!(
            "Unknown method: {}. Use 'ibp', 'crown', or 'alpha'",
            method
        )));
    }

    let result = Python::detach(py, || {
        // Load both models
        let ref_model = load_onnx(reference)?;
        let target_model = load_onnx(target)?;

        // Convert to propagation networks
        let ref_network = ref_model.to_propagate_network()?;
        let target_network = target_model.to_propagate_network()?;

        // Get input shape from reference model
        let ref_input_shape: Vec<usize> = ref_model
            .network
            .inputs
            .first()
            .map(|i| i.shape.iter().map(|&d| d.max(1) as usize).collect())
            .unwrap_or_else(|| vec![1]);

        // Create bounded input
        let input_data = ndarray::ArrayD::from_elem(ndarray::IxDyn(&ref_input_shape), 0.0f32);
        let input = BoundedTensor::from_epsilon(input_data, epsilon);

        // Run bound propagation on both models
        let ref_output = match method {
            "ibp" => ref_network.propagate_ibp(&input)?,
            "crown" => ref_network.propagate_crown(&input)?,
            "alpha" => ref_network.propagate_alpha_crown(&input)?,
            _ => unreachable!(),
        };

        let target_output = match method {
            "ibp" => target_network.propagate_ibp(&input)?,
            "crown" => target_network.propagate_crown(&input)?,
            "alpha" => target_network.propagate_alpha_crown(&input)?,
            _ => unreachable!(),
        };

        // Compare outputs
        let ref_lower = &ref_output.lower;
        let ref_upper = &ref_output.upper;
        let target_lower = &target_output.lower;
        let target_upper = &target_output.upper;

        let mut max_lower_diff: f32 = 0.0;
        let mut max_upper_diff: f32 = 0.0;
        let mut violations = Vec::new();

        for (idx, (((&rl, &ru), &tl), &tu)) in ref_lower
            .iter()
            .zip(ref_upper.iter())
            .zip(target_lower.iter())
            .zip(target_upper.iter())
            .enumerate()
        {
            let lower_diff = (rl - tl).abs();
            let upper_diff = (ru - tu).abs();

            max_lower_diff = max_lower_diff.max(lower_diff);
            max_upper_diff = max_upper_diff.max(upper_diff);

            if lower_diff > tolerance || upper_diff > tolerance {
                violations.push(BoundViolation {
                    index: idx,
                    ref_lower: rl,
                    ref_upper: ru,
                    target_lower: tl,
                    target_upper: tu,
                    lower_diff,
                    upper_diff,
                });
            }
        }

        // Compute overlap metric
        let mut overlap_count = 0usize;
        let total = ref_lower.len();
        for (((&rl, &ru), &tl), &tu) in ref_lower
            .iter()
            .zip(ref_upper.iter())
            .zip(target_lower.iter())
            .zip(target_upper.iter())
        {
            let overlap = rl.max(tl) <= ru.min(tu);
            if overlap {
                overlap_count += 1;
            }
        }
        let overlap_pct = 100.0 * overlap_count as f32 / total as f32;

        let is_equivalent = max_lower_diff <= tolerance && max_upper_diff <= tolerance;

        Ok(CompareResult {
            is_equivalent,
            max_lower_diff,
            max_upper_diff,
            tolerance,
            overlap_pct,
            ref_max_width: ref_output.max_width(),
            target_max_width: target_output.max_width(),
            method: method.to_string(),
            epsilon,
            violations,
        })
    })
    .map_err(|e: gamma_core::GammaError| PyValueError::new_err(format!("Compare error: {}", e)))?;

    Ok(result)
}

// ============================================================================
// Weights API
// ============================================================================

/// Information about a tensor in a weight file.
#[pyclass]
#[derive(Clone)]
pub struct TensorInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub elements: usize,
}

#[pymethods]
impl TensorInfo {
    fn __repr__(&self) -> String {
        format!(
            "TensorInfo(name='{}', shape={:?}, elements={})",
            self.name, self.shape, self.elements
        )
    }
}

/// Result of weight file inspection.
#[pyclass]
#[derive(Clone)]
pub struct WeightsInfo {
    #[pyo3(get)]
    pub format: String,
    #[pyo3(get)]
    pub tensor_count: usize,
    #[pyo3(get)]
    pub total_params: usize,
    #[pyo3(get)]
    pub tensors: Vec<TensorInfo>,
}

#[pymethods]
impl WeightsInfo {
    fn __repr__(&self) -> String {
        format!(
            "WeightsInfo(format='{}', tensors={}, params={})",
            self.format, self.tensor_count, self.total_params
        )
    }

    /// Get a formatted summary.
    fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Weights Info".to_string());
        lines.push("============".to_string());
        lines.push(format!("Format: {}", self.format));
        lines.push(format!("Tensors: {}", self.tensor_count));
        lines.push(format!(
            "Parameters: {} ({:.2}M)",
            self.total_params,
            self.total_params as f64 / 1e6
        ));
        lines.push("\nTensors:".to_string());
        for t in self.tensors.iter().take(20) {
            lines.push(format!(
                "  {}: {:?} ({} elements)",
                t.name, t.shape, t.elements
            ));
        }
        if self.tensors.len() > 20 {
            lines.push(format!("  ... and {} more", self.tensors.len() - 20));
        }
        lines.join("\n")
    }
}

/// Get information about weights in a file.
///
/// Supports ONNX (.onnx), SafeTensors (.safetensors), PyTorch (.pt, .pth, .bin),
/// GGUF (.gguf), and CoreML (.mlmodel, .mlpackage) formats.
/// Also supports directories containing sharded SafeTensors or HuggingFace PyTorch checkpoints.
///
/// Args:
///     path: Path to weights file
///
/// Returns:
///     WeightsInfo with tensor information
///
/// Example:
///     >>> info = gamma.weights_info("model.safetensors")
///     >>> print(f"Total params: {info.total_params:,}")
///     >>> for t in info.tensors[:5]:
///     ...     print(f"  {t.name}: {t.shape}")
#[pyfunction]
fn weights_info(py: Python<'_>, path: &str) -> PyResult<WeightsInfo> {
    let path = std::path::Path::new(path);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Check if it's an mlpackage directory (no extension check needed)
    let is_mlpackage = path.is_dir()
        && path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.ends_with(".mlpackage"))
            .unwrap_or(false);

    // Check if it's a directory (for sharded SafeTensors, MLX models, etc.)
    let is_directory = path.is_dir() && !is_mlpackage;

    let directory_format = if is_directory {
        // Mirror gamma_onnx::native directory format detection for better UX.
        let config_json = path.join("config.json");
        let has_safetensors = std::fs::read_dir(path)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("safetensors"))
            })
            .unwrap_or(false);

        // Check for sharded PyTorch models (index json or numbered bin files)
        let has_sharded_pytorch = path.join("pytorch_model.bin.index.json").exists()
            || std::fs::read_dir(path)
                .ok()
                .map(|entries| {
                    entries.filter_map(|e| e.ok()).any(|e| {
                        e.file_name()
                            .to_str()
                            .map(|n| n.starts_with("pytorch_model-") && n.ends_with(".bin"))
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false);

        if config_json.exists() && has_safetensors {
            "SafeTensors (sharded)".to_string()
        } else if has_sharded_pytorch {
            "PyTorch (sharded)".to_string()
        } else if path.join("pytorch_model.bin").exists() || path.join("model.pt").exists() {
            "PyTorch (checkpoint)".to_string()
        } else {
            // Best-effort; gamma_onnx::native::load_weights will provide the real error if unsupported.
            "SafeTensors (sharded)".to_string()
        }
    } else {
        String::new()
    };

    let (format, weights) = Python::detach(py, || -> gamma_core::Result<(String, WeightStore)> {
        // Handle mlpackage directories first
            if is_mlpackage {
                #[cfg(feature = "coreml")]
                {
                    let weights = load_coreml(path)?;
                    return Ok(("CoreML".to_string(), weights));
                }
                #[cfg(not(feature = "coreml"))]
                {
                    return Err(gamma_core::GammaError::ModelLoad(
                        "CoreML support not enabled. Rebuild with --features coreml".to_string(),
                    ));
                }
            }

            // Handle directories (sharded SafeTensors, MLX models, HuggingFace checkpoints)
            if is_directory {
                let weights = load_weights(path)?;
                return Ok((directory_format.clone(), weights));
            }

            match ext.as_str() {
                "safetensors" => {
                    let weights = load_safetensors(path)?;
                    Ok(("SafeTensors".to_string(), weights))
                }
                "onnx" => {
                    let model = load_onnx(path)?;
                    Ok(("ONNX".to_string(), model.weights))
                }
                #[cfg(feature = "pytorch")]
                "pt" | "pth" | "bin" => {
                    let weights = load_pytorch(path)?;
                    Ok(("PyTorch".to_string(), weights))
                }
                #[cfg(not(feature = "pytorch"))]
                "pt" | "pth" | "bin" => {
                    Err(gamma_core::GammaError::ModelLoad(
                        "PyTorch support not enabled. Rebuild with --features pytorch".to_string(),
                    ))
                }
                #[cfg(feature = "gguf")]
                "gguf" => {
                    let weights = load_gguf(path)?;
                    Ok(("GGUF".to_string(), weights))
                }
                #[cfg(not(feature = "gguf"))]
                "gguf" => {
                    Err(gamma_core::GammaError::ModelLoad(
                        "GGUF support not enabled. Rebuild with --features gguf".to_string(),
                    ))
                }
                #[cfg(feature = "coreml")]
                "mlmodel" => {
                    let weights = load_coreml(path)?;
                    Ok(("CoreML".to_string(), weights))
                }
                #[cfg(not(feature = "coreml"))]
                "mlmodel" => {
                    Err(gamma_core::GammaError::ModelLoad(
                        "CoreML support not enabled. Rebuild with --features coreml".to_string(),
                    ))
                }
                _ => Err(gamma_core::GammaError::ModelLoad(format!(
                    "Unsupported format: {}. Use .safetensors, .onnx, .pt, .pth, .bin, .gguf, .mlmodel, .mlpackage, or a directory with SafeTensors shards",
                    ext
                ))),
            }
        })
        .map_err(|e| PyValueError::new_err(format!("Load error: {}", e)))?;

    let mut tensors = Vec::new();
    let mut total_params = 0usize;

    for (name, tensor) in weights.iter() {
        let shape: Vec<usize> = tensor.shape().to_vec();
        let elements = shape.iter().product();
        total_params += elements;
        tensors.push(TensorInfo {
            name: name.clone(),
            shape,
            elements,
        });
    }

    // Sort by name for consistent output
    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(WeightsInfo {
        format,
        tensor_count: tensors.len(),
        total_params,
        tensors,
    })
}

/// Result of comparing a single tensor between two weight files.
#[pyclass]
#[derive(Clone)]
pub struct TensorComparison {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub max_diff: Option<f32>,
    #[pyo3(get)]
    pub shape_a: Option<Vec<usize>>,
    #[pyo3(get)]
    pub shape_b: Option<Vec<usize>>,
}

#[pymethods]
impl TensorComparison {
    fn __repr__(&self) -> String {
        match &self.max_diff {
            Some(diff) => format!(
                "TensorComparison(name='{}', status='{}', max_diff={:.2e})",
                self.name, self.status, diff
            ),
            None => format!(
                "TensorComparison(name='{}', status='{}')",
                self.name, self.status
            ),
        }
    }
}

/// Result of comparing two weight files.
#[pyclass]
#[derive(Clone)]
pub struct WeightsDiffResult {
    #[pyo3(get)]
    pub is_match: bool,
    #[pyo3(get)]
    pub max_diff: f32,
    #[pyo3(get)]
    pub tolerance: f32,
    #[pyo3(get)]
    pub differing_count: usize,
    #[pyo3(get)]
    pub total_tensors_a: usize,
    #[pyo3(get)]
    pub total_tensors_b: usize,
    #[pyo3(get)]
    pub comparisons: Vec<TensorComparison>,
}

#[pymethods]
impl WeightsDiffResult {
    fn __repr__(&self) -> String {
        format!(
            "WeightsDiffResult(is_match={}, max_diff={:.2e}, differing={})",
            self.is_match, self.max_diff, self.differing_count
        )
    }

    /// Get a formatted summary.
    fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Weights Diff Result".to_string());
        lines.push("===================".to_string());
        lines.push(format!(
            "Result: {}",
            if self.is_match { "MATCH" } else { "DIFFERS" }
        ));
        lines.push(format!("Max difference: {:.6e}", self.max_diff));
        lines.push(format!("Tolerance: {:.6e}", self.tolerance));
        lines.push(format!("Differing tensors: {}", self.differing_count));
        lines.push(format!("Tensors in A: {}", self.total_tensors_a));
        lines.push(format!("Tensors in B: {}", self.total_tensors_b));

        if self.differing_count > 0 {
            lines.push("\nDifferences:".to_string());
            for c in self
                .comparisons
                .iter()
                .filter(|c| c.status != "match")
                .take(20)
            {
                match &c.max_diff {
                    Some(diff) => {
                        lines.push(format!("  {}: {} (diff={:.2e})", c.name, c.status, diff))
                    }
                    None => lines.push(format!("  {}: {}", c.name, c.status)),
                }
            }
        }

        lines.join("\n")
    }

    /// Get matching tensors.
    fn matching_tensors(&self) -> Vec<TensorComparison> {
        self.comparisons
            .iter()
            .filter(|c| c.status == "match")
            .cloned()
            .collect()
    }

    /// Get differing tensors.
    fn differing_tensors(&self) -> Vec<TensorComparison> {
        self.comparisons
            .iter()
            .filter(|c| c.status != "match")
            .cloned()
            .collect()
    }
}

/// Load weights from a file or directory (helper for weights_diff).
///
/// Supports single files (.safetensors, .onnx, .pt, .pth, .bin, .gguf, .mlmodel, .mlpackage)
/// and directories containing sharded SafeTensors files, MLX models, or HuggingFace PyTorch checkpoints.
fn load_weights_from_file(path: &str) -> gamma_core::Result<WeightStore> {
    let path = std::path::Path::new(path);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Check if it's an mlpackage directory
    let is_mlpackage = path.is_dir()
        && path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.ends_with(".mlpackage"))
            .unwrap_or(false);

    // Check if it's a directory (for sharded SafeTensors, MLX models, etc.)
    let is_directory = path.is_dir() && !is_mlpackage;

    if is_mlpackage {
        #[cfg(feature = "coreml")]
        {
            return load_coreml(path);
        }
        #[cfg(not(feature = "coreml"))]
        {
            return Err(gamma_core::GammaError::ModelLoad(
                "CoreML support not enabled. Rebuild with --features coreml".to_string(),
            ));
        }
    }

    // Handle directories (sharded SafeTensors, MLX models, HuggingFace checkpoints)
    if is_directory {
        return load_weights(path);
    }

    match ext.as_str() {
        "safetensors" => load_safetensors(path),
        "onnx" => {
            let model = load_onnx(path)?;
            Ok(model.weights)
        }
        #[cfg(feature = "pytorch")]
        "pt" | "pth" | "bin" => load_pytorch(path),
        #[cfg(not(feature = "pytorch"))]
        "pt" | "pth" | "bin" => Err(gamma_core::GammaError::ModelLoad(
            "PyTorch support not enabled. Rebuild with --features pytorch".to_string(),
        )),
        #[cfg(feature = "gguf")]
        "gguf" => load_gguf(path),
        #[cfg(not(feature = "gguf"))]
        "gguf" => Err(gamma_core::GammaError::ModelLoad(
            "GGUF support not enabled. Rebuild with --features gguf".to_string(),
        )),
        #[cfg(feature = "coreml")]
        "mlmodel" => load_coreml(path),
        #[cfg(not(feature = "coreml"))]
        "mlmodel" => Err(gamma_core::GammaError::ModelLoad(
            "CoreML support not enabled. Rebuild with --features coreml".to_string(),
        )),
        _ => Err(gamma_core::GammaError::ModelLoad(format!(
            "Unsupported format: {}. Use .safetensors, .onnx, .pt, .pth, .bin, .gguf, .mlmodel, .mlpackage, or a directory with SafeTensors/PyTorch shards",
            ext
        ))),
    }
}

/// Compare weights between two files.
///
/// Supports ONNX (.onnx), SafeTensors (.safetensors), PyTorch (.pt, .pth, .bin),
/// GGUF (.gguf), and CoreML (.mlmodel, .mlpackage) formats.
///
/// Args:
///     file_a: Path to first weights file
///     file_b: Path to second weights file
///     tolerance: Maximum allowed absolute difference (default: 1e-6)
///
/// Returns:
///     WeightsDiffResult with comparison results
///
/// Example:
///     >>> result = gamma.weights_diff("model_a.safetensors", "model_b.safetensors")
///     >>> assert result.is_match, f"Max diff: {result.max_diff:.2e}"
///     >>> for diff in result.differing_tensors():
///     ...     print(f"  {diff.name}: {diff.status}")
#[pyfunction]
#[pyo3(signature = (file_a, file_b, tolerance=1e-6))]
fn weights_diff(
    py: Python<'_>,
    file_a: &str,
    file_b: &str,
    tolerance: f32,
) -> PyResult<WeightsDiffResult> {
    let result = Python::detach(py, || -> gamma_core::Result<WeightsDiffResult> {
        let weights_a = load_weights_from_file(file_a)?;
        let weights_b = load_weights_from_file(file_b)?;

        let mut comparisons = Vec::new();
        let mut max_diff = 0.0f32;
        let mut differing_count = 0;

        // Compare tensors in A
        for (name, tensor_a) in weights_a.iter() {
            if let Some(tensor_b) = weights_b.get(name) {
                // Compare shapes
                if tensor_a.shape() != tensor_b.shape() {
                    comparisons.push(TensorComparison {
                        name: name.clone(),
                        status: "shape_mismatch".to_string(),
                        max_diff: None,
                        shape_a: Some(tensor_a.shape().to_vec()),
                        shape_b: Some(tensor_b.shape().to_vec()),
                    });
                    differing_count += 1;
                    continue;
                }

                // Compare values
                let diff = tensor_a
                    .iter()
                    .zip(tensor_b.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);

                max_diff = max_diff.max(diff);

                if diff > tolerance {
                    differing_count += 1;
                    comparisons.push(TensorComparison {
                        name: name.clone(),
                        status: "differs".to_string(),
                        max_diff: Some(diff),
                        shape_a: Some(tensor_a.shape().to_vec()),
                        shape_b: Some(tensor_b.shape().to_vec()),
                    });
                } else {
                    comparisons.push(TensorComparison {
                        name: name.clone(),
                        status: "match".to_string(),
                        max_diff: Some(diff),
                        shape_a: Some(tensor_a.shape().to_vec()),
                        shape_b: Some(tensor_b.shape().to_vec()),
                    });
                }
            } else {
                comparisons.push(TensorComparison {
                    name: name.clone(),
                    status: "missing_in_b".to_string(),
                    max_diff: None,
                    shape_a: Some(tensor_a.shape().to_vec()),
                    shape_b: None,
                });
                differing_count += 1;
            }
        }

        // Check for tensors only in B
        for name in weights_b.keys() {
            if weights_a.get(name).is_none() {
                comparisons.push(TensorComparison {
                    name: name.clone(),
                    status: "missing_in_a".to_string(),
                    max_diff: None,
                    shape_a: None,
                    shape_b: weights_b.get(name).map(|t| t.shape().to_vec()),
                });
                differing_count += 1;
            }
        }

        let is_match = differing_count == 0;

        Ok(WeightsDiffResult {
            is_match,
            max_diff,
            tolerance,
            differing_count,
            total_tensors_a: weights_a.len(),
            total_tensors_b: weights_b.len(),
            comparisons,
        })
    })
    .map_err(|e| PyValueError::new_err(format!("Weights diff error: {}", e)))?;

    Ok(result)
}

// ============================================================================
// Benchmark API
// ============================================================================

/// Single benchmark result item.
#[pyclass]
#[derive(Clone)]
pub struct BenchResultItem {
    /// Name of the benchmark
    #[pyo3(get)]
    pub name: String,

    /// Number of iterations run
    #[pyo3(get)]
    pub iterations: usize,

    /// Time per iteration in nanoseconds
    #[pyo3(get)]
    pub per_iter_ns: u64,

    /// Time per iteration in microseconds
    #[pyo3(get)]
    pub per_iter_us: f64,

    /// Time per iteration in milliseconds
    #[pyo3(get)]
    pub per_iter_ms: f64,

    /// Total time in nanoseconds
    #[pyo3(get)]
    pub total_ns: u64,

    /// Total time in milliseconds
    #[pyo3(get)]
    pub total_ms: f64,
}

#[pymethods]
impl BenchResultItem {
    fn __repr__(&self) -> String {
        format!(
            "BenchResultItem(name='{}', per_iter_ms={:.3}, iterations={})",
            self.name, self.per_iter_ms, self.iterations
        )
    }
}

/// Dimensions used for benchmarks.
#[pyclass]
#[derive(Clone)]
pub struct BenchDimensions {
    /// Batch size
    #[pyo3(get)]
    pub batch: usize,

    /// Sequence length
    #[pyo3(get)]
    pub seq_len: usize,

    /// Hidden dimension
    #[pyo3(get)]
    pub hidden_dim: usize,

    /// Intermediate (feedforward) dimension
    #[pyo3(get)]
    pub intermediate_dim: usize,

    /// Number of attention heads
    #[pyo3(get)]
    pub num_heads: usize,

    /// Dimension per head
    #[pyo3(get)]
    pub head_dim: usize,

    /// Epsilon perturbation used
    #[pyo3(get)]
    pub epsilon: f32,
}

#[pymethods]
impl BenchDimensions {
    fn __repr__(&self) -> String {
        format!(
            "BenchDimensions(batch={}, seq={}, hidden={}, intermediate={}, heads={}, head_dim={}, eps={:.2e})",
            self.batch, self.seq_len, self.hidden_dim, self.intermediate_dim,
            self.num_heads, self.head_dim, self.epsilon
        )
    }
}

/// Full benchmark result.
#[pyclass]
#[derive(Clone)]
pub struct BenchResult {
    /// Type of benchmark (layer, attention, full)
    #[pyo3(get)]
    pub benchmark_type: String,

    /// Whether the benchmark type was valid
    #[pyo3(get)]
    pub valid_type: bool,

    /// Dimensions used for the benchmark
    #[pyo3(get)]
    pub dimensions: BenchDimensions,

    /// Individual benchmark results
    #[pyo3(get)]
    pub results: Vec<BenchResultItem>,
}

#[pymethods]
impl BenchResult {
    fn __repr__(&self) -> String {
        format!(
            "BenchResult(type='{}', valid={}, results={})",
            self.benchmark_type,
            self.valid_type,
            self.results.len()
        )
    }

    /// Get a summary of all benchmark results
    fn summary(&self) -> String {
        let mut lines = vec![
            format!("Benchmark: {}", self.benchmark_type),
            format!("Dimensions: {}", self.dimensions.__repr__()),
            "Results:".to_string(),
        ];
        for r in &self.results {
            lines.push(format!(
                "  {}: {:.3}ms/iter ({} iters)",
                r.name, r.per_iter_ms, r.iterations
            ));
        }
        lines.join("\n")
    }
}

/// Helper to create BoundedTensor for benchmarks
fn make_bench_input(shape: &[usize], center: f32, epsilon: f32) -> BoundedTensor {
    let values = ndarray::ArrayD::from_elem(NdIxDyn(shape), center);
    BoundedTensor::from_epsilon(values, epsilon)
}

/// Helper to run a benchmark with warmup
fn run_bench<F: FnMut()>(name: &str, iterations: usize, mut f: F) -> BenchResultItem {
    // Warmup
    for _ in 0..3 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter_ns = (elapsed.as_nanos() / iterations as u128) as u64;
    let total_ns = elapsed.as_nanos() as u64;

    BenchResultItem {
        name: name.to_string(),
        iterations,
        per_iter_ns,
        per_iter_us: per_iter_ns as f64 / 1000.0,
        per_iter_ms: per_iter_ns as f64 / 1_000_000.0,
        total_ns,
        total_ms: total_ns as f64 / 1_000_000.0,
    }
}

/// Run γ-CROWN benchmarks.
///
/// Runs performance benchmarks for neural network verification operations.
///
/// Args:
///     benchmark_type: Type of benchmark to run. Options:
///         - "layer" (default): Individual layer IBP performance
///         - "attention": Attention component (MatMul, Softmax) performance
///         - "full": Full pipeline scaling tests
///
/// Returns:
///     BenchResult with timing information for each benchmark
///
/// Example:
///     >>> result = gamma.bench()  # Run layer benchmarks
///     >>> print(result.summary())
///     >>> for r in result.results:
///     ...     print(f"{r.name}: {r.per_iter_ms:.3f}ms")
///
///     >>> result = gamma.bench("attention")  # Run attention benchmarks
///     >>> result = gamma.bench("full")  # Run full pipeline benchmarks
#[pyfunction]
#[pyo3(name = "bench", signature = (benchmark_type="layer"))]
fn run_benchmark(py: Python<'_>, benchmark_type: &str) -> PyResult<BenchResult> {
    // Whisper-tiny dimensions
    let batch = 1;
    let seq_len = 16;
    let hidden_dim = 384;
    let intermediate_dim = 1536;
    let num_heads = 6;
    let head_dim = 64;
    let epsilon = 0.01_f32;

    let dimensions = BenchDimensions {
        batch,
        seq_len,
        hidden_dim,
        intermediate_dim,
        num_heads,
        head_dim,
        epsilon,
    };

    let result = Python::detach(py, || -> gamma_core::Result<BenchResult> {
        let mut results: Vec<BenchResultItem> = Vec::new();
        let mut valid_type = true;

        // Create common layers
        let linear_weight = Array2::from_shape_fn((intermediate_dim, hidden_dim), |_| 0.01_f32);
        let linear_bias = Some(Array1::zeros(intermediate_dim));
        let linear1 = LinearLayer::new(linear_weight.clone(), linear_bias.clone())?;

        let linear_weight2 = Array2::from_shape_fn((hidden_dim, intermediate_dim), |_| 0.01_f32);
        let linear_bias2 = Some(Array1::zeros(hidden_dim));
        let linear2 = LinearLayer::new(linear_weight2, linear_bias2)?;

        let gelu = GELULayer::default();
        let layernorm =
            LayerNormLayer::new(Array1::ones(hidden_dim), Array1::zeros(hidden_dim), 1e-5);

        match benchmark_type {
            "layer" => {
                let input = make_bench_input(&[batch, seq_len, hidden_dim], 0.5, epsilon);

                // Linear layer
                let mut linear_output = input.clone();
                results.push(run_bench("Linear IBP [384->1536]", 100, || {
                    linear_output = linear1.propagate_ibp(&input).unwrap();
                }));

                // GELU
                results.push(run_bench("GELU IBP [1536]", 100, || {
                    let _ = gelu.propagate_ibp(&linear_output);
                }));
                let gelu_output = gelu.propagate_ibp(&linear_output)?;

                // Linear back
                results.push(run_bench("Linear IBP [1536->384]", 100, || {
                    let _ = linear2.propagate_ibp(&gelu_output);
                }));
                let final_output = linear2.propagate_ibp(&gelu_output)?;

                // LayerNorm
                results.push(run_bench("LayerNorm IBP [384]", 100, || {
                    let _ = layernorm.propagate_ibp(&final_output);
                }));

                // Full MLP
                let mut mlp = Network::new();
                mlp.add_layer(Layer::Linear(linear1.clone()));
                mlp.add_layer(Layer::GELU(gelu.clone()));
                mlp.add_layer(Layer::Linear(linear2.clone()));

                results.push(run_bench("Full MLP IBP [384->1536->384]", 100, || {
                    let _ = mlp.propagate_ibp(&input);
                }));
            }

            "attention" => {
                // MatMul: Q @ K^T
                let q_input = make_bench_input(&[batch, num_heads, seq_len, head_dim], 0.5, 0.1);
                let k_input = make_bench_input(&[batch, num_heads, head_dim, seq_len], 0.5, 0.1);

                let matmul = MatMulLayer::new(false, None);

                results.push(run_bench(
                    &format!(
                        "MatMul IBP [{},{},{},{}] @ [{},{},{},{}]",
                        batch, num_heads, seq_len, head_dim, batch, num_heads, head_dim, seq_len
                    ),
                    100,
                    || {
                        let _ = matmul.propagate_ibp_binary(&q_input, &k_input);
                    },
                ));

                // Softmax
                let attn_input = make_bench_input(&[batch, num_heads, seq_len, seq_len], 0.0, 1.0);
                let softmax = SoftmaxLayer::new(-1);

                results.push(run_bench(
                    &format!(
                        "Softmax IBP [{},{},{},{}]",
                        batch, num_heads, seq_len, seq_len
                    ),
                    100,
                    || {
                        let _ = softmax.propagate_ibp(&attn_input);
                    },
                ));

                // MatMul scaling
                for seq in [4, 16, 64] {
                    let q = make_bench_input(&[batch, num_heads, seq, head_dim], 0.5, 0.1);
                    let k = make_bench_input(&[batch, num_heads, head_dim, seq], 0.5, 0.1);
                    let iterations = if seq <= 16 { 100 } else { 20 };

                    results.push(run_bench(
                        &format!("MatMul IBP seq={}", seq),
                        iterations,
                        || {
                            let _ = matmul.propagate_ibp_binary(&q, &k);
                        },
                    ));
                }
            }

            "full" => {
                let mut mlp = Network::new();
                mlp.add_layer(Layer::Linear(linear1.clone()));
                mlp.add_layer(Layer::GELU(gelu.clone()));
                mlp.add_layer(Layer::Linear(linear2.clone()));

                // IBP scaling
                for seq in [4, 16, 64, 128] {
                    let input = make_bench_input(&[batch, seq, hidden_dim], 0.5, epsilon);
                    let iterations = if seq <= 16 {
                        100
                    } else if seq <= 64 {
                        20
                    } else {
                        5
                    };

                    results.push(run_bench(
                        &format!("MLP IBP seq={}", seq),
                        iterations,
                        || {
                            let _ = mlp.propagate_ibp(&input);
                        },
                    ));
                }

                // CROWN 1-D
                let input_1d = make_bench_input(&[hidden_dim], 0.5, epsilon);
                results.push(run_bench("Full MLP CROWN 1-D [384]", 100, || {
                    let _ = mlp.propagate_crown(&input_1d);
                }));
            }

            _ => {
                valid_type = false;
            }
        }

        Ok(BenchResult {
            benchmark_type: benchmark_type.to_string(),
            valid_type,
            dimensions,
            results,
        })
    })
    .map_err(|e| PyValueError::new_err(format!("Benchmark error: {}", e)))?;

    Ok(result)
}

/// γ-CROWN Python module.
///
/// Neural network verification and testing library.
#[pymodule]
fn gamma(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // P0: diff
    m.add_class::<LayerComparison>()?;
    m.add_class::<DiffResult>()?;
    m.add_class::<DiffStatus>()?;
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(run_with_intermediates, m)?)?;
    m.add_function(wrap_pyfunction!(load_model_info, m)?)?;
    m.add_function(wrap_pyfunction!(load_npy, m)?)?;

    // P2: sensitivity
    m.add_class::<LayerSensitivity>()?;
    m.add_class::<SensitivityResult>()?;
    m.add_function(wrap_pyfunction!(sensitivity_analysis, m)?)?;

    // P2: quantization
    m.add_class::<QuantSafety>()?;
    m.add_class::<LayerQuantization>()?;
    m.add_class::<QuantizationResult>()?;
    m.add_function(wrap_pyfunction!(quantize_check, m)?)?;

    // P2: profiling
    m.add_class::<BoundStatus>()?;
    m.add_class::<LayerProfile>()?;
    m.add_class::<ProfileResult>()?;
    m.add_function(wrap_pyfunction!(profile_bounds, m)?)?;

    // Verification API
    m.add_class::<OutputBound>()?;
    m.add_class::<VerifyStatus>()?;
    m.add_class::<VerifyResult>()?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;

    // Comparison API
    m.add_class::<CompareResult>()?;
    m.add_class::<BoundViolation>()?;
    m.add_function(wrap_pyfunction!(compare, m)?)?;

    // Weights API
    m.add_class::<TensorInfo>()?;
    m.add_class::<WeightsInfo>()?;
    m.add_class::<TensorComparison>()?;
    m.add_class::<WeightsDiffResult>()?;
    m.add_function(wrap_pyfunction!(weights_info, m)?)?;
    m.add_function(wrap_pyfunction!(weights_diff, m)?)?;

    // Benchmark API
    m.add_class::<BenchResultItem>()?;
    m.add_class::<BenchDimensions>()?;
    m.add_class::<BenchResult>()?;
    m.add_function(wrap_pyfunction!(run_benchmark, m)?)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_truncate_name_short() {
        assert_eq!(truncate_name("short", 10), "short");
    }

    #[test]
    fn test_truncate_name_exact() {
        assert_eq!(truncate_name("exactly10!", 10), "exactly10!");
    }

    #[test]
    fn test_truncate_name_long() {
        let result = truncate_name("this_is_a_very_long_layer_name", 15);
        assert!(result.starts_with("..."));
        assert_eq!(result.len(), 15);
        assert!(result.ends_with("layer_name"));
    }

    // =========================================================================
    // DiffStatus Tests
    // =========================================================================

    #[test]
    fn test_diff_status_repr() {
        assert_eq!(DiffStatus::Ok.__repr__(), "DiffStatus.Ok");
        assert_eq!(DiffStatus::DriftStarts.__repr__(), "DiffStatus.DriftStarts");
        assert_eq!(
            DiffStatus::ExceedsTolerance.__repr__(),
            "DiffStatus.ExceedsTolerance"
        );
        assert_eq!(
            DiffStatus::ShapeMismatch.__repr__(),
            "DiffStatus.ShapeMismatch"
        );
    }

    // =========================================================================
    // LayerComparison Tests
    // =========================================================================

    #[test]
    fn test_layer_comparison_creation() {
        let lc = LayerComparison {
            name: "layer1".to_string(),
            name_b: Some("layer1_b".to_string()),
            max_diff: 0.001,
            mean_diff: 0.0005,
            exceeds_tolerance: false,
            shape_a: vec![1, 64, 128],
            shape_b: vec![1, 64, 128],
        };
        assert_eq!(lc.name, "layer1");
        assert_eq!(lc.name_b, Some("layer1_b".to_string()));
        assert!(!lc.exceeds_tolerance);
    }

    #[test]
    fn test_layer_comparison_repr() {
        let lc = LayerComparison {
            name: "test_layer".to_string(),
            name_b: None,
            max_diff: 1e-4,
            mean_diff: 5e-5,
            exceeds_tolerance: true,
            shape_a: vec![1, 64],
            shape_b: vec![1, 64],
        };
        let repr = lc.__repr__();
        assert!(repr.contains("test_layer"));
        assert!(repr.contains("exceeds=true"));
    }

    // =========================================================================
    // DiffResult Tests
    // =========================================================================

    fn make_diff_result(first_bad_layer: Option<usize>) -> DiffResult {
        let layers = vec![
            LayerComparison {
                name: "layer0".to_string(),
                name_b: None,
                max_diff: 1e-6,
                mean_diff: 5e-7,
                exceeds_tolerance: false,
                shape_a: vec![1, 64],
                shape_b: vec![1, 64],
            },
            LayerComparison {
                name: "layer1".to_string(),
                name_b: None,
                max_diff: 1e-3,
                mean_diff: 5e-4,
                exceeds_tolerance: first_bad_layer == Some(1),
                shape_a: vec![1, 64],
                shape_b: vec![1, 64],
            },
        ];
        DiffResult {
            layers,
            first_bad_layer,
            drift_start_layer: None,
            max_divergence: 1e-3,
            tolerance: 1e-4,
            suggestion: None,
        }
    }

    #[test]
    fn test_diff_result_is_equivalent_true() {
        let result = make_diff_result(None);
        assert!(result.is_equivalent());
    }

    #[test]
    fn test_diff_result_is_equivalent_false() {
        let result = make_diff_result(Some(1));
        assert!(!result.is_equivalent());
    }

    #[test]
    fn test_diff_result_first_bad_layer_name() {
        let result = make_diff_result(Some(1));
        assert_eq!(result.first_bad_layer_name(), Some("layer1".to_string()));

        let result_none = make_diff_result(None);
        assert_eq!(result_none.first_bad_layer_name(), None);
    }

    #[test]
    fn test_diff_result_statuses() {
        let mut result = make_diff_result(Some(1));
        result.layers[1].exceeds_tolerance = true;

        let statuses = result.statuses();
        assert_eq!(statuses.len(), 2);
        assert!(matches!(statuses[0], DiffStatus::Ok));
        assert!(matches!(statuses[1], DiffStatus::ExceedsTolerance));
    }

    #[test]
    fn test_diff_result_statuses_with_drift() {
        let mut result = make_diff_result(None);
        result.drift_start_layer = Some(0);

        let statuses = result.statuses();
        assert!(matches!(statuses[0], DiffStatus::DriftStarts));
    }

    #[test]
    fn test_diff_result_statuses_shape_mismatch() {
        let mut result = make_diff_result(None);
        result.layers[0].shape_b = vec![1, 32]; // Different shape

        let statuses = result.statuses();
        assert!(matches!(statuses[0], DiffStatus::ShapeMismatch));
    }

    #[test]
    fn test_diff_result_repr() {
        let result = make_diff_result(None);
        let repr = result.__repr__();
        assert!(repr.contains("DiffResult"));
        assert!(repr.contains("layers=2"));
        assert!(repr.contains("is_equivalent=true"));
    }

    #[test]
    fn test_diff_result_summary() {
        let result = make_diff_result(Some(1));
        let summary = result.summary();
        assert!(summary.contains("Layer-by-Layer Comparison"));
        assert!(summary.contains("layer0"));
        assert!(summary.contains("layer1"));
    }

    // =========================================================================
    // LayerSensitivity Tests
    // =========================================================================

    fn make_layer_sensitivity(sensitivity: f32) -> LayerSensitivity {
        LayerSensitivity {
            name: "test_layer".to_string(),
            layer_type: "Linear".to_string(),
            input_width: 0.1,
            output_width: sensitivity * 0.1,
            sensitivity,
            mean_output_width: sensitivity * 0.1,
            output_shape: vec![1, 64],
            has_overflow: false,
        }
    }

    #[test]
    fn test_layer_sensitivity_is_high_sensitivity() {
        let layer = make_layer_sensitivity(15.0);
        assert!(layer.is_high_sensitivity(10.0));
        assert!(!layer.is_high_sensitivity(20.0));
    }

    #[test]
    fn test_layer_sensitivity_is_contractive() {
        let contractive = make_layer_sensitivity(0.5);
        assert!(contractive.is_contractive());

        let expansive = make_layer_sensitivity(2.0);
        assert!(!expansive.is_contractive());
    }

    #[test]
    fn test_layer_sensitivity_repr() {
        let layer = make_layer_sensitivity(5.0);
        let repr = layer.__repr__();
        assert!(repr.contains("test_layer"));
        assert!(repr.contains("5.00"));
    }

    // =========================================================================
    // SensitivityResult Tests
    // =========================================================================

    fn make_sensitivity_result() -> SensitivityResult {
        SensitivityResult {
            layers: vec![
                make_layer_sensitivity(2.0),
                make_layer_sensitivity(15.0),
                make_layer_sensitivity(0.8),
            ],
            total_sensitivity: 24.0,
            max_sensitivity: 15.0,
            max_sensitivity_layer: Some(1),
            input_epsilon: 0.01,
            final_width: 0.24,
            overflow_at_layer: None,
        }
    }

    #[test]
    fn test_sensitivity_result_hot_spots() {
        let result = make_sensitivity_result();
        let hot_spots = result.hot_spots(10.0);
        assert_eq!(hot_spots.len(), 1);
        assert_eq!(hot_spots[0].sensitivity, 15.0);
    }

    #[test]
    fn test_sensitivity_result_has_overflow() {
        let result = make_sensitivity_result();
        assert!(!result.has_overflow());

        let mut overflow_result = make_sensitivity_result();
        overflow_result.overflow_at_layer = Some(1);
        assert!(overflow_result.has_overflow());
    }

    #[test]
    fn test_sensitivity_result_max_layer_name() {
        let result = make_sensitivity_result();
        // max_sensitivity_layer is Some(1), but all layers have same name in test
        assert!(result.max_sensitivity_layer_name().is_some());
    }

    #[test]
    fn test_sensitivity_result_repr() {
        let result = make_sensitivity_result();
        let repr = result.__repr__();
        assert!(repr.contains("SensitivityResult"));
        assert!(repr.contains("max_sensitivity=15.00"));
    }

    // =========================================================================
    // QuantSafety Tests
    // =========================================================================

    #[test]
    fn test_quant_safety_repr() {
        assert_eq!(QuantSafety::Safe.__repr__(), "QuantSafety.Safe");
        assert_eq!(QuantSafety::Denormal.__repr__(), "QuantSafety.Denormal");
        assert_eq!(
            QuantSafety::ScalingRequired.__repr__(),
            "QuantSafety.ScalingRequired"
        );
        assert_eq!(QuantSafety::Overflow.__repr__(), "QuantSafety.Overflow");
        assert_eq!(QuantSafety::Unknown.__repr__(), "QuantSafety.Unknown");
    }

    #[test]
    fn test_quant_safety_str() {
        assert_eq!(QuantSafety::Safe.__str__(), "SAFE");
        assert_eq!(QuantSafety::Denormal.__str__(), "DENORMAL");
        assert_eq!(QuantSafety::ScalingRequired.__str__(), "SCALE");
        assert_eq!(QuantSafety::Overflow.__str__(), "OVERFLOW");
        assert_eq!(QuantSafety::Unknown.__str__(), "UNKNOWN");
    }

    // =========================================================================
    // LayerQuantization Tests
    // =========================================================================

    fn make_layer_quantization(f16: QuantSafety, i8: QuantSafety) -> LayerQuantization {
        LayerQuantization {
            name: "quant_layer".to_string(),
            layer_type: "Linear".to_string(),
            min_bound: -10.0,
            max_bound: 10.0,
            max_abs: 10.0,
            output_shape: vec![1, 64],
            float16_safety: f16,
            int8_safety: i8,
            int8_scale: Some(0.1),
            has_overflow: false,
        }
    }

    #[test]
    fn test_layer_quantization_is_float16_safe() {
        let safe = make_layer_quantization(QuantSafety::Safe, QuantSafety::Safe);
        assert!(safe.is_float16_safe());

        let unsafe_layer = make_layer_quantization(QuantSafety::Overflow, QuantSafety::Safe);
        assert!(!unsafe_layer.is_float16_safe());
    }

    #[test]
    fn test_layer_quantization_is_int8_safe() {
        let safe = make_layer_quantization(QuantSafety::Safe, QuantSafety::Safe);
        assert!(safe.is_int8_safe());

        let scaling = make_layer_quantization(QuantSafety::Safe, QuantSafety::ScalingRequired);
        assert!(scaling.is_int8_safe());

        let unsafe_layer = make_layer_quantization(QuantSafety::Safe, QuantSafety::Overflow);
        assert!(!unsafe_layer.is_int8_safe());
    }

    #[test]
    fn test_layer_quantization_repr() {
        let layer = make_layer_quantization(QuantSafety::Safe, QuantSafety::ScalingRequired);
        let repr = layer.__repr__();
        assert!(repr.contains("quant_layer"));
        assert!(repr.contains("SAFE"));
        assert!(repr.contains("SCALE"));
    }

    // =========================================================================
    // QuantizationResult Tests
    // =========================================================================

    fn make_quantization_result() -> QuantizationResult {
        QuantizationResult {
            layers: vec![
                make_layer_quantization(QuantSafety::Safe, QuantSafety::Safe),
                make_layer_quantization(QuantSafety::Overflow, QuantSafety::Safe),
                make_layer_quantization(QuantSafety::Safe, QuantSafety::Overflow),
            ],
            float16_safe: false,
            int8_safe: false,
            float16_overflow_count: 1,
            int8_overflow_count: 1,
            denormal_count: 0,
            input_epsilon: 0.01,
        }
    }

    #[test]
    fn test_quantization_result_float16_unsafe_layers() {
        let result = make_quantization_result();
        let unsafe_layers = result.float16_unsafe_layers();
        assert_eq!(unsafe_layers.len(), 1);
    }

    #[test]
    fn test_quantization_result_int8_unsafe_layers() {
        let result = make_quantization_result();
        let unsafe_layers = result.int8_unsafe_layers();
        assert_eq!(unsafe_layers.len(), 1);
    }

    #[test]
    fn test_quantization_result_repr() {
        let result = make_quantization_result();
        let repr = result.__repr__();
        assert!(repr.contains("QuantizationResult"));
        assert!(repr.contains("layers=3"));
    }

    // =========================================================================
    // BoundStatus Tests
    // =========================================================================

    #[test]
    fn test_bound_status_repr() {
        assert_eq!(BoundStatus::Tight.__repr__(), "BoundStatus.Tight");
        assert_eq!(BoundStatus::Moderate.__repr__(), "BoundStatus.Moderate");
        assert_eq!(BoundStatus::Wide.__repr__(), "BoundStatus.Wide");
        assert_eq!(BoundStatus::VeryWide.__repr__(), "BoundStatus.VeryWide");
        assert_eq!(BoundStatus::Overflow.__repr__(), "BoundStatus.Overflow");
    }

    #[test]
    fn test_bound_status_str() {
        assert_eq!(BoundStatus::Tight.__str__(), "TIGHT");
        assert_eq!(BoundStatus::Moderate.__str__(), "MODERATE");
        assert_eq!(BoundStatus::Wide.__str__(), "WIDE");
        assert_eq!(BoundStatus::VeryWide.__str__(), "VERY WIDE");
        assert_eq!(BoundStatus::Overflow.__str__(), "OVERFLOW");
    }

    // =========================================================================
    // LayerProfile Tests
    // =========================================================================

    fn make_layer_profile(growth: f32, status: BoundStatus) -> LayerProfile {
        LayerProfile {
            name: "profile_layer".to_string(),
            layer_type: "Linear".to_string(),
            input_width: 0.1,
            output_width: growth * 0.1,
            mean_output_width: growth * 0.1,
            median_output_width: growth * 0.1,
            growth_ratio: growth,
            cumulative_expansion: growth,
            output_shape: vec![1, 64],
            num_elements: 64,
            status,
        }
    }

    #[test]
    fn test_layer_profile_is_choke_point() {
        let layer = make_layer_profile(10.0, BoundStatus::Wide);
        assert!(layer.is_choke_point(5.0));
        assert!(!layer.is_choke_point(15.0));
    }

    #[test]
    fn test_layer_profile_repr() {
        let layer = make_layer_profile(5.0, BoundStatus::Moderate);
        let repr = layer.__repr__();
        assert!(repr.contains("profile_layer"));
        assert!(repr.contains("growth=5.00x"));
        assert!(repr.contains("MODERATE"));
    }

    // =========================================================================
    // ProfileResult Tests
    // =========================================================================

    fn make_profile_result() -> ProfileResult {
        ProfileResult {
            layers: vec![
                make_layer_profile(2.0, BoundStatus::Tight),
                make_layer_profile(10.0, BoundStatus::Wide),
                make_layer_profile(1.5, BoundStatus::Moderate),
            ],
            input_epsilon: 0.01,
            initial_width: 0.02,
            final_width: 0.6,
            total_expansion: 30.0,
            max_growth_layer: Some(1),
            max_growth_ratio: 10.0,
            overflow_at_layer: None,
            difficulty_score: 75.0,
        }
    }

    #[test]
    fn test_profile_result_choke_points() {
        let result = make_profile_result();
        let choke_points = result.choke_points(5.0);
        assert_eq!(choke_points.len(), 1);
        assert_eq!(choke_points[0].growth_ratio, 10.0);
    }

    #[test]
    fn test_profile_result_problematic_layers() {
        let result = make_profile_result();
        let problematic = result.problematic_layers();
        assert_eq!(problematic.len(), 1);
    }

    #[test]
    fn test_profile_result_has_overflow() {
        let result = make_profile_result();
        assert!(!result.has_overflow());

        let mut overflow_result = make_profile_result();
        overflow_result.overflow_at_layer = Some(2);
        assert!(overflow_result.has_overflow());
    }

    #[test]
    fn test_profile_result_max_growth_layer_name() {
        let result = make_profile_result();
        assert!(result.max_growth_layer_name().is_some());
    }

    #[test]
    fn test_profile_result_repr() {
        let result = make_profile_result();
        let repr = result.__repr__();
        assert!(repr.contains("ProfileResult"));
        assert!(repr.contains("expansion=30.00x"));
        assert!(repr.contains("difficulty=75/100"));
    }

    // =========================================================================
    // OutputBound Tests
    // =========================================================================

    #[test]
    fn test_output_bound_width() {
        let bound = OutputBound {
            lower: -1.0,
            upper: 2.0,
        };
        assert!((bound.width() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_output_bound_midpoint() {
        let bound = OutputBound {
            lower: -1.0,
            upper: 3.0,
        };
        assert!((bound.midpoint() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_output_bound_repr() {
        let bound = OutputBound {
            lower: 0.5,
            upper: 1.5,
        };
        let repr = bound.__repr__();
        assert!(repr.contains("OutputBound"));
        assert!(repr.contains("0.5"));
        assert!(repr.contains("1.5"));
    }

    // =========================================================================
    // VerifyStatus Tests
    // =========================================================================

    #[test]
    fn test_verify_status_repr() {
        assert_eq!(VerifyStatus::Verified.__repr__(), "VerifyStatus.Verified");
        assert_eq!(VerifyStatus::Violated.__repr__(), "VerifyStatus.Violated");
        assert_eq!(VerifyStatus::Unknown.__repr__(), "VerifyStatus.Unknown");
        assert_eq!(VerifyStatus::Timeout.__repr__(), "VerifyStatus.Timeout");
    }

    #[test]
    fn test_verify_status_str() {
        assert_eq!(VerifyStatus::Verified.__str__(), "VERIFIED");
        assert_eq!(VerifyStatus::Violated.__str__(), "VIOLATED");
        assert_eq!(VerifyStatus::Unknown.__str__(), "UNKNOWN");
        assert_eq!(VerifyStatus::Timeout.__str__(), "TIMEOUT");
    }

    // =========================================================================
    // VerifyResult Tests
    // =========================================================================

    fn make_verify_result(status: VerifyStatus) -> VerifyResult {
        let output_bounds = Some(vec![
            OutputBound {
                lower: 0.0,
                upper: 1.0,
            },
            OutputBound {
                lower: -0.5,
                upper: 0.5,
            },
        ]);

        VerifyResult {
            status,
            output_bounds,
            counterexample: None,
            counterexample_output: None,
            reason: None,
            method: "IBP".to_string(),
            epsilon: 0.01,
        }
    }

    #[test]
    fn test_verify_result_is_verified() {
        let verified = make_verify_result(VerifyStatus::Verified);
        assert!(verified.is_verified());
        assert!(!verified.is_violated());

        let unknown = make_verify_result(VerifyStatus::Unknown);
        assert!(!unknown.is_verified());
    }

    #[test]
    fn test_verify_result_is_violated() {
        let violated = make_verify_result(VerifyStatus::Violated);
        assert!(violated.is_violated());
        assert!(!violated.is_verified());
    }

    #[test]
    fn test_verify_result_max_output_width() {
        let result = make_verify_result(VerifyStatus::Verified);
        let max_width = result.max_output_width();
        assert!(max_width.is_some());
        assert!((max_width.unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_verify_result_max_output_width_none() {
        let mut result = make_verify_result(VerifyStatus::Verified);
        result.output_bounds = None;
        assert!(result.max_output_width().is_none());
    }

    #[test]
    fn test_verify_result_repr() {
        let result = make_verify_result(VerifyStatus::Verified);
        let repr = result.__repr__();
        assert!(repr.contains("VerifyResult"));
        assert!(repr.contains("VERIFIED"));
        assert!(repr.contains("IBP"));
    }

    // =========================================================================
    // Clone Tests
    // =========================================================================

    #[test]
    fn test_layer_comparison_clone() {
        let original = LayerComparison {
            name: "layer".to_string(),
            name_b: None,
            max_diff: 0.001,
            mean_diff: 0.0005,
            exceeds_tolerance: false,
            shape_a: vec![1, 64],
            shape_b: vec![1, 64],
        };
        let cloned = original.clone();
        assert_eq!(cloned.name, original.name);
        assert_eq!(cloned.max_diff, original.max_diff);
    }

    #[test]
    fn test_diff_result_clone() {
        let original = make_diff_result(Some(1));
        let cloned = original.clone();
        assert_eq!(cloned.layers.len(), original.layers.len());
        assert_eq!(cloned.first_bad_layer, original.first_bad_layer);
    }

    #[test]
    fn test_sensitivity_result_clone() {
        let original = make_sensitivity_result();
        let cloned = original.clone();
        assert_eq!(cloned.layers.len(), original.layers.len());
        assert_eq!(cloned.max_sensitivity, original.max_sensitivity);
    }

    #[test]
    fn test_quantization_result_clone() {
        let original = make_quantization_result();
        let cloned = original.clone();
        assert_eq!(cloned.layers.len(), original.layers.len());
        assert_eq!(cloned.float16_safe, original.float16_safe);
    }

    #[test]
    fn test_profile_result_clone() {
        let original = make_profile_result();
        let cloned = original.clone();
        assert_eq!(cloned.layers.len(), original.layers.len());
        assert_eq!(cloned.difficulty_score, original.difficulty_score);
    }

    #[test]
    fn test_verify_result_clone() {
        let original = make_verify_result(VerifyStatus::Verified);
        let cloned = original.clone();
        assert_eq!(cloned.method, original.method);
        assert_eq!(cloned.epsilon, original.epsilon);
    }
}
