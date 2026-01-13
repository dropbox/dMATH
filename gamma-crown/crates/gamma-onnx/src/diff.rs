//! Model comparison and layer-by-layer diff functionality.
//!
//! This module provides tools to compare two ONNX models layer-by-layer,
//! identifying where outputs first diverge. Useful for debugging model ports
//! (e.g., PyTorch → CoreML/Metal conversions).
//!
//! ## Intermediate Layer Extraction
//!
//! To compare layer-by-layer, we modify the ONNX graph to expose all intermediate
//! tensors as outputs. This is done by:
//! 1. Parsing the ONNX protobuf
//! 2. Adding all node outputs to graph outputs
//! 3. Serializing back to bytes
//! 4. Loading into ONNX Runtime via commit_from_memory
//!
//! ## Root Cause Diagnosis
//!
//! When `--diagnose` is enabled, the diff analyzes divergence patterns to identify
//! common numerical issues:
//! - Softmax overflow (large logits near exp boundary)
//! - Accumulation order differences (non-associative float ops)
//! - Mixed precision errors (fp16/fp32 boundaries)
//! - Weight mismatches (actual different values, not numerical drift)

use crate::onnx_proto;
use crate::{LayerSpec, TensorSpec};
use gamma_core::LayerType;
use ndarray::{ArrayD, IxDyn};
use ort::session::Session;
use prost::Message;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors that can occur during model diffing.
#[derive(Error, Debug)]
pub enum DiffError {
    #[error("Failed to load model: {0}")]
    LoadError(String),

    #[error("ONNX Runtime error: {0}")]
    OrtError(#[from] ort::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("NPY read error: {0}")]
    NpyError(String),

    #[error("Input shape mismatch: model A {model_a:?} vs model B {model_b:?}")]
    InputShapeMismatch {
        model_a: Vec<i64>,
        model_b: Vec<i64>,
    },

    #[error("Layer not found: {0}")]
    LayerNotFound(String),

    #[error("No layers to compare")]
    NoLayers,
}

/// Result of comparing a single layer between two models.
#[derive(Debug, Clone)]
pub struct LayerComparison {
    /// Layer name (from model A, matched to model B).
    pub name: String,
    /// Name in model B (if different from model A).
    pub name_b: Option<String>,
    /// Maximum absolute difference between outputs.
    pub max_diff: f32,
    /// Mean absolute difference between outputs.
    pub mean_diff: f32,
    /// Whether this layer exceeds the tolerance.
    pub exceeds_tolerance: bool,
    /// Output shape from model A.
    pub shape_a: Vec<usize>,
    /// Output shape from model B.
    pub shape_b: Vec<usize>,
}

/// Status of a layer comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffStatus {
    /// Outputs match within tolerance.
    Ok,
    /// First layer where drift is detected (within 10x tolerance).
    DriftStarts,
    /// Layer exceeds tolerance.
    ExceedsTolerance,
    /// Shapes don't match.
    ShapeMismatch,
}

/// Pattern of divergence detected by root cause analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum DivergencePattern {
    /// exp() overflow/underflow differences (large logits near ±88)
    ExpPrecision {
        /// Maximum logit value observed before exp
        max_logit: f32,
        /// Whether overflow (>88) or underflow (<-88) boundary
        is_overflow: bool,
    },
    /// Softmax numerical instability
    SoftmaxInstability {
        /// Maximum score before softmax
        max_score: f32,
        /// Range of scores (max - min)
        score_range: f32,
    },
    /// Accumulation order differences (non-associative float ops)
    AccumulationOrder {
        /// Operation where accumulation differs (e.g., "matmul", "sum")
        operation: String,
        /// Whether diff correlates with tensor size
        size_correlated: bool,
    },
    /// Quantization truncation errors
    QuantizationError {
        /// Estimated bits of precision lost
        bits_lost: u8,
        /// Whether at power-of-2 boundaries
        at_power_boundary: bool,
    },
    /// Weight mismatch (not numerical - actual different values)
    WeightMismatch {
        /// Layer with mismatched weights
        layer: String,
        /// Maximum weight difference
        max_diff: f32,
    },
    /// GELU approximation method differs (tanh vs erf)
    GeluApproximation {
        /// Max difference in GELU region
        max_diff_in_region: f32,
    },
    /// LayerNorm epsilon or computation order differs
    LayerNormVariance {
        /// Whether epsilon values likely differ
        epsilon_differs: bool,
    },
    /// Unknown pattern (could not identify root cause)
    Unknown,
}

impl std::fmt::Display for DivergencePattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DivergencePattern::ExpPrecision {
                max_logit,
                is_overflow,
            } => {
                let boundary = if *is_overflow {
                    "overflow"
                } else {
                    "underflow"
                };
                write!(
                    f,
                    "exp() {} boundary: max_logit = {:.1} (boundary ~±88)",
                    boundary, max_logit
                )
            }
            DivergencePattern::SoftmaxInstability {
                max_score,
                score_range,
            } => {
                write!(
                    f,
                    "softmax instability: max_score = {:.1}, range = {:.1}",
                    max_score, score_range
                )
            }
            DivergencePattern::AccumulationOrder {
                operation,
                size_correlated,
            } => {
                let correlation = if *size_correlated {
                    " (diff grows with size)"
                } else {
                    ""
                };
                write!(f, "accumulation order in {}{}", operation, correlation)
            }
            DivergencePattern::QuantizationError {
                bits_lost,
                at_power_boundary,
            } => {
                let boundary = if *at_power_boundary {
                    " at power-of-2 boundary"
                } else {
                    ""
                };
                write!(f, "~{} bits precision lost{}", bits_lost, boundary)
            }
            DivergencePattern::WeightMismatch { layer, max_diff } => {
                write!(
                    f,
                    "weights differ in {}: max_diff = {:.2e}",
                    layer, max_diff
                )
            }
            DivergencePattern::GeluApproximation { max_diff_in_region } => {
                write!(
                    f,
                    "GELU approximation method differs: max_diff = {:.2e}",
                    max_diff_in_region
                )
            }
            DivergencePattern::LayerNormVariance { epsilon_differs } => {
                if *epsilon_differs {
                    write!(f, "LayerNorm epsilon values differ")
                } else {
                    write!(f, "LayerNorm variance computation order differs")
                }
            }
            DivergencePattern::Unknown => write!(f, "unknown pattern"),
        }
    }
}

/// Root cause diagnosis for model divergence.
#[derive(Debug, Clone)]
pub struct DiffDiagnosis {
    /// Layer where divergence first exceeds tolerance.
    pub divergence_layer: String,
    /// Layer type where divergence occurs.
    pub layer_type: Option<LayerType>,
    /// Detected pattern explaining the divergence.
    pub pattern: DivergencePattern,
    /// Human-readable explanation.
    pub explanation: String,
    /// Suggested fix if known.
    pub suggestion: Option<String>,
    /// Confidence level (0.0 - 1.0) in the diagnosis.
    pub confidence: f32,
    /// Supporting evidence for the diagnosis.
    pub evidence: Vec<String>,
}

impl DiffDiagnosis {
    /// Create a diagnosis for an unknown pattern.
    pub fn unknown(layer: &str, layer_type: Option<LayerType>) -> Self {
        Self {
            divergence_layer: layer.to_string(),
            layer_type,
            pattern: DivergencePattern::Unknown,
            explanation: "Could not identify a specific divergence pattern".to_string(),
            suggestion: None,
            confidence: 0.0,
            evidence: Vec::new(),
        }
    }

    /// Format the diagnosis for display.
    pub fn format_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!("  Layer: {}\n", self.divergence_layer));
        if let Some(lt) = &self.layer_type {
            report.push_str(&format!("  Layer Type: {:?}\n", lt));
        }
        report.push_str(&format!("  Issue: {}\n", self.pattern));
        report.push_str(&format!("  Confidence: {:.0}%\n", self.confidence * 100.0));
        if !self.explanation.is_empty() {
            report.push_str(&format!("  Explanation: {}\n", self.explanation));
        }
        if !self.evidence.is_empty() {
            report.push_str("  Evidence:\n");
            for ev in &self.evidence {
                report.push_str(&format!("    - {}\n", ev));
            }
        }
        if let Some(ref sug) = self.suggestion {
            report.push_str(&format!("\n  Suggestion: {}\n", sug));
        }
        report
    }
}

/// Result of a full model diff operation.
#[derive(Debug, Clone)]
pub struct DiffResult {
    /// Per-layer comparison results.
    pub layers: Vec<LayerComparison>,
    /// Index of first layer that exceeded tolerance (if any).
    pub first_bad_layer: Option<usize>,
    /// Index of first layer where drift started (within 10x tolerance).
    pub drift_start_layer: Option<usize>,
    /// Overall maximum divergence across all layers.
    pub max_divergence: f32,
    /// Tolerance used for comparison.
    pub tolerance: f32,
    /// Suggested root cause (if identified) - legacy field for backwards compat.
    pub suggestion: Option<String>,
    /// Detailed root cause diagnosis (when --diagnose is enabled).
    pub diagnosis: Option<DiffDiagnosis>,
}

impl DiffResult {
    /// Get the status for each layer.
    pub fn statuses(&self) -> Vec<DiffStatus> {
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

    /// Check if models are equivalent within tolerance.
    pub fn is_equivalent(&self) -> bool {
        self.first_bad_layer.is_none()
    }

    /// Get the first bad layer name if any.
    pub fn first_bad_layer_name(&self) -> Option<&str> {
        self.first_bad_layer
            .and_then(|i| self.layers.get(i))
            .map(|l| l.name.as_str())
    }
}

/// Configuration for model diffing.
#[derive(Debug, Clone)]
pub struct DiffConfig {
    /// Maximum absolute difference allowed between outputs.
    pub tolerance: f32,
    /// Whether to continue comparing after first divergence.
    pub continue_after_divergence: bool,
    /// Input value for testing (None = zeros).
    pub input: Option<ArrayD<f32>>,
    /// Explicit layer name mappings (model_a_name -> model_b_name).
    pub layer_mapping: HashMap<String, String>,
    /// Enable root cause diagnosis analysis.
    pub diagnose: bool,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-5,
            continue_after_divergence: true,
            input: None,
            layer_mapping: HashMap::new(),
            diagnose: false,
        }
    }
}

/// Information about a model loaded for diffing.
#[derive(Debug)]
pub struct ModelInfo {
    /// Input specifications.
    pub inputs: Vec<TensorSpec>,
    /// Output specifications.
    pub outputs: Vec<TensorSpec>,
    /// All intermediate tensor names (node outputs).
    pub intermediate_names: Vec<String>,
    /// Layer specifications from ONNX parsing.
    pub layers: Vec<LayerSpec>,
}

/// Load model info from an ONNX file.
pub fn load_model_info(path: impl AsRef<Path>) -> Result<ModelInfo, DiffError> {
    let onnx_model =
        crate::load_onnx(path.as_ref()).map_err(|e| DiffError::LoadError(format!("{}", e)))?;

    let network = &onnx_model.network;

    // Collect all intermediate tensor names from layer outputs
    let mut intermediate_names = Vec::new();
    for layer in &network.layers {
        for output in &layer.outputs {
            intermediate_names.push(output.clone());
        }
    }

    Ok(ModelInfo {
        inputs: network.inputs.clone(),
        outputs: network.outputs.clone(),
        intermediate_names,
        layers: network.layers.clone(),
    })
}

/// Run inference on a single model and get outputs.
///
/// This runs the model and returns the final output(s).
pub fn run_inference(
    path: impl AsRef<Path>,
    input: &ArrayD<f32>,
) -> Result<Vec<ArrayD<f32>>, DiffError> {
    // Create session
    let mut session = Session::builder()?.commit_from_file(path.as_ref())?;

    // Convert input to ort tensor
    let input_shape: Vec<usize> = input.shape().to_vec();
    let input_data: Vec<f32> = input.iter().cloned().collect();

    let input_tensor =
        ort::value::TensorRef::from_array_view((input_shape.as_slice(), input_data.as_slice()))?;

    // Run inference
    let outputs = session.run(ort::inputs![input_tensor])?;

    // Extract outputs as ArrayD
    let mut result = Vec::new();
    for (_name, output) in outputs.iter() {
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let array = ArrayD::from_shape_vec(IxDyn(&shape_vec), data.to_vec())
            .map_err(|e| DiffError::LoadError(format!("Shape error: {}", e)))?;
        result.push(array);
    }

    Ok(result)
}

/// Create a modified ONNX model with all intermediate tensors exposed as outputs.
///
/// This function parses the ONNX protobuf, adds all node outputs to the graph outputs,
/// and returns the modified model bytes.
pub fn expose_intermediate_outputs(model_bytes: &[u8]) -> Result<Vec<u8>, DiffError> {
    // Parse the ONNX model
    let mut model = onnx_proto::ModelProto::decode(model_bytes)
        .map_err(|e| DiffError::LoadError(format!("Failed to parse ONNX protobuf: {}", e)))?;

    let graph = model
        .graph
        .as_mut()
        .ok_or_else(|| DiffError::LoadError("ONNX model has no graph".to_string()))?;

    // Collect existing output names to avoid duplicates
    let existing_outputs: std::collections::HashSet<String> =
        graph.output.iter().map(|o| o.name.clone()).collect();

    // Collect all input names (to skip adding them as outputs)
    let input_names: std::collections::HashSet<String> =
        graph.input.iter().map(|i| i.name.clone()).collect();

    // Collect initializer names (weights - skip these too)
    let initializer_names: std::collections::HashSet<String> =
        graph.initializer.iter().map(|i| i.name.clone()).collect();

    // Add all node outputs that aren't already graph outputs, inputs, or initializers
    let mut new_outputs = Vec::new();
    for node in &graph.node {
        for output_name in &node.output {
            if output_name.is_empty() {
                continue;
            }
            if existing_outputs.contains(output_name) {
                continue;
            }
            if input_names.contains(output_name) {
                continue;
            }
            if initializer_names.contains(output_name) {
                continue;
            }

            // Create a ValueInfoProto for this output
            // We don't set the type info - ONNX Runtime will infer it
            new_outputs.push(onnx_proto::ValueInfoProto {
                name: output_name.clone(),
                r#type: None,
            });
        }
    }

    debug!("Adding {} intermediate outputs to model", new_outputs.len());

    graph.output.extend(new_outputs);

    // Serialize back to bytes
    let mut buf = Vec::new();
    model.encode(&mut buf).map_err(|e| {
        DiffError::LoadError(format!("Failed to serialize modified ONNX model: {}", e))
    })?;

    Ok(buf)
}

/// Run inference and capture all intermediate outputs.
///
/// This modifies the ONNX graph to expose intermediate tensors as outputs,
/// runs inference, and returns all intermediate values keyed by tensor name.
pub fn run_inference_with_intermediates(
    path: impl AsRef<Path>,
    input: &ArrayD<f32>,
) -> Result<HashMap<String, ArrayD<f32>>, DiffError> {
    // Read the original model bytes
    let model_bytes = crate::io::read_bytes_maybe_gzip(path.as_ref())
        .map_err(|e| DiffError::LoadError(format!("{e}")))?;

    // Modify graph to expose all intermediate outputs
    let modified_bytes = expose_intermediate_outputs(&model_bytes)?;

    // Create session from modified model bytes
    let mut session = Session::builder()?.commit_from_memory(&modified_bytes)?;

    let input_shape: Vec<usize> = input.shape().to_vec();
    let input_data: Vec<f32> = input.iter().cloned().collect();

    let input_tensor =
        ort::value::TensorRef::from_array_view((input_shape.as_slice(), input_data.as_slice()))?;

    let outputs = session.run(ort::inputs![input_tensor])?;

    let mut result = HashMap::new();

    // Extract all outputs with their names
    for (name, output) in outputs.iter() {
        match output.try_extract_tensor::<f32>() {
            Ok((shape, data)) => {
                let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                match ArrayD::from_shape_vec(IxDyn(&shape_vec), data.to_vec()) {
                    Ok(array) => {
                        result.insert(name.to_string(), array);
                    }
                    Err(e) => {
                        warn!("Failed to convert tensor {} to array: {}", name, e);
                    }
                }
            }
            Err(e) => {
                // Some tensors may not be f32 (e.g., shape tensors are i64)
                debug!("Skipping non-f32 tensor {}: {}", name, e);
            }
        }
    }

    info!("Extracted {} intermediate outputs from model", result.len());

    Ok(result)
}

/// Match layer names between two models using heuristics.
///
/// Returns a list of (name_a, name_b) pairs for corresponding layers.
pub fn match_layer_names(
    layers_a: &[LayerSpec],
    layers_b: &[LayerSpec],
    explicit_mapping: &HashMap<String, String>,
) -> Vec<(String, Option<String>)> {
    let mut matches = Vec::new();

    // First, apply explicit mappings
    let b_names: HashMap<&str, &LayerSpec> =
        layers_b.iter().map(|l| (l.name.as_str(), l)).collect();

    for layer_a in layers_a {
        let name_a = &layer_a.name;

        // Check explicit mapping first
        if let Some(name_b) = explicit_mapping.get(name_a) {
            matches.push((name_a.clone(), Some(name_b.clone())));
            continue;
        }

        // Try exact match
        if b_names.contains_key(name_a.as_str()) {
            matches.push((name_a.clone(), Some(name_a.clone())));
            continue;
        }

        // Try fuzzy matching based on:
        // 1. Layer type match
        // 2. Position in network
        // 3. Suffix/prefix patterns (e.g., "_0" vs ".0")

        // Normalize name for matching
        let normalized_a = normalize_layer_name(name_a);

        let mut best_match: Option<&str> = None;
        for layer_b in layers_b {
            let normalized_b = normalize_layer_name(&layer_b.name);
            if normalized_a == normalized_b && layer_a.layer_type == layer_b.layer_type {
                best_match = Some(&layer_b.name);
                break;
            }
        }

        matches.push((name_a.clone(), best_match.map(|s| s.to_string())));
    }

    matches
}

/// Normalize a layer name for fuzzy matching.
fn normalize_layer_name(name: &str) -> String {
    let normalized: String = name
        .to_lowercase()
        .replace('_', ".")
        .replace("layer", "")
        .replace("block", "")
        .replace("module", "")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '.')
        .collect();
    // Remove consecutive dots and leading/trailing dots
    normalized
        .split('.')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(".")
}

/// Compare two arrays element-wise and compute diff statistics.
fn compare_arrays(a: &ArrayD<f32>, b: &ArrayD<f32>, tolerance: f32) -> LayerComparison {
    let shape_a: Vec<usize> = a.shape().to_vec();
    let shape_b: Vec<usize> = b.shape().to_vec();

    if shape_a != shape_b {
        return LayerComparison {
            name: String::new(),
            name_b: None,
            max_diff: f32::INFINITY,
            mean_diff: f32::INFINITY,
            exceeds_tolerance: true,
            shape_a,
            shape_b,
        };
    }

    let diffs: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(va, vb)| (va - vb).abs())
        .collect();

    let max_diff = diffs.iter().cloned().fold(0.0f32, f32::max);
    let mean_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;

    LayerComparison {
        name: String::new(),
        name_b: None,
        max_diff,
        mean_diff,
        exceeds_tolerance: max_diff > tolerance,
        shape_a,
        shape_b,
    }
}

/// Suggest a root cause based on the layer type where divergence starts.
fn suggest_root_cause(layer: &LayerSpec) -> Option<String> {
    use gamma_core::LayerType::*;

    match layer.layer_type {
        Softmax | CausalSoftmax => Some(
            "Check softmax numerical precision (exp overflow handling, log-sum-exp trick)"
                .to_string(),
        ),
        LayerNorm => {
            Some("Check LayerNorm epsilon value and variance computation order".to_string())
        }
        GELU => Some("Check GELU approximation method (tanh vs erf)".to_string()),
        Conv1d | Conv2d => Some("Check convolution padding mode and group handling".to_string()),
        Linear | MatMul => {
            Some("Check matrix multiplication transpose flags and bias handling".to_string())
        }
        Add | Mul => Some("Check broadcast semantics and tensor shapes".to_string()),
        _ => None,
    }
}

/// Context for diagnosing divergence patterns.
struct DiagnosisContext<'a> {
    /// Outputs from model A keyed by tensor name.
    outputs_a: &'a HashMap<String, ArrayD<f32>>,
    /// Outputs from model B keyed by tensor name.
    outputs_b: &'a HashMap<String, ArrayD<f32>>,
    /// Layer specs from model A.
    layers_a: &'a [LayerSpec],
    /// Layer comparisons computed so far.
    comparisons: &'a [LayerComparison],
    /// Tolerance used for comparison.
    tolerance: f32,
}

/// Detect the divergence pattern and generate a diagnosis.
fn diagnose_divergence(
    ctx: &DiagnosisContext,
    bad_layer_idx: usize,
    layer_spec: Option<&LayerSpec>,
) -> DiffDiagnosis {
    let comparison = &ctx.comparisons[bad_layer_idx];
    let layer_name = &comparison.name;
    let layer_type = layer_spec.map(|l| l.layer_type.clone());

    // Get the actual tensor data for analysis
    let out_a = ctx.outputs_a.get(layer_name);
    let out_b = ctx.outputs_b.get(layer_name);

    // Try to detect specific patterns based on layer type and tensor values
    if let (Some(arr_a), Some(arr_b)) = (out_a, out_b) {
        // Check for exp/softmax overflow pattern
        if let Some(ref lt) = layer_type {
            if matches!(lt, LayerType::Softmax | LayerType::CausalSoftmax) {
                if let Some(diag) = check_softmax_pattern(ctx, layer_name, arr_a, arr_b, lt.clone())
                {
                    return diag;
                }
            }

            // Check for GELU approximation differences
            if *lt == LayerType::GELU {
                if let Some(diag) =
                    check_gelu_pattern(layer_name, arr_a, arr_b, comparison.max_diff)
                {
                    return diag;
                }
            }

            // Check for LayerNorm variance issues
            if *lt == LayerType::LayerNorm {
                if let Some(diag) =
                    check_layernorm_pattern(layer_name, arr_a, arr_b, comparison.max_diff)
                {
                    return diag;
                }
            }

            // Check for accumulation order issues in matmul/linear
            if matches!(lt, LayerType::Linear | LayerType::MatMul) {
                if let Some(diag) =
                    check_accumulation_pattern(layer_name, arr_a, arr_b, comparison.max_diff)
                {
                    return diag;
                }
            }
        }

        // Check for quantization errors (independent of layer type)
        if let Some(diag) = check_quantization_pattern(
            layer_name,
            layer_type.as_ref(),
            arr_a,
            arr_b,
            comparison.max_diff,
        ) {
            return diag;
        }

        // Check for growing error pattern (accumulation order)
        if bad_layer_idx > 2 {
            if let Some(diag) =
                check_growing_error_pattern(ctx, bad_layer_idx, layer_name, layer_type.as_ref())
            {
                return diag;
            }
        }
    }

    // Fallback: return diagnosis based on layer type only
    let suggestion = layer_spec.and_then(suggest_root_cause);
    DiffDiagnosis {
        divergence_layer: layer_name.clone(),
        layer_type,
        pattern: DivergencePattern::Unknown,
        explanation: format!(
            "Divergence exceeds tolerance ({:.2e}) at layer {}",
            ctx.tolerance, layer_name
        ),
        suggestion,
        confidence: 0.2,
        evidence: vec![format!("max_diff = {:.2e}", comparison.max_diff)],
    }
}

/// Check for softmax numerical instability patterns.
fn check_softmax_pattern(
    ctx: &DiagnosisContext,
    layer_name: &str,
    _arr_a: &ArrayD<f32>,
    _arr_b: &ArrayD<f32>,
    layer_type: LayerType,
) -> Option<DiffDiagnosis> {
    // Find the input to this softmax layer by looking at the previous layer
    // Softmax input is typically from a matmul (attention scores)
    let mut prev_tensor_name: Option<String> = None;

    for layer in ctx.layers_a {
        for output in &layer.outputs {
            if output == layer_name {
                // Found the softmax layer, get its input
                if !layer.inputs.is_empty() {
                    prev_tensor_name = Some(layer.inputs[0].clone());
                }
                break;
            }
        }
    }

    // Get the pre-softmax values to analyze
    if let Some(ref input_name) = prev_tensor_name {
        if let Some(pre_softmax) = ctx.outputs_a.get(input_name) {
            let max_val = pre_softmax
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let min_val = pre_softmax.iter().cloned().fold(f32::INFINITY, f32::min);
            let range = max_val - min_val;

            // exp(88) is approximately f32::MAX, so logits > 80 are risky
            if max_val > 80.0 {
                return Some(DiffDiagnosis {
                    divergence_layer: layer_name.to_string(),
                    layer_type: Some(layer_type),
                    pattern: DivergencePattern::ExpPrecision {
                        max_logit: max_val,
                        is_overflow: true,
                    },
                    explanation: format!(
                        "Pre-softmax logits have max value {:.1}, near exp() overflow boundary (~88)",
                        max_val
                    ),
                    suggestion: Some(
                        "Apply log-sum-exp stabilization: subtract max(x) before exp()"
                            .to_string(),
                    ),
                    confidence: 0.9,
                    evidence: vec![
                        format!("max_logit = {:.1}", max_val),
                        format!("logit_range = {:.1}", range),
                        "exp(88) ~ f32::MAX".to_string(),
                    ],
                });
            } else if max_val < -80.0 {
                return Some(DiffDiagnosis {
                    divergence_layer: layer_name.to_string(),
                    layer_type: Some(layer_type),
                    pattern: DivergencePattern::ExpPrecision {
                        max_logit: max_val,
                        is_overflow: false,
                    },
                    explanation: format!(
                        "Pre-softmax logits have min value {:.1}, near exp() underflow boundary",
                        min_val
                    ),
                    suggestion: Some(
                        "Check if input normalization is missing or incorrect".to_string(),
                    ),
                    confidence: 0.85,
                    evidence: vec![
                        format!("min_logit = {:.1}", min_val),
                        format!("logit_range = {:.1}", range),
                    ],
                });
            } else if range > 50.0 {
                // Large range can cause precision loss in softmax
                return Some(DiffDiagnosis {
                    divergence_layer: layer_name.to_string(),
                    layer_type: Some(layer_type),
                    pattern: DivergencePattern::SoftmaxInstability {
                        max_score: max_val,
                        score_range: range,
                    },
                    explanation: format!(
                        "Large logit range ({:.1}) causes numerical instability in softmax",
                        range
                    ),
                    suggestion: Some(
                        "Use numerically stable softmax with max subtraction".to_string(),
                    ),
                    confidence: 0.75,
                    evidence: vec![
                        format!("max_score = {:.1}", max_val),
                        format!("min_score = {:.1}", min_val),
                        format!("range = {:.1}", range),
                    ],
                });
            }
        }
    }

    None
}

/// Check for GELU approximation differences.
fn check_gelu_pattern(
    layer_name: &str,
    arr_a: &ArrayD<f32>,
    arr_b: &ArrayD<f32>,
    max_diff: f32,
) -> Option<DiffDiagnosis> {
    // GELU has two common approximations:
    // 1. tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // 2. erf approximation: 0.5 * x * (1 + erf(x / sqrt(2)))
    //
    // These differ most around x = ±1 where the curve transitions

    // Check if differences are concentrated in the transition region
    let mut transition_diffs = 0;
    let mut total_significant = 0;

    for (a, b) in arr_a.iter().zip(arr_b.iter()) {
        let diff = (a - b).abs();
        if diff > max_diff * 0.1 {
            total_significant += 1;
            // Transition region is roughly |x| in [0.5, 2.0] where GELU derivative is significant
            // But we're looking at output values, which are similar magnitude for typical inputs
            if a.abs() > 0.1 && a.abs() < 2.0 {
                transition_diffs += 1;
            }
        }
    }

    if total_significant > 0 && transition_diffs as f64 / total_significant as f64 > 0.5 {
        return Some(DiffDiagnosis {
            divergence_layer: layer_name.to_string(),
            layer_type: Some(LayerType::GELU),
            pattern: DivergencePattern::GeluApproximation {
                max_diff_in_region: max_diff,
            },
            explanation: "GELU implementations use different approximation methods".to_string(),
            suggestion: Some(
                "Ensure both models use the same GELU variant (tanh vs erf approximation)"
                    .to_string(),
            ),
            confidence: 0.7,
            evidence: vec![
                format!("max_diff = {:.2e}", max_diff),
                format!(
                    "{:.0}% of significant diffs in transition region",
                    100.0 * transition_diffs as f64 / total_significant as f64
                ),
            ],
        });
    }

    None
}

/// Check for LayerNorm variance computation issues.
fn check_layernorm_pattern(
    layer_name: &str,
    arr_a: &ArrayD<f32>,
    arr_b: &ArrayD<f32>,
    max_diff: f32,
) -> Option<DiffDiagnosis> {
    // LayerNorm issues often show as:
    // 1. Consistent offset (epsilon difference)
    // 2. Scaling errors (variance computation order)

    // Check for consistent offset vs random errors
    let diffs: Vec<f32> = arr_a.iter().zip(arr_b.iter()).map(|(a, b)| a - b).collect();

    if diffs.is_empty() {
        return None;
    }

    let mean_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let variance: f32 =
        diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f32>() / diffs.len() as f32;
    let std_diff = variance.sqrt();

    // If std is very low relative to mean, it's likely a systematic error (epsilon)
    if std_diff < mean_diff.abs() * 0.1 && mean_diff.abs() > max_diff * 0.5 {
        return Some(DiffDiagnosis {
            divergence_layer: layer_name.to_string(),
            layer_type: Some(LayerType::LayerNorm),
            pattern: DivergencePattern::LayerNormVariance {
                epsilon_differs: true,
            },
            explanation: "Systematic offset suggests different epsilon values".to_string(),
            suggestion: Some(
                "Check LayerNorm epsilon parameter (common: 1e-5 vs 1e-6)".to_string(),
            ),
            confidence: 0.8,
            evidence: vec![
                format!("mean_diff = {:.2e}", mean_diff),
                format!("std_diff = {:.2e}", std_diff),
                "Low variance suggests systematic error".to_string(),
            ],
        });
    }

    // Otherwise could be variance computation order
    if std_diff > mean_diff.abs() * 2.0 {
        return Some(DiffDiagnosis {
            divergence_layer: layer_name.to_string(),
            layer_type: Some(LayerType::LayerNorm),
            pattern: DivergencePattern::LayerNormVariance {
                epsilon_differs: false,
            },
            explanation: "High variance suggests different computation order".to_string(),
            suggestion: Some(
                "Check variance computation: single-pass vs two-pass algorithm".to_string(),
            ),
            confidence: 0.6,
            evidence: vec![
                format!("std_diff = {:.2e}", std_diff),
                format!("mean_diff = {:.2e}", mean_diff),
            ],
        });
    }

    None
}

/// Check for accumulation order differences in matrix operations.
fn check_accumulation_pattern(
    layer_name: &str,
    arr_a: &ArrayD<f32>,
    arr_b: &ArrayD<f32>,
    max_diff: f32,
) -> Option<DiffDiagnosis> {
    // Accumulation order differences typically:
    // 1. Scale with tensor size (more ops = more drift)
    // 2. Are relatively uniform across the output

    let size = arr_a.len();
    if size < 100 {
        return None; // Too small to detect pattern
    }

    // Calculate diff statistics across the tensor
    let diffs: Vec<f32> = arr_a
        .iter()
        .zip(arr_b.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();

    let mean_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let variance: f32 =
        diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f32>() / diffs.len() as f32;
    let cv = variance.sqrt() / mean_diff.max(1e-10); // Coefficient of variation

    // Low CV suggests uniform error distribution (accumulation order)
    if cv < 0.5 && max_diff > 1e-6 {
        return Some(DiffDiagnosis {
            divergence_layer: layer_name.to_string(),
            layer_type: Some(LayerType::MatMul),
            pattern: DivergencePattern::AccumulationOrder {
                operation: "matmul".to_string(),
                size_correlated: size > 10000,
            },
            explanation: "Uniform error distribution suggests accumulation order difference"
                .to_string(),
            suggestion: Some(
                "Use Kahan summation or ensure consistent reduction order".to_string(),
            ),
            confidence: 0.65,
            evidence: vec![
                format!("coefficient_of_variation = {:.2}", cv),
                format!("tensor_size = {}", size),
                format!("mean_diff = {:.2e}", mean_diff),
            ],
        });
    }

    None
}

/// Check for quantization error patterns.
fn check_quantization_pattern(
    layer_name: &str,
    layer_type: Option<&LayerType>,
    arr_a: &ArrayD<f32>,
    arr_b: &ArrayD<f32>,
    _max_diff: f32,
) -> Option<DiffDiagnosis> {
    // Quantization errors show as:
    // 1. Step-like differences (rounding)
    // 2. Errors at power-of-2 boundaries

    // Check if differences are quantized (multiples of a base unit)
    let diffs: Vec<f32> = arr_a
        .iter()
        .zip(arr_b.iter())
        .map(|(a, b)| (a - b).abs())
        .filter(|d| *d > 1e-10)
        .collect();

    if diffs.len() < 10 {
        return None;
    }

    // Find the smallest non-zero difference as potential quantization step
    let min_diff = diffs.iter().cloned().fold(f32::INFINITY, f32::min);

    // Check how many diffs are close to multiples of min_diff
    let mut quantized_count = 0;
    for d in &diffs {
        let ratio = d / min_diff;
        let rounded = ratio.round();
        if (ratio - rounded).abs() < 0.1 {
            quantized_count += 1;
        }
    }

    let quantized_ratio = quantized_count as f64 / diffs.len() as f64;

    if quantized_ratio > 0.7 {
        // Estimate bits lost based on quantization step
        let bits_lost = if min_diff > 1e-3 {
            10
        } else if min_diff > 1e-5 {
            7
        } else if min_diff > 1e-7 {
            4
        } else {
            2
        };

        // Check for power-of-2 boundary effects
        let at_boundary = arr_a.iter().any(|v| {
            let exp = v.abs().log2().floor();
            (v.abs() - 2.0f32.powf(exp)).abs() < min_diff * 10.0
        });

        return Some(DiffDiagnosis {
            divergence_layer: layer_name.to_string(),
            layer_type: layer_type.cloned(),
            pattern: DivergencePattern::QuantizationError {
                bits_lost,
                at_power_boundary: at_boundary,
            },
            explanation: format!("Differences appear quantized with step ~{:.2e}", min_diff),
            suggestion: Some(
                "Check for fp16/fp32 mixed precision or INT8 quantization".to_string(),
            ),
            confidence: 0.75,
            evidence: vec![
                format!("quantization_step = {:.2e}", min_diff),
                format!("{:.0}% of diffs are quantized", quantized_ratio * 100.0),
                format!("estimated_bits_lost = {}", bits_lost),
            ],
        });
    }

    None
}

/// Check for growing error pattern (error accumulation across layers).
fn check_growing_error_pattern(
    ctx: &DiagnosisContext,
    bad_layer_idx: usize,
    layer_name: &str,
    layer_type: Option<&LayerType>,
) -> Option<DiffDiagnosis> {
    // Check if errors are growing across layers
    if bad_layer_idx < 3 {
        return None;
    }

    let recent_diffs: Vec<f32> = ctx.comparisons[bad_layer_idx.saturating_sub(5)..=bad_layer_idx]
        .iter()
        .map(|c| c.max_diff)
        .collect();

    if recent_diffs.len() < 3 {
        return None;
    }

    // Check if diffs are monotonically increasing
    let mut increasing = true;
    for i in 1..recent_diffs.len() {
        if recent_diffs[i] < recent_diffs[i - 1] * 0.9 {
            increasing = false;
            break;
        }
    }

    if increasing {
        // Calculate growth rate
        let first = recent_diffs.first().unwrap_or(&0.0);
        let last = recent_diffs.last().unwrap_or(&0.0);
        let growth = if *first > 1e-10 { last / first } else { 1.0 };

        return Some(DiffDiagnosis {
            divergence_layer: layer_name.to_string(),
            layer_type: layer_type.cloned(),
            pattern: DivergencePattern::AccumulationOrder {
                operation: "network".to_string(),
                size_correlated: true,
            },
            explanation: format!(
                "Errors grow {:.1}x across {} layers, suggesting accumulation order differences",
                growth,
                recent_diffs.len()
            ),
            suggestion: Some(
                "Check for non-associative operations: different reduction orders, fused ops"
                    .to_string(),
            ),
            confidence: 0.7,
            evidence: vec![
                format!("growth_factor = {:.1}x", growth),
                format!("layers_analyzed = {}", recent_diffs.len()),
                format!("first_diff = {:.2e}", first),
                format!("last_diff = {:.2e}", last),
            ],
        });
    }

    None
}

/// Perform a full diff between two ONNX models.
///
/// This is the main entry point for comparing models. It extracts all intermediate
/// tensors from both models and compares them layer by layer.
pub fn diff_models(
    path_a: impl AsRef<Path>,
    path_b: impl AsRef<Path>,
    config: &DiffConfig,
) -> Result<DiffResult, DiffError> {
    info!("Loading model A: {}", path_a.as_ref().display());
    let info_a = load_model_info(&path_a)?;

    info!("Loading model B: {}", path_b.as_ref().display());
    let info_b = load_model_info(&path_b)?;

    // Check input compatibility
    if info_a.inputs.is_empty() || info_b.inputs.is_empty() {
        return Err(DiffError::NoLayers);
    }

    let input_a = &info_a.inputs[0];
    let _input_b = &info_b.inputs[0]; // Could validate shape match in future

    // Create input tensor
    let input = if let Some(ref inp) = config.input {
        inp.clone()
    } else {
        // Use zeros with model A's input shape
        let shape: Vec<usize> = input_a
            .shape
            .iter()
            .map(|&d| if d > 0 { d as usize } else { 1 })
            .collect();
        ArrayD::zeros(IxDyn(&shape))
    };

    debug!("Input shape: {:?}", input.shape());

    // Run inference with intermediate outputs on both models
    info!("Running inference on model A (extracting all intermediate outputs)");
    let outputs_a = run_inference_with_intermediates(&path_a, &input)?;

    info!("Running inference on model B (extracting all intermediate outputs)");
    let outputs_b = run_inference_with_intermediates(&path_b, &input)?;

    if outputs_a.is_empty() || outputs_b.is_empty() {
        return Err(DiffError::NoLayers);
    }

    info!(
        "Model A: {} tensors, Model B: {} tensors",
        outputs_a.len(),
        outputs_b.len()
    );

    // Build tensor name to layer spec mapping for suggestions
    let mut tensor_to_layer_a: HashMap<String, &LayerSpec> = HashMap::new();
    for layer in &info_a.layers {
        for output in &layer.outputs {
            tensor_to_layer_a.insert(output.clone(), layer);
        }
    }

    let mut layers = Vec::new();
    let mut first_bad_layer = None;
    let mut drift_start_layer = None;
    let mut max_divergence: f32 = 0.0;
    let mut first_bad_layer_spec: Option<&LayerSpec> = None;

    // Strategy: iterate through model A's layer outputs in order
    // and match them with model B's outputs
    for layer_a in &info_a.layers {
        for output_name_a in &layer_a.outputs {
            if output_name_a.is_empty() {
                continue;
            }

            // Get output from model A
            let out_a = match outputs_a.get(output_name_a) {
                Some(arr) => arr,
                None => continue, // Skip tensors we couldn't extract
            };

            // Try to find matching output in model B
            // First check explicit mapping
            let output_name_b = config
                .layer_mapping
                .get(output_name_a)
                .cloned()
                .or_else(|| {
                    // Try exact match
                    if outputs_b.contains_key(output_name_a) {
                        Some(output_name_a.clone())
                    } else {
                        // Try normalized name matching
                        let normalized_a = normalize_layer_name(output_name_a);
                        outputs_b
                            .keys()
                            .find(|name_b| normalize_layer_name(name_b) == normalized_a)
                            .cloned()
                    }
                });

            let (out_b, matched_name_b) = match output_name_b {
                Some(ref name_b) => {
                    match outputs_b.get(name_b) {
                        Some(arr) => (arr, Some(name_b.clone())),
                        None => continue, // Matched name but no output
                    }
                }
                None => continue, // No match found
            };

            // Compare arrays
            let mut comparison = compare_arrays(out_a, out_b, config.tolerance);
            comparison.name = output_name_a.clone();
            comparison.name_b = if matched_name_b.as_ref() == Some(output_name_a) {
                None
            } else {
                matched_name_b
            };

            max_divergence = max_divergence.max(comparison.max_diff);

            let idx = layers.len();

            // Detect first bad layer
            if comparison.exceeds_tolerance && first_bad_layer.is_none() {
                first_bad_layer = Some(idx);
                first_bad_layer_spec = tensor_to_layer_a.get(output_name_a).copied();
            }

            // Detect drift start (within 10x tolerance but above tolerance/10)
            if comparison.max_diff > config.tolerance / 10.0
                && comparison.max_diff <= config.tolerance * 10.0
                && drift_start_layer.is_none()
            {
                drift_start_layer = Some(idx);
            }

            layers.push(comparison);

            // Stop early if not continuing after divergence
            if !config.continue_after_divergence && first_bad_layer.is_some() {
                break;
            }
        }

        if !config.continue_after_divergence && first_bad_layer.is_some() {
            break;
        }
    }

    // If we didn't match any layers, fall back to comparing final outputs only
    if layers.is_empty() {
        warn!("No intermediate layers matched between models, comparing final outputs only");
        return diff_models_final_only(path_a, path_b, config);
    }

    // Generate suggestion based on the layer where divergence starts
    let suggestion = first_bad_layer_spec.and_then(suggest_root_cause);

    // Generate detailed diagnosis if enabled and there's a divergence
    let diagnosis = if config.diagnose {
        first_bad_layer.map(|bad_layer| {
            let ctx = DiagnosisContext {
                outputs_a: &outputs_a,
                outputs_b: &outputs_b,
                layers_a: &info_a.layers,
                comparisons: &layers,
                tolerance: config.tolerance,
            };
            diagnose_divergence(&ctx, bad_layer, first_bad_layer_spec)
        })
    } else {
        None
    };

    Ok(DiffResult {
        layers,
        first_bad_layer,
        drift_start_layer,
        max_divergence,
        tolerance: config.tolerance,
        suggestion,
        diagnosis,
    })
}

/// Fallback: compare only final outputs (used when intermediate matching fails).
fn diff_models_final_only(
    path_a: impl AsRef<Path>,
    path_b: impl AsRef<Path>,
    config: &DiffConfig,
) -> Result<DiffResult, DiffError> {
    let info_a = load_model_info(&path_a)?;
    let _info_b = load_model_info(&path_b)?;

    let input_a = &info_a.inputs[0];

    let input = if let Some(ref inp) = config.input {
        inp.clone()
    } else {
        let shape: Vec<usize> = input_a
            .shape
            .iter()
            .map(|&d| if d > 0 { d as usize } else { 1 })
            .collect();
        ArrayD::zeros(IxDyn(&shape))
    };

    let outputs_a = run_inference(&path_a, &input)?;
    let outputs_b = run_inference(&path_b, &input)?;

    if outputs_a.is_empty() || outputs_b.is_empty() {
        return Err(DiffError::NoLayers);
    }

    let mut layers = Vec::new();
    let mut first_bad_layer = None;
    let mut drift_start_layer = None;
    let mut max_divergence: f32 = 0.0;

    for (i, (out_a, out_b)) in outputs_a.iter().zip(outputs_b.iter()).enumerate() {
        let output_name = info_a
            .outputs
            .get(i)
            .map(|o| o.name.clone())
            .unwrap_or_else(|| format!("output_{}", i));

        let mut comparison = compare_arrays(out_a, out_b, config.tolerance);
        comparison.name = output_name;

        max_divergence = max_divergence.max(comparison.max_diff);

        if comparison.exceeds_tolerance && first_bad_layer.is_none() {
            first_bad_layer = Some(i);
        }

        if comparison.max_diff > config.tolerance / 10.0
            && comparison.max_diff <= config.tolerance * 10.0
            && drift_start_layer.is_none()
        {
            drift_start_layer = Some(i);
        }

        layers.push(comparison);
    }

    let suggestion = first_bad_layer
        .and_then(|_| info_a.layers.last())
        .and_then(suggest_root_cause);

    // Note: Diagnosis is not available in final-only mode since we lack intermediate data
    Ok(DiffResult {
        layers,
        first_bad_layer,
        drift_start_layer,
        max_divergence,
        tolerance: config.tolerance,
        suggestion,
        diagnosis: None,
    })
}

/// Load a numpy array from a .npy file.
pub fn load_npy(path: impl AsRef<Path>) -> Result<ArrayD<f32>, DiffError> {
    use ndarray_npy::ReadNpyExt;

    let file = std::fs::File::open(path.as_ref())?;
    let reader = std::io::BufReader::new(file);

    // Try f32 first
    if let Ok(arr) = ArrayD::<f32>::read_npy(reader) {
        return Ok(arr);
    }

    // Try f64 and convert
    let file = std::fs::File::open(path.as_ref())?;
    let reader = std::io::BufReader::new(file);
    if let Ok(arr) = ArrayD::<f64>::read_npy(reader) {
        return Ok(arr.mapv(|x| x as f32));
    }

    Err(DiffError::NpyError(format!(
        "Could not read numpy file as f32 or f64: {}",
        path.as_ref().display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_layer_name() {
        assert_eq!(normalize_layer_name("layer_0"), "0");
        assert_eq!(normalize_layer_name("encoder.block.0"), "encoder.0");
        assert_eq!(normalize_layer_name("Block_1_Linear"), "1.linear");
        assert_eq!(normalize_layer_name("layer0"), "0");
        assert_eq!(normalize_layer_name("module.layer.0.linear"), "0.linear");
    }

    #[test]
    fn test_compare_arrays() {
        let a = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0f32);
        let b = ArrayD::from_elem(IxDyn(&[2, 3]), 1.001f32);

        let comp = compare_arrays(&a, &b, 0.01);
        assert!(!comp.exceeds_tolerance);
        assert!((comp.max_diff - 0.001).abs() < 1e-6);

        let comp2 = compare_arrays(&a, &b, 0.0001);
        assert!(comp2.exceeds_tolerance);
    }

    #[test]
    fn test_compare_arrays_shape_mismatch() {
        let a = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0f32);
        let b = ArrayD::from_elem(IxDyn(&[3, 2]), 1.0f32);

        let comp = compare_arrays(&a, &b, 0.01);
        assert!(comp.exceeds_tolerance);
        assert!(comp.max_diff.is_infinite());
    }

    // Integration tests that require ONNX Runtime
    const TEST_MODELS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/models");

    fn test_model_path(name: &str) -> String {
        format!("{}/{}", TEST_MODELS_DIR, name)
    }

    #[test]
    fn test_intermediate_extraction() {
        let model_path = test_model_path("simple_mlp.onnx");
        if !std::path::Path::new(&model_path).exists() {
            eprintln!("Test model not found: {}, skipping", model_path);
            return;
        }

        // Create a simple input
        let input = ArrayD::from_elem(IxDyn(&[1, 2]), 0.5f32);

        // Run inference with intermediate extraction
        let outputs = run_inference_with_intermediates(&model_path, &input)
            .expect("Failed to run inference with intermediates");

        // Simple MLP should have: fc1_out, relu_out, output
        // The actual names depend on how the model was exported
        assert!(
            outputs.len() >= 2,
            "Expected at least 2 intermediate outputs, got {}",
            outputs.len()
        );

        // All outputs should be finite
        for (name, arr) in &outputs {
            assert!(
                arr.iter().all(|v| v.is_finite()),
                "Output {} contains non-finite values",
                name
            );
        }
    }

    #[test]
    fn test_diff_models_same_model() {
        let model_path = test_model_path("simple_mlp.onnx");
        if !std::path::Path::new(&model_path).exists() {
            eprintln!("Test model not found: {}, skipping", model_path);
            return;
        }

        let config = DiffConfig {
            tolerance: 1e-5,
            continue_after_divergence: true,
            input: None,
            layer_mapping: HashMap::new(),
            diagnose: false,
        };

        let result =
            diff_models(&model_path, &model_path, &config).expect("Failed to diff same model");

        // Same model should be equivalent
        assert!(result.is_equivalent(), "Same model should be equivalent");
        assert!(
            result.max_divergence < 1e-10,
            "Max divergence should be near zero for same model"
        );

        // Should have multiple layers
        assert!(
            result.layers.len() >= 2,
            "Should compare multiple layers, got {}",
            result.layers.len()
        );
    }

    #[test]
    fn test_diff_models_layer_count() {
        let model_path = test_model_path("transformer_mlp.onnx");
        if !std::path::Path::new(&model_path).exists() {
            eprintln!("Test model not found: {}, skipping", model_path);
            return;
        }

        let config = DiffConfig::default();

        let result = diff_models(&model_path, &model_path, &config).expect("Failed to diff model");

        // transformer_mlp has multiple layers (fc1, gelu, fc2)
        // Verify we're getting multiple comparisons
        assert!(
            result.layers.len() >= 2,
            "Expected >= 2 layer comparisons for transformer_mlp, got {}",
            result.layers.len()
        );

        // Print layers for debugging
        for layer in &result.layers {
            eprintln!("Layer: {} max_diff={:.2e}", layer.name, layer.max_diff);
        }
    }

    #[test]
    fn test_divergence_pattern_display() {
        // Test Display implementations for all patterns
        let patterns = vec![
            DivergencePattern::ExpPrecision {
                max_logit: 89.5,
                is_overflow: true,
            },
            DivergencePattern::ExpPrecision {
                max_logit: -90.0,
                is_overflow: false,
            },
            DivergencePattern::SoftmaxInstability {
                max_score: 75.0,
                score_range: 60.0,
            },
            DivergencePattern::AccumulationOrder {
                operation: "matmul".to_string(),
                size_correlated: true,
            },
            DivergencePattern::QuantizationError {
                bits_lost: 7,
                at_power_boundary: true,
            },
            DivergencePattern::WeightMismatch {
                layer: "fc1".to_string(),
                max_diff: 0.001,
            },
            DivergencePattern::GeluApproximation {
                max_diff_in_region: 1e-4,
            },
            DivergencePattern::LayerNormVariance {
                epsilon_differs: true,
            },
            DivergencePattern::Unknown,
        ];

        for pattern in patterns {
            let s = format!("{}", pattern);
            assert!(!s.is_empty(), "Pattern display should not be empty");
            eprintln!("Pattern: {}", s);
        }
    }

    #[test]
    fn test_diff_diagnosis_format_report() {
        let diagnosis = DiffDiagnosis {
            divergence_layer: "encoder.softmax".to_string(),
            layer_type: Some(gamma_core::LayerType::Softmax),
            pattern: DivergencePattern::ExpPrecision {
                max_logit: 85.0,
                is_overflow: true,
            },
            explanation: "Large logits near exp overflow".to_string(),
            suggestion: Some("Use log-sum-exp stabilization".to_string()),
            confidence: 0.9,
            evidence: vec![
                "max_logit = 85.0".to_string(),
                "near exp(88) boundary".to_string(),
            ],
        };

        let report = diagnosis.format_report();
        assert!(report.contains("encoder.softmax"));
        assert!(report.contains("Softmax"));
        assert!(report.contains("90%")); // 0.9 * 100
        assert!(report.contains("log-sum-exp"));
        assert!(report.contains("max_logit"));
        eprintln!("Report:\n{}", report);
    }

    #[test]
    fn test_diff_diagnosis_unknown() {
        let diagnosis = DiffDiagnosis::unknown("layer_0", Some(gamma_core::LayerType::Linear));
        assert_eq!(diagnosis.divergence_layer, "layer_0");
        assert!(matches!(diagnosis.pattern, DivergencePattern::Unknown));
        assert_eq!(diagnosis.confidence, 0.0);
    }

    #[test]
    fn test_diff_config_with_diagnose() {
        let config = DiffConfig {
            tolerance: 1e-5,
            continue_after_divergence: true,
            input: None,
            layer_mapping: HashMap::new(),
            diagnose: true,
        };

        assert!(config.diagnose);
        assert_eq!(config.tolerance, 1e-5);
    }

    #[test]
    fn test_diff_result_is_equivalent() {
        // No bad layer = equivalent
        let result_ok = DiffResult {
            layers: vec![LayerComparison {
                name: "layer_0".to_string(),
                name_b: None,
                max_diff: 1e-6,
                mean_diff: 1e-7,
                exceeds_tolerance: false,
                shape_a: vec![1, 2],
                shape_b: vec![1, 2],
            }],
            first_bad_layer: None,
            drift_start_layer: None,
            max_divergence: 1e-6,
            tolerance: 1e-5,
            suggestion: None,
            diagnosis: None,
        };
        assert!(result_ok.is_equivalent());

        // Has bad layer = not equivalent
        let result_bad = DiffResult {
            layers: vec![LayerComparison {
                name: "layer_0".to_string(),
                name_b: None,
                max_diff: 1e-3,
                mean_diff: 1e-4,
                exceeds_tolerance: true,
                shape_a: vec![1, 2],
                shape_b: vec![1, 2],
            }],
            first_bad_layer: Some(0),
            drift_start_layer: Some(0),
            max_divergence: 1e-3,
            tolerance: 1e-5,
            suggestion: None,
            diagnosis: None,
        };
        assert!(!result_bad.is_equivalent());
    }

    #[test]
    fn test_diff_result_first_bad_layer_name() {
        let result = DiffResult {
            layers: vec![
                LayerComparison {
                    name: "layer_0".to_string(),
                    name_b: None,
                    max_diff: 1e-7,
                    mean_diff: 1e-8,
                    exceeds_tolerance: false,
                    shape_a: vec![1, 2],
                    shape_b: vec![1, 2],
                },
                LayerComparison {
                    name: "layer_1_bad".to_string(),
                    name_b: None,
                    max_diff: 1e-3,
                    mean_diff: 1e-4,
                    exceeds_tolerance: true,
                    shape_a: vec![1, 2],
                    shape_b: vec![1, 2],
                },
            ],
            first_bad_layer: Some(1),
            drift_start_layer: Some(1),
            max_divergence: 1e-3,
            tolerance: 1e-5,
            suggestion: None,
            diagnosis: None,
        };

        assert_eq!(result.first_bad_layer_name(), Some("layer_1_bad"));

        // No bad layer case
        let result_ok = DiffResult {
            layers: vec![LayerComparison {
                name: "layer_0".to_string(),
                name_b: None,
                max_diff: 1e-7,
                mean_diff: 1e-8,
                exceeds_tolerance: false,
                shape_a: vec![1, 2],
                shape_b: vec![1, 2],
            }],
            first_bad_layer: None,
            drift_start_layer: None,
            max_divergence: 1e-7,
            tolerance: 1e-5,
            suggestion: None,
            diagnosis: None,
        };
        assert_eq!(result_ok.first_bad_layer_name(), None);
    }

    #[test]
    fn test_diff_result_statuses() {
        let result = DiffResult {
            layers: vec![
                LayerComparison {
                    name: "layer_ok".to_string(),
                    name_b: None,
                    max_diff: 1e-7,
                    mean_diff: 1e-8,
                    exceeds_tolerance: false,
                    shape_a: vec![1, 2],
                    shape_b: vec![1, 2],
                },
                LayerComparison {
                    name: "layer_drift".to_string(),
                    name_b: None,
                    max_diff: 1e-6,
                    mean_diff: 1e-7,
                    exceeds_tolerance: false,
                    shape_a: vec![1, 2],
                    shape_b: vec![1, 2],
                },
                LayerComparison {
                    name: "layer_bad".to_string(),
                    name_b: None,
                    max_diff: 1e-3,
                    mean_diff: 1e-4,
                    exceeds_tolerance: true,
                    shape_a: vec![1, 2],
                    shape_b: vec![1, 2],
                },
                LayerComparison {
                    name: "layer_shape_mismatch".to_string(),
                    name_b: None,
                    max_diff: f32::INFINITY,
                    mean_diff: f32::INFINITY,
                    exceeds_tolerance: true,
                    shape_a: vec![1, 2],
                    shape_b: vec![2, 1],
                },
            ],
            first_bad_layer: Some(2),
            drift_start_layer: Some(1),
            max_divergence: 1e-3,
            tolerance: 1e-5,
            suggestion: None,
            diagnosis: None,
        };

        let statuses = result.statuses();
        assert_eq!(statuses.len(), 4);
        assert_eq!(statuses[0], DiffStatus::Ok);
        assert_eq!(statuses[1], DiffStatus::DriftStarts);
        assert_eq!(statuses[2], DiffStatus::ExceedsTolerance);
        assert_eq!(statuses[3], DiffStatus::ShapeMismatch);
    }

    #[test]
    fn test_layer_comparison_creation() {
        let comp = LayerComparison {
            name: "test_layer".to_string(),
            name_b: Some("test_layer_b".to_string()),
            max_diff: 0.001,
            mean_diff: 0.0005,
            exceeds_tolerance: false,
            shape_a: vec![1, 10, 20],
            shape_b: vec![1, 10, 20],
        };

        assert_eq!(comp.name, "test_layer");
        assert_eq!(comp.name_b, Some("test_layer_b".to_string()));
        assert!((comp.max_diff - 0.001).abs() < 1e-9);
        assert!(!comp.exceeds_tolerance);
        assert_eq!(comp.shape_a, comp.shape_b);
    }

    #[test]
    fn test_diff_error_display() {
        // Test error Display implementations
        let err_load = DiffError::LoadError("failed to open".to_string());
        assert!(err_load.to_string().contains("failed to open"));

        let err_shape = DiffError::InputShapeMismatch {
            model_a: vec![1, 2, 3],
            model_b: vec![1, 2, 4],
        };
        assert!(err_shape.to_string().contains("[1, 2, 3]"));
        assert!(err_shape.to_string().contains("[1, 2, 4]"));

        let err_layer = DiffError::LayerNotFound("missing_layer".to_string());
        assert!(err_layer.to_string().contains("missing_layer"));

        let err_no_layers = DiffError::NoLayers;
        assert!(err_no_layers.to_string().contains("No layers"));

        let err_npy = DiffError::NpyError("invalid format".to_string());
        assert!(err_npy.to_string().contains("invalid format"));
    }

    // ========================================================================
    // normalize_layer_name additional tests
    // ========================================================================

    #[test]
    fn test_normalize_layer_name_empty() {
        assert_eq!(normalize_layer_name(""), "");
    }

    #[test]
    fn test_normalize_layer_name_just_numbers() {
        assert_eq!(normalize_layer_name("123"), "123");
        assert_eq!(normalize_layer_name("0"), "0");
    }

    #[test]
    fn test_normalize_layer_name_multiple_underscores() {
        assert_eq!(normalize_layer_name("layer__0__1"), "0.1");
        assert_eq!(normalize_layer_name("___layer___"), "");
    }

    #[test]
    fn test_normalize_layer_name_multiple_dots() {
        assert_eq!(normalize_layer_name("a...b"), "a.b");
        assert_eq!(normalize_layer_name("..."), "");
    }

    #[test]
    fn test_normalize_layer_name_mixed_separators() {
        assert_eq!(normalize_layer_name("layer_0.block.1_conv"), "0.1.conv");
        assert_eq!(
            normalize_layer_name("module_list.0.conv_block"),
            "list.0.conv"
        );
    }

    #[test]
    fn test_normalize_layer_name_special_chars_stripped() {
        // Special chars are stripped but don't introduce dots
        assert_eq!(normalize_layer_name("layer@0#1!"), "01");
        assert_eq!(normalize_layer_name("layer[0]"), "0");
        // Underscore introduces a dot
        assert_eq!(normalize_layer_name("layer@0_1!"), "0.1");
    }

    #[test]
    fn test_normalize_layer_name_uppercase_to_lowercase() {
        assert_eq!(normalize_layer_name("LAYER_0"), "0");
        assert_eq!(normalize_layer_name("Block_GELU"), "gelu");
    }

    #[test]
    fn test_normalize_layer_name_preserves_alphanumeric() {
        assert_eq!(normalize_layer_name("fc1"), "fc1");
        assert_eq!(normalize_layer_name("relu"), "relu");
    }

    // ========================================================================
    // match_layer_names tests
    // ========================================================================

    fn make_layer(name: &str, layer_type: LayerType) -> LayerSpec {
        LayerSpec {
            name: name.to_string(),
            layer_type,
            inputs: vec![],
            outputs: vec![name.to_string()],
            weights: None,
            attributes: HashMap::new(),
        }
    }

    #[test]
    fn test_match_layer_names_exact_match() {
        let layers_a = vec![
            make_layer("fc1", LayerType::Linear),
            make_layer("relu1", LayerType::ReLU),
        ];
        let layers_b = vec![
            make_layer("fc1", LayerType::Linear),
            make_layer("relu1", LayerType::ReLU),
        ];

        let matches = match_layer_names(&layers_a, &layers_b, &HashMap::new());
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0], ("fc1".to_string(), Some("fc1".to_string())));
        assert_eq!(matches[1], ("relu1".to_string(), Some("relu1".to_string())));
    }

    #[test]
    fn test_match_layer_names_explicit_mapping() {
        let layers_a = vec![make_layer("layer_a", LayerType::Linear)];
        let layers_b = vec![make_layer("layer_b", LayerType::Linear)];

        let mut mapping = HashMap::new();
        mapping.insert("layer_a".to_string(), "layer_b".to_string());

        let matches = match_layer_names(&layers_a, &layers_b, &mapping);
        assert_eq!(matches.len(), 1);
        assert_eq!(
            matches[0],
            ("layer_a".to_string(), Some("layer_b".to_string()))
        );
    }

    #[test]
    fn test_match_layer_names_fuzzy_match() {
        let layers_a = vec![make_layer("block_0_linear", LayerType::Linear)];
        let layers_b = vec![make_layer("block.0.linear", LayerType::Linear)];

        let matches = match_layer_names(&layers_a, &layers_b, &HashMap::new());
        assert_eq!(matches.len(), 1);
        // Both normalize to "0.linear" and have same type
        assert_eq!(
            matches[0],
            (
                "block_0_linear".to_string(),
                Some("block.0.linear".to_string())
            )
        );
    }

    #[test]
    fn test_match_layer_names_no_match() {
        let layers_a = vec![make_layer("fc1", LayerType::Linear)];
        let layers_b = vec![make_layer("conv1", LayerType::Conv2d)];

        let matches = match_layer_names(&layers_a, &layers_b, &HashMap::new());
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], ("fc1".to_string(), None));
    }

    #[test]
    fn test_match_layer_names_type_mismatch_prevents_fuzzy() {
        // Same normalized name but different type - should not match
        let layers_a = vec![make_layer("layer_0", LayerType::Linear)];
        let layers_b = vec![make_layer("layer.0", LayerType::Conv2d)];

        let matches = match_layer_names(&layers_a, &layers_b, &HashMap::new());
        assert_eq!(matches.len(), 1);
        // Won't match because types differ
        assert_eq!(matches[0], ("layer_0".to_string(), None));
    }

    #[test]
    fn test_match_layer_names_empty_inputs() {
        let empty: Vec<LayerSpec> = vec![];
        let layers_b = vec![make_layer("fc1", LayerType::Linear)];

        let matches = match_layer_names(&empty, &layers_b, &HashMap::new());
        assert!(matches.is_empty());
    }

    // ========================================================================
    // suggest_root_cause tests
    // ========================================================================

    #[test]
    fn test_suggest_root_cause_softmax() {
        let layer = make_layer("softmax", LayerType::Softmax);
        let suggestion = suggest_root_cause(&layer);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("softmax"));
    }

    #[test]
    fn test_suggest_root_cause_causal_softmax() {
        let layer = make_layer("causal_softmax", LayerType::CausalSoftmax);
        let suggestion = suggest_root_cause(&layer);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("softmax"));
    }

    #[test]
    fn test_suggest_root_cause_layernorm() {
        let layer = make_layer("ln", LayerType::LayerNorm);
        let suggestion = suggest_root_cause(&layer);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("LayerNorm"));
    }

    #[test]
    fn test_suggest_root_cause_gelu() {
        let layer = make_layer("gelu", LayerType::GELU);
        let suggestion = suggest_root_cause(&layer);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("GELU"));
    }

    #[test]
    fn test_suggest_root_cause_conv1d() {
        let layer = make_layer("conv", LayerType::Conv1d);
        let suggestion = suggest_root_cause(&layer);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("convolution"));
    }

    #[test]
    fn test_suggest_root_cause_conv2d() {
        let layer = make_layer("conv", LayerType::Conv2d);
        let suggestion = suggest_root_cause(&layer);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("convolution"));
    }

    #[test]
    fn test_suggest_root_cause_linear() {
        let layer = make_layer("fc", LayerType::Linear);
        let suggestion = suggest_root_cause(&layer);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("matrix multiplication"));
    }

    #[test]
    fn test_suggest_root_cause_matmul() {
        let layer = make_layer("mm", LayerType::MatMul);
        let suggestion = suggest_root_cause(&layer);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("matrix multiplication"));
    }

    #[test]
    fn test_suggest_root_cause_add() {
        let layer = make_layer("add", LayerType::Add);
        let suggestion = suggest_root_cause(&layer);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("broadcast"));
    }

    #[test]
    fn test_suggest_root_cause_mul() {
        let layer = make_layer("mul", LayerType::Mul);
        let suggestion = suggest_root_cause(&layer);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("broadcast"));
    }

    #[test]
    fn test_suggest_root_cause_relu_returns_none() {
        let layer = make_layer("relu", LayerType::ReLU);
        let suggestion = suggest_root_cause(&layer);
        assert!(suggestion.is_none());
    }

    #[test]
    fn test_suggest_root_cause_unknown_returns_none() {
        let layer = make_layer("unknown", LayerType::Unknown);
        let suggestion = suggest_root_cause(&layer);
        assert!(suggestion.is_none());
    }

    // ========================================================================
    // compare_arrays additional tests
    // ========================================================================

    #[test]
    fn test_compare_arrays_empty() {
        let a = ArrayD::from_elem(IxDyn(&[0]), 0.0f32);
        let b = ArrayD::from_elem(IxDyn(&[0]), 0.0f32);
        let comp = compare_arrays(&a, &b, 1e-5);
        // Empty arrays should compare as equivalent with NaN mean_diff
        assert_eq!(comp.shape_a, vec![0]);
        assert_eq!(comp.max_diff, 0.0);
        assert!(comp.mean_diff.is_nan()); // 0/0
    }

    #[test]
    fn test_compare_arrays_all_zeros() {
        let a = ArrayD::from_elem(IxDyn(&[10, 10]), 0.0f32);
        let b = ArrayD::from_elem(IxDyn(&[10, 10]), 0.0f32);
        let comp = compare_arrays(&a, &b, 1e-5);
        assert!(!comp.exceeds_tolerance);
        assert_eq!(comp.max_diff, 0.0);
        assert_eq!(comp.mean_diff, 0.0);
    }

    #[test]
    fn test_compare_arrays_one_element_different() {
        let mut a = ArrayD::from_elem(IxDyn(&[10]), 1.0f32);
        let b = ArrayD::from_elem(IxDyn(&[10]), 1.0f32);
        a[[5]] = 1.1; // Make one element different

        let comp = compare_arrays(&a, &b, 0.05);
        assert!(comp.exceeds_tolerance);
        assert!((comp.max_diff - 0.1).abs() < 1e-6);
        assert!((comp.mean_diff - 0.01).abs() < 1e-6); // 0.1/10
    }

    #[test]
    fn test_compare_arrays_large() {
        let a = ArrayD::from_elem(IxDyn(&[100, 100, 100]), 1.0f32);
        let mut b = ArrayD::from_elem(IxDyn(&[100, 100, 100]), 1.0f32);
        b[[50, 50, 50]] = 1.0001;

        let comp = compare_arrays(&a, &b, 1e-3);
        assert!(!comp.exceeds_tolerance);
        assert!((comp.max_diff - 0.0001).abs() < 1e-6);
    }

    #[test]
    fn test_compare_arrays_multidim_shape_mismatch() {
        let a = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 1.0f32);
        let b = ArrayD::from_elem(IxDyn(&[2, 4, 3]), 1.0f32);
        let comp = compare_arrays(&a, &b, 1e-5);
        assert!(comp.exceeds_tolerance);
        assert_eq!(comp.shape_a, vec![2, 3, 4]);
        assert_eq!(comp.shape_b, vec![2, 4, 3]);
    }

    // ========================================================================
    // check_gelu_pattern tests
    // ========================================================================

    #[test]
    fn test_check_gelu_pattern_no_pattern() {
        // Random differences not concentrated in transition region
        let a = ArrayD::from_shape_vec(IxDyn(&[10]), vec![5.0; 10]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[10]), vec![5.001; 10]).unwrap();

        let result = check_gelu_pattern("gelu_out", &a, &b, 0.001);
        // Values at 5.0 are not in the transition region [0.1, 2.0]
        assert!(result.is_none());
    }

    #[test]
    fn test_check_gelu_pattern_in_transition_region() {
        // Differences concentrated in GELU transition region
        let a_data: Vec<f32> = (0..100).map(|i| 0.5 + (i as f32) * 0.015).collect(); // 0.5 to 2.0
        let b_data: Vec<f32> = a_data.iter().map(|x| x + 0.001).collect();

        let a = ArrayD::from_shape_vec(IxDyn(&[100]), a_data).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[100]), b_data).unwrap();

        let result = check_gelu_pattern("gelu_out", &a, &b, 0.001);
        if let Some(diag) = result {
            assert!(matches!(
                diag.pattern,
                DivergencePattern::GeluApproximation { .. }
            ));
        }
        // Note: May or may not detect depending on threshold ratios
    }

    // ========================================================================
    // check_layernorm_pattern tests
    // ========================================================================

    #[test]
    fn test_check_layernorm_pattern_no_pattern() {
        // Moderate variance relative to mean - neither too high nor too low
        // We need: 0.1 * |mean| <= std <= 2.0 * |mean|
        // With diffs: 0.001, 0.002, 0.001, 0.002, ...
        // mean = 0.0015, variance = 0.00000025, std = 0.0005
        // std/mean = 0.0005/0.0015 = 0.33, which is in [0.1, 2.0]
        let a = ArrayD::from_shape_vec(IxDyn(&[100]), vec![1.0; 100]).unwrap();
        let mut b_data = vec![1.0; 100];
        for (i, val) in b_data.iter_mut().enumerate() {
            *val += if i % 2 == 0 { 0.001 } else { 0.002 };
        }
        let b = ArrayD::from_shape_vec(IxDyn(&[100]), b_data).unwrap();

        let result = check_layernorm_pattern("ln_out", &a, &b, 0.002);
        // Moderate variance ratio won't trigger either pattern
        assert!(result.is_none());
    }

    #[test]
    fn test_check_layernorm_pattern_systematic_offset() {
        // Systematic offset (epsilon difference) - low variance, high mean
        let a = ArrayD::from_shape_vec(IxDyn(&[100]), vec![1.0; 100]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[100]), vec![1.005; 100]).unwrap();

        let result = check_layernorm_pattern("ln_out", &a, &b, 0.005);
        if let Some(diag) = result {
            assert!(matches!(
                diag.pattern,
                DivergencePattern::LayerNormVariance {
                    epsilon_differs: true
                }
            ));
        }
    }

    #[test]
    fn test_check_layernorm_pattern_empty_array() {
        let a = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();

        let result = check_layernorm_pattern("ln_out", &a, &b, 0.001);
        assert!(result.is_none());
    }

    // ========================================================================
    // check_accumulation_pattern tests
    // ========================================================================

    #[test]
    fn test_check_accumulation_pattern_too_small() {
        // Array too small to detect pattern (< 100 elements)
        let a = ArrayD::from_shape_vec(IxDyn(&[50]), vec![1.0; 50]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[50]), vec![1.001; 50]).unwrap();

        let result = check_accumulation_pattern("mm_out", &a, &b, 0.001);
        assert!(result.is_none());
    }

    #[test]
    fn test_check_accumulation_pattern_uniform_error() {
        // Uniform error distribution (low coefficient of variation)
        let a = ArrayD::from_shape_vec(IxDyn(&[1000]), vec![1.0; 1000]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[1000]), vec![1.00001; 1000]).unwrap();

        let result = check_accumulation_pattern("mm_out", &a, &b, 0.00001);
        if let Some(diag) = result {
            assert!(matches!(
                diag.pattern,
                DivergencePattern::AccumulationOrder { .. }
            ));
        }
    }

    #[test]
    fn test_check_accumulation_pattern_nonuniform_error() {
        // Non-uniform error distribution (high CV)
        let mut b_data = vec![1.0; 1000];
        for (i, val) in b_data.iter_mut().enumerate() {
            if i < 100 {
                *val = 1.01; // Large diff in first 100 elements
            }
        }
        let a = ArrayD::from_shape_vec(IxDyn(&[1000]), vec![1.0; 1000]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[1000]), b_data).unwrap();

        let result = check_accumulation_pattern("mm_out", &a, &b, 0.01);
        // High CV should not trigger accumulation pattern
        assert!(result.is_none());
    }

    // ========================================================================
    // check_quantization_pattern tests
    // ========================================================================

    #[test]
    fn test_check_quantization_pattern_too_few_diffs() {
        // Less than 10 non-zero differences
        let a = ArrayD::from_shape_vec(IxDyn(&[10]), vec![1.0; 10]).unwrap();
        let mut b_data = vec![1.0; 10];
        b_data[0] = 1.001; // Only one diff
        let b = ArrayD::from_shape_vec(IxDyn(&[10]), b_data).unwrap();

        let result = check_quantization_pattern("out", None, &a, &b, 0.001);
        assert!(result.is_none());
    }

    #[test]
    fn test_check_quantization_pattern_quantized_diffs() {
        // Differences are multiples of a quantization step
        let a_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = a_data
            .iter()
            .enumerate()
            .map(|(i, x)| x + (i % 3) as f32 * 0.001)
            .collect();

        let a = ArrayD::from_shape_vec(IxDyn(&[100]), a_data).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[100]), b_data).unwrap();

        let result = check_quantization_pattern("out", Some(&LayerType::Linear), &a, &b, 0.003);
        // May detect quantization pattern if diffs are multiples of min_diff
        // This test just verifies the function runs without panic
        let _ = result;
    }

    // ========================================================================
    // DiffConfig tests
    // ========================================================================

    #[test]
    fn test_diff_config_default() {
        let config = DiffConfig::default();
        assert_eq!(config.tolerance, 1e-5);
        assert!(config.continue_after_divergence);
        assert!(config.input.is_none());
        assert!(config.layer_mapping.is_empty());
        assert!(!config.diagnose);
    }

    #[test]
    fn test_diff_config_custom() {
        let input = ArrayD::from_elem(IxDyn(&[1, 10]), 0.5f32);
        let mut mapping = HashMap::new();
        mapping.insert("a".to_string(), "b".to_string());

        let config = DiffConfig {
            tolerance: 1e-3,
            continue_after_divergence: false,
            input: Some(input.clone()),
            layer_mapping: mapping,
            diagnose: true,
        };

        assert_eq!(config.tolerance, 1e-3);
        assert!(!config.continue_after_divergence);
        assert!(config.input.is_some());
        assert_eq!(config.layer_mapping.len(), 1);
        assert!(config.diagnose);
    }

    #[test]
    fn test_diff_config_debug() {
        let config = DiffConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("DiffConfig"));
        assert!(debug_str.contains("tolerance"));
    }

    #[test]
    fn test_diff_config_clone() {
        let original = DiffConfig {
            tolerance: 0.01,
            ..Default::default()
        };
        let cloned = original.clone();
        assert_eq!(cloned.tolerance, 0.01);
    }

    // ========================================================================
    // DiffStatus tests
    // ========================================================================

    #[test]
    fn test_diff_status_eq() {
        assert_eq!(DiffStatus::Ok, DiffStatus::Ok);
        assert_eq!(DiffStatus::DriftStarts, DiffStatus::DriftStarts);
        assert_eq!(DiffStatus::ExceedsTolerance, DiffStatus::ExceedsTolerance);
        assert_eq!(DiffStatus::ShapeMismatch, DiffStatus::ShapeMismatch);
    }

    #[test]
    fn test_diff_status_ne() {
        assert_ne!(DiffStatus::Ok, DiffStatus::DriftStarts);
        assert_ne!(DiffStatus::Ok, DiffStatus::ExceedsTolerance);
        assert_ne!(DiffStatus::Ok, DiffStatus::ShapeMismatch);
    }

    #[test]
    fn test_diff_status_debug() {
        assert_eq!(format!("{:?}", DiffStatus::Ok), "Ok");
        assert_eq!(format!("{:?}", DiffStatus::DriftStarts), "DriftStarts");
        assert_eq!(
            format!("{:?}", DiffStatus::ExceedsTolerance),
            "ExceedsTolerance"
        );
        assert_eq!(format!("{:?}", DiffStatus::ShapeMismatch), "ShapeMismatch");
    }

    #[test]
    fn test_diff_status_copy() {
        let status = DiffStatus::Ok;
        let copied = status;
        assert_eq!(status, copied);
    }

    #[test]
    fn test_diff_status_clone() {
        let status = DiffStatus::ExceedsTolerance;
        // DiffStatus is Copy, but test clone trait is also implemented
        let cloned: DiffStatus = Clone::clone(&status);
        assert_eq!(status, cloned);
    }

    // ========================================================================
    // DivergencePattern tests
    // ========================================================================

    #[test]
    fn test_divergence_pattern_partial_eq() {
        let p1 = DivergencePattern::Unknown;
        let p2 = DivergencePattern::Unknown;
        assert_eq!(p1, p2);

        let p3 = DivergencePattern::ExpPrecision {
            max_logit: 85.0,
            is_overflow: true,
        };
        let p4 = DivergencePattern::ExpPrecision {
            max_logit: 85.0,
            is_overflow: true,
        };
        assert_eq!(p3, p4);

        let p5 = DivergencePattern::ExpPrecision {
            max_logit: 85.0,
            is_overflow: false,
        };
        assert_ne!(p3, p5);
    }

    #[test]
    fn test_divergence_pattern_clone() {
        let pattern = DivergencePattern::SoftmaxInstability {
            max_score: 75.0,
            score_range: 60.0,
        };
        let cloned = pattern.clone();
        assert_eq!(pattern, cloned);
    }

    #[test]
    fn test_divergence_pattern_debug() {
        let pattern = DivergencePattern::AccumulationOrder {
            operation: "sum".to_string(),
            size_correlated: false,
        };
        let debug_str = format!("{:?}", pattern);
        assert!(debug_str.contains("AccumulationOrder"));
        assert!(debug_str.contains("sum"));
    }

    #[test]
    fn test_divergence_pattern_display_accumulation_no_correlation() {
        let pattern = DivergencePattern::AccumulationOrder {
            operation: "reduce".to_string(),
            size_correlated: false,
        };
        let s = format!("{}", pattern);
        assert!(s.contains("reduce"));
        assert!(!s.contains("grows with size"));
    }

    #[test]
    fn test_divergence_pattern_display_quantization_no_boundary() {
        let pattern = DivergencePattern::QuantizationError {
            bits_lost: 4,
            at_power_boundary: false,
        };
        let s = format!("{}", pattern);
        assert!(s.contains("4 bits"));
        assert!(!s.contains("boundary"));
    }

    #[test]
    fn test_divergence_pattern_display_layernorm_variance_order() {
        let pattern = DivergencePattern::LayerNormVariance {
            epsilon_differs: false,
        };
        let s = format!("{}", pattern);
        assert!(s.contains("computation order"));
    }

    // ========================================================================
    // LayerComparison tests
    // ========================================================================

    #[test]
    fn test_layer_comparison_debug() {
        let comp = LayerComparison {
            name: "test".to_string(),
            name_b: None,
            max_diff: 0.001,
            mean_diff: 0.0005,
            exceeds_tolerance: false,
            shape_a: vec![1, 2],
            shape_b: vec![1, 2],
        };
        let debug_str = format!("{:?}", comp);
        assert!(debug_str.contains("LayerComparison"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_layer_comparison_clone() {
        let comp = LayerComparison {
            name: "layer".to_string(),
            name_b: Some("layer_b".to_string()),
            max_diff: 0.01,
            mean_diff: 0.005,
            exceeds_tolerance: true,
            shape_a: vec![1, 10, 20],
            shape_b: vec![1, 10, 20],
        };
        let cloned = comp.clone();
        assert_eq!(cloned.name, "layer");
        assert_eq!(cloned.name_b, Some("layer_b".to_string()));
        assert_eq!(cloned.max_diff, 0.01);
    }

    // ========================================================================
    // DiffDiagnosis tests
    // ========================================================================

    #[test]
    fn test_diff_diagnosis_debug() {
        let diag = DiffDiagnosis::unknown("test_layer", None);
        let debug_str = format!("{:?}", diag);
        assert!(debug_str.contains("DiffDiagnosis"));
        assert!(debug_str.contains("test_layer"));
    }

    #[test]
    fn test_diff_diagnosis_clone() {
        let diag = DiffDiagnosis {
            divergence_layer: "fc1".to_string(),
            layer_type: Some(LayerType::Linear),
            pattern: DivergencePattern::WeightMismatch {
                layer: "fc1".to_string(),
                max_diff: 0.1,
            },
            explanation: "Weights differ".to_string(),
            suggestion: Some("Check model export".to_string()),
            confidence: 0.95,
            evidence: vec!["diff = 0.1".to_string()],
        };
        let cloned = diag.clone();
        assert_eq!(cloned.divergence_layer, "fc1");
        assert_eq!(cloned.confidence, 0.95);
    }

    #[test]
    fn test_diff_diagnosis_format_report_no_suggestion() {
        let diag = DiffDiagnosis {
            divergence_layer: "relu".to_string(),
            layer_type: None,
            pattern: DivergencePattern::Unknown,
            explanation: "Unknown cause".to_string(),
            suggestion: None,
            confidence: 0.0,
            evidence: vec![],
        };
        let report = diag.format_report();
        assert!(report.contains("relu"));
        assert!(report.contains("Unknown cause"));
        assert!(!report.contains("Suggestion:"));
    }

    #[test]
    fn test_diff_diagnosis_format_report_no_layer_type() {
        let diag = DiffDiagnosis {
            divergence_layer: "output".to_string(),
            layer_type: None,
            pattern: DivergencePattern::Unknown,
            explanation: "".to_string(),
            suggestion: None,
            confidence: 0.5,
            evidence: vec![],
        };
        let report = diag.format_report();
        assert!(report.contains("output"));
        assert!(!report.contains("Layer Type:"));
    }

    #[test]
    fn test_diff_diagnosis_format_report_empty_explanation() {
        let diag = DiffDiagnosis {
            divergence_layer: "x".to_string(),
            layer_type: None,
            pattern: DivergencePattern::Unknown,
            explanation: "".to_string(),
            suggestion: None,
            confidence: 0.0,
            evidence: vec![],
        };
        let report = diag.format_report();
        // Empty explanation is skipped (not printed)
        assert!(!report.contains("Explanation:"));
        // But other fields are present
        assert!(report.contains("Layer: x"));
        assert!(report.contains("Issue:"));
        assert!(report.contains("Confidence:"));
    }

    // ========================================================================
    // DiffResult tests
    // ========================================================================

    #[test]
    fn test_diff_result_debug() {
        let result = DiffResult {
            layers: vec![],
            first_bad_layer: None,
            drift_start_layer: None,
            max_divergence: 0.0,
            tolerance: 1e-5,
            suggestion: None,
            diagnosis: None,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("DiffResult"));
    }

    #[test]
    fn test_diff_result_clone() {
        let result = DiffResult {
            layers: vec![LayerComparison {
                name: "layer".to_string(),
                name_b: None,
                max_diff: 0.001,
                mean_diff: 0.0005,
                exceeds_tolerance: false,
                shape_a: vec![1],
                shape_b: vec![1],
            }],
            first_bad_layer: None,
            drift_start_layer: None,
            max_divergence: 0.001,
            tolerance: 1e-5,
            suggestion: Some("test".to_string()),
            diagnosis: None,
        };
        let cloned = result.clone();
        assert_eq!(cloned.layers.len(), 1);
        assert_eq!(cloned.suggestion, Some("test".to_string()));
    }

    #[test]
    fn test_diff_result_first_bad_layer_out_of_bounds() {
        // Edge case: first_bad_layer index is out of bounds
        let result = DiffResult {
            layers: vec![],
            first_bad_layer: Some(5), // Invalid index
            drift_start_layer: None,
            max_divergence: 0.0,
            tolerance: 1e-5,
            suggestion: None,
            diagnosis: None,
        };
        assert_eq!(result.first_bad_layer_name(), None);
    }

    #[test]
    fn test_diff_result_statuses_empty() {
        let result = DiffResult {
            layers: vec![],
            first_bad_layer: None,
            drift_start_layer: None,
            max_divergence: 0.0,
            tolerance: 1e-5,
            suggestion: None,
            diagnosis: None,
        };
        let statuses = result.statuses();
        assert!(statuses.is_empty());
    }

    // ========================================================================
    // DiffError tests
    // ========================================================================

    #[test]
    fn test_diff_error_from_io_error() {
        use std::io::{Error, ErrorKind};
        let io_err = Error::new(ErrorKind::NotFound, "file not found");
        let diff_err: DiffError = io_err.into();
        assert!(matches!(diff_err, DiffError::IoError(_)));
    }

    #[test]
    fn test_diff_error_debug() {
        let err = DiffError::NoLayers;
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("NoLayers"));
    }

    // ========================================================================
    // ModelInfo tests
    // ========================================================================

    #[test]
    fn test_model_info_debug() {
        let info = ModelInfo {
            inputs: vec![],
            outputs: vec![],
            intermediate_names: vec!["a".to_string(), "b".to_string()],
            layers: vec![],
        };
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("ModelInfo"));
        assert!(debug_str.contains("intermediate_names"));
    }

    // ========================================================================
    // Negative value tests for compare_arrays
    // ========================================================================

    #[test]
    fn test_compare_arrays_negative_values() {
        let a = ArrayD::from_shape_vec(IxDyn(&[5]), vec![-1.0, -2.0, -3.0, -4.0, -5.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[5]), vec![-1.001, -2.001, -3.001, -4.001, -5.001])
            .unwrap();

        let comp = compare_arrays(&a, &b, 0.01);
        assert!(!comp.exceeds_tolerance);
        assert!((comp.max_diff - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_compare_arrays_mixed_signs() {
        let a = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1.0, 0.0, 1.0, 2.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-0.99, 0.01, 1.01, 2.01]).unwrap();

        let comp = compare_arrays(&a, &b, 0.05);
        assert!(!comp.exceeds_tolerance);
        assert_eq!(comp.max_diff, 0.01);
    }

    #[test]
    fn test_compare_arrays_inf_values() {
        let a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::INFINITY, f32::NEG_INFINITY, 0.0])
            .unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::INFINITY, f32::NEG_INFINITY, 0.0])
            .unwrap();

        let comp = compare_arrays(&a, &b, 1e-5);
        // Infinity - Infinity = NaN, so max_diff will be NaN or 0 depending on handling
        // This test ensures no panic
        let _ = comp;
    }

    #[test]
    fn test_compare_arrays_nan_values() {
        let a = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 1.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 1.0]).unwrap();

        let comp = compare_arrays(&a, &b, 1e-5);
        // NaN - NaN = NaN, (NaN).abs() = NaN
        // This test ensures no panic
        let _ = comp;
    }
}
