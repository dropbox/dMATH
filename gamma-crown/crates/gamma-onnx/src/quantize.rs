//! Quantization safety analysis for neural networks.
//!
//! This module analyzes whether a neural network's layer outputs can safely
//! be quantized to lower precision formats (float16, int8) without overflow
//! or significant precision loss.
//!
//! ## Key Analysis
//!
//! For each layer, we compute output bounds and check:
//! - **float16 safety**: Can outputs fit in [-65504, 65504]?
//! - **int8 safety**: Can scaled outputs fit in [-128, 127]?
//! - **Denormal risk**: Are outputs in the denormal range for float16?
//!
//! ## Usage
//!
//! ```ignore
//! use gamma_onnx::quantize::{analyze_quantization, QuantizeConfig};
//!
//! let config = QuantizeConfig::default();
//! let result = analyze_quantization("model.onnx", &config)?;
//! println!("{}", result.summary());
//! ```

use crate::{load_onnx, OnnxModel};
use gamma_propagate::BoundPropagation;
use gamma_tensor::BoundedTensor;
use ndarray::{ArrayD, IxDyn};
use std::path::Path;
use thiserror::Error;
use tracing::{debug, info};

/// float16 representation limits
const FLOAT16_MAX: f32 = 65504.0;
const FLOAT16_MIN_POSITIVE: f32 = 6.10e-5; // Minimum positive normal float16
#[allow(dead_code)]
const FLOAT16_DENORM_MIN: f32 = 5.96e-8; // Minimum positive denormal float16

/// int8 representation limits (for documentation, used for scale calculations)
#[allow(dead_code)]
const INT8_MIN: f32 = -128.0;
#[allow(dead_code)]
const INT8_MAX: f32 = 127.0;

/// Errors that can occur during quantization analysis.
#[derive(Error, Debug)]
pub enum QuantizeError {
    #[error("Failed to load model: {0}")]
    LoadError(String),

    #[error("Propagation error: {0}")]
    PropagationError(String),

    #[error("No layers in network")]
    NoLayers,

    #[error("Invalid input shape: {0}")]
    InvalidInputShape(String),
}

/// Quantization format being checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    Float16,
    Int8,
}

impl std::fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantFormat::Float16 => write!(f, "float16"),
            QuantFormat::Int8 => write!(f, "int8"),
        }
    }
}

/// Safety status for a quantization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantSafety {
    /// Safe to quantize - all values within representable range
    Safe,
    /// Warning - values may be in denormal range (precision loss)
    Denormal,
    /// Warning - values require careful scaling for int8
    ScalingRequired,
    /// Unsafe - values may overflow the format
    Overflow,
    /// Unknown - bounds are infinite or NaN
    Unknown,
}

impl std::fmt::Display for QuantSafety {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantSafety::Safe => write!(f, "SAFE"),
            QuantSafety::Denormal => write!(f, "DENORMAL"),
            QuantSafety::ScalingRequired => write!(f, "SCALE"),
            QuantSafety::Overflow => write!(f, "OVERFLOW"),
            QuantSafety::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// Result of analyzing a single layer's quantization safety.
#[derive(Debug, Clone)]
pub struct LayerQuantization {
    /// Layer name from the model.
    pub name: String,
    /// Layer type (e.g., "Linear", "ReLU", "Softmax").
    pub layer_type: String,
    /// Minimum output bound across all elements.
    pub min_bound: f32,
    /// Maximum output bound across all elements.
    pub max_bound: f32,
    /// Maximum absolute value in bounds.
    pub max_abs: f32,
    /// Output shape.
    pub output_shape: Vec<usize>,
    /// float16 safety assessment.
    pub float16_safety: QuantSafety,
    /// int8 safety assessment.
    pub int8_safety: QuantSafety,
    /// Suggested int8 scale factor (if applicable).
    pub int8_scale: Option<f32>,
    /// Whether any output bounds are infinite or NaN.
    pub has_overflow: bool,
}

impl LayerQuantization {
    /// Check if layer is safe for the given format.
    pub fn is_safe_for(&self, format: QuantFormat) -> bool {
        match format {
            QuantFormat::Float16 => matches!(self.float16_safety, QuantSafety::Safe),
            QuantFormat::Int8 => matches!(
                self.int8_safety,
                QuantSafety::Safe | QuantSafety::ScalingRequired
            ),
        }
    }
}

/// Result of a full quantization analysis.
#[derive(Debug, Clone)]
pub struct QuantizationResult {
    /// Per-layer quantization analysis.
    pub layers: Vec<LayerQuantization>,
    /// Overall float16 safety (all layers must be safe).
    pub float16_safe: bool,
    /// Overall int8 safety (all layers must be safe with scaling).
    pub int8_safe: bool,
    /// Number of layers with float16 overflow risk.
    pub float16_overflow_count: usize,
    /// Number of layers with int8 overflow risk.
    pub int8_overflow_count: usize,
    /// Number of layers in float16 denormal range.
    pub denormal_count: usize,
    /// Input perturbation epsilon used.
    pub input_epsilon: f32,
}

impl QuantizationResult {
    /// Get a summary of the analysis.
    pub fn summary(&self) -> String {
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
                layer.float16_safety,
                layer.int8_safety
            ));
        }

        lines.push(String::new());
        lines.push(format!(
            "Float16: {} ({} layers with overflow risk, {} in denormal range)",
            if self.float16_safe { "SAFE" } else { "UNSAFE" },
            self.float16_overflow_count,
            self.denormal_count
        ));
        lines.push(format!(
            "Int8:    {} ({} layers with overflow risk)",
            if self.int8_safe { "SAFE" } else { "UNSAFE" },
            self.int8_overflow_count
        ));

        lines.join("\n")
    }

    /// Get layers that are unsafe for float16.
    pub fn float16_unsafe_layers(&self) -> Vec<&LayerQuantization> {
        self.layers
            .iter()
            .filter(|l| {
                matches!(
                    l.float16_safety,
                    QuantSafety::Overflow | QuantSafety::Unknown
                )
            })
            .collect()
    }

    /// Get layers that are unsafe for int8.
    pub fn int8_unsafe_layers(&self) -> Vec<&LayerQuantization> {
        self.layers
            .iter()
            .filter(|l| matches!(l.int8_safety, QuantSafety::Overflow | QuantSafety::Unknown))
            .collect()
    }

    /// Get layers with denormal warning.
    pub fn denormal_layers(&self) -> Vec<&LayerQuantization> {
        self.layers
            .iter()
            .filter(|l| matches!(l.float16_safety, QuantSafety::Denormal))
            .collect()
    }
}

/// Configuration for quantization analysis.
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// Input perturbation epsilon.
    pub epsilon: f32,
    /// Whether to continue after overflow.
    pub continue_after_overflow: bool,
    /// Custom input tensor (None = zeros with epsilon bounds).
    pub input: Option<BoundedTensor>,
    /// Formats to check.
    pub formats: Vec<QuantFormat>,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.01,
            continue_after_overflow: true,
            input: None,
            formats: vec![QuantFormat::Float16, QuantFormat::Int8],
        }
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

/// Assess float16 safety for given bounds.
fn assess_float16(min_bound: f32, max_bound: f32) -> QuantSafety {
    if !min_bound.is_finite() || !max_bound.is_finite() {
        return QuantSafety::Unknown;
    }

    let max_abs = min_bound.abs().max(max_bound.abs());

    if max_abs > FLOAT16_MAX {
        QuantSafety::Overflow
    } else if max_abs > 0.0 && max_abs < FLOAT16_MIN_POSITIVE {
        // Values in denormal range - precision loss likely
        QuantSafety::Denormal
    } else {
        QuantSafety::Safe
    }
}

/// Assess int8 safety for given bounds.
fn assess_int8(min_bound: f32, max_bound: f32) -> (QuantSafety, Option<f32>) {
    if !min_bound.is_finite() || !max_bound.is_finite() {
        return (QuantSafety::Unknown, None);
    }

    // Calculate the scale needed to fit in int8
    let max_abs = min_bound.abs().max(max_bound.abs());

    if max_abs == 0.0 {
        // Zero tensor - safe
        return (QuantSafety::Safe, Some(1.0));
    }

    // Scale to fit in [-127, 127] (reserving -128 for special values)
    let scale = 127.0 / max_abs;

    if scale < 1e-10 {
        // Scale too small - overflow
        (QuantSafety::Overflow, None)
    } else if scale < 1.0 {
        // Needs scaling down
        (QuantSafety::ScalingRequired, Some(scale))
    } else if scale > 1e6 {
        // Very small values - precision issues
        (QuantSafety::ScalingRequired, Some(scale))
    } else {
        (QuantSafety::Safe, Some(scale))
    }
}

/// Analyze quantization safety of a model loaded from ONNX file.
pub fn analyze_quantization(
    path: impl AsRef<Path>,
    config: &QuantizeConfig,
) -> Result<QuantizationResult, QuantizeError> {
    info!("Loading model: {}", path.as_ref().display());
    let onnx_model =
        load_onnx(path.as_ref()).map_err(|e| QuantizeError::LoadError(format!("{}", e)))?;

    analyze_quantization_model(&onnx_model, config)
}

/// Analyze quantization safety of an already-loaded ONNX model.
pub fn analyze_quantization_model(
    model: &OnnxModel,
    config: &QuantizeConfig,
) -> Result<QuantizationResult, QuantizeError> {
    // Convert to propagate network
    let network = model
        .to_propagate_network()
        .map_err(|e| QuantizeError::PropagationError(format!("{}", e)))?;

    if network.layers.is_empty() {
        return Err(QuantizeError::NoLayers);
    }

    // Create input tensor
    let input = if let Some(ref inp) = config.input {
        inp.clone()
    } else {
        let input_spec = model.network.inputs.first().ok_or_else(|| {
            QuantizeError::InvalidInputShape("No input specification".to_string())
        })?;

        let shape: Vec<usize> = input_spec
            .shape
            .iter()
            .map(|&d| if d > 0 { d as usize } else { 1 })
            .collect();

        let data = ArrayD::zeros(IxDyn(&shape));
        BoundedTensor::from_epsilon(data, config.epsilon)
    };

    info!(
        "Starting quantization analysis with input shape {:?}, epsilon {}",
        input.shape(),
        config.epsilon
    );

    // Track layer-by-layer quantization
    let mut layers = Vec::new();
    let mut current = input.clone();
    let mut float16_overflow_count = 0;
    let mut int8_overflow_count = 0;
    let mut denormal_count = 0;

    for (layer, spec) in network.layers.iter().zip(model.network.layers.iter()) {
        // Propagate through this layer
        let output = match layer.propagate_ibp(&current) {
            Ok(out) => out,
            Err(e) => {
                debug!("Layer {} propagation failed: {}", spec.name, e);
                if !config.continue_after_overflow {
                    return Err(QuantizeError::PropagationError(format!(
                        "Layer {} failed: {}",
                        spec.name, e
                    )));
                }
                current.clone()
            }
        };

        // Get bounds
        let min_bound = output.lower.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_bound = output
            .upper
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let max_abs = min_bound.abs().max(max_bound.abs());
        let has_overflow = !min_bound.is_finite() || !max_bound.is_finite();

        // Assess quantization safety
        let float16_safety = assess_float16(min_bound, max_bound);
        let (int8_safety, int8_scale) = assess_int8(min_bound, max_bound);

        // Count issues
        if matches!(float16_safety, QuantSafety::Overflow | QuantSafety::Unknown) {
            float16_overflow_count += 1;
        }
        if matches!(float16_safety, QuantSafety::Denormal) {
            denormal_count += 1;
        }
        if matches!(int8_safety, QuantSafety::Overflow | QuantSafety::Unknown) {
            int8_overflow_count += 1;
        }

        layers.push(LayerQuantization {
            name: spec.name.clone(),
            layer_type: format!("{:?}", spec.layer_type),
            min_bound,
            max_bound,
            max_abs,
            output_shape: output.shape().to_vec(),
            float16_safety,
            int8_safety,
            int8_scale,
            has_overflow,
        });

        debug!(
            "Layer {}: bounds [{:.3e}, {:.3e}], f16={}, i8={}",
            spec.name, min_bound, max_bound, float16_safety, int8_safety
        );

        // Stop if overflow and not continuing
        if has_overflow && !config.continue_after_overflow {
            break;
        }

        current = output;
    }

    Ok(QuantizationResult {
        layers,
        float16_safe: float16_overflow_count == 0,
        int8_safe: int8_overflow_count == 0,
        float16_overflow_count,
        int8_overflow_count,
        denormal_count,
        input_epsilon: config.epsilon,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== truncate_name tests =====

    #[test]
    fn test_truncate_name_shorter_than_width() {
        assert_eq!(truncate_name("short", 10), "short");
    }

    #[test]
    fn test_truncate_name_exact_width() {
        assert_eq!(truncate_name("exactly_10", 10), "exactly_10");
    }

    #[test]
    fn test_truncate_name_longer_than_width() {
        assert_eq!(truncate_name("very_long_layer_name", 10), "...er_name");
    }

    #[test]
    fn test_truncate_name_empty_string() {
        assert_eq!(truncate_name("", 10), "");
    }

    #[test]
    fn test_truncate_name_width_equals_length() {
        assert_eq!(truncate_name("abcde", 5), "abcde");
    }

    #[test]
    fn test_truncate_name_single_char_over() {
        // "abcdef" length 6, width 5 -> "...ef"
        assert_eq!(truncate_name("abcdef", 5), "...ef");
    }

    #[test]
    fn test_truncate_name_minimum_width() {
        // With width=4, we get "..." + 1 char
        assert_eq!(truncate_name("abcdefgh", 4), "...h");
    }

    // ===== assess_float16 tests =====

    #[test]
    fn test_float16_assessment_safe_small_values() {
        assert!(matches!(assess_float16(-100.0, 100.0), QuantSafety::Safe));
    }

    #[test]
    fn test_float16_assessment_safe_at_boundary() {
        assert!(matches!(
            assess_float16(-65504.0, 65504.0),
            QuantSafety::Safe
        ));
    }

    #[test]
    fn test_float16_assessment_overflow() {
        assert!(matches!(
            assess_float16(-70000.0, 70000.0),
            QuantSafety::Overflow
        ));
    }

    #[test]
    fn test_float16_assessment_denormal() {
        assert!(matches!(assess_float16(-1e-6, 1e-6), QuantSafety::Denormal));
    }

    #[test]
    fn test_float16_assessment_unknown_infinity() {
        assert!(matches!(
            assess_float16(f32::NEG_INFINITY, f32::INFINITY),
            QuantSafety::Unknown
        ));
    }

    #[test]
    fn test_float16_assessment_unknown_nan() {
        assert!(matches!(
            assess_float16(f32::NAN, 1.0),
            QuantSafety::Unknown
        ));
        assert!(matches!(
            assess_float16(1.0, f32::NAN),
            QuantSafety::Unknown
        ));
    }

    #[test]
    fn test_float16_assessment_unknown_neg_infinity() {
        assert!(matches!(
            assess_float16(f32::NEG_INFINITY, 0.0),
            QuantSafety::Unknown
        ));
    }

    #[test]
    fn test_float16_assessment_zero() {
        assert!(matches!(assess_float16(0.0, 0.0), QuantSafety::Safe));
    }

    #[test]
    fn test_float16_assessment_just_above_max() {
        // Exactly at the boundary + epsilon
        assert!(matches!(
            assess_float16(-65505.0, 65505.0),
            QuantSafety::Overflow
        ));
    }

    #[test]
    fn test_float16_assessment_asymmetric_bounds() {
        // Only one side exceeds
        assert!(matches!(
            assess_float16(-100.0, 70000.0),
            QuantSafety::Overflow
        ));
    }

    #[test]
    fn test_float16_assessment_negative_only_overflow() {
        assert!(matches!(
            assess_float16(-70000.0, 0.0),
            QuantSafety::Overflow
        ));
    }

    #[test]
    fn test_float16_assessment_small_positive_denormal() {
        // Very small positive value in denormal range
        assert!(matches!(assess_float16(0.0, 1e-6), QuantSafety::Denormal));
    }

    // ===== assess_int8 tests =====

    #[test]
    fn test_int8_assessment_safe() {
        let (safety, scale) = assess_int8(-100.0, 100.0);
        assert!(matches!(safety, QuantSafety::Safe));
        assert!(scale.is_some());
        // Scale should be 127.0 / 100.0 = 1.27
        assert!((scale.unwrap() - 1.27).abs() < 0.01);
    }

    #[test]
    fn test_int8_assessment_zero() {
        let (safety, scale) = assess_int8(0.0, 0.0);
        assert!(matches!(safety, QuantSafety::Safe));
        assert_eq!(scale, Some(1.0));
    }

    #[test]
    fn test_int8_assessment_scaling_required() {
        let (safety, scale) = assess_int8(-1000.0, 1000.0);
        assert!(matches!(safety, QuantSafety::ScalingRequired));
        assert!(scale.is_some());
        // Scale should be 127.0 / 1000.0 = 0.127
        assert!((scale.unwrap() - 0.127).abs() < 0.001);
    }

    #[test]
    fn test_int8_assessment_unknown_infinity() {
        let (safety, scale) = assess_int8(f32::NEG_INFINITY, f32::INFINITY);
        assert!(matches!(safety, QuantSafety::Unknown));
        assert!(scale.is_none());
    }

    #[test]
    fn test_int8_assessment_unknown_nan() {
        let (safety, scale) = assess_int8(f32::NAN, 0.0);
        assert!(matches!(safety, QuantSafety::Unknown));
        assert!(scale.is_none());
    }

    #[test]
    fn test_int8_assessment_very_small_values() {
        // Very small values that need large scale (precision issues)
        let (safety, scale) = assess_int8(-1e-8, 1e-8);
        assert!(matches!(safety, QuantSafety::ScalingRequired));
        assert!(scale.is_some());
    }

    #[test]
    fn test_int8_assessment_asymmetric_bounds() {
        // Asymmetric bounds - max_abs should use the larger absolute value
        let (safety, scale) = assess_int8(-50.0, 100.0);
        assert!(matches!(safety, QuantSafety::Safe));
        // Scale based on max_abs = 100
        assert!((scale.unwrap() - 1.27).abs() < 0.01);
    }

    #[test]
    fn test_int8_assessment_negative_only() {
        let (safety, scale) = assess_int8(-100.0, -50.0);
        assert!(matches!(safety, QuantSafety::Safe));
        // Scale based on max_abs = 100
        assert!((scale.unwrap() - 1.27).abs() < 0.01);
    }

    #[test]
    fn test_int8_assessment_overflow_very_large() {
        // If scale would be too small (< 1e-10), it's overflow
        let (safety, scale) = assess_int8(-1e15, 1e15);
        assert!(matches!(safety, QuantSafety::Overflow));
        assert!(scale.is_none());
    }

    // ===== QuantFormat Display tests =====

    #[test]
    fn test_quant_format_display_float16() {
        assert_eq!(format!("{}", QuantFormat::Float16), "float16");
    }

    #[test]
    fn test_quant_format_display_int8() {
        assert_eq!(format!("{}", QuantFormat::Int8), "int8");
    }

    // ===== QuantSafety Display tests =====

    #[test]
    fn test_quant_safety_display_safe() {
        assert_eq!(format!("{}", QuantSafety::Safe), "SAFE");
    }

    #[test]
    fn test_quant_safety_display_denormal() {
        assert_eq!(format!("{}", QuantSafety::Denormal), "DENORMAL");
    }

    #[test]
    fn test_quant_safety_display_scaling_required() {
        assert_eq!(format!("{}", QuantSafety::ScalingRequired), "SCALE");
    }

    #[test]
    fn test_quant_safety_display_overflow() {
        assert_eq!(format!("{}", QuantSafety::Overflow), "OVERFLOW");
    }

    #[test]
    fn test_quant_safety_display_unknown() {
        assert_eq!(format!("{}", QuantSafety::Unknown), "UNKNOWN");
    }

    // ===== LayerQuantization::is_safe_for tests =====

    #[test]
    fn test_layer_quantization_is_safe_for_float16_safe() {
        let layer = LayerQuantization {
            name: "test".to_string(),
            layer_type: "Linear".to_string(),
            min_bound: -100.0,
            max_bound: 100.0,
            max_abs: 100.0,
            output_shape: vec![1, 10],
            float16_safety: QuantSafety::Safe,
            int8_safety: QuantSafety::Safe,
            int8_scale: Some(1.27),
            has_overflow: false,
        };
        assert!(layer.is_safe_for(QuantFormat::Float16));
    }

    #[test]
    fn test_layer_quantization_is_safe_for_float16_overflow() {
        let layer = LayerQuantization {
            name: "test".to_string(),
            layer_type: "Linear".to_string(),
            min_bound: -70000.0,
            max_bound: 70000.0,
            max_abs: 70000.0,
            output_shape: vec![1, 10],
            float16_safety: QuantSafety::Overflow,
            int8_safety: QuantSafety::Overflow,
            int8_scale: None,
            has_overflow: false,
        };
        assert!(!layer.is_safe_for(QuantFormat::Float16));
    }

    #[test]
    fn test_layer_quantization_is_safe_for_float16_denormal() {
        let layer = LayerQuantization {
            name: "test".to_string(),
            layer_type: "Linear".to_string(),
            min_bound: -1e-6,
            max_bound: 1e-6,
            max_abs: 1e-6,
            output_shape: vec![1, 10],
            float16_safety: QuantSafety::Denormal,
            int8_safety: QuantSafety::Safe,
            int8_scale: Some(1.0),
            has_overflow: false,
        };
        // Denormal is not considered "safe" for float16
        assert!(!layer.is_safe_for(QuantFormat::Float16));
    }

    #[test]
    fn test_layer_quantization_is_safe_for_int8_safe() {
        let layer = LayerQuantization {
            name: "test".to_string(),
            layer_type: "Linear".to_string(),
            min_bound: -100.0,
            max_bound: 100.0,
            max_abs: 100.0,
            output_shape: vec![1, 10],
            float16_safety: QuantSafety::Safe,
            int8_safety: QuantSafety::Safe,
            int8_scale: Some(1.27),
            has_overflow: false,
        };
        assert!(layer.is_safe_for(QuantFormat::Int8));
    }

    #[test]
    fn test_layer_quantization_is_safe_for_int8_scaling_required() {
        let layer = LayerQuantization {
            name: "test".to_string(),
            layer_type: "Linear".to_string(),
            min_bound: -1000.0,
            max_bound: 1000.0,
            max_abs: 1000.0,
            output_shape: vec![1, 10],
            float16_safety: QuantSafety::Safe,
            int8_safety: QuantSafety::ScalingRequired,
            int8_scale: Some(0.127),
            has_overflow: false,
        };
        // ScalingRequired is considered safe for int8
        assert!(layer.is_safe_for(QuantFormat::Int8));
    }

    #[test]
    fn test_layer_quantization_is_safe_for_int8_overflow() {
        let layer = LayerQuantization {
            name: "test".to_string(),
            layer_type: "Linear".to_string(),
            min_bound: -1e15,
            max_bound: 1e15,
            max_abs: 1e15,
            output_shape: vec![1, 10],
            float16_safety: QuantSafety::Overflow,
            int8_safety: QuantSafety::Overflow,
            int8_scale: None,
            has_overflow: false,
        };
        assert!(!layer.is_safe_for(QuantFormat::Int8));
    }

    // ===== QuantizeConfig tests =====

    #[test]
    fn test_quantize_config_default() {
        let config = QuantizeConfig::default();
        assert!((config.epsilon - 0.01).abs() < 1e-6);
        assert!(config.continue_after_overflow);
        assert!(config.input.is_none());
        assert_eq!(config.formats.len(), 2);
        assert!(config.formats.contains(&QuantFormat::Float16));
        assert!(config.formats.contains(&QuantFormat::Int8));
    }

    // ===== QuantizationResult tests =====

    fn create_test_result() -> QuantizationResult {
        QuantizationResult {
            layers: vec![
                LayerQuantization {
                    name: "layer1".to_string(),
                    layer_type: "Linear".to_string(),
                    min_bound: -100.0,
                    max_bound: 100.0,
                    max_abs: 100.0,
                    output_shape: vec![1, 10],
                    float16_safety: QuantSafety::Safe,
                    int8_safety: QuantSafety::Safe,
                    int8_scale: Some(1.27),
                    has_overflow: false,
                },
                LayerQuantization {
                    name: "layer2".to_string(),
                    layer_type: "ReLU".to_string(),
                    min_bound: -70000.0,
                    max_bound: 70000.0,
                    max_abs: 70000.0,
                    output_shape: vec![1, 10],
                    float16_safety: QuantSafety::Overflow,
                    int8_safety: QuantSafety::ScalingRequired,
                    int8_scale: Some(0.00181),
                    has_overflow: false,
                },
                LayerQuantization {
                    name: "layer3".to_string(),
                    layer_type: "Softmax".to_string(),
                    min_bound: -1e-6,
                    max_bound: 1e-6,
                    max_abs: 1e-6,
                    output_shape: vec![1, 10],
                    float16_safety: QuantSafety::Denormal,
                    int8_safety: QuantSafety::Safe,
                    int8_scale: Some(1.0),
                    has_overflow: false,
                },
            ],
            float16_safe: false,
            int8_safe: true,
            float16_overflow_count: 1,
            int8_overflow_count: 0,
            denormal_count: 1,
            input_epsilon: 0.01,
        }
    }

    #[test]
    fn test_quantization_result_summary_contains_header() {
        let result = create_test_result();
        let summary = result.summary();
        assert!(summary.contains("Quantization Safety Analysis"));
        assert!(summary.contains("Layer"));
        assert!(summary.contains("Min"));
        assert!(summary.contains("Max"));
        assert!(summary.contains("F16"));
        assert!(summary.contains("I8"));
    }

    #[test]
    fn test_quantization_result_summary_contains_layers() {
        let result = create_test_result();
        let summary = result.summary();
        assert!(summary.contains("layer1"));
        assert!(summary.contains("layer2"));
        assert!(summary.contains("layer3"));
    }

    #[test]
    fn test_quantization_result_summary_contains_safety_status() {
        let result = create_test_result();
        let summary = result.summary();
        assert!(summary.contains("UNSAFE")); // float16
        assert!(summary.contains("SAFE")); // int8
    }

    #[test]
    fn test_quantization_result_float16_unsafe_layers() {
        let result = create_test_result();
        let unsafe_layers = result.float16_unsafe_layers();
        assert_eq!(unsafe_layers.len(), 1);
        assert_eq!(unsafe_layers[0].name, "layer2");
    }

    #[test]
    fn test_quantization_result_int8_unsafe_layers() {
        let result = create_test_result();
        let unsafe_layers = result.int8_unsafe_layers();
        assert!(unsafe_layers.is_empty());
    }

    #[test]
    fn test_quantization_result_denormal_layers() {
        let result = create_test_result();
        let denormal_layers = result.denormal_layers();
        assert_eq!(denormal_layers.len(), 1);
        assert_eq!(denormal_layers[0].name, "layer3");
    }

    #[test]
    fn test_quantization_result_empty_layers() {
        let result = QuantizationResult {
            layers: vec![],
            float16_safe: true,
            int8_safe: true,
            float16_overflow_count: 0,
            int8_overflow_count: 0,
            denormal_count: 0,
            input_epsilon: 0.01,
        };
        assert!(result.float16_unsafe_layers().is_empty());
        assert!(result.int8_unsafe_layers().is_empty());
        assert!(result.denormal_layers().is_empty());
    }

    // ===== QuantizeError tests =====

    #[test]
    fn test_quantize_error_display_load_error() {
        let err = QuantizeError::LoadError("test error".to_string());
        assert_eq!(format!("{}", err), "Failed to load model: test error");
    }

    #[test]
    fn test_quantize_error_display_propagation_error() {
        let err = QuantizeError::PropagationError("prop error".to_string());
        assert_eq!(format!("{}", err), "Propagation error: prop error");
    }

    #[test]
    fn test_quantize_error_display_no_layers() {
        let err = QuantizeError::NoLayers;
        assert_eq!(format!("{}", err), "No layers in network");
    }

    #[test]
    fn test_quantize_error_display_invalid_input_shape() {
        let err = QuantizeError::InvalidInputShape("bad shape".to_string());
        assert_eq!(format!("{}", err), "Invalid input shape: bad shape");
    }
}
