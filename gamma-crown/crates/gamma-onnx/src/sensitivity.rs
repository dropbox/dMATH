//! Sensitivity analysis for neural networks.
//!
//! This module provides tools to analyze how each layer in a neural network
//! amplifies input uncertainty. This is useful for:
//! - Finding unstable layers that explode bounds
//! - Identifying where verification becomes difficult
//! - Pre-quantization analysis (high sensitivity = quantization risk)
//!
//! ## Key Metric: Sensitivity (Amplification Factor)
//!
//! For each layer, we compute:
//!   sensitivity = output_bound_width / input_bound_width
//!
//! - sensitivity < 1.0: Layer contracts bounds (stable)
//! - sensitivity = 1.0: Layer preserves bounds (neutral)
//! - sensitivity > 1.0: Layer amplifies bounds (potentially unstable)
//!
//! High sensitivity layers are "choke points" for verification and
//! may be problematic for quantization.

use crate::{load_onnx, OnnxModel};
use gamma_propagate::BoundPropagation;
use gamma_propagate::GraphNetwork;
use gamma_tensor::BoundedTensor;
use ndarray::{ArrayD, IxDyn};
use std::path::Path;
use thiserror::Error;
use tracing::{debug, info};

/// Errors that can occur during sensitivity analysis.
#[derive(Error, Debug)]
pub enum SensitivityError {
    #[error("Failed to load model: {0}")]
    LoadError(String),

    #[error("Propagation error: {0}")]
    PropagationError(String),

    #[error("No layers in network")]
    NoLayers,

    #[error("Invalid input shape: {0}")]
    InvalidInputShape(String),
}

/// Result of analyzing a single layer's sensitivity.
#[derive(Debug, Clone)]
pub struct LayerSensitivity {
    /// Layer name from the model.
    pub name: String,
    /// Layer type (e.g., "Linear", "ReLU", "Softmax").
    pub layer_type: String,
    /// Input bound width (max width across all elements).
    pub input_width: f32,
    /// Output bound width (max width across all elements).
    pub output_width: f32,
    /// Sensitivity = output_width / input_width.
    /// >1 means the layer amplifies uncertainty.
    pub sensitivity: f32,
    /// Mean bound width at output.
    pub mean_output_width: f32,
    /// Output shape.
    pub output_shape: Vec<usize>,
    /// Whether any output bounds are infinite or NaN (actual numerical overflow).
    pub has_overflow: bool,
    /// Whether propagation failed (e.g., shape mismatch).
    /// If true, sensitivity is unreliable (fallback input used as output).
    pub propagation_failed: bool,
}

impl LayerSensitivity {
    /// Check if this layer is a high-sensitivity layer (amplifies significantly).
    pub fn is_high_sensitivity(&self, threshold: f32) -> bool {
        self.sensitivity > threshold
    }

    /// Check if this layer contracts bounds.
    pub fn is_contractive(&self) -> bool {
        self.sensitivity < 1.0
    }
}

/// Result of a full sensitivity analysis.
#[derive(Debug, Clone)]
pub struct SensitivityResult {
    /// Per-layer sensitivity analysis.
    pub layers: Vec<LayerSensitivity>,
    /// Total sensitivity (product of all layer sensitivities).
    /// This is the theoretical worst-case bound amplification.
    pub total_sensitivity: f32,
    /// Maximum single-layer sensitivity.
    pub max_sensitivity: f32,
    /// Index of the layer with maximum sensitivity.
    pub max_sensitivity_layer: Option<usize>,
    /// Initial input bound width.
    pub input_epsilon: f32,
    /// Final output bound width.
    pub final_width: f32,
    /// Index of first layer where overflow occurred (if any).
    pub overflow_at_layer: Option<usize>,
}

impl SensitivityResult {
    /// Get a summary of the analysis.
    pub fn summary(&self) -> String {
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
            // Status priority: propagation_failed > has_overflow > sensitivity thresholds
            let status = if layer.propagation_failed {
                "SKIPPED" // Propagation failed (e.g., shape mismatch) - sensitivity is unreliable
            } else if layer.has_overflow {
                "OVERFLOW" // Actual numerical overflow (Inf/NaN)
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
        lines.push(format!(
            "Total sensitivity: {:.2e} (product of all layers)",
            self.total_sensitivity
        ));
        lines.push(format!(
            "Max single-layer sensitivity: {:.2} at layer {}",
            self.max_sensitivity,
            self.max_sensitivity_layer
                .and_then(|i| self.layers.get(i))
                .map(|l| l.name.as_str())
                .unwrap_or("N/A")
        ));
        lines.push(format!(
            "Input epsilon: {:.2e} → Final width: {:.2e}",
            self.input_epsilon, self.final_width
        ));

        // Count skipped (propagation failed) and overflow layers
        let skipped_count = self.layers.iter().filter(|l| l.propagation_failed).count();
        let overflow_count = self
            .layers
            .iter()
            .filter(|l| l.has_overflow && !l.propagation_failed)
            .count();

        if skipped_count > 0 {
            lines.push(format!(
                "NOTE: {} layer(s) skipped due to propagation failure (shape mismatch)",
                skipped_count
            ));
        }

        if overflow_count > 0 {
            lines.push(format!(
                "WARNING: {} layer(s) have numerical overflow (Inf/NaN values)",
                overflow_count
            ));
        }

        if let Some(overflow_idx) = self.overflow_at_layer {
            let layer = self.layers.get(overflow_idx);
            if layer
                .map(|l| l.has_overflow && !l.propagation_failed)
                .unwrap_or(false)
            {
                lines.push(format!(
                    "First overflow at layer {} ({})",
                    overflow_idx,
                    layer.map(|l| l.name.as_str()).unwrap_or("unknown")
                ));
            }
        }

        lines.join("\n")
    }

    /// Get layers sorted by sensitivity (highest first).
    pub fn layers_by_sensitivity(&self) -> Vec<&LayerSensitivity> {
        let mut sorted: Vec<_> = self.layers.iter().collect();
        sorted.sort_by(|a, b| {
            b.sensitivity
                .partial_cmp(&a.sensitivity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Get "hot spots" - layers with sensitivity above a threshold.
    pub fn hot_spots(&self, threshold: f32) -> Vec<&LayerSensitivity> {
        self.layers
            .iter()
            .filter(|l| l.sensitivity > threshold)
            .collect()
    }
}

/// Configuration for sensitivity analysis.
#[derive(Debug, Clone)]
pub struct SensitivityConfig {
    /// Input perturbation epsilon.
    pub epsilon: f32,
    /// Whether to continue after overflow.
    pub continue_after_overflow: bool,
    /// Custom input tensor (None = zeros with epsilon bounds).
    pub input: Option<BoundedTensor>,
}

impl Default for SensitivityConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.01,
            continue_after_overflow: false,
            input: None,
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

/// Analyze sensitivity of a model loaded from ONNX file.
pub fn analyze_sensitivity(
    path: impl AsRef<Path>,
    config: &SensitivityConfig,
) -> Result<SensitivityResult, SensitivityError> {
    info!("Loading model: {}", path.as_ref().display());
    let onnx_model =
        load_onnx(path.as_ref()).map_err(|e| SensitivityError::LoadError(format!("{}", e)))?;

    analyze_sensitivity_model(&onnx_model, config)
}

/// Analyze sensitivity of an already-loaded ONNX model.
pub fn analyze_sensitivity_model(
    model: &OnnxModel,
    config: &SensitivityConfig,
) -> Result<SensitivityResult, SensitivityError> {
    // Create input tensor
    let input = if let Some(ref inp) = config.input {
        inp.clone()
    } else {
        // Get input shape from model
        let input_spec = model.network.inputs.first().ok_or_else(|| {
            SensitivityError::InvalidInputShape("No input specification".to_string())
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
        "Starting sensitivity analysis with input shape {:?}, epsilon {}",
        input.shape(),
        config.epsilon
    );

    // Prefer a graph-based analysis when the model contains binary nodes
    // (e.g., residual Add, bounded MatMul, MulBinary). Sequential sensitivity
    // uses `Network` and cannot represent DAGs correctly.
    if let Ok(graph) = model.to_graph_network() {
        if graph_requires_dag_sensitivity(&graph) {
            return analyze_sensitivity_graph(&graph, &input, config);
        }
    }

    // Fall back to sequential sensitivity analysis. If propagation panics (e.g., due to a
    // binary op slipping through), retry with DAG-based analysis instead of crashing.
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        analyze_sensitivity_sequential(model, &input, config)
    })) {
        Ok(res) => res,
        Err(_) => {
            debug!("Sequential sensitivity panicked; retrying with GraphNetwork analysis");
            let graph = model
                .to_graph_network()
                .map_err(|e| SensitivityError::PropagationError(format!("{}", e)))?;
            analyze_sensitivity_graph(&graph, &input, config)
        }
    }
}

fn graph_requires_dag_sensitivity(graph: &GraphNetwork) -> bool {
    graph
        .node_names()
        .iter()
        .filter_map(|n| graph.get_node(n))
        .any(|node| node.layer.is_binary())
}

fn analyze_sensitivity_sequential(
    model: &OnnxModel,
    input: &BoundedTensor,
    config: &SensitivityConfig,
) -> Result<SensitivityResult, SensitivityError> {
    // Convert to propagate network
    let network = model
        .to_propagate_network()
        .map_err(|e| SensitivityError::PropagationError(format!("{}", e)))?;

    if network.layers.is_empty() {
        return Err(SensitivityError::NoLayers);
    }

    // Track layer-by-layer sensitivity
    let mut layers = Vec::new();
    let mut current = input.clone();
    let mut total_sensitivity: f32 = 1.0;
    let mut max_sensitivity: f32 = 0.0;
    let mut max_sensitivity_layer: Option<usize> = None;
    let mut overflow_at_layer: Option<usize> = None;

    for (i, (layer, spec)) in network
        .layers
        .iter()
        .zip(model.network.layers.iter())
        .enumerate()
    {
        let input_width = current.max_width();

        // Propagate through this layer
        let mut propagation_failed = false;
        let output = match layer.propagate_ibp(&current) {
            Ok(out) => out,
            Err(e) => {
                debug!("Layer {} propagation failed: {}", spec.name, e);
                if !config.continue_after_overflow {
                    return Err(SensitivityError::PropagationError(format!(
                        "Layer {} failed: {}",
                        spec.name, e
                    )));
                }
                // Use current as fallback and mark as propagation failure
                propagation_failed = true;
                current.clone()
            }
        };

        let output_width = output.max_width();
        let mean_width = output.width().iter().sum::<f32>() / output.width().len().max(1) as f32;
        // Only check for numerical overflow if propagation succeeded
        let has_overflow = !propagation_failed
            && (!output_width.is_finite() || output.width().iter().any(|w| !w.is_finite()));

        // Calculate sensitivity
        let sensitivity = if input_width > 0.0 && input_width.is_finite() {
            output_width / input_width
        } else if output_width == 0.0 {
            1.0
        } else {
            f32::INFINITY
        };

        // Track max sensitivity (only for successful propagations)
        if !propagation_failed && sensitivity > max_sensitivity && sensitivity.is_finite() {
            max_sensitivity = sensitivity;
            max_sensitivity_layer = Some(i);
        }

        // Accumulate total sensitivity (product) - skip failed propagations
        if !propagation_failed && sensitivity.is_finite() {
            total_sensitivity *= sensitivity;
        } else if has_overflow {
            total_sensitivity = f32::INFINITY;
        }

        // Check for actual overflow (not propagation failure)
        if has_overflow && overflow_at_layer.is_none() {
            overflow_at_layer = Some(i);
        }

        layers.push(LayerSensitivity {
            name: spec.name.clone(),
            layer_type: format!("{:?}", spec.layer_type),
            input_width,
            output_width,
            sensitivity,
            mean_output_width: mean_width,
            output_shape: output.shape().to_vec(),
            has_overflow,
            propagation_failed,
        });

        debug!(
            "Layer {}: {:?} -> sensitivity = {:.3}",
            spec.name, spec.layer_type, sensitivity
        );

        // Stop if overflow and not continuing
        if has_overflow && !config.continue_after_overflow {
            break;
        }

        current = output;
    }

    let final_width = current.max_width();

    Ok(SensitivityResult {
        layers,
        total_sensitivity,
        max_sensitivity,
        max_sensitivity_layer,
        input_epsilon: config.epsilon,
        final_width,
        overflow_at_layer,
    })
}

fn analyze_sensitivity_graph(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    config: &SensitivityConfig,
) -> Result<SensitivityResult, SensitivityError> {
    let exec_order = graph
        .topological_sort()
        .map_err(|e| SensitivityError::PropagationError(format!("{}", e)))?;

    if exec_order.is_empty() {
        return Err(SensitivityError::NoLayers);
    }

    let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
        std::collections::HashMap::with_capacity(exec_order.len());

    let mut layers = Vec::with_capacity(exec_order.len());
    let mut total_sensitivity: f32 = 1.0;
    let mut max_sensitivity: f32 = 0.0;
    let mut max_sensitivity_layer: Option<usize> = None;
    let mut overflow_at_layer: Option<usize> = None;

    let input_width0 = input.max_width();

    for (i, node_name) in exec_order.iter().enumerate() {
        let node = graph.get_node(node_name).ok_or_else(|| {
            SensitivityError::PropagationError(format!("Node not found: {}", node_name))
        })?;

        let input_width = if node.inputs.is_empty() {
            input_width0
        } else {
            node.inputs
                .iter()
                .map(|inp| {
                    if inp == "_input" {
                        input_width0
                    } else {
                        bounds_cache
                            .get(inp)
                            .map(|b| b.max_width())
                            .unwrap_or(input_width0)
                    }
                })
                .fold(0.0_f32, f32::max)
        };

        let mut propagation_failed = false;

        let output = if node.layer.is_binary() {
            if node.inputs.len() < 2 {
                return Err(SensitivityError::PropagationError(format!(
                    "Binary node {} requires 2 inputs, got {}",
                    node_name,
                    node.inputs.len()
                )));
            }

            let input_a = if node.inputs[0] == "_input" {
                input
            } else {
                bounds_cache.get(&node.inputs[0]).ok_or_else(|| {
                    SensitivityError::PropagationError(format!(
                        "Bounds for node {} not computed yet",
                        node.inputs[0]
                    ))
                })?
            };
            let input_b = if node.inputs[1] == "_input" {
                input
            } else {
                bounds_cache.get(&node.inputs[1]).ok_or_else(|| {
                    SensitivityError::PropagationError(format!(
                        "Bounds for node {} not computed yet",
                        node.inputs[1]
                    ))
                })?
            };

            match node.layer.propagate_ibp_binary(input_a, input_b) {
                Ok(out) => out,
                Err(e) => {
                    debug!("Node {} propagation failed: {}", node_name, e);
                    if !config.continue_after_overflow {
                        return Err(SensitivityError::PropagationError(format!(
                            "Node {} failed: {}",
                            node_name, e
                        )));
                    }
                    propagation_failed = true;
                    input_a.clone()
                }
            }
        } else {
            let node_input = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                input
            } else {
                bounds_cache.get(&node.inputs[0]).ok_or_else(|| {
                    SensitivityError::PropagationError(format!(
                        "Bounds for node {} not computed yet",
                        node.inputs[0]
                    ))
                })?
            };

            match node.layer.propagate_ibp(node_input) {
                Ok(out) => out,
                Err(e) => {
                    debug!("Node {} propagation failed: {}", node_name, e);
                    if !config.continue_after_overflow {
                        return Err(SensitivityError::PropagationError(format!(
                            "Node {} failed: {}",
                            node_name, e
                        )));
                    }
                    propagation_failed = true;
                    node_input.clone()
                }
            }
        };

        let output_width = output.max_width();
        let width_vec = output.width();
        let mean_width = width_vec.iter().sum::<f32>() / width_vec.len().max(1) as f32;
        let non_finite_count = width_vec.iter().filter(|w| !w.is_finite()).count();
        // Only check for numerical overflow if propagation succeeded
        let has_overflow =
            !propagation_failed && (!output_width.is_finite() || non_finite_count > 0);

        if non_finite_count > 0 && !propagation_failed {
            debug!(
                "Node {} has {}/{} non-finite width values",
                node_name,
                non_finite_count,
                width_vec.len()
            );
        }

        let sensitivity = if input_width > 0.0 && input_width.is_finite() {
            output_width / input_width
        } else if output_width == 0.0 {
            1.0
        } else {
            f32::INFINITY
        };

        // Track max sensitivity (only for successful propagations)
        if !propagation_failed && sensitivity > max_sensitivity && sensitivity.is_finite() {
            max_sensitivity = sensitivity;
            max_sensitivity_layer = Some(i);
        }

        // Accumulate total sensitivity (product) - skip failed propagations
        if !propagation_failed && sensitivity.is_finite() {
            total_sensitivity *= sensitivity;
        } else if has_overflow {
            // Only set to infinity for actual numerical overflow
            total_sensitivity = f32::INFINITY;
        }
        // Note: propagation failures are skipped (not counted in total)

        // Check for actual overflow (not propagation failure)
        if has_overflow && overflow_at_layer.is_none() {
            overflow_at_layer = Some(i);
        }

        layers.push(LayerSensitivity {
            name: node.name.clone(),
            layer_type: node.layer.layer_type().to_string(),
            input_width,
            output_width,
            sensitivity,
            mean_output_width: mean_width,
            output_shape: output.shape().to_vec(),
            has_overflow,
            propagation_failed,
        });

        bounds_cache.insert(node.name.clone(), output);

        // Stop early if overflow detected and not configured to continue.
        if has_overflow && !config.continue_after_overflow {
            break;
        }
    }

    let output_node = graph.get_output_name();
    let final_width = if !output_node.is_empty() {
        bounds_cache
            .get(output_node)
            .map(|b| b.max_width())
            .unwrap_or_else(|| {
                layers
                    .last()
                    .and_then(|l| bounds_cache.get(&l.name))
                    .map(|b| b.max_width())
                    .unwrap_or(input_width0)
            })
    } else {
        layers
            .last()
            .and_then(|l| bounds_cache.get(&l.name))
            .map(|b| b.max_width())
            .unwrap_or(input_width0)
    };

    Ok(SensitivityResult {
        layers,
        total_sensitivity,
        max_sensitivity,
        max_sensitivity_layer,
        input_epsilon: config.epsilon,
        final_width,
        overflow_at_layer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_name() {
        assert_eq!(truncate_name("short", 10), "short");
        assert_eq!(truncate_name("very_long_layer_name", 10), "...er_name");
    }

    const TEST_MODELS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/models");

    fn test_model_path(name: &str) -> String {
        format!("{}/{}", TEST_MODELS_DIR, name)
    }

    #[test]
    fn test_sensitivity_simple_mlp() {
        let model_path = test_model_path("simple_mlp.onnx");
        if !std::path::Path::new(&model_path).exists() {
            eprintln!("Test model not found: {}, skipping", model_path);
            return;
        }

        let config = SensitivityConfig {
            epsilon: 0.01,
            continue_after_overflow: true,
            input: None,
        };

        let result =
            analyze_sensitivity(&model_path, &config).expect("Failed to analyze sensitivity");

        // Should have multiple layers
        assert!(
            !result.layers.is_empty(),
            "Expected at least one layer in sensitivity analysis"
        );

        // All sensitivities should be positive
        for layer in &result.layers {
            assert!(
                layer.sensitivity >= 0.0,
                "Sensitivity should be non-negative for layer {}",
                layer.name
            );
        }

        // Print summary for debugging
        eprintln!("{}", result.summary());
    }

    #[test]
    fn test_sensitivity_transformer_block_dag() {
        // This model contains residual connections (binary Add), requiring DAG propagation.
        let model_path = test_model_path("transformer_block.onnx");
        if !std::path::Path::new(&model_path).exists() {
            eprintln!("Test model not found: {}, skipping", model_path);
            return;
        }

        let config = SensitivityConfig {
            epsilon: 0.01,
            continue_after_overflow: true,
            input: None,
        };

        let result =
            analyze_sensitivity(&model_path, &config).expect("Failed to analyze sensitivity");
        assert!(
            !result.layers.is_empty(),
            "Expected at least one node in DAG sensitivity analysis"
        );
    }

    #[test]
    fn test_sensitivity_config_default() {
        let config = SensitivityConfig::default();
        assert_eq!(config.epsilon, 0.01);
        assert!(!config.continue_after_overflow);
        assert!(config.input.is_none());
    }

    #[test]
    fn test_layer_sensitivity_is_high_sensitivity() {
        let layer = LayerSensitivity {
            name: "test_layer".to_string(),
            layer_type: "Linear".to_string(),
            input_width: 0.02,
            output_width: 0.2,
            sensitivity: 10.0,
            mean_output_width: 0.15,
            output_shape: vec![10],
            has_overflow: false,
            propagation_failed: false,
        };

        // sensitivity=10 is above threshold=5
        assert!(layer.is_high_sensitivity(5.0));
        // sensitivity=10 is not above threshold=15
        assert!(!layer.is_high_sensitivity(15.0));
        // exact threshold
        assert!(!layer.is_high_sensitivity(10.0));
    }

    #[test]
    fn test_layer_sensitivity_is_contractive() {
        let contractive = LayerSensitivity {
            name: "relu".to_string(),
            layer_type: "ReLU".to_string(),
            input_width: 1.0,
            output_width: 0.5,
            sensitivity: 0.5,
            mean_output_width: 0.3,
            output_shape: vec![10],
            has_overflow: false,
            propagation_failed: false,
        };
        assert!(contractive.is_contractive());

        let expanding = LayerSensitivity {
            name: "linear".to_string(),
            layer_type: "Linear".to_string(),
            input_width: 0.5,
            output_width: 1.0,
            sensitivity: 2.0,
            mean_output_width: 0.8,
            output_shape: vec![10],
            has_overflow: false,
            propagation_failed: false,
        };
        assert!(!expanding.is_contractive());

        let neutral = LayerSensitivity {
            name: "identity".to_string(),
            layer_type: "Identity".to_string(),
            input_width: 1.0,
            output_width: 1.0,
            sensitivity: 1.0,
            mean_output_width: 1.0,
            output_shape: vec![10],
            has_overflow: false,
            propagation_failed: false,
        };
        assert!(!neutral.is_contractive());
    }

    #[test]
    fn test_sensitivity_result_layers_by_sensitivity() {
        let result = SensitivityResult {
            layers: vec![
                LayerSensitivity {
                    name: "low".to_string(),
                    layer_type: "ReLU".to_string(),
                    input_width: 1.0,
                    output_width: 0.5,
                    sensitivity: 0.5,
                    mean_output_width: 0.3,
                    output_shape: vec![10],
                    has_overflow: false,
                    propagation_failed: false,
                },
                LayerSensitivity {
                    name: "high".to_string(),
                    layer_type: "Linear".to_string(),
                    input_width: 1.0,
                    output_width: 10.0,
                    sensitivity: 10.0,
                    mean_output_width: 8.0,
                    output_shape: vec![10],
                    has_overflow: false,
                    propagation_failed: false,
                },
                LayerSensitivity {
                    name: "medium".to_string(),
                    layer_type: "Softmax".to_string(),
                    input_width: 1.0,
                    output_width: 3.0,
                    sensitivity: 3.0,
                    mean_output_width: 2.0,
                    output_shape: vec![10],
                    has_overflow: false,
                    propagation_failed: false,
                },
            ],
            total_sensitivity: 15.0,
            max_sensitivity: 10.0,
            max_sensitivity_layer: Some(1),
            input_epsilon: 0.01,
            final_width: 5.0,
            overflow_at_layer: None,
        };

        let sorted = result.layers_by_sensitivity();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].name, "high"); // sensitivity=10
        assert_eq!(sorted[1].name, "medium"); // sensitivity=3
        assert_eq!(sorted[2].name, "low"); // sensitivity=0.5
    }

    #[test]
    fn test_sensitivity_result_hot_spots() {
        let result = SensitivityResult {
            layers: vec![
                LayerSensitivity {
                    name: "low".to_string(),
                    layer_type: "ReLU".to_string(),
                    input_width: 1.0,
                    output_width: 0.5,
                    sensitivity: 0.5,
                    mean_output_width: 0.3,
                    output_shape: vec![10],
                    has_overflow: false,
                    propagation_failed: false,
                },
                LayerSensitivity {
                    name: "high".to_string(),
                    layer_type: "Linear".to_string(),
                    input_width: 1.0,
                    output_width: 10.0,
                    sensitivity: 10.0,
                    mean_output_width: 8.0,
                    output_shape: vec![10],
                    has_overflow: false,
                    propagation_failed: false,
                },
                LayerSensitivity {
                    name: "very_high".to_string(),
                    layer_type: "Softmax".to_string(),
                    input_width: 1.0,
                    output_width: 100.0,
                    sensitivity: 100.0,
                    mean_output_width: 80.0,
                    output_shape: vec![10],
                    has_overflow: false,
                    propagation_failed: false,
                },
            ],
            total_sensitivity: 500.0,
            max_sensitivity: 100.0,
            max_sensitivity_layer: Some(2),
            input_epsilon: 0.01,
            final_width: 100.0,
            overflow_at_layer: None,
        };

        let hot_spots_5 = result.hot_spots(5.0);
        assert_eq!(hot_spots_5.len(), 2);
        assert!(hot_spots_5.iter().any(|l| l.name == "high"));
        assert!(hot_spots_5.iter().any(|l| l.name == "very_high"));

        let hot_spots_50 = result.hot_spots(50.0);
        assert_eq!(hot_spots_50.len(), 1);
        assert_eq!(hot_spots_50[0].name, "very_high");

        let hot_spots_1000 = result.hot_spots(1000.0);
        assert!(hot_spots_1000.is_empty());
    }

    #[test]
    fn test_sensitivity_result_summary_basic() {
        let result = SensitivityResult {
            layers: vec![LayerSensitivity {
                name: "linear_1".to_string(),
                layer_type: "Linear".to_string(),
                input_width: 0.02,
                output_width: 0.1,
                sensitivity: 5.0,
                mean_output_width: 0.08,
                output_shape: vec![10],
                has_overflow: false,
                propagation_failed: false,
            }],
            total_sensitivity: 5.0,
            max_sensitivity: 5.0,
            max_sensitivity_layer: Some(0),
            input_epsilon: 0.01,
            final_width: 0.1,
            overflow_at_layer: None,
        };

        let summary = result.summary();
        assert!(summary.contains("Sensitivity Analysis"));
        assert!(summary.contains("linear_1"));
        assert!(summary.contains("5.00"));
        assert!(summary.contains("MODERATE"));
    }

    #[test]
    fn test_sensitivity_result_summary_with_overflow() {
        let result = SensitivityResult {
            layers: vec![
                LayerSensitivity {
                    name: "layer_1".to_string(),
                    layer_type: "Linear".to_string(),
                    input_width: 0.02,
                    output_width: 0.1,
                    sensitivity: 5.0,
                    mean_output_width: 0.08,
                    output_shape: vec![10],
                    has_overflow: false,
                    propagation_failed: false,
                },
                LayerSensitivity {
                    name: "layer_2".to_string(),
                    layer_type: "Softmax".to_string(),
                    input_width: 0.1,
                    output_width: f32::INFINITY,
                    sensitivity: f32::INFINITY,
                    mean_output_width: f32::INFINITY,
                    output_shape: vec![10],
                    has_overflow: true,
                    propagation_failed: false,
                },
            ],
            total_sensitivity: f32::INFINITY,
            max_sensitivity: 5.0,
            max_sensitivity_layer: Some(0),
            input_epsilon: 0.01,
            final_width: f32::INFINITY,
            overflow_at_layer: Some(1),
        };

        let summary = result.summary();
        assert!(summary.contains("OVERFLOW"));
        assert!(summary.contains("WARNING:"));
        assert!(summary.contains("layer_2"));
    }

    #[test]
    fn test_sensitivity_result_summary_with_propagation_failure() {
        let result = SensitivityResult {
            layers: vec![LayerSensitivity {
                name: "broken_layer".to_string(),
                layer_type: "Unknown".to_string(),
                input_width: 0.02,
                output_width: 0.02,
                sensitivity: 1.0,
                mean_output_width: 0.02,
                output_shape: vec![10],
                has_overflow: false,
                propagation_failed: true,
            }],
            total_sensitivity: 1.0,
            max_sensitivity: 0.0,
            max_sensitivity_layer: None,
            input_epsilon: 0.01,
            final_width: 0.02,
            overflow_at_layer: None,
        };

        let summary = result.summary();
        assert!(summary.contains("SKIPPED"));
        assert!(summary.contains("propagation failure"));
    }

    #[test]
    fn test_sensitivity_result_summary_status_thresholds() {
        // Test that different sensitivity values get correct status
        let make_layer = |name: &str, sens: f32| LayerSensitivity {
            name: name.to_string(),
            layer_type: "Linear".to_string(),
            input_width: 1.0,
            output_width: sens,
            sensitivity: sens,
            mean_output_width: sens * 0.8,
            output_shape: vec![10],
            has_overflow: false,
            propagation_failed: false,
        };

        let result = SensitivityResult {
            layers: vec![
                make_layer("stable", 0.5),   // sensitivity < 1.0 -> STABLE
                make_layer("ok", 1.5),       // 1.0 <= sensitivity <= 2.0 -> OK
                make_layer("moderate", 5.0), // 2.0 < sensitivity <= 10.0 -> MODERATE
                make_layer("high", 15.0),    // sensitivity > 10.0 -> HIGH
            ],
            total_sensitivity: 56.25,
            max_sensitivity: 15.0,
            max_sensitivity_layer: Some(3),
            input_epsilon: 0.01,
            final_width: 15.0,
            overflow_at_layer: None,
        };

        let summary = result.summary();
        assert!(summary.contains("STABLE"));
        assert!(summary.contains("OK"));
        assert!(summary.contains("MODERATE"));
        assert!(summary.contains("HIGH"));
    }

    #[test]
    fn test_truncate_name_various_lengths() {
        // Exact fit
        assert_eq!(truncate_name("exactly_10", 10), "exactly_10");
        // Under limit
        assert_eq!(truncate_name("short", 10), "short");
        // Over limit: "this_is_way_too_long_name" (25 chars) -> keep last 7 chars = "ng_name"
        assert_eq!(truncate_name("this_is_way_too_long_name", 10), "...ng_name");
        // Exactly one over: "abcdefghijk" (11 chars) -> keep last 7 = "efghijk"
        assert_eq!(truncate_name("abcdefghijk", 10), "...efghijk");
        // Very short width: "longname" (8 chars), width 5 -> keep last 2 = "me"
        assert_eq!(truncate_name("longname", 5), "...me");
    }

    #[test]
    fn test_sensitivity_error_display() {
        let load_err = SensitivityError::LoadError("file not found".to_string());
        assert!(load_err.to_string().contains("file not found"));

        let prop_err = SensitivityError::PropagationError("shape mismatch".to_string());
        assert!(prop_err.to_string().contains("shape mismatch"));

        let no_layers = SensitivityError::NoLayers;
        assert!(no_layers.to_string().contains("No layers"));

        let invalid_shape = SensitivityError::InvalidInputShape("bad shape".to_string());
        assert!(invalid_shape.to_string().contains("bad shape"));
    }
}
