//! Bound width profiling for neural networks.
//!
//! This module provides detailed analysis of how bound widths propagate
//! through a neural network, helping identify where verification becomes
//! difficult.
//!
//! ## Key Metrics
//!
//! For each layer, we track:
//! - **Input/Output width**: Max bound width at layer boundaries
//! - **Width growth**: Ratio of output to input width (expansion factor)
//! - **Cumulative width**: Total bound expansion from input to this layer
//!
//! ## Usage
//!
//! ```ignore
//! use gamma_onnx::profile::{profile_bounds, ProfileConfig};
//!
//! let config = ProfileConfig::default();
//! let result = profile_bounds("model.onnx", &config)?;
//! println!("{}", result.summary());
//! ```

use crate::{load_onnx, OnnxModel};
use gamma_propagate::BoundPropagation;
use gamma_tensor::BoundedTensor;
use ndarray::{ArrayD, IxDyn};
use std::path::Path;
use thiserror::Error;
use tracing::{debug, info};

/// Errors that can occur during bound profiling.
#[derive(Error, Debug)]
pub enum ProfileError {
    #[error("Failed to load model: {0}")]
    LoadError(String),

    #[error("Propagation error: {0}")]
    PropagationError(String),

    #[error("No layers in network")]
    NoLayers,

    #[error("Invalid input shape: {0}")]
    InvalidInputShape(String),
}

/// Bound width status indicators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundStatus {
    /// Bounds are tight and stable
    Tight,
    /// Bounds are moderate
    Moderate,
    /// Bounds are wide - verification getting harder
    Wide,
    /// Bounds are very wide - verification difficult
    VeryWide,
    /// Bounds have overflowed (infinity/NaN)
    Overflow,
}

impl std::fmt::Display for BoundStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BoundStatus::Tight => write!(f, "TIGHT"),
            BoundStatus::Moderate => write!(f, "MODERATE"),
            BoundStatus::Wide => write!(f, "WIDE"),
            BoundStatus::VeryWide => write!(f, "VERY WIDE"),
            BoundStatus::Overflow => write!(f, "OVERFLOW"),
        }
    }
}

impl BoundStatus {
    /// Determine status from bound width relative to input epsilon.
    fn from_width(width: f32, input_epsilon: f32) -> Self {
        if !width.is_finite() {
            BoundStatus::Overflow
        } else {
            let ratio = width / (2.0 * input_epsilon);
            if ratio < 10.0 {
                BoundStatus::Tight
            } else if ratio < 100.0 {
                BoundStatus::Moderate
            } else if ratio < 10000.0 {
                BoundStatus::Wide
            } else {
                BoundStatus::VeryWide
            }
        }
    }
}

/// Result of profiling a single layer's bounds.
#[derive(Debug, Clone)]
pub struct LayerProfile {
    /// Layer name from the model.
    pub name: String,
    /// Layer type (e.g., "Linear", "ReLU", "Softmax").
    pub layer_type: String,
    /// Input bound width (max across all elements).
    pub input_width: f32,
    /// Output bound width (max across all elements).
    pub output_width: f32,
    /// Mean output bound width.
    pub mean_output_width: f32,
    /// Median output bound width.
    pub median_output_width: f32,
    /// Width growth ratio (output/input).
    pub growth_ratio: f32,
    /// Cumulative width from input (output_width / initial_epsilon * 2).
    pub cumulative_expansion: f32,
    /// Output shape.
    pub output_shape: Vec<usize>,
    /// Number of elements in output.
    pub num_elements: usize,
    /// Bound status indicator.
    pub status: BoundStatus,
}

impl LayerProfile {
    /// Check if this layer is a "choke point" where bounds explode.
    pub fn is_choke_point(&self, growth_threshold: f32) -> bool {
        self.growth_ratio > growth_threshold
    }
}

/// Result of a full bound profiling analysis.
#[derive(Debug, Clone)]
pub struct ProfileResult {
    /// Per-layer bound profiles.
    pub layers: Vec<LayerProfile>,
    /// Input perturbation epsilon.
    pub input_epsilon: f32,
    /// Initial input bound width (2 * epsilon).
    pub initial_width: f32,
    /// Final output bound width.
    pub final_width: f32,
    /// Total expansion (final_width / initial_width).
    pub total_expansion: f32,
    /// Layer with highest growth ratio.
    pub max_growth_layer: Option<usize>,
    /// Maximum single-layer growth ratio.
    pub max_growth_ratio: f32,
    /// Index of first layer with overflow.
    pub overflow_at_layer: Option<usize>,
    /// Verification difficulty score (0-100).
    pub difficulty_score: f32,
}

impl ProfileResult {
    /// Get a summary of the profiling.
    pub fn summary(&self) -> String {
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
                layer.status,
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
            "Max growth layer: {} ({:.2}x)",
            self.max_growth_layer
                .and_then(|i| self.layers.get(i))
                .map(|l| l.name.as_str())
                .unwrap_or("N/A"),
            self.max_growth_ratio
        ));
        lines.push(format!(
            "Verification difficulty: {:.0}/100",
            self.difficulty_score
        ));

        if let Some(idx) = self.overflow_at_layer {
            lines.push(format!(
                "WARNING: Overflow at layer {} ({})",
                idx,
                self.layers
                    .get(idx)
                    .map(|l| l.name.as_str())
                    .unwrap_or("unknown")
            ));
        }

        lines.join("\n")
    }

    /// Get layers sorted by growth ratio (highest first).
    pub fn layers_by_growth(&self) -> Vec<&LayerProfile> {
        let mut sorted: Vec<_> = self.layers.iter().collect();
        sorted.sort_by(|a, b| {
            b.growth_ratio
                .partial_cmp(&a.growth_ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Get "choke points" - layers with high growth ratio.
    pub fn choke_points(&self, threshold: f32) -> Vec<&LayerProfile> {
        self.layers
            .iter()
            .filter(|l| l.growth_ratio > threshold)
            .collect()
    }

    /// Get layers with wide or very wide bounds.
    pub fn problematic_layers(&self) -> Vec<&LayerProfile> {
        self.layers
            .iter()
            .filter(|l| {
                matches!(
                    l.status,
                    BoundStatus::Wide | BoundStatus::VeryWide | BoundStatus::Overflow
                )
            })
            .collect()
    }
}

/// Configuration for bound profiling.
#[derive(Debug, Clone)]
pub struct ProfileConfig {
    /// Input perturbation epsilon.
    pub epsilon: f32,
    /// Whether to continue after overflow.
    pub continue_after_overflow: bool,
    /// Custom input tensor (None = zeros with epsilon bounds).
    pub input: Option<BoundedTensor>,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.01,
            continue_after_overflow: true,
            input: None,
        }
    }
}

/// Create deterministic input with unit variance for realistic LayerNorm/RMSNorm bounds.
///
/// Uses alternating ±1 pattern to ensure non-zero variance, avoiding artificial
/// amplification when the center point has variance near zero.
///
/// For RMSNorm/LayerNorm with zero-valued inputs:
/// - var(zeros) = 0, std = sqrt(eps) ≈ 0.003 → 300x amplification
///
/// For alternating ±1 inputs:
/// - mean ≈ 0, var = 1, std ≈ 1 → 1x amplification (realistic)
fn make_unit_variance_input(shape: &[usize], epsilon: f32) -> BoundedTensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let array =
        ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape and data size should match");
    BoundedTensor::from_epsilon(array, epsilon)
}

/// Truncate a name to fit in a given width.
fn truncate_name(name: &str, width: usize) -> String {
    if name.len() <= width {
        name.to_string()
    } else {
        format!("...{}", &name[name.len() - width + 3..])
    }
}

/// Calculate median of a slice.
fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.iter().filter(|v| v.is_finite()).cloned().collect();
    if sorted.is_empty() {
        return f32::INFINITY;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Calculate verification difficulty score (0-100).
fn difficulty_score(total_expansion: f32, max_growth: f32, overflow: bool) -> f32 {
    if overflow {
        return 100.0;
    }

    // Log-scale scoring
    let expansion_score = if total_expansion <= 1.0 {
        0.0
    } else {
        (total_expansion.log10() * 10.0).min(50.0)
    };

    let growth_score = if max_growth <= 1.0 {
        0.0
    } else {
        (max_growth.log10() * 20.0).min(50.0)
    };

    (expansion_score + growth_score).min(100.0)
}

/// Profile bounds of a model loaded from ONNX file.
pub fn profile_bounds(
    path: impl AsRef<Path>,
    config: &ProfileConfig,
) -> Result<ProfileResult, ProfileError> {
    info!("Loading model: {}", path.as_ref().display());
    let onnx_model =
        load_onnx(path.as_ref()).map_err(|e| ProfileError::LoadError(format!("{}", e)))?;

    profile_bounds_model(&onnx_model, config)
}

/// Profile bounds of an already-loaded ONNX model.
pub fn profile_bounds_model(
    model: &OnnxModel,
    config: &ProfileConfig,
) -> Result<ProfileResult, ProfileError> {
    // Convert to propagate network
    let network = model
        .to_propagate_network()
        .map_err(|e| ProfileError::PropagationError(format!("{}", e)))?;

    if network.layers.is_empty() {
        return Err(ProfileError::NoLayers);
    }

    // Create input tensor
    let input = if let Some(ref inp) = config.input {
        inp.clone()
    } else {
        let input_spec =
            model.network.inputs.first().ok_or_else(|| {
                ProfileError::InvalidInputShape("No input specification".to_string())
            })?;

        let shape: Vec<usize> = input_spec
            .shape
            .iter()
            .map(|&d| if d > 0 { d as usize } else { 1 })
            .collect();

        // Use unit-variance input to avoid artificial amplification in LayerNorm/RMSNorm
        make_unit_variance_input(&shape, config.epsilon)
    };

    let initial_width = input.max_width();

    info!(
        "Starting bound profile with input shape {:?}, epsilon {}, initial width {}",
        input.shape(),
        config.epsilon,
        initial_width
    );

    // Track layer-by-layer bounds
    let mut layers = Vec::new();
    let mut current = input.clone();
    let mut max_growth_ratio: f32 = 1.0;
    let mut max_growth_layer: Option<usize> = None;
    let mut overflow_at_layer: Option<usize> = None;

    for (i, (layer, spec)) in network
        .layers
        .iter()
        .zip(model.network.layers.iter())
        .enumerate()
    {
        let input_width = current.max_width();

        // Propagate through this layer
        let output = match layer.propagate_ibp(&current) {
            Ok(out) => out,
            Err(e) => {
                debug!("Layer {} propagation failed: {}", spec.name, e);
                if !config.continue_after_overflow {
                    return Err(ProfileError::PropagationError(format!(
                        "Layer {} failed: {}",
                        spec.name, e
                    )));
                }
                if overflow_at_layer.is_none() {
                    overflow_at_layer = Some(i);
                }
                current.clone()
            }
        };

        let output_width = output.max_width();
        let widths: Vec<f32> = output.width().iter().cloned().collect();
        let mean_width = widths.iter().sum::<f32>() / widths.len().max(1) as f32;
        let median_width = median(&widths);

        // Calculate growth ratio
        let growth_ratio = if input_width > 0.0 && input_width.is_finite() {
            output_width / input_width
        } else {
            1.0
        };

        // Track max growth
        if growth_ratio > max_growth_ratio && growth_ratio.is_finite() {
            max_growth_ratio = growth_ratio;
            max_growth_layer = Some(i);
        }

        // Calculate cumulative expansion from input
        let cumulative_expansion = if initial_width > 0.0 && initial_width.is_finite() {
            output_width / initial_width
        } else {
            1.0
        };

        // Determine status
        let has_overflow = !output_width.is_finite();
        let status = if has_overflow {
            if overflow_at_layer.is_none() {
                overflow_at_layer = Some(i);
            }
            BoundStatus::Overflow
        } else {
            BoundStatus::from_width(output_width, config.epsilon)
        };

        layers.push(LayerProfile {
            name: spec.name.clone(),
            layer_type: format!("{:?}", spec.layer_type),
            input_width,
            output_width,
            mean_output_width: mean_width,
            median_output_width: median_width,
            growth_ratio,
            cumulative_expansion,
            output_shape: output.shape().to_vec(),
            num_elements: output.lower.len(),
            status,
        });

        debug!(
            "Layer {}: width {} -> {}, growth {:.2}x",
            spec.name, input_width, output_width, growth_ratio
        );

        // Stop if overflow and not continuing
        if has_overflow && !config.continue_after_overflow {
            break;
        }

        current = output;
    }

    let final_width = current.max_width();
    let total_expansion = if initial_width > 0.0 && initial_width.is_finite() {
        final_width / initial_width
    } else {
        1.0
    };

    let difficulty = difficulty_score(
        total_expansion,
        max_growth_ratio,
        overflow_at_layer.is_some(),
    );

    Ok(ProfileResult {
        layers,
        input_epsilon: config.epsilon,
        initial_width,
        final_width,
        total_expansion,
        max_growth_layer,
        max_growth_ratio,
        overflow_at_layer,
        difficulty_score: difficulty,
    })
}

/// Profile bounds of a GraphNetwork (for native format models).
///
/// This function profiles IBP bounds through a GraphNetwork, tracking bound width
/// at each node. Useful for diagnosing bound explosion in GGUF/SafeTensors models.
pub fn profile_bounds_graph(
    graph: &gamma_propagate::GraphNetwork,
    config: &ProfileConfig,
    input_shape: &[usize],
) -> Result<ProfileResult, ProfileError> {
    // Create input tensor
    // Use unit-variance input to avoid artificial amplification in LayerNorm/RMSNorm
    let input = if let Some(ref inp) = config.input {
        inp.clone()
    } else {
        make_unit_variance_input(input_shape, config.epsilon)
    };

    let initial_width = input.max_width();

    info!(
        "Starting graph bound profile with input shape {:?}, epsilon {}, initial width {}",
        input.shape(),
        config.epsilon,
        initial_width
    );

    // Get topological order for processing
    let exec_order = graph
        .topological_sort()
        .map_err(|e| ProfileError::PropagationError(format!("Topological sort failed: {}", e)))?;

    if exec_order.is_empty() {
        return Err(ProfileError::NoLayers);
    }

    // Track layer-by-layer bounds
    let mut layers = Vec::new();
    let mut max_growth_ratio: f32 = 1.0;
    let mut max_growth_layer: Option<usize> = None;
    let mut overflow_at_layer: Option<usize> = None;

    // Cache bounds for each node
    let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
        std::collections::HashMap::new();

    // Helper to get bounds for an input (either from cache or network input)
    fn get_bounds<'a>(
        input_name: &str,
        network_input: &BoundedTensor,
        cache: &'a std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<std::borrow::Cow<'a, BoundedTensor>, ProfileError> {
        if input_name == "_input" {
            Ok(std::borrow::Cow::Owned(network_input.clone()))
        } else {
            cache
                .get(input_name)
                .map(std::borrow::Cow::Borrowed)
                .ok_or_else(|| {
                    ProfileError::PropagationError(format!(
                        "Input {} not found in cache",
                        input_name
                    ))
                })
        }
    }

    // Process nodes in topological order
    for (i, node_name) in exec_order.iter().enumerate() {
        let node = graph.get_node(node_name).ok_or_else(|| {
            ProfileError::PropagationError(format!("Node not found: {}", node_name))
        })?;

        let layer_type = format!("{:?}", node.layer.layer_type());

        // Get input width (from first input)
        let input_width = if node.inputs.is_empty() {
            initial_width
        } else {
            get_bounds(&node.inputs[0], &input, &bounds_cache)
                .map(|b| b.max_width())
                .unwrap_or(initial_width)
        };

        // Propagate bounds through this node
        let output = if node.layer.is_binary() {
            if node.inputs.len() < 2 {
                return Err(ProfileError::PropagationError(format!(
                    "Binary node {} requires 2 inputs",
                    node_name
                )));
            }
            let input_a = get_bounds(&node.inputs[0], &input, &bounds_cache)?;
            let input_b = get_bounds(&node.inputs[1], &input, &bounds_cache)?;
            match node.layer.propagate_ibp_binary(&input_a, &input_b) {
                Ok(out) => out,
                Err(e) => {
                    debug!("Node {} propagation failed: {}", node_name, e);
                    if !config.continue_after_overflow {
                        return Err(ProfileError::PropagationError(format!(
                            "Node {} failed: {}",
                            node_name, e
                        )));
                    }
                    if overflow_at_layer.is_none() {
                        overflow_at_layer = Some(i);
                    }
                    input_a.into_owned()
                }
            }
        } else {
            if node.inputs.is_empty() {
                return Err(ProfileError::PropagationError(format!(
                    "Node {} has no inputs",
                    node_name
                )));
            }
            let node_input = get_bounds(&node.inputs[0], &input, &bounds_cache)?;
            match node.layer.propagate_ibp(&node_input) {
                Ok(out) => out,
                Err(e) => {
                    debug!("Node {} propagation failed: {}", node_name, e);
                    if !config.continue_after_overflow {
                        return Err(ProfileError::PropagationError(format!(
                            "Node {} failed: {}",
                            node_name, e
                        )));
                    }
                    if overflow_at_layer.is_none() {
                        overflow_at_layer = Some(i);
                    }
                    node_input.into_owned()
                }
            }
        };

        let output_width = output.max_width();
        let widths: Vec<f32> = output.width().iter().cloned().collect();
        let mean_width = widths.iter().sum::<f32>() / widths.len().max(1) as f32;
        let median_width = median(&widths);

        // Calculate growth ratio
        let growth_ratio = if input_width > 0.0 && input_width.is_finite() {
            output_width / input_width
        } else {
            1.0
        };

        // Track max growth
        if growth_ratio > max_growth_ratio && growth_ratio.is_finite() {
            max_growth_ratio = growth_ratio;
            max_growth_layer = Some(i);
        }

        // Calculate cumulative expansion from input
        let cumulative_expansion = if initial_width > 0.0 && initial_width.is_finite() {
            output_width / initial_width
        } else {
            1.0
        };

        // Determine status
        let has_overflow = !output_width.is_finite();
        let status = if has_overflow {
            if overflow_at_layer.is_none() {
                overflow_at_layer = Some(i);
            }
            BoundStatus::Overflow
        } else {
            BoundStatus::from_width(output_width, config.epsilon)
        };

        layers.push(LayerProfile {
            name: node_name.clone(),
            layer_type,
            input_width,
            output_width,
            mean_output_width: mean_width,
            median_output_width: median_width,
            growth_ratio,
            cumulative_expansion,
            output_shape: output.shape().to_vec(),
            num_elements: output.lower.len(),
            status,
        });

        debug!(
            "Node {}: width {} -> {}, growth {:.2}x",
            node_name, input_width, output_width, growth_ratio
        );

        // Stop if overflow and not continuing
        if has_overflow && !config.continue_after_overflow {
            break;
        }

        bounds_cache.insert(node_name.clone(), output);
    }

    let final_width = layers
        .last()
        .map(|l| l.output_width)
        .unwrap_or(initial_width);
    let total_expansion = if initial_width > 0.0 && initial_width.is_finite() {
        final_width / initial_width
    } else {
        1.0
    };

    let difficulty = difficulty_score(
        total_expansion,
        max_growth_ratio,
        overflow_at_layer.is_some(),
    );

    Ok(ProfileResult {
        layers,
        input_epsilon: config.epsilon,
        initial_width,
        final_width,
        total_expansion,
        max_growth_layer,
        max_growth_ratio,
        overflow_at_layer,
        difficulty_score: difficulty,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bound_status() {
        let epsilon = 0.01;
        // Initial width is 2 * epsilon = 0.02

        // Tight: width < 10 * initial = 0.2
        assert!(matches!(
            BoundStatus::from_width(0.1, epsilon),
            BoundStatus::Tight
        ));

        // Moderate: width < 100 * initial = 2.0
        assert!(matches!(
            BoundStatus::from_width(1.0, epsilon),
            BoundStatus::Moderate
        ));

        // Wide: width < 10000 * initial = 200
        assert!(matches!(
            BoundStatus::from_width(50.0, epsilon),
            BoundStatus::Wide
        ));

        // Very wide
        assert!(matches!(
            BoundStatus::from_width(1000.0, epsilon),
            BoundStatus::VeryWide
        ));

        // Overflow
        assert!(matches!(
            BoundStatus::from_width(f32::INFINITY, epsilon),
            BoundStatus::Overflow
        ));
    }

    #[test]
    fn test_median() {
        assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median(&[5.0]), 5.0);
        assert_eq!(median(&[]), 0.0);
    }

    #[test]
    fn test_difficulty_score() {
        // No expansion
        assert_eq!(difficulty_score(1.0, 1.0, false), 0.0);

        // Overflow = max difficulty
        assert_eq!(difficulty_score(1.0, 1.0, true), 100.0);

        // Some expansion
        let score = difficulty_score(100.0, 10.0, false);
        assert!(score > 0.0 && score < 100.0);
    }

    #[test]
    fn test_truncate_name() {
        assert_eq!(truncate_name("short", 10), "short");
        assert_eq!(truncate_name("very_long_layer_name", 10), "...er_name");
    }

    #[test]
    fn test_profile_config_default() {
        let config = ProfileConfig::default();
        assert_eq!(config.epsilon, 0.01);
        assert!(config.continue_after_overflow);
        assert!(config.input.is_none());
    }

    #[test]
    fn test_bound_status_display() {
        assert_eq!(format!("{}", BoundStatus::Tight), "TIGHT");
        assert_eq!(format!("{}", BoundStatus::Moderate), "MODERATE");
        assert_eq!(format!("{}", BoundStatus::Wide), "WIDE");
        assert_eq!(format!("{}", BoundStatus::VeryWide), "VERY WIDE");
        assert_eq!(format!("{}", BoundStatus::Overflow), "OVERFLOW");
    }

    #[test]
    fn test_bound_status_from_width_boundary_values() {
        let epsilon = 0.01;
        // Initial width is 2 * epsilon = 0.02

        // Tight/Moderate boundary: ratio = 10 -> width = 10 * 0.02 = 0.2
        assert!(matches!(
            BoundStatus::from_width(0.19, epsilon),
            BoundStatus::Tight
        ));
        assert!(matches!(
            BoundStatus::from_width(0.21, epsilon),
            BoundStatus::Moderate
        ));

        // Moderate/Wide boundary: ratio = 100 -> width = 100 * 0.02 = 2.0
        assert!(matches!(
            BoundStatus::from_width(1.99, epsilon),
            BoundStatus::Moderate
        ));
        assert!(matches!(
            BoundStatus::from_width(2.01, epsilon),
            BoundStatus::Wide
        ));

        // Wide/VeryWide boundary: ratio = 10000 -> width = 10000 * 0.02 = 200
        assert!(matches!(
            BoundStatus::from_width(199.0, epsilon),
            BoundStatus::Wide
        ));
        assert!(matches!(
            BoundStatus::from_width(201.0, epsilon),
            BoundStatus::VeryWide
        ));

        // NaN should also be overflow
        assert!(matches!(
            BoundStatus::from_width(f32::NAN, epsilon),
            BoundStatus::Overflow
        ));

        // Negative infinity
        assert!(matches!(
            BoundStatus::from_width(f32::NEG_INFINITY, epsilon),
            BoundStatus::Overflow
        ));
    }

    #[test]
    fn test_layer_profile_is_choke_point() {
        let layer = LayerProfile {
            name: "test".to_string(),
            layer_type: "Linear".to_string(),
            input_width: 0.1,
            output_width: 1.0,
            mean_output_width: 0.8,
            median_output_width: 0.7,
            growth_ratio: 10.0,
            cumulative_expansion: 50.0,
            output_shape: vec![10],
            num_elements: 10,
            status: BoundStatus::Moderate,
        };

        assert!(layer.is_choke_point(5.0));
        assert!(!layer.is_choke_point(15.0));
        assert!(!layer.is_choke_point(10.0)); // exact threshold
    }

    #[test]
    fn test_profile_result_layers_by_growth() {
        let result = ProfileResult {
            layers: vec![
                LayerProfile {
                    name: "low".to_string(),
                    layer_type: "ReLU".to_string(),
                    input_width: 1.0,
                    output_width: 0.5,
                    mean_output_width: 0.4,
                    median_output_width: 0.4,
                    growth_ratio: 0.5,
                    cumulative_expansion: 0.5,
                    output_shape: vec![10],
                    num_elements: 10,
                    status: BoundStatus::Tight,
                },
                LayerProfile {
                    name: "high".to_string(),
                    layer_type: "Linear".to_string(),
                    input_width: 0.5,
                    output_width: 5.0,
                    mean_output_width: 4.0,
                    median_output_width: 4.0,
                    growth_ratio: 10.0,
                    cumulative_expansion: 5.0,
                    output_shape: vec![10],
                    num_elements: 10,
                    status: BoundStatus::Moderate,
                },
                LayerProfile {
                    name: "medium".to_string(),
                    layer_type: "Softmax".to_string(),
                    input_width: 5.0,
                    output_width: 15.0,
                    mean_output_width: 12.0,
                    median_output_width: 12.0,
                    growth_ratio: 3.0,
                    cumulative_expansion: 15.0,
                    output_shape: vec![10],
                    num_elements: 10,
                    status: BoundStatus::Wide,
                },
            ],
            input_epsilon: 0.01,
            initial_width: 1.0,
            final_width: 15.0,
            total_expansion: 15.0,
            max_growth_layer: Some(1),
            max_growth_ratio: 10.0,
            overflow_at_layer: None,
            difficulty_score: 30.0,
        };

        let sorted = result.layers_by_growth();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].name, "high"); // growth=10
        assert_eq!(sorted[1].name, "medium"); // growth=3
        assert_eq!(sorted[2].name, "low"); // growth=0.5
    }

    #[test]
    fn test_profile_result_choke_points() {
        let result = ProfileResult {
            layers: vec![
                LayerProfile {
                    name: "low".to_string(),
                    layer_type: "ReLU".to_string(),
                    input_width: 1.0,
                    output_width: 0.5,
                    mean_output_width: 0.4,
                    median_output_width: 0.4,
                    growth_ratio: 0.5,
                    cumulative_expansion: 0.5,
                    output_shape: vec![10],
                    num_elements: 10,
                    status: BoundStatus::Tight,
                },
                LayerProfile {
                    name: "high".to_string(),
                    layer_type: "Linear".to_string(),
                    input_width: 0.5,
                    output_width: 50.0,
                    mean_output_width: 40.0,
                    median_output_width: 40.0,
                    growth_ratio: 100.0,
                    cumulative_expansion: 50.0,
                    output_shape: vec![10],
                    num_elements: 10,
                    status: BoundStatus::Wide,
                },
            ],
            input_epsilon: 0.01,
            initial_width: 1.0,
            final_width: 50.0,
            total_expansion: 50.0,
            max_growth_layer: Some(1),
            max_growth_ratio: 100.0,
            overflow_at_layer: None,
            difficulty_score: 50.0,
        };

        let chokes = result.choke_points(10.0);
        assert_eq!(chokes.len(), 1);
        assert_eq!(chokes[0].name, "high");

        let no_chokes = result.choke_points(1000.0);
        assert!(no_chokes.is_empty());
    }

    #[test]
    fn test_profile_result_problematic_layers() {
        let result = ProfileResult {
            layers: vec![
                LayerProfile {
                    name: "tight".to_string(),
                    layer_type: "ReLU".to_string(),
                    input_width: 0.01,
                    output_width: 0.01,
                    mean_output_width: 0.01,
                    median_output_width: 0.01,
                    growth_ratio: 1.0,
                    cumulative_expansion: 1.0,
                    output_shape: vec![10],
                    num_elements: 10,
                    status: BoundStatus::Tight,
                },
                LayerProfile {
                    name: "moderate".to_string(),
                    layer_type: "Linear".to_string(),
                    input_width: 0.01,
                    output_width: 0.5,
                    mean_output_width: 0.4,
                    median_output_width: 0.4,
                    growth_ratio: 50.0,
                    cumulative_expansion: 50.0,
                    output_shape: vec![10],
                    num_elements: 10,
                    status: BoundStatus::Moderate,
                },
                LayerProfile {
                    name: "wide".to_string(),
                    layer_type: "Softmax".to_string(),
                    input_width: 0.5,
                    output_width: 100.0,
                    mean_output_width: 80.0,
                    median_output_width: 80.0,
                    growth_ratio: 200.0,
                    cumulative_expansion: 10000.0,
                    output_shape: vec![10],
                    num_elements: 10,
                    status: BoundStatus::Wide,
                },
                LayerProfile {
                    name: "overflow".to_string(),
                    layer_type: "Exp".to_string(),
                    input_width: 100.0,
                    output_width: f32::INFINITY,
                    mean_output_width: f32::INFINITY,
                    median_output_width: f32::INFINITY,
                    growth_ratio: f32::INFINITY,
                    cumulative_expansion: f32::INFINITY,
                    output_shape: vec![10],
                    num_elements: 10,
                    status: BoundStatus::Overflow,
                },
            ],
            input_epsilon: 0.01,
            initial_width: 0.02,
            final_width: f32::INFINITY,
            total_expansion: f32::INFINITY,
            max_growth_layer: Some(2),
            max_growth_ratio: 200.0,
            overflow_at_layer: Some(3),
            difficulty_score: 100.0,
        };

        let problems = result.problematic_layers();
        assert_eq!(problems.len(), 2); // Wide and Overflow
        assert!(problems.iter().any(|l| l.name == "wide"));
        assert!(problems.iter().any(|l| l.name == "overflow"));
    }

    #[test]
    fn test_profile_result_summary_basic() {
        let result = ProfileResult {
            layers: vec![LayerProfile {
                name: "linear_1".to_string(),
                layer_type: "Linear".to_string(),
                input_width: 0.02,
                output_width: 0.1,
                mean_output_width: 0.08,
                median_output_width: 0.07,
                growth_ratio: 5.0,
                cumulative_expansion: 5.0,
                output_shape: vec![10],
                num_elements: 10,
                status: BoundStatus::Tight,
            }],
            input_epsilon: 0.01,
            initial_width: 0.02,
            final_width: 0.1,
            total_expansion: 5.0,
            max_growth_layer: Some(0),
            max_growth_ratio: 5.0,
            overflow_at_layer: None,
            difficulty_score: 17.0,
        };

        let summary = result.summary();
        assert!(summary.contains("Bound Width Profile"));
        assert!(summary.contains("linear_1"));
        assert!(summary.contains("5.00x"));
        assert!(summary.contains("TIGHT"));
        assert!(summary.contains("Verification difficulty"));
    }

    #[test]
    fn test_profile_result_summary_with_overflow() {
        let result = ProfileResult {
            layers: vec![LayerProfile {
                name: "exploding_layer".to_string(),
                layer_type: "Exp".to_string(),
                input_width: 100.0,
                output_width: f32::INFINITY,
                mean_output_width: f32::INFINITY,
                median_output_width: f32::INFINITY,
                growth_ratio: f32::INFINITY,
                cumulative_expansion: f32::INFINITY,
                output_shape: vec![10],
                num_elements: 10,
                status: BoundStatus::Overflow,
            }],
            input_epsilon: 0.01,
            initial_width: 0.02,
            final_width: f32::INFINITY,
            total_expansion: f32::INFINITY,
            max_growth_layer: Some(0),
            max_growth_ratio: f32::INFINITY,
            overflow_at_layer: Some(0),
            difficulty_score: 100.0,
        };

        let summary = result.summary();
        assert!(summary.contains("WARNING"));
        assert!(summary.contains("Overflow"));
        assert!(summary.contains("exploding_layer"));
    }

    #[test]
    fn test_median_edge_cases() {
        // Empty input
        assert_eq!(median(&[]), 0.0);

        // Single element
        assert_eq!(median(&[5.0]), 5.0);

        // Two elements (even)
        assert_eq!(median(&[1.0, 3.0]), 2.0);

        // Odd number
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);

        // Even number (already sorted)
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 3.5);

        // Unsorted input
        assert_eq!(median(&[5.0, 1.0, 3.0]), 3.0);

        // With infinite values (should filter them)
        assert_eq!(median(&[1.0, f32::INFINITY, 3.0, 2.0]), 2.0);

        // All infinite
        assert_eq!(median(&[f32::INFINITY, f32::NEG_INFINITY]), f32::INFINITY);

        // With NaN (should filter)
        assert_eq!(median(&[1.0, f32::NAN, 3.0, 2.0]), 2.0);
    }

    #[test]
    fn test_difficulty_score_edge_cases() {
        // Both under 1.0
        assert_eq!(difficulty_score(0.5, 0.5, false), 0.0);

        // Expansion exactly 1.0
        assert_eq!(difficulty_score(1.0, 1.0, false), 0.0);

        // High expansion
        let high_expansion = difficulty_score(1e6, 1.0, false);
        assert!(high_expansion > 40.0);
        assert!(high_expansion <= 50.0);

        // High growth
        let high_growth = difficulty_score(1.0, 1e6, false);
        assert!(high_growth > 40.0);
        assert!(high_growth <= 50.0);

        // Both high
        let both_high = difficulty_score(1e6, 1e6, false);
        assert_eq!(both_high, 100.0);

        // Overflow always 100
        assert_eq!(difficulty_score(1.0, 1.0, true), 100.0);
        assert_eq!(difficulty_score(0.1, 0.1, true), 100.0);
    }

    #[test]
    fn test_make_unit_variance_input() {
        let shape = &[2, 4];
        let epsilon = 0.01;
        let input = make_unit_variance_input(shape, epsilon);

        assert_eq!(input.shape(), shape);

        // Check alternating pattern
        let data: Vec<f32> = input.lower.iter().cloned().collect();
        assert_eq!(data[0], 1.0 - epsilon);
        assert_eq!(data[1], -1.0 - epsilon);
        assert_eq!(data[2], 1.0 - epsilon);
        assert_eq!(data[3], -1.0 - epsilon);

        // Check that bounds have correct width
        let width = input.max_width();
        assert!((width - 2.0 * epsilon).abs() < 1e-6);
    }

    #[test]
    fn test_truncate_name_edge_cases() {
        // Empty string
        assert_eq!(truncate_name("", 10), "");

        // Exactly width length
        assert_eq!(truncate_name("1234567890", 10), "1234567890");

        // One over: 11 chars, width 10 -> keep last 7 = "5678901"
        assert_eq!(truncate_name("12345678901", 10), "...5678901");

        // Width smaller than "...": 5 chars, width 3 -> formula gives: 5-3+3=5, so &name[5..] = ""
        // Result is just "..."
        assert_eq!(truncate_name("hello", 3), "...");

        // Width 4 with "hello": 5-4+3=4, &name[4..] = "o" -> "...o"
        assert_eq!(truncate_name("hello", 4), "...o");
    }

    const TEST_MODELS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/models");

    fn test_model_path(name: &str) -> String {
        format!("{}/{}", TEST_MODELS_DIR, name)
    }

    #[test]
    fn test_profile_bounds_simple_mlp() {
        let model_path = test_model_path("simple_mlp.onnx");
        if !std::path::Path::new(&model_path).exists() {
            eprintln!("Test model not found: {}, skipping", model_path);
            return;
        }

        let config = ProfileConfig::default();
        let result = profile_bounds(&model_path, &config).expect("Failed to profile bounds");

        // Should have layers
        assert!(!result.layers.is_empty());

        // All growth ratios should be positive
        for layer in &result.layers {
            assert!(
                layer.growth_ratio > 0.0 || layer.growth_ratio.is_nan(),
                "Growth ratio should be positive for layer {}",
                layer.name
            );
        }

        // Print summary for debugging
        eprintln!("{}", result.summary());
    }
}
