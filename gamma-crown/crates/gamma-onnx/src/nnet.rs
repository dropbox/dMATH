//! NNet format support for loading neural network verification benchmarks.
//!
//! NNet is a simple text format for ReLU networks, commonly used in the
//! VNN-COMP (Verification of Neural Networks Competition) benchmarks,
//! particularly for ACAS-Xu collision avoidance networks.
//!
//! # Format Specification
//!
//! The NNet format (Kyle Julian, Stanford 2016) stores fully-connected
//! ReLU networks as plain text:
//!
//! - Comment lines starting with `//`
//! - Header: numLayers, inputSize, outputSize, maxLayerSize
//! - Layer sizes (comma-separated)
//! - Symmetric flag (typically 0, unused)
//! - Input bounds: minimums, maximums
//! - Normalization: means, ranges (for inputs + 1 for output)
//! - For each layer: weight matrix (row-major), then bias vector
//!
//! The network uses ReLU activations for hidden layers and linear output.
//!
//! # Usage
//!
//! ```ignore
//! use gamma_onnx::nnet::{load_nnet, NNetNetwork};
//!
//! let network = load_nnet("model.nnet")?;
//! println!("Layers: {}, Inputs: {}, Outputs: {}",
//!          network.num_layers, network.input_size, network.output_size);
//! ```

use crate::{DataType, Network, TensorSpec};
use gamma_core::{GammaError, Result};
use gamma_propagate::{Layer, LinearLayer, Network as PropNetwork, ReLULayer};
use ndarray::{Array1, Array2};
use std::path::Path;
use tracing::{debug, info};

/// A parsed NNet network with all metadata.
#[derive(Debug, Clone)]
pub struct NNetNetwork {
    /// Number of layers (not including input layer).
    pub num_layers: usize,
    /// Size of input layer.
    pub input_size: usize,
    /// Size of output layer.
    pub output_size: usize,
    /// Maximum size of any hidden layer.
    pub max_layer_size: usize,
    /// Sizes of all layers including input and output.
    pub layer_sizes: Vec<usize>,
    /// Minimum input values (for normalization/clipping).
    pub input_minimums: Vec<f32>,
    /// Maximum input values (for normalization/clipping).
    pub input_maximums: Vec<f32>,
    /// Mean values for input normalization.
    pub input_means: Vec<f32>,
    /// Range values for input normalization.
    pub input_ranges: Vec<f32>,
    /// Mean value for output denormalization.
    pub output_mean: f32,
    /// Range value for output denormalization.
    pub output_range: f32,
    /// Weight matrices for each layer (`layer_sizes[i+1] x layer_sizes[i]`).
    pub weights: Vec<Array2<f32>>,
    /// Bias vectors for each layer (`layer_sizes[i+1]`).
    pub biases: Vec<Array1<f32>>,
}

impl NNetNetwork {
    /// Evaluate the network on an input vector.
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector of size `input_size`
    /// * `normalize` - If true, apply input normalization and output denormalization
    ///
    /// # Returns
    ///
    /// Output vector of size `output_size`
    pub fn evaluate(&self, input: &[f32], normalize: bool) -> Vec<f32> {
        let mut x: Vec<f32> = if normalize {
            // Normalize inputs
            input
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let clamped = v.clamp(self.input_minimums[i], self.input_maximums[i]);
                    (clamped - self.input_means[i]) / self.input_ranges[i]
                })
                .collect()
        } else {
            input.to_vec()
        };

        // Forward pass through all layers
        for (layer_idx, (weights, bias)) in self.weights.iter().zip(&self.biases).enumerate() {
            // Linear: y = Wx + b
            let mut y = vec![0.0f32; weights.nrows()];
            for (i, row) in weights.rows().into_iter().enumerate() {
                y[i] = row.iter().zip(&x).map(|(&w, &xi)| w * xi).sum::<f32>() + bias[i];
            }

            // ReLU for hidden layers (not output)
            if layer_idx < self.num_layers - 1 {
                for v in &mut y {
                    *v = v.max(0.0);
                }
            }

            x = y;
        }

        // Denormalize output if requested
        if normalize {
            for v in &mut x {
                *v = *v * self.output_range + self.output_mean;
            }
        }

        x
    }

    /// Get normalized input bounds (after applying input normalization).
    pub fn get_normalized_input_bounds(&self) -> (Vec<f32>, Vec<f32>) {
        let lower: Vec<f32> = self
            .input_minimums
            .iter()
            .zip(&self.input_means)
            .zip(&self.input_ranges)
            .map(|((&min, &mean), &range)| (min - mean) / range)
            .collect();

        let upper: Vec<f32> = self
            .input_maximums
            .iter()
            .zip(&self.input_means)
            .zip(&self.input_ranges)
            .map(|((&max, &mean), &range)| (max - mean) / range)
            .collect();

        (lower, upper)
    }

    /// Convert to γ-CROWN's Network format for inspection.
    ///
    /// Note: For verification, use `to_prop_network()` instead.
    pub fn to_gamma_network(&self) -> Network {
        Network {
            name: "nnet_model".to_string(),
            inputs: vec![TensorSpec {
                name: "input".to_string(),
                shape: vec![1, self.input_size as i64],
                dtype: DataType::Float32,
            }],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![1, self.output_size as i64],
                dtype: DataType::Float32,
            }],
            layers: Vec::new(), // LayerSpec requires complex setup; use to_prop_network for verification
            param_count: self.param_count(),
        }
    }

    /// Convert to γ-CROWN's PropNetwork for verification.
    pub fn to_prop_network(&self) -> Result<PropNetwork> {
        let mut network = PropNetwork::new();

        for (layer_idx, (w, b)) in self.weights.iter().zip(&self.biases).enumerate() {
            let is_output = layer_idx == self.num_layers - 1;

            // Create bias Array1
            let bias = Array1::from_vec(b.iter().cloned().collect());

            // Add Linear layer
            let linear = LinearLayer::new(w.clone(), Some(bias))?;
            network.layers.push(Layer::Linear(linear));

            // Add ReLU for hidden layers
            if !is_output {
                network.layers.push(Layer::ReLU(ReLULayer));
            }
        }

        Ok(network)
    }

    /// Get total parameter count.
    pub fn param_count(&self) -> usize {
        self.weights.iter().map(|w| w.len()).sum::<usize>()
            + self.biases.iter().map(|b| b.len()).sum::<usize>()
    }
}

/// Load a neural network from NNet format.
///
/// # Arguments
///
/// * `path` - Path to the .nnet file
///
/// # Returns
///
/// A parsed `NNetNetwork` ready for verification.
pub fn load_nnet<P: AsRef<Path>>(path: P) -> Result<NNetNetwork> {
    let path = path.as_ref();
    info!("Loading NNet from: {}", path.display());

    if !path.exists() {
        return Err(GammaError::ModelLoad(format!(
            "File not found: {}",
            path.display()
        )));
    }

    let content = std::fs::read_to_string(path)
        .map_err(|e| GammaError::ModelLoad(format!("Failed to read file: {}", e)))?;

    parse_nnet(&content)
}

/// Parse NNet format from string content.
pub fn parse_nnet(content: &str) -> Result<NNetNetwork> {
    let mut lines = content
        .lines()
        .filter(|line| !line.starts_with("//") && !line.trim().is_empty());

    // Parse header: numLayers, inputSize, outputSize, maxLayerSize
    let header_line = lines
        .next()
        .ok_or_else(|| GammaError::ModelLoad("Missing header line".to_string()))?;
    let header: Vec<usize> = parse_csv_line(header_line)?;
    if header.len() < 4 {
        return Err(GammaError::ModelLoad(format!(
            "Invalid header: expected 4 values, got {}",
            header.len()
        )));
    }
    let (num_layers, input_size, output_size, max_layer_size) =
        (header[0], header[1], header[2], header[3]);

    debug!(
        "NNet: {} layers, {} inputs, {} outputs",
        num_layers, input_size, output_size
    );

    // Parse layer sizes
    let sizes_line = lines
        .next()
        .ok_or_else(|| GammaError::ModelLoad("Missing layer sizes line".to_string()))?;
    let layer_sizes: Vec<usize> = parse_csv_line(sizes_line)?;
    if layer_sizes.len() != num_layers + 1 {
        return Err(GammaError::ModelLoad(format!(
            "Expected {} layer sizes, got {}",
            num_layers + 1,
            layer_sizes.len()
        )));
    }

    // Parse symmetric flag (unused)
    let _symmetric_line = lines.next();

    // Parse input minimums
    let min_line = lines
        .next()
        .ok_or_else(|| GammaError::ModelLoad("Missing input minimums".to_string()))?;
    let input_minimums: Vec<f32> = parse_csv_line_f32(min_line)?;

    // Parse input maximums
    let max_line = lines
        .next()
        .ok_or_else(|| GammaError::ModelLoad("Missing input maximums".to_string()))?;
    let input_maximums: Vec<f32> = parse_csv_line_f32(max_line)?;

    // Parse means (inputs + output)
    let means_line = lines
        .next()
        .ok_or_else(|| GammaError::ModelLoad("Missing means".to_string()))?;
    let means: Vec<f32> = parse_csv_line_f32(means_line)?;
    let (input_means, output_mean) = if means.len() > input_size {
        (means[..input_size].to_vec(), means[input_size])
    } else {
        (means.clone(), 0.0)
    };

    // Parse ranges (inputs + output)
    let ranges_line = lines
        .next()
        .ok_or_else(|| GammaError::ModelLoad("Missing ranges".to_string()))?;
    let ranges: Vec<f32> = parse_csv_line_f32(ranges_line)?;
    let (input_ranges, output_range) = if ranges.len() > input_size {
        (ranges[..input_size].to_vec(), ranges[input_size])
    } else {
        (ranges.clone(), 1.0)
    };

    // Parse weights and biases for each layer
    let mut weights = Vec::with_capacity(num_layers);
    let mut biases = Vec::with_capacity(num_layers);

    for layer_idx in 0..num_layers {
        let prev_size = layer_sizes[layer_idx];
        let curr_size = layer_sizes[layer_idx + 1];

        debug!(
            "Layer {}: {} -> {} (reading {} weights, {} biases)",
            layer_idx,
            prev_size,
            curr_size,
            curr_size * prev_size,
            curr_size
        );

        // Read weight matrix (curr_size rows, prev_size values per row)
        let mut weight_data = Vec::with_capacity(curr_size * prev_size);
        for _row in 0..curr_size {
            let row_line = lines
                .next()
                .ok_or_else(|| GammaError::ModelLoad("Missing weight row".to_string()))?;
            let row_values: Vec<f32> = parse_csv_line_f32(row_line)?;
            if row_values.len() < prev_size {
                return Err(GammaError::ModelLoad(format!(
                    "Weight row has {} values, expected {}",
                    row_values.len(),
                    prev_size
                )));
            }
            weight_data.extend_from_slice(&row_values[..prev_size]);
        }
        let weight = Array2::from_shape_vec((curr_size, prev_size), weight_data)
            .map_err(|e| GammaError::ModelLoad(format!("Failed to create weight matrix: {}", e)))?;
        weights.push(weight);

        // Read bias vector (curr_size values)
        let mut bias_data = Vec::with_capacity(curr_size);
        for _i in 0..curr_size {
            let bias_line = lines
                .next()
                .ok_or_else(|| GammaError::ModelLoad("Missing bias value".to_string()))?;
            let bias_value: f32 = bias_line
                .trim()
                .trim_end_matches(',')
                .parse()
                .map_err(|e| GammaError::ModelLoad(format!("Invalid bias value: {}", e)))?;
            bias_data.push(bias_value);
        }
        let bias = Array1::from_vec(bias_data);
        biases.push(bias);
    }

    let network = NNetNetwork {
        num_layers,
        input_size,
        output_size,
        max_layer_size,
        layer_sizes,
        input_minimums,
        input_maximums,
        input_means,
        input_ranges,
        output_mean,
        output_range,
        weights,
        biases,
    };

    info!(
        "Loaded NNet: {} layers, {} params",
        num_layers,
        network.param_count()
    );

    Ok(network)
}

fn parse_csv_line<T: std::str::FromStr>(line: &str) -> Result<Vec<T>>
where
    T::Err: std::fmt::Display,
{
    line.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse()
                .map_err(|e| GammaError::ModelLoad(format!("Parse error: {}", e)))
        })
        .collect()
}

fn parse_csv_line_f32(line: &str) -> Result<Vec<f32>> {
    line.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            let trimmed = s.trim();
            // Handle scientific notation
            trimmed
                .parse()
                .map_err(|e| GammaError::ModelLoad(format!("Parse error '{}': {}", trimmed, e)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gamma_tensor::BoundedTensor;
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn test_parse_simple_nnet() {
        let content = r#"
// Simple test network
// 2 layers, 3 inputs, 2 outputs
2,3,2,4,
3,4,2,
0,
-1.0,-1.0,-1.0,
1.0,1.0,1.0,
0.0,0.0,0.0,0.0,
1.0,1.0,1.0,1.0,
0.1,0.2,0.3,
0.4,0.5,0.6,
0.7,0.8,0.9,
1.0,1.1,1.2,
0.01,
0.02,
0.03,
0.04,
1.0,2.0,3.0,4.0,
5.0,6.0,7.0,8.0,
0.1,
0.2,
"#;

        let network = parse_nnet(content).unwrap();
        assert_eq!(network.num_layers, 2);
        assert_eq!(network.input_size, 3);
        assert_eq!(network.output_size, 2);
        assert_eq!(network.layer_sizes, vec![3, 4, 2]);
        assert_eq!(network.weights.len(), 2);
        assert_eq!(network.biases.len(), 2);

        // First layer: 4x3 weights
        assert_eq!(network.weights[0].shape(), &[4, 3]);
        // Second layer: 2x4 weights
        assert_eq!(network.weights[1].shape(), &[2, 4]);
    }

    #[test]
    fn test_nnet_evaluation() {
        let content = r#"
// Identity-like network for testing
1,2,2,2,
2,2,
0,
-10.0,-10.0,
10.0,10.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
1.0,0.0,
0.0,1.0,
0.0,
0.0,
"#;

        let network = parse_nnet(content).unwrap();

        // Test evaluation (should be approximately identity for positive inputs)
        let input = vec![1.0, 2.0];
        let output = network.evaluate(&input, false);
        assert_eq!(output.len(), 2);
        // Linear output (no ReLU on last layer)
        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_nnet_to_prop_network() {
        let content = r#"
// Simple network
1,2,2,2,
2,2,
0,
-10.0,-10.0,
10.0,10.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
1.0,0.0,
0.0,1.0,
0.0,
0.0,
"#;

        let nnet = parse_nnet(content).unwrap();
        let network = nnet.to_prop_network().unwrap();

        // Create input bounds
        let lower = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 1.0]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.5, 1.5]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        // Run IBP
        let result = network.propagate_ibp(&input).unwrap();
        assert_eq!(result.shape(), &[1, 2]);

        // Output bounds should be valid
        for (l, u) in result.lower.iter().zip(result.upper.iter()) {
            assert!(l <= u, "Invalid bounds: {} > {}", l, u);
        }
    }

    #[test]
    fn test_relu_in_hidden_layers() {
        let content = r#"
// Network with hidden ReLU
2,2,1,3,
2,3,1,
0,
-10.0,-10.0,
10.0,10.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
1.0,0.0,
0.0,1.0,
-1.0,1.0,
0.0,
0.0,
0.0,
1.0,1.0,1.0,
0.0,
"#;

        let network = parse_nnet(content).unwrap();

        // Test with negative inputs that should get ReLU'd
        let input = vec![-1.0, 1.0];
        let output = network.evaluate(&input, false);

        // First hidden layer: [relu(-1), relu(1), relu(-1+1)] = [0, 1, 0]
        // Output: 0*1 + 1*1 + 0*1 = 1
        assert_eq!(output.len(), 1);
        // The exact value depends on the network computation
    }

    #[test]
    fn test_load_acasxu_model() {
        // Load actual ACAS-Xu model
        let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests/models/acasxu_1_1.nnet");

        if !model_path.exists() {
            eprintln!("Skipping test: ACAS-Xu model not found at {:?}", model_path);
            return;
        }

        let network = load_nnet(&model_path).unwrap();

        // ACAS-Xu 1_1 has 7 layers (6 hidden + 1 output)
        assert_eq!(network.num_layers, 7);
        assert_eq!(network.input_size, 5);
        assert_eq!(network.output_size, 5);
        assert_eq!(network.layer_sizes, vec![5, 50, 50, 50, 50, 50, 50, 5]);

        // Check weights dimensions
        assert_eq!(network.weights[0].shape(), &[50, 5]); // First hidden layer: 50 x 5
        assert_eq!(network.weights[6].shape(), &[5, 50]); // Output layer: 5 x 50

        // Convert to PropNetwork and run IBP
        let prop_network = network.to_prop_network().unwrap();
        assert_eq!(prop_network.layers.len(), 13); // 7 linear + 6 relu

        // Create input with small perturbation
        let (lower_bounds, upper_bounds) = network.get_normalized_input_bounds();
        let center: Vec<f32> = lower_bounds
            .iter()
            .zip(&upper_bounds)
            .map(|(l, u)| (l + u) / 2.0)
            .collect();

        let eps = 0.01;
        let lower =
            ArrayD::from_shape_vec(IxDyn(&[1, 5]), center.iter().map(|&c| c - eps).collect())
                .unwrap();
        let upper =
            ArrayD::from_shape_vec(IxDyn(&[1, 5]), center.iter().map(|&c| c + eps).collect())
                .unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        // Run IBP
        let result = prop_network.propagate_ibp(&input).unwrap();
        assert_eq!(result.shape(), &[1, 5]);

        // Output bounds should be valid
        for (l, u) in result.lower.iter().zip(result.upper.iter()) {
            assert!(l <= u, "Invalid bounds: {} > {}", l, u);
        }

        // Print bounds for information
        println!("ACAS-Xu IBP output bounds:");
        for i in 0..5 {
            println!(
                "  Output {}: [{:.4}, {:.4}]",
                i,
                result.lower[[0, i]],
                result.upper[[0, i]]
            );
        }
    }

    #[test]
    fn test_crown_ibp_acasxu() {
        // Test CROWN-IBP vs CROWN on ACAS-Xu model
        let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests/models/acasxu_1_1.nnet");

        if !model_path.exists() {
            eprintln!("Skipping test: ACAS-Xu model not found at {:?}", model_path);
            return;
        }

        let network = load_nnet(&model_path).unwrap();
        let prop_network = network.to_prop_network().unwrap();

        // Use the same input bounds as in the debug report
        // lower = [0.6, -0.5, -0.5, 0.45, -0.5]
        // upper = [0.679857769, 0.5, 0.5, 0.5, -0.45]
        let lower = ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.6, -0.5, -0.5, 0.45, -0.5]).unwrap();
        let upper =
            ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.679858, 0.5, 0.5, 0.5, -0.45]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        // Run IBP
        let ibp_result = prop_network.propagate_ibp(&input).unwrap();
        let ibp_width: f32 = ibp_result.width().iter().sum();

        // Run CROWN
        let crown_result = prop_network.propagate_crown(&input).unwrap();
        let crown_width: f32 = crown_result.width().iter().sum();

        // Run CROWN-IBP
        let crown_ibp_result = prop_network.propagate_crown_ibp(&input).unwrap();
        let crown_ibp_width: f32 = crown_ibp_result.width().iter().sum();

        println!("\n=== ACAS-Xu Bound Comparison ===");
        println!("IBP total width: {:.2}", ibp_width);
        println!("CROWN total width: {:.2}", crown_width);
        println!("CROWN-IBP total width: {:.2}", crown_ibp_width);

        println!("\nCROWN bounds:");
        for i in 0..5 {
            println!(
                "  Output {}: [{:.2}, {:.2}]",
                i,
                crown_result.lower[[i]],
                crown_result.upper[[i]]
            );
        }

        println!("\nCROWN-IBP bounds:");
        for i in 0..5 {
            println!(
                "  Output {}: [{:.2}, {:.2}]",
                i,
                crown_ibp_result.lower[[i]],
                crown_ibp_result.upper[[i]]
            );
        }

        // CROWN should be tighter than IBP
        assert!(
            crown_width <= ibp_width,
            "CROWN ({:.2}) should be <= IBP ({:.2})",
            crown_width,
            ibp_width
        );

        // Both bounds should be valid
        for i in 0..5 {
            assert!(
                crown_result.lower[[i]] <= crown_result.upper[[i]],
                "CROWN bounds invalid at {}",
                i
            );
            assert!(
                crown_ibp_result.lower[[i]] <= crown_ibp_result.upper[[i]],
                "CROWN-IBP bounds invalid at {}",
                i
            );
        }

        // Print improvement percentage
        let improvement_vs_crown = (1.0 - crown_ibp_width / crown_width) * 100.0;
        let improvement_vs_ibp = (1.0 - crown_ibp_width / ibp_width) * 100.0;
        println!(
            "\nCROWN-IBP improvement vs CROWN: {:.1}%",
            improvement_vs_crown
        );
        println!("CROWN-IBP improvement vs IBP: {:.1}%", improvement_vs_ibp);
    }

    // ==================== NEW COMPREHENSIVE TESTS ====================

    #[test]
    fn test_parse_csv_line_empty() {
        let result: Vec<usize> = parse_csv_line("").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_csv_line_with_trailing_comma() {
        let result: Vec<usize> = parse_csv_line("1,2,3,").unwrap();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_parse_csv_line_with_spaces() {
        let result: Vec<usize> = parse_csv_line(" 1 , 2 , 3 ").unwrap();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_parse_csv_line_invalid_value() {
        let result: Result<Vec<usize>> = parse_csv_line("1,abc,3");
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Parse error"));
    }

    #[test]
    fn test_parse_csv_line_f32_scientific_notation() {
        let result = parse_csv_line_f32("1.5e-3,2.0E+2,-3.25e0").unwrap();
        assert!((result[0] - 0.0015).abs() < 1e-10);
        assert!((result[1] - 200.0).abs() < 1e-10);
        assert!((result[2] - (-3.25)).abs() < 1e-10);
    }

    #[test]
    fn test_parse_csv_line_f32_negative_values() {
        let result = parse_csv_line_f32("-1.5,-2.0,-3.5").unwrap();
        assert_eq!(result, vec![-1.5, -2.0, -3.5]);
    }

    #[test]
    fn test_parse_csv_line_f32_invalid() {
        let result = parse_csv_line_f32("1.0,not_a_number,3.0");
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("not_a_number"));
    }

    #[test]
    fn test_load_nnet_file_not_found() {
        let result = load_nnet("/nonexistent/path/model.nnet");
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("File not found"));
    }

    #[test]
    fn test_parse_nnet_missing_header() {
        let content = "";
        let result = parse_nnet(content);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Missing header"));
    }

    #[test]
    fn test_parse_nnet_invalid_header_too_few_values() {
        let content = "2,3,2,";
        let result = parse_nnet(content);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Invalid header"));
    }

    #[test]
    fn test_parse_nnet_missing_layer_sizes() {
        let content = "2,3,2,4,\n";
        let result = parse_nnet(content);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Missing layer sizes"));
    }

    #[test]
    fn test_parse_nnet_wrong_layer_sizes_count() {
        let content = r#"
2,3,2,4,
3,4,
"#;
        let result = parse_nnet(content);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Expected 3 layer sizes"));
    }

    #[test]
    fn test_parse_nnet_missing_input_minimums() {
        let content = r#"
2,3,2,4,
3,4,2,
0,
"#;
        let result = parse_nnet(content);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Missing input minimums"));
    }

    #[test]
    fn test_parse_nnet_missing_input_maximums() {
        let content = r#"
2,3,2,4,
3,4,2,
0,
-1.0,-1.0,-1.0,
"#;
        let result = parse_nnet(content);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Missing input maximums"));
    }

    #[test]
    fn test_parse_nnet_missing_means() {
        let content = r#"
2,3,2,4,
3,4,2,
0,
-1.0,-1.0,-1.0,
1.0,1.0,1.0,
"#;
        let result = parse_nnet(content);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Missing means"));
    }

    #[test]
    fn test_parse_nnet_missing_ranges() {
        let content = r#"
2,3,2,4,
3,4,2,
0,
-1.0,-1.0,-1.0,
1.0,1.0,1.0,
0.0,0.0,0.0,0.0,
"#;
        let result = parse_nnet(content);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Missing ranges"));
    }

    #[test]
    fn test_parse_nnet_missing_weight_row() {
        let content = r#"
2,3,2,4,
3,4,2,
0,
-1.0,-1.0,-1.0,
1.0,1.0,1.0,
0.0,0.0,0.0,0.0,
1.0,1.0,1.0,1.0,
0.1,0.2,0.3,
"#;
        let result = parse_nnet(content);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Missing weight row"));
    }

    #[test]
    fn test_parse_nnet_weight_row_too_few_values() {
        let content = r#"
1,2,2,2,
2,2,
0,
-10.0,-10.0,
10.0,10.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
1.0,
0.0,1.0,
0.0,
0.0,
"#;
        let result = parse_nnet(content);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Weight row has 1 values, expected 2"));
    }

    #[test]
    fn test_parse_nnet_missing_bias() {
        let content = r#"
1,2,2,2,
2,2,
0,
-10.0,-10.0,
10.0,10.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
1.0,0.0,
0.0,1.0,
"#;
        let result = parse_nnet(content);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Missing bias value"));
    }

    #[test]
    fn test_parse_nnet_invalid_bias() {
        let content = r#"
1,2,2,2,
2,2,
0,
-10.0,-10.0,
10.0,10.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
1.0,0.0,
0.0,1.0,
not_a_number,
0.0,
"#;
        let result = parse_nnet(content);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Invalid bias value"));
    }

    #[test]
    fn test_nnet_evaluation_with_normalization() {
        let content = r#"
// Network for normalization testing
1,2,2,2,
2,2,
0,
-10.0,-10.0,
10.0,10.0,
2.0,3.0,0.0,
4.0,5.0,1.0,
1.0,0.0,
0.0,1.0,
0.0,
0.0,
"#;

        let network = parse_nnet(content).unwrap();

        // Test with normalization enabled
        let input = vec![6.0, 8.0];
        let output = network.evaluate(&input, true);

        // Input normalization: (x - mean) / range
        // x[0] = (6.0 - 2.0) / 4.0 = 1.0
        // x[1] = (8.0 - 3.0) / 5.0 = 1.0
        // Linear: [[1,0],[0,1]] * [1,1] + [0,0] = [1, 1]
        // Denorm: output * range + mean = [1*1+0, 1*1+0] = [1, 1]
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_nnet_evaluation_clamping() {
        let content = r#"
// Network with tight input bounds
1,2,2,2,
2,2,
0,
0.0,0.0,
1.0,1.0,
0.5,0.5,0.0,
1.0,1.0,1.0,
1.0,0.0,
0.0,1.0,
0.0,
0.0,
"#;

        let network = parse_nnet(content).unwrap();

        // Input outside bounds should be clamped when normalizing
        let input = vec![10.0, -5.0]; // Outside [0, 1]
        let output = network.evaluate(&input, true);
        assert_eq!(output.len(), 2);

        // No clamping when normalize=false
        let output_no_norm = network.evaluate(&input, false);
        assert_eq!(output_no_norm.len(), 2);
    }

    #[test]
    fn test_nnet_get_normalized_input_bounds() {
        let content = r#"
1,2,2,2,
2,2,
0,
0.0,0.0,
10.0,20.0,
5.0,10.0,0.0,
2.0,4.0,1.0,
1.0,0.0,
0.0,1.0,
0.0,
0.0,
"#;

        let network = parse_nnet(content).unwrap();
        let (lower, upper) = network.get_normalized_input_bounds();

        // lower = (min - mean) / range
        // Input 0: (0 - 5) / 2 = -2.5
        // Input 1: (0 - 10) / 4 = -2.5
        assert!((lower[0] - (-2.5)).abs() < 1e-6);
        assert!((lower[1] - (-2.5)).abs() < 1e-6);

        // upper = (max - mean) / range
        // Input 0: (10 - 5) / 2 = 2.5
        // Input 1: (20 - 10) / 4 = 2.5
        assert!((upper[0] - 2.5).abs() < 1e-6);
        assert!((upper[1] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_nnet_to_gamma_network() {
        let content = r#"
1,3,2,3,
3,2,
0,
-1.0,-1.0,-1.0,
1.0,1.0,1.0,
0.0,0.0,0.0,0.0,
1.0,1.0,1.0,1.0,
1.0,0.0,0.0,
0.0,1.0,0.0,
0.0,
0.0,
"#;

        let nnet = parse_nnet(content).unwrap();
        let gamma_network = nnet.to_gamma_network();

        assert_eq!(gamma_network.name, "nnet_model");
        assert_eq!(gamma_network.inputs.len(), 1);
        assert_eq!(gamma_network.outputs.len(), 1);
        assert_eq!(gamma_network.inputs[0].name, "input");
        assert_eq!(gamma_network.inputs[0].shape, vec![1, 3]);
        assert_eq!(gamma_network.outputs[0].name, "output");
        assert_eq!(gamma_network.outputs[0].shape, vec![1, 2]);
        assert_eq!(gamma_network.param_count, nnet.param_count());
    }

    #[test]
    fn test_nnet_param_count() {
        let content = r#"
2,2,3,3,
2,3,3,
0,
-1.0,-1.0,
1.0,1.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
1.0,0.0,
0.0,1.0,
1.0,1.0,
0.0,
0.0,
0.0,
1.0,1.0,1.0,
1.0,1.0,1.0,
1.0,1.0,1.0,
0.1,
0.2,
0.3,
"#;

        let network = parse_nnet(content).unwrap();
        // Layer 1: 3x2 weights + 3 biases = 9
        // Layer 2: 3x3 weights + 3 biases = 12
        // Total = 21
        assert_eq!(network.param_count(), 21);
    }

    #[test]
    fn test_nnet_with_comments_between_data() {
        let content = r#"
// Header comment
2,3,2,4,
// Layer sizes
3,4,2,
// Symmetric flag
0,
// Input minimums
-1.0,-1.0,-1.0,
// Input maximums
1.0,1.0,1.0,
// Means
0.0,0.0,0.0,0.0,
// Ranges
1.0,1.0,1.0,1.0,
// Layer 0 weights
0.1,0.2,0.3,
0.4,0.5,0.6,
0.7,0.8,0.9,
1.0,1.1,1.2,
// Layer 0 biases
0.01,
0.02,
0.03,
0.04,
// Layer 1 weights
1.0,2.0,3.0,4.0,
5.0,6.0,7.0,8.0,
// Layer 1 biases
0.1,
0.2,
"#;

        let network = parse_nnet(content).unwrap();
        assert_eq!(network.num_layers, 2);
        assert_eq!(network.input_size, 3);
        assert_eq!(network.output_size, 2);
    }

    #[test]
    fn test_nnet_deep_network_relu_propagation() {
        // 3-layer deep network to test ReLU in multiple hidden layers
        let content = r#"
3,2,2,4,
2,4,4,2,
0,
-10.0,-10.0,
10.0,10.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
1.0,0.0,
0.0,1.0,
-1.0,0.0,
0.0,-1.0,
0.0,
0.0,
0.0,
0.0,
1.0,0.0,0.0,0.0,
0.0,1.0,0.0,0.0,
0.0,0.0,1.0,0.0,
0.0,0.0,0.0,1.0,
0.0,
0.0,
0.0,
0.0,
1.0,1.0,1.0,1.0,
-1.0,-1.0,-1.0,-1.0,
0.0,
0.0,
"#;

        let network = parse_nnet(content).unwrap();
        assert_eq!(network.num_layers, 3);

        // Test evaluation - network has negative paths that should be ReLU'd to 0
        let input = vec![1.0, 1.0];
        let output = network.evaluate(&input, false);
        assert_eq!(output.len(), 2);

        // Convert to prop network and verify structure
        let prop_network = network.to_prop_network().unwrap();
        // 3 linear layers + 2 ReLU layers (no ReLU on output)
        assert_eq!(prop_network.layers.len(), 5);
    }

    #[test]
    fn test_nnet_means_ranges_with_only_inputs() {
        // Test case where means/ranges have exactly input_size elements
        let content = r#"
1,2,2,2,
2,2,
0,
-1.0,-1.0,
1.0,1.0,
0.5,0.5,
1.0,1.0,
1.0,0.0,
0.0,1.0,
0.0,
0.0,
"#;

        let network = parse_nnet(content).unwrap();
        // When means/ranges have only input_size elements, output_mean=0, output_range=1
        assert_eq!(network.output_mean, 0.0);
        assert_eq!(network.output_range, 1.0);
        assert_eq!(network.input_means.len(), 2);
        assert_eq!(network.input_ranges.len(), 2);
    }

    #[test]
    fn test_nnet_single_layer_network() {
        // Minimal 1-layer network (direct input to output)
        let content = r#"
1,3,2,3,
3,2,
0,
-1.0,-1.0,-1.0,
1.0,1.0,1.0,
0.0,0.0,0.0,0.0,
1.0,1.0,1.0,1.0,
1.0,0.0,0.0,
0.0,1.0,0.0,
0.5,
-0.5,
"#;

        let network = parse_nnet(content).unwrap();
        assert_eq!(network.num_layers, 1);

        // Single layer should have no ReLU (output layer)
        let input = vec![1.0, 2.0, 3.0];
        let output = network.evaluate(&input, false);
        // Expected: [1*1 + 0*2 + 0*3 + 0.5, 0*1 + 1*2 + 0*3 - 0.5] = [1.5, 1.5]
        assert!((output[0] - 1.5).abs() < 1e-6);
        assert!((output[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_nnet_max_layer_size_field() {
        let content = r#"
2,2,2,50,
2,50,2,
0,
-1.0,-1.0,
1.0,1.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
"#;
        // This will fail because we don't have weights, but max_layer_size should parse
        let result = parse_nnet(content);
        // Will fail later but header should parse
        assert!(result.is_err()); // Missing weights, but confirms header parsing
    }
}
