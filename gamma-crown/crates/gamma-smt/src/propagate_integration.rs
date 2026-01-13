//! Integration between gamma-propagate and gamma-smt.
//!
//! This module provides high-level APIs for verifying neural networks
//! by combining:
//! - gamma-propagate: Fast IBP/CROWN bound computation for intermediate layers
//! - gamma-smt: Complete SMT-based verification via Z4
//!
//! The typical workflow is:
//! 1. Load a network with gamma-propagate
//! 2. Compute intermediate bounds with IBP/CROWN (fast, sound, incomplete)
//! 3. Use SMT to verify specific properties with those bounds (slower, complete)

use crate::verifier::SmtVerifier;
use crate::{Result, SmtError};
use gamma_core::{Bound, VerificationResult};
use gamma_propagate::layers::Layer;
use gamma_propagate::Network;
use gamma_tensor::BoundedTensor;

/// Network parameters extracted for SMT encoding: (weights, biases, layer_dims).
type NetworkParams = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>);

/// Method for computing intermediate bounds.
#[derive(Debug, Clone, Copy, Default)]
pub enum BoundMethod {
    /// Fast IBP (interval bound propagation) - O(n)
    #[default]
    Ibp,
    /// CROWN-IBP hybrid - tighter but slower - O(n^2)
    CrownIbp,
}

/// Configuration for integrated verification.
#[derive(Debug, Clone)]
pub struct IntegratedVerifierConfig {
    /// Method for computing intermediate bounds.
    pub bound_method: BoundMethod,
    /// Use Big-M encoding for complete ReLU encoding (slower).
    pub use_bigm: bool,
    /// Big-M constant.
    pub big_m: f64,
    /// Timeout in milliseconds for SMT solving.
    pub timeout_ms: Option<u64>,
    /// Produce UNSAT proof certificates.
    /// When enabled, verified results include an Alethe-format proof.
    pub produce_proofs: bool,
}

impl Default for IntegratedVerifierConfig {
    fn default() -> Self {
        Self {
            bound_method: BoundMethod::Ibp,
            use_bigm: false,
            big_m: 1e6,
            timeout_ms: Some(60_000), // 60 seconds
            produce_proofs: false,
        }
    }
}

/// Integrated verifier combining bound propagation and SMT solving.
pub struct IntegratedVerifier {
    smt_verifier: SmtVerifier,
    config: IntegratedVerifierConfig,
}

impl IntegratedVerifier {
    /// Create a new integrated verifier with default config.
    pub fn new() -> Self {
        Self::with_config(IntegratedVerifierConfig::default())
    }

    /// Create a new integrated verifier with custom config.
    pub fn with_config(config: IntegratedVerifierConfig) -> Self {
        let smt_config = crate::verifier::SmtVerifierConfig {
            use_bigm: config.use_bigm,
            big_m: config.big_m,
            timeout_ms: config.timeout_ms,
            produce_proofs: config.produce_proofs,
        };
        Self {
            smt_verifier: SmtVerifier::with_config(smt_config),
            config,
        }
    }

    /// Verify a feedforward network property.
    ///
    /// # Arguments
    /// * `network` - The gamma-propagate Network to verify
    /// * `input_bounds` - Input region as BoundedTensor
    /// * `output_property` - Required output bounds
    ///
    /// # Returns
    /// Verification result with potential counterexample.
    pub fn verify(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_property: &[Bound],
    ) -> Result<VerificationResult> {
        // Step 1: Compute intermediate bounds using IBP or CROWN-IBP
        let intermediate_bounds = match self.config.bound_method {
            BoundMethod::Ibp => network
                .collect_ibp_bounds(input_bounds)
                .map_err(|e| SmtError::EncodingError(format!("IBP failed: {}", e)))?,
            BoundMethod::CrownIbp => network
                .collect_crown_ibp_bounds(input_bounds)
                .map_err(|e| SmtError::EncodingError(format!("CROWN-IBP failed: {}", e)))?,
        };

        // Step 2: Convert network to SMT format
        let (weights, biases, layer_dims) = extract_network_params(network)?;
        let input_bounds_vec = bounded_tensor_to_bounds(input_bounds);
        let intermediate_bounds_vec = convert_intermediate_bounds(&intermediate_bounds, network)?;

        // Step 3: Run SMT verification
        self.smt_verifier.verify_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds_vec,
            output_property,
            &intermediate_bounds_vec,
        )
    }

    /// Verify that network output satisfies given bounds.
    ///
    /// This is a convenience method that automatically creates output bounds.
    pub fn verify_output_bounds(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_lower: &[f32],
        output_upper: &[f32],
    ) -> Result<VerificationResult> {
        if output_lower.len() != output_upper.len() {
            return Err(SmtError::InvalidBounds(format!(
                "output bounds length mismatch: {} vs {}",
                output_lower.len(),
                output_upper.len()
            )));
        }

        let output_property: Vec<Bound> = output_lower
            .iter()
            .zip(output_upper.iter())
            .map(|(&l, &u)| Bound::new(l, u))
            .collect();

        self.verify(network, input_bounds, &output_property)
    }
}

impl Default for IntegratedVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract weights, biases, and layer dimensions from a Network.
///
/// Only supports networks with Linear and ReLU layers.
fn extract_network_params(network: &Network) -> Result<NetworkParams> {
    let mut weights: Vec<Vec<f64>> = Vec::new();
    let mut biases: Vec<Vec<f64>> = Vec::new();
    let mut layer_dims: Vec<usize> = Vec::new();

    let mut first_layer = true;

    for layer in &network.layers {
        match layer {
            Layer::Linear(linear) => {
                let (out_dim, in_dim) = linear.weight.dim();

                if first_layer {
                    layer_dims.push(in_dim);
                    first_layer = false;
                }

                layer_dims.push(out_dim);

                // Flatten weight matrix (row-major: out_dim x in_dim)
                let weight_vec: Vec<f64> = linear.weight.iter().map(|&x| x as f64).collect();
                weights.push(weight_vec);

                // Handle optional bias
                let bias_vec: Vec<f64> = match &linear.bias {
                    Some(b) => b.iter().map(|&x| x as f64).collect(),
                    None => vec![0.0; out_dim],
                };
                biases.push(bias_vec);
            }
            Layer::ReLU(_) => {
                // ReLU is handled implicitly in SMT encoding
                // (applied after each linear layer except the last)
            }
            _ => {
                return Err(SmtError::UnsupportedLayer(format!(
                    "SMT verification only supports Linear and ReLU layers, got {:?}",
                    std::mem::discriminant(layer)
                )));
            }
        }
    }

    if weights.is_empty() {
        return Err(SmtError::EncodingError(
            "No linear layers found".to_string(),
        ));
    }

    Ok((weights, biases, layer_dims))
}

/// Convert a BoundedTensor to a vector of Bounds (flattened).
fn bounded_tensor_to_bounds(tensor: &BoundedTensor) -> Vec<Bound> {
    tensor
        .lower
        .iter()
        .zip(tensor.upper.iter())
        .map(|(&l, &u)| Bound::new(l, u))
        .collect()
}

/// Convert intermediate bounds from IBP/CROWN to the format needed by SMT.
///
/// The intermediate bounds are pre-activation bounds for each hidden layer.
/// We need bounds after each linear layer, before ReLU.
fn convert_intermediate_bounds(
    layer_outputs: &[BoundedTensor],
    network: &Network,
) -> Result<Vec<Vec<Bound>>> {
    let mut intermediate_bounds = Vec::new();

    // Map layer output index to the bounds
    // Skip the last layer (output layer) - it doesn't need ReLU bounds
    let mut linear_layer_idx = 0;
    let num_linear_layers = network
        .layers
        .iter()
        .filter(|l| matches!(l, Layer::Linear(_)))
        .count();

    for (i, layer) in network.layers.iter().enumerate() {
        if matches!(layer, Layer::Linear(_)) {
            // Check if there's a ReLU following this linear layer
            let has_following_relu = network
                .layers
                .get(i + 1)
                .map(|l| matches!(l, Layer::ReLU(_)))
                .unwrap_or(false);

            // If this is not the last linear layer and has ReLU, record bounds
            if has_following_relu && linear_layer_idx < num_linear_layers - 1 {
                // The bounds at this point are the output of the linear layer
                // (before ReLU activation)
                if linear_layer_idx < layer_outputs.len() {
                    let bounds = bounded_tensor_to_bounds(&layer_outputs[linear_layer_idx]);
                    intermediate_bounds.push(bounds);
                }
            }
            linear_layer_idx += 1;
        }
    }

    Ok(intermediate_bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gamma_propagate::layers::{LinearLayer, ReLULayer};
    use ndarray::{Array1, Array2};

    fn create_simple_network() -> Network {
        // Simple 2 -> 2 -> 1 network with ReLU
        let layer1 = LinearLayer::new(
            Array2::from_shape_vec((2, 2), vec![1.0, 0.5, 0.5, 1.0]).unwrap(),
            Some(Array1::from_vec(vec![0.0, 0.0])),
        )
        .unwrap();
        let layer2 = LinearLayer::new(
            Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
            Some(Array1::from_vec(vec![0.0])),
        )
        .unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(layer1));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(layer2));
        network
    }

    #[test]
    fn test_extract_network_params() {
        let network = create_simple_network();
        let (weights, biases, layer_dims) = extract_network_params(&network).unwrap();

        assert_eq!(weights.len(), 2);
        assert_eq!(biases.len(), 2);
        assert_eq!(layer_dims, vec![2, 2, 1]);

        // Check first layer weights
        assert_eq!(weights[0].len(), 4); // 2x2
        assert_eq!(biases[0].len(), 2);

        // Check second layer weights
        assert_eq!(weights[1].len(), 2); // 1x2
        assert_eq!(biases[1].len(), 1);
    }

    #[test]
    fn test_bounded_tensor_to_bounds() {
        use ndarray::arr1;

        let tensor = BoundedTensor::new(
            arr1(&[-1.0f32, 0.0, 1.0]).into_dyn(),
            arr1(&[1.0f32, 2.0, 3.0]).into_dyn(),
        )
        .unwrap();

        let bounds = bounded_tensor_to_bounds(&tensor);

        assert_eq!(bounds.len(), 3);
        assert!((bounds[0].lower - (-1.0)).abs() < 1e-6);
        assert!((bounds[0].upper - 1.0).abs() < 1e-6);
        assert!((bounds[1].lower - 0.0).abs() < 1e-6);
        assert!((bounds[1].upper - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_integrated_verification() {
        use ndarray::arr1;

        let network = create_simple_network();
        let verifier = IntegratedVerifier::new();

        // Input: [0, 1] x [0, 1]
        let input = BoundedTensor::new(
            arr1(&[0.0f32, 0.0]).into_dyn(),
            arr1(&[1.0f32, 1.0]).into_dyn(),
        )
        .unwrap();

        // First compute what the output range should be
        // Layer 1: [[1, 0.5], [0.5, 1]] @ [0..1, 0..1] + [0, 0]
        //   = [0..1.5, 0..1.5]
        // ReLU: same (already non-negative)
        // Layer 2: [1, 1] @ [0..1.5, 0..1.5] + 0 = [0..3]

        // Property: output in [0, 4] should VERIFY
        let result = verifier
            .verify(&network, &input, &[Bound::new(0.0, 4.0)])
            .unwrap();

        assert!(
            result.is_verified(),
            "Expected verified for loose bounds, got {:?}",
            result
        );
    }

    #[test]
    fn test_integrated_verification_violated() {
        use ndarray::arr1;

        let network = create_simple_network();
        let verifier = IntegratedVerifier::new();

        let input = BoundedTensor::new(
            arr1(&[0.0f32, 0.0]).into_dyn(),
            arr1(&[1.0f32, 1.0]).into_dyn(),
        )
        .unwrap();

        // Property: output in [1, 2] should be VIOLATED
        // (output can be 0 when input is [0, 0])
        let result = verifier
            .verify(&network, &input, &[Bound::new(1.0, 2.0)])
            .unwrap();

        assert!(
            matches!(result, VerificationResult::Violated { .. }),
            "Expected violated, got {:?}",
            result
        );
    }

    // ===== Tests for Default implementations =====

    #[test]
    fn test_bound_method_default() {
        let method = BoundMethod::default();
        assert!(matches!(method, BoundMethod::Ibp));
    }

    #[test]
    fn test_integrated_verifier_config_default() {
        let config = IntegratedVerifierConfig::default();
        assert!(matches!(config.bound_method, BoundMethod::Ibp));
        assert!(!config.use_bigm);
        assert!((config.big_m - 1e6).abs() < 1e-10);
        assert_eq!(config.timeout_ms, Some(60_000));
        assert!(!config.produce_proofs);
    }

    #[test]
    fn test_integrated_verifier_default() {
        let verifier = IntegratedVerifier::default();
        // Just ensure it constructs without panicking
        assert!(matches!(verifier.config.bound_method, BoundMethod::Ibp));
    }

    // ===== Tests for custom config =====

    #[test]
    fn test_integrated_verifier_with_custom_config() {
        let config = IntegratedVerifierConfig {
            bound_method: BoundMethod::CrownIbp,
            use_bigm: true,
            big_m: 1e4,
            timeout_ms: Some(30_000),
            produce_proofs: false,
        };
        let verifier = IntegratedVerifier::with_config(config);

        assert!(matches!(
            verifier.config.bound_method,
            BoundMethod::CrownIbp
        ));
        assert!(verifier.config.use_bigm);
        assert!((verifier.config.big_m - 1e4).abs() < 1e-10);
        assert_eq!(verifier.config.timeout_ms, Some(30_000));
    }

    #[test]
    fn test_config_no_timeout() {
        let config = IntegratedVerifierConfig {
            bound_method: BoundMethod::Ibp,
            use_bigm: false,
            big_m: 1e6,
            timeout_ms: None,
            produce_proofs: false,
        };
        let verifier = IntegratedVerifier::with_config(config);
        assert_eq!(verifier.config.timeout_ms, None);
    }

    // ===== Tests for verify_output_bounds =====

    #[test]
    fn test_verify_output_bounds() {
        use ndarray::arr1;

        let network = create_simple_network();
        let verifier = IntegratedVerifier::new();

        let input = BoundedTensor::new(
            arr1(&[0.0f32, 0.0]).into_dyn(),
            arr1(&[1.0f32, 1.0]).into_dyn(),
        )
        .unwrap();

        // Output is in [0, 3], so [0, 4] should verify
        let result = verifier
            .verify_output_bounds(&network, &input, &[0.0], &[4.0])
            .unwrap();

        assert!(result.is_verified(), "Expected verified, got {:?}", result);
    }

    #[test]
    fn test_verify_output_bounds_violated() {
        use ndarray::arr1;

        let network = create_simple_network();
        let verifier = IntegratedVerifier::new();

        let input = BoundedTensor::new(
            arr1(&[0.0f32, 0.0]).into_dyn(),
            arr1(&[1.0f32, 1.0]).into_dyn(),
        )
        .unwrap();

        // Output is in [0, 3], requiring [1, 2] should be violated
        let result = verifier
            .verify_output_bounds(&network, &input, &[1.0], &[2.0])
            .unwrap();

        assert!(
            matches!(result, VerificationResult::Violated { .. }),
            "Expected violated, got {:?}",
            result
        );
    }

    #[test]
    fn test_verify_output_bounds_length_mismatch() {
        use ndarray::arr1;

        let network = create_simple_network();
        let verifier = IntegratedVerifier::new();

        let input = BoundedTensor::new(
            arr1(&[0.0f32, 0.0]).into_dyn(),
            arr1(&[1.0f32, 1.0]).into_dyn(),
        )
        .unwrap();

        // Mismatched lengths should error
        let result = verifier.verify_output_bounds(&network, &input, &[0.0, 0.0], &[1.0]);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{:?}", err).contains("length mismatch"),
            "Expected length mismatch error, got {:?}",
            err
        );
    }

    // ===== Tests for extract_network_params edge cases =====

    #[test]
    fn test_extract_network_params_empty_network() {
        let network = Network::new();
        let result = extract_network_params(&network);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{:?}", err).contains("No linear layers"),
            "Expected no linear layers error, got {:?}",
            err
        );
    }

    #[test]
    fn test_extract_network_params_linear_only() {
        // Network with just a single linear layer, no ReLU
        let layer = LinearLayer::new(
            Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            Some(Array1::from_vec(vec![0.1, 0.2, 0.3])),
        )
        .unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(layer));

        let (weights, biases, layer_dims) = extract_network_params(&network).unwrap();

        assert_eq!(weights.len(), 1);
        assert_eq!(biases.len(), 1);
        assert_eq!(layer_dims, vec![2, 3]);
        assert_eq!(weights[0].len(), 6); // 3x2
        assert_eq!(biases[0].len(), 3);
    }

    #[test]
    fn test_extract_network_params_no_bias() {
        // Linear layer without bias
        let layer = LinearLayer::new(
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            None, // No bias
        )
        .unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(layer));

        let (weights, biases, layer_dims) = extract_network_params(&network).unwrap();

        assert_eq!(weights.len(), 1);
        assert_eq!(biases.len(), 1);
        assert_eq!(layer_dims, vec![2, 2]);
        // When no bias, should be zeros
        assert_eq!(biases[0], vec![0.0, 0.0]);
    }

    #[test]
    fn test_extract_network_params_deeper_network() {
        // 3 -> 4 -> 4 -> 2 network
        let layer1 = LinearLayer::new(
            Array2::from_shape_vec((4, 3), vec![1.0; 12]).unwrap(),
            Some(Array1::from_vec(vec![0.0; 4])),
        )
        .unwrap();
        let layer2 = LinearLayer::new(
            Array2::from_shape_vec((4, 4), vec![1.0; 16]).unwrap(),
            Some(Array1::from_vec(vec![0.0; 4])),
        )
        .unwrap();
        let layer3 = LinearLayer::new(
            Array2::from_shape_vec((2, 4), vec![1.0; 8]).unwrap(),
            Some(Array1::from_vec(vec![0.0; 2])),
        )
        .unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(layer1));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(layer2));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(layer3));

        let (weights, biases, layer_dims) = extract_network_params(&network).unwrap();

        assert_eq!(weights.len(), 3);
        assert_eq!(biases.len(), 3);
        assert_eq!(layer_dims, vec![3, 4, 4, 2]);
    }

    #[test]
    fn test_extract_network_params_unsupported_layer() {
        use gamma_propagate::layers::Conv2dLayer;
        use ndarray::ArrayD;

        // Create a network with an unsupported layer type
        let linear =
            LinearLayer::new(Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap(), None).unwrap();

        // Conv2d is not supported by SMT verification
        let conv = Conv2dLayer::new(
            ArrayD::from_shape_vec(vec![1, 1, 1, 1], vec![1.0f32]).unwrap(),
            None,
            (1, 1),
            (0, 0),
        )
        .unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));
        network.add_layer(Layer::Conv2d(conv));

        let result = extract_network_params(&network);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{:?}", err).contains("only supports Linear and ReLU"),
            "Expected unsupported layer error, got {:?}",
            err
        );
    }

    // ===== Tests for bounded_tensor_to_bounds edge cases =====

    #[test]
    fn test_bounded_tensor_to_bounds_single_element() {
        use ndarray::arr1;

        let tensor =
            BoundedTensor::new(arr1(&[-5.0f32]).into_dyn(), arr1(&[5.0f32]).into_dyn()).unwrap();

        let bounds = bounded_tensor_to_bounds(&tensor);

        assert_eq!(bounds.len(), 1);
        assert!((bounds[0].lower - (-5.0)).abs() < 1e-6);
        assert!((bounds[0].upper - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_bounded_tensor_to_bounds_large() {
        use ndarray::Array1;

        let n = 100;
        let lower: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let upper: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();

        let tensor = BoundedTensor::new(
            Array1::from_vec(lower.clone()).into_dyn(),
            Array1::from_vec(upper.clone()).into_dyn(),
        )
        .unwrap();

        let bounds = bounded_tensor_to_bounds(&tensor);

        assert_eq!(bounds.len(), n);
        for i in 0..n {
            assert!((bounds[i].lower - lower[i]).abs() < 1e-6);
            assert!((bounds[i].upper - upper[i]).abs() < 1e-6);
        }
    }

    // ===== Tests for convert_intermediate_bounds =====

    #[test]
    fn test_convert_intermediate_bounds_simple_network() {
        use ndarray::arr1;

        let network = create_simple_network();

        // Simulate intermediate bounds from IBP
        // After first linear layer (before ReLU): BoundedTensor
        let layer_outputs = vec![
            BoundedTensor::new(
                arr1(&[-1.0f32, -1.0]).into_dyn(),
                arr1(&[2.0f32, 2.0]).into_dyn(),
            )
            .unwrap(),
            // Output layer bounds (not used for intermediate)
            BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[4.0f32]).into_dyn()).unwrap(),
        ];

        let intermediate = convert_intermediate_bounds(&layer_outputs, &network).unwrap();

        // Should have 1 intermediate bound (after first linear, before ReLU)
        assert_eq!(intermediate.len(), 1);
        assert_eq!(intermediate[0].len(), 2);
    }

    #[test]
    fn test_convert_intermediate_bounds_deeper_network() {
        use ndarray::arr1;

        // Create 3-layer network
        let layer1 =
            LinearLayer::new(Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap(), None).unwrap();
        let layer2 =
            LinearLayer::new(Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap(), None).unwrap();
        let layer3 =
            LinearLayer::new(Array2::from_shape_vec((1, 2), vec![1.0; 2]).unwrap(), None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(layer1));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(layer2));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(layer3));

        let layer_outputs = vec![
            BoundedTensor::new(
                arr1(&[-1.0f32, -1.0]).into_dyn(),
                arr1(&[2.0f32, 2.0]).into_dyn(),
            )
            .unwrap(),
            BoundedTensor::new(
                arr1(&[-2.0f32, -2.0]).into_dyn(),
                arr1(&[4.0f32, 4.0]).into_dyn(),
            )
            .unwrap(),
            BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[8.0f32]).into_dyn()).unwrap(),
        ];

        let intermediate = convert_intermediate_bounds(&layer_outputs, &network).unwrap();

        // Should have 2 intermediate bounds (after 1st and 2nd linear layers)
        assert_eq!(intermediate.len(), 2);
    }

    #[test]
    fn test_convert_intermediate_bounds_no_relus() {
        use ndarray::arr1;

        // Network without ReLUs
        let layer1 =
            LinearLayer::new(Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap(), None).unwrap();
        let layer2 =
            LinearLayer::new(Array2::from_shape_vec((1, 2), vec![1.0; 2]).unwrap(), None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(layer1));
        network.add_layer(Layer::Linear(layer2));

        let layer_outputs = vec![BoundedTensor::new(
            arr1(&[-1.0f32, -1.0]).into_dyn(),
            arr1(&[1.0f32, 1.0]).into_dyn(),
        )
        .unwrap()];

        let intermediate = convert_intermediate_bounds(&layer_outputs, &network).unwrap();

        // No ReLUs means no intermediate bounds needed
        assert!(intermediate.is_empty());
    }

    // ===== Tests for BoundMethod::CrownIbp =====

    #[test]
    fn test_verification_with_crown_ibp() {
        use ndarray::arr1;

        let network = create_simple_network();
        let config = IntegratedVerifierConfig {
            bound_method: BoundMethod::CrownIbp,
            ..Default::default()
        };
        let verifier = IntegratedVerifier::with_config(config);

        let input = BoundedTensor::new(
            arr1(&[0.0f32, 0.0]).into_dyn(),
            arr1(&[1.0f32, 1.0]).into_dyn(),
        )
        .unwrap();

        // Property: output in [0, 4] should VERIFY
        let result = verifier
            .verify(&network, &input, &[Bound::new(0.0, 4.0)])
            .unwrap();

        assert!(
            result.is_verified(),
            "Expected verified with CROWN-IBP, got {:?}",
            result
        );
    }

    // ===== Tests for BoundMethod Debug/Clone =====

    #[test]
    fn test_bound_method_debug() {
        let ibp = BoundMethod::Ibp;
        let crown = BoundMethod::CrownIbp;

        assert!(format!("{:?}", ibp).contains("Ibp"));
        assert!(format!("{:?}", crown).contains("CrownIbp"));
    }

    #[test]
    fn test_bound_method_clone() {
        let method = BoundMethod::CrownIbp;
        let cloned = method;
        assert!(matches!(cloned, BoundMethod::CrownIbp));
    }

    #[test]
    fn test_config_debug() {
        let config = IntegratedVerifierConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("IntegratedVerifierConfig"));
        assert!(debug_str.contains("bound_method"));
    }

    #[test]
    fn test_config_clone() {
        let config = IntegratedVerifierConfig {
            bound_method: BoundMethod::CrownIbp,
            use_bigm: true,
            big_m: 1e5,
            timeout_ms: Some(10_000),
            produce_proofs: true,
        };
        let cloned = config.clone();
        assert!(matches!(cloned.bound_method, BoundMethod::CrownIbp));
        assert!(cloned.use_bigm);
        assert!((cloned.big_m - 1e5).abs() < 1e-10);
        assert_eq!(cloned.timeout_ms, Some(10_000));
        assert!(cloned.produce_proofs);
    }
}
