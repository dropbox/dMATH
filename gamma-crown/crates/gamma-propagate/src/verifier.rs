//! Main verifier struct for neural network verification.

use crate::beta_crown::{BetaCrownConfig, BetaCrownVerifier};
use crate::network::{GraphNetwork, Network};
use crate::types::{PropagationConfig, PropagationMethod};
use gamma_core::{Bound, GammaError, GemmEngine, Result, VerificationResult, VerificationSpec};
use gamma_tensor::BoundedTensor;
use ndarray::ArrayD;
use tracing::{debug, info, warn};

/// Main verifier struct.
///
/// Provides a unified interface for neural network verification using
/// different bound propagation methods (IBP, CROWN, α-CROWN, β-CROWN).
///
/// # Example
/// ```ignore
/// use gamma_propagate::{Verifier, PropagationConfig, PropagationMethod, Network};
/// use gamma_core::{VerificationSpec, Bound};
///
/// let config = PropagationConfig { method: PropagationMethod::Crown };
/// let verifier = Verifier::new(config);
///
/// let network = Network::new(); // Add layers...
/// let spec = VerificationSpec {
///     input_bounds: vec![Bound::new(-1.0, 1.0); 10],
///     output_bounds: vec![Bound::new(0.0, f32::INFINITY); 1],
///     timeout_ms: None,
///     input_shape: None,
/// };
///
/// let result = verifier.verify(&network, &spec)?;
/// ```
pub struct Verifier {
    config: PropagationConfig,
}

impl Verifier {
    pub fn new(config: PropagationConfig) -> Self {
        Self { config }
    }

    /// Verify a specification on a Network.
    ///
    /// Propagates input bounds through the network using the configured method
    /// and checks if output bounds satisfy the specification.
    pub fn verify(&self, network: &Network, spec: &VerificationSpec) -> Result<VerificationResult> {
        self.verify_with_engine(network, spec, None)
    }

    pub fn verify_with_engine(
        &self,
        network: &Network,
        spec: &VerificationSpec,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<VerificationResult> {
        info!(
            "Starting verification with {:?}, {} layers",
            self.config.method,
            network.layers.len()
        );

        // Convert input bounds to bounded tensor
        let input_bounds = self.spec_to_tensor(&spec.input_bounds, spec.input_shape.as_deref())?;

        // Propagate through network
        let output_bounds = match self.config.method {
            PropagationMethod::Ibp => network.propagate_ibp(&input_bounds)?,
            PropagationMethod::Crown => {
                network.propagate_crown_with_engine(&input_bounds, engine)?
            }
            PropagationMethod::AlphaCrown => {
                network.propagate_alpha_crown_with_engine(&input_bounds, engine)?
            }
            PropagationMethod::SdpCrown => {
                let (x_hat, rho) = Self::infer_l2_ball_from_box(&input_bounds)?;
                network.propagate_sdp_crown(&input_bounds, &x_hat, rho)?
            }
            PropagationMethod::BetaCrown => {
                return self.verify_beta_crown(network, &input_bounds, spec);
            }
        };

        // Sanitize NaN bounds: replace NaN with conservative infinities
        let output_bounds = Self::sanitize_output_bounds(output_bounds);

        // Check if output bounds satisfy spec
        self.check_spec(&output_bounds, &spec.output_bounds)
    }

    fn spec_to_tensor(&self, bounds: &[Bound], shape: Option<&[usize]>) -> Result<BoundedTensor> {
        let lower: Vec<f32> = bounds.iter().map(|b| b.lower).collect();
        let upper: Vec<f32> = bounds.iter().map(|b| b.upper).collect();

        // Use provided shape or default to 1D
        let tensor_shape: Vec<usize> = match shape {
            Some(s) => {
                // Verify total elements match
                let total: usize = s.iter().product();
                if total != bounds.len() {
                    return Err(GammaError::InvalidSpec(format!(
                        "Input shape {:?} has {} elements but bounds has {}",
                        s,
                        total,
                        bounds.len()
                    )));
                }
                s.to_vec()
            }
            None => vec![bounds.len()],
        };

        BoundedTensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&tensor_shape), lower)
                .map_err(|e| GammaError::InvalidSpec(e.to_string()))?,
            ArrayD::from_shape_vec(ndarray::IxDyn(&tensor_shape), upper)
                .map_err(|e| GammaError::InvalidSpec(e.to_string()))?,
        )
    }

    /// Verify using β-CROWN with branch-and-bound search.
    ///
    /// β-CROWN handles verification directly (returns VerificationResult)
    /// rather than just computing output bounds.
    fn verify_beta_crown(
        &self,
        network: &Network,
        input: &BoundedTensor,
        spec: &VerificationSpec,
    ) -> Result<VerificationResult> {
        debug!("β-CROWN verification");

        // Create β-CROWN verifier with default config
        // The threshold is derived from output specification
        let config = BetaCrownConfig {
            timeout: spec
                .timeout_ms
                .map(std::time::Duration::from_millis)
                .unwrap_or(std::time::Duration::from_secs(60)),
            ..BetaCrownConfig::default()
        };

        let beta_verifier = BetaCrownVerifier::new(config);

        // Derive threshold from output bounds
        // For now, use the first output's lower bound as threshold
        // (verifying output > threshold)
        let threshold = spec
            .output_bounds
            .first()
            .map(|b| b.lower)
            .unwrap_or(f32::NEG_INFINITY);

        let result = beta_verifier.verify(network, input, threshold)?;

        // Convert BabResult to VerificationResult
        let output_bounds: Vec<Bound> = match &result.result {
            crate::beta_crown::BabVerificationStatus::Verified => {
                // Verified: output is guaranteed > threshold
                vec![Bound::new(threshold, f32::INFINITY); spec.output_bounds.len()]
            }
            crate::beta_crown::BabVerificationStatus::Violated { .. } => {
                // Found concrete counterexample
                vec![Bound::new(f32::NEG_INFINITY, threshold); spec.output_bounds.len()]
            }
            crate::beta_crown::BabVerificationStatus::PotentialViolation => {
                // Found potential counterexample
                vec![Bound::new(f32::NEG_INFINITY, threshold); spec.output_bounds.len()]
            }
            crate::beta_crown::BabVerificationStatus::Unknown { .. } => {
                // Unknown result - return wide bounds
                vec![Bound::new(f32::NEG_INFINITY, f32::INFINITY); spec.output_bounds.len()]
            }
        };

        match result.result {
            crate::beta_crown::BabVerificationStatus::Verified => {
                Ok(VerificationResult::Verified {
                    output_bounds,
                    proof: None,
                })
            }
            crate::beta_crown::BabVerificationStatus::Violated {
                counterexample,
                output,
            } => {
                // Create informative details if bounds are available
                let details = gamma_core::InformativeCounterexample::new(
                    counterexample.clone(),
                    output.clone(),
                    Some(&output_bounds),
                );
                Ok(VerificationResult::Violated {
                    counterexample,
                    output,
                    details: Some(Box::new(details)),
                })
            }
            crate::beta_crown::BabVerificationStatus::PotentialViolation => {
                Ok(VerificationResult::Unknown {
                    bounds: output_bounds,
                    reason: "β-CROWN found potential violation region".to_string(),
                })
            }
            crate::beta_crown::BabVerificationStatus::Unknown { reason } => {
                Ok(VerificationResult::Unknown {
                    bounds: output_bounds,
                    reason: format!("β-CROWN: {}", reason),
                })
            }
        }
    }

    /// Verify a specification on a GraphNetwork (DAG-based network with binary ops support).
    ///
    /// This method supports models with binary operations like attention MatMul (Q@K^T)
    /// where both inputs are bounded tensors. Use this for transformer models.
    ///
    /// # Example
    /// ```ignore
    /// let graph = model.to_graph_network()?;
    /// let spec = VerificationSpec { ... };
    /// let result = verifier.verify_graph(&graph, &spec)?;
    /// ```
    pub fn verify_graph(
        &self,
        graph: &GraphNetwork,
        spec: &VerificationSpec,
    ) -> Result<VerificationResult> {
        self.verify_graph_with_engine(graph, spec, None)
    }

    pub fn verify_graph_with_engine(
        &self,
        graph: &GraphNetwork,
        spec: &VerificationSpec,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<VerificationResult> {
        info!(
            "Starting graph verification with {:?}, {} nodes",
            self.config.method,
            graph.num_nodes()
        );

        // Convert input bounds to bounded tensor
        let input_bounds = self.spec_to_tensor(&spec.input_bounds, spec.input_shape.as_deref())?;

        // Propagate through graph network
        let output_bounds = match self.config.method {
            PropagationMethod::Ibp => graph.propagate_ibp(&input_bounds)?,
            PropagationMethod::Crown => {
                // Try batched CROWN first (preserves N-D structure for transformers)
                match graph.propagate_crown_batched(&input_bounds) {
                    Ok(b) => b,
                    Err(e) => {
                        debug!("Batched CROWN failed ({}); trying flat CROWN", e);
                        match graph.propagate_crown_with_engine(&input_bounds, engine) {
                            Ok(b) => b,
                            Err(e2) => {
                                warn!("Graph CROWN failed ({}); falling back to IBP", e2);
                                graph.propagate_ibp(&input_bounds)?
                            }
                        }
                    }
                }
            }
            PropagationMethod::AlphaCrown => {
                // Use α-CROWN for GraphNetwork (supports DAG models with skip connections)
                debug!("Using α-CROWN for GraphNetwork");
                match graph.propagate_alpha_crown_with_engine(&input_bounds, engine) {
                    Ok(b) => b,
                    Err(e) => {
                        debug!("α-CROWN failed ({}); trying batched CROWN", e);
                        match graph.propagate_crown_batched(&input_bounds) {
                            Ok(b) => b,
                            Err(e2) => {
                                debug!("Batched CROWN failed ({}); trying flat CROWN", e2);
                                match graph.propagate_crown_with_engine(&input_bounds, engine) {
                                    Ok(b) => b,
                                    Err(e3) => {
                                        warn!("Graph CROWN failed ({}); falling back to IBP", e3);
                                        graph.propagate_ibp(&input_bounds)?
                                    }
                                }
                            }
                        }
                    }
                }
            }
            PropagationMethod::SdpCrown => {
                // Try to convert GraphNetwork to sequential Network for SDP-CROWN
                match graph.try_to_sequential_network() {
                    Some(network) => {
                        let (x_hat, rho) = Self::infer_l2_ball_from_box(&input_bounds)?;
                        network.propagate_sdp_crown(&input_bounds, &x_hat, rho)?
                    }
                    None => {
                        return Err(GammaError::UnsupportedOp(
                            "SDP-CROWN requires a sequential Linear/ReLU network; \
                             this GraphNetwork contains non-supported layers or has branches"
                                .to_string(),
                        ));
                    }
                }
            }
            PropagationMethod::BetaCrown => {
                // For verify_graph_with_engine with BetaCrown, compute CROWN bounds
                // for all outputs. This ensures VNN-LIB constraint checking has
                // access to bounds for all output indices.
                //
                // For complete BaB-based verification with VNN-LIB constraints,
                // use the dedicated `beta-crown` command instead.
                debug!("BetaCrown via verify: computing CROWN bounds for all outputs");
                match graph.propagate_alpha_crown_with_engine(&input_bounds, engine) {
                    Ok(b) => b,
                    Err(e) => {
                        debug!("α-CROWN failed ({}); trying batched CROWN", e);
                        match graph.propagate_crown_batched(&input_bounds) {
                            Ok(b) => b,
                            Err(e2) => {
                                debug!("Batched CROWN failed ({}); trying flat CROWN", e2);
                                match graph.propagate_crown_with_engine(&input_bounds, engine) {
                                    Ok(b) => b,
                                    Err(e3) => {
                                        warn!("Graph CROWN failed ({}); falling back to IBP", e3);
                                        graph.propagate_ibp(&input_bounds)?
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        // Sanitize NaN bounds: replace NaN with conservative infinities
        let output_bounds = Self::sanitize_output_bounds(output_bounds);

        // Check if output bounds satisfy spec
        self.check_spec(&output_bounds, &spec.output_bounds)
    }

    /// Interpret a uniform ℓ∞ box `x ∈ [l, u]` as an ℓ2 ball `||x - x_hat||_2 <= rho`.
    ///
    /// This is used for `PropagationMethod::SdpCrown`, since `VerificationSpec` currently stores
    /// input constraints as per-element bounds.
    ///
    /// Requirements:
    /// - All elements must have the same half-width `(u_i - l_i)/2` (within tolerance).
    fn infer_l2_ball_from_box(input_bounds: &BoundedTensor) -> Result<(ndarray::Array1<f32>, f32)> {
        let flat = input_bounds.flatten();
        let lower = flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| {
                GammaError::InvalidSpec("input lower must be 1-D after flatten".to_string())
            })?;
        let upper = flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| {
                GammaError::InvalidSpec("input upper must be 1-D after flatten".to_string())
            })?;

        let n = lower.len();
        let mut x_hat = ndarray::Array1::<f32>::zeros(n);
        let mut rho_opt: Option<f32> = None;

        for i in 0..n {
            let l = lower[i];
            let u = upper[i];
            if !(l.is_finite() && u.is_finite()) {
                return Err(GammaError::InvalidSpec(
                    "SDP-CROWN requires finite input bounds".to_string(),
                ));
            }
            if u < l {
                return Err(GammaError::InvalidSpec(format!(
                    "Invalid input bounds at index {i}: [{l}, {u}]"
                )));
            }
            x_hat[i] = 0.5 * (l + u);
            let rho_i = 0.5 * (u - l);
            let tol = 1e-5f32 * rho_i.abs().max(1.0);
            match rho_opt {
                None => rho_opt = Some(rho_i),
                Some(rho) => {
                    if (rho_i - rho).abs() > tol {
                        return Err(GammaError::InvalidSpec(
                            "SDP-CROWN currently requires a uniform epsilon box (same (u-l)/2 for all inputs)".to_string(),
                        ));
                    }
                }
            }
        }

        Ok((x_hat, rho_opt.unwrap_or(0.0)))
    }

    /// Sanitize output bounds: replace NaN values with conservative infinities.
    ///
    /// NaN bounds indicate numerical issues during propagation (e.g., inf + (-inf)).
    /// Sound verification requires converting these to conservative bounds:
    /// - NaN lower bound -> -inf (sound: any value >= -inf)
    /// - NaN upper bound -> +inf (sound: any value <= +inf)
    fn sanitize_output_bounds(mut bounds: BoundedTensor) -> BoundedTensor {
        use ndarray::Zip;
        Zip::from(&mut bounds.lower)
            .and(&mut bounds.upper)
            .for_each(|l, u| {
                if l.is_nan() {
                    *l = f32::NEG_INFINITY;
                }
                if u.is_nan() {
                    *u = f32::INFINITY;
                }
                // Ensure lower <= upper (sanitize inverted bounds)
                if *l > *u {
                    *l = f32::NEG_INFINITY;
                    *u = f32::INFINITY;
                }
            });
        bounds
    }

    fn check_spec(&self, output: &BoundedTensor, required: &[Bound]) -> Result<VerificationResult> {
        // Flatten the output bounds to 1D for comparison
        // Works for any dimensional output tensor
        let output_bounds: Vec<Bound> = output
            .lower
            .iter()
            .zip(output.upper.iter())
            .map(|(&l, &u)| Bound::new(l, u))
            .collect();

        // Check if computed bounds are within required bounds
        for (i, (computed, req)) in output_bounds.iter().zip(required.iter()).enumerate() {
            if computed.lower < req.lower || computed.upper > req.upper {
                return Ok(VerificationResult::Unknown {
                    bounds: output_bounds.clone(),
                    reason: format!(
                        "Output {} bounds [{}, {}] exceed required [{}, {}]",
                        i, computed.lower, computed.upper, req.lower, req.upper
                    ),
                });
            }
        }

        Ok(VerificationResult::Verified {
            output_bounds,
            proof: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::types::PropagationMethod;
    use ndarray::{Array1, Array2, IxDyn};

    // ==================== Verifier::new tests ====================

    #[test]
    fn test_verifier_new_default_config() {
        let config = PropagationConfig::default();
        let verifier = Verifier::new(config);
        // Default method is AlphaCrown
        assert!(matches!(
            verifier.config.method,
            PropagationMethod::AlphaCrown
        ));
    }

    #[test]
    fn test_verifier_new_crown_config() {
        let config = PropagationConfig {
            method: PropagationMethod::Crown,
            ..Default::default()
        };
        let verifier = Verifier::new(config);
        assert!(matches!(verifier.config.method, PropagationMethod::Crown));
    }

    #[test]
    fn test_verifier_new_alpha_crown_config() {
        let config = PropagationConfig {
            method: PropagationMethod::AlphaCrown,
            ..Default::default()
        };
        let verifier = Verifier::new(config);
        assert!(matches!(
            verifier.config.method,
            PropagationMethod::AlphaCrown
        ));
    }

    #[test]
    fn test_verifier_new_sdp_crown_config() {
        let config = PropagationConfig {
            method: PropagationMethod::SdpCrown,
            ..Default::default()
        };
        let verifier = Verifier::new(config);
        assert!(matches!(
            verifier.config.method,
            PropagationMethod::SdpCrown
        ));
    }

    #[test]
    fn test_verifier_new_beta_crown_config() {
        let config = PropagationConfig {
            method: PropagationMethod::BetaCrown,
            ..Default::default()
        };
        let verifier = Verifier::new(config);
        assert!(matches!(
            verifier.config.method,
            PropagationMethod::BetaCrown
        ));
    }

    // ==================== spec_to_tensor tests ====================

    #[test]
    fn test_spec_to_tensor_simple_1d() {
        let verifier = Verifier::new(PropagationConfig::default());
        let bounds = vec![Bound::new(-1.0, 1.0), Bound::new(0.0, 2.0)];

        let result = verifier.spec_to_tensor(&bounds, None).unwrap();

        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.lower[[0]], -1.0);
        assert_eq!(result.lower[[1]], 0.0);
        assert_eq!(result.upper[[0]], 1.0);
        assert_eq!(result.upper[[1]], 2.0);
    }

    #[test]
    fn test_spec_to_tensor_with_shape_2d() {
        let verifier = Verifier::new(PropagationConfig::default());
        let bounds = vec![
            Bound::new(-1.0, 1.0),
            Bound::new(-2.0, 2.0),
            Bound::new(-3.0, 3.0),
            Bound::new(-4.0, 4.0),
        ];

        let result = verifier.spec_to_tensor(&bounds, Some(&[2, 2])).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_spec_to_tensor_with_shape_3d() {
        let verifier = Verifier::new(PropagationConfig::default());
        let bounds: Vec<Bound> = (0..24)
            .map(|i| Bound::new(i as f32, i as f32 + 1.0))
            .collect();

        let result = verifier.spec_to_tensor(&bounds, Some(&[2, 3, 4])).unwrap();

        assert_eq!(result.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_spec_to_tensor_shape_mismatch_error() {
        let verifier = Verifier::new(PropagationConfig::default());
        let bounds = vec![Bound::new(-1.0, 1.0), Bound::new(0.0, 2.0)];

        // 3x3=9 elements but only 2 bounds provided
        let result = verifier.spec_to_tensor(&bounds, Some(&[3, 3]));

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, GammaError::InvalidSpec(_)));
        let msg = format!("{}", err);
        assert!(msg.contains("9 elements"));
        assert!(msg.contains("2"));
    }

    #[test]
    fn test_spec_to_tensor_empty_bounds() {
        let verifier = Verifier::new(PropagationConfig::default());
        let bounds: Vec<Bound> = vec![];

        let result = verifier.spec_to_tensor(&bounds, None).unwrap();

        assert_eq!(result.shape(), &[0]);
    }

    #[test]
    fn test_spec_to_tensor_single_element() {
        let verifier = Verifier::new(PropagationConfig::default());
        let bounds = vec![Bound::new(0.5, 1.5)];

        let result = verifier.spec_to_tensor(&bounds, None).unwrap();

        assert_eq!(result.shape(), &[1]);
        assert_eq!(result.lower[[0]], 0.5);
        assert_eq!(result.upper[[0]], 1.5);
    }

    #[test]
    fn test_spec_to_tensor_large_values() {
        // Test with large but finite values instead of infinities
        // (BoundedTensor::new rejects infinities)
        let verifier = Verifier::new(PropagationConfig::default());
        let large = 1e30f32;
        let bounds = vec![
            Bound::new(-large, large),
            Bound::new(0.0, large),
            Bound::new(-large, 0.0),
        ];

        let result = verifier.spec_to_tensor(&bounds, None).unwrap();

        assert_eq!(result.lower[[0]], -large);
        assert_eq!(result.upper[[0]], large);
        assert_eq!(result.lower[[1]], 0.0);
        assert_eq!(result.upper[[1]], large);
        assert_eq!(result.lower[[2]], -large);
        assert_eq!(result.upper[[2]], 0.0);
    }

    // ==================== sanitize_output_bounds tests ====================

    #[test]
    fn test_sanitize_output_bounds_normal_values() {
        let tensor = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        )
        .unwrap();

        let result = Verifier::sanitize_output_bounds(tensor);

        assert_eq!(result.lower[[0]], -1.0);
        assert_eq!(result.lower[[1]], 0.0);
        assert_eq!(result.lower[[2]], 1.0);
        assert_eq!(result.upper[[0]], 1.0);
        assert_eq!(result.upper[[1]], 2.0);
        assert_eq!(result.upper[[2]], 3.0);
    }

    #[test]
    fn test_sanitize_output_bounds_nan_lower() {
        // Use new_unchecked to allow NaN values for testing sanitization
        let tensor = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, 0.0, 1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        )
        .unwrap();

        let result = Verifier::sanitize_output_bounds(tensor);

        assert_eq!(result.lower[[0]], f32::NEG_INFINITY);
        assert_eq!(result.lower[[1]], 0.0);
        assert_eq!(result.lower[[2]], 1.0);
        assert_eq!(result.upper[[0]], 1.0);
    }

    #[test]
    fn test_sanitize_output_bounds_nan_upper() {
        let tensor = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, 2.0, 3.0]).unwrap(),
        )
        .unwrap();

        let result = Verifier::sanitize_output_bounds(tensor);

        assert_eq!(result.lower[[0]], -1.0);
        assert_eq!(result.upper[[0]], f32::INFINITY);
        assert_eq!(result.upper[[1]], 2.0);
    }

    #[test]
    fn test_sanitize_output_bounds_both_nan() {
        let tensor = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 1.0]).unwrap(),
        )
        .unwrap();

        let result = Verifier::sanitize_output_bounds(tensor);

        assert_eq!(result.lower[[0]], f32::NEG_INFINITY);
        assert_eq!(result.upper[[0]], f32::INFINITY);
        assert_eq!(result.lower[[1]], 0.0);
        assert_eq!(result.upper[[1]], 1.0);
    }

    #[test]
    fn test_sanitize_output_bounds_inverted_bounds() {
        // lower > upper should be sanitized
        let tensor = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![5.0, 0.0]).unwrap(), // lower > upper
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 1.0]).unwrap(),
        )
        .unwrap();

        let result = Verifier::sanitize_output_bounds(tensor);

        // Inverted bounds should become (-inf, +inf)
        assert_eq!(result.lower[[0]], f32::NEG_INFINITY);
        assert_eq!(result.upper[[0]], f32::INFINITY);
        // Normal bounds unchanged
        assert_eq!(result.lower[[1]], 0.0);
        assert_eq!(result.upper[[1]], 1.0);
    }

    #[test]
    fn test_sanitize_output_bounds_all_nan() {
        let tensor = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, f32::NAN, f32::NAN]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, f32::NAN, f32::NAN]).unwrap(),
        )
        .unwrap();

        let result = Verifier::sanitize_output_bounds(tensor);

        for i in 0..3 {
            assert_eq!(result.lower[[i]], f32::NEG_INFINITY);
            assert_eq!(result.upper[[i]], f32::INFINITY);
        }
    }

    #[test]
    fn test_sanitize_output_bounds_multidimensional() {
        let tensor = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![f32::NAN, -1.0, 0.0, 2.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, f32::NAN, 1.0, 1.0]).unwrap(), // [1,1] inverted
        )
        .unwrap();

        let result = Verifier::sanitize_output_bounds(tensor);

        // [0,0]: NaN lower
        assert_eq!(result.lower[[0, 0]], f32::NEG_INFINITY);
        assert_eq!(result.upper[[0, 0]], 1.0);
        // [0,1]: NaN upper
        assert_eq!(result.lower[[0, 1]], -1.0);
        assert_eq!(result.upper[[0, 1]], f32::INFINITY);
        // [1,0]: normal
        assert_eq!(result.lower[[1, 0]], 0.0);
        assert_eq!(result.upper[[1, 0]], 1.0);
        // [1,1]: inverted (2.0 > 1.0)
        assert_eq!(result.lower[[1, 1]], f32::NEG_INFINITY);
        assert_eq!(result.upper[[1, 1]], f32::INFINITY);
    }

    // ==================== check_spec tests ====================

    #[test]
    fn test_check_spec_verified() {
        let verifier = Verifier::new(PropagationConfig::default());
        let output = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.8, 1.5]).unwrap(),
        )
        .unwrap();
        let required = vec![Bound::new(0.0, 1.0), Bound::new(0.5, 2.0)];

        let result = verifier.check_spec(&output, &required).unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
        if let VerificationResult::Verified { output_bounds, .. } = result {
            assert_eq!(output_bounds.len(), 2);
            assert_eq!(output_bounds[0].lower, 0.5);
            assert_eq!(output_bounds[0].upper, 0.8);
        }
    }

    #[test]
    fn test_check_spec_unknown_lower_too_low() {
        let verifier = Verifier::new(PropagationConfig::default());
        let output = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-0.5, 1.0]).unwrap(), // -0.5 < required 0.0
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.8, 1.5]).unwrap(),
        )
        .unwrap();
        let required = vec![Bound::new(0.0, 1.0), Bound::new(0.5, 2.0)];

        let result = verifier.check_spec(&output, &required).unwrap();

        assert!(matches!(result, VerificationResult::Unknown { .. }));
        if let VerificationResult::Unknown { reason, .. } = result {
            assert!(reason.contains("Output 0"));
            assert!(reason.contains("-0.5"));
        }
    }

    #[test]
    fn test_check_spec_unknown_upper_too_high() {
        let verifier = Verifier::new(PropagationConfig::default());
        let output = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.5, 1.5]).unwrap(), // 1.5 > required 1.0
        )
        .unwrap();
        let required = vec![Bound::new(0.0, 1.0), Bound::new(0.5, 2.0)];

        let result = verifier.check_spec(&output, &required).unwrap();

        assert!(matches!(result, VerificationResult::Unknown { .. }));
        if let VerificationResult::Unknown { reason, .. } = result {
            assert!(reason.contains("Output 0"));
            assert!(reason.contains("1.5"));
        }
    }

    #[test]
    fn test_check_spec_second_output_violates() {
        let verifier = Verifier::new(PropagationConfig::default());
        let output = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.0]).unwrap(), // second: 0.0 < required 0.5
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.8, 1.5]).unwrap(),
        )
        .unwrap();
        let required = vec![Bound::new(0.0, 1.0), Bound::new(0.5, 2.0)];

        let result = verifier.check_spec(&output, &required).unwrap();

        assert!(matches!(result, VerificationResult::Unknown { .. }));
        if let VerificationResult::Unknown { reason, .. } = result {
            assert!(reason.contains("Output 1"));
        }
    }

    #[test]
    fn test_check_spec_exactly_at_bounds() {
        let verifier = Verifier::new(PropagationConfig::default());
        let output = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.5]).unwrap(), // exactly at lower bounds
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(), // exactly at upper bounds
        )
        .unwrap();
        let required = vec![Bound::new(0.0, 1.0), Bound::new(0.5, 2.0)];

        let result = verifier.check_spec(&output, &required).unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
    }

    #[test]
    fn test_check_spec_with_infinities() {
        let verifier = Verifier::new(PropagationConfig::default());
        let output = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.5]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.5]).unwrap(),
        )
        .unwrap();
        let required = vec![Bound::new(f32::NEG_INFINITY, f32::INFINITY)];

        let result = verifier.check_spec(&output, &required).unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
    }

    #[test]
    fn test_check_spec_empty() {
        let verifier = Verifier::new(PropagationConfig::default());
        let output = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
        )
        .unwrap();
        let required: Vec<Bound> = vec![];

        let result = verifier.check_spec(&output, &required).unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
    }

    #[test]
    fn test_check_spec_multidimensional_output() {
        let verifier = Verifier::new(PropagationConfig::default());
        // 2x2 output, will be flattened to 4 elements
        let output = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5, 0.6, 0.7, 0.8]).unwrap(),
        )
        .unwrap();
        let required = vec![
            Bound::new(0.0, 1.0),
            Bound::new(0.0, 1.0),
            Bound::new(0.0, 1.0),
            Bound::new(0.0, 1.0),
        ];

        let result = verifier.check_spec(&output, &required).unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
    }

    // ==================== infer_l2_ball_from_box tests ====================

    #[test]
    fn test_infer_l2_ball_uniform_box() {
        let tensor = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, -1.0, -1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 1.0, 1.0]).unwrap(),
        )
        .unwrap();

        let (x_hat, rho) = Verifier::infer_l2_ball_from_box(&tensor).unwrap();

        assert_eq!(x_hat.len(), 3);
        assert!((x_hat[0] - 0.0).abs() < 1e-6);
        assert!((x_hat[1] - 0.0).abs() < 1e-6);
        assert!((x_hat[2] - 0.0).abs() < 1e-6);
        assert!((rho - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_infer_l2_ball_uniform_box_shifted() {
        let tensor = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 3.0]).unwrap(),
        )
        .unwrap();

        let (x_hat, rho) = Verifier::infer_l2_ball_from_box(&tensor).unwrap();

        assert!((x_hat[0] - 2.0).abs() < 1e-6);
        assert!((x_hat[1] - 2.0).abs() < 1e-6);
        assert!((rho - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_infer_l2_ball_non_uniform_error() {
        let tensor = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -2.0]).unwrap(), // different widths
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        )
        .unwrap();

        let result = Verifier::infer_l2_ball_from_box(&tensor);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, GammaError::InvalidSpec(_)));
        let msg = format!("{}", err);
        assert!(msg.contains("uniform epsilon box"));
    }

    #[test]
    fn test_infer_l2_ball_infinite_bounds_error() {
        // Use new_unchecked to allow infinity for testing error handling
        let tensor = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, -1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
        )
        .unwrap();

        let result = Verifier::infer_l2_ball_from_box(&tensor);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, GammaError::InvalidSpec(_)));
        let msg = format!("{}", err);
        assert!(msg.contains("finite"));
    }

    #[test]
    fn test_infer_l2_ball_inverted_bounds_error() {
        // Use new_unchecked to allow inverted bounds for testing error handling
        let tensor = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, -1.0]).unwrap(), // first element inverted
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).unwrap(),
        )
        .unwrap();

        let result = Verifier::infer_l2_ball_from_box(&tensor);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, GammaError::InvalidSpec(_)));
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid input bounds"));
    }

    #[test]
    fn test_infer_l2_ball_single_element() {
        let tensor = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
        )
        .unwrap();

        let (x_hat, rho) = Verifier::infer_l2_ball_from_box(&tensor).unwrap();

        assert_eq!(x_hat.len(), 1);
        assert!((x_hat[0] - 1.0).abs() < 1e-6);
        assert!((rho - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_infer_l2_ball_empty() {
        let tensor = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
        )
        .unwrap();

        let (x_hat, rho) = Verifier::infer_l2_ball_from_box(&tensor).unwrap();

        assert_eq!(x_hat.len(), 0);
        assert_eq!(rho, 0.0);
    }

    #[test]
    fn test_infer_l2_ball_point_bounds() {
        // Point bounds: l == u
        let tensor = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        )
        .unwrap();

        let (x_hat, rho) = Verifier::infer_l2_ball_from_box(&tensor).unwrap();

        assert!((x_hat[0] - 1.0).abs() < 1e-6);
        assert!((x_hat[1] - 2.0).abs() < 1e-6);
        assert!((x_hat[2] - 3.0).abs() < 1e-6);
        assert!((rho - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_infer_l2_ball_multidimensional_input() {
        // 2x2 input, should be flattened
        let tensor = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-0.5, -0.5, -0.5, -0.5]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5, 0.5, 0.5, 0.5]).unwrap(),
        )
        .unwrap();

        let (x_hat, rho) = Verifier::infer_l2_ball_from_box(&tensor).unwrap();

        assert_eq!(x_hat.len(), 4);
        for i in 0..4 {
            assert!((x_hat[i] - 0.0).abs() < 1e-6);
        }
        assert!((rho - 0.5).abs() < 1e-6);
    }

    // ==================== verify integration tests (simple cases) ====================

    #[test]
    fn test_verify_simple_linear_ibp() {
        // Simple 2x2 identity linear layer
        let weight = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let linear = LinearLayer::new(weight, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));

        let verifier = Verifier::new(PropagationConfig {
            method: PropagationMethod::Ibp,
            ..Default::default()
        });

        let spec = VerificationSpec {
            input_bounds: vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
            output_bounds: vec![Bound::new(-2.0, 2.0), Bound::new(-2.0, 2.0)],
            timeout_ms: None,
            input_shape: None,
        };

        let result = verifier.verify(&network, &spec).unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
    }

    #[test]
    fn test_verify_simple_linear_crown() {
        // Simple 2x2 identity linear layer
        let weight = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let linear = LinearLayer::new(weight, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));

        let verifier = Verifier::new(PropagationConfig {
            method: PropagationMethod::Crown,
            ..Default::default()
        });

        let spec = VerificationSpec {
            input_bounds: vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
            output_bounds: vec![Bound::new(-2.0, 2.0), Bound::new(-2.0, 2.0)],
            timeout_ms: None,
            input_shape: None,
        };

        let result = verifier.verify(&network, &spec).unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
    }

    #[test]
    fn test_verify_tight_bounds_unknown() {
        // Simple 2x2 linear layer with scaling
        let weight = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).unwrap(); // 2x scaling
        let linear = LinearLayer::new(weight, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));

        let verifier = Verifier::new(PropagationConfig {
            method: PropagationMethod::Ibp,
            ..Default::default()
        });

        // Input: [-1, 1], output should be [-2, 2] but we require [-1, 1]
        let spec = VerificationSpec {
            input_bounds: vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
            output_bounds: vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)], // Too tight
            timeout_ms: None,
            input_shape: None,
        };

        let result = verifier.verify(&network, &spec).unwrap();

        assert!(matches!(result, VerificationResult::Unknown { .. }));
    }

    #[test]
    fn test_verify_with_relu() {
        // Linear + ReLU network
        let weight = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let linear = LinearLayer::new(weight, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));
        network.add_layer(Layer::ReLU(ReLULayer));

        let verifier = Verifier::new(PropagationConfig {
            method: PropagationMethod::Ibp,
            ..Default::default()
        });

        let spec = VerificationSpec {
            input_bounds: vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
            output_bounds: vec![Bound::new(0.0, 2.0), Bound::new(0.0, 2.0)], // ReLU clips negative
            timeout_ms: None,
            input_shape: None,
        };

        let result = verifier.verify(&network, &spec).unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
    }

    #[test]
    fn test_verify_with_bias() {
        // Linear layer with bias
        let weight = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let bias = Array1::from_vec(vec![1.0, -1.0]);
        let linear = LinearLayer::new(weight, Some(bias)).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));

        let verifier = Verifier::new(PropagationConfig {
            method: PropagationMethod::Ibp,
            ..Default::default()
        });

        // Input [-1, 1] + bias [1, -1] = output [0, 2] and [-2, 0]
        let spec = VerificationSpec {
            input_bounds: vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
            output_bounds: vec![Bound::new(-1.0, 3.0), Bound::new(-3.0, 1.0)],
            timeout_ms: None,
            input_shape: None,
        };

        let result = verifier.verify(&network, &spec).unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
    }

    #[test]
    fn test_verify_with_larger_network() {
        // Test with a larger 4x4 identity network
        let weight = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap();
        let linear = LinearLayer::new(weight, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));

        let verifier = Verifier::new(PropagationConfig {
            method: PropagationMethod::Ibp,
            ..Default::default()
        });

        let spec = VerificationSpec {
            input_bounds: vec![
                Bound::new(-1.0, 1.0),
                Bound::new(-1.0, 1.0),
                Bound::new(-1.0, 1.0),
                Bound::new(-1.0, 1.0),
            ],
            output_bounds: vec![
                Bound::new(-2.0, 2.0),
                Bound::new(-2.0, 2.0),
                Bound::new(-2.0, 2.0),
                Bound::new(-2.0, 2.0),
            ],
            timeout_ms: None,
            input_shape: None, // Use default 1D shape for linear layers
        };

        let result = verifier.verify(&network, &spec).unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
    }

    // ==================== Additional edge case tests ====================

    #[test]
    fn test_sanitize_preserves_infinity_bounds() {
        // Valid infinite bounds should be preserved, not sanitized
        // Use new_unchecked since BoundedTensor::new rejects infinities
        let tensor = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, 0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 1.0]).unwrap(),
        )
        .unwrap();

        let result = Verifier::sanitize_output_bounds(tensor);

        assert_eq!(result.lower[[0]], f32::NEG_INFINITY);
        assert_eq!(result.upper[[0]], f32::INFINITY);
        assert_eq!(result.lower[[1]], 0.0);
        assert_eq!(result.upper[[1]], 1.0);
    }

    #[test]
    fn test_check_spec_partial_match() {
        // First output matches, second doesn't
        let verifier = Verifier::new(PropagationConfig::default());
        let output = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.5]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.8, 2.5]).unwrap(), // second upper too high
        )
        .unwrap();
        let required = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 2.0)];

        let result = verifier.check_spec(&output, &required).unwrap();

        // Should detect the violation in second output
        assert!(matches!(result, VerificationResult::Unknown { .. }));
        if let VerificationResult::Unknown { reason, .. } = result {
            assert!(reason.contains("Output 1"));
        }
    }

    #[test]
    fn test_infer_l2_ball_tolerance_check() {
        // Small difference within tolerance should pass
        let epsilon = 1e-6;
        let tensor = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0 - epsilon]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0 + epsilon]).unwrap(),
        )
        .unwrap();

        // Should succeed because the difference is within tolerance
        let result = Verifier::infer_l2_ball_from_box(&tensor);
        assert!(result.is_ok());
    }
}
