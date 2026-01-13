//! Parallel position verification for sequence models.
//!
//! This module provides parallel verification across sequence positions,
//! enabling near-linear speedup with cores for position-independent verification.
//!
//! # When to Use
//!
//! Parallel position verification is beneficial when:
//! - Verifying position-independent properties (bounds at each output position)
//! - Sequence length is >= number of CPU cores
//! - Memory allows independent position verification
//!
//! # Example
//! ```ignore
//! use gamma_propagate::parallel::{ParallelVerifier, ParallelConfig};
//!
//! let config = ParallelConfig::default();
//! let verifier = ParallelVerifier::new(config);
//!
//! // Verify each position in parallel
//! let results = verifier.verify_positions_parallel(
//!     &graph,
//!     &input,  // [batch, seq_len, hidden]
//!     1,       // seq axis
//! )?;
//!
//! // Or verify batch dimension in parallel
//! let batch_results = verifier.verify_batch_parallel(
//!     &graph,
//!     &input,  // [batch, seq_len, hidden]
//!     0,       // batch axis
//! )?;
//! ```

use crate::network::GraphNetwork;
use crate::types::PropagationMethod;
use gamma_core::{GammaError, Result};
use gamma_tensor::BoundedTensor;
use ndarray::Array1;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{debug, info, trace};

/// Configuration for parallel verification.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Propagation method to use (IBP, CROWN, etc.)
    pub method: PropagationMethod,

    /// Minimum number of positions before enabling parallelism.
    /// Below this threshold, serial execution is used to avoid overhead.
    pub min_positions_for_parallel: usize,

    /// Maximum number of threads to use.
    /// None means use rayon's default (typically number of cores).
    pub max_threads: Option<usize>,

    /// Whether to report progress during verification.
    pub report_progress: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            method: PropagationMethod::Ibp,
            min_positions_for_parallel: 4,
            max_threads: None,
            report_progress: false,
        }
    }
}

/// Result of parallel position verification.
#[derive(Debug)]
pub struct ParallelVerificationResult {
    /// Output bounds for each position, stacked back together.
    pub output_bounds: BoundedTensor,

    /// Number of positions verified.
    pub num_positions: usize,

    /// Number of positions verified in parallel (vs serial).
    pub parallel_positions: usize,

    /// Total verification time in milliseconds.
    pub total_time_ms: u64,

    /// Average time per position in milliseconds.
    pub avg_position_time_ms: f64,
}

/// Parallel verifier for sequence models.
///
/// Distributes verification across sequence positions using rayon,
/// achieving near-linear speedup with available cores.
pub struct ParallelVerifier {
    config: ParallelConfig,
}

impl ParallelVerifier {
    /// Create a new parallel verifier with the given configuration.
    pub fn new(config: ParallelConfig) -> Self {
        Self { config }
    }

    /// Verify each position along the specified axis in parallel.
    ///
    /// This is ideal for verifying position-independent properties on
    /// sequence models like transformers.
    ///
    /// # Arguments
    /// * `graph` - The network graph to verify
    /// * `input` - Input bounded tensor with shape [..., axis_size, ...]
    /// * `axis` - The axis to parallelize over (typically seq_len axis)
    ///
    /// # Returns
    /// Combined output bounds with same shape as serial verification.
    pub fn verify_positions_parallel(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        axis: usize,
    ) -> Result<ParallelVerificationResult> {
        let start_time = std::time::Instant::now();
        let shape = input.shape();

        if axis >= shape.len() {
            return Err(GammaError::InvalidSpec(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                shape.len()
            )));
        }

        let num_positions = shape[axis];
        info!(
            "Parallel verification: {} positions along axis {}, method {:?}",
            num_positions, axis, self.config.method
        );

        // Decide whether to use parallel or serial
        let use_parallel = num_positions >= self.config.min_positions_for_parallel;
        let parallel_positions = if use_parallel { num_positions } else { 0 };

        let output_positions = if use_parallel {
            self.verify_parallel_impl(graph, input, axis, num_positions)?
        } else {
            debug!(
                "Using serial verification ({} positions < threshold {})",
                num_positions, self.config.min_positions_for_parallel
            );
            self.verify_serial_impl(graph, input, axis, num_positions)?
        };

        // Stack outputs back together
        let output_bounds = BoundedTensor::stack(&output_positions, axis)?;

        let total_time_ms = start_time.elapsed().as_millis() as u64;
        let avg_position_time_ms = total_time_ms as f64 / num_positions as f64;

        info!(
            "Parallel verification complete: {}ms total, {:.2}ms/position",
            total_time_ms, avg_position_time_ms
        );

        Ok(ParallelVerificationResult {
            output_bounds,
            num_positions,
            parallel_positions,
            total_time_ms,
            avg_position_time_ms,
        })
    }

    /// Verify each sample in the batch dimension in parallel.
    ///
    /// Useful when batch size > 1 and each sample is independent.
    pub fn verify_batch_parallel(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        batch_axis: usize,
    ) -> Result<ParallelVerificationResult> {
        // Same implementation, just typically called with axis=0
        self.verify_positions_parallel(graph, input, batch_axis)
    }

    /// Internal parallel implementation using rayon.
    fn verify_parallel_impl(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        axis: usize,
        num_positions: usize,
    ) -> Result<Vec<BoundedTensor>> {
        let progress = AtomicUsize::new(0);

        // Build the thread pool with optional thread limit
        let pool = if let Some(max_threads) = self.config.max_threads {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(max_threads)
                    .build()
                    .map_err(|e| {
                        GammaError::InvalidSpec(format!("Failed to create thread pool: {}", e))
                    })?,
            )
        } else {
            None
        };

        let report_progress = self.config.report_progress;
        let method = self.config.method;

        // Closure to verify a single position
        let verify_position = |pos: usize| -> Result<BoundedTensor> {
            trace!("Verifying position {}/{}", pos + 1, num_positions);

            // Extract single position
            let pos_input = input.slice_axis(axis, pos)?;

            // Propagate through network
            let pos_output = match method {
                PropagationMethod::Ibp => graph.propagate_ibp(&pos_input)?,
                PropagationMethod::Crown => {
                    // Try batched CROWN first, fall back to flat CROWN, then IBP
                    graph
                        .propagate_crown_batched(&pos_input)
                        .or_else(|_| graph.propagate_crown(&pos_input))
                        .or_else(|_| graph.propagate_ibp(&pos_input))?
                }
                PropagationMethod::AlphaCrown => {
                    // Graph doesn't have alpha-CROWN, use batched CROWN
                    graph
                        .propagate_crown_batched(&pos_input)
                        .or_else(|_| graph.propagate_crown(&pos_input))
                        .or_else(|_| graph.propagate_ibp(&pos_input))?
                }
                PropagationMethod::SdpCrown => {
                    // SDP-CROWN requires: (1) sequential network, (2) uniform epsilon box
                    if let Some(network) = graph.try_to_sequential_network() {
                        if let Some((x_hat, rho)) = infer_l2_ball_from_box(&pos_input) {
                            // Use SDP-CROWN for tighter bounds
                            match network.propagate_sdp_crown(&pos_input, &x_hat, rho) {
                                Ok(output) => output,
                                Err(e) => {
                                    trace!("SDP-CROWN failed, falling back to CROWN: {}", e);
                                    graph
                                        .propagate_crown_batched(&pos_input)
                                        .or_else(|_| graph.propagate_crown(&pos_input))
                                        .or_else(|_| graph.propagate_ibp(&pos_input))?
                                }
                            }
                        } else {
                            // Non-uniform epsilon: fall back to CROWN
                            trace!("Non-uniform epsilon, using CROWN instead of SDP-CROWN");
                            graph
                                .propagate_crown_batched(&pos_input)
                                .or_else(|_| graph.propagate_crown(&pos_input))
                                .or_else(|_| graph.propagate_ibp(&pos_input))?
                        }
                    } else {
                        // Non-sequential graph (has GELU/SiLU/etc): fall back to CROWN
                        graph
                            .propagate_crown_batched(&pos_input)
                            .or_else(|_| graph.propagate_crown(&pos_input))
                            .or_else(|_| graph.propagate_ibp(&pos_input))?
                    }
                }
                PropagationMethod::BetaCrown => {
                    // Beta-CROWN falls back to CROWN for parallel verification
                    graph
                        .propagate_crown_batched(&pos_input)
                        .or_else(|_| graph.propagate_crown(&pos_input))
                        .or_else(|_| graph.propagate_ibp(&pos_input))?
                }
            };

            if report_progress {
                let completed = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if completed % 10 == 0 || completed == num_positions {
                    debug!("Progress: {}/{} positions", completed, num_positions);
                }
            }

            Ok(pos_output)
        };

        // Execute in parallel
        let results: Result<Vec<BoundedTensor>> = if let Some(pool) = pool {
            pool.install(|| {
                (0..num_positions)
                    .into_par_iter()
                    .map(verify_position)
                    .collect()
            })
        } else {
            (0..num_positions)
                .into_par_iter()
                .map(verify_position)
                .collect()
        };

        results
    }

    /// Internal serial implementation for small position counts.
    fn verify_serial_impl(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        axis: usize,
        num_positions: usize,
    ) -> Result<Vec<BoundedTensor>> {
        let mut results = Vec::with_capacity(num_positions);

        for pos in 0..num_positions {
            trace!("Verifying position {}/{} (serial)", pos + 1, num_positions);

            let pos_input = input.slice_axis(axis, pos)?;

            let pos_output = match self.config.method {
                PropagationMethod::Ibp => graph.propagate_ibp(&pos_input)?,
                PropagationMethod::Crown => graph
                    .propagate_crown_batched(&pos_input)
                    .or_else(|_| graph.propagate_crown(&pos_input))
                    .or_else(|_| graph.propagate_ibp(&pos_input))?,
                PropagationMethod::AlphaCrown => graph
                    .propagate_crown_batched(&pos_input)
                    .or_else(|_| graph.propagate_crown(&pos_input))
                    .or_else(|_| graph.propagate_ibp(&pos_input))?,
                PropagationMethod::SdpCrown => {
                    // SDP-CROWN requires: (1) sequential network, (2) uniform epsilon box
                    if let Some(network) = graph.try_to_sequential_network() {
                        if let Some((x_hat, rho)) = infer_l2_ball_from_box(&pos_input) {
                            match network.propagate_sdp_crown(&pos_input, &x_hat, rho) {
                                Ok(output) => output,
                                Err(_) => graph
                                    .propagate_crown_batched(&pos_input)
                                    .or_else(|_| graph.propagate_crown(&pos_input))
                                    .or_else(|_| graph.propagate_ibp(&pos_input))?,
                            }
                        } else {
                            graph
                                .propagate_crown_batched(&pos_input)
                                .or_else(|_| graph.propagate_crown(&pos_input))
                                .or_else(|_| graph.propagate_ibp(&pos_input))?
                        }
                    } else {
                        graph
                            .propagate_crown_batched(&pos_input)
                            .or_else(|_| graph.propagate_crown(&pos_input))
                            .or_else(|_| graph.propagate_ibp(&pos_input))?
                    }
                }
                PropagationMethod::BetaCrown => {
                    // Beta-CROWN falls back to CROWN for position-wise verification
                    graph
                        .propagate_crown_batched(&pos_input)
                        .or_else(|_| graph.propagate_crown(&pos_input))
                        .or_else(|_| graph.propagate_ibp(&pos_input))?
                }
            };

            results.push(pos_output);
        }

        Ok(results)
    }
}

/// Convenience function for parallel position verification with default config.
///
/// # Example
/// ```ignore
/// let output = verify_parallel(&graph, &input, 1)?; // Parallelize over seq axis
/// ```
pub fn verify_parallel(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    axis: usize,
) -> Result<BoundedTensor> {
    let config = ParallelConfig::default();
    let verifier = ParallelVerifier::new(config);
    Ok(verifier
        .verify_positions_parallel(graph, input, axis)?
        .output_bounds)
}

/// Convenience function for parallel position verification with custom method.
pub fn verify_parallel_with_method(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    axis: usize,
    method: PropagationMethod,
) -> Result<BoundedTensor> {
    let config = ParallelConfig {
        method,
        ..Default::default()
    };
    let verifier = ParallelVerifier::new(config);
    Ok(verifier
        .verify_positions_parallel(graph, input, axis)?
        .output_bounds)
}

/// Infer ℓ2 ball parameters from a BoundedTensor (uniform epsilon box).
///
/// For SDP-CROWN to work, the input must be a uniform epsilon box:
/// - All dimensions have the same half-width (epsilon)
/// - The center is (lower + upper) / 2
/// - The radius rho equals the common epsilon value
///
/// Returns (x_hat, rho) where x_hat is the center vector and rho is the radius.
/// Returns None if the box is non-uniform (different epsilon per dimension).
fn infer_l2_ball_from_box(input: &BoundedTensor) -> Option<(Array1<f32>, f32)> {
    let flat = input.flatten();
    let lower = flat
        .lower
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .ok()?;
    let upper = flat
        .upper
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .ok()?;

    let n = lower.len();
    if n == 0 {
        return Some((Array1::zeros(0), 0.0));
    }

    let mut x_hat = Array1::<f32>::zeros(n);
    let mut rho_opt: Option<f32> = None;

    for i in 0..n {
        let l = lower[i];
        let u = upper[i];

        // Require finite bounds
        if !(l.is_finite() && u.is_finite()) || u < l {
            return None;
        }

        x_hat[i] = 0.5 * (l + u);
        let rho_i = 0.5 * (u - l);
        let tol = 1e-5f32 * rho_i.abs().max(1.0);

        match rho_opt {
            None => rho_opt = Some(rho_i),
            Some(rho) => {
                // Non-uniform epsilon: cannot use SDP-CROWN
                if (rho_i - rho).abs() > tol {
                    return None;
                }
            }
        }
    }

    Some((x_hat, rho_opt.unwrap_or(0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::network::{GraphNetwork, GraphNode};
    use ndarray::{arr2, arr3, ArrayD, IxDyn};

    fn create_simple_graph() -> GraphNetwork {
        // Simple 2-layer MLP: Linear -> ReLU -> Linear
        let mut graph = GraphNetwork::new();

        // Input layer (3 -> 4)
        let weight1 = arr2(&[
            [1.0_f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]);
        let linear1 = LinearLayer::new(weight1, None).unwrap();
        graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

        // ReLU
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ));

        // Output layer (4 -> 2)
        let weight2 = arr2(&[[1.0_f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]);
        let linear2 = LinearLayer::new(weight2, None).unwrap();
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(linear2),
            vec!["relu".to_string()],
        ));

        graph.set_output("linear2");
        graph
    }

    #[test]
    fn test_parallel_verifier_basic() {
        let graph = create_simple_graph();

        // Input: [batch=1, seq=8, hidden=3]
        let lower = arr3(&[[
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4],
            [0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6],
            [0.7, 0.7, 0.7],
        ]])
        .into_dyn();
        let upper = lower.mapv(|x| x + 0.1);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let config = ParallelConfig {
            method: PropagationMethod::Ibp,
            min_positions_for_parallel: 2, // Use parallel for 8 positions
            ..Default::default()
        };
        let verifier = ParallelVerifier::new(config);

        let result = verifier
            .verify_positions_parallel(&graph, &input, 1)
            .unwrap();

        // Check output shape: [batch=1, seq=8, output=2]
        assert_eq!(result.output_bounds.shape(), &[1, 8, 2]);
        assert_eq!(result.num_positions, 8);
        assert_eq!(result.parallel_positions, 8);
    }

    #[test]
    fn test_parallel_vs_serial_equivalence() {
        let graph = create_simple_graph();

        // Input: [batch=1, seq=4, hidden=3]
        let lower = arr3(&[[
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
        ]])
        .into_dyn();
        let upper = lower.mapv(|x| x + 0.1);
        let input = BoundedTensor::new(lower, upper).unwrap();

        // Serial verification (high threshold forces serial)
        let serial_config = ParallelConfig {
            method: PropagationMethod::Ibp,
            min_positions_for_parallel: 100,
            ..Default::default()
        };
        let serial_verifier = ParallelVerifier::new(serial_config);
        let serial_result = serial_verifier
            .verify_positions_parallel(&graph, &input, 1)
            .unwrap();

        // Parallel verification
        let parallel_config = ParallelConfig {
            method: PropagationMethod::Ibp,
            min_positions_for_parallel: 1,
            ..Default::default()
        };
        let parallel_verifier = ParallelVerifier::new(parallel_config);
        let parallel_result = parallel_verifier
            .verify_positions_parallel(&graph, &input, 1)
            .unwrap();

        // Results should be identical
        assert_eq!(
            serial_result.output_bounds.shape(),
            parallel_result.output_bounds.shape()
        );

        let serial_bounds = &serial_result.output_bounds;
        let parallel_bounds = &parallel_result.output_bounds;

        for (s, p) in serial_bounds.lower.iter().zip(parallel_bounds.lower.iter()) {
            assert!((s - p).abs() < 1e-6, "Lower bounds differ: {} vs {}", s, p);
        }
        for (s, p) in serial_bounds.upper.iter().zip(parallel_bounds.upper.iter()) {
            assert!((s - p).abs() < 1e-6, "Upper bounds differ: {} vs {}", s, p);
        }
    }

    #[test]
    fn test_convenience_function() {
        let graph = create_simple_graph();

        let lower = arr3(&[[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]]).into_dyn();
        let upper = lower.mapv(|x| x + 0.1);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let output = verify_parallel(&graph, &input, 1).unwrap();
        assert_eq!(output.shape(), &[1, 2, 2]);
    }

    #[test]
    fn test_batch_parallel() {
        let graph = create_simple_graph();

        // Input: [batch=4, hidden=3] - parallelize over batch
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[4, 3]),
            vec![0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3],
        )
        .unwrap();
        let upper = lower.mapv(|x| x + 0.1);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let config = ParallelConfig {
            method: PropagationMethod::Ibp,
            min_positions_for_parallel: 1,
            ..Default::default()
        };
        let verifier = ParallelVerifier::new(config);

        let result = verifier.verify_batch_parallel(&graph, &input, 0).unwrap();

        // Output: [batch=4, output=2]
        assert_eq!(result.output_bounds.shape(), &[4, 2]);
    }

    #[test]
    fn test_axis_out_of_bounds() {
        let graph = create_simple_graph();

        let lower = arr3(&[[[0.0, 0.0, 0.0]]]).into_dyn();
        let upper = lower.mapv(|x| x + 0.1);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let verifier = ParallelVerifier::new(ParallelConfig::default());
        let result = verifier.verify_positions_parallel(&graph, &input, 10);

        assert!(result.is_err());
    }

    #[test]
    fn test_sdp_crown_parallel_uniform_epsilon() {
        // Simple ReLU MLP that SDP-CROWN supports
        let graph = create_simple_graph();

        // Input with uniform epsilon (same half-width in all dimensions)
        // Shape: [batch=1, seq=4, hidden=3]
        let epsilon = 0.05_f32;
        let center = arr3(&[[
            [0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6],
            [0.7, 0.7, 0.7],
            [0.8, 0.8, 0.8],
        ]])
        .into_dyn();
        let lower = center.mapv(|x| x - epsilon);
        let upper = center.mapv(|x| x + epsilon);
        let input = BoundedTensor::new(lower, upper).unwrap();

        // Verify with SDP-CROWN (should use actual SDP-CROWN on ReLU network)
        let config = ParallelConfig {
            method: PropagationMethod::SdpCrown,
            min_positions_for_parallel: 2,
            ..Default::default()
        };
        let verifier = ParallelVerifier::new(config);
        let sdp_result = verifier
            .verify_positions_parallel(&graph, &input, 1)
            .unwrap();

        // Also verify with CROWN for comparison
        let crown_config = ParallelConfig {
            method: PropagationMethod::Crown,
            min_positions_for_parallel: 2,
            ..Default::default()
        };
        let crown_verifier = ParallelVerifier::new(crown_config);
        let crown_result = crown_verifier
            .verify_positions_parallel(&graph, &input, 1)
            .unwrap();

        // Both should produce valid bounds with same shape
        assert_eq!(
            sdp_result.output_bounds.shape(),
            crown_result.output_bounds.shape()
        );
        assert_eq!(sdp_result.output_bounds.shape(), &[1, 4, 2]);

        // Note: SDP-CROWN computes bounds for an ℓ2 ball circumscribing the ℓ∞ box.
        // When rho = half-width, the ℓ2 ball is larger (√n times in diameter).
        // So SDP-CROWN bounds may be looser on ℓ∞ box inputs, which is expected.
        // The important thing is that SDP-CROWN runs without errors.
        let sdp_width = sdp_result.output_bounds.max_width();
        let crown_width = crown_result.output_bounds.max_width();

        // Both should produce finite, reasonable bounds
        assert!(
            sdp_width.is_finite(),
            "SDP-CROWN should produce finite bounds"
        );
        assert!(
            crown_width.is_finite(),
            "CROWN should produce finite bounds"
        );

        // SDP-CROWN bounds should be within reasonable range of CROWN
        // (may be looser due to larger input set, but not dramatically so)
        assert!(
            sdp_width < crown_width * 2.0,
            "SDP-CROWN bounds should be within 2x of CROWN: SDP={}, CROWN={}",
            sdp_width,
            crown_width
        );
    }

    #[test]
    fn test_sdp_crown_parallel_non_uniform_epsilon_fallback() {
        let graph = create_simple_graph();

        // Input with NON-uniform epsilon (different half-widths per dimension)
        // Shape: [batch=1, seq=2, hidden=3]
        // First position: epsilon=0.1, second position: epsilon=0.2
        let lower = arr3(&[[[0.4, 0.4, 0.4], [0.3, 0.3, 0.3]]]).into_dyn();
        let upper = arr3(&[
            [[0.6, 0.6, 0.6], [0.7, 0.7, 0.7]], // Different epsilon per position
        ])
        .into_dyn();
        let input = BoundedTensor::new(lower, upper).unwrap();

        // SDP-CROWN should gracefully fall back to CROWN (non-uniform epsilon)
        let config = ParallelConfig {
            method: PropagationMethod::SdpCrown,
            min_positions_for_parallel: 1,
            ..Default::default()
        };
        let verifier = ParallelVerifier::new(config);
        let result = verifier.verify_positions_parallel(&graph, &input, 1);

        // Should succeed (falls back to CROWN)
        assert!(result.is_ok());
        assert_eq!(result.unwrap().output_bounds.shape(), &[1, 2, 2]);
    }

    #[test]
    fn test_infer_l2_ball_uniform() {
        // Uniform epsilon box
        let epsilon = 0.1_f32;
        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 0.5, 1.0, 1.5]).unwrap();
        let upper = lower.mapv(|x| x + 2.0 * epsilon);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let result = super::infer_l2_ball_from_box(&input);
        assert!(result.is_some());
        let (x_hat, rho) = result.unwrap();

        // Center should be midpoint
        assert!((x_hat[0] - 0.1).abs() < 1e-5);
        assert!((x_hat[1] - 0.6).abs() < 1e-5);

        // Rho should equal epsilon
        assert!((rho - epsilon).abs() < 1e-5);
    }

    #[test]
    fn test_infer_l2_ball_non_uniform() {
        // Non-uniform epsilon box (should return None)
        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0, 0.0, 0.0]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.1, 0.2, 0.3]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let result = super::infer_l2_ball_from_box(&input);
        assert!(result.is_none(), "Non-uniform epsilon should return None");
    }

    // ============== ParallelConfig Tests ==============

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert!(matches!(config.method, PropagationMethod::Ibp));
        assert_eq!(config.min_positions_for_parallel, 4);
        assert!(config.max_threads.is_none());
        assert!(!config.report_progress);
    }

    #[test]
    fn test_parallel_config_custom() {
        let config = ParallelConfig {
            method: PropagationMethod::Crown,
            min_positions_for_parallel: 10,
            max_threads: Some(4),
            report_progress: true,
        };

        assert!(matches!(config.method, PropagationMethod::Crown));
        assert_eq!(config.min_positions_for_parallel, 10);
        assert_eq!(config.max_threads, Some(4));
        assert!(config.report_progress);
    }

    // ============== ParallelVerificationResult Tests ==============

    #[test]
    fn test_parallel_verification_result_fields() {
        let graph = create_simple_graph();

        let lower = arr3(&[[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]]).into_dyn();
        let upper = lower.mapv(|x| x + 0.1);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let config = ParallelConfig {
            method: PropagationMethod::Ibp,
            min_positions_for_parallel: 1,
            ..Default::default()
        };
        let verifier = ParallelVerifier::new(config);

        let result = verifier
            .verify_positions_parallel(&graph, &input, 1)
            .unwrap();

        // Check all fields are populated correctly
        assert_eq!(result.num_positions, 3);
        assert_eq!(result.parallel_positions, 3);
        assert!(result.avg_position_time_ms >= 0.0);
        assert_eq!(result.output_bounds.shape(), &[1, 3, 2]);
    }

    // ============== infer_l2_ball_from_box Tests ==============

    #[test]
    fn test_infer_l2_ball_empty() {
        // Empty input
        let lower = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let result = super::infer_l2_ball_from_box(&input);
        assert!(result.is_some());
        let (x_hat, rho) = result.unwrap();
        assert_eq!(x_hat.len(), 0);
        assert!((rho - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_infer_l2_ball_single_element() {
        // Single element
        let lower = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let result = super::infer_l2_ball_from_box(&input);
        assert!(result.is_some());
        let (x_hat, rho) = result.unwrap();

        // Center = (1 + 3) / 2 = 2
        assert!((x_hat[0] - 2.0).abs() < 1e-5);
        // Radius = (3 - 1) / 2 = 1
        assert!((rho - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_infer_l2_ball_concrete_point() {
        // Concrete point (lower == upper)
        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let upper = lower.clone();
        let input = BoundedTensor::new(lower.clone(), upper).unwrap();

        let result = super::infer_l2_ball_from_box(&input);
        assert!(result.is_some());
        let (x_hat, rho) = result.unwrap();

        assert!((x_hat[0] - 1.0).abs() < 1e-5);
        assert!((x_hat[1] - 2.0).abs() < 1e-5);
        assert!((x_hat[2] - 3.0).abs() < 1e-5);
        assert!((rho - 0.0).abs() < 1e-5);
    }

    // ============== verify_parallel_with_method Tests ==============

    #[test]
    fn test_verify_parallel_with_method_ibp() {
        let graph = create_simple_graph();

        let lower = arr3(&[[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]]).into_dyn();
        let upper = lower.mapv(|x| x + 0.1);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let output =
            verify_parallel_with_method(&graph, &input, 1, PropagationMethod::Ibp).unwrap();
        assert_eq!(output.shape(), &[1, 2, 2]);
    }

    #[test]
    fn test_verify_parallel_with_method_crown() {
        let graph = create_simple_graph();

        let lower = arr3(&[[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]]).into_dyn();
        let upper = lower.mapv(|x| x + 0.1);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let output =
            verify_parallel_with_method(&graph, &input, 1, PropagationMethod::Crown).unwrap();
        assert_eq!(output.shape(), &[1, 2, 2]);
    }

    // ============== Serial Fallback Tests ==============

    #[test]
    fn test_parallel_serial_fallback_for_few_positions() {
        let graph = create_simple_graph();

        // Only 2 positions, with threshold of 4 -> should use serial
        let lower = arr3(&[[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]]).into_dyn();
        let upper = lower.mapv(|x| x + 0.1);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let config = ParallelConfig {
            method: PropagationMethod::Ibp,
            min_positions_for_parallel: 4, // Higher than 2 positions
            ..Default::default()
        };
        let verifier = ParallelVerifier::new(config);

        let result = verifier
            .verify_positions_parallel(&graph, &input, 1)
            .unwrap();

        // Should use serial (parallel_positions = 0)
        assert_eq!(result.num_positions, 2);
        assert_eq!(result.parallel_positions, 0);
        assert_eq!(result.output_bounds.shape(), &[1, 2, 2]);
    }

    // ============== Max Threads Configuration Tests ==============

    #[test]
    fn test_parallel_with_max_threads() {
        let graph = create_simple_graph();

        let lower = arr3(&[[
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
        ]])
        .into_dyn();
        let upper = lower.mapv(|x| x + 0.1);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let config = ParallelConfig {
            method: PropagationMethod::Ibp,
            min_positions_for_parallel: 1,
            max_threads: Some(2), // Limit to 2 threads
            ..Default::default()
        };
        let verifier = ParallelVerifier::new(config);

        let result = verifier
            .verify_positions_parallel(&graph, &input, 1)
            .unwrap();

        // Should still work correctly with limited threads
        assert_eq!(result.output_bounds.shape(), &[1, 4, 2]);
        assert_eq!(result.num_positions, 4);
    }

    // ============== AlphaCrown and BetaCrown Tests ==============

    #[test]
    fn test_parallel_alpha_crown_method() {
        let graph = create_simple_graph();

        let lower = arr3(&[[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]]).into_dyn();
        let upper = lower.mapv(|x| x + 0.1);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let config = ParallelConfig {
            method: PropagationMethod::AlphaCrown,
            min_positions_for_parallel: 1,
            ..Default::default()
        };
        let verifier = ParallelVerifier::new(config);

        // Should work (falls back to CROWN or IBP internally)
        let result = verifier.verify_positions_parallel(&graph, &input, 1);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().output_bounds.shape(), &[1, 2, 2]);
    }

    #[test]
    fn test_parallel_beta_crown_method() {
        let graph = create_simple_graph();

        let lower = arr3(&[[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]]).into_dyn();
        let upper = lower.mapv(|x| x + 0.1);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let config = ParallelConfig {
            method: PropagationMethod::BetaCrown,
            min_positions_for_parallel: 1,
            ..Default::default()
        };
        let verifier = ParallelVerifier::new(config);

        // Should work (falls back to CROWN or IBP internally)
        let result = verifier.verify_positions_parallel(&graph, &input, 1);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().output_bounds.shape(), &[1, 2, 2]);
    }

    // ============== Progress Reporting Tests ==============

    #[test]
    fn test_parallel_with_progress_reporting() {
        let graph = create_simple_graph();

        let lower = arr3(&[[
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4],
        ]])
        .into_dyn();
        let upper = lower.mapv(|x| x + 0.1);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let config = ParallelConfig {
            method: PropagationMethod::Ibp,
            min_positions_for_parallel: 1,
            report_progress: true, // Enable progress reporting
            ..Default::default()
        };
        let verifier = ParallelVerifier::new(config);

        // Should work with progress reporting enabled
        let result = verifier
            .verify_positions_parallel(&graph, &input, 1)
            .unwrap();
        assert_eq!(result.output_bounds.shape(), &[1, 5, 2]);
    }

    // ============== Multidimensional Batch Tests ==============

    #[test]
    fn test_parallel_over_different_axes() {
        let graph = create_simple_graph();

        // Input: [batch=2, seq=3, hidden=3]
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 3]),
            vec![
                0.0, 0.0, 0.0, // batch 0, seq 0
                0.1, 0.1, 0.1, // batch 0, seq 1
                0.2, 0.2, 0.2, // batch 0, seq 2
                0.3, 0.3, 0.3, // batch 1, seq 0
                0.4, 0.4, 0.4, // batch 1, seq 1
                0.5, 0.5, 0.5, // batch 1, seq 2
            ],
        )
        .unwrap();
        let upper = lower.mapv(|x| x + 0.1);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let config = ParallelConfig {
            method: PropagationMethod::Ibp,
            min_positions_for_parallel: 1,
            ..Default::default()
        };
        let verifier = ParallelVerifier::new(config);

        // Parallel over batch axis (0)
        let result_batch = verifier
            .verify_positions_parallel(&graph, &input, 0)
            .unwrap();
        assert_eq!(result_batch.output_bounds.shape(), &[2, 3, 2]);

        // Parallel over seq axis (1)
        let result_seq = verifier
            .verify_positions_parallel(&graph, &input, 1)
            .unwrap();
        assert_eq!(result_seq.output_bounds.shape(), &[2, 3, 2]);
    }
}
