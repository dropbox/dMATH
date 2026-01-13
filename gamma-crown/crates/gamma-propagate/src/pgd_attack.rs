//! PGD (Projected Gradient Descent) attack for finding counterexamples.
//!
//! This module implements adversarial attacks to find inputs that violate
//! verification properties. When verification is inconclusive, we can try to
//! find a concrete counterexample that proves the property is violated.
//!
//! ## Algorithm
//!
//! 1. **Random Initialization**: Sample random points within input bounds
//! 2. **Gradient Estimation**: Use SPSA to estimate gradients without backprop
//! 3. **Gradient Step**: Move toward property violation
//! 4. **Projection**: Clip to input bounds
//! 5. **Repeat**: Multiple restarts for robustness
//!
//! ## References
//!
//! - α,β-CROWN uses PGD with 10000 restarts for ACAS-Xu benchmarks
//! - SPSA: Spall, J.C. (1992). "Multivariate Stochastic Approximation Using a
//!   Simultaneous Perturbation Gradient Approximation"

use gamma_core::Result;
use gamma_tensor::BoundedTensor;
use ndarray::{Array1, ArrayD, IxDyn};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{debug, trace};

use crate::Network;

/// Configuration for PGD attack.
#[derive(Debug, Clone)]
pub struct PgdConfig {
    /// Number of random restarts.
    pub num_restarts: usize,
    /// Number of gradient steps per restart.
    pub num_steps: usize,
    /// Step size for gradient updates.
    pub step_size: f32,
    /// Perturbation size for SPSA gradient estimation.
    pub spsa_delta: f32,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Whether to run restarts in parallel using Rayon.
    pub parallel: bool,
}

impl Default for PgdConfig {
    fn default() -> Self {
        Self {
            num_restarts: 100,
            num_steps: 50,
            step_size: 0.01,
            spsa_delta: 0.001,
            seed: 42,
            parallel: true,
        }
    }
}

impl PgdConfig {
    /// Create config for fast attack (fewer restarts).
    pub fn fast() -> Self {
        Self {
            num_restarts: 10,
            num_steps: 20,
            step_size: 0.01,
            spsa_delta: 0.001,
            seed: 42,
            parallel: false, // Too few restarts to benefit
        }
    }

    /// Create config for thorough attack (many restarts).
    pub fn thorough() -> Self {
        Self {
            num_restarts: 1000,
            num_steps: 100,
            step_size: 0.005,
            spsa_delta: 0.0005,
            seed: 42,
            parallel: true,
        }
    }

    /// Create config optimized for ACAS-Xu (high restart count like α,β-CROWN).
    pub fn acas_xu() -> Self {
        Self {
            num_restarts: 5000,
            num_steps: 50,
            step_size: 0.01,
            spsa_delta: 0.001,
            seed: 42,
            parallel: true,
        }
    }
}

/// Result of a PGD attack.
#[derive(Debug, Clone)]
pub struct PgdResult {
    /// Whether a counterexample was found.
    pub found_counterexample: bool,
    /// The counterexample input (if found).
    pub counterexample: Option<ArrayD<f32>>,
    /// Output at counterexample (if found).
    pub output: Option<ArrayD<f32>>,
    /// Best (most violating) output value found.
    pub best_output_value: f32,
    /// Number of restarts completed.
    pub restarts_completed: usize,
    /// Total network evaluations.
    pub total_evaluations: usize,
}

/// Internal result from a single PGD restart.
///
/// This struct replaces the complex tuple type `(ArrayD<f32>, ArrayD<f32>, f32, bool, usize)`
/// for better readability and self-documentation.
struct RestartResult {
    /// The input point found by this restart.
    input: ArrayD<f32>,
    /// Network output at the found input point.
    output: ArrayD<f32>,
    /// The objective value (output[idx], difference, or conjunctive max).
    value: f32,
    /// Whether this result represents a property violation.
    is_violation: bool,
    /// Number of network evaluations used in this restart.
    evaluations: usize,
}

/// PGD attacker for finding counterexamples.
pub struct PgdAttacker {
    config: PgdConfig,
}

impl PgdAttacker {
    /// Create a new PGD attacker with the given configuration.
    pub fn new(config: PgdConfig) -> Self {
        Self { config }
    }

    /// Evaluate network at a concrete point.
    ///
    /// Returns the network output as a flat f32 array.
    fn evaluate(&self, network: &Network, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let input_bounds = BoundedTensor::concrete(input.clone());
        let output_bounds = network.propagate_ibp(&input_bounds)?;
        // For concrete input, lower == upper
        Ok(output_bounds.lower)
    }

    /// Estimate gradient using SPSA (Simultaneous Perturbation Stochastic Approximation).
    ///
    /// SPSA uses random perturbations to estimate gradients with only 2 function evaluations,
    /// regardless of input dimension. This is much more efficient than finite differences
    /// for high-dimensional inputs.
    fn estimate_gradient_spsa(
        &self,
        network: &Network,
        input: &ArrayD<f32>,
        output_idx: usize,
        rng: &mut StdRng,
    ) -> Result<(ArrayD<f32>, usize)> {
        let delta = self.config.spsa_delta;
        let n = input.len();
        let mut evals = 0;

        // Generate random perturbation vector (Bernoulli ±1)
        let perturbation: Array1<f32> = (0..n)
            .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
            .collect();
        let perturbation = perturbation
            .into_shape_with_order(IxDyn(input.shape()))
            .unwrap();

        // Evaluate at x + delta * perturbation
        let input_plus = input + &perturbation * delta;
        let output_plus = self.evaluate(network, &input_plus)?;
        evals += 1;

        // Evaluate at x - delta * perturbation
        let input_minus = input - &perturbation * delta;
        let output_minus = self.evaluate(network, &input_minus)?;
        evals += 1;

        // SPSA gradient estimate: (f(x+) - f(x-)) / (2 * delta * perturbation)
        let y_plus = output_plus.iter().nth(output_idx).copied().unwrap_or(0.0);
        let y_minus = output_minus.iter().nth(output_idx).copied().unwrap_or(0.0);
        let diff = y_plus - y_minus;

        // Gradient estimate for each dimension
        let gradient = &perturbation * (diff / (2.0 * delta));

        Ok((gradient, evals))
    }

    /// Project point back to input bounds.
    fn project(&self, x: &ArrayD<f32>, bounds: &BoundedTensor) -> ArrayD<f32> {
        let lower = &bounds.lower;
        let upper = &bounds.upper;

        // Element-wise clipping
        let mut result = x.clone();
        for (val, (l, u)) in result.iter_mut().zip(lower.iter().zip(upper.iter())) {
            *val = val.clamp(*l, *u);
        }
        result
    }

    /// Sample a random point within input bounds.
    fn sample_uniform(&self, bounds: &BoundedTensor, rng: &mut StdRng) -> ArrayD<f32> {
        let lower = &bounds.lower;
        let upper = &bounds.upper;

        let mut result = ArrayD::zeros(IxDyn(bounds.shape()));
        for (val, (l, u)) in result.iter_mut().zip(lower.iter().zip(upper.iter())) {
            *val = rng.random_range(*l..=*u);
        }
        result
    }

    /// Run a single PGD restart (internal helper).
    fn run_single_restart(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx: usize,
        threshold: f32,
        verify_upper_bound: bool,
        seed: u64,
    ) -> Result<RestartResult> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut x = self.sample_uniform(input_bounds, &mut rng);
        let mut evals = 0;

        // Run gradient descent steps
        for _step in 0..self.config.num_steps {
            let (gradient, step_evals) =
                self.estimate_gradient_spsa(network, &x, output_idx, &mut rng)?;
            evals += step_evals;

            let step = if verify_upper_bound {
                &gradient * self.config.step_size
            } else {
                &gradient * (-self.config.step_size)
            };

            x = &x + &step;
            x = self.project(&x, input_bounds);
        }

        // Evaluate final point
        let output = self.evaluate(network, &x)?;
        evals += 1;
        let value = output.iter().nth(output_idx).copied().unwrap_or(0.0);

        let is_violation = if verify_upper_bound {
            value >= threshold
        } else {
            value <= threshold
        };

        Ok(RestartResult {
            input: x,
            output,
            value,
            is_violation,
            evaluations: evals,
        })
    }

    /// Run PGD attack to find counterexample where `output[output_idx]` violates threshold.
    ///
    /// For `verify_upper_bound = true`: looking for output >= threshold (property violation)
    /// For `verify_upper_bound = false`: looking for output <= threshold (property violation)
    pub fn attack(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx: usize,
        threshold: f32,
        verify_upper_bound: bool,
    ) -> Result<PgdResult> {
        if self.config.parallel && self.config.num_restarts >= 10 {
            self.attack_parallel(
                network,
                input_bounds,
                output_idx,
                threshold,
                verify_upper_bound,
            )
        } else {
            self.attack_sequential(
                network,
                input_bounds,
                output_idx,
                threshold,
                verify_upper_bound,
            )
        }
    }

    /// Sequential PGD attack (original implementation).
    fn attack_sequential(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx: usize,
        threshold: f32,
        verify_upper_bound: bool,
    ) -> Result<PgdResult> {
        let mut best_counterexample: Option<ArrayD<f32>> = None;
        let mut best_output: Option<ArrayD<f32>> = None;
        let mut best_value = if verify_upper_bound {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        let mut total_evaluations = 0;

        for restart in 0..self.config.num_restarts {
            let seed = self.config.seed.wrapping_add(restart as u64);
            let result = self.run_single_restart(
                network,
                input_bounds,
                output_idx,
                threshold,
                verify_upper_bound,
                seed,
            )?;
            total_evaluations += result.evaluations;

            let is_better = if verify_upper_bound {
                result.value > best_value
            } else {
                result.value < best_value
            };

            if is_better {
                best_value = result.value;
                best_counterexample = Some(result.input.clone());
                best_output = Some(result.output.clone());
            }

            if result.is_violation {
                debug!(
                    "PGD found counterexample at restart {}: output[{}] = {} {} threshold {}",
                    restart,
                    output_idx,
                    result.value,
                    if verify_upper_bound { ">=" } else { "<=" },
                    threshold
                );
                return Ok(PgdResult {
                    found_counterexample: true,
                    counterexample: Some(result.input),
                    output: Some(result.output),
                    best_output_value: result.value,
                    restarts_completed: restart + 1,
                    total_evaluations,
                });
            }

            trace!(
                "PGD restart {} complete: best output[{}] = {}, target {} threshold {}",
                restart,
                output_idx,
                best_value,
                if verify_upper_bound { ">=" } else { "<=" },
                threshold
            );
        }

        debug!(
            "PGD completed {} restarts without finding counterexample. Best: {} vs threshold {}",
            self.config.num_restarts, best_value, threshold
        );

        Ok(PgdResult {
            found_counterexample: false,
            counterexample: best_counterexample,
            output: best_output,
            best_output_value: best_value,
            restarts_completed: self.config.num_restarts,
            total_evaluations,
        })
    }

    /// Parallel PGD attack using Rayon.
    fn attack_parallel(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx: usize,
        threshold: f32,
        verify_upper_bound: bool,
    ) -> Result<PgdResult> {
        let found = AtomicBool::new(false);

        // Run restarts in parallel, with early termination when counterexample found
        let results: Vec<_> = (0..self.config.num_restarts)
            .into_par_iter()
            .filter_map(|restart| {
                // Skip if another thread found a counterexample
                if found.load(Ordering::Relaxed) {
                    return None;
                }

                let seed = self.config.seed.wrapping_add(restart as u64);
                match self.run_single_restart(
                    network, input_bounds, output_idx, threshold, verify_upper_bound, seed,
                ) {
                    Ok(result) => {
                        if result.is_violation {
                            found.store(true, Ordering::Relaxed);
                            debug!(
                                "PGD found counterexample at restart {}: output[{}] = {} {} threshold {}",
                                restart, output_idx, result.value,
                                if verify_upper_bound { ">=" } else { "<=" },
                                threshold
                            );
                        }
                        Some((restart, result))
                    }
                    Err(_) => None,
                }
            })
            .collect();

        // Find the best result (counterexample if any, otherwise best candidate)
        let mut best_counterexample: Option<ArrayD<f32>> = None;
        let mut best_output: Option<ArrayD<f32>> = None;
        let mut best_value = if verify_upper_bound {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        let mut total_evaluations = 0;
        let mut found_violation = false;
        let mut earliest_violation_restart = usize::MAX;

        for (restart, result) in results {
            total_evaluations += result.evaluations;

            if result.is_violation && restart < earliest_violation_restart {
                earliest_violation_restart = restart;
                found_violation = true;
                best_value = result.value;
                best_counterexample = Some(result.input);
                best_output = Some(result.output);
            } else if !found_violation {
                let is_better = if verify_upper_bound {
                    result.value > best_value
                } else {
                    result.value < best_value
                };
                if is_better {
                    best_value = result.value;
                    best_counterexample = Some(result.input);
                    best_output = Some(result.output);
                }
            }
        }

        if !found_violation {
            debug!(
                "PGD completed {} restarts without finding counterexample. Best: {} vs threshold {}",
                self.config.num_restarts, best_value, threshold
            );
        }

        Ok(PgdResult {
            found_counterexample: found_violation,
            counterexample: best_counterexample,
            output: best_output,
            best_output_value: best_value,
            restarts_completed: if found_violation {
                earliest_violation_restart + 1
            } else {
                self.config.num_restarts
            },
            total_evaluations,
        })
    }

    /// Run a single PGD restart for difference constraint (internal helper).
    #[allow(clippy::too_many_arguments)]
    fn run_single_restart_difference(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx_i: usize,
        output_idx_j: usize,
        threshold: f32,
        verify_upper_bound: bool,
        seed: u64,
    ) -> Result<RestartResult> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut x = self.sample_uniform(input_bounds, &mut rng);
        let mut evals = 0;

        for _step in 0..self.config.num_steps {
            let (grad_i, evals_i) =
                self.estimate_gradient_spsa(network, &x, output_idx_i, &mut rng)?;
            let (grad_j, evals_j) =
                self.estimate_gradient_spsa(network, &x, output_idx_j, &mut rng)?;
            evals += evals_i + evals_j;

            let gradient_diff = &grad_i - &grad_j;
            let step = if verify_upper_bound {
                &gradient_diff * self.config.step_size
            } else {
                &gradient_diff * (-self.config.step_size)
            };

            x = &x + &step;
            x = self.project(&x, input_bounds);
        }

        let output = self.evaluate(network, &x)?;
        evals += 1;

        let val_i = output.iter().nth(output_idx_i).copied().unwrap_or(0.0);
        let val_j = output.iter().nth(output_idx_j).copied().unwrap_or(0.0);
        let diff = val_i - val_j;

        let is_violation = if verify_upper_bound {
            diff >= threshold
        } else {
            diff <= threshold
        };

        Ok(RestartResult {
            input: x,
            output,
            value: diff,
            is_violation,
            evaluations: evals,
        })
    }

    /// Attack to find counterexample for a difference constraint: `output[i] - output[j]` violates threshold.
    ///
    /// This is common in ACAS-Xu where we verify Y_i <= Y_j (difference <= 0).
    pub fn attack_difference(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx_i: usize,
        output_idx_j: usize,
        threshold: f32,
        verify_upper_bound: bool,
    ) -> Result<PgdResult> {
        if self.config.parallel && self.config.num_restarts >= 10 {
            self.attack_difference_parallel(
                network,
                input_bounds,
                output_idx_i,
                output_idx_j,
                threshold,
                verify_upper_bound,
            )
        } else {
            self.attack_difference_sequential(
                network,
                input_bounds,
                output_idx_i,
                output_idx_j,
                threshold,
                verify_upper_bound,
            )
        }
    }

    /// Sequential attack for difference constraint.
    fn attack_difference_sequential(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx_i: usize,
        output_idx_j: usize,
        threshold: f32,
        verify_upper_bound: bool,
    ) -> Result<PgdResult> {
        let mut best_counterexample: Option<ArrayD<f32>> = None;
        let mut best_output: Option<ArrayD<f32>> = None;
        let mut best_diff = if verify_upper_bound {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        let mut total_evaluations = 0;

        for restart in 0..self.config.num_restarts {
            let seed = self.config.seed.wrapping_add(restart as u64);
            let result = self.run_single_restart_difference(
                network,
                input_bounds,
                output_idx_i,
                output_idx_j,
                threshold,
                verify_upper_bound,
                seed,
            )?;
            total_evaluations += result.evaluations;

            let is_better = if verify_upper_bound {
                result.value > best_diff
            } else {
                result.value < best_diff
            };

            if is_better {
                best_diff = result.value;
                best_counterexample = Some(result.input.clone());
                best_output = Some(result.output.clone());
            }

            if result.is_violation {
                debug!(
                    "PGD found counterexample at restart {}: Y[{}] - Y[{}] = {} {} threshold {}",
                    restart,
                    output_idx_i,
                    output_idx_j,
                    result.value,
                    if verify_upper_bound { ">=" } else { "<=" },
                    threshold
                );
                return Ok(PgdResult {
                    found_counterexample: true,
                    counterexample: Some(result.input),
                    output: Some(result.output),
                    best_output_value: result.value,
                    restarts_completed: restart + 1,
                    total_evaluations,
                });
            }
        }

        Ok(PgdResult {
            found_counterexample: false,
            counterexample: best_counterexample,
            output: best_output,
            best_output_value: best_diff,
            restarts_completed: self.config.num_restarts,
            total_evaluations,
        })
    }

    /// Parallel attack for difference constraint using Rayon.
    fn attack_difference_parallel(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx_i: usize,
        output_idx_j: usize,
        threshold: f32,
        verify_upper_bound: bool,
    ) -> Result<PgdResult> {
        let found = AtomicBool::new(false);

        let results: Vec<_> = (0..self.config.num_restarts)
            .into_par_iter()
            .filter_map(|restart| {
                if found.load(Ordering::Relaxed) {
                    return None;
                }

                let seed = self.config.seed.wrapping_add(restart as u64);
                match self.run_single_restart_difference(
                    network, input_bounds, output_idx_i, output_idx_j, threshold, verify_upper_bound, seed,
                ) {
                    Ok(result) => {
                        if result.is_violation {
                            found.store(true, Ordering::Relaxed);
                            debug!(
                                "PGD found counterexample at restart {}: Y[{}] - Y[{}] = {} {} threshold {}",
                                restart, output_idx_i, output_idx_j, result.value,
                                if verify_upper_bound { ">=" } else { "<=" },
                                threshold
                            );
                        }
                        Some((restart, result))
                    }
                    Err(_) => None,
                }
            })
            .collect();

        let mut best_counterexample: Option<ArrayD<f32>> = None;
        let mut best_output: Option<ArrayD<f32>> = None;
        let mut best_diff = if verify_upper_bound {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        let mut total_evaluations = 0;
        let mut found_violation = false;
        let mut earliest_violation_restart = usize::MAX;

        for (restart, result) in results {
            total_evaluations += result.evaluations;

            if result.is_violation && restart < earliest_violation_restart {
                earliest_violation_restart = restart;
                found_violation = true;
                best_diff = result.value;
                best_counterexample = Some(result.input);
                best_output = Some(result.output);
            } else if !found_violation {
                let is_better = if verify_upper_bound {
                    result.value > best_diff
                } else {
                    result.value < best_diff
                };
                if is_better {
                    best_diff = result.value;
                    best_counterexample = Some(result.input);
                    best_output = Some(result.output);
                }
            }
        }

        Ok(PgdResult {
            found_counterexample: found_violation,
            counterexample: best_counterexample,
            output: best_output,
            best_output_value: best_diff,
            restarts_completed: if found_violation {
                earliest_violation_restart + 1
            } else {
                self.config.num_restarts
            },
            total_evaluations,
        })
    }

    /// Run a single PGD restart for conjunctive LessEq constraints (internal helper).
    ///
    /// For constraints Y_target <= Y_j for each j in comparison_indices,
    /// finds input minimizing max(Y_target - Y_j for all j).
    /// A counterexample is found when max <= 0 (all constraints satisfied).
    fn run_single_restart_conjunctive_less_eq(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        target_idx: usize,
        comparison_indices: &[usize],
        seed: u64,
    ) -> Result<RestartResult> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut x = self.sample_uniform(input_bounds, &mut rng);
        let mut evals = 0;

        for _step in 0..self.config.num_steps {
            // Evaluate current point to find which constraint is most violated
            let output = self.evaluate(network, &x)?;
            evals += 1;

            let target_val = output.iter().nth(target_idx).copied().unwrap_or(0.0);

            // Find the constraint with max(Y_target - Y_j) - this is the "most violated"
            let mut max_diff = f32::NEG_INFINITY;
            let mut worst_j = comparison_indices[0];
            for &j in comparison_indices {
                let j_val = output.iter().nth(j).copied().unwrap_or(0.0);
                let diff = target_val - j_val;
                if diff > max_diff {
                    max_diff = diff;
                    worst_j = j;
                }
            }

            // Gradient descent on Y_target - Y_worst_j
            // We want to minimize max(Y_target - Y_j), so descend on the active constraint
            let (grad_target, evals_t) =
                self.estimate_gradient_spsa(network, &x, target_idx, &mut rng)?;
            let (grad_j, evals_j) = self.estimate_gradient_spsa(network, &x, worst_j, &mut rng)?;
            evals += evals_t + evals_j;

            // Gradient of (Y_target - Y_j) = grad_target - grad_j
            // To minimize, take negative gradient step
            let gradient_diff = &grad_target - &grad_j;
            let step = &gradient_diff * (-self.config.step_size);

            x = &x + &step;
            x = self.project(&x, input_bounds);
        }

        // Evaluate final point and compute max(Y_target - Y_j)
        let output = self.evaluate(network, &x)?;
        evals += 1;

        let target_val = output.iter().nth(target_idx).copied().unwrap_or(0.0);
        let mut max_diff = f32::NEG_INFINITY;
        for &j in comparison_indices {
            let j_val = output.iter().nth(j).copied().unwrap_or(0.0);
            let diff = target_val - j_val;
            if diff > max_diff {
                max_diff = diff;
            }
        }

        // Counterexample found if max_diff <= 0 (all Y_target <= Y_j satisfied)
        let is_violation = max_diff <= 0.0;

        Ok(RestartResult {
            input: x,
            output,
            value: max_diff,
            is_violation,
            evaluations: evals,
        })
    }

    /// Attack to find counterexample for conjunctive LessEq constraints.
    ///
    /// For constraints of the form: Y_target <= Y_j for each j in comparison_indices
    /// (common in ACAS-Xu prop_3/prop_4 where Y_0 must be minimal).
    ///
    /// Returns counterexample if found (input where ALL constraints are satisfied),
    /// which proves the VNNLIB property is violated (unsafe condition can occur).
    pub fn attack_conjunctive_less_eq(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        target_idx: usize,
        comparison_indices: &[usize],
    ) -> Result<PgdResult> {
        if self.config.parallel && self.config.num_restarts >= 10 {
            self.attack_conjunctive_less_eq_parallel(
                network,
                input_bounds,
                target_idx,
                comparison_indices,
            )
        } else {
            self.attack_conjunctive_less_eq_sequential(
                network,
                input_bounds,
                target_idx,
                comparison_indices,
            )
        }
    }

    /// Sequential attack for conjunctive LessEq constraints.
    fn attack_conjunctive_less_eq_sequential(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        target_idx: usize,
        comparison_indices: &[usize],
    ) -> Result<PgdResult> {
        let mut best_counterexample: Option<ArrayD<f32>> = None;
        let mut best_output: Option<ArrayD<f32>> = None;
        let mut best_max_diff = f32::INFINITY; // Lower is better (want max_diff <= 0)
        let mut total_evaluations = 0;

        for restart in 0..self.config.num_restarts {
            let seed = self.config.seed.wrapping_add(restart as u64);
            let result = self.run_single_restart_conjunctive_less_eq(
                network,
                input_bounds,
                target_idx,
                comparison_indices,
                seed,
            )?;
            total_evaluations += result.evaluations;

            if result.value < best_max_diff {
                best_max_diff = result.value;
                best_counterexample = Some(result.input.clone());
                best_output = Some(result.output.clone());
            }

            if result.is_violation {
                debug!(
                    "Conjunctive PGD found counterexample at restart {}: max(Y_{} - Y_j) = {} <= 0",
                    restart, target_idx, result.value
                );
                return Ok(PgdResult {
                    found_counterexample: true,
                    counterexample: Some(result.input),
                    output: Some(result.output),
                    best_output_value: result.value,
                    restarts_completed: restart + 1,
                    total_evaluations,
                });
            }
        }

        debug!(
            "Conjunctive PGD completed {} restarts without counterexample. Best max diff: {}",
            self.config.num_restarts, best_max_diff
        );

        Ok(PgdResult {
            found_counterexample: false,
            counterexample: best_counterexample,
            output: best_output,
            best_output_value: best_max_diff,
            restarts_completed: self.config.num_restarts,
            total_evaluations,
        })
    }

    /// Parallel attack for conjunctive LessEq constraints using Rayon.
    fn attack_conjunctive_less_eq_parallel(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        target_idx: usize,
        comparison_indices: &[usize],
    ) -> Result<PgdResult> {
        let found = AtomicBool::new(false);

        let results: Vec<_> = (0..self.config.num_restarts)
            .into_par_iter()
            .filter_map(|restart| {
                if found.load(Ordering::Relaxed) {
                    return None;
                }

                let seed = self.config.seed.wrapping_add(restart as u64);
                match self.run_single_restart_conjunctive_less_eq(
                    network, input_bounds, target_idx, comparison_indices, seed,
                ) {
                    Ok(result) => {
                        if result.is_violation {
                            found.store(true, Ordering::Relaxed);
                            debug!(
                                "Conjunctive PGD found counterexample at restart {}: max(Y_{} - Y_j) = {} <= 0",
                                restart, target_idx, result.value
                            );
                        }
                        Some((restart, result))
                    }
                    Err(_) => None,
                }
            })
            .collect();

        let mut best_counterexample: Option<ArrayD<f32>> = None;
        let mut best_output: Option<ArrayD<f32>> = None;
        let mut best_max_diff = f32::INFINITY;
        let mut total_evaluations = 0;
        let mut found_violation = false;
        let mut earliest_violation_restart = usize::MAX;

        for (restart, result) in results {
            total_evaluations += result.evaluations;

            if result.is_violation && restart < earliest_violation_restart {
                earliest_violation_restart = restart;
                found_violation = true;
                best_max_diff = result.value;
                best_counterexample = Some(result.input);
                best_output = Some(result.output);
            } else if !found_violation && result.value < best_max_diff {
                best_max_diff = result.value;
                best_counterexample = Some(result.input);
                best_output = Some(result.output);
            }
        }

        Ok(PgdResult {
            found_counterexample: found_violation,
            counterexample: best_counterexample,
            output: best_output,
            best_output_value: best_max_diff,
            restarts_completed: if found_violation {
                earliest_violation_restart + 1
            } else {
                self.config.num_restarts
            },
            total_evaluations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::*;
    use ndarray::{arr1, arr2};

    fn simple_linear_network() -> Network {
        // Simple network: y = W @ x + b
        // W = [[1, 2], [3, 4]], b = [0, 0]
        // So y[0] = x[0] + 2*x[1], y[1] = 3*x[0] + 4*x[1]
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
        ));
        network
    }

    #[test]
    fn test_pgd_evaluate_concrete() {
        let network = simple_linear_network();
        let attacker = PgdAttacker::new(PgdConfig::fast());

        let input = arr1(&[1.0_f32, 1.0]).into_dyn();
        let output = attacker.evaluate(&network, &input).unwrap();

        // y[0] = 1 + 2 = 3, y[1] = 3 + 4 = 7
        assert!((output[[0]] - 3.0).abs() < 1e-5);
        assert!((output[[1]] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_pgd_finds_counterexample() {
        // Network: y = x^2 (approximated by piecewise linear via ReLU)
        // For simplicity, use linear network y = 2*x
        // Input bounds: [-1, 1]
        // Property: y > 0 (should be violated at x < 0)
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[2.0]]), Some(arr1(&[0.0]))).unwrap(),
        ));

        let attacker = PgdAttacker::new(PgdConfig {
            num_restarts: 10,
            num_steps: 20,
            step_size: 0.1,
            spsa_delta: 0.01,
            seed: 42,
            parallel: false,
        });

        let input_bounds =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

        // Attack: find x where y <= 0 (verify_upper_bound = false, threshold = 0)
        let result = attacker
            .attack(&network, &input_bounds, 0, 0.0, false)
            .unwrap();

        // Should find counterexample at x < 0
        assert!(result.found_counterexample);
        let cx = result.counterexample.unwrap();
        assert!(cx[[0]] < 0.0);
    }

    #[test]
    fn test_pgd_no_counterexample_when_property_holds() {
        // Network: y = x + 5
        // Input bounds: [0, 1]
        // Property: y > 0 (always true since y >= 5)
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[5.0]))).unwrap(),
        ));

        let attacker = PgdAttacker::new(PgdConfig::fast());

        let input_bounds =
            BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

        // Attack: try to find x where y <= 0 (should fail)
        let result = attacker
            .attack(&network, &input_bounds, 0, 0.0, false)
            .unwrap();

        // Should NOT find counterexample
        assert!(!result.found_counterexample);
        // Best value should still be > 0
        assert!(result.best_output_value > 0.0);
    }

    #[test]
    fn test_pgd_difference_attack() {
        // Network with 2 outputs: y[0] = x, y[1] = 2*x
        // Property: y[0] <= y[1] (equivalent to y[0] - y[1] <= 0)
        // This should hold for x >= 0, violated for x < 0
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0], [2.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
        ));

        let attacker = PgdAttacker::new(PgdConfig {
            num_restarts: 20,
            num_steps: 30,
            step_size: 0.1,
            spsa_delta: 0.01,
            seed: 42,
            parallel: false,
        });

        let input_bounds =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

        // Attack: find x where y[0] - y[1] >= 0 (verify_upper_bound = true, threshold = 0)
        // At x < 0: y[0] - y[1] = x - 2x = -x > 0
        let result = attacker
            .attack_difference(&network, &input_bounds, 0, 1, 0.0, true)
            .unwrap();

        // Should find counterexample at x < 0
        assert!(result.found_counterexample);
        let cx = result.counterexample.unwrap();
        assert!(cx[[0]] < 0.0);
    }

    #[test]
    fn test_pgd_conjunctive_less_eq_attack() {
        // Network with 3 outputs: y[0] = -x + 1, y[1] = x + 2, y[2] = x + 3
        // For x in [-1, 1]:
        //   y[0] = -x + 1 in [0, 2]
        //   y[1] = x + 2 in [1, 3]
        //   y[2] = x + 3 in [2, 4]
        // Property: y[0] <= y[1] AND y[0] <= y[2] (COC is minimal)
        // At x = 1: y[0] = 0, y[1] = 3, y[2] = 4 -> y[0] < y[1] and y[0] < y[2] -> satisfied
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[-1.0], [1.0], [1.0]]), Some(arr1(&[1.0, 2.0, 3.0]))).unwrap(),
        ));

        let attacker = PgdAttacker::new(PgdConfig {
            num_restarts: 20,
            num_steps: 30,
            step_size: 0.1,
            spsa_delta: 0.01,
            seed: 42,
            parallel: false,
        });

        let input_bounds =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

        // Attack: find x where y[0] <= y[1] AND y[0] <= y[2]
        // At x = 1: y[0] = 0 < y[1] = 3 and y[0] = 0 < y[2] = 4 -> should find counterexample
        let result = attacker
            .attack_conjunctive_less_eq(&network, &input_bounds, 0, &[1, 2])
            .unwrap();

        // Should find counterexample since at x=1, y[0] is minimal
        assert!(result.found_counterexample);
        let _cx = result.counterexample.unwrap();
        // At counterexample, max(y[0] - y[1], y[0] - y[2]) <= 0
        assert!(result.best_output_value <= 0.0);
    }

    // ============== PgdConfig Tests ==============

    #[test]
    fn test_pgd_config_default() {
        let config = PgdConfig::default();
        assert_eq!(config.num_restarts, 100);
        assert_eq!(config.num_steps, 50);
        assert!((config.step_size - 0.01).abs() < 1e-6);
        assert!((config.spsa_delta - 0.001).abs() < 1e-6);
        assert_eq!(config.seed, 42);
        assert!(config.parallel);
    }

    #[test]
    fn test_pgd_config_fast() {
        let config = PgdConfig::fast();
        assert_eq!(config.num_restarts, 10);
        assert_eq!(config.num_steps, 20);
        assert!(!config.parallel); // Too few restarts to benefit
    }

    #[test]
    fn test_pgd_config_thorough() {
        let config = PgdConfig::thorough();
        assert_eq!(config.num_restarts, 1000);
        assert_eq!(config.num_steps, 100);
        assert!((config.step_size - 0.005).abs() < 1e-6);
        assert!((config.spsa_delta - 0.0005).abs() < 1e-6);
        assert!(config.parallel);
    }

    #[test]
    fn test_pgd_config_acas_xu() {
        let config = PgdConfig::acas_xu();
        assert_eq!(config.num_restarts, 5000);
        assert_eq!(config.num_steps, 50);
        assert!(config.parallel);
    }

    // ============== PgdResult Tests ==============

    #[test]
    fn test_pgd_result_fields_when_found() {
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[2.0]]), Some(arr1(&[0.0]))).unwrap(),
        ));

        let attacker = PgdAttacker::new(PgdConfig {
            num_restarts: 5,
            num_steps: 10,
            step_size: 0.1,
            spsa_delta: 0.01,
            seed: 42,
            parallel: false,
        });

        let input_bounds =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

        // Attack should find counterexample where y <= 0 (at x < 0)
        let result = attacker
            .attack(&network, &input_bounds, 0, 0.0, false)
            .unwrap();

        assert!(result.found_counterexample);
        assert!(result.counterexample.is_some());
        assert!(result.output.is_some());
        assert!(result.restarts_completed <= 5);
        assert!(result.total_evaluations > 0);
    }

    #[test]
    fn test_pgd_result_fields_when_not_found() {
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[10.0]))).unwrap(),
        ));

        let attacker = PgdAttacker::new(PgdConfig {
            num_restarts: 3,
            num_steps: 5,
            step_size: 0.1,
            spsa_delta: 0.01,
            seed: 42,
            parallel: false,
        });

        let input_bounds =
            BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

        // y = x + 10, so y >= 10 for x >= 0. Looking for y <= 0 should fail.
        let result = attacker
            .attack(&network, &input_bounds, 0, 0.0, false)
            .unwrap();

        assert!(!result.found_counterexample);
        // Still should have best candidate
        assert!(result.counterexample.is_some());
        assert!(result.output.is_some());
        assert_eq!(result.restarts_completed, 3);
        assert!(result.total_evaluations > 0);
        assert!(result.best_output_value > 0.0); // Couldn't get below 0
    }

    // ============== Helper Method Tests ==============

    #[test]
    fn test_pgd_project_clipping() {
        let attacker = PgdAttacker::new(PgdConfig::fast());
        let bounds = BoundedTensor::new(
            arr1(&[-1.0_f32, 0.0]).into_dyn(),
            arr1(&[1.0_f32, 2.0]).into_dyn(),
        )
        .unwrap();

        // Point outside bounds
        let x = arr1(&[5.0_f32, -5.0]).into_dyn();
        let projected = attacker.project(&x, &bounds);

        assert!((projected[[0]] - 1.0).abs() < 1e-6); // Clipped to upper
        assert!((projected[[1]] - 0.0).abs() < 1e-6); // Clipped to lower
    }

    #[test]
    fn test_pgd_project_within_bounds() {
        let attacker = PgdAttacker::new(PgdConfig::fast());
        let bounds = BoundedTensor::new(
            arr1(&[-1.0_f32, 0.0]).into_dyn(),
            arr1(&[1.0_f32, 2.0]).into_dyn(),
        )
        .unwrap();

        // Point inside bounds
        let x = arr1(&[0.5_f32, 1.0]).into_dyn();
        let projected = attacker.project(&x, &bounds);

        assert!((projected[[0]] - 0.5).abs() < 1e-6);
        assert!((projected[[1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pgd_sample_uniform_within_bounds() {
        let attacker = PgdAttacker::new(PgdConfig::fast());
        let bounds = BoundedTensor::new(
            arr1(&[-1.0_f32, 0.0]).into_dyn(),
            arr1(&[1.0_f32, 2.0]).into_dyn(),
        )
        .unwrap();

        let mut rng = StdRng::seed_from_u64(12345);

        // Sample multiple times and verify all within bounds
        for _ in 0..100 {
            let sample = attacker.sample_uniform(&bounds, &mut rng);

            assert!(sample[[0]] >= -1.0 && sample[[0]] <= 1.0);
            assert!(sample[[1]] >= 0.0 && sample[[1]] <= 2.0);
        }
    }

    #[test]
    fn test_pgd_estimate_gradient_spsa_direction() {
        // Test that SPSA gradient estimate points in reasonable direction
        // Network: y = x (identity for simplicity)
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0]]), None).unwrap(),
        ));

        let attacker = PgdAttacker::new(PgdConfig {
            spsa_delta: 0.01,
            ..PgdConfig::fast()
        });

        let x = arr1(&[0.5_f32]).into_dyn();
        let mut rng = StdRng::seed_from_u64(42);

        let (gradient, evals) = attacker
            .estimate_gradient_spsa(&network, &x, 0, &mut rng)
            .unwrap();

        // For y = x, gradient should be approximately 1
        // SPSA may have variance, but should be positive
        assert!(gradient[[0]] > 0.0, "Gradient should be positive for y=x");
        assert_eq!(evals, 2); // SPSA uses exactly 2 evaluations
    }

    // ============== Parallel Attack Tests ==============

    #[test]
    fn test_pgd_parallel_attack_finds_counterexample() {
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[2.0]]), Some(arr1(&[0.0]))).unwrap(),
        ));

        // Use parallel config with enough restarts
        let attacker = PgdAttacker::new(PgdConfig {
            num_restarts: 20,
            num_steps: 20,
            step_size: 0.1,
            spsa_delta: 0.01,
            seed: 42,
            parallel: true, // Enable parallel
        });

        let input_bounds =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

        // Looking for y <= 0 (should find at x < 0)
        let result = attacker
            .attack(&network, &input_bounds, 0, 0.0, false)
            .unwrap();

        assert!(result.found_counterexample);
        let cx = result.counterexample.unwrap();
        assert!(cx[[0]] < 0.0);
    }

    #[test]
    fn test_pgd_parallel_vs_sequential_consistency() {
        // Both parallel and sequential should find counterexamples for same problem
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[3.0]]), Some(arr1(&[-1.0]))).unwrap(),
        ));

        let input_bounds =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

        // Sequential attack
        let seq_attacker = PgdAttacker::new(PgdConfig {
            num_restarts: 15,
            num_steps: 15,
            step_size: 0.1,
            spsa_delta: 0.01,
            seed: 42,
            parallel: false,
        });
        let seq_result = seq_attacker
            .attack(&network, &input_bounds, 0, 0.0, false)
            .unwrap();

        // Parallel attack
        let par_attacker = PgdAttacker::new(PgdConfig {
            num_restarts: 15,
            num_steps: 15,
            step_size: 0.1,
            spsa_delta: 0.01,
            seed: 42,
            parallel: true,
        });
        let par_result = par_attacker
            .attack(&network, &input_bounds, 0, 0.0, false)
            .unwrap();

        // Both should find counterexamples
        assert!(seq_result.found_counterexample);
        assert!(par_result.found_counterexample);
    }

    #[test]
    fn test_pgd_parallel_difference_attack() {
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0], [2.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
        ));

        let attacker = PgdAttacker::new(PgdConfig {
            num_restarts: 20,
            num_steps: 20,
            step_size: 0.1,
            spsa_delta: 0.01,
            seed: 42,
            parallel: true, // Enable parallel
        });

        let input_bounds =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

        // Attack: find x where y[0] - y[1] >= 0
        // At x < 0: y[0] - y[1] = x - 2x = -x > 0
        let result = attacker
            .attack_difference(&network, &input_bounds, 0, 1, 0.0, true)
            .unwrap();

        assert!(result.found_counterexample);
    }

    #[test]
    fn test_pgd_parallel_conjunctive_attack() {
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[-1.0], [1.0], [1.0]]), Some(arr1(&[1.0, 2.0, 3.0]))).unwrap(),
        ));

        let attacker = PgdAttacker::new(PgdConfig {
            num_restarts: 25,
            num_steps: 25,
            step_size: 0.1,
            spsa_delta: 0.01,
            seed: 42,
            parallel: true, // Enable parallel
        });

        let input_bounds =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

        let result = attacker
            .attack_conjunctive_less_eq(&network, &input_bounds, 0, &[1, 2])
            .unwrap();

        assert!(result.found_counterexample);
        assert!(result.best_output_value <= 0.0);
    }

    // ============== RestartResult Tests ==============

    #[test]
    fn test_restart_result_struct() {
        // Test that RestartResult is properly constructed via run_single_restart
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
        ));

        let attacker = PgdAttacker::new(PgdConfig {
            num_restarts: 1,
            num_steps: 5,
            step_size: 0.1,
            spsa_delta: 0.01,
            seed: 42,
            parallel: false,
        });

        let input_bounds =
            BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

        // Use attack method which internally uses RestartResult
        let result = attacker
            .attack(&network, &input_bounds, 0, 10.0, true)
            .unwrap();

        // Check that RestartResult was properly created (via the output)
        assert!(result.output.is_some());
        let output = result.output.unwrap();
        assert_eq!(output.len(), 1); // Single output dimension
    }

    // ============== Edge Case Tests ==============

    #[test]
    fn test_pgd_multidimensional_input() {
        // Test with higher-dimensional input
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0, 0.5, -0.5], [0.5, 1.0, 0.5]]),
                Some(arr1(&[0.0, 0.0])),
            )
            .unwrap(),
        ));

        let attacker = PgdAttacker::new(PgdConfig::fast());

        let input_bounds = BoundedTensor::new(
            arr1(&[-1.0_f32, -1.0, -1.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0, 1.0]).into_dyn(),
        )
        .unwrap();

        // Just verify it runs without error on multidimensional input
        let result = attacker
            .attack(&network, &input_bounds, 0, 0.0, true)
            .unwrap();

        assert!(result.total_evaluations > 0);
    }

    #[test]
    fn test_pgd_tight_bounds() {
        // Test with very tight input bounds (nearly concrete)
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[5.0]))).unwrap(),
        ));

        let attacker = PgdAttacker::new(PgdConfig::fast());

        // Very tight bounds: [0.499, 0.501]
        let input_bounds =
            BoundedTensor::new(arr1(&[0.499_f32]).into_dyn(), arr1(&[0.501_f32]).into_dyn())
                .unwrap();

        let result = attacker
            .attack(&network, &input_bounds, 0, 5.5, true)
            .unwrap();

        // y = x + 5, so y in [5.499, 5.501]. Looking for y >= 5.5 should find it.
        assert!(result.found_counterexample);
    }

    #[test]
    fn test_pgd_reproducibility_with_seed() {
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
        ));

        let input_bounds =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

        // Run same attack twice with same seed
        let config = PgdConfig {
            num_restarts: 5,
            num_steps: 10,
            step_size: 0.1,
            spsa_delta: 0.01,
            seed: 12345,
            parallel: false,
        };

        let attacker1 = PgdAttacker::new(config.clone());
        let result1 = attacker1
            .attack(&network, &input_bounds, 0, 0.0, true)
            .unwrap();

        let attacker2 = PgdAttacker::new(config);
        let result2 = attacker2
            .attack(&network, &input_bounds, 0, 0.0, true)
            .unwrap();

        // Same seed should give same result
        assert!(
            (result1.best_output_value - result2.best_output_value).abs() < 1e-6,
            "Same seed should produce reproducible results"
        );
    }
}
