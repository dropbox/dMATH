//! SDP-CROWN utilities for tighter LiRPA bounds under ℓ2 input sets.
//!
//! This module implements the key offset tightening from:
//!   SDP-CROWN: Efficient Bound Propagation for Neural Network Verification
//!   with Tightness of Semidefinite Programming (arXiv:2506.06665)
//!
//! We currently implement the ReLU-specific offset `h(g, λ)` (Theorem 1) used to
//! convert a standard LiRPA/CROWN linear relaxation (computed on an ℓ∞ box
//! containing the ℓ2 ball) into a valid and often tighter relaxation for the
//! ℓ2 ball directly.

use gamma_core::{GammaError, Result};

const MIN_LAMBDA: f64 = 1e-8;
const MAX_LAMBDA: f64 = 1e8;

fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn dot(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(GammaError::shape_mismatch(vec![a.len()], vec![b.len()]));
    }
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum())
}

fn l2_norm_sq(x: &[f32]) -> f64 {
    x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()
}

fn phi_norm_sq(c: &[f32], g: &[f32], x_hat: &[f32], lambda: f64) -> Result<f64> {
    if c.len() != g.len() {
        return Err(GammaError::shape_mismatch(vec![c.len()], vec![g.len()]));
    }
    if c.len() != x_hat.len() {
        return Err(GammaError::shape_mismatch(vec![c.len()], vec![x_hat.len()]));
    }
    let mut sum = 0.0f64;
    for i in 0..c.len() {
        let ci = c[i] as f64;
        let gi = g[i] as f64;
        let xhi = x_hat[i] as f64;

        // φ_i(g,λ) = min{c_i - g_i - λ x̂_i, g_i + λ x̂_i, 0}
        let a = ci - gi - lambda * xhi;
        let b = gi + lambda * xhi;
        let phi = a.min(b).min(0.0);
        sum += phi * phi;
    }
    Ok(sum)
}

fn phi0_l2_norm(c: &[f32], g: &[f32]) -> Result<f64> {
    if c.len() != g.len() {
        return Err(GammaError::shape_mismatch(vec![c.len()], vec![g.len()]));
    }
    // When x̂ = 0, φ is independent of λ:
    // φ_i = min{c_i - g_i, g_i, 0}
    let mut sum = 0.0f64;
    for i in 0..c.len() {
        let a = (c[i] - g[i]) as f64;
        let b = g[i] as f64;
        let phi = a.min(b).min(0.0);
        sum += phi * phi;
    }
    Ok(sum.sqrt())
}

/// Compute SDP-CROWN ReLU offset `h(g, λ)` for `c^T ReLU(x) >= g^T x + h` on an ℓ2 ball.
///
/// The ball is `B2(x_hat, rho) = { x : ||x - x_hat||_2 <= rho }`.
pub fn relu_sdp_offset_for_lambda(
    c: &[f32],
    g: &[f32],
    x_hat: &[f32],
    rho: f32,
    lambda: f64,
) -> Result<f32> {
    if rho < 0.0 {
        return Err(GammaError::InvalidSpec(format!(
            "SDP-CROWN: rho must be >= 0 (got {rho})"
        )));
    }
    let lambda = lambda.clamp(MIN_LAMBDA, MAX_LAMBDA);
    let rho2_minus_xhat2 = (rho as f64) * (rho as f64) - l2_norm_sq(x_hat);
    let phi2 = phi_norm_sq(c, g, x_hat, lambda)?;
    let h = -0.5f64 * (lambda * rho2_minus_xhat2 + phi2 / lambda);
    Ok(h as f32)
}

/// Compute a near-optimal SDP-CROWN ReLU offset by maximizing over λ.
///
/// - If `rho == 0`, returns the exact offset for the singleton set `{x_hat}`.
/// - If `x_hat == 0`, uses the closed-form optimum `h* = -rho * ||min{c-g, g, 0}||_2`.
/// - Otherwise, uses a log-spaced grid search over λ (robust and fast enough for small nets).
pub fn relu_sdp_offset_opt(c: &[f32], g: &[f32], x_hat: &[f32], rho: f32) -> Result<f32> {
    if rho == 0.0 {
        let lhs = dot(c, &x_hat.iter().copied().map(relu).collect::<Vec<_>>())?;
        let rhs = dot(g, x_hat)?;
        return Ok(lhs - rhs);
    }

    let xhat_norm_sq = l2_norm_sq(x_hat);
    if xhat_norm_sq < 1e-20 {
        let phi_norm = phi0_l2_norm(c, g)?;
        return Ok(-(rho as f64 * phi_norm) as f32);
    }

    let phi0 = phi0_l2_norm(c, g)?;
    let base = if rho > 1e-12 && phi0 > 1e-12 {
        (phi0 / rho as f64).clamp(MIN_LAMBDA, MAX_LAMBDA)
    } else {
        1.0
    };

    let min_lambda = (base * 1e-3).clamp(MIN_LAMBDA, MAX_LAMBDA);
    let max_lambda = (base * 1e3).clamp(min_lambda, MAX_LAMBDA);

    // 41-point log grid (cheap, robust for piecewise objective).
    let steps = 41usize;
    let log_min = min_lambda.ln();
    let log_max = max_lambda.ln();
    let mut best_h = f64::NEG_INFINITY;
    for t in 0..steps {
        let frac = if steps == 1 {
            0.0
        } else {
            t as f64 / (steps as f64 - 1.0)
        };
        let lambda = (log_min + frac * (log_max - log_min)).exp();
        let h = relu_sdp_offset_for_lambda(c, g, x_hat, rho, lambda)? as f64;
        if h.is_finite() && h > best_h {
            best_h = h;
        }
    }

    Ok(best_h as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============== relu tests ==============

    #[test]
    fn test_relu_positive() {
        assert_eq!(relu(5.0), 5.0);
        assert_eq!(relu(0.1), 0.1);
        assert_eq!(relu(100.0), 100.0);
    }

    #[test]
    fn test_relu_negative() {
        assert_eq!(relu(-5.0), 0.0);
        assert_eq!(relu(-0.1), 0.0);
        assert_eq!(relu(-100.0), 0.0);
    }

    #[test]
    fn test_relu_zero() {
        assert_eq!(relu(0.0), 0.0);
    }

    // ============== dot tests ==============

    #[test]
    fn test_dot_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = dot(&a, &b).unwrap();
        assert!((result - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_dot_zeros() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 3.0];
        let result = dot(&a, &b).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_dot_negative() {
        let a = [1.0, -2.0, 3.0];
        let b = [-1.0, 2.0, -3.0];
        let result = dot(&a, &b).unwrap();
        assert!((result - (-14.0)).abs() < 1e-6); // 1*(-1) + (-2)*2 + 3*(-3) = -14
    }

    #[test]
    fn test_dot_single_element() {
        let a = [3.0];
        let b = [4.0];
        let result = dot(&a, &b).unwrap();
        assert!((result - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_empty() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        let result = dot(&a, &b).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_dot_mismatched_lengths() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0];
        let result = dot(&a, &b);
        assert!(result.is_err());
    }

    // ============== l2_norm_sq tests ==============

    #[test]
    fn test_l2_norm_sq_basic() {
        let x = [3.0, 4.0];
        let result = l2_norm_sq(&x);
        assert!((result - 25.0).abs() < 1e-10); // 9 + 16 = 25
    }

    #[test]
    fn test_l2_norm_sq_zeros() {
        let x = [0.0, 0.0, 0.0];
        let result = l2_norm_sq(&x);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_l2_norm_sq_single() {
        let x = [5.0];
        let result = l2_norm_sq(&x);
        assert!((result - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_norm_sq_negative() {
        let x = [-3.0, 4.0];
        let result = l2_norm_sq(&x);
        assert!((result - 25.0).abs() < 1e-10); // Square makes positive
    }

    #[test]
    fn test_l2_norm_sq_empty() {
        let x: [f32; 0] = [];
        let result = l2_norm_sq(&x);
        assert_eq!(result, 0.0);
    }

    // ============== phi_norm_sq tests ==============

    #[test]
    fn test_phi_norm_sq_basic() {
        let c = [1.0, 1.0];
        let g = [0.5, 0.5];
        let x_hat = [0.0, 0.0];
        let lambda = 1.0;
        let result = phi_norm_sq(&c, &g, &x_hat, lambda).unwrap();
        // φ_i = min{c_i - g_i - λx̂_i, g_i + λx̂_i, 0} = min{0.5, 0.5, 0} = 0
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_phi_norm_sq_negative_phi() {
        let c = [0.0]; // c=0, g=1 -> min{0-1-0, 1+0, 0} = min{-1, 1, 0} = -1
        let g = [1.0];
        let x_hat = [0.0];
        let lambda = 1.0;
        let result = phi_norm_sq(&c, &g, &x_hat, lambda).unwrap();
        assert!((result - 1.0).abs() < 1e-10); // (-1)^2 = 1
    }

    #[test]
    fn test_phi_norm_sq_mismatched_c_g() {
        let c = [1.0, 2.0, 3.0];
        let g = [1.0, 2.0];
        let x_hat = [0.0, 0.0];
        let result = phi_norm_sq(&c, &g, &x_hat, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_phi_norm_sq_mismatched_xhat() {
        let c = [1.0, 2.0];
        let g = [1.0, 2.0];
        let x_hat = [0.0, 0.0, 0.0];
        let result = phi_norm_sq(&c, &g, &x_hat, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_phi_norm_sq_with_nonzero_xhat() {
        let c = [2.0];
        let g = [1.0];
        let x_hat = [0.5];
        let lambda = 2.0;
        // φ = min{2 - 1 - 2*0.5, 1 + 2*0.5, 0} = min{0, 2, 0} = 0
        let result = phi_norm_sq(&c, &g, &x_hat, lambda).unwrap();
        assert!((result - 0.0).abs() < 1e-10);
    }

    // ============== phi0_l2_norm tests ==============

    #[test]
    fn test_phi0_l2_norm_basic() {
        let c = [1.0, 1.0];
        let g = [0.5, 0.5];
        // φ_i = min{c_i - g_i, g_i, 0} = min{0.5, 0.5, 0} = 0
        let result = phi0_l2_norm(&c, &g).unwrap();
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_phi0_l2_norm_negative_phi() {
        let c = [0.0];
        let g = [1.0];
        // φ = min{0-1, 1, 0} = -1
        let result = phi0_l2_norm(&c, &g).unwrap();
        assert!((result - 1.0).abs() < 1e-10); // sqrt(1) = 1
    }

    #[test]
    fn test_phi0_l2_norm_multiple() {
        let c = [0.0, 0.0];
        let g = [1.0, 1.0];
        // Each φ_i = min{-1, 1, 0} = -1
        // ||φ||_2 = sqrt(1 + 1) = sqrt(2)
        let result = phi0_l2_norm(&c, &g).unwrap();
        assert!((result - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_phi0_l2_norm_mismatched() {
        let c = [1.0, 2.0, 3.0];
        let g = [1.0, 2.0];
        let result = phi0_l2_norm(&c, &g);
        assert!(result.is_err());
    }

    // ============== relu_sdp_offset_for_lambda tests ==============

    #[test]
    fn test_relu_sdp_offset_for_lambda_basic() {
        let c = [1.0, 1.0];
        let g = [0.5, 0.5];
        let x_hat = [0.0, 0.0];
        let rho = 0.1;
        let lambda = 1.0;
        let result = relu_sdp_offset_for_lambda(&c, &g, &x_hat, rho, lambda).unwrap();
        // Should return a finite value
        assert!(result.is_finite());
    }

    #[test]
    fn test_relu_sdp_offset_for_lambda_negative_rho() {
        let c = [1.0];
        let g = [0.5];
        let x_hat = [0.0];
        let rho = -0.1;
        let result = relu_sdp_offset_for_lambda(&c, &g, &x_hat, rho, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_relu_sdp_offset_for_lambda_zero_rho() {
        let c = [1.0, 1.0];
        let g = [0.5, 0.5];
        let x_hat = [0.0, 0.0];
        let rho = 0.0;
        let lambda = 1.0;
        let result = relu_sdp_offset_for_lambda(&c, &g, &x_hat, rho, lambda).unwrap();
        // With rho=0 and x_hat=0: rho^2 - ||x_hat||^2 = 0, phi^2/λ ≥ 0
        // h = -0.5 * (λ * 0 + phi^2/λ) ≤ 0
        assert!(result <= 0.0001);
    }

    #[test]
    fn test_relu_sdp_offset_for_lambda_lambda_clamping_min() {
        let c = [1.0];
        let g = [0.5];
        let x_hat = [0.0];
        let rho = 0.1;
        // Very small lambda should be clamped to MIN_LAMBDA
        let result = relu_sdp_offset_for_lambda(&c, &g, &x_hat, rho, 1e-20).unwrap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_relu_sdp_offset_for_lambda_lambda_clamping_max() {
        let c = [1.0];
        let g = [0.5];
        let x_hat = [0.0];
        let rho = 0.1;
        // Very large lambda should be clamped to MAX_LAMBDA
        let result = relu_sdp_offset_for_lambda(&c, &g, &x_hat, rho, 1e20).unwrap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_relu_sdp_offset_for_lambda_mismatched_dims() {
        let c = [1.0, 2.0];
        let g = [0.5];
        let x_hat = [0.0, 0.0];
        let result = relu_sdp_offset_for_lambda(&c, &g, &x_hat, 0.1, 1.0);
        assert!(result.is_err());
    }

    // ============== relu_sdp_offset_opt tests ==============

    #[test]
    fn test_relu_sdp_offset_opt_rho_zero() {
        let c = [1.0, 0.0];
        let g = [0.5, 0.5];
        let x_hat = [2.0, -1.0];
        // rho=0 path: lhs = c · ReLU(x_hat) = 1*2 + 0*0 = 2
        // rhs = g · x_hat = 0.5*2 + 0.5*(-1) = 0.5
        // result = 2 - 0.5 = 1.5
        let result = relu_sdp_offset_opt(&c, &g, &x_hat, 0.0).unwrap();
        assert!((result - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_relu_sdp_offset_opt_xhat_zero() {
        let c = [0.0];
        let g = [1.0];
        let x_hat = [0.0];
        let rho = 1.0;
        // x_hat = 0 path: h* = -rho * ||min{c-g, g, 0}||_2
        // φ_i = min{0-1, 1, 0} = -1
        // ||φ||_2 = 1
        // h* = -1 * 1 = -1
        let result = relu_sdp_offset_opt(&c, &g, &x_hat, rho).unwrap();
        assert!((result - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_relu_sdp_offset_opt_general() {
        let c = [1.0, 1.0];
        let g = [0.5, 0.5];
        let x_hat = [0.1, 0.1];
        let rho = 0.5;
        let result = relu_sdp_offset_opt(&c, &g, &x_hat, rho).unwrap();
        // Should return finite value from optimization
        assert!(result.is_finite());
    }

    #[test]
    fn test_relu_sdp_offset_opt_consistency() {
        // Result from opt should be >= any specific lambda
        let c = [1.0, 0.5];
        let g = [0.3, 0.4];
        let x_hat = [0.2, 0.2];
        let rho = 0.3;

        let opt_result = relu_sdp_offset_opt(&c, &g, &x_hat, rho).unwrap();

        // Check that opt_result is >= result for several lambda values
        for lambda in [0.1, 1.0, 10.0, 100.0] {
            let specific = relu_sdp_offset_for_lambda(&c, &g, &x_hat, rho, lambda).unwrap();
            assert!(
                opt_result >= specific - 1e-5,
                "opt {} should be >= specific {} for lambda {}",
                opt_result,
                specific,
                lambda
            );
        }
    }

    #[test]
    fn test_relu_sdp_offset_opt_mismatched() {
        let c = [1.0, 2.0];
        let g = [0.5];
        let x_hat = [0.0, 0.0];
        let result = relu_sdp_offset_opt(&c, &g, &x_hat, 0.1);
        assert!(result.is_err());
    }

    // ============== Mathematical property tests ==============

    #[test]
    fn test_offset_nonpositive_property() {
        // For valid ReLU relaxation, offset should typically be non-positive
        // (it's a correction term that tightens bounds)
        let c = [1.0, 1.0, 1.0];
        let g = [0.5, 0.5, 0.5];
        let x_hat = [0.0, 0.0, 0.0];

        for rho in [0.1, 0.5, 1.0, 2.0] {
            let result = relu_sdp_offset_opt(&c, &g, &x_hat, rho).unwrap();
            assert!(
                result <= 0.001,
                "Offset {} should be non-positive for rho={}",
                result,
                rho
            );
        }
    }

    #[test]
    fn test_offset_monotonic_in_rho() {
        // Larger rho (larger uncertainty ball) should give more negative offset
        let c = [0.0, 0.0];
        let g = [1.0, 1.0];
        let x_hat = [0.0, 0.0];

        let offset_small = relu_sdp_offset_opt(&c, &g, &x_hat, 0.1).unwrap();
        let offset_large = relu_sdp_offset_opt(&c, &g, &x_hat, 1.0).unwrap();

        assert!(
            offset_large <= offset_small + 1e-5,
            "Larger rho should give smaller (more negative) offset: {} vs {}",
            offset_large,
            offset_small
        );
    }

    #[test]
    fn test_singleton_set_exact() {
        // When rho=0, the set is a singleton {x_hat}
        // The offset should equal c^T ReLU(x_hat) - g^T x_hat exactly
        let c = [1.0, 2.0];
        let g = [0.5, 0.5];
        let x_hat = [3.0, -1.0]; // ReLU(x_hat) = [3, 0]

        let result = relu_sdp_offset_opt(&c, &g, &x_hat, 0.0).unwrap();
        // c^T ReLU(x_hat) = 1*3 + 2*0 = 3
        // g^T x_hat = 0.5*3 + 0.5*(-1) = 1
        // offset = 3 - 1 = 2
        assert!((result - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_centered_ball_closed_form() {
        // When x_hat = 0, there's a closed-form solution
        // h* = -rho * ||min{c-g, g, 0}||_2
        let c = [2.0, 0.0];
        let g = [1.0, 1.0];
        let x_hat = [0.0, 0.0];
        let rho = 2.0;

        // φ_0 = min{2-1, 1, 0} = 0
        // φ_1 = min{0-1, 1, 0} = -1
        // ||φ||_2 = sqrt(0 + 1) = 1
        // h* = -2 * 1 = -2

        let result = relu_sdp_offset_opt(&c, &g, &x_hat, rho).unwrap();
        assert!((result - (-2.0)).abs() < 1e-5);
    }

    #[test]
    fn test_small_xhat_norm() {
        // Very small x_hat should use the x_hat=0 branch
        let c = [1.0];
        let g = [0.5];
        let x_hat = [1e-15];
        let rho = 1.0;

        // Should not fail due to numerical issues
        let result = relu_sdp_offset_opt(&c, &g, &x_hat, rho).unwrap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_empty_inputs() {
        let c: [f32; 0] = [];
        let g: [f32; 0] = [];
        let x_hat: [f32; 0] = [];

        let result = relu_sdp_offset_opt(&c, &g, &x_hat, 1.0).unwrap();
        // Empty case: rho^2 - 0 > 0, phi = 0
        // h = -0.5 * (λ * rho^2 + 0) < 0
        assert!(result.is_finite());
    }
}
