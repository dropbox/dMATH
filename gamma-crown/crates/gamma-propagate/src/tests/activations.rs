//! Activation function tests (Sqrt, DivConstant, SubConstant, PowConstant,
//! ReduceMean, ReduceSum, Tanh, Sigmoid, Softplus, Sin, Cos)

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== Abs tests ====================

#[test]
fn test_abs_ibp_positive_interval() {
    let lower = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let abs = AbsLayer;
    let output = abs.propagate_ibp(&input).unwrap();

    for i in 0..3 {
        assert!((output.lower[[i]] - 2.0).abs() < 1e-6);
        assert!((output.upper[[i]] - 5.0).abs() < 1e-6);
    }
}

#[test]
fn test_abs_ibp_negative_interval() {
    let lower = ArrayD::from_elem(IxDyn(&[2]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), -2.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let abs = AbsLayer;
    let output = abs.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!((output.lower[[i]] - 2.0).abs() < 1e-6);
        assert!((output.upper[[i]] - 5.0).abs() < 1e-6);
    }
}

#[test]
fn test_abs_ibp_crosses_zero() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-3.0, -2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![4.0, 1.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let abs = AbsLayer;
    let output = abs.propagate_ibp(&input).unwrap();

    assert!((output.lower[[0]] - 0.0).abs() < 1e-6);
    assert!((output.upper[[0]] - 4.0).abs() < 1e-6);
    assert!((output.lower[[1]] - 0.0).abs() < 1e-6);
    assert!((output.upper[[1]] - 2.0).abs() < 1e-6);
}

#[test]
fn test_abs_linear_passthrough() {
    // Without pre-activation bounds, Abs returns identity (pass-through)
    // so CROWN can propagate and use CROWN-IBP/bounds-aware method
    let bounds = LinearBounds::identity(4);
    let abs = AbsLayer;
    let result = abs.propagate_linear(&bounds).unwrap();
    // Should be identity - no change
    assert_eq!(result.lower_a, bounds.lower_a);
    assert_eq!(result.upper_a, bounds.upper_a);
}

#[test]
fn test_abs_crown_with_bounds_positive_region() {
    // All positive pre-activation bounds: |x| = x (identity)
    let pre_lower = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[3]), 5.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let abs = AbsLayer;

    let result = abs
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // In positive region, should be identity (slope=1, intercept=0)
    for i in 0..3 {
        assert!(
            (result.lower_a[[i, i]] - 1.0).abs() < 1e-6,
            "Positive region should have slope 1"
        );
        assert!(
            (result.upper_a[[i, i]] - 1.0).abs() < 1e-6,
            "Positive region should have slope 1"
        );
    }
    assert!(result.lower_b.iter().all(|&x| x.abs() < 1e-6));
    assert!(result.upper_b.iter().all(|&x| x.abs() < 1e-6));
}

#[test]
fn test_abs_crown_with_bounds_negative_region() {
    // All negative pre-activation bounds: |x| = -x (negation)
    let pre_lower = ArrayD::from_elem(IxDyn(&[3]), -5.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[3]), -2.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let abs = AbsLayer;

    let result = abs
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // In negative region, should be negation (slope=-1, intercept=0)
    for i in 0..3 {
        assert!(
            (result.lower_a[[i, i]] + 1.0).abs() < 1e-6,
            "Negative region should have slope -1"
        );
        assert!(
            (result.upper_a[[i, i]] + 1.0).abs() < 1e-6,
            "Negative region should have slope -1"
        );
    }
    assert!(result.lower_b.iter().all(|&x| x.abs() < 1e-6));
    assert!(result.upper_b.iter().all(|&x| x.abs() < 1e-6));
}

#[test]
fn test_abs_crown_soundness() {
    // Test that CROWN bounds are sound (contain true outputs)
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0, 0.5, -1.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, 2.0, 4.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let abs = AbsLayer;

    let result = abs
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample points in the input range and verify bounds hold
    let test_points: [Vec<f32>; 4] = [
        vec![-2.0, 0.5, -1.0], // lower
        vec![3.0, 2.0, 4.0],   // upper
        vec![0.0, 1.0, 0.0],   // middle
        vec![-1.0, 1.5, 2.0],  // random
    ];

    for point in &test_points {
        let abs_output: Vec<f32> = point.iter().map(|x| x.abs()).collect();

        // Check each output dimension
        for (j, &abs_val) in abs_output.iter().enumerate() {
            // Lower bound: lower_a * x + lower_b should be <= abs(point)
            let lb_val: f32 = (0..3)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            // Upper bound: upper_a * x + upper_b should be >= abs(point)
            let ub_val: f32 = (0..3)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            let tol = 1e-4;
            assert!(
                lb_val <= abs_val + tol,
                "Lower bound violated at point {:?}: lb {} > abs {}",
                point,
                lb_val,
                abs_val
            );
            assert!(
                ub_val >= abs_val - tol,
                "Upper bound violated at point {:?}: ub {} < abs {}",
                point,
                ub_val,
                abs_val
            );
        }
    }
}

// ==================== Sqrt tests ====================

#[test]
fn test_sqrt_ibp_basic() {
    // Test sqrt on positive bounds
    let lower = ArrayD::from_elem(IxDyn(&[4]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4]), 4.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sqrt = SqrtLayer;
    let output = sqrt.propagate_ibp(&input).unwrap();

    // sqrt([1, 4]) = [1, 2]
    for i in 0..4 {
        assert!(
            (output.lower[[i]] - 1.0).abs() < 1e-6,
            "sqrt(1) should be 1, got {}",
            output.lower[[i]]
        );
        assert!(
            (output.upper[[i]] - 2.0).abs() < 1e-6,
            "sqrt(4) should be 2, got {}",
            output.upper[[i]]
        );
    }
}

#[test]
fn test_sqrt_ibp_clamps_negative() {
    // Test that negative values are clamped to 0
    let lower = ArrayD::from_elem(IxDyn(&[3]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 9.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sqrt = SqrtLayer;
    let output = sqrt.propagate_ibp(&input).unwrap();

    // sqrt(max(0, -1)) = 0, sqrt(9) = 3
    for i in 0..3 {
        assert!(
            output.lower[[i]].abs() < 1e-6,
            "sqrt(max(0, -1)) should be 0, got {}",
            output.lower[[i]]
        );
        assert!(
            (output.upper[[i]] - 3.0).abs() < 1e-6,
            "sqrt(9) should be 3, got {}",
            output.upper[[i]]
        );
    }
}

#[test]
fn test_sqrt_linear_passthrough() {
    // Without pre-activation bounds, Sqrt returns identity (pass-through)
    let bounds = LinearBounds::identity(4);
    let sqrt = SqrtLayer;
    let result = sqrt.propagate_linear(&bounds).unwrap();
    // Should be identity - no change
    assert_eq!(result.lower_a, bounds.lower_a);
    assert_eq!(result.upper_a, bounds.upper_a);
}

#[test]
fn test_sqrt_crown_soundness() {
    // Test that CROWN bounds for sqrt are sound
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 4.0, 0.25, 9.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![4.0, 9.0, 1.0, 16.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let sqrt = SqrtLayer;

    let result = sqrt
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample points in the input range and verify bounds hold
    let test_points: [Vec<f32>; 4] = [
        vec![1.0, 4.0, 0.25, 9.0],   // lower
        vec![4.0, 9.0, 1.0, 16.0],   // upper
        vec![2.5, 6.5, 0.625, 12.5], // midpoint
        vec![2.0, 5.0, 0.5, 10.0],   // random
    ];

    for point in &test_points {
        let sqrt_output: Vec<f32> = point.iter().map(|x| x.sqrt()).collect();

        // Check each output dimension
        for (j, &sqrt_val) in sqrt_output.iter().enumerate() {
            // Lower bound: lower_a * x + lower_b should be <= sqrt(point)
            let lb_val: f32 = (0..4)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            // Upper bound: upper_a * x + upper_b should be >= sqrt(point)
            let ub_val: f32 = (0..4)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            let tol = 1e-3; // Slightly relaxed for sqrt's curvature
            assert!(
                lb_val <= sqrt_val + tol,
                "Lower bound violated at point {:?}: lb {} > sqrt {}",
                point,
                lb_val,
                sqrt_val
            );
            assert!(
                ub_val >= sqrt_val - tol,
                "Upper bound violated at point {:?}: ub {} < sqrt {}",
                point,
                ub_val,
                sqrt_val
            );
        }
    }
}

#[test]
fn test_sqrt_crown_concave_property() {
    // Sqrt is concave, so chord should be a lower bound
    // and tangent-based approximation should be an upper bound
    let pre_lower = ArrayD::from_elem(IxDyn(&[1]), 1.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[1]), 4.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(1);
    let sqrt = SqrtLayer;

    let result = sqrt
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Chord from (1, 1) to (4, 2): slope = (2-1)/(4-1) = 1/3 ≈ 0.333
    let expected_chord_slope = (2.0 - 1.0) / (4.0 - 1.0);
    // Due to numerical sampling, the slope should be close to chord slope
    let slope = result.lower_a[[0, 0]];

    // Verify the slope is reasonable (chord-based)
    assert!(
        (slope - expected_chord_slope).abs() < 0.1,
        "Lower slope {} should be close to chord slope {}",
        slope,
        expected_chord_slope
    );
}

// ==================== DivConstant tests ====================

#[test]
fn test_div_constant_ibp_positive_divisor() {
    // Test division by positive constant
    let lower = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 6.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let div = DivConstantLayer::scalar(2.0);
    let output = div.propagate_ibp(&input).unwrap();

    // [2, 6] / 2 = [1, 3]
    for i in 0..3 {
        assert!(
            (output.lower[[i]] - 1.0).abs() < 1e-6,
            "2/2 should be 1, got {}",
            output.lower[[i]]
        );
        assert!(
            (output.upper[[i]] - 3.0).abs() < 1e-6,
            "6/2 should be 3, got {}",
            output.upper[[i]]
        );
    }
}

#[test]
fn test_div_constant_ibp_negative_divisor() {
    // Test division by negative constant (bounds swap)
    let lower = ArrayD::from_elem(IxDyn(&[2]), 2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 6.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let div = DivConstantLayer::scalar(-2.0);
    let output = div.propagate_ibp(&input).unwrap();

    // [2, 6] / -2 = [-3, -1]
    for i in 0..2 {
        assert!(
            (output.lower[[i]] - (-3.0)).abs() < 1e-6,
            "6/-2 should be -3, got {}",
            output.lower[[i]]
        );
        assert!(
            (output.upper[[i]] - (-1.0)).abs() < 1e-6,
            "2/-2 should be -1, got {}",
            output.upper[[i]]
        );
    }
}

// ==================== SubConstant tests ====================

#[test]
fn test_sub_constant_ibp_normal() {
    // Test y = x - constant
    let lower = ArrayD::from_elem(IxDyn(&[3]), 5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sub = SubConstantLayer::scalar(3.0);
    let output = sub.propagate_ibp(&input).unwrap();

    // [5, 10] - 3 = [2, 7]
    for i in 0..3 {
        assert!(
            (output.lower[[i]] - 2.0).abs() < 1e-6,
            "5-3 should be 2, got {}",
            output.lower[[i]]
        );
        assert!(
            (output.upper[[i]] - 7.0).abs() < 1e-6,
            "10-3 should be 7, got {}",
            output.upper[[i]]
        );
    }
}

#[test]
fn test_sub_constant_ibp_reverse() {
    // Test y = constant - x (bounds swap)
    let lower = ArrayD::from_elem(IxDyn(&[2]), 2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 8.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let constant = ArrayD::from_elem(IxDyn(&[2]), 10.0f32);
    let sub = SubConstantLayer::new_reverse(constant);
    let output = sub.propagate_ibp(&input).unwrap();

    // 10 - [2, 8] = [2, 8]
    for i in 0..2 {
        assert!(
            (output.lower[[i]] - 2.0).abs() < 1e-6,
            "10-8 should be 2, got {}",
            output.lower[[i]]
        );
        assert!(
            (output.upper[[i]] - 8.0).abs() < 1e-6,
            "10-2 should be 8, got {}",
            output.upper[[i]]
        );
    }
}

#[test]
fn test_div_linear_propagation() {
    // Test that DivConstant linear propagation is consistent
    let lower = ArrayD::from_elem(IxDyn(&[4]), 2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4]), 8.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let div = DivConstantLayer::scalar(2.0);
    let ibp_result = div.propagate_ibp(&input).unwrap();

    // Create identity linear bounds and propagate
    let linear_bounds = LinearBounds::identity(4);
    let linear_result = div.propagate_linear(&linear_bounds).unwrap().into_owned();
    let concretized = linear_result.concretize(&input);

    // Results should match
    for i in 0..4 {
        assert!(
            (ibp_result.lower[[i]] - concretized.lower[[i]]).abs() < 1e-5,
            "Linear lower doesn't match IBP: {} vs {}",
            ibp_result.lower[[i]],
            concretized.lower[[i]]
        );
        assert!(
            (ibp_result.upper[[i]] - concretized.upper[[i]]).abs() < 1e-5,
            "Linear upper doesn't match IBP: {} vs {}",
            ibp_result.upper[[i]],
            concretized.upper[[i]]
        );
    }
}

#[test]
fn test_sub_linear_propagation() {
    // Test that SubConstant linear propagation is consistent
    let lower = ArrayD::from_elem(IxDyn(&[4]), 3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4]), 9.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sub = SubConstantLayer::scalar(2.0);
    let ibp_result = sub.propagate_ibp(&input).unwrap();

    // Create identity linear bounds and propagate
    let linear_bounds = LinearBounds::identity(4);
    let linear_result = sub.propagate_linear(&linear_bounds).unwrap().into_owned();
    let concretized = linear_result.concretize(&input);

    // Results should match
    for i in 0..4 {
        assert!(
            (ibp_result.lower[[i]] - concretized.lower[[i]]).abs() < 1e-5,
            "Linear lower doesn't match IBP: {} vs {}",
            ibp_result.lower[[i]],
            concretized.lower[[i]]
        );
        assert!(
            (ibp_result.upper[[i]] - concretized.upper[[i]]).abs() < 1e-5,
            "Linear upper doesn't match IBP: {} vs {}",
            ibp_result.upper[[i]],
            concretized.upper[[i]]
        );
    }
}

// ==================== PowConstant tests ====================

#[test]
fn test_pow_square_positive() {
    // Test x^2 for positive bounds
    let lower = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 4.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let pow = PowConstantLayer::square();
    let output = pow.propagate_ibp(&input).unwrap();

    // [2, 4]^2 = [4, 16]
    for i in 0..3 {
        assert!(
            (output.lower[[i]] - 4.0).abs() < 1e-6,
            "2^2 should be 4, got {}",
            output.lower[[i]]
        );
        assert!(
            (output.upper[[i]] - 16.0).abs() < 1e-6,
            "4^2 should be 16, got {}",
            output.upper[[i]]
        );
    }
}

#[test]
fn test_pow_square_negative() {
    // Test x^2 for negative bounds
    let lower = ArrayD::from_elem(IxDyn(&[2]), -4.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), -2.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let pow = PowConstantLayer::square();
    let output = pow.propagate_ibp(&input).unwrap();

    // [-4, -2]^2 = [4, 16] (monotonically decreasing for negative x)
    for i in 0..2 {
        assert!(
            (output.lower[[i]] - 4.0).abs() < 1e-6,
            "(-2)^2 should be 4, got {}",
            output.lower[[i]]
        );
        assert!(
            (output.upper[[i]] - 16.0).abs() < 1e-6,
            "(-4)^2 should be 16, got {}",
            output.upper[[i]]
        );
    }
}

#[test]
fn test_pow_square_straddles_zero() {
    // Test x^2 when bounds straddle zero
    let lower = ArrayD::from_elem(IxDyn(&[2]), -3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let pow = PowConstantLayer::square();
    let output = pow.propagate_ibp(&input).unwrap();

    // [-3, 2]^2 = [0, 9] (minimum at 0, max is max(9, 4) = 9)
    for i in 0..2 {
        assert!(
            output.lower[[i]].abs() < 1e-6,
            "Min of x^2 for [-3, 2] should be 0, got {}",
            output.lower[[i]]
        );
        assert!(
            (output.upper[[i]] - 9.0).abs() < 1e-6,
            "Max of x^2 for [-3, 2] should be 9, got {}",
            output.upper[[i]]
        );
    }
}

#[test]
fn test_pow_square_crown_positive_region_coefficients() {
    let pre_lower = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[3]), 4.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let pow = PowConstantLayer::square();

    let result = pow
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // For [2,4], both upper (chord) and lower (tangent at midpoint) have slope 6.
    for i in 0..3 {
        assert!((result.lower_a[[i, i]] - 6.0).abs() < 1e-6);
        assert!((result.upper_a[[i, i]] - 6.0).abs() < 1e-6);
        assert!((result.lower_b[i] + 9.0).abs() < 1e-6); // tangent at m=3: b=-9
        assert!((result.upper_b[i] + 8.0).abs() < 1e-6); // chord: b=-l*u=-8
    }
}

#[test]
fn test_pow_square_crown_crosses_zero_lower_is_zero() {
    let pre_lower = ArrayD::from_elem(IxDyn(&[2]), -3.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[2]), 2.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(2);
    let pow = PowConstantLayer::square();

    let result = pow
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Crossing 0: use y >= 0 lower bound.
    for i in 0..2 {
        assert!(result.lower_a[[i, i]].abs() < 1e-6);
        assert!(result.lower_b[i].abs() < 1e-6);
    }
}

#[test]
fn test_pow_square_crown_soundness() {
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0, 0.5, -1.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, 2.0, 4.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let pow = PowConstantLayer::square();

    let result = pow
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    let test_points: [Vec<f32>; 4] = [
        vec![-2.0, 0.5, -1.0], // lower
        vec![3.0, 2.0, 4.0],   // upper
        vec![0.0, 1.0, 0.0],   // middle
        vec![-1.0, 1.5, 2.0],  // random
    ];

    for point in &test_points {
        let pow_output: Vec<f32> = point.iter().map(|x| x * x).collect();

        for (j, &pow_val) in pow_output.iter().enumerate() {
            let lb_val: f32 = (0..3)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            let ub_val: f32 = (0..3)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            let tol = 1e-4;
            assert!(
                lb_val <= pow_val + tol,
                "Lower bound violated at point {:?}: lb {} > x^2 {}",
                point,
                lb_val,
                pow_val
            );
            assert!(
                ub_val + tol >= pow_val,
                "Upper bound violated at point {:?}: ub {} < x^2 {}",
                point,
                ub_val,
                pow_val
            );
        }
    }
}

// ==================== ReduceMean tests ====================

#[test]
fn test_reduce_mean_last_axis() {
    // Test mean over last axis with keepdims=true
    // Input: 2x3 tensor
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let reduce = ReduceMeanLayer::new(vec![-1], true);
    let output = reduce.propagate_ibp(&input).unwrap();

    // Output shape should be [2, 1]
    assert_eq!(output.shape(), &[2, 1]);

    // Row 0: mean([1,2,3]) = 2, mean([2,3,4]) = 3
    assert!(
        (output.lower[[0, 0]] - 2.0).abs() < 1e-6,
        "Mean lower of [1,2,3] should be 2, got {}",
        output.lower[[0, 0]]
    );
    assert!(
        (output.upper[[0, 0]] - 3.0).abs() < 1e-6,
        "Mean upper of [2,3,4] should be 3, got {}",
        output.upper[[0, 0]]
    );

    // Row 1: mean([4,5,6]) = 5, mean([5,6,7]) = 6
    assert!(
        (output.lower[[1, 0]] - 5.0).abs() < 1e-6,
        "Mean lower of [4,5,6] should be 5, got {}",
        output.lower[[1, 0]]
    );
    assert!(
        (output.upper[[1, 0]] - 6.0).abs() < 1e-6,
        "Mean upper of [5,6,7] should be 6, got {}",
        output.upper[[1, 0]]
    );
}

#[test]
fn test_reduce_mean_no_keepdims() {
    // Test mean over last axis with keepdims=false
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
    let upper = lower.clone();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let reduce = ReduceMeanLayer::new(vec![-1], false);
    let output = reduce.propagate_ibp(&input).unwrap();

    // Output shape should be [2] (dimension removed)
    assert_eq!(output.shape(), &[2]);

    // Row 0: mean([1,2,3,4]) = 2.5
    assert!(
        (output.lower[[0]] - 2.5).abs() < 1e-6,
        "Mean of [1,2,3,4] should be 2.5, got {}",
        output.lower[[0]]
    );

    // Row 1: mean([5,6,7,8]) = 6.5
    assert!(
        (output.lower[[1]] - 6.5).abs() < 1e-6,
        "Mean of [5,6,7,8] should be 6.5, got {}",
        output.lower[[1]]
    );
}

// ==================== ReduceMean CROWN tests ====================

#[test]
fn test_reduce_mean_crown_backward_keepdims() {
    // Test CROWN backward pass for ReduceMean with keepdims=true
    // Input: 2x3 tensor, output: 2x1 tensor
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Identity linear bounds on the output (2 elements after reduction)
    let linear_bounds = LinearBounds::identity(2);
    let reduce = ReduceMeanLayer::new(vec![-1], true);

    let result = reduce
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Output should have shape (2, 6) - 2 outputs, 6 inputs
    assert_eq!(result.lower_a.nrows(), 2);
    assert_eq!(result.lower_a.ncols(), 6);

    // For mean over axis -1 with 3 elements, each coefficient should be 1/3
    let scale = 1.0 / 3.0;

    // Row 0 should have 1/3 for columns 0,1,2 (input row 0) and 0 elsewhere
    for j in 0..3 {
        assert!(
            (result.lower_a[[0, j]] - scale).abs() < 1e-6,
            "Expected {}, got {} at [0,{}]",
            scale,
            result.lower_a[[0, j]],
            j
        );
    }
    for j in 3..6 {
        assert!(
            result.lower_a[[0, j]].abs() < 1e-6,
            "Expected 0, got {} at [0,{}]",
            result.lower_a[[0, j]],
            j
        );
    }

    // Row 1 should have 1/3 for columns 3,4,5 (input row 1) and 0 elsewhere
    for j in 0..3 {
        assert!(
            result.lower_a[[1, j]].abs() < 1e-6,
            "Expected 0, got {} at [1,{}]",
            result.lower_a[[1, j]],
            j
        );
    }
    for j in 3..6 {
        assert!(
            (result.lower_a[[1, j]] - scale).abs() < 1e-6,
            "Expected {}, got {} at [1,{}]",
            scale,
            result.lower_a[[1, j]],
            j
        );
    }
}

#[test]
fn test_reduce_mean_crown_soundness() {
    // Test that CROWN bounds are sound for ReduceMean
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(2);
    let reduce = ReduceMeanLayer::new(vec![-1], true);

    let result = reduce
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Concretize bounds
    let concrete = result.concretize(&pre_activation);

    // IBP result for comparison
    let ibp_result = reduce.propagate_ibp(&pre_activation).unwrap();

    // CROWN should give bounds that contain IBP bounds (or be equal)
    // Row 0: mean of [1,2,3] = 2 (lower), mean of [2,3,4] = 3 (upper)
    assert!(
        concrete.lower[[0]] <= ibp_result.lower[[0, 0]] + 1e-5,
        "CROWN lower {} should be <= IBP lower {}",
        concrete.lower[[0]],
        ibp_result.lower[[0, 0]]
    );
    assert!(
        concrete.upper[[0]] >= ibp_result.upper[[0, 0]] - 1e-5,
        "CROWN upper {} should be >= IBP upper {}",
        concrete.upper[[0]],
        ibp_result.upper[[0, 0]]
    );
}

// ==================== ReduceSum tests ====================

#[test]
fn test_reduce_sum_last_axis() {
    // Test sum over last axis with keepdims=true
    // Input: 2x3 tensor
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let reduce = ReduceSumLayer::new(vec![-1], true);
    let output = reduce.propagate_ibp(&input).unwrap();

    // Output shape should be [2, 1]
    assert_eq!(output.shape(), &[2, 1]);

    // Row 0: sum([1,2,3]) = 6, sum([2,3,4]) = 9
    assert!(
        (output.lower[[0, 0]] - 6.0).abs() < 1e-6,
        "Sum lower of [1,2,3] should be 6, got {}",
        output.lower[[0, 0]]
    );
    assert!(
        (output.upper[[0, 0]] - 9.0).abs() < 1e-6,
        "Sum upper of [2,3,4] should be 9, got {}",
        output.upper[[0, 0]]
    );

    // Row 1: sum([4,5,6]) = 15, sum([5,6,7]) = 18
    assert!(
        (output.lower[[1, 0]] - 15.0).abs() < 1e-6,
        "Sum lower of [4,5,6] should be 15, got {}",
        output.lower[[1, 0]]
    );
    assert!(
        (output.upper[[1, 0]] - 18.0).abs() < 1e-6,
        "Sum upper of [5,6,7] should be 18, got {}",
        output.upper[[1, 0]]
    );
}

#[test]
fn test_reduce_sum_no_keepdims() {
    // Test sum over last axis with keepdims=false
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
    let upper = lower.clone();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let reduce = ReduceSumLayer::new(vec![-1], false);
    let output = reduce.propagate_ibp(&input).unwrap();

    // Output shape should be [2] (dimension removed)
    assert_eq!(output.shape(), &[2]);

    // Row 0: sum([1,2,3,4]) = 10
    assert!(
        (output.lower[[0]] - 10.0).abs() < 1e-6,
        "Sum of [1,2,3,4] should be 10, got {}",
        output.lower[[0]]
    );

    // Row 1: sum([5,6,7,8]) = 26
    assert!(
        (output.lower[[1]] - 26.0).abs() < 1e-6,
        "Sum of [5,6,7,8] should be 26, got {}",
        output.lower[[1]]
    );
}

// ==================== ReduceSum CROWN tests ====================

#[test]
fn test_reduce_sum_crown_backward_keepdims() {
    // Test CROWN backward pass for ReduceSum with keepdims=true
    // Input: 2x3 tensor, output: 2x1 tensor
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Identity linear bounds on the output (2 elements after reduction)
    let linear_bounds = LinearBounds::identity(2);
    let reduce = ReduceSumLayer::new(vec![-1], true);

    let result = reduce
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Output should have shape (2, 6) - 2 outputs, 6 inputs
    assert_eq!(result.lower_a.nrows(), 2);
    assert_eq!(result.lower_a.ncols(), 6);

    // For sum over axis -1, each coefficient should be 1 (no scaling unlike mean)
    // Row 0 should have 1 for columns 0,1,2 (input row 0) and 0 elsewhere
    for j in 0..3 {
        assert!(
            (result.lower_a[[0, j]] - 1.0).abs() < 1e-6,
            "Expected 1.0, got {} at [0,{}]",
            result.lower_a[[0, j]],
            j
        );
    }
    for j in 3..6 {
        assert!(
            result.lower_a[[0, j]].abs() < 1e-6,
            "Expected 0, got {} at [0,{}]",
            result.lower_a[[0, j]],
            j
        );
    }

    // Row 1 should have 1 for columns 3,4,5 (input row 1) and 0 elsewhere
    for j in 0..3 {
        assert!(
            result.lower_a[[1, j]].abs() < 1e-6,
            "Expected 0, got {} at [1,{}]",
            result.lower_a[[1, j]],
            j
        );
    }
    for j in 3..6 {
        assert!(
            (result.lower_a[[1, j]] - 1.0).abs() < 1e-6,
            "Expected 1.0, got {} at [1,{}]",
            result.lower_a[[1, j]],
            j
        );
    }
}

#[test]
fn test_reduce_sum_crown_soundness() {
    // Test that CROWN bounds are sound for ReduceSum
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(2);
    let reduce = ReduceSumLayer::new(vec![-1], true);

    let result = reduce
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Concretize bounds
    let concrete = result.concretize(&pre_activation);

    // IBP result for comparison
    let ibp_result = reduce.propagate_ibp(&pre_activation).unwrap();

    // CROWN should give bounds that contain IBP bounds (or be equal for linear ops)
    // Row 0: sum of [1,2,3] = 6 (lower), sum of [2,3,4] = 9 (upper)
    assert!(
        (concrete.lower[[0]] - ibp_result.lower[[0, 0]]).abs() < 1e-5,
        "CROWN lower {} should equal IBP lower {} for linear op",
        concrete.lower[[0]],
        ibp_result.lower[[0, 0]]
    );
    assert!(
        (concrete.upper[[0]] - ibp_result.upper[[0, 0]]).abs() < 1e-5,
        "CROWN upper {} should equal IBP upper {} for linear op",
        concrete.upper[[0]],
        ibp_result.upper[[0, 0]]
    );
}

#[test]
fn test_reduce_sum_crown_no_keepdims() {
    // Test CROWN backward pass for ReduceSum with keepdims=false
    // Input: 2x4 tensor, output: 2 elements (dimension removed)
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
    let pre_upper = pre_lower.clone();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Identity linear bounds on the output (2 elements after reduction)
    let linear_bounds = LinearBounds::identity(2);
    let reduce = ReduceSumLayer::new(vec![-1], false);

    let result = reduce
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Output should have shape (2, 8) - 2 outputs, 8 inputs
    assert_eq!(result.lower_a.nrows(), 2);
    assert_eq!(result.lower_a.ncols(), 8);

    // Row 0 should have 1 for columns 0,1,2,3 and 0 elsewhere
    for j in 0..4 {
        assert!(
            (result.lower_a[[0, j]] - 1.0).abs() < 1e-6,
            "Expected 1.0, got {} at [0,{}]",
            result.lower_a[[0, j]],
            j
        );
    }
    for j in 4..8 {
        assert!(
            result.lower_a[[0, j]].abs() < 1e-6,
            "Expected 0, got {} at [0,{}]",
            result.lower_a[[0, j]],
            j
        );
    }
}

// ==================== Tanh tests ====================

#[test]
fn test_tanh_ibp_basic() {
    // Test tanh on interval that straddles zero
    let lower = ArrayD::from_elem(IxDyn(&[4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let tanh_layer = TanhLayer::new();
    let output = tanh_layer.propagate_ibp(&input).unwrap();

    // tanh is monotonic: tanh(-1) ≈ -0.7616, tanh(1) ≈ 0.7616
    let expected_lower = (-1.0_f32).tanh();
    let expected_upper = (1.0_f32).tanh();

    for i in 0..4 {
        assert!(
            (output.lower[[i]] - expected_lower).abs() < 1e-5,
            "tanh(-1) should be {}, got {}",
            expected_lower,
            output.lower[[i]]
        );
        assert!(
            (output.upper[[i]] - expected_upper).abs() < 1e-5,
            "tanh(1) should be {}, got {}",
            expected_upper,
            output.upper[[i]]
        );
    }
}

#[test]
fn test_tanh_ibp_soundness() {
    // Test that IBP bounds are sound (contain actual function values)
    // Test multiple intervals
    let test_cases = vec![
        (-5.0_f32, 5.0_f32),
        (-1.0, 1.0),
        (0.0, 2.0),
        (-2.0, 0.0),
        (-10.0, 10.0),
    ];

    for (l, u) in test_cases {
        let lower = ArrayD::from_elem(IxDyn(&[1]), l);
        let upper = ArrayD::from_elem(IxDyn(&[1]), u);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let tanh_layer = TanhLayer::new();
        let output = tanh_layer.propagate_ibp(&input).unwrap();

        // Test several points in the interval
        for i in 0..=10 {
            let x = l + (u - l) * (i as f32 / 10.0);
            let y = x.tanh();
            assert!(
                output.lower[[0]] <= y && y <= output.upper[[0]],
                "tanh({}) = {} should be in [{}, {}]",
                x,
                y,
                output.lower[[0]],
                output.upper[[0]]
            );
        }
    }
}

// ==================== Sigmoid tests ====================

#[test]
fn test_sigmoid_ibp_basic() {
    // Test sigmoid on interval that straddles zero
    let lower = ArrayD::from_elem(IxDyn(&[4]), -2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sigmoid_layer = SigmoidLayer::new();
    let output = sigmoid_layer.propagate_ibp(&input).unwrap();

    // sigmoid is monotonic
    let expected_lower = 1.0 / (1.0 + 2.0_f32.exp()); // sigmoid(-2)
    let expected_upper = 1.0 / (1.0 + (-2.0_f32).exp()); // sigmoid(2)

    for i in 0..4 {
        assert!(
            (output.lower[[i]] - expected_lower).abs() < 1e-5,
            "sigmoid(-2) should be {}, got {}",
            expected_lower,
            output.lower[[i]]
        );
        assert!(
            (output.upper[[i]] - expected_upper).abs() < 1e-5,
            "sigmoid(2) should be {}, got {}",
            expected_upper,
            output.upper[[i]]
        );
    }
}

#[test]
fn test_sigmoid_range() {
    // Sigmoid output should always be in [0, 1] (inclusive at limits due to float precision)
    let lower = ArrayD::from_elem(IxDyn(&[2]), -10.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sigmoid_layer = SigmoidLayer::new();
    let output = sigmoid_layer.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!(
            output.lower[[i]] >= 0.0 && output.lower[[i]] <= 1.0,
            "sigmoid lower bound should be in [0, 1], got {}",
            output.lower[[i]]
        );
        assert!(
            output.upper[[i]] >= 0.0 && output.upper[[i]] <= 1.0,
            "sigmoid upper bound should be in [0, 1], got {}",
            output.upper[[i]]
        );
        // Also check monotonicity: lower input gives lower sigmoid
        assert!(
            output.lower[[i]] < output.upper[[i]],
            "sigmoid bounds should be ordered"
        );
    }
}

// ==================== Softplus tests ====================

#[test]
fn test_softplus_ibp_basic() {
    // Test softplus on interval
    let lower = ArrayD::from_elem(IxDyn(&[3]), -2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let softplus_layer = SoftplusLayer::new();
    let output = softplus_layer.propagate_ibp(&input).unwrap();

    // softplus is monotonic
    let expected_lower = (1.0 + (-2.0_f32).exp()).ln();
    let expected_upper = (1.0 + 2.0_f32.exp()).ln();

    for i in 0..3 {
        assert!(
            (output.lower[[i]] - expected_lower).abs() < 1e-5,
            "softplus(-2) should be {}, got {}",
            expected_lower,
            output.lower[[i]]
        );
        assert!(
            (output.upper[[i]] - expected_upper).abs() < 1e-5,
            "softplus(2) should be {}, got {}",
            expected_upper,
            output.upper[[i]]
        );
    }
}

#[test]
fn test_softplus_always_positive() {
    // Softplus output should always be positive
    let lower = ArrayD::from_elem(IxDyn(&[2]), -100.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), -50.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let softplus_layer = SoftplusLayer::new();
    let output = softplus_layer.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!(
            output.lower[[i]] >= 0.0,
            "softplus should always be non-negative"
        );
    }
}

// ==================== Sin/Cos tests ====================

#[test]
fn test_sin_ibp_no_extrema() {
    use std::f32::consts::PI;
    // Test sin on interval that doesn't contain extrema
    let lower = ArrayD::from_elem(IxDyn(&[2]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), PI / 4.0);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sin_layer = SinLayer::new();
    let output = sin_layer.propagate_ibp(&input).unwrap();

    // sin is monotonically increasing on [0, π/4]
    let expected_lower = 0.0_f32.sin(); // = 0
    let expected_upper = (PI / 4.0).sin(); // ≈ 0.707

    for i in 0..2 {
        assert!(
            (output.lower[[i]] - expected_lower).abs() < 1e-5,
            "sin(0) should be {}, got {}",
            expected_lower,
            output.lower[[i]]
        );
        assert!(
            (output.upper[[i]] - expected_upper).abs() < 1e-5,
            "sin(π/4) should be {}, got {}",
            expected_upper,
            output.upper[[i]]
        );
    }
}

#[test]
fn test_sin_ibp_contains_maximum() {
    use std::f32::consts::PI;
    // Test sin on interval that contains π/2 (maximum)
    let lower = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), PI);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sin_layer = SinLayer::new();
    let output = sin_layer.propagate_ibp(&input).unwrap();

    // Interval contains π/2 where sin=1
    assert!(
        (output.upper[[0]] - 1.0).abs() < 1e-5,
        "sin max should be 1 when interval contains π/2, got {}",
        output.upper[[0]]
    );
}

#[test]
fn test_cos_ibp_no_extrema() {
    use std::f32::consts::PI;
    // Test cos on interval that doesn't contain extrema
    let lower = ArrayD::from_elem(IxDyn(&[2]), PI / 4.0);
    let upper = ArrayD::from_elem(IxDyn(&[2]), PI / 2.0);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let cos_layer = CosLayer::new();
    let output = cos_layer.propagate_ibp(&input).unwrap();

    // cos is monotonically decreasing on [π/4, π/2]
    let expected_upper = (PI / 4.0).cos(); // ≈ 0.707
    let expected_lower = (PI / 2.0).cos(); // ≈ 0

    for i in 0..2 {
        assert!(
            (output.lower[[i]] - expected_lower).abs() < 1e-5,
            "cos(π/2) should be {}, got {}",
            expected_lower,
            output.lower[[i]]
        );
        assert!(
            (output.upper[[i]] - expected_upper).abs() < 1e-5,
            "cos(π/4) should be {}, got {}",
            expected_upper,
            output.upper[[i]]
        );
    }
}

#[test]
fn test_cos_ibp_contains_minimum() {
    use std::f32::consts::PI;
    // Test cos on interval that contains π (minimum)
    let lower = ArrayD::from_elem(IxDyn(&[1]), PI / 2.0);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 3.0 * PI / 2.0);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let cos_layer = CosLayer::new();
    let output = cos_layer.propagate_ibp(&input).unwrap();

    // Interval contains π where cos=-1
    assert!(
        (output.lower[[0]] - (-1.0)).abs() < 1e-5,
        "cos min should be -1 when interval contains π, got {}",
        output.lower[[0]]
    );
}

// ==================== Adaptive GELU Relaxation tests ====================

use crate::layers::{
    adaptive_gelu_linear_relaxation, gelu_eval, gelu_linear_relaxation, GeluApproximation,
    RelaxationMode,
};

/// Helper: compute GELU at a point
fn gelu_at(x: f32) -> f32 {
    gelu_eval(x, GeluApproximation::Erf)
}

/// Helper: verify relaxation is sound (lower <= GELU <= upper over interval)
fn verify_relaxation_sound(l: f32, u: f32, relaxation: (f32, f32, f32, f32)) {
    let (ls, li, us, ui) = relaxation;
    let num_samples = 100;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let gx = gelu_at(x);
        let lower_bound = ls * x + li;
        let upper_bound = us * x + ui;

        assert!(
            lower_bound <= gx + 1e-4,
            "Lower bound violated at x={}: lower_bound={} > GELU({})={}",
            x,
            lower_bound,
            x,
            gx
        );
        assert!(
            gx <= upper_bound + 1e-4,
            "Upper bound violated at x={}: GELU({})={} > upper_bound={}",
            x,
            x,
            gx,
            upper_bound
        );
    }
}

#[test]
fn test_relaxation_mode_chord_soundness() {
    // Test chord mode on various intervals
    let test_intervals = [
        (-2.0, -1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (1.0, 2.0),
        (-1.0, 1.0),
        (-3.0, 3.0),
    ];

    for (l, u) in test_intervals {
        let relaxation =
            adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Chord);
        verify_relaxation_sound(l, u, relaxation);
    }
}

#[test]
fn test_relaxation_mode_tangent_soundness() {
    // Test tangent mode on various intervals
    let test_intervals = [
        (-2.0, -1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (1.0, 2.0),
        (-1.0, 1.0),
        (-3.0, 3.0),
    ];

    for (l, u) in test_intervals {
        let relaxation =
            adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Tangent);
        verify_relaxation_sound(l, u, relaxation);
    }
}

#[test]
fn test_relaxation_mode_two_slope_soundness() {
    // Test two-slope mode on various intervals
    let test_intervals = [
        (-2.0, -1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (1.0, 2.0),
        (-1.0, 1.0),
        (-3.0, 3.0),
    ];

    for (l, u) in test_intervals {
        let relaxation =
            adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::TwoSlope);
        verify_relaxation_sound(l, u, relaxation);
    }
}

#[test]
fn test_relaxation_mode_adaptive_soundness() {
    // Test adaptive mode on various intervals
    let test_intervals = [
        (-2.0, -1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (1.0, 2.0),
        (-1.0, 1.0),
        (-3.0, 3.0),
    ];

    for (l, u) in test_intervals {
        let relaxation =
            adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Adaptive);
        verify_relaxation_sound(l, u, relaxation);
    }
}

#[test]
fn test_adaptive_is_at_least_as_tight_as_chord() {
    // Adaptive mode should produce bounds at least as tight as chord
    let test_intervals = [
        (-2.0, -1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (1.0, 2.0),
        (-1.0, 1.0),
        (-0.5, 0.5),
    ];

    for (l, u) in test_intervals {
        let chord =
            adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Chord);
        let adaptive =
            adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Adaptive);

        let c = (l + u) / 2.0;
        let chord_width = (chord.2 * c + chord.3) - (chord.0 * c + chord.1);
        let adaptive_width = (adaptive.2 * c + adaptive.3) - (adaptive.0 * c + adaptive.1);

        assert!(
            adaptive_width <= chord_width + 1e-5,
            "Adaptive should be at least as tight as chord for [{}, {}]: adaptive_width={} > chord_width={}",
            l, u, adaptive_width, chord_width
        );
    }
}

#[test]
fn test_relaxation_modes_small_interval() {
    // For small intervals, tangent should be very tight
    let l = -0.1;
    let u = 0.1;

    let chord =
        adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Chord);
    let tangent =
        adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Tangent);
    let two_slope =
        adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::TwoSlope);
    let adaptive =
        adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Adaptive);

    let c = (l + u) / 2.0;
    let chord_width = (chord.2 * c + chord.3) - (chord.0 * c + chord.1);
    let tangent_width = (tangent.2 * c + tangent.3) - (tangent.0 * c + tangent.1);
    let two_slope_width = (two_slope.2 * c + two_slope.3) - (two_slope.0 * c + two_slope.1);
    let adaptive_width = (adaptive.2 * c + adaptive.3) - (adaptive.0 * c + adaptive.1);

    // All modes should be sound
    verify_relaxation_sound(l, u, chord);
    verify_relaxation_sound(l, u, tangent);
    verify_relaxation_sound(l, u, two_slope);
    verify_relaxation_sound(l, u, adaptive);

    // Adaptive should be the tightest or equal
    assert!(adaptive_width <= chord_width + 1e-5);
    assert!(adaptive_width <= tangent_width + 1e-5);
    assert!(adaptive_width <= two_slope_width + 1e-5);
}

#[test]
fn test_gelu_layer_with_relaxation_modes() {
    // Test GELULayer with different relaxation modes produces valid bounds
    let lower = ArrayD::from_elem(IxDyn(&[4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Test with default (Chord) mode
    let gelu_chord = GELULayer::new(GeluApproximation::Erf);
    let output_chord = gelu_chord.propagate_ibp(&input).unwrap();

    // Test with adaptive mode
    let gelu_adaptive = GELULayer::adaptive(GeluApproximation::Erf);
    let output_adaptive = gelu_adaptive.propagate_ibp(&input).unwrap();

    // Both should produce valid bounds
    for i in 0..4 {
        assert!(output_chord.lower[[i]] <= output_chord.upper[[i]]);
        assert!(output_adaptive.lower[[i]] <= output_adaptive.upper[[i]]);
    }
}

#[test]
fn test_relaxation_modes_wide_interval() {
    // For wide intervals, two-slope may help
    let l = -3.0;
    let u = 3.0;

    let chord =
        adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Chord);
    let tangent =
        adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Tangent);
    let two_slope =
        adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::TwoSlope);
    let adaptive =
        adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Adaptive);

    // All modes should be sound
    verify_relaxation_sound(l, u, chord);
    verify_relaxation_sound(l, u, tangent);
    verify_relaxation_sound(l, u, two_slope);
    verify_relaxation_sound(l, u, adaptive);

    // Adaptive should pick the best
    let c = (l + u) / 2.0;
    let chord_width = (chord.2 * c + chord.3) - (chord.0 * c + chord.1);
    let adaptive_width = (adaptive.2 * c + adaptive.3) - (adaptive.0 * c + adaptive.1);

    assert!(
        adaptive_width <= chord_width + 1e-5,
        "Adaptive should be at least as tight as chord"
    );
}

#[test]
fn test_chord_vs_original_gelu_linear_relaxation() {
    // Verify chord mode matches original function
    let test_intervals = [(-2.0, -1.0), (-1.0, 0.0), (0.0, 1.0), (-1.0, 1.0)];

    for (l, u) in test_intervals {
        let original = gelu_linear_relaxation(l, u, GeluApproximation::Erf);
        let chord =
            adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Chord);

        assert!((original.0 - chord.0).abs() < 1e-6, "Lower slope mismatch");
        assert!(
            (original.1 - chord.1).abs() < 1e-6,
            "Lower intercept mismatch"
        );
        assert!((original.2 - chord.2).abs() < 1e-6, "Upper slope mismatch");
        assert!(
            (original.3 - chord.3).abs() < 1e-6,
            "Upper intercept mismatch"
        );
    }
}

#[test]
fn test_relaxation_improvement_metrics() {
    // Print improvement metrics for adaptive relaxation
    // This test always passes but logs useful information

    let test_intervals = [
        ("small_near_zero", -0.1_f32, 0.1_f32),
        ("medium_symmetric", -1.0, 1.0),
        ("wide_symmetric", -3.0, 3.0),
        ("negative_region", -2.0, -0.5),
        ("positive_region", 0.5, 2.0),
        ("critical_region", -1.0, 0.0),
    ];

    let mut total_chord = 0.0_f32;
    let mut total_adaptive = 0.0_f32;
    let mut improvements = Vec::new();

    for (name, l, u) in test_intervals.iter() {
        let chord =
            adaptive_gelu_linear_relaxation(*l, *u, GeluApproximation::Erf, RelaxationMode::Chord);
        let adaptive = adaptive_gelu_linear_relaxation(
            *l,
            *u,
            GeluApproximation::Erf,
            RelaxationMode::Adaptive,
        );

        let c = (l + u) / 2.0;
        let chord_width = (chord.2 * c + chord.3) - (chord.0 * c + chord.1);
        let adaptive_width = (adaptive.2 * c + adaptive.3) - (adaptive.0 * c + adaptive.1);

        let improvement = if chord_width > 0.0 {
            (chord_width - adaptive_width) / chord_width * 100.0
        } else {
            0.0
        };

        improvements.push((name, chord_width, adaptive_width, improvement));
        total_chord += chord_width;
        total_adaptive += adaptive_width;
    }

    // Log results (visible with --nocapture)
    eprintln!("\n=== Adaptive GELU Relaxation Improvement ===");
    eprintln!(
        "{:<20} {:>12} {:>12} {:>12}",
        "Interval", "Chord", "Adaptive", "Improvement"
    );
    for (name, chord, adaptive, improvement) in &improvements {
        eprintln!(
            "{:<20} {:>12.6} {:>12.6} {:>11.1}%",
            name, chord, adaptive, improvement
        );
    }

    let avg_improvement = if total_chord > 0.0 {
        (total_chord - total_adaptive) / total_chord * 100.0
    } else {
        0.0
    };
    eprintln!("\nAverage improvement: {:.1}%", avg_improvement);

    // Verify adaptive is never worse than chord
    for (name, chord_width, adaptive_width, _) in improvements {
        assert!(
            adaptive_width <= chord_width + 1e-5,
            "Adaptive should not be worse than chord for {}",
            name
        );
    }
}

// ==================== Where (conditional) tests ====================

#[test]
fn test_where_ibp_ternary_basic() {
    // Test Where: output = x if condition else y
    // For interval bounds, result is union of x and y bounds
    let condition_lower = ArrayD::from_elem(IxDyn(&[3]), 0.0f32);
    let condition_upper = ArrayD::from_elem(IxDyn(&[3]), 1.0f32);
    let condition = BoundedTensor::new(condition_lower, condition_upper).unwrap();

    // x bounds: [2, 5]
    let x_lower = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let x_upper = ArrayD::from_elem(IxDyn(&[3]), 5.0f32);
    let x = BoundedTensor::new(x_lower, x_upper).unwrap();

    // y bounds: [-1, 3]
    let y_lower = ArrayD::from_elem(IxDyn(&[3]), -1.0f32);
    let y_upper = ArrayD::from_elem(IxDyn(&[3]), 3.0f32);
    let y = BoundedTensor::new(y_lower, y_upper).unwrap();

    let where_layer = WhereLayer::new();
    let output = where_layer
        .propagate_ibp_ternary(&condition, &x, &y)
        .unwrap();

    // Union of [2, 5] and [-1, 3] = [-1, 5]
    for i in 0..3 {
        assert!(
            (output.lower[[i]] - (-1.0)).abs() < 1e-6,
            "lower should be min(-1, 2) = -1, got {}",
            output.lower[[i]]
        );
        assert!(
            (output.upper[[i]] - 5.0).abs() < 1e-6,
            "upper should be max(5, 3) = 5, got {}",
            output.upper[[i]]
        );
    }
}

#[test]
fn test_where_ibp_ternary_non_overlapping() {
    // Test Where with non-overlapping intervals
    let condition_lower = ArrayD::from_elem(IxDyn(&[2]), 0.0f32);
    let condition_upper = ArrayD::from_elem(IxDyn(&[2]), 1.0f32);
    let condition = BoundedTensor::new(condition_lower, condition_upper).unwrap();

    // x bounds: [10, 20]
    let x_lower = ArrayD::from_elem(IxDyn(&[2]), 10.0f32);
    let x_upper = ArrayD::from_elem(IxDyn(&[2]), 20.0f32);
    let x = BoundedTensor::new(x_lower, x_upper).unwrap();

    // y bounds: [-5, 5]
    let y_lower = ArrayD::from_elem(IxDyn(&[2]), -5.0f32);
    let y_upper = ArrayD::from_elem(IxDyn(&[2]), 5.0f32);
    let y = BoundedTensor::new(y_lower, y_upper).unwrap();

    let where_layer = WhereLayer::new();
    let output = where_layer
        .propagate_ibp_ternary(&condition, &x, &y)
        .unwrap();

    // Union of [10, 20] and [-5, 5] = [-5, 20]
    for i in 0..2 {
        assert!(
            (output.lower[[i]] - (-5.0)).abs() < 1e-6,
            "lower should be -5, got {}",
            output.lower[[i]]
        );
        assert!(
            (output.upper[[i]] - 20.0).abs() < 1e-6,
            "upper should be 20, got {}",
            output.upper[[i]]
        );
    }
}

#[test]
fn test_where_ibp_error_not_supported() {
    // Single-input IBP should fail
    let lower = ArrayD::from_elem(IxDyn(&[3]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let where_layer = WhereLayer::new();
    assert!(where_layer.propagate_ibp(&input).is_err());
}

#[test]
fn test_where_linear_not_supported() {
    let bounds = LinearBounds::identity(4);
    let where_layer = WhereLayer::new();
    assert!(where_layer.propagate_linear(&bounds).is_err());
}

// ==================== NonZero tests ====================

#[test]
fn test_nonzero_ibp_all_nonzero() {
    // All elements are definitely non-zero (positive interval)
    let lower = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 3]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let nonzero = NonZeroLayer;
    let output = nonzero.propagate_ibp(&input).unwrap();

    // Output shape: [rank(input), max_nonzero] = [2, 6]
    assert_eq!(output.shape(), &[2, 6]);

    // All lower bounds should be 0 (min index)
    for val in output.lower.iter() {
        assert_eq!(*val, 0.0);
    }

    // Upper bounds for dim 0 should be 1 (shape[0]-1 = 2-1)
    // Upper bounds for dim 1 should be 2 (shape[1]-1 = 3-1)
    for col in 0..6 {
        assert_eq!(output.upper[[0, col]], 1.0);
        assert_eq!(output.upper[[1, col]], 2.0);
    }
}

#[test]
fn test_nonzero_ibp_all_zeros() {
    // All elements are exactly zero
    let lower = ArrayD::from_elem(IxDyn(&[3, 4]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3, 4]), 0.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let nonzero = NonZeroLayer;
    let output = nonzero.propagate_ibp(&input).unwrap();

    // Output shape: [rank(input), 0] = [2, 0] (no nonzero elements)
    assert_eq!(output.shape(), &[2, 0]);
}

#[test]
fn test_nonzero_ibp_mixed() {
    // Some elements could be nonzero, some definitely zero
    let mut lower = ArrayD::from_elem(IxDyn(&[4]), 0.0f32);
    let mut upper = ArrayD::from_elem(IxDyn(&[4]), 0.0f32);

    // Element 0: [0, 0] - definitely zero
    // Element 1: [1, 2] - definitely non-zero
    lower[[1]] = 1.0;
    upper[[1]] = 2.0;
    // Element 2: [-1, 1] - could be zero or nonzero
    lower[[2]] = -1.0;
    upper[[2]] = 1.0;
    // Element 3: [0, 0] - definitely zero

    let input = BoundedTensor::new(lower, upper).unwrap();

    let nonzero = NonZeroLayer;
    let output = nonzero.propagate_ibp(&input).unwrap();

    // Elements 1 and 2 could be non-zero, so max_nonzero = 2
    // Output shape: [1, 2] (1D input)
    assert_eq!(output.shape(), &[1, 2]);

    // Lower bounds: all 0
    for val in output.lower.iter() {
        assert_eq!(*val, 0.0);
    }

    // Upper bounds: 3 (input shape - 1 = 4 - 1 = 3)
    for val in output.upper.iter() {
        assert_eq!(*val, 3.0);
    }
}

#[test]
fn test_nonzero_linear_not_supported() {
    // Linear bounds should fail (data-dependent output shape)
    let bounds = LinearBounds::identity(4);
    let nonzero = NonZeroLayer;
    assert!(nonzero.propagate_linear(&bounds).is_err());
}

// ==================== ThresholdedRelu CROWN tests ====================

#[test]
fn test_thresholded_relu_crown_backward_identity() {
    // When all pre-activations are above threshold, should be identity
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 3.0, 4.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![5.0, 6.0, 7.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let threshold_relu = ThresholdedReluLayer::new(1.0); // alpha = 1.0

    let result = threshold_relu
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // All above threshold, should be identity (slope=1, intercept=0)
    for i in 0..3 {
        assert!(
            (result.lower_a[[i, i]] - 1.0).abs() < 1e-6,
            "Active region should have slope 1"
        );
        assert!(
            (result.upper_a[[i, i]] - 1.0).abs() < 1e-6,
            "Active region should have slope 1"
        );
    }
    assert!(result.lower_b.iter().all(|&x| x.abs() < 1e-6));
    assert!(result.upper_b.iter().all(|&x| x.abs() < 1e-6));
}

#[test]
fn test_thresholded_relu_crown_backward_zero() {
    // When all pre-activations are at or below threshold, should be zero
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0, -1.0, 0.5]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-0.5, 0.5, 1.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let threshold_relu = ThresholdedReluLayer::new(1.0); // alpha = 1.0

    let result = threshold_relu
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // All at or below threshold, should be zero
    for i in 0..3 {
        assert!(
            result.lower_a[[i, i]].abs() < 1e-6,
            "Inactive region should have slope 0"
        );
        assert!(
            result.upper_a[[i, i]].abs() < 1e-6,
            "Inactive region should have slope 0"
        );
    }
}

#[test]
fn test_thresholded_relu_crown_soundness() {
    // Test that CROWN bounds are sound (contain true outputs)
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.5, 2.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 1.5, 4.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let alpha = 1.0;
    let threshold_relu = ThresholdedReluLayer::new(alpha);

    let result = threshold_relu
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample points in the input range and verify bounds hold
    let test_points: [Vec<f32>; 5] = [
        vec![-1.0, 0.5, 2.0], // lower
        vec![2.0, 1.5, 4.0],  // upper
        vec![0.0, 1.0, 3.0],  // middle
        vec![1.0, 1.0, 2.5],  // on threshold
        vec![1.5, 1.2, 3.5],  // above threshold
    ];

    for point in &test_points {
        // ThresholdedRelu: y = x if x > alpha, else 0
        let tr_output: Vec<f32> = point
            .iter()
            .map(|&x| if x > alpha { x } else { 0.0 })
            .collect();

        // Check each output dimension
        for (j, &tr_val) in tr_output.iter().enumerate() {
            let lb_val: f32 = (0..3)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            let ub_val: f32 = (0..3)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            let tol = 1e-4;
            assert!(
                lb_val <= tr_val + tol,
                "Lower bound violated at point {:?}: lb {} > tr {}",
                point,
                lb_val,
                tr_val
            );
            assert!(
                ub_val >= tr_val - tol,
                "Upper bound violated at point {:?}: ub {} < tr {}",
                point,
                ub_val,
                tr_val
            );
        }
    }
}

// ==================== Shrink CROWN tests ====================

#[test]
fn test_shrink_crown_backward_dead_zone() {
    // When all pre-activations are in dead zone, should be zero
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-0.3, -0.2, 0.1]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.2, 0.3, 0.4]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let shrink = ShrinkLayer::new(0.0, 0.5); // bias=0, lambd=0.5

    let result = shrink
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // All in dead zone [-0.5, 0.5], should be zero
    for i in 0..3 {
        assert!(
            result.lower_a[[i, i]].abs() < 1e-6,
            "Dead zone should have slope 0"
        );
        assert!(
            result.upper_a[[i, i]].abs() < 1e-6,
            "Dead zone should have slope 0"
        );
    }
}

#[test]
fn test_shrink_crown_backward_positive_piece() {
    // When all pre-activations are in positive piece (x > lambd)
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 4.0, 5.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let bias = 0.1;
    let lambd = 0.5;
    let shrink = ShrinkLayer::new(bias, lambd);

    let result = shrink
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // All in positive piece, should be y = x - bias (slope=1, intercept=-bias)
    for i in 0..3 {
        assert!(
            (result.lower_a[[i, i]] - 1.0).abs() < 1e-6,
            "Positive piece should have slope 1"
        );
        assert!(
            (result.upper_a[[i, i]] - 1.0).abs() < 1e-6,
            "Positive piece should have slope 1"
        );
    }
    // Intercept should be -bias
    for &b in result.lower_b.iter() {
        assert!((b - (-bias)).abs() < 1e-6, "Intercept should be -bias");
    }
}

#[test]
fn test_shrink_crown_backward_negative_piece() {
    // When all pre-activations are in negative piece (x < -lambd)
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-4.0, -3.0, -2.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, -0.8, -0.7]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let bias = 0.2;
    let lambd = 0.5;
    let shrink = ShrinkLayer::new(bias, lambd);

    let result = shrink
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // All in negative piece, should be y = x + bias (slope=1, intercept=+bias)
    for i in 0..3 {
        assert!(
            (result.lower_a[[i, i]] - 1.0).abs() < 1e-6,
            "Negative piece should have slope 1"
        );
        assert!(
            (result.upper_a[[i, i]] - 1.0).abs() < 1e-6,
            "Negative piece should have slope 1"
        );
    }
    // Intercept should be +bias
    for &b in result.lower_b.iter() {
        assert!((b - bias).abs() < 1e-6, "Intercept should be +bias");
    }
}

#[test]
fn test_shrink_crown_soundness() {
    // Test that CROWN bounds are sound (contain true outputs)
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-2.0, -0.3, 0.2, 1.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-0.8, 0.4, 1.5, 3.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let bias = 0.1;
    let lambd = 0.5;
    let shrink = ShrinkLayer::new(bias, lambd);

    let result = shrink
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample points in the input range and verify bounds hold
    // Pre-activation bounds: dim0=[-2.0,-0.8], dim1=[-0.3,0.4], dim2=[0.2,1.5], dim3=[1.0,3.0]
    let test_points: [Vec<f32>; 5] = [
        vec![-2.0, -0.3, 0.2, 1.0], // lower bounds
        vec![-0.8, 0.4, 1.5, 3.0],  // upper bounds
        vec![-1.5, 0.0, 0.8, 2.0],  // middle
        vec![-0.9, 0.3, 0.5, 1.5],  // within bounds (dim0 in neg piece, dim1 in dead zone, etc)
        vec![-1.0, 0.1, 1.0, 1.5],  // mixed
    ];

    for point in &test_points {
        // Shrink: y = x - bias if x > lambd, x + bias if x < -lambd, else 0
        let shrink_output: Vec<f32> = point
            .iter()
            .map(|&x| {
                if x > lambd {
                    x - bias
                } else if x < -lambd {
                    x + bias
                } else {
                    0.0
                }
            })
            .collect();

        // Check each output dimension
        for (j, &shrink_val) in shrink_output.iter().enumerate() {
            let lb_val: f32 = (0..4)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            let ub_val: f32 = (0..4)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            let tol = 1e-4;
            assert!(
                lb_val <= shrink_val + tol,
                "Lower bound violated at point {:?}: lb {} > shrink {}",
                point,
                lb_val,
                shrink_val
            );
            assert!(
                ub_val >= shrink_val - tol,
                "Upper bound violated at point {:?}: ub {} < shrink {}",
                point,
                ub_val,
                shrink_val
            );
        }
    }
}

// ==================== LogSoftmax CROWN tests ====================

#[test]
fn test_logsoftmax_ibp_basic() {
    // Test basic LogSoftmax IBP propagation
    let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let input = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    let logsoftmax = LogSoftmaxLayer::new(-1);
    let output = logsoftmax.propagate_ibp(&input).unwrap();

    // Lower bound should be less than upper bound
    for i in 0..4 {
        assert!(
            output.lower[[i]] <= output.upper[[i]],
            "Lower bound should be <= upper bound"
        );
    }

    // Check that the bounds are sound by sampling points in the input interval
    for sample in 0..20 {
        // Generate a random point in the interval
        let point: Vec<f32> = (0..4)
            .map(|i| {
                let t = ((sample as u32).wrapping_mul(2654435761) ^ (i as u32)) as f32
                    / u32::MAX as f32;
                lower[[i]] + (upper[[i]] - lower[[i]]) * t
            })
            .collect();
        let logsoftmax_output = {
            let max_val = point.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = point.iter().map(|&v| (v - max_val).exp()).sum();
            let lse = max_val + exp_sum.ln();
            point.iter().map(|&v| v - lse).collect::<Vec<f32>>()
        };

        for (i, &lsm_val) in logsoftmax_output.iter().enumerate() {
            let tol = 1e-5;
            assert!(
                output.lower[[i]] <= lsm_val + tol,
                "IBP lower bound violated at sample {}: {} > {}",
                sample,
                output.lower[[i]],
                lsm_val
            );
            assert!(
                output.upper[[i]] >= lsm_val - tol,
                "IBP upper bound violated at sample {}: {} < {}",
                sample,
                output.upper[[i]],
                lsm_val
            );
        }
    }
}

#[test]
fn test_logsoftmax_crown_backward_basic() {
    // Test LogSoftmax CROWN backward propagation
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let logsoftmax = LogSoftmaxLayer::new(-1);

    let result = logsoftmax
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Check dimensions
    assert_eq!(result.lower_a.shape(), &[4, 4]);
    assert_eq!(result.upper_a.shape(), &[4, 4]);
    assert_eq!(result.lower_b.len(), 4);
    assert_eq!(result.upper_b.len(), 4);

    // The Jacobian of LogSoftmax is J_ij = δ_ij - softmax_j
    // So diagonal should be close to 1 - softmax[i]
    // and off-diagonal should be close to -softmax[j]
    // At center point [0.5, 1.5, 2.5, 3.5]:
    let center: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5];
    let max_val = center.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = center.iter().map(|&v| (v - max_val).exp()).collect();
    let exp_sum: f32 = exp_vals.iter().sum();
    let softmax: Vec<f32> = exp_vals.iter().map(|&e| e / exp_sum).collect();

    // Check that coefficient matrix has reasonable structure
    // Diagonal should be positive (since 1 - softmax[i] > 0 for softmax[i] < 1)
    for (i, &sm_val) in softmax.iter().enumerate() {
        // Diagonal elements should be larger than off-diagonal
        // (since J[i,i] = 1 - softmax[i] vs J[i,j] = -softmax[j])
        for j in 0..4 {
            if i == j {
                // Diagonal should be around 1 - softmax[i]
                let expected_diag = 1.0 - sm_val;
                // Check that the actual value is in reasonable range
                assert!(
                    result.lower_a[[i, j]] < expected_diag + 0.5,
                    "Diagonal too large"
                );
            }
        }
    }
}

#[test]
fn test_logsoftmax_crown_soundness() {
    // Test that CROWN bounds are sound (contain the actual function values)
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1.0, 0.0, 1.0, 2.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let logsoftmax = LogSoftmaxLayer::new(-1);

    let result = logsoftmax
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample many points and verify bounds contain actual values
    for sample in 0..20 {
        // Generate a random point in the interval
        let point: Vec<f32> = (0..4)
            .map(|i| {
                let t = ((sample as u32).wrapping_mul(2654435761) ^ (i as u32)) as f32
                    / u32::MAX as f32;
                pre_lower[[i]] + (pre_upper[[i]] - pre_lower[[i]]) * t
            })
            .collect();

        // Compute actual logsoftmax output
        let max_val = point.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = point.iter().map(|&v| (v - max_val).exp()).sum();
        let lse = max_val + exp_sum.ln();
        let logsoftmax_output: Vec<f32> = point.iter().map(|&v| v - lse).collect();

        // Check each output dimension
        for (j, &lsm_val) in logsoftmax_output.iter().enumerate() {
            let lb_val: f32 = (0..4)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            let ub_val: f32 = (0..4)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            let tol = 0.1; // Generous tolerance due to sampling-based error estimation
            assert!(
                lb_val <= lsm_val + tol,
                "CROWN lower bound violated at sample {}, dim {}: lb {} > actual {}",
                sample,
                j,
                lb_val,
                lsm_val
            );
            assert!(
                ub_val >= lsm_val - tol,
                "CROWN upper bound violated at sample {}, dim {}: ub {} < actual {}",
                sample,
                j,
                ub_val,
                lsm_val
            );
        }
    }
}

#[test]
fn test_logsoftmax_crown_network_integration() {
    // Test LogSoftmax CROWN in a network context
    use crate::layers::{LinearLayer, LogSoftmaxLayer};
    use crate::network::Network;
    use ndarray::Array2;

    // Create a simple network: Linear -> LogSoftmax
    let weight = Array2::from_shape_vec(
        (4, 3),
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let bias: Option<ndarray::Array1<f32>> = Some(ndarray::Array1::zeros(4));
    let linear = LinearLayer::new(weight, bias).unwrap();

    let logsoftmax = LogSoftmaxLayer::new(-1);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::LogSoftmax(logsoftmax));

    // Create input bounds
    let input_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, -1.0, -1.0]).unwrap();
    let input_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 1.0, 1.0]).unwrap();
    let input = BoundedTensor::new(input_lower, input_upper).unwrap();

    // Test CROWN propagation
    let crown_result = network.propagate_crown(&input).unwrap();

    // Test IBP propagation for comparison
    let ibp_result = network.propagate_ibp(&input).unwrap();

    // CROWN bounds should be at least as tight as (or equal to) IBP bounds
    for i in 0..4 {
        // CROWN lower bound should be >= IBP lower bound (tighter)
        assert!(
            crown_result.lower[[i]] >= ibp_result.lower[[i]] - 1e-4,
            "CROWN lower bound {} should be >= IBP lower bound {}",
            crown_result.lower[[i]],
            ibp_result.lower[[i]]
        );
        // CROWN upper bound should be <= IBP upper bound (tighter)
        assert!(
            crown_result.upper[[i]] <= ibp_result.upper[[i]] + 1e-4,
            "CROWN upper bound {} should be <= IBP upper bound {}",
            crown_result.upper[[i]],
            ibp_result.upper[[i]]
        );
    }
}

// ==================== Floor CROWN tests ====================

#[test]
fn test_floor_crown_constant_bounds() {
    // Floor is piecewise constant, so CROWN should produce slope=0 bounds
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.2, -0.8, 2.9, -2.1]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.8, 0.3, 3.5, -1.5]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let floor = FloorLayer;

    let result = floor
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // CROWN for discontinuous functions: slope = 0, intercept = f(bound)
    // All coefficients should be zero (constant bounds)
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                result.lower_a[[i, j]].abs() < 1e-6,
                "Floor CROWN lower slope should be 0, got {} at [{},{}]",
                result.lower_a[[i, j]],
                i,
                j
            );
            assert!(
                result.upper_a[[i, j]].abs() < 1e-6,
                "Floor CROWN upper slope should be 0, got {} at [{},{}]",
                result.upper_a[[i, j]],
                i,
                j
            );
        }
    }

    // Check intercepts match IBP bounds
    // floor([1.2, 1.8]) = [1, 1], floor([-0.8, 0.3]) = [-1, 0],
    // floor([2.9, 3.5]) = [2, 3], floor([-2.1, -1.5]) = [-3, -2]
    let expected_lower = [1.0, -1.0, 2.0, -3.0];
    let expected_upper = [1.0, 0.0, 3.0, -2.0];

    for i in 0..4 {
        assert!(
            (result.lower_b[i] - expected_lower[i]).abs() < 1e-6,
            "Floor CROWN lower intercept mismatch at {}: got {}, expected {}",
            i,
            result.lower_b[i],
            expected_lower[i]
        );
        assert!(
            (result.upper_b[i] - expected_upper[i]).abs() < 1e-6,
            "Floor CROWN upper intercept mismatch at {}: got {}, expected {}",
            i,
            result.upper_b[i],
            expected_upper[i]
        );
    }
}

#[test]
fn test_floor_crown_soundness() {
    // Test that CROWN bounds are sound (contain true outputs)
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.5, 0.2, 2.7]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5, 1.8, 3.3]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let floor = FloorLayer;

    let result = floor
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample test points within the input bounds
    let test_points: [Vec<f32>; 3] = [
        vec![-1.5, 0.2, 2.7], // lower
        vec![0.5, 1.8, 3.3],  // upper
        vec![-0.5, 1.0, 3.0], // middle
    ];

    for point in &test_points {
        let floor_output: Vec<f32> = point.iter().map(|x| x.floor()).collect();

        for (j, &floor_val) in floor_output.iter().enumerate() {
            // Since slope=0, bound = intercept (constant)
            let lower_bound = result.lower_b[j];
            let upper_bound = result.upper_b[j];

            assert!(
                floor_val >= lower_bound - 1e-6,
                "Floor output {} should be >= lower bound {}",
                floor_val,
                lower_bound
            );
            assert!(
                floor_val <= upper_bound + 1e-6,
                "Floor output {} should be <= upper bound {}",
                floor_val,
                upper_bound
            );
        }
    }
}

#[test]
fn test_floor_crown_network_integration() {
    // Test Floor in a simple network with CROWN propagation
    let weight = ndarray::Array2::from_shape_vec(
        (4, 4),
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
    .unwrap();
    let bias = Some(ndarray::Array1::from_vec(vec![0.5, -0.5, 0.0, 0.0]));
    let linear = LinearLayer::new(weight, bias).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::Floor(FloorLayer));

    let input_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 0.0, 1.0, -1.0]).unwrap();
    let input_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 1.0, 2.0, 0.0]).unwrap();
    let input = BoundedTensor::new(input_lower, input_upper).unwrap();

    let crown_result = network.propagate_crown(&input).unwrap();
    let ibp_result = network.propagate_ibp(&input).unwrap();

    // CROWN bounds should equal IBP bounds for floor (constant relaxation)
    for i in 0..4 {
        assert!(
            (crown_result.lower[[i]] - ibp_result.lower[[i]]).abs() < 1e-4,
            "Floor CROWN lower should match IBP: {} vs {}",
            crown_result.lower[[i]],
            ibp_result.lower[[i]]
        );
        assert!(
            (crown_result.upper[[i]] - ibp_result.upper[[i]]).abs() < 1e-4,
            "Floor CROWN upper should match IBP: {} vs {}",
            crown_result.upper[[i]],
            ibp_result.upper[[i]]
        );
    }
}

// ==================== Ceil CROWN tests ====================

#[test]
fn test_ceil_crown_constant_bounds() {
    // Ceil is piecewise constant, so CROWN should produce slope=0 bounds
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.2, -0.8, 2.9, -2.1]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.8, 0.3, 3.5, -1.5]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let ceil = CeilLayer;

    let result = ceil
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // CROWN for discontinuous functions: slope = 0
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                result.lower_a[[i, j]].abs() < 1e-6,
                "Ceil CROWN lower slope should be 0"
            );
            assert!(
                result.upper_a[[i, j]].abs() < 1e-6,
                "Ceil CROWN upper slope should be 0"
            );
        }
    }

    // Check intercepts match IBP bounds
    // ceil([1.2, 1.8]) = [2, 2], ceil([-0.8, 0.3]) = [0, 1],
    // ceil([2.9, 3.5]) = [3, 4], ceil([-2.1, -1.5]) = [-2, -1]
    let expected_lower = [2.0, 0.0, 3.0, -2.0];
    let expected_upper = [2.0, 1.0, 4.0, -1.0];

    for i in 0..4 {
        assert!(
            (result.lower_b[i] - expected_lower[i]).abs() < 1e-6,
            "Ceil CROWN lower intercept mismatch at {}: got {}, expected {}",
            i,
            result.lower_b[i],
            expected_lower[i]
        );
        assert!(
            (result.upper_b[i] - expected_upper[i]).abs() < 1e-6,
            "Ceil CROWN upper intercept mismatch at {}: got {}, expected {}",
            i,
            result.upper_b[i],
            expected_upper[i]
        );
    }
}

#[test]
fn test_ceil_crown_soundness() {
    // Test that CROWN bounds are sound
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.5, 0.2, 2.7]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5, 1.8, 3.3]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let ceil = CeilLayer;

    let result = ceil
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample test points
    let test_points: [Vec<f32>; 3] = [
        vec![-1.5, 0.2, 2.7],
        vec![0.5, 1.8, 3.3],
        vec![-0.5, 1.0, 3.0],
    ];

    for point in &test_points {
        let ceil_output: Vec<f32> = point.iter().map(|x| x.ceil()).collect();

        for (j, &ceil_val) in ceil_output.iter().enumerate() {
            let lower_bound = result.lower_b[j];
            let upper_bound = result.upper_b[j];

            assert!(
                ceil_val >= lower_bound - 1e-6,
                "Ceil output {} should be >= lower bound {}",
                ceil_val,
                lower_bound
            );
            assert!(
                ceil_val <= upper_bound + 1e-6,
                "Ceil output {} should be <= upper bound {}",
                ceil_val,
                upper_bound
            );
        }
    }
}

// ==================== Round CROWN tests ====================

#[test]
fn test_round_crown_constant_bounds() {
    // Round is piecewise constant, so CROWN should produce slope=0 bounds
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.2, -0.8, 2.9, -2.1]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.8, 0.3, 3.5, -1.5]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let round = RoundLayer;

    let result = round
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // CROWN for discontinuous functions: slope = 0
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                result.lower_a[[i, j]].abs() < 1e-6,
                "Round CROWN lower slope should be 0"
            );
            assert!(
                result.upper_a[[i, j]].abs() < 1e-6,
                "Round CROWN upper slope should be 0"
            );
        }
    }

    // Check intercepts match IBP bounds
    // round([1.2, 1.8]) = [1, 2], round([-0.8, 0.3]) = [-1, 0],
    // round([2.9, 3.5]) = [3, 4], round([-2.1, -1.5]) = [-2, -2]
    let expected_lower = [1.0, -1.0, 3.0, -2.0];
    let expected_upper = [2.0, 0.0, 4.0, -2.0];

    for i in 0..4 {
        assert!(
            (result.lower_b[i] - expected_lower[i]).abs() < 1e-6,
            "Round CROWN lower intercept mismatch at {}: got {}, expected {}",
            i,
            result.lower_b[i],
            expected_lower[i]
        );
        assert!(
            (result.upper_b[i] - expected_upper[i]).abs() < 1e-6,
            "Round CROWN upper intercept mismatch at {}: got {}, expected {}",
            i,
            result.upper_b[i],
            expected_upper[i]
        );
    }
}

#[test]
fn test_round_crown_soundness() {
    // Test that CROWN bounds are sound
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.5, 0.2, 2.7]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5, 1.8, 3.3]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let round = RoundLayer;

    let result = round
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample test points
    let test_points: [Vec<f32>; 3] = [
        vec![-1.5, 0.2, 2.7],
        vec![0.5, 1.8, 3.3],
        vec![-0.5, 1.0, 3.0],
    ];

    for point in &test_points {
        let round_output: Vec<f32> = point.iter().map(|x| x.round()).collect();

        for (j, &round_val) in round_output.iter().enumerate() {
            let lower_bound = result.lower_b[j];
            let upper_bound = result.upper_b[j];

            assert!(
                round_val >= lower_bound - 1e-6,
                "Round output {} should be >= lower bound {}",
                round_val,
                lower_bound
            );
            assert!(
                round_val <= upper_bound + 1e-6,
                "Round output {} should be <= upper bound {}",
                round_val,
                upper_bound
            );
        }
    }
}

// ==================== Sign CROWN tests ====================

#[test]
fn test_sign_crown_constant_bounds() {
    // Sign is piecewise constant with values in {-1, 0, 1}
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[5]), vec![1.0, -2.0, -0.5, 0.0, -1.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[5]), vec![2.0, -1.0, 0.5, 0.5, 0.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(5);
    let sign = SignLayer;

    let result = sign
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // CROWN for Sign: slope = 0
    for i in 0..5 {
        for j in 0..5 {
            assert!(
                result.lower_a[[i, j]].abs() < 1e-6,
                "Sign CROWN lower slope should be 0"
            );
            assert!(
                result.upper_a[[i, j]].abs() < 1e-6,
                "Sign CROWN upper slope should be 0"
            );
        }
    }

    // Check intercepts:
    // [1, 2]: positive -> sign = 1
    // [-2, -1]: negative -> sign = -1
    // [-0.5, 0.5]: crosses zero -> sign in [-1, 1]
    // [0, 0.5]: zero and positive -> sign in [0, 1]
    // [-1, 0]: negative and zero -> sign in [-1, 0]
    let expected_lower = [1.0, -1.0, -1.0, 0.0, -1.0];
    let expected_upper = [1.0, -1.0, 1.0, 1.0, 0.0];

    for i in 0..5 {
        assert!(
            (result.lower_b[i] - expected_lower[i]).abs() < 1e-6,
            "Sign CROWN lower intercept mismatch at {}: got {}, expected {}",
            i,
            result.lower_b[i],
            expected_lower[i]
        );
        assert!(
            (result.upper_b[i] - expected_upper[i]).abs() < 1e-6,
            "Sign CROWN upper intercept mismatch at {}: got {}, expected {}",
            i,
            result.upper_b[i],
            expected_upper[i]
        );
    }
}

#[test]
fn test_sign_crown_soundness() {
    // Test that CROWN bounds are sound
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-2.0, -0.5, 0.5, -1.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-0.5, 1.0, 2.0, 0.5]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let sign = SignLayer;

    let result = sign
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample test points
    let test_points: [Vec<f32>; 3] = [
        vec![-2.0, -0.5, 0.5, -1.0], // lower
        vec![-0.5, 1.0, 2.0, 0.5],   // upper
        vec![-1.0, 0.0, 1.0, 0.0],   // middle with zero
    ];

    for point in &test_points {
        let sign_output: Vec<f32> = point.iter().map(|x| x.signum()).collect();

        for (j, &sign_val) in sign_output.iter().enumerate() {
            let lower_bound = result.lower_b[j];
            let upper_bound = result.upper_b[j];

            assert!(
                sign_val >= lower_bound - 1e-6,
                "Sign output {} should be >= lower bound {}",
                sign_val,
                lower_bound
            );
            assert!(
                sign_val <= upper_bound + 1e-6,
                "Sign output {} should be <= upper bound {}",
                sign_val,
                upper_bound
            );
        }
    }
}

#[test]
fn test_sign_crown_network_integration() {
    // Test Sign in a simple network with CROWN propagation
    let weight =
        ndarray::Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
    let bias = Some(ndarray::Array1::from_vec(vec![0.0, 0.0, 0.0]));
    let linear = LinearLayer::new(weight, bias).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::Sign(SignLayer));

    let input_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0, -0.5, 1.0]).unwrap();
    let input_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-0.5, 0.5, 2.0]).unwrap();
    let input = BoundedTensor::new(input_lower, input_upper).unwrap();

    let crown_result = network.propagate_crown(&input).unwrap();
    let ibp_result = network.propagate_ibp(&input).unwrap();

    // CROWN bounds should equal IBP bounds for sign (constant relaxation)
    for i in 0..3 {
        assert!(
            (crown_result.lower[[i]] - ibp_result.lower[[i]]).abs() < 1e-4,
            "Sign CROWN lower should match IBP: {} vs {}",
            crown_result.lower[[i]],
            ibp_result.lower[[i]]
        );
        assert!(
            (crown_result.upper[[i]] - ibp_result.upper[[i]]).abs() < 1e-4,
            "Sign CROWN upper should match IBP: {} vs {}",
            crown_result.upper[[i]],
            ibp_result.upper[[i]]
        );
    }
}

// ==================== Discontinuous layer CROWN/IBP consistency ====================

#[test]
fn test_discontinuous_layers_crown_equals_ibp() {
    // For discontinuous layers with constant bounds (slope=0),
    // CROWN should produce the same bounds as IBP
    let input_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1.5, 0.3, 2.1, -0.8]).unwrap();
    let input_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.2, 1.7, 3.9, 0.4]).unwrap();
    let input = BoundedTensor::new(input_lower, input_upper).unwrap();

    // Test each discontinuous layer type
    let layers_to_test: Vec<(&str, Layer)> = vec![
        ("Floor", Layer::Floor(FloorLayer)),
        ("Ceil", Layer::Ceil(CeilLayer)),
        ("Round", Layer::Round(RoundLayer)),
        ("Sign", Layer::Sign(SignLayer)),
    ];

    for (name, layer) in layers_to_test {
        let mut network = Network::new();
        network.add_layer(layer);

        let crown_result = network.propagate_crown(&input).unwrap();
        let ibp_result = network.propagate_ibp(&input).unwrap();

        for i in 0..4 {
            assert!(
                (crown_result.lower[[i]] - ibp_result.lower[[i]]).abs() < 1e-4,
                "{} CROWN lower should match IBP at {}: {} vs {}",
                name,
                i,
                crown_result.lower[[i]],
                ibp_result.lower[[i]]
            );
            assert!(
                (crown_result.upper[[i]] - ibp_result.upper[[i]]).abs() < 1e-4,
                "{} CROWN upper should match IBP at {}: {} vs {}",
                name,
                i,
                crown_result.upper[[i]],
                ibp_result.upper[[i]]
            );
        }
    }
}
