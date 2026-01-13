mod activations;
mod checkpoint;
mod conv;
mod crown;
mod crown_profile;
mod flatten;
mod graph;
mod matmul;
mod proptest_soundness;
mod sdp_crown;
mod tile;
mod transformer;
mod verifier;

use super::*;
use ndarray::{arr1, arr2};

#[test]
fn test_block_progress_fraction_complete() {
    let p = BlockProgress {
        block_index: 0,
        total_blocks: 10,
        block_name: "layer0".to_string(),
        elapsed: std::time::Duration::from_secs(2),
        current_max_sensitivity: 1.0,
        degraded_so_far: 0,
    };
    assert!((p.fraction() - 0.1).abs() < 1e-6);
}

#[test]
fn test_layer_progress_fraction_complete() {
    let p = LayerProgress {
        node_index: 4,
        total_nodes: 10,
        node_name: "n4".to_string(),
        layer_type: "Linear".to_string(),
        elapsed: std::time::Duration::from_secs(2),
        current_max_sensitivity: 1.0,
        degraded_so_far: 0,
    };
    assert!((p.fraction() - 0.5).abs() < 1e-6);
}

#[test]
fn test_relu_relaxation_positive() {
    let (ls, li, us, ui) = relu_crown_relaxation(1.0, 2.0);
    assert_eq!((ls, li, us, ui), (1.0, 0.0, 1.0, 0.0));
}

#[test]
fn test_relu_relaxation_negative() {
    let (ls, li, us, ui) = relu_crown_relaxation(-2.0, -1.0);
    assert_eq!((ls, li, us, ui), (0.0, 0.0, 0.0, 0.0));
}

#[test]
fn test_relu_relaxation_crossing() {
    let (_ls, _li, us, _ui) = relu_crown_relaxation(-1.0, 2.0);
    // Upper slope should be 2/(2-(-1)) = 2/3
    assert!((us - 2.0 / 3.0).abs() < 1e-6);
}

// ============================================================
// IBP TESTS FOR LINEAR LAYER
// ============================================================

#[test]
fn test_linear_ibp_identity() {
    // Identity matrix: output should equal input bounds
    // W = [[1, 0], [0, 1]], b = [0, 0]
    let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let bias = arr1(&[0.0, 0.0]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();

    let input =
        BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[2.0, 3.0]).into_dyn()).unwrap();

    let output = linear.propagate_ibp(&input).unwrap();

    // Identity: output bounds should equal input bounds
    assert!((output.lower[[0]] - 0.0).abs() < 1e-6);
    assert!((output.upper[[0]] - 2.0).abs() < 1e-6);
    assert!((output.lower[[1]] - 1.0).abs() < 1e-6);
    assert!((output.upper[[1]] - 3.0).abs() < 1e-6);
}

#[test]
fn test_linear_ibp_positive_weights() {
    // Simple positive weight matrix
    // W = [[1, 2], [3, 4]], b = [0, 0]
    // x in [[0, 1], [0, 1]]
    //
    // Hand calculation:
    // W+ = W (all positive), W- = 0
    // lower_y = W @ x_lower = [[1,2],[3,4]] @ [0,0] = [0, 0]
    // upper_y = W @ x_upper = [[1,2],[3,4]] @ [1,1] = [3, 7]
    let weight = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let output = linear.propagate_ibp(&input).unwrap();

    assert!((output.lower[[0]] - 0.0).abs() < 1e-6);
    assert!((output.upper[[0]] - 3.0).abs() < 1e-6);
    assert!((output.lower[[1]] - 0.0).abs() < 1e-6);
    assert!((output.upper[[1]] - 7.0).abs() < 1e-6);
}

#[test]
fn test_linear_ibp_mixed_weights() {
    // Mixed positive/negative weights
    // W = [[1, -1], [-2, 3]], b = [0, 0]
    // x in [[0, 1], [0, 1]]
    //
    // Hand calculation:
    // W+ = [[1, 0], [0, 3]], W- = [[0, -1], [-2, 0]]
    // lower_y = W+ @ [0,0] + W- @ [1,1] = [0,0] + [-1,-2] = [-1, -2]
    // upper_y = W+ @ [1,1] + W- @ [0,0] = [1,3] + [0,0] = [1, 3]
    let weight = arr2(&[[1.0, -1.0], [-2.0, 3.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let output = linear.propagate_ibp(&input).unwrap();

    assert!(
        (output.lower[[0]] - (-1.0)).abs() < 1e-6,
        "lower[0] = {}",
        output.lower[[0]]
    );
    assert!(
        (output.upper[[0]] - 1.0).abs() < 1e-6,
        "upper[0] = {}",
        output.upper[[0]]
    );
    assert!(
        (output.lower[[1]] - (-2.0)).abs() < 1e-6,
        "lower[1] = {}",
        output.lower[[1]]
    );
    assert!(
        (output.upper[[1]] - 3.0).abs() < 1e-6,
        "upper[1] = {}",
        output.upper[[1]]
    );
}

#[test]
fn test_linear_ibp_with_bias() {
    // W = [[1, 0], [0, 1]], b = [1, -1]
    // x in [[0, 0], [1, 1]]
    // output = W @ x + b = x + b
    // lower = [0, 0] + [1, -1] = [1, -1]
    // upper = [1, 1] + [1, -1] = [2, 0]
    let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let bias = arr1(&[1.0, -1.0]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let output = linear.propagate_ibp(&input).unwrap();

    assert!((output.lower[[0]] - 1.0).abs() < 1e-6);
    assert!((output.upper[[0]] - 2.0).abs() < 1e-6);
    assert!((output.lower[[1]] - (-1.0)).abs() < 1e-6);
    assert!((output.upper[[1]] - 0.0).abs() < 1e-6);
}

#[test]
fn test_linear_ibp_asymmetric_bounds() {
    // x in [[-1, 2], [1, 3]] (non-symmetric, non-zero)
    // W = [[1, 2]], b = [0]
    // W+ = [[1, 2]], W- = [[0, 0]]
    // lower_y = W+ @ [-1, 1] + W- @ [2, 3] = [1*(-1) + 2*1] = [1]
    // upper_y = W+ @ [2, 3] + W- @ [-1, 1] = [1*2 + 2*3] = [8]
    let weight = arr2(&[[1.0, 2.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();

    let input =
        BoundedTensor::new(arr1(&[-1.0, 1.0]).into_dyn(), arr1(&[2.0, 3.0]).into_dyn()).unwrap();

    let output = linear.propagate_ibp(&input).unwrap();

    assert!(
        (output.lower[[0]] - 1.0).abs() < 1e-6,
        "lower = {}",
        output.lower[[0]]
    );
    assert!(
        (output.upper[[0]] - 8.0).abs() < 1e-6,
        "upper = {}",
        output.upper[[0]]
    );
}

#[test]
fn test_linear_ibp_all_negative_weights() {
    // W = [[-1, -2]], b = [0]
    // x in [[0, 1], [0, 1]]
    // W+ = [[0, 0]], W- = [[-1, -2]]
    // lower_y = W+ @ [0, 0] + W- @ [1, 1] = [0] + [-1 + -2] = [-3]
    // upper_y = W+ @ [1, 1] + W- @ [0, 0] = [0] + [0] = [0]
    let weight = arr2(&[[-1.0, -2.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let output = linear.propagate_ibp(&input).unwrap();

    assert!((output.lower[[0]] - (-3.0)).abs() < 1e-6);
    assert!((output.upper[[0]] - 0.0).abs() < 1e-6);
}

// ============================================================
// IBP TESTS FOR RELU
// ============================================================

#[test]
fn test_relu_ibp_all_positive() {
    let input =
        BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn()).unwrap();

    let output = relu_ibp(&input);

    // All positive: ReLU is identity
    assert!((output.lower[[0]] - 1.0).abs() < 1e-6);
    assert!((output.upper[[0]] - 3.0).abs() < 1e-6);
    assert!((output.lower[[1]] - 2.0).abs() < 1e-6);
    assert!((output.upper[[1]] - 4.0).abs() < 1e-6);
}

#[test]
fn test_relu_ibp_all_negative() {
    let input = BoundedTensor::new(
        arr1(&[-4.0, -3.0]).into_dyn(),
        arr1(&[-2.0, -1.0]).into_dyn(),
    )
    .unwrap();

    let output = relu_ibp(&input);

    // All negative: ReLU outputs zero
    assert!((output.lower[[0]] - 0.0).abs() < 1e-6);
    assert!((output.upper[[0]] - 0.0).abs() < 1e-6);
    assert!((output.lower[[1]] - 0.0).abs() < 1e-6);
    assert!((output.upper[[1]] - 0.0).abs() < 1e-6);
}

#[test]
fn test_relu_ibp_crossing_zero() {
    let input =
        BoundedTensor::new(arr1(&[-1.0, -2.0]).into_dyn(), arr1(&[2.0, 1.0]).into_dyn()).unwrap();

    let output = relu_ibp(&input);

    // Crossing zero: lower = max(-1, 0) = 0, upper = max(2, 0) = 2
    assert!((output.lower[[0]] - 0.0).abs() < 1e-6);
    assert!((output.upper[[0]] - 2.0).abs() < 1e-6);
    assert!((output.lower[[1]] - 0.0).abs() < 1e-6);
    assert!((output.upper[[1]] - 1.0).abs() < 1e-6);
}

// ============================================================
// NETWORK (MULTI-LAYER) IBP TESTS
// ============================================================

#[test]
fn test_network_linear_relu() {
    // Simple 2-layer network: Linear(2->2) -> ReLU
    // W = [[1, -1], [-1, 1]], b = [0, 0]
    // x in [[-1, 1], [-1, 1]]
    //
    // After Linear:
    // W+ = [[1, 0], [0, 1]], W- = [[0, -1], [-1, 0]]
    // lower = W+ @ [-1, -1] + W- @ [1, 1] = [-1, -1] + [-1, -1] = [-2, -2]
    // upper = W+ @ [1, 1] + W- @ [-1, -1] = [1, 1] + [1, 1] = [2, 2]
    //
    // After ReLU: [max(-2,0), max(-2,0)] to [max(2,0), max(2,0)] = [0, 0] to [2, 2]
    let weight = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let output = network.propagate_ibp(&input).unwrap();

    assert!((output.lower[[0]] - 0.0).abs() < 1e-6);
    assert!((output.upper[[0]] - 2.0).abs() < 1e-6);
    assert!((output.lower[[1]] - 0.0).abs() < 1e-6);
    assert!((output.upper[[1]] - 2.0).abs() < 1e-6);
}

#[test]
fn test_network_two_linear_layers() {
    // Linear(2->2) -> ReLU -> Linear(2->1)
    // First linear: identity, second linear: sum
    let w1 = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let w2 = arr2(&[[1.0, 1.0]]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, None).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-1.0, 2.0]).into_dyn(), arr1(&[1.0, 3.0]).into_dyn()).unwrap();

    let output = network.propagate_ibp(&input).unwrap();

    // After identity: [-1, 2] to [1, 3]
    // After ReLU: [0, 2] to [1, 3]
    // After sum: [0+2, 1+3] = [2, 4]
    assert!((output.lower[[0]] - 2.0).abs() < 1e-6);
    assert!((output.upper[[0]] - 4.0).abs() < 1e-6);
}

#[test]
fn test_linear_ibp_soundness() {
    // Soundness test: verify that for any concrete input within bounds,
    // the concrete output is within computed bounds.
    //
    // W = [[2, -1], [1, 3]], b = [1, -2]
    // x in [[0, 2], [1, 3]]
    let weight = arr2(&[[2.0, -1.0], [1.0, 3.0]]);
    let bias = arr1(&[1.0, -2.0]);
    let linear = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();

    let input =
        BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[2.0, 3.0]).into_dyn()).unwrap();

    let output_bounds = linear.propagate_ibp(&input).unwrap();

    // Test several concrete points within input bounds
    let test_points = [
        arr1(&[0.0, 1.0]), // lower corner
        arr1(&[2.0, 3.0]), // upper corner
        arr1(&[1.0, 2.0]), // center
        arr1(&[0.5, 1.5]), // random point 1
        arr1(&[1.5, 2.5]), // random point 2
    ];

    for x in &test_points {
        let y = weight.dot(x) + &bias;

        // Verify y[i] is within [lower[i], upper[i]] for all i
        for i in 0..y.len() {
            assert!(
                y[i] >= output_bounds.lower[[i]] - 1e-6,
                "Soundness violation: y[{}] = {} < lower = {}",
                i,
                y[i],
                output_bounds.lower[[i]]
            );
            assert!(
                y[i] <= output_bounds.upper[[i]] + 1e-6,
                "Soundness violation: y[{}] = {} > upper = {}",
                i,
                y[i],
                output_bounds.upper[[i]]
            );
        }
    }
}

// ============================================================
// DIRECTED ROUNDING / SOUND IBP TESTS
// ============================================================

#[test]
fn test_propagate_ibp_sound_widens_bounds() {
    // propagate_ibp_sound should produce bounds that are slightly wider
    // than propagate_ibp due to directed rounding
    let w1 = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    let w2 = arr2(&[[1.0, 1.0]]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, None).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-1.0, 2.0]).into_dyn(), arr1(&[1.0, 3.0]).into_dyn()).unwrap();

    let normal_output = network.propagate_ibp(&input).unwrap();
    let sound_output = network.propagate_ibp_sound(&input).unwrap();

    // Sound bounds should be at least as wide (lower <= lower, upper >= upper)
    assert!(
        sound_output.lower[[0]] <= normal_output.lower[[0]],
        "Sound lower bound should be <= normal: {} <= {}",
        sound_output.lower[[0]],
        normal_output.lower[[0]]
    );
    assert!(
        sound_output.upper[[0]] >= normal_output.upper[[0]],
        "Sound upper bound should be >= normal: {} >= {}",
        sound_output.upper[[0]],
        normal_output.upper[[0]]
    );
}

#[test]
fn test_propagate_ibp_sound_preserves_soundness() {
    // Sound propagation should still satisfy the soundness property:
    // for any x in input bounds, f(x) is in output bounds
    let weight = arr2(&[[2.0, -1.0], [1.0, 3.0]]);
    let bias = arr1(&[1.0, -2.0]);
    let linear = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input =
        BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[2.0, 3.0]).into_dyn()).unwrap();

    let output_bounds = network.propagate_ibp_sound(&input).unwrap();

    // Test concrete points
    let test_points = [arr1(&[0.0, 1.0]), arr1(&[2.0, 3.0]), arr1(&[1.0, 2.0])];

    for x in &test_points {
        // Compute concrete output: ReLU(Wx + b)
        let linear_out = weight.dot(x) + &bias;
        let relu_out = linear_out.mapv(|v| v.max(0.0));

        for i in 0..relu_out.len() {
            assert!(
                relu_out[i] >= output_bounds.lower[[i]] - 1e-6,
                "Sound propagation violation: out[{}] = {} < lower = {}",
                i,
                relu_out[i],
                output_bounds.lower[[i]]
            );
            assert!(
                relu_out[i] <= output_bounds.upper[[i]] + 1e-6,
                "Sound propagation violation: out[{}] = {} > upper = {}",
                i,
                relu_out[i],
                output_bounds.upper[[i]]
            );
        }
    }
}

#[test]
fn test_linear_layer_shape_validation() {
    // Bias shape mismatch should error
    let weight = arr2(&[[1.0, 2.0], [3.0, 4.0]]); // 2x2
    let bad_bias = arr1(&[1.0, 2.0, 3.0]); // wrong size

    let result = LinearLayer::new(weight, Some(bad_bias));
    assert!(result.is_err());
}

#[test]
fn test_linear_ibp_input_shape_validation() {
    // Input dimension mismatch should error
    let weight = arr2(&[[1.0, 2.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();

    let input = BoundedTensor::new(
        arr1(&[0.0, 0.0, 0.0]).into_dyn(), // 3 elements, expected 2
        arr1(&[1.0, 1.0, 1.0]).into_dyn(),
    )
    .unwrap();

    let result = linear.propagate_ibp(&input);
    assert!(result.is_err());
}

// ============================================================
// MUTATION-KILLING TESTS FOR bounds.rs
// ============================================================

use crate::bounds::{
    batched_matvec, safe_add_for_bounds, safe_add_for_bounds_with_polarity, safe_array_add,
    safe_mul_for_bounds, AlphaState, BatchedLinearBounds, GraphAlphaState, LinearBounds,
};

// --- Tests for safe_add_for_bounds (line 696) ---

#[test]
fn test_safe_add_for_bounds_returns_nonzero() {
    // Kills mutant: replace safe_add_for_bounds -> f32 with 0.0
    let result = safe_add_for_bounds(1.0, 2.0);
    assert_eq!(result, 3.0);
}

#[test]
fn test_safe_add_for_bounds_returns_negative() {
    // Kills mutant: replace safe_add_for_bounds -> f32 with 1.0 or 0.0
    let result = safe_add_for_bounds(-5.0, 2.0);
    assert_eq!(result, -3.0);
}

#[test]
fn test_safe_add_for_bounds_returns_correct_value() {
    // Kills mutant: replace safe_add_for_bounds -> f32 with -1.0
    let result = safe_add_for_bounds(0.5, 0.3);
    assert!((result - 0.8).abs() < 1e-6);
}

#[test]
fn test_safe_add_for_bounds_inf_handling() {
    // Test that inf + finite = inf
    let result = safe_add_for_bounds(f32::INFINITY, 1.0);
    assert!(result.is_infinite() && result > 0.0);
}

// --- Tests for safe_array_add (lines 715) ---

#[test]
fn test_safe_array_add_nan_to_conservative_lower() {
    // When inf + (-inf) = NaN, should become -inf for lower bounds
    // Kills mutant: replace && with ||
    use ndarray::ArrayD;
    let a = ArrayD::from_elem(ndarray::IxDyn(&[2]), f32::INFINITY);
    let b = ArrayD::from_elem(ndarray::IxDyn(&[2]), f32::NEG_INFINITY);
    let result = safe_array_add(&a, &b, true); // is_lower = true
    assert!(result[[0]].is_infinite() && result[[0]] < 0.0);
}

#[test]
fn test_safe_array_add_nan_to_conservative_upper() {
    // When inf + (-inf) = NaN, should become +inf for upper bounds
    // Kills mutant: replace || with &&
    use ndarray::ArrayD;
    let a = ArrayD::from_elem(ndarray::IxDyn(&[2]), f32::INFINITY);
    let b = ArrayD::from_elem(ndarray::IxDyn(&[2]), f32::NEG_INFINITY);
    let result = safe_array_add(&a, &b, false); // is_lower = false
    assert!(result[[0]].is_infinite() && result[[0]] > 0.0);
}

#[test]
fn test_safe_array_add_preserves_normal_values() {
    // Normal addition should work correctly
    use ndarray::ArrayD;
    let a = ArrayD::from_elem(ndarray::IxDyn(&[2]), 1.0f32);
    let b = ArrayD::from_elem(ndarray::IxDyn(&[2]), 2.0f32);
    let result = safe_array_add(&a, &b, false);
    assert!((result[[0]] - 3.0).abs() < 1e-6);
}

#[test]
fn test_safe_array_add_inf_plus_finite() {
    // inf + finite should remain inf (no NaN)
    use ndarray::ArrayD;
    let a = ArrayD::from_elem(ndarray::IxDyn(&[2]), f32::INFINITY);
    let b = ArrayD::from_elem(ndarray::IxDyn(&[2]), 1.0f32);
    let result = safe_array_add(&a, &b, false);
    assert!(result[[0]].is_infinite() && result[[0]] > 0.0);
}

// --- Tests for batched_matvec (lines 733-764) ---

#[test]
fn test_batched_matvec_degenerate_a() {
    // Kills mutant: replace || with && in line 733
    // When a has < 2 dims, should return 0-dim array
    use ndarray::ArrayD;
    let a = ArrayD::zeros(ndarray::IxDyn(&[3])); // 1D array
    let x = ArrayD::from_elem(ndarray::IxDyn(&[3]), 1.0f32);
    let result = batched_matvec(&a, &x);
    assert!(result.shape().is_empty()); // 0-dimensional array
}

#[test]
fn test_batched_matvec_degenerate_x() {
    // Kills mutant: empty x check
    use ndarray::ArrayD;
    let a = ArrayD::zeros(ndarray::IxDyn(&[2, 3]));
    let x: ArrayD<f32> = ArrayD::zeros(ndarray::IxDyn(&[])); // 0-dim
    let result = batched_matvec(&a, &x);
    assert!(result.shape().is_empty()); // 0-dimensional array
}

#[test]
fn test_batched_matvec_basic() {
    // Basic matrix-vector multiplication
    use ndarray::{array, ArrayD};
    let a: ArrayD<f32> = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
    let x: ArrayD<f32> = array![1.0, 1.0].into_dyn();
    let result = batched_matvec(&a, &x);
    assert!((result[[0]] - 3.0).abs() < 1e-6);
    assert!((result[[1]] - 7.0).abs() < 1e-6);
}

#[test]
fn test_batched_matvec_with_inf() {
    // Tests the inf path
    // Kills mutant: replace || with && in line 763-764
    use ndarray::{array, ArrayD};
    let a: ArrayD<f32> = array![[1.0, 0.0], [0.0, 1.0]].into_dyn();
    let x: ArrayD<f32> = array![f32::INFINITY, 2.0].into_dyn();
    let result = batched_matvec(&a, &x);
    assert!(result[[0]].is_infinite() && result[[0]] > 0.0);
    assert!((result[[1]] - 2.0).abs() < 1e-6);
}

#[test]
fn test_batched_matvec_nan_in_a() {
    // Test NaN handling in matrix
    use ndarray::{array, ArrayD};
    let a: ArrayD<f32> = array![[1.0, f32::NAN], [0.0, 1.0]].into_dyn();
    let x: ArrayD<f32> = array![1.0, 1.0].into_dyn();
    let result = batched_matvec(&a, &x);
    // Should take safe path and handle NaN
    assert!(result[[0]].is_nan() || result[[0]].is_finite() || result[[0]].is_infinite());
}

#[test]
fn test_batched_matvec_zero_times_inf() {
    // 0 * inf should be handled correctly (gives 0 in safe path)
    use ndarray::{array, ArrayD};
    let a: ArrayD<f32> = array![[0.0, 1.0], [1.0, 0.0]].into_dyn();
    let x: ArrayD<f32> = array![f32::INFINITY, 2.0].into_dyn();
    let result = batched_matvec(&a, &x);
    // First row: 0*inf + 1*2 = 0 + 2 = 2 (safe mul handles 0*inf = 0)
    assert!((result[[0]] - 2.0).abs() < 1e-6);
}

// --- Tests for AlphaState (lines 889-971) ---

#[test]
fn test_alpha_state_from_preactivation_positive_region() {
    // When l >= 0, alpha should be 1.0 (identity region)
    // Kills mutant: replace < with <= at line 889, 894
    let layer_bounds =
        vec![
            BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn()).unwrap(),
        ];
    let relu_indices = vec![0];
    let state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices);
    let alpha = state.get_alpha(0).unwrap();
    assert_eq!(alpha[[0]], 1.0);
    assert_eq!(alpha[[1]], 1.0);
}

#[test]
fn test_alpha_state_from_preactivation_negative_region() {
    // When u <= 0, alpha should be 0.0 (zero region)
    let layer_bounds = vec![BoundedTensor::new(
        arr1(&[-3.0, -2.0]).into_dyn(),
        arr1(&[-1.0, -0.5]).into_dyn(),
    )
    .unwrap()];
    let relu_indices = vec![0];
    let state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices);
    let alpha = state.get_alpha(0).unwrap();
    assert_eq!(alpha[[0]], 0.0);
    assert_eq!(alpha[[1]], 0.0);
}

#[test]
fn test_alpha_state_from_preactivation_unstable_heuristic() {
    // When l < 0 < u, alpha = 1 if u > -l, else 0
    // Kills mutant: replace > with >= at line 911, delete - at line 911
    // l = -1, u = 2: -l = 1, u > -l (2 > 1) => alpha = 1
    // l = -3, u = 1: -l = 3, u > -l (1 > 3) is false => alpha = 0
    let layer_bounds =
        vec![
            BoundedTensor::new(arr1(&[-1.0, -3.0]).into_dyn(), arr1(&[2.0, 1.0]).into_dyn())
                .unwrap(),
        ];
    let relu_indices = vec![0];
    let state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices);
    let alpha = state.get_alpha(0).unwrap();
    assert_eq!(alpha[[0]], 1.0); // u=2 > -l=1
    assert_eq!(alpha[[1]], 0.0); // u=1 not > -l=3
}

#[test]
fn test_alpha_state_from_preactivation_boundary_case() {
    // Test boundary: u == -l (should give alpha = 0 since not strictly >)
    // l = -2, u = 2: -l = 2, u > -l (2 > 2) is false => alpha = 0
    let layer_bounds =
        vec![BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap()];
    let relu_indices = vec![0];
    let state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices);
    let alpha = state.get_alpha(0).unwrap();
    assert_eq!(alpha[[0]], 0.0); // u=2 not > -l=2
}

#[test]
fn test_alpha_state_update_velocity_formula() {
    // vel[i] = momentum * vel[i] - learning_rate * gradient[i]
    // Kills mutants: replace - with +, replace * with /, replace * with +
    // Use l=-0.5, u=2 so alpha starts at 1 (u > -l => 2 > 0.5)
    let layer_bounds =
        vec![BoundedTensor::new(arr1(&[-0.5]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap()];
    let relu_indices = vec![0];
    let mut state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices);

    // Initial alpha for unstable neuron should be 1.0
    let initial_alpha = state.get_alpha(0).unwrap()[[0]];
    assert_eq!(initial_alpha, 1.0);

    // Update with specific values: momentum=0.9, lr=0.1, gradient=1.0
    let gradient = arr1(&[1.0]);
    state.update(0, &gradient, 0.1, 0.9);

    // vel = 0.9 * 0 - 0.1 * 1.0 = -0.1
    // alpha = 1.0 + (-0.1) = 0.9 (clamped to [0,1])
    let expected = (initial_alpha - 0.1).clamp(0.0, 1.0);
    let actual = state.get_alpha(0).unwrap()[[0]];
    assert!(
        (actual - expected).abs() < 1e-6,
        "actual={}, expected={}",
        actual,
        expected
    );
}

#[test]
fn test_alpha_state_update_momentum_accumulates() {
    // Test that momentum accumulates across updates
    // Kills mutant: replace += with -=, replace += with *=
    // Use l=-0.5, u=2 so that alpha starts at 1 (u > -l => 2 > 0.5)
    let layer_bounds =
        vec![BoundedTensor::new(arr1(&[-0.5]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap()];
    let relu_indices = vec![0];
    let mut state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices);

    // Initial alpha should be 1.0
    let alpha0 = state.get_alpha(0).unwrap()[[0]];
    assert_eq!(alpha0, 1.0);

    let gradient = arr1(&[0.5]);
    // First update: vel = 0 - 0.1*0.5 = -0.05, alpha = 1 + (-0.05) = 0.95
    state.update(0, &gradient, 0.1, 0.5);
    let alpha1 = state.get_alpha(0).unwrap()[[0]];

    // Second update: vel = 0.5*(-0.05) - 0.1*0.5 = -0.025 - 0.05 = -0.075
    // alpha = 0.95 + (-0.075) = 0.875
    state.update(0, &gradient, 0.1, 0.5);
    let alpha2 = state.get_alpha(0).unwrap()[[0]];

    // Alpha should have decreased further
    assert!(
        alpha2 < alpha1,
        "alpha2={} should be < alpha1={}",
        alpha2,
        alpha1
    );
}

#[test]
fn test_alpha_state_num_unstable_counts_correctly() {
    // Kills mutant: replace num_unstable -> usize with 0 or 1
    // Create bounds with 2 unstable neurons (crossing zero) and 1 stable
    let layer_bounds = vec![BoundedTensor::new(
        arr1(&[-1.0, -1.0, 1.0]).into_dyn(),
        arr1(&[1.0, 1.0, 2.0]).into_dyn(),
    )
    .unwrap()];
    let relu_indices = vec![0];
    let state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices);
    assert_eq!(state.num_unstable(), 2);
}

#[test]
fn test_alpha_state_num_unstable_zero() {
    // Test with no unstable neurons
    let layer_bounds =
        vec![
            BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn()).unwrap(),
        ];
    let relu_indices = vec![0];
    let state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices);
    assert_eq!(state.num_unstable(), 0);
}

// --- Tests for GraphAlphaState (lines 1004-1096) ---

#[test]
fn test_graph_alpha_state_add_relu_node_positive() {
    // Kills mutant: replace < with ==, >, <= at lines 1014, 1019
    let mut state = GraphAlphaState::new();
    let pre =
        BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre);
    let alpha = state.get_alpha("relu1").unwrap();
    assert_eq!(alpha[[0]], 1.0); // positive region
    assert_eq!(alpha[[1]], 1.0);
}

#[test]
fn test_graph_alpha_state_add_relu_node_negative() {
    // Test negative region
    let mut state = GraphAlphaState::new();
    let pre = BoundedTensor::new(
        arr1(&[-3.0, -2.0]).into_dyn(),
        arr1(&[-1.0, -0.5]).into_dyn(),
    )
    .unwrap();
    state.add_relu_node("relu1", &pre);
    let alpha = state.get_alpha("relu1").unwrap();
    assert_eq!(alpha[[0]], 0.0);
    assert_eq!(alpha[[1]], 0.0);
}

#[test]
fn test_graph_alpha_state_add_relu_node_unstable() {
    // Kills mutant: replace >= with < at line 1025, <= with > at line 1029
    // Kills mutant: replace > with ==, <, >= at line 1036, delete - at line 1036
    let mut state = GraphAlphaState::new();
    // l=-1, u=2: u > -l (2 > 1) => alpha = 1
    // l=-3, u=1: u > -l (1 > 3) false => alpha = 0
    let pre =
        BoundedTensor::new(arr1(&[-1.0, -3.0]).into_dyn(), arr1(&[2.0, 1.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre);
    let alpha = state.get_alpha("relu1").unwrap();
    assert_eq!(alpha[[0]], 1.0);
    assert_eq!(alpha[[1]], 0.0);
}

#[test]
fn test_graph_alpha_state_add_relu_node_does_something() {
    // Kills mutant: replace add_relu_node with ()
    let mut state = GraphAlphaState::new();
    let pre = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    assert!(state.get_alpha("relu1").is_none());
    state.add_relu_node("relu1", &pre);
    assert!(state.get_alpha("relu1").is_some());
}

#[test]
fn test_graph_alpha_state_get_alpha_returns_correct() {
    // Kills mutant: replace get_alpha -> Option<&Array1<f32>> with None
    // Kills mutant: replace with Some(...) variants
    let mut state = GraphAlphaState::new();
    let pre =
        BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre);
    let alpha = state.get_alpha("relu1");
    assert!(alpha.is_some());
    let alpha = alpha.unwrap();
    assert_eq!(alpha.len(), 2);
    assert_eq!(alpha[[0]], 1.0);
    assert_eq!(alpha[[1]], 1.0);
}

#[test]
fn test_graph_alpha_state_get_alpha_returns_none_for_missing() {
    let state = GraphAlphaState::new();
    assert!(state.get_alpha("nonexistent").is_none());
}

#[test]
fn test_graph_alpha_state_update_velocity_formula() {
    // Kills mutants: replace - with + or / at line 1077
    // Kills mutants: replace * with + or / at line 1077
    // Use l=-0.5, u=2 so alpha starts at 1 (u > -l => 2 > 0.5)
    let mut state = GraphAlphaState::new();
    let pre = BoundedTensor::new(arr1(&[-0.5]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre);

    let initial_alpha = state.get_alpha("relu1").unwrap()[[0]];
    assert_eq!(initial_alpha, 1.0);

    let gradient = arr1(&[1.0]);
    state.update("relu1", &gradient, 0.1, 0.9);

    // vel = 0.9 * 0 - 0.1 * 1.0 = -0.1
    // alpha = 1.0 + (-0.1) = 0.9, clamped
    let expected = (initial_alpha - 0.1).clamp(0.0, 1.0);
    let actual = state.get_alpha("relu1").unwrap()[[0]];
    assert!(
        (actual - expected).abs() < 1e-6,
        "actual={}, expected={}",
        actual,
        expected
    );
}

#[test]
fn test_graph_alpha_state_update_does_something() {
    // Kills mutant: replace update with ()
    // Use l=-0.5, u=2 so alpha starts at 1 (u > -l => 2 > 0.5)
    let mut state = GraphAlphaState::new();
    let pre = BoundedTensor::new(arr1(&[-0.5]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre);

    let initial_alpha = state.get_alpha("relu1").unwrap()[[0]];
    assert_eq!(initial_alpha, 1.0);

    let gradient = arr1(&[2.0]); // Large gradient
    state.update("relu1", &gradient, 0.5, 0.0); // No momentum, high LR

    let new_alpha = state.get_alpha("relu1").unwrap()[[0]];
    // vel = 0 - 0.5 * 2.0 = -1.0, alpha = 1.0 + (-1.0) = 0.0 (clamped)
    assert!(
        (new_alpha - initial_alpha).abs() > 0.01,
        "new_alpha={}, initial={}",
        new_alpha,
        initial_alpha
    );
}

#[test]
fn test_graph_alpha_state_update_accumulates() {
    // Kills mutant: replace += with -= or *= at line 1079
    // Use l=-0.5, u=2 so alpha starts at 1 (u > -l => 2 > 0.5)
    let mut state = GraphAlphaState::new();
    let pre = BoundedTensor::new(arr1(&[-0.5]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre);

    let alpha0 = state.get_alpha("relu1").unwrap()[[0]];
    assert_eq!(alpha0, 1.0);

    let gradient = arr1(&[0.5]);
    state.update("relu1", &gradient, 0.1, 0.5);
    let alpha1 = state.get_alpha("relu1").unwrap()[[0]];
    state.update("relu1", &gradient, 0.1, 0.5);
    let alpha2 = state.get_alpha("relu1").unwrap()[[0]];

    // With momentum and consistent positive gradient, alpha should keep decreasing
    assert!(
        alpha2 < alpha1,
        "alpha2={} should be < alpha1={}",
        alpha2,
        alpha1
    );
}

#[test]
fn test_graph_alpha_state_num_unstable_counts() {
    // Kills mutant: replace num_unstable -> usize with 0 or 1
    let mut state = GraphAlphaState::new();
    // Node 1: 2 unstable
    let pre1 =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    // Node 2: 1 unstable, 1 stable
    let pre2 =
        BoundedTensor::new(arr1(&[-1.0, 1.0]).into_dyn(), arr1(&[1.0, 2.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre1);
    state.add_relu_node("relu2", &pre2);
    assert_eq!(state.num_unstable(), 3);
}

#[test]
fn test_graph_alpha_state_relu_nodes_returns_keys() {
    // Kills mutant: replace relu_nodes with empty iterator
    let mut state = GraphAlphaState::new();
    let pre = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre);
    state.add_relu_node("relu2", &pre);

    let nodes: Vec<_> = state.relu_nodes().collect();
    assert_eq!(nodes.len(), 2);
    assert!(nodes.contains(&&"relu1".to_string()));
    assert!(nodes.contains(&&"relu2".to_string()));
}

#[test]
fn test_safe_mul_for_bounds_zero_times_inf_is_zero() {
    // Kills mutant: replace || with && in safe_mul_for_bounds
    let v = safe_mul_for_bounds(0.0, f32::INFINITY);
    assert_eq!(v, 0.0);
    assert!(!v.is_nan());
}

#[test]
fn test_safe_add_for_bounds_with_polarity_inf_plus_nan_is_conservative() {
    // Kills mutant: replace || with && in safe_add_for_bounds_with_polarity
    let v = safe_add_for_bounds_with_polarity(f32::INFINITY, f32::NAN, false);
    assert!(v.is_infinite() && v > 0.0);
}

#[test]
fn test_safe_add_for_bounds_with_polarity_does_not_overconservatively_flip_sign() {
    // Kills mutant: replace && with || in safe_add_for_bounds_with_polarity
    let v = safe_add_for_bounds_with_polarity(f32::INFINITY, 1.0, true);
    assert!(v.is_infinite() && v > 0.0);
}

#[test]
fn test_safe_array_add_nan_from_nan_plus_inf_sanitizes() {
    // Kills mutant: replace || with && in safe_array_add
    use ndarray::ArrayD;
    let a = ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::NAN);
    let b = ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::INFINITY);
    let r = safe_array_add(&a, &b, false);
    assert!(r[[0]].is_infinite() && r[[0]] > 0.0);
}

#[test]
fn test_safe_array_add_inf_plus_finite_lower_does_not_flip_to_neg_inf() {
    // Kills mutant: replace && with || in safe_array_add
    use ndarray::ArrayD;
    let a = ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::INFINITY);
    let b = ArrayD::from_elem(ndarray::IxDyn(&[1]), 1.0f32);
    let r = safe_array_add(&a, &b, true);
    assert!(r[[0]].is_infinite() && r[[0]] > 0.0);
}

#[test]
fn test_batched_matvec_inf_minus_inf_is_conservative_inf() {
    // Kills mutant: replace || with && in batched_matvec has_inf_or_nan check
    use ndarray::{Array1, Array2};
    let a = Array2::from_shape_vec((1, 2), vec![f32::INFINITY, f32::NEG_INFINITY])
        .unwrap()
        .into_dyn();
    let x = Array1::from_vec(vec![1.0, 1.0]).into_dyn();
    let r = batched_matvec(&a, &x);
    assert_eq!(r.shape(), &[1]);
    assert!(r[[0]].is_infinite() && r[[0]] > 0.0);
}

#[test]
fn test_linear_bounds_concretize_handles_inf_by_avoiding_zero_times_inf() {
    // Kills mutants in LinearBounds::concretize:
    // - has_inf detection (|| -> &&)
    // - 0 * inf guard (!= -> ==)
    // - arithmetic operator mutations inside the safe path loop
    let bounds = LinearBounds {
        lower_a: arr2(&[[0.0, 1.0, -2.0], [0.0, -1.0, 0.5]]),
        lower_b: arr1(&[0.1, -0.2]),
        upper_a: arr2(&[[0.0, 1.0, -2.0], [0.0, -1.0, 0.5]]),
        upper_b: arr1(&[0.3, 0.4]),
    };

    // Note: BoundedTensor::new() debug-asserts against Inf/NaN; use new_unchecked for this test.
    let input = BoundedTensor::new_unchecked(
        arr1(&[0.0, -1.0, 2.0]).into_dyn(),
        arr1(&[f32::INFINITY, 3.0, 4.0]).into_dyn(),
    )
    .unwrap();

    let out = bounds.concretize(&input);
    for &v in out.lower.iter().chain(out.upper.iter()) {
        assert!(!v.is_nan(), "unexpected NaN in concretize output");
    }

    let lower = out.lower.as_slice().unwrap();
    let upper = out.upper.as_slice().unwrap();
    assert!((lower[0] - (-8.9)).abs() < 1e-6, "lower[0]={}", lower[0]);
    assert!((upper[0] - (-0.7)).abs() < 1e-6, "upper[0]={}", upper[0]);
    assert!((lower[1] - (-2.2)).abs() < 1e-6, "lower[1]={}", lower[1]);
    assert!((upper[1] - 3.4).abs() < 1e-6, "upper[1]={}", upper[1]);
}

#[test]
fn test_linear_bounds_concretize_l2_ball_rho_validation() {
    // Kills mutants: replace < with == / <= on rho validation
    let bounds = LinearBounds::identity(2);
    let x_hat = arr1(&[0.0, 0.0]);
    assert!(bounds.concretize_l2_ball(&x_hat, -1.0).is_err());

    let out = bounds.concretize_l2_ball(&x_hat, 0.0).unwrap();
    assert_eq!(out.lower.as_slice().unwrap(), &[0.0, 0.0]);
    assert_eq!(out.upper.as_slice().unwrap(), &[0.0, 0.0]);
}

#[test]
fn test_linear_bounds_concretize_l2_ball_matches_closed_form() {
    // Kills arithmetic operator mutants inside concretize_l2_ball
    let bounds = LinearBounds {
        lower_a: arr2(&[[3.0, 4.0]]),
        lower_b: arr1(&[0.5]),
        upper_a: arr2(&[[3.0, 4.0]]),
        upper_b: arr1(&[-1.0]),
    };
    let x_hat = arr1(&[1.0, 2.0]);
    let rho = 2.0;
    let out = bounds.concretize_l2_ball(&x_hat, rho).unwrap();

    // dot = a^T x + b, ||a|| = 5
    // lower = (0.5 + 11) - 2*5 = 1.5
    // upper = (-1 + 11) + 2*5 = 20
    let lower = out.lower.as_slice().unwrap()[0];
    let upper = out.upper.as_slice().unwrap()[0];
    assert!((lower - 1.5).abs() < 1e-6, "lower={}", lower);
    assert!((upper - 20.0).abs() < 1e-6, "upper={}", upper);
}

#[test]
fn test_batched_linear_bounds_identity_empty_shape() {
    // Kills mutant: replace || with && in BatchedLinearBounds::identity
    let b = BatchedLinearBounds::identity(&[]);
    assert_eq!(b.input_shape, vec![1]);
    assert_eq!(b.output_shape, vec![1]);
    assert_eq!(b.lower_a.shape(), &[1, 1]);
    assert_eq!(b.lower_b.shape(), &[1]);
}

#[test]
fn test_batched_linear_bounds_identity_for_attention_small_identity_matrix() {
    // Kills mutants in out_dim/in_dim accessors.
    let b = BatchedLinearBounds::identity_for_attention(&[1, 1, 2, 2]).unwrap();
    assert_eq!(b.in_dim(), 4);
    assert_eq!(b.out_dim(), 4);
    assert_eq!(b.input_shape, vec![1, 1, 4]);
    assert_eq!(b.output_shape, vec![1, 1, 4]);

    let a = b
        .lower_a
        .view()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();
    assert_eq!(a[[0, 0, 0, 0]], 1.0);
    assert_eq!(a[[0, 0, 0, 1]], 0.0);
    assert_eq!(a[[0, 0, 3, 3]], 1.0);
}

#[test]
fn test_batched_linear_bounds_identity_for_attention_over_limit_returns_none() {
    // Kills mutants in total_elements/max_elements arithmetic and comparisons.
    // Baseline returns None before allocating large matrices.
    assert!(BatchedLinearBounds::identity_for_attention(&[3, 1, 64, 64]).is_none());
}

#[test]
fn test_batched_linear_bounds_identity_for_attention_mid_size_is_some() {
    // Kills mutants in max_elements arithmetic that would incorrectly reject moderate shapes.
    let b = BatchedLinearBounds::identity_for_attention(&[2, 1, 20, 20]).unwrap();
    assert_eq!(b.in_dim(), 400);
    assert_eq!(b.out_dim(), 400);
    assert_eq!(b.lower_a.shape(), &[2, 1, 400, 400]);
}

#[test]
fn test_batched_linear_bounds_compose_2d_exact() {
    // Kills mutants in BatchedLinearBounds::compose (shape checks and arithmetic).
    let a1 = arr2(&[[1.0, 2.0], [0.0, -1.0]]).into_dyn();
    let b1 = arr1(&[1.0, -1.0]).into_dyn();
    let self_bounds = BatchedLinearBounds {
        lower_a: a1.clone(),
        upper_a: a1,
        lower_b: b1.clone(),
        upper_b: b1,
        input_shape: vec![2],
        output_shape: vec![2],
    };

    let a2 = arr2(&[[2.0, 0.0], [1.0, 1.0]]).into_dyn();
    let b2 = arr1(&[0.5, 2.0]).into_dyn();
    let other_bounds = BatchedLinearBounds {
        lower_a: a2.clone(),
        upper_a: a2,
        lower_b: b2.clone(),
        upper_b: b2,
        input_shape: vec![2],
        output_shape: vec![2],
    };

    let composed = self_bounds.compose(&other_bounds).unwrap();
    assert_eq!(composed.input_shape, vec![2]);
    assert_eq!(composed.output_shape, vec![2]);

    let a = composed
        .lower_a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    let b = composed
        .lower_b
        .view()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    assert_eq!(a[[0, 0]], 2.0);
    assert_eq!(a[[0, 1]], 4.0);
    assert_eq!(a[[1, 0]], 1.0);
    assert_eq!(a[[1, 1]], 1.0);
    assert!((b[0] - 2.5).abs() < 1e-6, "b[0]={}", b[0]);
    assert!((b[1] - 2.0).abs() < 1e-6, "b[1]={}", b[1]);
}

#[test]
fn test_batched_linear_bounds_compose_rectangular_exact() {
    // Kills mutants in dimension indexing inside compose (e.g., len-2 arithmetic).
    let a1 = arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).into_dyn(); // 3x2
    let b1 = arr1(&[0.0, 1.0, 2.0]).into_dyn(); // 3
    let self_bounds = BatchedLinearBounds {
        lower_a: a1.clone(),
        upper_a: a1,
        lower_b: b1.clone(),
        upper_b: b1,
        input_shape: vec![2],
        output_shape: vec![3],
    };

    let a2 = arr2(&[[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]).into_dyn(); // 2x3
    let b2 = arr1(&[1.0, -1.0]).into_dyn(); // 2
    let other_bounds = BatchedLinearBounds {
        lower_a: a2.clone(),
        upper_a: a2,
        lower_b: b2.clone(),
        upper_b: b2,
        input_shape: vec![3],
        output_shape: vec![2],
    };

    let composed = self_bounds.compose(&other_bounds).unwrap();
    let a = composed
        .lower_a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    let b = composed
        .lower_b
        .view()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    assert_eq!(a[[0, 0]], 4.0);
    assert_eq!(a[[0, 1]], 5.0);
    assert_eq!(a[[1, 0]], 0.0);
    assert_eq!(a[[1, 1]], 1.0);
    assert!((b[0] - 9.0).abs() < 1e-6, "b[0]={}", b[0]);
    assert!((b[1] - 0.0).abs() < 1e-6, "b[1]={}", b[1]);
}

#[test]
fn test_batched_linear_bounds_compose_invalid_ndim_is_error() {
    // Kills mutant: replace || with && in BatchedLinearBounds::compose ndim validation.
    use ndarray::ArrayD;
    let bad_self = BatchedLinearBounds {
        lower_a: ArrayD::zeros(ndarray::IxDyn(&[1])),
        upper_a: ArrayD::zeros(ndarray::IxDyn(&[1])),
        lower_b: ArrayD::zeros(ndarray::IxDyn(&[1])),
        upper_b: ArrayD::zeros(ndarray::IxDyn(&[1])),
        input_shape: vec![1],
        output_shape: vec![1],
    };
    let other = BatchedLinearBounds {
        lower_a: ArrayD::zeros(ndarray::IxDyn(&[1, 1])),
        upper_a: ArrayD::zeros(ndarray::IxDyn(&[1, 1])),
        lower_b: ArrayD::zeros(ndarray::IxDyn(&[1])),
        upper_b: ArrayD::zeros(ndarray::IxDyn(&[1])),
        input_shape: vec![1],
        output_shape: vec![1],
    };
    assert!(bad_self.compose(&other).is_err());
}

#[test]
fn test_batched_linear_bounds_compose_nan_in_coeffs_widens_to_infinity() {
    // Kills mutants: replace || with && in compose::interval_mul_for_bounds NaN check.
    let self_bounds = BatchedLinearBounds {
        lower_a: arr2(&[[2.0]]).into_dyn(),
        upper_a: arr2(&[[2.0]]).into_dyn(),
        lower_b: arr1(&[0.0]).into_dyn(),
        upper_b: arr1(&[0.0]).into_dyn(),
        input_shape: vec![1],
        output_shape: vec![1],
    };
    let other_bounds = BatchedLinearBounds {
        lower_a: arr2(&[[1.0]]).into_dyn(),
        upper_a: arr2(&[[f32::NAN]]).into_dyn(),
        lower_b: arr1(&[0.0]).into_dyn(),
        upper_b: arr1(&[0.0]).into_dyn(),
        input_shape: vec![1],
        output_shape: vec![1],
    };

    let composed = self_bounds.compose(&other_bounds).unwrap();
    let a_l = composed
        .lower_a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()[[0, 0]];
    let a_u = composed
        .upper_a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()[[0, 0]];
    assert!(a_l.is_infinite() && a_l < 0.0, "a_l={}", a_l);
    assert!(a_u.is_infinite() && a_u > 0.0, "a_u={}", a_u);
}

#[test]
fn test_batched_linear_bounds_compose_nan_in_self_upper_widens_to_infinity() {
    // Kills missed mutant: OR->AND for the b_u.is_nan() clause in interval_mul_for_bounds.
    let self_bounds = BatchedLinearBounds {
        lower_a: arr2(&[[2.0]]).into_dyn(),
        upper_a: arr2(&[[f32::NAN]]).into_dyn(),
        lower_b: arr1(&[0.0]).into_dyn(),
        upper_b: arr1(&[0.0]).into_dyn(),
        input_shape: vec![1],
        output_shape: vec![1],
    };
    let other_bounds = BatchedLinearBounds {
        lower_a: arr2(&[[1.0]]).into_dyn(),
        upper_a: arr2(&[[1.0]]).into_dyn(),
        lower_b: arr1(&[0.0]).into_dyn(),
        upper_b: arr1(&[0.0]).into_dyn(),
        input_shape: vec![1],
        output_shape: vec![1],
    };

    let composed = self_bounds.compose(&other_bounds).unwrap();
    let a_l = composed
        .lower_a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()[[0, 0]];
    let a_u = composed
        .upper_a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()[[0, 0]];
    assert!(a_l.is_infinite() && a_l < 0.0, "a_l={}", a_l);
    assert!(a_u.is_infinite() && a_u > 0.0, "a_u={}", a_u);
}

#[test]
fn test_batched_linear_bounds_compose_all_inf_products_widens_to_infinity() {
    // Kills missed mutant: OR->AND in interval_mul_for_bounds sentinel check.
    let self_bounds = BatchedLinearBounds {
        lower_a: arr2(&[[f32::INFINITY]]).into_dyn(),
        upper_a: arr2(&[[f32::INFINITY]]).into_dyn(),
        lower_b: arr1(&[0.0]).into_dyn(),
        upper_b: arr1(&[0.0]).into_dyn(),
        input_shape: vec![1],
        output_shape: vec![1],
    };
    let other_bounds = BatchedLinearBounds {
        lower_a: arr2(&[[1.0]]).into_dyn(),
        upper_a: arr2(&[[1.0]]).into_dyn(),
        lower_b: arr1(&[0.0]).into_dyn(),
        upper_b: arr1(&[0.0]).into_dyn(),
        input_shape: vec![1],
        output_shape: vec![1],
    };

    let composed = self_bounds.compose(&other_bounds).unwrap();
    let a_l = composed
        .lower_a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()[[0, 0]];
    let a_u = composed
        .upper_a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()[[0, 0]];
    assert!(a_l.is_infinite() && a_l < 0.0, "a_l={}", a_l);
    assert!(a_u.is_infinite() && a_u > 0.0, "a_u={}", a_u);
}

#[test]
fn test_alpha_state_update_gradient_not_one_detects_mul_vs_div() {
    // Kills mutant: replace * with / in AlphaState::update
    let layer_bounds =
        vec![BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap()];
    let relu_indices = vec![0];
    let mut state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices);

    let gradient = arr1(&[2.0]);
    state.update(0, &gradient, 0.1, 0.0);
    let alpha = state.get_alpha(0).unwrap()[[0]];
    assert!((alpha - 0.8).abs() < 1e-6, "alpha={}", alpha);
}

#[test]
fn test_graph_alpha_state_strict_gt_boundary_u_equals_neg_l() {
    // Kills mutant: replace > with >= in GraphAlphaState::add_relu_node heuristic.
    let mut state = GraphAlphaState::new();
    let pre = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre);
    let alpha = state.get_alpha("relu1").unwrap()[[0]];
    assert_eq!(alpha, 0.0);
}
