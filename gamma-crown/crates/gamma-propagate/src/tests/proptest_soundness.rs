//! Property-based soundness tests for bound propagation.
//!
//! These tests verify that IBP bounds are sound: for any concrete input x
//! within the input bounds, the function output f(x) is within the output bounds.
//!
//! Note: A small tolerance (FP_TOLERANCE) is used to account for floating-point
//! precision errors. For strict mathematical soundness, use directed rounding
//! via `propagate_ibp_sound()` which applies `next_down/next_up` to bounds.

use crate::layers::{
    gelu_eval, CosLayer, GELULayer, GeluApproximation, ReLULayer, SigmoidLayer, SinLayer,
    SoftplusLayer, TanhLayer,
};
use crate::*;
use gamma_tensor::BoundedTensor;
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use proptest::prelude::*;

/// Tolerance for floating-point precision in soundness checks.
/// This accounts for FP rounding in both bound computation and function evaluation.
/// For strict soundness guarantees, use directed rounding (`propagate_ibp_sound`).
const FP_TOLERANCE: f32 = 1e-5;

/// Strategy to generate valid interval bounds [lower, upper] where lower <= upper.
/// Constrained to avoid extreme values that could cause overflow.
fn valid_interval(range: f32) -> impl Strategy<Value = (f32, f32)> {
    (-range..range).prop_flat_map(move |a| (-range..range).prop_map(move |b| (a.min(b), a.max(b))))
}

/// Sample points within an interval for soundness verification.
/// Uses clamping to handle FP rounding issues.
fn sample_points(lower: f32, upper: f32, num_samples: usize) -> Vec<f32> {
    if lower == upper {
        return vec![lower];
    }
    (0..=num_samples)
        .map(|i| {
            let t = i as f32 / num_samples as f32;
            let sample = lower + (upper - lower) * t;
            sample.clamp(lower, upper)
        })
        .collect()
}

// =============================================================================
// ACTIVATION FUNCTION SOUNDNESS TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// ReLU IBP soundness: for any x in [l, u], ReLU(x) is in computed bounds.
    #[test]
    fn soundness_relu_ibp((l, u) in valid_interval(100.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let relu = ReLULayer;
        let output = relu.propagate_ibp(&input).unwrap();

        // Verify soundness: ReLU(x) in [output.lower, output.upper] for all x in [l, u]
        for x in sample_points(l, u, 20) {
            let relu_x = x.max(0.0);
            prop_assert!(
                output.lower[[0]] - FP_TOLERANCE <= relu_x && relu_x <= output.upper[[0]] + FP_TOLERANCE,
                "ReLU soundness violation: ReLU({})={} not in [{}, {}]",
                x, relu_x, output.lower[[0]], output.upper[[0]]
            );
        }
    }

    /// Tanh IBP soundness: for any x in [l, u], tanh(x) is in computed bounds.
    #[test]
    fn soundness_tanh_ibp((l, u) in valid_interval(10.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let tanh = TanhLayer::new();
        let output = tanh.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let tanh_x = x.tanh();
            prop_assert!(
                output.lower[[0]] - FP_TOLERANCE <= tanh_x && tanh_x <= output.upper[[0]] + FP_TOLERANCE,
                "Tanh soundness violation: tanh({})={} not in [{}, {}]",
                x, tanh_x, output.lower[[0]], output.upper[[0]]
            );
        }
    }

    /// Sigmoid IBP soundness: for any x in [l, u], sigmoid(x) is in computed bounds.
    #[test]
    fn soundness_sigmoid_ibp((l, u) in valid_interval(10.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let sigmoid = SigmoidLayer::new();
        let output = sigmoid.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let sigmoid_x = 1.0 / (1.0 + (-x).exp());
            prop_assert!(
                output.lower[[0]] - FP_TOLERANCE <= sigmoid_x && sigmoid_x <= output.upper[[0]] + FP_TOLERANCE,
                "Sigmoid soundness violation: sigmoid({})={} not in [{}, {}]",
                x, sigmoid_x, output.lower[[0]], output.upper[[0]]
            );
        }
    }

    /// Softplus IBP soundness: for any x in [l, u], softplus(x) is in computed bounds.
    #[test]
    fn soundness_softplus_ibp((l, u) in valid_interval(10.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let softplus = SoftplusLayer::new();
        let output = softplus.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let softplus_x = (1.0 + x.exp()).ln();
            prop_assert!(
                output.lower[[0]] - FP_TOLERANCE <= softplus_x && softplus_x <= output.upper[[0]] + FP_TOLERANCE,
                "Softplus soundness violation: softplus({})={} not in [{}, {}]",
                x, softplus_x, output.lower[[0]], output.upper[[0]]
            );
        }
    }

    /// Sin IBP soundness: for any x in [l, u], sin(x) is in computed bounds.
    #[test]
    fn soundness_sin_ibp((l, u) in valid_interval(10.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let sin = SinLayer::new();
        let output = sin.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 50) {  // More samples for periodic function
            let sin_x = x.sin();
            prop_assert!(
                output.lower[[0]] - FP_TOLERANCE <= sin_x && sin_x <= output.upper[[0]] + FP_TOLERANCE,
                "Sin soundness violation: sin({})={} not in [{}, {}]",
                x, sin_x, output.lower[[0]], output.upper[[0]]
            );
        }
    }

    /// Cos IBP soundness: for any x in [l, u], cos(x) is in computed bounds.
    #[test]
    fn soundness_cos_ibp((l, u) in valid_interval(10.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let cos = CosLayer::new();
        let output = cos.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 50) {  // More samples for periodic function
            let cos_x = x.cos();
            prop_assert!(
                output.lower[[0]] - FP_TOLERANCE <= cos_x && cos_x <= output.upper[[0]] + FP_TOLERANCE,
                "Cos soundness violation: cos({})={} not in [{}, {}]",
                x, cos_x, output.lower[[0]], output.upper[[0]]
            );
        }
    }

    /// GELU IBP soundness: for any x in [l, u], GELU(x) is in computed bounds.
    #[test]
    fn soundness_gelu_ibp((l, u) in valid_interval(5.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let gelu = GELULayer::new(GeluApproximation::Erf);
        let output = gelu.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 30) {
            let gelu_x = gelu_eval(x, GeluApproximation::Erf);
            prop_assert!(
                output.lower[[0]] - FP_TOLERANCE <= gelu_x && gelu_x <= output.upper[[0]] + FP_TOLERANCE,
                "GELU soundness violation: GELU({})={} not in [{}, {}]",
                x, gelu_x, output.lower[[0]], output.upper[[0]]
            );
        }
    }

    /// GELU Tanh approximation IBP soundness.
    #[test]
    fn soundness_gelu_tanh_ibp((l, u) in valid_interval(5.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let gelu = GELULayer::new(GeluApproximation::Tanh);
        let output = gelu.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 30) {
            let gelu_x = gelu_eval(x, GeluApproximation::Tanh);
            prop_assert!(
                output.lower[[0]] - FP_TOLERANCE <= gelu_x && gelu_x <= output.upper[[0]] + FP_TOLERANCE,
                "GELU-Tanh soundness violation: GELU({})={} not in [{}, {}]",
                x, gelu_x, output.lower[[0]], output.upper[[0]]
            );
        }
    }
}

// =============================================================================
// LINEAR LAYER SOUNDNESS TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Linear layer IBP soundness: for any x in input bounds, Wx+b is in output bounds.
    #[test]
    fn soundness_linear_ibp_2x2(
        w11 in -5.0f32..5.0,
        w12 in -5.0f32..5.0,
        w21 in -5.0f32..5.0,
        w22 in -5.0f32..5.0,
        b1 in -5.0f32..5.0,
        b2 in -5.0f32..5.0,
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
    ) {
        let weight = arr2(&[[w11, w12], [w21, w22]]);
        let bias = arr1(&[b1, b2]);
        let linear = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn()
        ).unwrap();

        let output = linear.propagate_ibp(&input).unwrap();

        // Test multiple concrete points
        for x1 in sample_points(l1, u1, 5) {
            for x2 in sample_points(l2, u2, 5) {
                let x = arr1(&[x1, x2]);
                let y = weight.dot(&x) + &bias;

                for i in 0..2 {
                    prop_assert!(
                        output.lower[[i]] - FP_TOLERANCE <= y[i] && y[i] <= output.upper[[i]] + FP_TOLERANCE,
                        "Linear soundness violation at output {}: y=Wx+b where x=[{}, {}] gives y[{}]={}, not in [{}, {}]",
                        i, x1, x2, i, y[i], output.lower[[i]], output.upper[[i]]
                    );
                }
            }
        }
    }

    /// Linear layer with larger dimensions (5x3).
    #[test]
    fn soundness_linear_ibp_5x3(
        weights in prop::collection::vec(-3.0f32..3.0, 15),  // 5*3 = 15 weights
        biases in prop::collection::vec(-3.0f32..3.0, 5),
        bounds in prop::collection::vec(valid_interval(5.0), 3),
    ) {
        // Reshape weights to 5x3 matrix
        let weight = ndarray::Array2::from_shape_vec((5, 3), weights).unwrap();
        let bias = ndarray::Array1::from_vec(biases);
        let linear = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();

        // Create input bounds
        let lower_vec: Vec<f32> = bounds.iter().map(|(l, _)| *l).collect();
        let upper_vec: Vec<f32> = bounds.iter().map(|(_, u)| *u).collect();
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), lower_vec).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), upper_vec).unwrap()
        ).unwrap();

        let output = linear.propagate_ibp(&input).unwrap();

        // Test corner points and center
        for corner in 0..8 {  // 2^3 corners
            let mut x_vec = Vec::new();
            for (i, (l, u)) in bounds.iter().enumerate() {
                let use_upper = (corner >> i) & 1 == 1;
                x_vec.push(if use_upper { *u } else { *l });
            }
            let x = ndarray::Array1::from_vec(x_vec);
            let y = weight.dot(&x) + &bias;

            for i in 0..5 {
                prop_assert!(
                    output.lower[[i]] - FP_TOLERANCE <= y[i] && y[i] <= output.upper[[i]] + FP_TOLERANCE,
                    "Linear 5x3 soundness violation at output {}: y[{}]={} not in [{}, {}]",
                    i, i, y[i], output.lower[[i]], output.upper[[i]]
                );
            }
        }
    }
}

// =============================================================================
// MULTI-LAYER NETWORK SOUNDNESS TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Two-layer network (Linear -> ReLU) soundness.
    #[test]
    fn soundness_linear_relu_network(
        w11 in -3.0f32..3.0,
        w12 in -3.0f32..3.0,
        w21 in -3.0f32..3.0,
        w22 in -3.0f32..3.0,
        b1 in -3.0f32..3.0,
        b2 in -3.0f32..3.0,
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
    ) {
        let weight = arr2(&[[w11, w12], [w21, w22]]);
        let bias = arr1(&[b1, b2]);
        let linear = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));
        network.add_layer(Layer::ReLU(ReLULayer));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn()
        ).unwrap();

        let output = network.propagate_ibp(&input).unwrap();

        // Test multiple concrete points
        for x1 in sample_points(l1, u1, 4) {
            for x2 in sample_points(l2, u2, 4) {
                let x = arr1(&[x1, x2]);
                let linear_out = weight.dot(&x) + &bias;
                let relu_out = linear_out.mapv(|v| v.max(0.0));

                for i in 0..2 {
                    prop_assert!(
                        output.lower[[i]] - FP_TOLERANCE <= relu_out[i] && relu_out[i] <= output.upper[[i]] + FP_TOLERANCE,
                        "Linear-ReLU network soundness violation: ReLU(Wx+b)[{}]={} not in [{}, {}]",
                        i, relu_out[i], output.lower[[i]], output.upper[[i]]
                    );
                }
            }
        }
    }

    /// Three-layer network (Linear -> ReLU -> Linear) soundness.
    #[test]
    fn soundness_linear_relu_linear_network(
        w1_vec in prop::collection::vec(-2.0f32..2.0, 4),
        b1_vec in prop::collection::vec(-2.0f32..2.0, 2),
        w2_vec in prop::collection::vec(-2.0f32..2.0, 2),
        b2 in -2.0f32..2.0,
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let w1 = ndarray::Array2::from_shape_vec((2, 2), w1_vec).unwrap();
        let b1 = ndarray::Array1::from_vec(b1_vec);
        let w2 = ndarray::Array2::from_shape_vec((1, 2), w2_vec).unwrap();
        let b2_arr = ndarray::Array1::from_vec(vec![b2]);

        let mut network = Network::new();
        network.add_layer(Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(LinearLayer::new(w2.clone(), Some(b2_arr.clone())).unwrap()));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn()
        ).unwrap();

        let output = network.propagate_ibp(&input).unwrap();

        // Test concrete points
        for x1 in sample_points(l1, u1, 4) {
            for x2 in sample_points(l2, u2, 4) {
                let x = arr1(&[x1, x2]);
                let y1 = w1.dot(&x) + &b1;
                let relu_out = y1.mapv(|v| v.max(0.0));
                let final_out = w2.dot(&relu_out) + &b2_arr;

                prop_assert!(
                    output.lower[[0]] - FP_TOLERANCE <= final_out[0] && final_out[0] <= output.upper[[0]] + FP_TOLERANCE,
                    "3-layer network soundness violation: output={} not in [{}, {}]",
                    final_out[0], output.lower[[0]], output.upper[[0]]
                );
            }
        }
    }
}

// =============================================================================
// SOUND PROPAGATION (DIRECTED ROUNDING) TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// propagate_ibp_sound should produce bounds at least as wide as propagate_ibp.
    #[test]
    fn sound_propagation_widens_bounds(
        w11 in -3.0f32..3.0,
        w12 in -3.0f32..3.0,
        w21 in -3.0f32..3.0,
        w22 in -3.0f32..3.0,
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
    ) {
        let weight = arr2(&[[w11, w12], [w21, w22]]);
        let linear = LinearLayer::new(weight, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));
        network.add_layer(Layer::ReLU(ReLULayer));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn()
        ).unwrap();

        let normal = network.propagate_ibp(&input).unwrap();
        let sound = network.propagate_ibp_sound(&input).unwrap();

        for i in 0..2 {
            prop_assert!(
                sound.lower[[i]] <= normal.lower[[i]],
                "Sound lower bound should be <= normal at {}: {} <= {}",
                i, sound.lower[[i]], normal.lower[[i]]
            );
            prop_assert!(
                sound.upper[[i]] >= normal.upper[[i]],
                "Sound upper bound should be >= normal at {}: {} >= {}",
                i, sound.upper[[i]], normal.upper[[i]]
            );
        }
    }
}
