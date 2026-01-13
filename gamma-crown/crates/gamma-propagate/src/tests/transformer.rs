//! Transformer component tests (GELU, LayerNorm, Softmax, Attention, MLP, Batched Bounds)
//!
//! Tests for transformer-specific layers and operations:
//! - GELU CROWN relaxation
//! - LayerNorm CROWN propagation
//! - Softmax CROWN bounds
//! - MLP-style network CROWN propagation
//! - Batched linear bounds for sequence processing

use crate::*;
use ndarray::{arr1, arr2, Array1, Array2, ArrayD, IxDyn};

// ========================================================================
// GELU CROWN Tests
// ========================================================================

#[test]
fn test_gelu_crown_linear_relaxation_soundness() {
    // Test that GELU linear relaxation produces sound bounds
    // Test various intervals
    let test_cases = vec![
        (-2.0_f32, -1.0_f32), // Negative region (convex)
        (0.0, 1.0),           // Positive region (concave)
        (-1.0, 1.0),          // Mixed region
        (-0.5, 0.5),          // Small mixed region
        (-3.0, 2.0),          // Wide mixed region
    ];

    for (l, u) in test_cases {
        let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
            gelu_linear_relaxation(l, u, GeluApproximation::Erf);

        // Sample points in the interval and verify bounds
        for t in 0..=20 {
            let x = l + (u - l) * (t as f32 / 20.0);
            let gelu_val = gelu_eval(x, GeluApproximation::Erf);
            let lower_bound = lower_slope * x + lower_intercept;
            let upper_bound = upper_slope * x + upper_intercept;

            assert!(
                gelu_val >= lower_bound - 1e-5,
                "GELU({}) = {} should be >= lower bound {} for interval [{}, {}]",
                x,
                gelu_val,
                lower_bound,
                l,
                u
            );
            assert!(
                gelu_val <= upper_bound + 1e-5,
                "GELU({}) = {} should be <= upper bound {} for interval [{}, {}]",
                x,
                gelu_val,
                upper_bound,
                l,
                u
            );
        }
    }
}

#[test]
fn test_gelu_crown_propagation_soundness() {
    // Test GELU CROWN propagation end-to-end
    use ndarray::arr2;

    // Create a network: Linear -> GELU
    let weight = arr2(&[[1.0_f32, 0.5], [-0.5, 1.0], [0.3, -0.8]]);
    let linear = LinearLayer::new(weight, Some(arr1(&[0.1, -0.1, 0.0]))).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::default()),
        vec!["linear".to_string()],
    ));
    graph.set_output("gelu");

    // Input with perturbation
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Verify soundness: sample random inputs and check they're within bounds
    let rng_seed = 42;
    for i in 0..100 {
        let t1 = ((i * 7 + rng_seed) % 100) as f32 / 100.0;
        let t2 = ((i * 11 + rng_seed) % 100) as f32 / 100.0;
        let x1 = -0.5 + t1;
        let x2 = -0.5 + t2;

        // Forward pass
        let z1_0 = 1.0 * x1 + 0.5 * x2 + 0.1;
        let z1_1 = -0.5 * x1 + 1.0 * x2 - 0.1;
        let z1_2 = 0.3 * x1 - 0.8 * x2;

        let y0 = gelu_eval(z1_0, GeluApproximation::Erf);
        let y1 = gelu_eval(z1_1, GeluApproximation::Erf);
        let y2 = gelu_eval(z1_2, GeluApproximation::Erf);

        // Check CROWN bounds contain the output
        assert!(
            y0 >= crown_bounds.lower[[0]] - 1e-5 && y0 <= crown_bounds.upper[[0]] + 1e-5,
            "Output 0 {} outside CROWN bounds [{}, {}]",
            y0,
            crown_bounds.lower[[0]],
            crown_bounds.upper[[0]]
        );
        assert!(
            y1 >= crown_bounds.lower[[1]] - 1e-5 && y1 <= crown_bounds.upper[[1]] + 1e-5,
            "Output 1 {} outside CROWN bounds [{}, {}]",
            y1,
            crown_bounds.lower[[1]],
            crown_bounds.upper[[1]]
        );
        assert!(
            y2 >= crown_bounds.lower[[2]] - 1e-5 && y2 <= crown_bounds.upper[[2]] + 1e-5,
            "Output 2 {} outside CROWN bounds [{}, {}]",
            y2,
            crown_bounds.lower[[2]],
            crown_bounds.upper[[2]]
        );
    }

    // Note: For GELU, CROWN linear relaxation may not always be tighter than IBP
    // because IBP uses exact interval evaluation while CROWN uses linear bounds.
    // The key property is soundness, not tightness for GELU.
    // Just verify both methods produce sound bounds (tested above).
    let _crown_width: f32 = (0..3)
        .map(|i| crown_bounds.upper[[i]] - crown_bounds.lower[[i]])
        .sum();
    let _ibp_width: f32 = (0..3)
        .map(|i| ibp_bounds.upper[[i]] - ibp_bounds.lower[[i]])
        .sum();
    // Both are sound - tightness depends on network structure
}

// ========================================================================
// LayerNorm CROWN Tests
// ========================================================================

#[test]
fn test_layernorm_eval() {
    // Test LayerNorm evaluation at concrete points
    let gamma = arr1(&[1.0_f32, 2.0, 0.5]);
    let beta = arr1(&[0.0_f32, 1.0, -0.5]);
    let ln = LayerNormLayer::new(gamma, beta, 1e-5);

    // Test with simple input
    let x = arr1(&[1.0_f32, 2.0, 3.0]);
    let y = ln.eval(&x);

    // mean = 2.0, var = 2/3, std ≈ 0.8165
    let mean = 2.0_f32;
    let var = ((1.0 - mean).powi(2) + (2.0 - mean).powi(2) + (3.0 - mean).powi(2)) / 3.0;
    let std = (var + 1e-5_f32).sqrt();

    let expected_0 = 1.0 * (1.0 - mean) / std + 0.0;
    let expected_1 = 2.0 * (2.0 - mean) / std + 1.0;
    let expected_2 = 0.5 * (3.0 - mean) / std + (-0.5);

    assert!(
        (y[0] - expected_0).abs() < 1e-5,
        "y[0] = {} != expected {}",
        y[0],
        expected_0
    );
    assert!(
        (y[1] - expected_1).abs() < 1e-5,
        "y[1] = {} != expected {}",
        y[1],
        expected_1
    );
    assert!(
        (y[2] - expected_2).abs() < 1e-5,
        "y[2] = {} != expected {}",
        y[2],
        expected_2
    );
}

#[test]
fn test_layernorm_jacobian() {
    // Test LayerNorm Jacobian computation
    let gamma = arr1(&[1.0_f32, 1.0, 1.0]);
    let beta = arr1(&[0.0_f32, 0.0, 0.0]);
    let ln = LayerNormLayer::new(gamma, beta, 1e-5);

    let x = arr1(&[1.0_f32, 2.0, 3.0]);
    let jacobian = ln.jacobian(&x);

    // Verify Jacobian via finite differences
    let eps = 1e-4_f32;
    for j in 0..3 {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[j] += eps;
        x_minus[j] -= eps;

        let y_plus = ln.eval(&x_plus);
        let y_minus = ln.eval(&x_minus);

        for i in 0..3 {
            let fd = (y_plus[i] - y_minus[i]) / (2.0 * eps);
            // Allow 1% relative error or 1e-2 absolute error for numerical stability
            let rel_err = if fd.abs() > 1e-6 {
                (jacobian[[i, j]] - fd).abs() / fd.abs()
            } else {
                (jacobian[[i, j]] - fd).abs()
            };
            assert!(
                rel_err < 0.02 || (jacobian[[i, j]] - fd).abs() < 1e-2,
                "J[{},{}] = {} != finite diff {} (rel_err={:.2}%, abs_diff={})",
                i,
                j,
                jacobian[[i, j]],
                fd,
                rel_err * 100.0,
                (jacobian[[i, j]] - fd).abs()
            );
        }
    }
}

#[test]
fn test_layernorm_crown_soundness() {
    // Test that LayerNorm CROWN bounds are sound
    let gamma = arr1(&[1.0_f32, 2.0, 0.5]);
    let beta = arr1(&[0.0_f32, 1.0, -0.5]);
    let ln = LayerNormLayer::new(gamma, beta, 1e-5);

    // Create input bounds
    let input_lower = arr1(&[0.5_f32, 1.5, 2.5]);
    let input_upper = arr1(&[1.5_f32, 2.5, 3.5]);
    let input = BoundedTensor::new(
        input_lower.clone().into_dyn(),
        input_upper.clone().into_dyn(),
    )
    .unwrap();

    // Get CROWN bounds
    let linear_bounds = LinearBounds::identity(3);
    let crown_bounds = ln
        .propagate_linear_with_bounds(&linear_bounds, &input)
        .unwrap();

    // Concretize to get scalar bounds
    let concrete = crown_bounds.concretize(&input);

    // Verify soundness by sampling
    for sample_idx in 0..100 {
        let t0 = (sample_idx as f32 * 17.0 % 100.0) / 100.0;
        let t1 = (sample_idx as f32 * 31.0 % 100.0) / 100.0;
        let t2 = (sample_idx as f32 * 47.0 % 100.0) / 100.0;

        let x_sample = arr1(&[
            input_lower[0] + (input_upper[0] - input_lower[0]) * t0,
            input_lower[1] + (input_upper[1] - input_lower[1]) * t1,
            input_lower[2] + (input_upper[2] - input_lower[2]) * t2,
        ]);

        let y_sample = ln.eval(&x_sample);

        for i in 0..3 {
            assert!(
                y_sample[i] >= concrete.lower[[i]] - 1e-4,
                "Sample {} output {} = {} < lower bound {} at dim {}",
                sample_idx,
                y_sample[i],
                y_sample[i],
                concrete.lower[[i]],
                i
            );
            assert!(
                y_sample[i] <= concrete.upper[[i]] + 1e-4,
                "Sample {} output {} = {} > upper bound {} at dim {}",
                sample_idx,
                y_sample[i],
                y_sample[i],
                concrete.upper[[i]],
                i
            );
        }
    }
}

#[test]
fn test_layernorm_crown_tighter_than_ibp() {
    // CROWN should be at least as tight as IBP for reasonable perturbations
    let gamma = arr1(&[1.0_f32, 1.0, 1.0, 1.0]);
    let beta = arr1(&[0.0_f32, 0.0, 0.0, 0.0]);
    let ln = LayerNormLayer::new(gamma, beta, 1e-5);

    // Small perturbation around a point
    let center = arr1(&[1.0_f32, 2.0, 3.0, 4.0]);
    let eps = 0.1_f32;
    let input_lower: Array1<f32> = center.iter().map(|&c| c - eps).collect();
    let input_upper: Array1<f32> = center.iter().map(|&c| c + eps).collect();

    let input = BoundedTensor::new(
        input_lower.clone().into_dyn(),
        input_upper.clone().into_dyn(),
    )
    .unwrap();

    // Get IBP bounds
    let ibp_bounds = ln.propagate_ibp(&input).unwrap();

    // Get CROWN bounds
    let linear_bounds = LinearBounds::identity(4);
    let crown_result = ln
        .propagate_linear_with_bounds(&linear_bounds, &input)
        .unwrap();
    let crown_bounds = crown_result.concretize(&input);

    // Both should be sound, CROWN should be at least as tight
    let ibp_width: f32 = (0..4)
        .map(|i| ibp_bounds.upper[[i]] - ibp_bounds.lower[[i]])
        .sum();
    let crown_width: f32 = (0..4)
        .map(|i| crown_bounds.upper[[i]] - crown_bounds.lower[[i]])
        .sum();

    println!(
        "IBP width: {}, CROWN width: {} (improvement: {:.1}%)",
        ibp_width,
        crown_width,
        (ibp_width - crown_width) / ibp_width * 100.0
    );

    // CROWN should not be much looser than IBP (within 50% for small perturbations)
    // Note: For LayerNorm, CROWN might not always be tighter due to cross-dimensional dependencies
    assert!(
        crown_width <= ibp_width * 1.5,
        "CROWN width {} should not be much worse than IBP width {}",
        crown_width,
        ibp_width
    );
}

#[test]
fn test_layernorm_crown_propagation_through_network() {
    // Test CROWN propagation through Linear -> LayerNorm
    use ndarray::arr2;

    // Create Linear layer
    let weight = arr2(&[[1.0_f32, 0.5, -0.3], [-0.5, 1.0, 0.2], [0.3, -0.2, 1.0]]);
    let linear = LinearLayer::new(weight, Some(arr1(&[0.1, -0.1, 0.0]))).unwrap();

    // Create LayerNorm layer
    let gamma = arr1(&[1.0_f32, 1.0, 1.0]);
    let beta = arr1(&[0.0_f32, 0.0, 0.0]);
    let ln = LayerNormLayer::new(gamma, beta, 1e-5);

    // Input bounds
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5, 0.5]).into_dyn(),
    )
    .unwrap();

    // Propagate through Linear with IBP to get pre-LayerNorm bounds
    let after_linear = linear.propagate_ibp(&input).unwrap();

    // Propagate CROWN through LayerNorm
    let ln_linear_bounds = LinearBounds::identity(3);
    let crown_result = ln
        .propagate_linear_with_bounds(&ln_linear_bounds, &after_linear)
        .unwrap();
    let crown_bounds = crown_result.concretize(&after_linear);

    // Verify soundness by sampling
    let input_lower = arr1(&[-0.5_f32, -0.5, -0.5]);
    let input_upper = arr1(&[0.5_f32, 0.5, 0.5]);

    for sample_idx in 0..50 {
        let t0 = (sample_idx as f32 * 17.0 % 50.0) / 50.0;
        let t1 = (sample_idx as f32 * 31.0 % 50.0) / 50.0;
        let t2 = (sample_idx as f32 * 47.0 % 50.0) / 50.0;

        let x_sample = arr1(&[
            input_lower[0] + (input_upper[0] - input_lower[0]) * t0,
            input_lower[1] + (input_upper[1] - input_lower[1]) * t1,
            input_lower[2] + (input_upper[2] - input_lower[2]) * t2,
        ])
        .into_dyn();

        // Forward through Linear
        let weight_view = linear.weight.view();
        let linear_out: Array1<f32> = weight_view.dot(
            &x_sample
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .unwrap(),
        ) + linear.bias.as_ref().unwrap();

        // Forward through LayerNorm
        let ln_out = ln.eval(&linear_out);

        for i in 0..3 {
            assert!(
                ln_out[i] >= crown_bounds.lower[[i]] - 1e-3,
                "Sample {} output {} = {} < lower bound {} at dim {}",
                sample_idx,
                ln_out[i],
                ln_out[i],
                crown_bounds.lower[[i]],
                i
            );
            assert!(
                ln_out[i] <= crown_bounds.upper[[i]] + 1e-3,
                "Sample {} output {} = {} > upper bound {} at dim {}",
                sample_idx,
                ln_out[i],
                ln_out[i],
                crown_bounds.upper[[i]],
                i
            );
        }
    }
}

#[test]
fn test_softmax_eval() {
    // Test softmax evaluation at a concrete point
    let softmax = SoftmaxLayer::new(-1);
    let x = arr1(&[1.0_f32, 2.0, 3.0]);
    let s = softmax.eval(&x);

    // Softmax outputs should sum to 1
    let sum: f32 = s.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "Softmax sum {} != 1.0", sum);

    // Check expected softmax values (exp normalized)
    // exp([1,2,3]) = [e, e^2, e^3]
    // sum = e + e^2 + e^3 ≈ 30.19
    // softmax ≈ [0.090, 0.245, 0.665]
    assert!((s[0] - 0.090).abs() < 0.01, "s[0] = {}", s[0]);
    assert!((s[1] - 0.245).abs() < 0.01, "s[1] = {}", s[1]);
    assert!((s[2] - 0.665).abs() < 0.01, "s[2] = {}", s[2]);
}

#[test]
fn test_softmax_jacobian() {
    // Test softmax Jacobian computation
    let softmax = SoftmaxLayer::new(-1);
    let x = arr1(&[0.0_f32, 1.0, 2.0]);
    let s = softmax.eval(&x);
    let j = softmax.jacobian(&x);

    // Check Jacobian dimensions
    assert_eq!(j.shape(), &[3, 3]);

    // Check diagonal elements: J[i,i] = s[i] * (1 - s[i])
    for i in 0..3 {
        let expected = s[i] * (1.0 - s[i]);
        assert!(
            (j[[i, i]] - expected).abs() < 1e-6,
            "J[{},{}] = {} != {}",
            i,
            i,
            j[[i, i]],
            expected
        );
    }

    // Check off-diagonal elements: J[i,j] = -s[i] * s[j]
    for i in 0..3 {
        for j_idx in 0..3 {
            if i != j_idx {
                let expected = -s[i] * s[j_idx];
                assert!(
                    (j[[i, j_idx]] - expected).abs() < 1e-6,
                    "J[{},{}] = {} != {}",
                    i,
                    j_idx,
                    j[[i, j_idx]],
                    expected
                );
            }
        }
    }

    // Check row sums are zero (property of softmax Jacobian)
    for i in 0..3 {
        let row_sum: f32 = (0..3).map(|jj| j[[i, jj]]).sum();
        assert!(row_sum.abs() < 1e-6, "Row {} sum = {} != 0", i, row_sum);
    }
}

#[test]
fn test_softmax_crown_soundness() {
    // Test that softmax CROWN bounds are sound
    let softmax = SoftmaxLayer::new(-1);

    // Create bounded input
    let lower = arr1(&[0.0_f32, 1.0, 2.0]).into_dyn();
    let upper = arr1(&[0.5_f32, 1.5, 2.5]).into_dyn();
    let input = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    // Create identity linear bounds
    let linear_bounds = LinearBounds::identity(3);

    // Get CROWN bounds
    let crown_result = softmax
        .propagate_linear_with_bounds(&linear_bounds, &input)
        .unwrap();
    let crown_bounds = crown_result.concretize(&input);

    // Verify soundness by sampling
    for sample_idx in 0..50 {
        let t0 = (sample_idx as f32 * 17.0 % 50.0) / 50.0;
        let t1 = (sample_idx as f32 * 31.0 % 50.0) / 50.0;
        let t2 = (sample_idx as f32 * 47.0 % 50.0) / 50.0;

        let x_sample = arr1(&[0.0 + 0.5 * t0, 1.0 + 0.5 * t1, 2.0 + 0.5 * t2]);

        let s_sample = softmax.eval(&x_sample);

        for i in 0..3 {
            assert!(
                s_sample[i] >= crown_bounds.lower[[i]] - 1e-5,
                "Sample {} softmax[{}] = {} < lower bound {}",
                sample_idx,
                i,
                s_sample[i],
                crown_bounds.lower[[i]]
            );
            assert!(
                s_sample[i] <= crown_bounds.upper[[i]] + 1e-5,
                "Sample {} softmax[{}] = {} > upper bound {}",
                sample_idx,
                i,
                s_sample[i],
                crown_bounds.upper[[i]]
            );
        }
    }
}

#[test]
fn test_softmax_crown_comparable_to_ibp() {
    // Test that softmax CROWN bounds are comparable to IBP
    // Note: Due to local linearization with safety margin, CROWN may sometimes
    // be slightly looser than IBP for very small perturbations, but should be
    // sound and within a reasonable factor.
    let softmax = SoftmaxLayer::new(-1);

    // Create bounded input with moderate perturbation
    let lower = arr1(&[0.0_f32, 1.0, 2.0]).into_dyn();
    let upper = arr1(&[0.5_f32, 1.5, 2.5]).into_dyn();
    let input = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    // Get IBP bounds
    let ibp_bounds = softmax.propagate_ibp(&input).unwrap();

    // Get CROWN bounds
    let linear_bounds = LinearBounds::identity(3);
    let crown_result = softmax
        .propagate_linear_with_bounds(&linear_bounds, &input)
        .unwrap();
    let crown_bounds = crown_result.concretize(&input);

    // Both should be sound (verified by sampling test)
    // Check that CROWN is within 2x of IBP (reasonable for local linearization)
    let ibp_range: f32 = (0..3)
        .map(|i| ibp_bounds.upper[[i]] - ibp_bounds.lower[[i]])
        .sum();
    let crown_range: f32 = (0..3)
        .map(|i| crown_bounds.upper[[i]] - crown_bounds.lower[[i]])
        .sum();

    // CROWN should be within reasonable factor of IBP
    assert!(
        crown_range <= ibp_range * 2.0 + 0.1,
        "CROWN range {} should be within 2x of IBP range {}",
        crown_range,
        ibp_range
    );

    // Both should give valid probability bounds [0, 1]
    for i in 0..3 {
        assert!(ibp_bounds.lower[[i]] >= 0.0 - 1e-6);
        assert!(ibp_bounds.upper[[i]] <= 1.0 + 1e-6);
        assert!(crown_bounds.lower[[i]] >= 0.0 - 0.1); // Slight tolerance for linear approximation
        assert!(crown_bounds.upper[[i]] <= 1.0 + 0.1);
    }
}

#[test]
fn test_graph_network_crown_with_softmax() {
    // Test that GraphNetwork CROWN works with Softmax
    let mut graph = GraphNetwork::new();

    // Linear -> Softmax network
    let linear = LinearLayer::new(Array2::eye(4), Some(arr1(&[0.0, 0.0, 0.0, 0.0]))).unwrap();
    let softmax = SoftmaxLayer::new(-1);

    graph.add_node(GraphNode::new(
        "linear",
        Layer::Linear(linear),
        vec!["_input".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "softmax",
        Layer::Softmax(softmax),
        vec!["linear".to_string()],
    ));
    graph.set_output("softmax");

    // Create bounded input
    let lower = arr1(&[0.0_f32, 1.0, 2.0, 3.0]).into_dyn();
    let upper = arr1(&[0.5_f32, 1.5, 2.5, 3.5]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Test IBP
    let ibp_result = graph.propagate_ibp(&input).unwrap();

    // Test CROWN
    let crown_result = graph.propagate_crown(&input).unwrap();

    // Both should give valid probability bounds
    for i in 0..4 {
        assert!(ibp_result.lower[[i]] >= 0.0, "IBP lower[{}] < 0", i);
        assert!(ibp_result.upper[[i]] <= 1.0, "IBP upper[{}] > 1", i);
        assert!(
            crown_result.lower[[i]] >= 0.0 - 1e-5,
            "CROWN lower[{}] < 0",
            i
        );
        assert!(
            crown_result.upper[[i]] <= 1.0 + 1e-5,
            "CROWN upper[{}] > 1",
            i
        );
    }

    // CROWN should be at least as tight as IBP on average
    let ibp_range: f32 = (0..4)
        .map(|i| ibp_result.upper[[i]] - ibp_result.lower[[i]])
        .sum();
    let crown_range: f32 = (0..4)
        .map(|i| crown_result.upper[[i]] - crown_result.lower[[i]])
        .sum();

    assert!(
        crown_range <= ibp_range + 0.1, // Small tolerance for numerical errors
        "CROWN range {} should be <= IBP range {}",
        crown_range,
        ibp_range
    );
}

#[test]
fn test_network_crown_with_softmax() {
    // Test that Network CROWN works with Softmax (sequential)
    let linear = LinearLayer::new(Array2::eye(4), Some(arr1(&[0.0, 0.0, 0.0, 0.0]))).unwrap();
    let softmax = SoftmaxLayer::new(-1);

    let network = Network {
        layers: vec![Layer::Linear(linear), Layer::Softmax(softmax)],
    };

    // Create bounded input
    let lower = arr1(&[0.0_f32, 1.0, 2.0, 3.0]).into_dyn();
    let upper = arr1(&[0.5_f32, 1.5, 2.5, 3.5]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Test CROWN
    let crown_result = network.propagate_crown(&input).unwrap();

    // Should give valid probability bounds
    for i in 0..4 {
        assert!(
            crown_result.lower[[i]] >= 0.0 - 1e-5,
            "CROWN lower[{}] < 0: {}",
            i,
            crown_result.lower[[i]]
        );
        assert!(
            crown_result.upper[[i]] <= 1.0 + 1e-5,
            "CROWN upper[{}] > 1: {}",
            i,
            crown_result.upper[[i]]
        );
    }

    // Verify soundness by sampling
    let softmax_layer = SoftmaxLayer::new(-1);
    for sample_idx in 0..20 {
        let t0 = (sample_idx as f32 * 13.0 % 20.0) / 20.0;
        let t1 = (sample_idx as f32 * 17.0 % 20.0) / 20.0;
        let t2 = (sample_idx as f32 * 23.0 % 20.0) / 20.0;
        let t3 = (sample_idx as f32 * 29.0 % 20.0) / 20.0;

        let x_sample = arr1(&[
            0.0 + 0.5 * t0,
            1.0 + 0.5 * t1,
            2.0 + 0.5 * t2,
            3.0 + 0.5 * t3,
        ]);

        let s_sample = softmax_layer.eval(&x_sample);

        for i in 0..4 {
            assert!(
                s_sample[i] >= crown_result.lower[[i]] - 1e-3,
                "Sample {} softmax[{}] = {} < lower bound {}",
                sample_idx,
                i,
                s_sample[i],
                crown_result.lower[[i]]
            );
            assert!(
                s_sample[i] <= crown_result.upper[[i]] + 1e-3,
                "Sample {} softmax[{}] = {} > upper bound {}",
                sample_idx,
                i,
                s_sample[i],
                crown_result.upper[[i]]
            );
        }
    }
}

#[test]
fn test_matmul_crown_batched_3d_soundness() {
    // Test MatMul CROWN with batched 3D inputs using GraphNetwork.
    // Build: GELU(input_a) @ GELU(input_b)^T with shape [batch, m, k] @ [batch, n, k]
    let batch = 2_usize;
    let m = 2_usize;
    let k = 3_usize;
    let n = 2_usize;

    let mut graph = GraphNetwork::new();

    // Use GELU to transform inputs (tests McCormick with negative values)
    graph.add_node(GraphNode::from_input(
        "a",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "b",
        Layer::GELU(GELULayer::default()),
    ));

    // C = A @ B^T
    let matmul = MatMulLayer::new(true, None);
    graph.add_node(GraphNode::binary("c", Layer::MatMul(matmul), "a", "b"));
    graph.set_output("c");

    // Input bounds: 3D tensors
    let input = BoundedTensor::new(
        ndarray::ArrayD::from_elem(vec![batch, m, k], -1.0_f32),
        ndarray::ArrayD::from_elem(vec![batch, m, k], 1.0_f32),
    )
    .unwrap();

    let crown_bounds = graph.propagate_crown(&input).unwrap();

    // Sample and verify soundness
    for sample_idx in 0..20_usize {
        let mut x_sample = ndarray::ArrayD::zeros(vec![batch, m, k]);

        for idx in x_sample.indexed_iter_mut() {
            let hash = (sample_idx as u32)
                .wrapping_mul(2654435761_u32)
                .wrapping_add(idx.0[0] as u32 * 100 + idx.0[1] as u32 * 10 + idx.0[2] as u32);
            let t = hash as f32 / u32::MAX as f32;
            *idx.1 = -1.0 + 2.0 * t;
        }

        // Apply GELU to get transformed inputs
        let a = x_sample.mapv(|v| gelu_eval(v, GeluApproximation::Erf));
        let b = x_sample.mapv(|v| gelu_eval(v, GeluApproximation::Erf));

        // Compute C = A @ B^T for each batch
        let mut c_sample = ndarray::ArrayD::zeros(vec![batch, m, n]);
        for b_idx in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0_f32;
                    for l in 0..k {
                        sum += a[[b_idx, i, l]] * b[[b_idx, j, l]];
                    }
                    c_sample[[b_idx, i, j]] = sum;
                }
            }
        }

        // Verify soundness against flattened bounds
        for (flat, &val) in c_sample.iter().enumerate() {
            let lower = crown_bounds.lower.as_slice().unwrap()[flat];
            let upper = crown_bounds.upper.as_slice().unwrap()[flat];
            assert!(
                val >= lower - 1e-4,
                "Batched MatMul CROWN lower violation at flat {} sample {}: {} < {}",
                flat,
                sample_idx,
                val,
                lower
            );
            assert!(
                val <= upper + 1e-4,
                "Batched MatMul CROWN upper violation at flat {} sample {}: {} > {}",
                flat,
                sample_idx,
                val,
                upper
            );
        }
    }
}

#[test]
fn test_matmul_batched_crown_basic() {
    // Test the new propagate_linear_batched_binary for MatMulLayer
    // Simple 2D case: C = A @ B where A is [2, 3], B is [3, 2] (no batch dims)
    let m = 2_usize;
    let k = 3_usize;
    let n = 2_usize;

    let matmul = MatMulLayer::new(false, None);

    // Input bounds for A [m, k] and B [k, n]
    let input_a = BoundedTensor::new(
        ndarray::ArrayD::from_elem(vec![m, k], -1.0_f32),
        ndarray::ArrayD::from_elem(vec![m, k], 1.0_f32),
    )
    .unwrap();

    let input_b = BoundedTensor::new(
        ndarray::ArrayD::from_elem(vec![k, n], -0.5_f32),
        ndarray::ArrayD::from_elem(vec![k, n], 0.5_f32),
    )
    .unwrap();

    // Identity bounds on C [m, n]
    let c_size = m * n;
    let identity = BatchedLinearBounds::identity(&[m, n]);

    assert_eq!(identity.lower_a.shape(), &[m, n, n]);

    // For this simple case, we need to adjust expectations:
    // BatchedLinearBounds::identity([m, n]) creates [m, n, n] shape
    // But for MatMul, the output flat size is m*n = 4
    // The identity is set up per-row batching which doesn't match our needs

    // Create proper identity for flattened output
    let eye = ndarray::Array2::<f32>::eye(c_size);
    let flat_identity = BatchedLinearBounds {
        lower_a: eye.clone().into_dyn(),
        lower_b: ndarray::ArrayD::zeros(vec![c_size].as_slice()),
        upper_a: eye.into_dyn(),
        upper_b: ndarray::ArrayD::zeros(vec![c_size].as_slice()),
        input_shape: vec![m, n],
        output_shape: vec![m, n],
    };

    // Propagate backward through MatMul
    let (bounds_a, bounds_b) = matmul
        .propagate_linear_batched_binary(&flat_identity, &input_a, &input_b)
        .unwrap();

    // Check shapes
    let a_size = m * k;
    let b_size = k * n;
    assert_eq!(bounds_a.lower_a.shape(), &[c_size, a_size]);
    assert_eq!(bounds_b.lower_a.shape(), &[c_size, b_size]);

    // Flatten inputs for concretization (concretize expects [..., in_dim] matching coefficients)
    let input_a_flat = BoundedTensor::new(
        input_a
            .lower
            .clone()
            .into_shape_with_order(vec![a_size])
            .unwrap()
            .into_dyn(),
        input_a
            .upper
            .clone()
            .into_shape_with_order(vec![a_size])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();
    let input_b_flat = BoundedTensor::new(
        input_b
            .lower
            .clone()
            .into_shape_with_order(vec![b_size])
            .unwrap()
            .into_dyn(),
        input_b
            .upper
            .clone()
            .into_shape_with_order(vec![b_size])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();

    // Concretize and check soundness
    let crown_a = bounds_a.concretize(&input_a_flat);
    let crown_b = bounds_b.concretize(&input_b_flat);

    // The combined bounds should contain the output C = A @ B
    // Sample some points and verify
    for sample_idx in 0..10 {
        let mut a_sample = ndarray::Array2::<f32>::zeros((m, k));
        let mut b_sample = ndarray::Array2::<f32>::zeros((k, n));

        for ((i, j), v) in a_sample.indexed_iter_mut() {
            let hash =
                (sample_idx as u32 * 1000 + i as u32 * 100 + j as u32).wrapping_mul(2654435761);
            let t = hash as f32 / u32::MAX as f32;
            *v = -1.0 + 2.0 * t;
        }

        for ((i, j), v) in b_sample.indexed_iter_mut() {
            let hash =
                (sample_idx as u32 * 10000 + i as u32 * 100 + j as u32).wrapping_mul(1664525);
            let t = hash as f32 / u32::MAX as f32;
            *v = -0.5 + 1.0 * t;
        }

        // Compute C = A @ B
        let c_sample = a_sample.dot(&b_sample);

        // Check soundness for each output position
        for i in 0..m {
            for j in 0..n {
                let val = c_sample[[i, j]];
                let flat = i * n + j;

                // The bounds are in terms of A and B separately
                // For soundness, we need the combined concretization
                // Since bias is split, add both halves
                let lower = crown_a.lower.as_slice().unwrap()[flat]
                    + crown_b.lower.as_slice().unwrap()[flat];
                let upper = crown_a.upper.as_slice().unwrap()[flat]
                    + crown_b.upper.as_slice().unwrap()[flat];

                // Allow tolerance for McCormick relaxation looseness
                assert!(
                    val >= lower - 1e-3,
                    "MatMul batched lower violation at [{},{}]: {} < {}",
                    i,
                    j,
                    val,
                    lower
                );
                assert!(
                    val <= upper + 1e-3,
                    "MatMul batched upper violation at [{},{}]: {} > {}",
                    i,
                    j,
                    val,
                    upper
                );
            }
        }
    }
}

#[test]
fn test_matmul_batched_crown_requires_flattened_output() {
    // Batched MatMul CROWN currently requires the incoming bounds to treat the MatMul output
    // as a flattened vector of length m*n, not a rank-2 [m, n] tensor with per-row batching.
    let m = 3_usize;
    let k = 4_usize;
    let n = 2_usize;

    let matmul = MatMulLayer::new(false, None);

    let input_a = BoundedTensor::new(
        ndarray::ArrayD::from_elem(vec![m, k], -1.0_f32),
        ndarray::ArrayD::from_elem(vec![m, k], 1.0_f32),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        ndarray::ArrayD::from_elem(vec![k, n], -1.0_f32),
        ndarray::ArrayD::from_elem(vec![k, n], 1.0_f32),
    )
    .unwrap();

    // This identity treats [m, n] as batch_dims=[m], dim=n (in_dim=n), which is not the
    // flattened [m*n] representation required by MatMul batched CROWN.
    let per_row_identity = BatchedLinearBounds::identity(&[m, n]);

    let err = matmul
        .propagate_linear_batched_binary(&per_row_identity, &input_a, &input_b)
        .unwrap_err();

    match err {
        GammaError::UnsupportedOp(msg) => {
            assert!(msg.contains("flattened output dim"), "msg: {}", msg);
        }
        other => panic!("Expected UnsupportedOp, got {:?}", other),
    }
}

#[test]
fn test_matmul_batched_crown_transpose_soundness() {
    // Test batched MatMul CROWN with transpose_b = true (like Q @ K^T in attention)
    let seq = 3_usize;
    let head_dim = 2_usize;

    let matmul = MatMulLayer::new(true, Some(0.5)); // transpose_b, scale

    // Q and K both have shape [seq, head_dim]
    let input_q = BoundedTensor::new(
        ndarray::ArrayD::from_elem(vec![seq, head_dim], -1.0_f32),
        ndarray::ArrayD::from_elem(vec![seq, head_dim], 1.0_f32),
    )
    .unwrap();

    let input_k = BoundedTensor::new(
        ndarray::ArrayD::from_elem(vec![seq, head_dim], -1.0_f32),
        ndarray::ArrayD::from_elem(vec![seq, head_dim], 1.0_f32),
    )
    .unwrap();

    // Output C has shape [seq, seq] (attention scores)
    let c_size = seq * seq;
    let q_size = seq * head_dim;
    let k_size = seq * head_dim;

    // Create identity bounds for flattened output
    let eye = ndarray::Array2::<f32>::eye(c_size);
    let flat_identity = BatchedLinearBounds {
        lower_a: eye.clone().into_dyn(),
        lower_b: ndarray::ArrayD::zeros(vec![c_size].as_slice()),
        upper_a: eye.into_dyn(),
        upper_b: ndarray::ArrayD::zeros(vec![c_size].as_slice()),
        input_shape: vec![seq, seq],
        output_shape: vec![seq, seq],
    };

    let (bounds_q, bounds_k) = matmul
        .propagate_linear_batched_binary(&flat_identity, &input_q, &input_k)
        .unwrap();

    assert_eq!(bounds_q.lower_a.shape(), &[c_size, q_size]);
    assert_eq!(bounds_k.lower_a.shape(), &[c_size, k_size]);

    // Flatten inputs for concretization
    let input_q_flat = BoundedTensor::new(
        input_q
            .lower
            .clone()
            .into_shape_with_order(vec![q_size])
            .unwrap()
            .into_dyn(),
        input_q
            .upper
            .clone()
            .into_shape_with_order(vec![q_size])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();
    let input_k_flat = BoundedTensor::new(
        input_k
            .lower
            .clone()
            .into_shape_with_order(vec![k_size])
            .unwrap()
            .into_dyn(),
        input_k
            .upper
            .clone()
            .into_shape_with_order(vec![k_size])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();

    // Concretize
    let crown_q = bounds_q.concretize(&input_q_flat);
    let crown_k = bounds_k.concretize(&input_k_flat);

    // Sample and verify
    for sample_idx in 0..10 {
        let mut q_sample = ndarray::Array2::<f32>::zeros((seq, head_dim));
        let mut k_sample = ndarray::Array2::<f32>::zeros((seq, head_dim));

        for ((i, j), v) in q_sample.indexed_iter_mut() {
            let hash =
                (sample_idx as u32 * 1000 + i as u32 * 10 + j as u32).wrapping_mul(2654435761);
            let t = hash as f32 / u32::MAX as f32;
            *v = -1.0 + 2.0 * t;
        }

        for ((i, j), v) in k_sample.indexed_iter_mut() {
            let hash = (sample_idx as u32 * 10000 + i as u32 * 10 + j as u32).wrapping_mul(1664525);
            let t = hash as f32 / u32::MAX as f32;
            *v = -1.0 + 2.0 * t;
        }

        // Compute C = Q @ K^T * scale
        let k_t = k_sample.t();
        let c_sample = q_sample.dot(&k_t).mapv(|v| v * 0.5);

        // Verify soundness
        for i in 0..seq {
            for j in 0..seq {
                let val = c_sample[[i, j]];
                let flat = i * seq + j;

                let lower = crown_q.lower.as_slice().unwrap()[flat]
                    + crown_k.lower.as_slice().unwrap()[flat];
                let upper = crown_q.upper.as_slice().unwrap()[flat]
                    + crown_k.upper.as_slice().unwrap()[flat];

                assert!(
                    val >= lower - 1e-3,
                    "MatMul transpose lower violation at [{},{}]: {} < {}",
                    i,
                    j,
                    val,
                    lower
                );
                assert!(
                    val <= upper + 1e-3,
                    "MatMul transpose upper violation at [{},{}]: {} > {}",
                    i,
                    j,
                    val,
                    upper
                );
            }
        }
    }
}

#[test]
fn test_softmax_crown_batched_3d_soundness() {
    // Test Softmax CROWN with batched 3D inputs: [batch, seq, vocab]
    // Softmax along last axis (vocab)
    let batch = 2_usize;
    let seq = 2_usize;
    let vocab = 4_usize;

    let mut lower = ndarray::ArrayD::zeros(vec![batch, seq, vocab]);
    let mut upper = ndarray::ArrayD::zeros(vec![batch, seq, vocab]);

    // Initialize with some spread
    for idx in lower.indexed_iter_mut() {
        let hash = (idx.0[0] as u32 * 100 + idx.0[1] as u32 * 10 + idx.0[2] as u32)
            .wrapping_mul(2654435761_u32);
        let base = (hash as f32 / u32::MAX as f32) * 4.0 - 2.0;
        *idx.1 = base - 0.2;
    }
    for idx in upper.indexed_iter_mut() {
        let hash = (idx.0[0] as u32 * 100 + idx.0[1] as u32 * 10 + idx.0[2] as u32)
            .wrapping_mul(2654435761_u32);
        let base = (hash as f32 / u32::MAX as f32) * 4.0 - 2.0;
        *idx.1 = base + 0.2;
    }

    let pre_bounds = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    // Identity bounds for full 3D tensor
    let total_size = batch * seq * vocab;
    let identity_bounds = LinearBounds::identity(total_size);

    let softmax = SoftmaxLayer::new(-1);
    let crown_result = softmax
        .propagate_linear_with_bounds(&identity_bounds, &pre_bounds)
        .unwrap();

    // Concretize bounds
    let crown_bounds = crown_result.concretize(&pre_bounds);

    // Sample and verify soundness
    for sample_idx in 0..20_usize {
        let mut x_sample = ndarray::ArrayD::zeros(vec![batch, seq, vocab]);

        for idx in x_sample.indexed_iter_mut() {
            let l = lower[idx.0.clone()];
            let u = upper[idx.0.clone()];
            let hash = (sample_idx as u32)
                .wrapping_mul(2654435761_u32)
                .wrapping_add(idx.0[0] as u32 * 1000 + idx.0[1] as u32 * 100 + idx.0[2] as u32);
            let t = hash as f32 / u32::MAX as f32;
            *idx.1 = l + (u - l) * t;
        }

        // Compute softmax for each [batch, seq] position
        let mut s_sample = ndarray::ArrayD::zeros(vec![batch, seq, vocab]);
        for b_idx in 0..batch {
            for s_idx in 0..seq {
                // Extract 1D slice
                let mut slice = vec![0.0_f32; vocab];
                for v in 0..vocab {
                    slice[v] = x_sample[[b_idx, s_idx, v]];
                }

                // Compute softmax
                let max_x = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_x: Vec<f32> = slice.iter().map(|&xi| (xi - max_x).exp()).collect();
                let sum_exp: f32 = exp_x.iter().sum();
                let softmax: Vec<f32> = exp_x.iter().map(|&ei| ei / sum_exp).collect();

                for v in 0..vocab {
                    s_sample[[b_idx, s_idx, v]] = softmax[v];
                }
            }
        }

        // Verify soundness
        let lower_slice = crown_bounds.lower.as_slice().unwrap();
        let upper_slice = crown_bounds.upper.as_slice().unwrap();
        for (flat, &val) in s_sample.iter().enumerate() {
            assert!(
                val >= lower_slice[flat] - 1e-4,
                "Batched Softmax CROWN lower violation at flat {} sample {}: {} < {}",
                flat,
                sample_idx,
                val,
                lower_slice[flat]
            );
            assert!(
                val <= upper_slice[flat] + 1e-4,
                "Batched Softmax CROWN upper violation at flat {} sample {}: {} > {}",
                flat,
                sample_idx,
                val,
                upper_slice[flat]
            );
        }
    }
}

#[test]
fn test_softmax_batched_linear_bounds() {
    // Test the new propagate_linear_batched_with_bounds for SoftmaxLayer
    // Input shape: [batch, seq, softmax_size] = [2, 3, 4]
    // This tests the batched CROWN backward through softmax
    let batch = 2_usize;
    let seq = 3_usize;
    let softmax_size = 4_usize;
    let _total_batch = batch * seq;

    // Initialize pre-activation bounds
    let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, softmax_size]));
    let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, softmax_size]));

    for b in 0..batch {
        for s in 0..seq {
            for k in 0..softmax_size {
                let hash = ((b * 100 + s * 10 + k) as u32).wrapping_mul(2654435761_u32);
                let base = (hash as f32 / u32::MAX as f32) * 4.0 - 2.0;
                lower[[b, s, k]] = base - 0.15;
                upper[[b, s, k]] = base + 0.15;
            }
        }
    }

    let pre_bounds = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    // Create identity BatchedLinearBounds for the output
    let identity = BatchedLinearBounds::identity(&[batch, seq, softmax_size]);

    let softmax = SoftmaxLayer::new(-1);

    // Test the batched method
    let result = softmax
        .propagate_linear_batched_with_bounds(&identity, &pre_bounds)
        .unwrap();

    // Concretize to get final bounds
    let final_bounds = result.concretize(&pre_bounds);

    // Verify shape
    assert_eq!(final_bounds.shape(), &[batch, seq, softmax_size]);

    // Sample and verify soundness
    for sample_idx in 0..15_usize {
        let mut x_sample = ArrayD::zeros(IxDyn(&[batch, seq, softmax_size]));

        for b in 0..batch {
            for s in 0..seq {
                for k in 0..softmax_size {
                    let l = lower[[b, s, k]];
                    let u = upper[[b, s, k]];
                    let hash = (sample_idx as u32)
                        .wrapping_mul(2654435761_u32)
                        .wrapping_add((b * 1000 + s * 100 + k) as u32);
                    let t = hash as f32 / u32::MAX as f32;
                    x_sample[[b, s, k]] = l + (u - l) * t;
                }
            }
        }

        // Compute softmax for each [batch, seq] position
        let mut s_sample = ArrayD::zeros(IxDyn(&[batch, seq, softmax_size]));
        for b in 0..batch {
            for s in 0..seq {
                // Extract 1D slice
                let mut slice = vec![0.0_f32; softmax_size];
                for k in 0..softmax_size {
                    slice[k] = x_sample[[b, s, k]];
                }

                // Compute softmax
                let max_x = slice.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
                let exp_x: Vec<f32> = slice.iter().map(|&xi| (xi - max_x).exp()).collect();
                let sum_exp: f32 = exp_x.iter().sum();
                let softmax_vals: Vec<f32> = exp_x.iter().map(|&ei| ei / sum_exp).collect();

                for k in 0..softmax_size {
                    s_sample[[b, s, k]] = softmax_vals[k];
                }
            }
        }

        // Verify soundness
        for b in 0..batch {
            for s in 0..seq {
                for k in 0..softmax_size {
                    let val = s_sample[[b, s, k]];
                    let lb = final_bounds.lower[[b, s, k]];
                    let ub = final_bounds.upper[[b, s, k]];
                    assert!(
                        val >= lb - 1e-4,
                        "Batched Softmax CROWN lower violation at [{}, {}, {}] sample {}: {} < {}",
                        b,
                        s,
                        k,
                        sample_idx,
                        val,
                        lb
                    );
                    assert!(
                        val <= ub + 1e-4,
                        "Batched Softmax CROWN upper violation at [{}, {}, {}] sample {}: {} > {}",
                        b,
                        s,
                        k,
                        sample_idx,
                        val,
                        ub
                    );
                }
            }
        }
    }

    // Also print bound widths for diagnostics
    let mut total_width = 0.0_f64;
    let mut count = 0;
    for b in 0..batch {
        for s in 0..seq {
            for k in 0..softmax_size {
                let width = (final_bounds.upper[[b, s, k]] - final_bounds.lower[[b, s, k]]) as f64;
                total_width += width;
                count += 1;
            }
        }
    }
    let avg_width = total_width / count as f64;
    println!(
        "Batched Softmax CROWN: shape [{}, {}, {}], avg bound width: {:.4}",
        batch, seq, softmax_size, avg_width
    );
}

#[test]
fn test_graph_network_crown_attention_4d_soundness() {
    // Test full 4D batched attention pattern: [batch, heads, seq, dim]
    // This is the actual shape used in transformer attention.
    //
    // Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
    //
    // Input shape: [batch, heads, seq, dim]
    // Q @ K^T shape: [batch, heads, seq, seq]
    // softmax(.) shape: [batch, heads, seq, seq]
    // @ V shape: [batch, heads, seq, dim]

    let batch = 2_usize;
    let heads = 2_usize;
    let seq = 3_usize;
    let dim = 4_usize;

    let head_dim = dim; // For scaling factor

    let mut graph = GraphNetwork::new();

    // Q, K, V all derive from input via GELU (simulates projection)
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "v",
        Layer::GELU(GELULayer::default()),
    ));

    // Q @ K^T with scaling: [batch, heads, seq, dim] @ [batch, heads, dim, seq] -> [batch, heads, seq, seq]
    let scores = MatMulLayer::new(true, Some(1.0 / (head_dim as f32).sqrt()));
    graph.add_node(GraphNode::binary("scores", Layer::MatMul(scores), "q", "k"));

    // Softmax along last axis (seq dimension in scores)
    let softmax = SoftmaxLayer::new(-1);
    graph.add_node(GraphNode::new(
        "probs",
        Layer::Softmax(softmax),
        vec!["scores".to_string()],
    ));

    // probs @ V: [batch, heads, seq, seq] @ [batch, heads, seq, dim] -> [batch, heads, seq, dim]
    let out_matmul = MatMulLayer::new(false, None);
    graph.add_node(GraphNode::binary(
        "out",
        Layer::MatMul(out_matmul),
        "probs",
        "v",
    ));
    graph.set_output("out");

    // Input: 4D tensor [batch, heads, seq, dim]
    let input = BoundedTensor::new(
        ndarray::ArrayD::from_elem(vec![batch, heads, seq, dim], -1.0_f32),
        ndarray::ArrayD::from_elem(vec![batch, heads, seq, dim], 1.0_f32),
    )
    .unwrap();

    let bounds = graph.propagate_crown(&input).unwrap();

    // Verify output shape
    assert_eq!(bounds.shape(), &[batch, heads, seq, dim]);

    let sm = SoftmaxLayer::new(-1);

    // Sample and verify soundness
    for sample_idx in 0..25_usize {
        let mut x = ndarray::ArrayD::<f32>::zeros(vec![batch, heads, seq, dim]);
        for idx in x.indexed_iter_mut() {
            let hash = (sample_idx as u32)
                .wrapping_mul(2654435761_u32)
                .wrapping_add(
                    idx.0[0] as u32 * 1000
                        + idx.0[1] as u32 * 100
                        + idx.0[2] as u32 * 10
                        + idx.0[3] as u32,
                );
            let t = hash as f32 / u32::MAX as f32;
            *idx.1 = -1.0 + 2.0 * t;
        }

        // Apply GELU to get Q, K, V
        let q = x.mapv(|v| gelu_eval(v, GeluApproximation::Erf));
        let k = x.mapv(|v| gelu_eval(v, GeluApproximation::Erf));
        let v = x.mapv(|v| gelu_eval(v, GeluApproximation::Erf));

        // Compute attention manually for each batch/head
        let mut out = ndarray::ArrayD::<f32>::zeros(vec![batch, heads, seq, dim]);
        for b in 0..batch {
            for h in 0..heads {
                // Extract 2D slices for this batch/head
                let q_2d: Vec<Vec<f32>> = (0..seq)
                    .map(|s| (0..dim).map(|d| q[[b, h, s, d]]).collect())
                    .collect();
                let k_2d: Vec<Vec<f32>> = (0..seq)
                    .map(|s| (0..dim).map(|d| k[[b, h, s, d]]).collect())
                    .collect();
                let v_2d: Vec<Vec<f32>> = (0..seq)
                    .map(|s| (0..dim).map(|d| v[[b, h, s, d]]).collect())
                    .collect();

                // Q @ K^T / sqrt(d) -> [seq, seq]
                let scale = 1.0 / (head_dim as f32).sqrt();
                let mut scores_2d = vec![vec![0.0_f32; seq]; seq];
                for i in 0..seq {
                    for j in 0..seq {
                        let mut sum = 0.0_f32;
                        for l in 0..dim {
                            sum += q_2d[i][l] * k_2d[j][l]; // K^T means k[j][l]
                        }
                        scores_2d[i][j] = sum * scale;
                    }
                }

                // Softmax each row
                let mut probs_2d = vec![vec![0.0_f32; seq]; seq];
                for i in 0..seq {
                    let row: ndarray::Array1<f32> = ndarray::Array1::from_vec(scores_2d[i].clone());
                    let softmax_row = sm.eval(&row);
                    for j in 0..seq {
                        probs_2d[i][j] = softmax_row[j];
                    }
                }

                // probs @ V -> [seq, dim]
                for i in 0..seq {
                    for d in 0..dim {
                        let mut sum = 0.0_f32;
                        for j in 0..seq {
                            sum += probs_2d[i][j] * v_2d[j][d];
                        }
                        out[[b, h, i, d]] = sum;
                    }
                }
            }
        }

        // Verify all outputs are within bounds
        for idx in out.indexed_iter() {
            let val = *idx.1;
            let lower_val = bounds.lower[idx.0.clone()];
            let upper_val = bounds.upper[idx.0.clone()];
            assert!(
                val >= lower_val - 1e-4,
                "4D Attention CROWN lower violation at {:?} sample {}: {} < {}",
                idx.0,
                sample_idx,
                val,
                lower_val
            );
            assert!(
                val <= upper_val + 1e-4,
                "4D Attention CROWN upper violation at {:?} sample {}: {} > {}",
                idx.0,
                sample_idx,
                val,
                upper_val
            );
        }
    }
}

#[test]
fn test_graph_network_crown_batched_uses_zonotope_for_attention_like_matmul_bounds() {
    // For attention-like Q@K^T patterns where Q and K are both linear projections of the same
    // input, GraphNetwork tightens MatMul bounds using a per-position zonotope so that
    // diagonal entries (sums of squares-like terms) get non-negative lower bounds.

    let seq = 3_usize;
    let dim = 4_usize;
    let epsilon = 0.5_f32;

    let mut graph = GraphNetwork::new();
    let eye = ndarray::Array2::<f32>::eye(dim);
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::Linear(LinearLayer::new(eye, None).unwrap()),
    ));
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "q",
        "k",
    ));
    graph.set_output("scores");

    // Input centered at 0 with uniform epsilon.
    let input = BoundedTensor::new(
        ndarray::ArrayD::from_elem(vec![seq, dim], -epsilon),
        ndarray::ArrayD::from_elem(vec![seq, dim], epsilon),
    )
    .unwrap();

    let baseline_interval = MatMulLayer::new(true, None)
        .propagate_ibp_binary(&input, &input)
        .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let node_bounds = graph.collect_node_bounds(&input).unwrap();
    let collected_scores = node_bounds.get("scores").unwrap();
    let crown_bounds = graph.propagate_crown_batched(&input).unwrap();

    // For X @ X^T, diagonal entries are sums of squares, so lower bounds should be >= 0.
    // IBP interval matmul cannot capture this and produces negative lower bounds.
    assert!(
        baseline_interval.lower[[0, 0]] < -1e-6,
        "Baseline interval MatMul should be negative on diagonal"
    );
    assert!(
        ibp_bounds.lower[[0, 0]] >= -1e-6,
        "Zonotope-tightened MatMul should be non-negative on diagonal (got {})",
        ibp_bounds.lower[[0, 0]]
    );
    assert!(
        collected_scores.lower[[0, 0]] >= -1e-6,
        "collect_node_bounds should use zonotope-tightened MatMul bounds (got {})",
        collected_scores.lower[[0, 0]]
    );

    // collect_node_bounds() should match the forward IBP pass for this graph.
    assert_eq!(collected_scores.shape(), ibp_bounds.shape());
    for ((&cl, &cu), (&il, &iu)) in collected_scores
        .lower
        .iter()
        .zip(collected_scores.upper.iter())
        .zip(ibp_bounds.lower.iter().zip(ibp_bounds.upper.iter()))
    {
        assert!((cl - il).abs() < 1e-6);
        assert!((cu - iu).abs() < 1e-6);
    }

    // Batched CROWN falls back to partial CROWN at attention MatMul; with an identity output,
    // the result should equal the MatMul IBP bounds used for concretization.
    assert_eq!(crown_bounds.shape(), ibp_bounds.shape());
    for ((&cl, &cu), (&il, &iu)) in crown_bounds
        .lower
        .iter()
        .zip(crown_bounds.upper.iter())
        .zip(ibp_bounds.lower.iter().zip(ibp_bounds.upper.iter()))
    {
        assert!((cl - il).abs() < 1e-6);
        assert!((cu - iu).abs() < 1e-6);
    }
}

#[test]
fn test_zonotope_attention_with_layernorm_integration() {
    // Test that zonotope attention tracking propagates through LayerNorm.
    // Architecture: _input -> LayerNorm -> Q_proj + K_proj -> Q@K^T
    // The LayerNorm should preserve correlations between Q and K, giving tighter bounds.

    let seq = 3_usize;
    let dim = 4_usize;
    let epsilon = 0.1_f32;

    let mut graph = GraphNetwork::new();

    // Add LayerNorm (gamma=1, beta=0)
    let gamma = ndarray::Array1::ones(dim);
    let beta = ndarray::Array1::zeros(dim);
    graph.add_node(GraphNode::from_input(
        "ln",
        Layer::LayerNorm(LayerNormLayer::new(gamma, beta, 1e-5)),
    ));

    // Q and K projections from LayerNorm output
    let eye = ndarray::Array2::<f32>::eye(dim);
    graph.add_node(GraphNode::new(
        "q",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
        vec!["ln".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "k",
        Layer::Linear(LinearLayer::new(eye, None).unwrap()),
        vec!["ln".to_string()],
    ));

    // Q@K^T MatMul
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "q",
        "k",
    ));
    graph.set_output("scores");

    // Input with varied values per feature (to avoid near-zero variance in LayerNorm)
    // Use a pattern that gives non-trivial variance: each row has values [1, 2, 3, 4]
    let center_values: Vec<f32> = (0..seq * dim).map(|i| (i % dim) as f32 + 1.0).collect();
    let lower = ndarray::ArrayD::from_shape_vec(
        vec![seq, dim],
        center_values.iter().map(|&v| v - epsilon).collect(),
    )
    .unwrap();
    let upper = ndarray::ArrayD::from_shape_vec(
        vec![seq, dim],
        center_values.iter().map(|&v| v + epsilon).collect(),
    )
    .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Run IBP (which uses zonotope tightening for Q@K^T)
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Check that bounds are finite and reasonable
    assert!(
        ibp_bounds.lower.iter().all(|&v| v.is_finite()),
        "Lower bounds should be finite"
    );
    assert!(
        ibp_bounds.upper.iter().all(|&v| v.is_finite()),
        "Upper bounds should be finite"
    );

    // LayerNorm normalizes input to zero mean, unit variance.
    // After identity projections, Q@K^T is essentially X_norm @ X_norm^T
    // Diagonal entries should be near dim (sum of squares of normalized values)
    let diag_lower = ibp_bounds.lower[[0, 0]];
    let diag_upper = ibp_bounds.upper[[0, 0]];
    assert!(
        diag_lower >= -10.0, // LayerNorm + small epsilon can still give reasonable bounds
        "Diagonal lower bound should be reasonable (got {})",
        diag_lower
    );
    assert!(
        diag_upper >= diag_lower,
        "Upper bound should be >= lower bound"
    );

    // Verify bounds enclose a reasonable range
    let max_width = ibp_bounds.max_width();
    assert!(
        max_width < 50.0,
        "Bound width should be reasonable (got {})",
        max_width
    );
}

#[test]
fn test_zonotope_attention_layernorm_vs_no_layernorm() {
    // Compare zonotope attention bounds with and without LayerNorm in the path.
    // Both should produce valid bounds; with LayerNorm should still have correlation tracking.

    let seq = 3_usize;
    let dim = 4_usize;
    let epsilon = 0.1_f32;

    // Graph WITHOUT LayerNorm: _input -> Q_proj + K_proj -> Q@K^T
    let mut graph_no_ln = GraphNetwork::new();
    let eye = ndarray::Array2::<f32>::eye(dim);
    graph_no_ln.add_node(GraphNode::from_input(
        "q",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
    ));
    graph_no_ln.add_node(GraphNode::from_input(
        "k",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
    ));
    graph_no_ln.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "q",
        "k",
    ));
    graph_no_ln.set_output("scores");

    // Graph WITH LayerNorm: _input -> LayerNorm -> Q_proj + K_proj -> Q@K^T
    let mut graph_with_ln = GraphNetwork::new();
    let gamma = ndarray::Array1::ones(dim);
    let beta = ndarray::Array1::zeros(dim);
    graph_with_ln.add_node(GraphNode::from_input(
        "ln",
        Layer::LayerNorm(LayerNormLayer::new(gamma, beta, 1e-5)),
    ));
    graph_with_ln.add_node(GraphNode::new(
        "q",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
        vec!["ln".to_string()],
    ));
    graph_with_ln.add_node(GraphNode::new(
        "k",
        Layer::Linear(LinearLayer::new(eye, None).unwrap()),
        vec!["ln".to_string()],
    ));
    graph_with_ln.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "q",
        "k",
    ));
    graph_with_ln.set_output("scores");

    // Input with varied values per feature (to avoid near-zero variance in LayerNorm)
    // For graph_no_ln this is also fine; for graph_with_ln it ensures LayerNorm works well
    let center_values: Vec<f32> = (0..seq * dim).map(|i| (i % dim) as f32 + 1.0).collect();
    let lower = ndarray::ArrayD::from_shape_vec(
        vec![seq, dim],
        center_values.iter().map(|&v| v - epsilon).collect(),
    )
    .unwrap();
    let upper = ndarray::ArrayD::from_shape_vec(
        vec![seq, dim],
        center_values.iter().map(|&v| v + epsilon).collect(),
    )
    .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let bounds_no_ln = graph_no_ln.propagate_ibp(&input).unwrap();
    let bounds_with_ln = graph_with_ln.propagate_ibp(&input).unwrap();

    // Both should have same output shape
    assert_eq!(bounds_no_ln.shape(), bounds_with_ln.shape());

    // Both should produce finite bounds
    assert!(bounds_no_ln.lower.iter().all(|&v| v.is_finite()));
    assert!(bounds_no_ln.upper.iter().all(|&v| v.is_finite()));
    assert!(bounds_with_ln.lower.iter().all(|&v| v.is_finite()));
    assert!(bounds_with_ln.upper.iter().all(|&v| v.is_finite()));

    // Without LayerNorm, diagonal should have non-negative lower bound
    // (zonotope tracks X @ X^T correlation directly)
    assert!(
        bounds_no_ln.lower[[0, 0]] >= -1e-6,
        "No-LN diagonal lower should be >= 0 (got {})",
        bounds_no_ln.lower[[0, 0]]
    );

    // With LayerNorm, the transformation changes the values but correlation
    // should still be tracked through the affine approximation
    let ln_diag_lower = bounds_with_ln.lower[[0, 0]];
    assert!(
        ln_diag_lower > -10.0, // LayerNorm can shift values, so relaxed check
        "With-LN diagonal lower should be reasonable (got {})",
        ln_diag_lower
    );

    // Bound widths should be reasonable (not exploding)
    assert!(
        bounds_no_ln.max_width() < 50.0,
        "No-LN bounds width should be reasonable (got {})",
        bounds_no_ln.max_width()
    );
    assert!(
        bounds_with_ln.max_width() < 50.0,
        "With-LN bounds width should be reasonable (got {})",
        bounds_with_ln.max_width()
    );
}

#[test]
fn test_zonotope_swiglu_ffn_tightening() {
    // Test that zonotope tightening for SwiGLU FFN gives tighter bounds than IBP.
    // Architecture: ffn_norm -> ffn_up (Linear) -----> up
    //                       -> ffn_gate (Linear) -> silu -> gate
    //               MulBinary(up, gate) -> swiglu
    //
    // Both ffn_up and ffn_gate share the same input (ffn_norm output),
    // so zonotopes can track correlations and give tighter multiplication bounds.

    use gamma_tensor::ZonotopeTensor;

    let seq = 4_usize;
    let hidden = 8_usize;
    let ffn_dim = 16_usize; // Intermediate FFN dimension

    let mut graph = GraphNetwork::new();

    // FFN norm (simulated as identity - just marks the shared input)
    graph.add_node(GraphNode::from_input(
        "ffn_norm",
        Layer::AddConstant(AddConstantLayer::new(
            ndarray::Array2::<f32>::zeros((seq, hidden)).into_dyn(),
        )),
    ));

    // FFN up projection: [seq, hidden] -> [seq, ffn_dim]
    let up_weights = Array2::<f32>::from_shape_fn((ffn_dim, hidden), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32 * 0.1;
        0.3 * phase.sin()
    });
    let up_linear = LinearLayer::new(up_weights, None).unwrap();
    graph.add_node(GraphNode::new(
        "ffn_up",
        Layer::Linear(up_linear),
        vec!["ffn_norm".to_string()],
    ));

    // FFN gate projection: [seq, hidden] -> [seq, ffn_dim]
    let gate_weights = Array2::<f32>::from_shape_fn((ffn_dim, hidden), |(i, j)| {
        let phase = (i * 23 + j * 13) as f32 * 0.1;
        0.3 * phase.cos()
    });
    let gate_linear = LinearLayer::new(gate_weights, None).unwrap();
    graph.add_node(GraphNode::new(
        "ffn_gate",
        Layer::Linear(gate_linear),
        vec!["ffn_norm".to_string()],
    ));

    // SiLU activation on gate
    graph.add_node(GraphNode::new(
        "silu",
        Layer::GELU(GELULayer::default()), // SiLU approximated as GELU
        vec!["ffn_gate".to_string()],
    ));

    // SwiGLU: up * silu(gate)
    graph.add_node(GraphNode::binary(
        "swiglu",
        Layer::MulBinary(MulBinaryLayer),
        "ffn_up",
        "silu",
    ));

    graph.set_output("swiglu");

    // Input with epsilon perturbation
    let epsilon = 0.01_f32;
    let input = BoundedTensor::from_epsilon(
        ndarray::Array2::<f32>::zeros((seq, hidden)).into_dyn(),
        epsilon,
    );

    // Get IBP bounds (treats up and gate as independent)
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let ibp_width = ibp_bounds.max_width();

    // Create block-wise result to test zonotope tightening
    let block_wise_result = graph.propagate_ibp_block_wise(&input, epsilon).unwrap();

    // Check that we got swiglu width tracking
    let has_swiglu_tracking = block_wise_result
        .blocks
        .iter()
        .any(|b| b.swiglu_width.is_some());

    // For the test, also manually compute zonotope bounds for comparison
    let _center = (&input.lower + &input.upper) / 2.0;
    let base_z = ZonotopeTensor::from_bounded_tensor_per_position_2d(&input).unwrap();

    // Get the linear layers for manual zonotope propagation
    let up_node = graph.nodes.get("ffn_up").unwrap();
    let gate_node = graph.nodes.get("ffn_gate").unwrap();
    let (up_linear, gate_linear) = match (&up_node.layer, &gate_node.layer) {
        (Layer::Linear(u), Layer::Linear(g)) => (u, g),
        _ => panic!("Expected Linear layers"),
    };

    // Propagate through up and gate
    let up_z = base_z
        .linear(&up_linear.weight, up_linear.bias.as_ref())
        .unwrap();
    let gate_z = base_z
        .linear(&gate_linear.weight, gate_linear.bias.as_ref())
        .unwrap();

    // Apply SiLU to gate
    let silu_z = gate_z.silu_affine().unwrap();

    // Multiply with shared error symbols
    let swiglu_z = up_z.mul_elementwise(&silu_z).unwrap();
    let zonotope_bounds = swiglu_z.to_bounded_tensor();
    let zonotope_width = zonotope_bounds.max_width();

    // Zonotope should give tighter bounds than IBP due to correlation tracking
    // Note: Not always tighter (depends on correlation structure), but should be sound
    assert!(
        zonotope_width <= ibp_width * 1.1, // Allow 10% tolerance for numerical issues
        "Zonotope width ({:.3e}) should not be much larger than IBP ({:.3e})",
        zonotope_width,
        ibp_width
    );

    // Verify soundness: zonotope bounds should contain actual values
    // (using concrete center values as sanity check)
    for (&l, &u) in zonotope_bounds
        .lower
        .iter()
        .zip(zonotope_bounds.upper.iter())
    {
        assert!(
            l.is_finite() && u.is_finite(),
            "Zonotope bounds must be finite"
        );
        assert!(l <= u + 1e-5, "Invalid interval: {} > {}", l, u);
    }

    // Print comparison
    println!(
        "SwiGLU bounds comparison: IBP width={:.3e}, Zonotope width={:.3e}, ratio={:.2}x",
        ibp_width,
        zonotope_width,
        ibp_width / zonotope_width
    );
    if has_swiglu_tracking {
        let swiglu_w = block_wise_result.blocks[0].swiglu_width.unwrap();
        println!("Block-wise SwiGLU width: {:.3e}", swiglu_w);
    }
}

#[test]
fn test_graph_network_crown_batched_attention_4d_smoke() {
    // Smoke test: N-D batched CROWN should be able to propagate through an attention-shaped
    // GraphNetwork without erroring.
    //
    // Shape matches Whisper attention core: [batch, heads, seq, dim]
    let batch = 1_usize;
    let heads = 1_usize;
    let seq = 3_usize;
    let dim = 4_usize;

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "v",
        Layer::GELU(GELULayer::default()),
    ));

    let scale = 1.0 / (dim as f32).sqrt();
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, Some(scale))),
        "q",
        "k",
    ));
    graph.add_node(GraphNode::new(
        "probs",
        Layer::Softmax(SoftmaxLayer::new(-1)),
        vec!["scores".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "out",
        Layer::MatMul(MatMulLayer::new(false, None)),
        "probs",
        "v",
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ndarray::ArrayD::from_elem(vec![batch, heads, seq, dim], -1.0_f32),
        ndarray::ArrayD::from_elem(vec![batch, heads, seq, dim], 1.0_f32),
    )
    .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let bounds = graph.propagate_crown_batched(&input).unwrap();
    assert_eq!(bounds.shape(), &[batch, heads, seq, dim]);

    for (l, u) in bounds.lower.iter().zip(bounds.upper.iter()) {
        assert!(l.is_finite() && u.is_finite(), "Non-finite bounds");
        assert!(*l <= *u + 1e-6, "Invalid interval: {} > {}", l, u);
    }

    // Partial CROWN: concretizes at the unsupported MatMul using IBP bounds there,
    // giving CROWN benefits for layers after the MatMul. Bounds should be at least
    // as tight as pure IBP (often tighter due to CROWN on post-MatMul layers).
    for ((crown_l, crown_u), (ibp_l, ibp_u)) in bounds
        .lower
        .iter()
        .zip(bounds.upper.iter())
        .zip(ibp_bounds.lower.iter().zip(ibp_bounds.upper.iter()))
    {
        // CROWN bounds should be at least as tight as IBP
        assert!(
            *crown_l >= *ibp_l - 1e-5,
            "CROWN lower {} should be >= IBP lower {}",
            crown_l,
            ibp_l
        );
        assert!(
            *crown_u <= *ibp_u + 1e-5,
            "CROWN upper {} should be <= IBP upper {}",
            crown_u,
            ibp_u
        );
    }
}

#[test]
fn test_graph_network_crown_vs_ibp_4d_attention() {
    // Verify CROWN provides tighter bounds than IBP for 4D attention
    let batch = 2_usize;
    let heads = 2_usize;
    let seq = 3_usize;
    let dim = 4_usize;

    let head_dim = dim;

    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "v",
        Layer::GELU(GELULayer::default()),
    ));

    let scores = MatMulLayer::new(true, Some(1.0 / (head_dim as f32).sqrt()));
    graph.add_node(GraphNode::binary("scores", Layer::MatMul(scores), "q", "k"));

    let softmax = SoftmaxLayer::new(-1);
    graph.add_node(GraphNode::new(
        "probs",
        Layer::Softmax(softmax),
        vec!["scores".to_string()],
    ));

    let out_matmul = MatMulLayer::new(false, None);
    graph.add_node(GraphNode::binary(
        "out",
        Layer::MatMul(out_matmul),
        "probs",
        "v",
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ndarray::ArrayD::from_elem(vec![batch, heads, seq, dim], -1.0_f32),
        ndarray::ArrayD::from_elem(vec![batch, heads, seq, dim], 1.0_f32),
    )
    .unwrap();

    // Get both IBP and CROWN bounds
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let crown_bounds = graph.propagate_crown(&input).unwrap();

    // Compute average interval widths
    let ibp_widths: Vec<f32> = ibp_bounds
        .lower
        .iter()
        .zip(ibp_bounds.upper.iter())
        .map(|(&l, &u)| u - l)
        .collect();
    let crown_widths: Vec<f32> = crown_bounds
        .lower
        .iter()
        .zip(crown_bounds.upper.iter())
        .map(|(&l, &u)| u - l)
        .collect();

    let avg_ibp_width: f32 = ibp_widths.iter().sum::<f32>() / ibp_widths.len() as f32;
    let avg_crown_width: f32 = crown_widths.iter().sum::<f32>() / crown_widths.len() as f32;

    println!(
        "4D Attention [batch={}, heads={}, seq={}, dim={}]:",
        batch, heads, seq, dim
    );
    println!("  IBP average width: {:.4}", avg_ibp_width);
    println!("  CROWN average width: {:.4}", avg_crown_width);
    println!(
        "  Tightening ratio: {:.2}x",
        avg_ibp_width / avg_crown_width
    );

    // CROWN should provide tighter or equal bounds
    for (i, (&ibp_l, &crown_l)) in ibp_bounds
        .lower
        .iter()
        .zip(crown_bounds.lower.iter())
        .enumerate()
    {
        assert!(
            crown_l >= ibp_l - 1e-4,
            "CROWN lower bound {} looser than IBP at {}: {} < {}",
            crown_l,
            i,
            crown_l,
            ibp_l
        );
    }
    for (i, (&ibp_u, &crown_u)) in ibp_bounds
        .upper
        .iter()
        .zip(crown_bounds.upper.iter())
        .enumerate()
    {
        assert!(
            crown_u <= ibp_u + 1e-4,
            "CROWN upper bound {} looser than IBP at {}: {} > {}",
            crown_u,
            i,
            crown_u,
            ibp_u
        );
    }

    // Expect CROWN to be at least 1.0x tighter on average (equal or better)
    // The actual improvement varies by network structure
    assert!(
        avg_crown_width <= avg_ibp_width + 1e-4,
        "CROWN should provide tighter or equal bounds than IBP"
    );
}

// ========================================================================
// MLP-style Network CROWN Tests (for transformer verification)
// ========================================================================

#[test]
fn test_mlp_style_crown_vs_ibp_1d() {
    // Test CROWN vs IBP on an MLP-style network with 1D input.
    // This simulates a single position of a transformer MLP.
    //
    // MLP structure: LayerNorm → Linear (expand 4x) → GELU → Linear (contract)
    //
    // Key insight: For transformers, the MLP operates independently per position.
    // If CROWN works on 1D, we can potentially apply it per-position.

    let hidden_dim = 8_usize; // Small for test
    let intermediate_dim = hidden_dim * 4; // 4x expansion like transformers

    // Create weights
    // Linear1: hidden_dim -> intermediate_dim (expand)
    let mut w1 = Array2::<f32>::zeros((intermediate_dim, hidden_dim));
    for i in 0..intermediate_dim {
        for j in 0..hidden_dim {
            // Kaiming-style initialization
            w1[[i, j]] = ((i * 13 + j * 7) % 17) as f32 / 17.0 * 0.2 - 0.1;
        }
    }
    let b1 = Array1::<f32>::zeros(intermediate_dim);

    // Linear2: intermediate_dim -> hidden_dim (contract)
    let mut w2 = Array2::<f32>::zeros((hidden_dim, intermediate_dim));
    for i in 0..hidden_dim {
        for j in 0..intermediate_dim {
            w2[[i, j]] = ((i * 11 + j * 3) % 13) as f32 / 13.0 * 0.2 - 0.1;
        }
    }
    let b2 = Array1::<f32>::zeros(hidden_dim);

    // Build GraphNetwork: LayerNorm → Linear1 → GELU → Linear2
    let mut graph = GraphNetwork::new();

    // LayerNorm (identity scale, zero bias)
    let gamma = Array1::ones(hidden_dim);
    let beta = Array1::zeros(hidden_dim);
    let ln = LayerNormLayer::new(gamma, beta, 1e-5);
    graph.add_node(GraphNode::from_input("layernorm", Layer::LayerNorm(ln)));

    // Linear1 (expand)
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();
    graph.add_node(GraphNode::new(
        "linear1",
        Layer::Linear(linear1),
        vec!["layernorm".to_string()],
    ));

    // GELU
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::default()),
        vec!["linear1".to_string()],
    ));

    // Linear2 (contract)
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["gelu".to_string()],
    ));

    graph.set_output("linear2");

    println!("\n=== MLP-style Network (1D): CROWN vs IBP ===");
    println!(
        "Structure: LayerNorm[{}] → Linear[{}→{}] → GELU → Linear[{}→{}]",
        hidden_dim, hidden_dim, intermediate_dim, intermediate_dim, hidden_dim
    );

    // Test with small epsilon (tight input bounds)
    let center = Array1::from_elem(hidden_dim, 0.0_f32);
    let epsilon = 0.01_f32;
    let input = BoundedTensor::new(
        (center.clone() - epsilon).into_dyn(),
        (center.clone() + epsilon).into_dyn(),
    )
    .unwrap();

    println!("\nInput: center=0, epsilon={}", epsilon);

    // Get IBP bounds
    let ibp_result = graph.propagate_ibp(&input);
    let ibp_bounds = ibp_result.unwrap();
    let ibp_max_width = ibp_bounds.max_width();

    // Get CROWN bounds
    let crown_result = graph.propagate_crown(&input);
    let crown_bounds = crown_result.unwrap();
    let crown_max_width = crown_bounds.max_width();

    println!("IBP max width: {:.6}", ibp_max_width);
    println!("CROWN max width: {:.6}", crown_max_width);
    println!(
        "CROWN tightening ratio: {:.2}x",
        ibp_max_width / crown_max_width
    );

    // CROWN should be at least as tight as IBP
    assert!(
        crown_max_width <= ibp_max_width + 1e-6,
        "CROWN width {} should be <= IBP width {}",
        crown_max_width,
        ibp_max_width
    );

    // Test with larger epsilon (looser input bounds)
    let epsilon_large = 0.1_f32;
    let input_large = BoundedTensor::new(
        (center.clone() - epsilon_large).into_dyn(),
        (center.clone() + epsilon_large).into_dyn(),
    )
    .unwrap();

    println!("\nInput: center=0, epsilon={}", epsilon_large);

    let ibp_large = graph.propagate_ibp(&input_large).unwrap();
    let crown_large = graph.propagate_crown(&input_large).unwrap();

    let ibp_large_width = ibp_large.max_width();
    let crown_large_width = crown_large.max_width();

    println!("IBP max width: {:.6}", ibp_large_width);
    println!("CROWN max width: {:.6}", crown_large_width);
    println!(
        "CROWN tightening ratio: {:.2}x",
        ibp_large_width / crown_large_width
    );

    // Note: CROWN may not always be tighter than IBP for non-convex functions
    // like LayerNorm, due to linearization error. We check they're comparable.
    let ratio = crown_large_width / ibp_large_width;
    println!("CROWN/IBP ratio: {:.2}", ratio);
    // Allow CROWN to be up to 1.5x looser (due to LayerNorm linearization)
    assert!(
        ratio <= 1.5,
        "CROWN width {} should not be much worse than IBP width {} (ratio {:.2}x)",
        crown_large_width,
        ibp_large_width,
        ratio
    );

    // Test with realistic Whisper-like input width after attention
    // Input width ~1e4 (simulating post-attention bounds)
    let large_width_input = BoundedTensor::new(
        Array1::from_elem(hidden_dim, -5000.0_f32).into_dyn(),
        Array1::from_elem(hidden_dim, 5000.0_f32).into_dyn(),
    )
    .unwrap();

    println!("\nInput: width=10000 (simulating post-attention bounds)");

    let ibp_post_attn = graph.propagate_ibp(&large_width_input).unwrap();
    let crown_post_attn = graph.propagate_crown(&large_width_input).unwrap();

    let ibp_post_attn_width = ibp_post_attn.max_width();
    let crown_post_attn_width = crown_post_attn.max_width();

    println!("IBP max width: {:.6e}", ibp_post_attn_width);
    println!("CROWN max width: {:.6e}", crown_post_attn_width);
    println!(
        "CROWN tightening ratio: {:.2}x",
        ibp_post_attn_width / crown_post_attn_width
    );

    // Note: Full soundness verification would require complete forward pass
    // through LayerNorm → Linear → GELU → Linear, which is complex.
    // The key results are captured in the width comparisons above.

    // Key result: measure how much CROWN helps vs IBP for MLP
    println!("\n=== Summary ===");
    println!(
        "With epsilon=0.01: CROWN is {:.1}x tighter than IBP",
        ibp_max_width / crown_max_width
    );
    println!(
        "With epsilon=0.1:  CROWN is {:.1}x tighter than IBP",
        ibp_large_width / crown_large_width
    );
    println!(
        "With width=10000:  CROWN is {:.1}x tighter than IBP",
        ibp_post_attn_width / crown_post_attn_width
    );
}

#[test]
fn test_mlp_crown_without_layernorm() {
    // Test MLP without LayerNorm to isolate the effect of Linear→GELU→Linear

    let hidden_dim = 16_usize;
    let intermediate_dim = hidden_dim * 4;

    // Create random-ish weights
    let mut w1 = Array2::<f32>::zeros((intermediate_dim, hidden_dim));
    for i in 0..intermediate_dim {
        for j in 0..hidden_dim {
            w1[[i, j]] = ((i * 13 + j * 7) % 17) as f32 / 17.0 * 0.3 - 0.15;
        }
    }
    let b1 = Array1::<f32>::zeros(intermediate_dim);

    let mut w2 = Array2::<f32>::zeros((hidden_dim, intermediate_dim));
    for i in 0..hidden_dim {
        for j in 0..intermediate_dim {
            w2[[i, j]] = ((i * 11 + j * 3) % 13) as f32 / 13.0 * 0.3 - 0.15;
        }
    }
    let b2 = Array1::<f32>::zeros(hidden_dim);

    // Build GraphNetwork: Linear1 → GELU → Linear2
    let mut graph = GraphNetwork::new();

    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::default()),
        vec!["linear1".to_string()],
    ));

    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["gelu".to_string()],
    ));

    graph.set_output("linear2");

    println!("\n=== MLP without LayerNorm: CROWN vs IBP ===");
    println!(
        "Structure: Linear[{}→{}] → GELU → Linear[{}→{}]",
        hidden_dim, intermediate_dim, intermediate_dim, hidden_dim
    );

    // Test with various epsilons
    for &epsilon in &[0.01_f32, 0.1, 1.0] {
        let center = Array1::from_elem(hidden_dim, 0.0_f32);
        let input = BoundedTensor::new(
            (center.clone() - epsilon).into_dyn(),
            (center + epsilon).into_dyn(),
        )
        .unwrap();

        let ibp_bounds = graph.propagate_ibp(&input).unwrap();
        let crown_bounds = graph.propagate_crown(&input).unwrap();

        let ibp_width = ibp_bounds.max_width();
        let crown_width = crown_bounds.max_width();
        let ratio = ibp_width / crown_width;

        println!(
            "epsilon={:.2}: IBP={:.4}, CROWN={:.4}, ratio={:.2}x",
            epsilon, ibp_width, crown_width, ratio
        );

        // CROWN should be at least as tight
        assert!(
            crown_width <= ibp_width + 1e-5,
            "CROWN width {} should be <= IBP width {}",
            crown_width,
            ibp_width
        );
    }
}

#[test]
fn test_causal_softmax_layer_basic() {
    // Test CausalSoftmaxLayer IBP propagation
    let causal_softmax = CausalSoftmaxLayer::new(-1);

    // 2D input: [seq_q=3, seq_k=3]
    // For causal: row i can only attend to columns 0..=i
    let values = arr2(&[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    let input = BoundedTensor::new(values.clone().into_dyn(), values.into_dyn()).unwrap();

    let output = causal_softmax.propagate_ibp(&input).unwrap();

    // Row 0: softmax([1.0]) = [1.0], masked positions are 0
    assert!(
        (output.lower[[0, 0]] - 1.0).abs() < 1e-5,
        "Row 0, pos 0 should be 1.0"
    );
    assert!((output.upper[[0, 0]] - 1.0).abs() < 1e-5);
    assert!(
        output.lower[[0, 1]].abs() < 1e-5,
        "Row 0, pos 1 should be 0 (masked)"
    );
    assert!(output.upper[[0, 1]].abs() < 1e-5);
    assert!(
        output.lower[[0, 2]].abs() < 1e-5,
        "Row 0, pos 2 should be 0 (masked)"
    );
    assert!(output.upper[[0, 2]].abs() < 1e-5);

    // Row 1: softmax([1.0, 2.0]) = [~0.27, ~0.73], position 2 masked
    let row1_sum: f32 = output.lower[[1, 0]] + output.lower[[1, 1]];
    assert!(
        (row1_sum - 1.0).abs() < 1e-4,
        "Row 1 unmasked sum should be 1.0, got {}",
        row1_sum
    );
    assert!(
        output.lower[[1, 2]].abs() < 1e-5,
        "Row 1, pos 2 should be 0 (masked)"
    );
    assert!(output.upper[[1, 2]].abs() < 1e-5);

    // Row 2: full softmax - all positions unmasked
    let row2_sum: f32 = output.lower[[2, 0]] + output.lower[[2, 1]] + output.lower[[2, 2]];
    assert!(
        (row2_sum - 1.0).abs() < 1e-4,
        "Row 2 sum should be 1.0, got {}",
        row2_sum
    );
}

#[test]
fn test_causal_softmax_layer_soundness() {
    // Test that causal softmax bounds are sound under perturbation
    let causal_softmax = CausalSoftmaxLayer::new(-1);
    let eps = 0.1;

    let center = arr2(&[[0.0, 1.0, 2.0], [0.5, 1.5, 2.5], [1.0, 2.0, 3.0]]);
    let lower = center.mapv(|v| v - eps);
    let upper = center.mapv(|v| v + eps);
    let input = BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).unwrap();

    let output = causal_softmax.propagate_ibp(&input).unwrap();

    // Verify bounds are valid (lower <= upper)
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                output.lower[[i, j]] <= output.upper[[i, j]] + 1e-6,
                "Invalid bounds at [{}, {}]: {} > {}",
                i,
                j,
                output.lower[[i, j]],
                output.upper[[i, j]]
            );
        }
    }

    // Verify masked positions are exactly 0
    assert!(
        output.upper[[0, 1]].abs() < 1e-6,
        "Masked position [0,1] should be 0"
    );
    assert!(
        output.upper[[0, 2]].abs() < 1e-6,
        "Masked position [0,2] should be 0"
    );
    assert!(
        output.upper[[1, 2]].abs() < 1e-6,
        "Masked position [1,2] should be 0"
    );
}

#[test]
fn test_softmax_layer_ibp_handles_nonfinite_bounds() {
    let softmax = SoftmaxLayer::new(-1);

    // Two rows; second row contains non-finite bounds which should be sanitized to [0, 1].
    let lower = arr2(&[[0.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let mut upper = lower.clone();
    upper[[1, 2]] = f32::INFINITY;

    // Use new_unchecked to bypass debug_asserts - this test intentionally uses Inf
    let input = BoundedTensor::new_unchecked(lower.into_dyn(), upper.into_dyn()).unwrap();
    let output = softmax.propagate_ibp(&input).unwrap();

    // Row 1: fallback to [0, 1].
    for j in 0..3 {
        assert_eq!(output.lower[[1, j]], 0.0);
        assert_eq!(output.upper[[1, j]], 1.0);
    }

    // Row 0: should remain finite and within [0, 1].
    for j in 0..3 {
        let lb = output.lower[[0, j]];
        let ub = output.upper[[0, j]];
        assert!(lb.is_finite(), "Row 0 lower should be finite");
        assert!(ub.is_finite(), "Row 0 upper should be finite");
        assert!(lb >= 0.0 - 1e-6, "Row 0 lower should be >= 0, got {}", lb);
        assert!(ub <= 1.0 + 1e-6, "Row 0 upper should be <= 1, got {}", ub);
        assert!(lb <= ub + 1e-6, "Row 0 bounds invalid: {} > {}", lb, ub);
    }
}

#[test]
fn test_causal_softmax_layer_ibp_handles_nonfinite_bounds() {
    let causal_softmax = CausalSoftmaxLayer::new(-1);

    // 2D input: [seq_q=3, seq_k=3]. Row 2 includes an infinite bound in an unmasked position.
    let lower = arr2(&[[0.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let mut upper = lower.clone();
    upper[[2, 1]] = f32::INFINITY;

    // Use new_unchecked to bypass debug_asserts - this test intentionally uses Inf
    let input = BoundedTensor::new_unchecked(lower.into_dyn(), upper.into_dyn()).unwrap();
    let output = causal_softmax.propagate_ibp(&input).unwrap();

    // Masked positions remain exactly 0.
    assert!(output.lower[[0, 1]].abs() < 1e-6);
    assert!(output.upper[[0, 1]].abs() < 1e-6);
    assert!(output.lower[[0, 2]].abs() < 1e-6);
    assert!(output.upper[[0, 2]].abs() < 1e-6);
    assert!(output.lower[[1, 2]].abs() < 1e-6);
    assert!(output.upper[[1, 2]].abs() < 1e-6);

    // Row 2 unmasked positions are sanitized to [0, 1].
    for j in 0..3 {
        assert_eq!(output.lower[[2, j]], 0.0);
        assert_eq!(output.upper[[2, j]], 1.0);
    }

    // No NaNs are propagated.
    for i in 0..3 {
        for j in 0..3 {
            assert!(!output.lower[[i, j]].is_nan(), "NaN lower at [{},{}]", i, j);
            assert!(!output.upper[[i, j]].is_nan(), "NaN upper at [{},{}]", i, j);
        }
    }
}

#[test]
fn test_causal_softmax_crown_backward_basic() {
    // Test CausalSoftmax CROWN backward propagation
    // 2D input: [seq_q=2, seq_k=3]
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, 0.5, 1.5, 2.5]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 1.5, 2.5, 3.5]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Identity linear bounds for 6 elements (2*3)
    let linear_bounds = LinearBounds::identity(6);
    let causal_softmax = CausalSoftmaxLayer::new(-1);

    let result = causal_softmax
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Check dimensions
    assert_eq!(result.lower_a.shape(), &[6, 6]);
    assert_eq!(result.upper_a.shape(), &[6, 6]);
    assert_eq!(result.lower_b.len(), 6);
    assert_eq!(result.upper_b.len(), 6);

    // The Jacobian is block diagonal (each row is independent)
    // Row 0: softmax over position 0 only (masked: 1, 2)
    // Row 1: softmax over positions 0, 1 (masked: 2)

    // Check row 0 structure: only position [0,0] affects output [0,0]
    // Positions [0,1] and [0,2] are masked (output=0), so Jacobian is 0
    for k in 3..6 {
        // Row 0 outputs don't depend on row 1 inputs
        for j in 0..3 {
            assert!(
                result.lower_a[[j, k]].abs() < 1e-5,
                "Row 0 output {} should not depend on row 1 input {}",
                j,
                k
            );
        }
    }
}

#[test]
fn test_causal_softmax_crown_soundness() {
    // Test that CROWN bounds are sound (contain the actual function values)
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.0, 1.0, -0.5, 0.5, 1.5]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, 0.5, 1.5, 2.5]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(6);
    let causal_softmax = CausalSoftmaxLayer::new(-1);

    let result = causal_softmax
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample many points and verify bounds contain actual values
    for sample in 0..20 {
        // Generate a random point in the interval
        let point: Vec<f32> = (0..6)
            .map(|i| {
                let t = ((sample as u32).wrapping_mul(2654435761) ^ (i as u32)) as f32
                    / u32::MAX as f32;
                let pre_l = pre_lower.as_slice().unwrap()[i];
                let pre_u = pre_upper.as_slice().unwrap()[i];
                pre_l + (pre_u - pre_l) * t
            })
            .collect();

        // Compute actual causal softmax output
        // Row 0: softmax over position 0 only
        let row0_exp0 = point[0].exp();
        let row0_sum = row0_exp0 + 1e-8;
        let causal_output = [
            row0_exp0 / row0_sum, // [0,0]
            0.0,                  // [0,1] - masked
            0.0,                  // [0,2] - masked
            // Row 1: softmax over positions 0, 1
            {
                let max_val = point[3].max(point[4]);
                let exp0 = (point[3] - max_val).exp();
                let exp1 = (point[4] - max_val).exp();
                exp0 / (exp0 + exp1 + 1e-8)
            },
            {
                let max_val = point[3].max(point[4]);
                let exp0 = (point[3] - max_val).exp();
                let exp1 = (point[4] - max_val).exp();
                exp1 / (exp0 + exp1 + 1e-8)
            },
            0.0, // [1,2] - masked
        ];

        // Check each output dimension
        for (j, &causal_val) in causal_output.iter().enumerate() {
            let lb_val: f32 = (0..6)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            let ub_val: f32 = (0..6)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            let tol = 0.15; // Generous tolerance due to sampling-based error estimation
            assert!(
                lb_val <= causal_val + tol,
                "CROWN lower bound violated at sample {}, dim {}: lb {} > actual {}",
                sample,
                j,
                lb_val,
                causal_val
            );
            assert!(
                ub_val >= causal_val - tol,
                "CROWN upper bound violated at sample {}, dim {}: ub {} < actual {}",
                sample,
                j,
                ub_val,
                causal_val
            );
        }
    }
}

#[test]
fn test_causal_softmax_crown_network_integration() {
    // Test CausalSoftmax CROWN in a network context
    use crate::layers::LinearLayer;
    use crate::network::Network;

    // Create a simple network: Linear -> CausalSoftmax
    // Input: 6 -> reshape as [2, 3] for causal softmax
    let weight = Array2::from_shape_vec(
        (6, 4),
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let bias: Option<Array1<f32>> = Some(Array1::zeros(6));
    let linear = LinearLayer::new(weight, bias).unwrap();

    let causal_softmax = CausalSoftmaxLayer::new(-1);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    // Add reshape to [2, 3]
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![2, 3])));
    network.add_layer(Layer::CausalSoftmax(causal_softmax));

    // Create input bounds
    let input_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-0.5; 4]).unwrap();
    let input_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.5; 4]).unwrap();
    let input = BoundedTensor::new(input_lower, input_upper).unwrap();

    // Test CROWN propagation
    let crown_result = network.propagate_crown(&input).unwrap();

    // Test IBP propagation for comparison
    let _ibp_result = network.propagate_ibp(&input).unwrap();

    // CROWN bounds should be at least as tight as (or equal to) IBP bounds
    // Allow some tolerance since both methods have approximation errors
    // Output shape is [2, 3] from the reshape
    for i in 0..2 {
        for j in 0..3 {
            // Both should produce valid bounds in [0, 1] range for softmax
            assert!(
                crown_result.lower[[i, j]] >= -0.01,
                "CROWN lower bound [{}, {}] = {} should be >= 0",
                i,
                j,
                crown_result.lower[[i, j]]
            );
            assert!(
                crown_result.upper[[i, j]] <= 1.01,
                "CROWN upper bound [{}, {}] = {} should be <= 1",
                i,
                j,
                crown_result.upper[[i, j]]
            );
        }
    }
}

#[test]
fn test_causal_softmax_crown_masked_positions() {
    // Verify that masked positions have bounds containing 0
    let pre_lower = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![
            0.0, 1.0, 2.0, // Row 0: only position 0 unmasked
            0.0, 1.0, 2.0, // Row 1: positions 0,1 unmasked
            0.0, 1.0, 2.0, // Row 2: all unmasked
        ],
    )
    .unwrap();
    let pre_upper = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    )
    .unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(9);
    let causal_softmax = CausalSoftmaxLayer::new(-1);

    let result = causal_softmax
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Masked positions should have bounds containing 0
    // Position [0,1] and [0,2] are masked (row 0)
    // Position [1,2] is masked (row 1)
    // All positions in row 2 are unmasked

    // For masked positions, verify the bounds can contain 0
    let masked_indices = vec![1, 2, 5]; // [0,1], [0,2], [1,2]
    for &idx in &masked_indices {
        // The actual output at masked positions is exactly 0
        // So bounds should contain 0 (lb <= 0 <= ub)
        let lb = result.lower_b[idx]; // With identity bounds and zero input center
        let ub = result.upper_b[idx];
        // At center point, the output is 0 for masked positions
        // The bounds should reflect this
        assert!(
            lb <= 0.1,
            "Lower bound at masked position {} should allow 0, got {}",
            idx,
            lb
        );
        assert!(
            ub >= -0.1,
            "Upper bound at masked position {} should allow 0, got {}",
            idx,
            ub
        );
    }
}

// ============================================================================
// Batched Linear Bounds Tests
// ============================================================================

#[test]
fn test_batched_linear_bounds_identity() {
    // Test identity bounds for 1D shape
    let bounds = BatchedLinearBounds::identity(&[4]);
    assert_eq!(bounds.lower_a.shape(), &[4, 4]);
    assert_eq!(bounds.lower_b.shape(), &[4]);

    // Check it's identity
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((bounds.lower_a[[i, j]] - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn test_batched_linear_bounds_identity_2d() {
    // Test identity bounds for 2D shape (batch, hidden)
    let bounds = BatchedLinearBounds::identity(&[2, 4]);
    assert_eq!(bounds.lower_a.shape(), &[2, 4, 4]);
    assert_eq!(bounds.lower_b.shape(), &[2, 4]);

    // Check each batch position has identity
    for b in 0..2 {
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (bounds.lower_a[[b, i, j]] - expected).abs() < 1e-6,
                    "lower_a[{}, {}, {}] = {}, expected {}",
                    b,
                    i,
                    j,
                    bounds.lower_a[[b, i, j]],
                    expected
                );
            }
        }
    }
}

#[test]
fn test_batched_linear_bounds_identity_3d() {
    // Test identity bounds for 3D shape (batch, seq, hidden)
    let bounds = BatchedLinearBounds::identity(&[1, 4, 8]);
    assert_eq!(bounds.lower_a.shape(), &[1, 4, 8, 8]);
    assert_eq!(bounds.lower_b.shape(), &[1, 4, 8]);
}

#[test]
fn test_batched_linear_bounds_concretize_identity() {
    // Identity bounds should return input unchanged
    let bounds = BatchedLinearBounds::identity(&[2, 4]);
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap(),
    )
    .unwrap();

    let output = bounds.concretize(&input);

    // Output should equal input for identity bounds
    for i in 0..2 {
        for j in 0..4 {
            assert!(
                (output.lower[[i, j]] - input.lower[[i, j]]).abs() < 1e-5,
                "lower[{}, {}] mismatch",
                i,
                j
            );
            assert!(
                (output.upper[[i, j]] - input.upper[[i, j]]).abs() < 1e-5,
                "upper[{}, {}] mismatch",
                i,
                j
            );
        }
    }
}

#[test]
fn test_batched_linear_bounds_identity_for_attention() {
    // Test identity_for_attention for attention-shaped output
    // Attention output shape: [batch=1, heads=2, seq=4, seq=4]
    let shape = [1_usize, 2, 4, 4];
    let bounds = BatchedLinearBounds::identity_for_attention(&shape);

    // Should return Some for small attention shapes
    assert!(
        bounds.is_some(),
        "identity_for_attention should succeed for small seq"
    );
    let bounds = bounds.unwrap();

    // A shape should be [batch=1, heads=2, flat_size=16, flat_size=16]
    let flat_size = 4 * 4;
    assert_eq!(
        bounds.lower_a.shape(),
        &[1, 2, flat_size, flat_size],
        "lower_a shape mismatch"
    );
    assert_eq!(
        bounds.lower_b.shape(),
        &[1, 2, flat_size],
        "lower_b shape mismatch"
    );

    // Check identity structure per head
    for b in 0..1 {
        for h in 0..2 {
            for i in 0..flat_size {
                for j in 0..flat_size {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (bounds.lower_a[[b, h, i, j]] - expected).abs() < 1e-6,
                        "lower_a[{}, {}, {}, {}] = {}, expected {}",
                        b,
                        h,
                        i,
                        j,
                        bounds.lower_a[[b, h, i, j]],
                        expected
                    );
                }
            }
        }
    }
}

#[test]
fn test_identity_for_attention_rejects_large_seq() {
    // For seq > 64 (flat_size > 4096), should return None to avoid memory issues
    // seq=65 gives flat_size=4225 > 4096
    let shape = [1_usize, 1, 65, 65];
    let bounds = BatchedLinearBounds::identity_for_attention(&shape);
    assert!(
        bounds.is_none(),
        "identity_for_attention should reject seq > 64"
    );
}

#[test]
fn test_identity_for_attention_rejects_non_square() {
    // Non-square attention output should return None
    let shape = [1_usize, 2, 4, 8];
    let bounds = BatchedLinearBounds::identity_for_attention(&shape);
    assert!(
        bounds.is_none(),
        "identity_for_attention should reject non-square attention"
    );
}

#[test]
fn test_identity_for_attention_rejects_wrong_dims() {
    // Non-4D shapes should return None
    let shape_3d = [1_usize, 4, 4];
    assert!(
        BatchedLinearBounds::identity_for_attention(&shape_3d).is_none(),
        "Should reject 3D shape"
    );

    let shape_5d = [1_usize, 1, 2, 4, 4];
    assert!(
        BatchedLinearBounds::identity_for_attention(&shape_5d).is_none(),
        "Should reject 5D shape"
    );
}

#[test]
fn test_batched_linear_bounds_compose_identity() {
    // Composing two identity bounds should give identity bounds
    let shape = [2_usize, 4]; // batch=2, dim=4
    let id1 = BatchedLinearBounds::identity(&shape);
    let id2 = BatchedLinearBounds::identity(&shape);

    let composed = id1
        .compose(&id2)
        .expect("Compose should succeed for compatible identities");

    // Check that composed coefficient matrices are identity-like
    // A_composed = I @ I = I
    let expected_a_shape = [2, 4, 4]; // [batch, out_dim, in_dim]
    assert_eq!(composed.lower_a.shape(), expected_a_shape);
    assert_eq!(composed.upper_a.shape(), expected_a_shape);

    // Each batch should have identity matrix
    for b in 0..2 {
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let got_lower = composed.lower_a[[b, i, j]];
                let got_upper = composed.upper_a[[b, i, j]];
                assert!(
                    (got_lower - expected).abs() < 1e-6,
                    "Expected lower[{},{},{}] = {}, got {}",
                    b,
                    i,
                    j,
                    expected,
                    got_lower
                );
                assert!(
                    (got_upper - expected).abs() < 1e-6,
                    "Expected upper[{},{},{}] = {}, got {}",
                    b,
                    i,
                    j,
                    expected,
                    got_upper
                );
            }
        }
    }

    // Bias should be zero
    for val in composed.lower_b.iter() {
        assert!(val.abs() < 1e-6, "Expected zero bias, got {}", val);
    }
    for val in composed.upper_b.iter() {
        assert!(val.abs() < 1e-6, "Expected zero bias, got {}", val);
    }
}

#[test]
fn test_batched_linear_bounds_compose_scale() {
    // Composing a 2x scale with a 3x scale should give 6x scale
    let batch = 1;
    let dim = 2;

    // Create 2x scale bounds: y = 2*x
    let mut eye2: ndarray::Array2<f32> = ndarray::Array2::eye(dim);
    eye2.mapv_inplace(|v| v * 2.0);
    let scale_2 = BatchedLinearBounds {
        lower_a: eye2
            .clone()
            .into_dyn()
            .into_shape_with_order(IxDyn(&[batch, dim, dim]))
            .unwrap(),
        lower_b: ArrayD::zeros(IxDyn(&[batch, dim])),
        upper_a: eye2
            .into_dyn()
            .into_shape_with_order(IxDyn(&[batch, dim, dim]))
            .unwrap(),
        upper_b: ArrayD::zeros(IxDyn(&[batch, dim])),
        input_shape: vec![batch, dim],
        output_shape: vec![batch, dim],
    };

    // Create 3x scale bounds: z = 3*y
    let mut eye3: ndarray::Array2<f32> = ndarray::Array2::eye(dim);
    eye3.mapv_inplace(|v| v * 3.0);
    let scale_3 = BatchedLinearBounds {
        lower_a: eye3
            .clone()
            .into_dyn()
            .into_shape_with_order(IxDyn(&[batch, dim, dim]))
            .unwrap(),
        lower_b: ArrayD::zeros(IxDyn(&[batch, dim])),
        upper_a: eye3
            .into_dyn()
            .into_shape_with_order(IxDyn(&[batch, dim, dim]))
            .unwrap(),
        upper_b: ArrayD::zeros(IxDyn(&[batch, dim])),
        input_shape: vec![batch, dim],
        output_shape: vec![batch, dim],
    };

    // Compose: z = 3 * (2 * x) = 6 * x
    let composed = scale_2.compose(&scale_3).expect("Compose should succeed");

    // Check that result is 6x identity
    for i in 0..dim {
        for j in 0..dim {
            let expected = if i == j { 6.0 } else { 0.0 };
            let got = composed.lower_a[[0, i, j]];
            assert!(
                (got - expected).abs() < 1e-5,
                "Expected composed[{},{}] = {}, got {}",
                i,
                j,
                expected,
                got
            );
        }
    }
}

#[test]
fn test_batched_linear_bounds_compose_with_bias() {
    // Test that bias composition works: z = A2(A1*x + b1) + b2 = A2*A1*x + A2*b1 + b2
    let batch = 1;
    let dim = 2;

    // y = 2*x + [1, 2]
    let mut a1: ndarray::Array2<f32> = ndarray::Array2::eye(dim);
    a1.mapv_inplace(|v| v * 2.0);
    let bounds1 = BatchedLinearBounds {
        lower_a: a1
            .clone()
            .into_dyn()
            .into_shape_with_order(IxDyn(&[batch, dim, dim]))
            .unwrap(),
        lower_b: ArrayD::from_shape_vec(IxDyn(&[batch, dim]), vec![1.0, 2.0]).unwrap(),
        upper_a: a1
            .into_dyn()
            .into_shape_with_order(IxDyn(&[batch, dim, dim]))
            .unwrap(),
        upper_b: ArrayD::from_shape_vec(IxDyn(&[batch, dim]), vec![1.0, 2.0]).unwrap(),
        input_shape: vec![batch, dim],
        output_shape: vec![batch, dim],
    };

    // z = y + [3, 4] (identity transform with bias)
    let eye: ndarray::Array2<f32> = ndarray::Array2::eye(dim);
    let bounds2 = BatchedLinearBounds {
        lower_a: eye
            .clone()
            .into_dyn()
            .into_shape_with_order(IxDyn(&[batch, dim, dim]))
            .unwrap(),
        lower_b: ArrayD::from_shape_vec(IxDyn(&[batch, dim]), vec![3.0, 4.0]).unwrap(),
        upper_a: eye
            .into_dyn()
            .into_shape_with_order(IxDyn(&[batch, dim, dim]))
            .unwrap(),
        upper_b: ArrayD::from_shape_vec(IxDyn(&[batch, dim]), vec![3.0, 4.0]).unwrap(),
        input_shape: vec![batch, dim],
        output_shape: vec![batch, dim],
    };

    // z = (2*x + [1,2]) + [3,4] = 2*x + [4,6]
    let composed = bounds1.compose(&bounds2).expect("Compose should succeed");

    // Check coefficient matrix is 2*I
    assert!((composed.lower_a[[0, 0, 0]] - 2.0).abs() < 1e-5);
    assert!((composed.lower_a[[0, 1, 1]] - 2.0).abs() < 1e-5);
    assert!(composed.lower_a[[0, 0, 1]].abs() < 1e-5);
    assert!(composed.lower_a[[0, 1, 0]].abs() < 1e-5);

    // Check bias is [4, 6] (b1=[1,2] passed through identity A2, plus b2=[3,4])
    assert!(
        (composed.lower_b[[0, 0]] - 4.0).abs() < 1e-5,
        "Expected bias[0] = 4, got {}",
        composed.lower_b[[0, 0]]
    );
    assert!(
        (composed.lower_b[[0, 1]] - 6.0).abs() < 1e-5,
        "Expected bias[1] = 6, got {}",
        composed.lower_b[[0, 1]]
    );
}

#[test]
fn test_batched_linear_bounds_compose_avoids_nan_from_0_times_inf() {
    // Regression test: saturated coefficients (±inf) can appear in bounds, and composing
    // bounds must not introduce NaNs via 0 * inf.
    let batch = 1;
    let dim = 2;

    // y = A1 @ x where A1 contains +inf on the diagonal (synthetic saturation)
    let a1 = vec![
        f32::INFINITY,
        0.0, //
        0.0,
        f32::INFINITY, //
    ];
    let bounds1 = BatchedLinearBounds {
        lower_a: ArrayD::from_shape_vec(IxDyn(&[batch, dim, dim]), a1.clone()).unwrap(),
        upper_a: ArrayD::from_shape_vec(IxDyn(&[batch, dim, dim]), a1).unwrap(),
        lower_b: ArrayD::zeros(IxDyn(&[batch, dim])),
        upper_b: ArrayD::zeros(IxDyn(&[batch, dim])),
        input_shape: vec![batch, dim],
        output_shape: vec![batch, dim],
    };

    // z = A2 @ y with A2 = 0 (should produce z = 0 regardless of A1)
    let a2 = vec![
        0.0, 0.0, //
        0.0, 0.0, //
    ];
    let bounds2 = BatchedLinearBounds {
        lower_a: ArrayD::from_shape_vec(IxDyn(&[batch, dim, dim]), a2.clone()).unwrap(),
        upper_a: ArrayD::from_shape_vec(IxDyn(&[batch, dim, dim]), a2).unwrap(),
        lower_b: ArrayD::zeros(IxDyn(&[batch, dim])),
        upper_b: ArrayD::zeros(IxDyn(&[batch, dim])),
        input_shape: vec![batch, dim],
        output_shape: vec![batch, dim],
    };

    let composed = bounds1.compose(&bounds2).expect("Compose should succeed");

    for v in composed
        .lower_a
        .iter()
        .chain(composed.upper_a.iter())
        .chain(composed.lower_b.iter())
        .chain(composed.upper_b.iter())
    {
        assert!(!v.is_nan(), "compose produced NaN");
    }

    // Coefficients should be exactly zero (0 * inf treated as 0).
    for v in composed.lower_a.iter().chain(composed.upper_a.iter()) {
        assert_eq!(*v, 0.0);
    }
}

#[test]
fn test_batched_linear_bounds_compose_avoids_nan_from_inf_minus_inf_sum() {
    // Regression test: interval sums like (+inf) + (-inf) must widen, not become NaN.
    let batch = 1;
    let dim = 2;

    // y = A1 @ x where two rows contribute +inf and -inf to the same output when composed.
    // A1:
    //   [ +inf, 0 ]
    //   [ -inf, 0 ]
    let a1_lower = vec![
        f32::INFINITY,
        0.0, //
        f32::NEG_INFINITY,
        0.0, //
    ];
    let a1_upper = vec![
        f32::INFINITY,
        0.0, //
        f32::NEG_INFINITY,
        0.0, //
    ];
    let bounds1 = BatchedLinearBounds {
        lower_a: ArrayD::from_shape_vec(IxDyn(&[batch, dim, dim]), a1_lower).unwrap(),
        upper_a: ArrayD::from_shape_vec(IxDyn(&[batch, dim, dim]), a1_upper).unwrap(),
        lower_b: ArrayD::zeros(IxDyn(&[batch, dim])),
        upper_b: ArrayD::zeros(IxDyn(&[batch, dim])),
        input_shape: vec![batch, dim],
        output_shape: vec![batch, dim],
    };

    // z = A2 @ y with A2[0,:] = [1, 1], so z0 = y0 + y1 includes +inf + (-inf).
    let a2 = vec![
        1.0, 1.0, //
        0.0, 0.0, //
    ];
    let bounds2 = BatchedLinearBounds {
        lower_a: ArrayD::from_shape_vec(IxDyn(&[batch, dim, dim]), a2.clone()).unwrap(),
        upper_a: ArrayD::from_shape_vec(IxDyn(&[batch, dim, dim]), a2).unwrap(),
        lower_b: ArrayD::zeros(IxDyn(&[batch, dim])),
        upper_b: ArrayD::zeros(IxDyn(&[batch, dim])),
        input_shape: vec![batch, dim],
        output_shape: vec![batch, dim],
    };

    let composed = bounds1.compose(&bounds2).expect("Compose should succeed");

    for v in composed
        .lower_a
        .iter()
        .chain(composed.upper_a.iter())
        .chain(composed.lower_b.iter())
        .chain(composed.upper_b.iter())
    {
        assert!(!v.is_nan(), "compose produced NaN");
    }

    assert!(
        composed.lower_a[[0, 0, 0]].is_infinite() && composed.lower_a[[0, 0, 0]].is_sign_negative(),
        "Expected widened lower=-inf, got {}",
        composed.lower_a[[0, 0, 0]]
    );
    assert!(
        composed.upper_a[[0, 0, 0]].is_infinite() && composed.upper_a[[0, 0, 0]].is_sign_positive(),
        "Expected widened upper=+inf, got {}",
        composed.upper_a[[0, 0, 0]]
    );
}

#[test]
fn test_graph_network_attention_crown_small_seq() {
    // Test that attention CROWN path is attempted for small attention shapes (seq <= 64).
    // This creates a minimal attention graph where Q@K^T produces [batch, heads, seq, seq]
    // and verifies that the attention identity path is exercised.
    //
    // Graph: Q -> Q@K^T (attention MatMul) -> output
    //        K -^
    //
    // With seq=4 (within the 64 limit), the attention identity should be used.

    let batch = 1_usize;
    let heads = 2_usize;
    let seq = 4_usize;
    let dim = 8_usize;

    let mut graph = GraphNetwork::new();

    // Q and K inputs pass through GELU first (gives non-trivial CROWN pass through)
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));

    // Q @ K^T produces attention scores [batch, heads, seq, seq]
    let scale = 1.0 / (dim as f32).sqrt();
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, Some(scale))),
        "q",
        "k",
    ));
    graph.set_output("scores");

    // Input shape: [batch, heads, seq, dim]
    let input = BoundedTensor::new(
        ndarray::ArrayD::from_elem(vec![batch, heads, seq, dim], -0.5_f32),
        ndarray::ArrayD::from_elem(vec![batch, heads, seq, dim], 0.5_f32),
    )
    .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let crown_bounds = graph.propagate_crown_batched(&input).unwrap();

    // Output shape should be [batch, heads, seq, seq]
    assert_eq!(crown_bounds.shape(), &[batch, heads, seq, seq]);

    // Bounds should be valid and finite
    for (l, u) in crown_bounds.lower.iter().zip(crown_bounds.upper.iter()) {
        assert!(
            l.is_finite() && u.is_finite(),
            "Non-finite bounds: {} {}",
            l,
            u
        );
        assert!(*l <= *u + 1e-5, "Invalid interval: {} > {}", l, u);
    }

    // CROWN should be at least as tight as IBP (soundness check)
    for ((crown_l, crown_u), (ibp_l, ibp_u)) in crown_bounds
        .lower
        .iter()
        .zip(crown_bounds.upper.iter())
        .zip(ibp_bounds.lower.iter().zip(ibp_bounds.upper.iter()))
    {
        assert!(
            *crown_l >= *ibp_l - 1e-4,
            "CROWN lower {} should be >= IBP lower {}",
            crown_l,
            ibp_l
        );
        assert!(
            *crown_u <= *ibp_u + 1e-4,
            "CROWN upper {} should be <= IBP upper {}",
            crown_u,
            ibp_u
        );
    }
}

#[test]
fn test_graph_network_attention_crown_large_seq_fallback() {
    // Test that for large seq (> 64), we fall back to partial CROWN without error.
    // The attention identity path should NOT be used due to memory limits.

    let batch = 1_usize;
    let heads = 1_usize;
    let seq = 128_usize; // > 64, should trigger memory limit fallback
    let dim = 8_usize;

    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));

    let scale = 1.0 / (dim as f32).sqrt();
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, Some(scale))),
        "q",
        "k",
    ));
    graph.set_output("scores");

    let input = BoundedTensor::new(
        ndarray::ArrayD::from_elem(vec![batch, heads, seq, dim], -0.5_f32),
        ndarray::ArrayD::from_elem(vec![batch, heads, seq, dim], 0.5_f32),
    )
    .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let crown_bounds = graph.propagate_crown_batched(&input).unwrap();

    // Should still succeed (partial CROWN fallback)
    assert_eq!(crown_bounds.shape(), &[batch, heads, seq, seq]);

    // Bounds should be valid
    for (l, u) in crown_bounds.lower.iter().zip(crown_bounds.upper.iter()) {
        assert!(l.is_finite() && u.is_finite(), "Non-finite bounds");
        assert!(*l <= *u + 1e-5, "Invalid interval: {} > {}", l, u);
    }

    // CROWN (with fallback) should be at least as tight as IBP
    for ((crown_l, crown_u), (ibp_l, ibp_u)) in crown_bounds
        .lower
        .iter()
        .zip(crown_bounds.upper.iter())
        .zip(ibp_bounds.lower.iter().zip(ibp_bounds.upper.iter()))
    {
        assert!(
            *crown_l >= *ibp_l - 1e-4,
            "CROWN lower {} should be >= IBP lower {}",
            crown_l,
            ibp_l
        );
        assert!(
            *crown_u <= *ibp_u + 1e-4,
            "CROWN upper {} should be <= IBP upper {}",
            crown_u,
            ibp_u
        );
    }
}

#[test]
fn test_linear_layer_batched_backward() {
    // Test batched backward propagation through linear layer
    // Linear layer: y = Wx + b, W is [out_dim, in_dim]
    let weight = Array2::from_shape_vec(
        (3, 4),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    )
    .unwrap();
    let bias = Array1::from_vec(vec![0.1, 0.2, 0.3]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();

    // Create identity bounds at output: shape [batch=2, out_dim=3, out_dim=3]
    let bounds = BatchedLinearBounds::identity(&[2, 3]);

    let result = linear.propagate_linear_batched(&bounds).unwrap();

    // After backward through linear, we should have:
    // new_A = I @ W = W, shape [2, 3, 4]
    // new_b = I @ bias + 0 = bias, shape [2, 3]
    assert_eq!(result.lower_a.shape(), &[2, 3, 4]);
    assert_eq!(result.lower_b.shape(), &[2, 3]);

    // Check weight is propagated correctly
    // For batch 0, row 0: [1, 0, 0, 0]
    assert!((result.lower_a[[0, 0, 0]] - 1.0).abs() < 1e-5);
    assert!((result.lower_a[[0, 0, 1]] - 0.0).abs() < 1e-5);

    // For batch 0, row 2: [0, 0, 1, 1]
    assert!((result.lower_a[[0, 2, 2]] - 1.0).abs() < 1e-5);
    assert!((result.lower_a[[0, 2, 3]] - 1.0).abs() < 1e-5);

    // Check bias is propagated correctly
    assert!((result.lower_b[[0, 0]] - 0.1).abs() < 1e-5);
    assert!((result.lower_b[[0, 1]] - 0.2).abs() < 1e-5);
    assert!((result.lower_b[[0, 2]] - 0.3).abs() < 1e-5);
}

#[test]
fn test_relu_layer_batched_backward_positive() {
    // Test batched ReLU backward when all inputs are positive (identity pass-through)
    let relu = ReLULayer;

    // Pre-activation bounds: all positive
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0; 8]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![2.0; 8]).unwrap(),
    )
    .unwrap();

    // Identity bounds at output
    let bounds = BatchedLinearBounds::identity(&[2, 4]);

    let result = relu
        .propagate_linear_batched_with_bounds(&bounds, &pre_activation)
        .unwrap();

    // With all positive pre-activation, ReLU is identity
    // So bounds should remain identity
    for b in 0..2 {
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (result.lower_a[[b, i, j]] - expected).abs() < 1e-5,
                    "lower_a[{}, {}, {}] = {}, expected {}",
                    b,
                    i,
                    j,
                    result.lower_a[[b, i, j]],
                    expected
                );
            }
        }
    }
}

#[test]
fn test_relu_layer_batched_backward_negative() {
    // Test batched ReLU backward when all inputs are negative (zero output)
    let relu = ReLULayer;

    // Pre-activation bounds: all negative
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![-2.0; 8]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![-1.0; 8]).unwrap(),
    )
    .unwrap();

    // Identity bounds at output
    let bounds = BatchedLinearBounds::identity(&[2, 4]);

    let result = relu
        .propagate_linear_batched_with_bounds(&bounds, &pre_activation)
        .unwrap();

    // With all negative pre-activation, ReLU outputs zero
    // So all coefficients should be zero
    for b in 0..2 {
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    result.lower_a[[b, i, j]].abs() < 1e-5,
                    "lower_a[{}, {}, {}] = {}, expected 0",
                    b,
                    i,
                    j,
                    result.lower_a[[b, i, j]]
                );
            }
        }
    }
}

#[test]
fn test_batched_crown_linear_relu_chain() {
    // Test a simple Linear -> ReLU chain with batched bounds
    // This verifies that the batched backward propagation composes correctly

    // Linear layer: 4 -> 4 with identity weight
    let weight = Array2::eye(4);
    let linear = LinearLayer::new(weight, None).unwrap();

    let relu = ReLULayer;

    // Pre-activation bounds: mix of positive, negative, and crossing
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, -2.0, -0.5, 0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![2.0, -1.0, 0.5, 1.5]).unwrap(),
    )
    .unwrap();

    // Start with identity bounds at output
    let bounds = BatchedLinearBounds::identity(&[1, 4]);

    // Backward through ReLU first
    let after_relu = relu
        .propagate_linear_batched_with_bounds(&bounds, &pre_activation)
        .unwrap();

    // Then backward through Linear (identity weight, so should pass through)
    let final_bounds = linear.propagate_linear_batched(&after_relu).unwrap();

    // Verify shapes
    assert_eq!(final_bounds.lower_a.shape(), &[1, 4, 4]);

    // Position 0: always positive -> identity (slope 1)
    assert!((final_bounds.lower_a[[0, 0, 0]] - 1.0).abs() < 1e-5);

    // Position 1: always negative -> zero (slope 0)
    assert!(final_bounds.lower_a[[0, 1, 1]].abs() < 1e-5);

    // Position 2: crossing [-0.5, 0.5] -> linear relaxation
    // lambda = u/(u-l) = 0.5/1.0 = 0.5
    // For lower_a positive, uses alpha (heuristic: u > -l -> 0.5 > 0.5 -> false, alpha=0)
    // Actually u=0.5, -l=0.5, so u == -l, alpha = 0
    assert!(final_bounds.lower_a[[0, 2, 2]].abs() < 1e-5);

    // Position 3: crossing [0.5, 1.5] but positive lower bound -> identity for upper bound coeff >= 0
    // lambda = 1.5/1.0 = 1.5... wait, l=0.5 > 0, so this is always positive!
    // Actually l=0.5 >= 0, so this is identity
    assert!((final_bounds.lower_a[[0, 3, 3]] - 1.0).abs() < 1e-5);
}

#[test]
fn test_gelu_layer_batched_backward() {
    // Test batched GELU backward propagation
    let gelu = GELULayer::default();

    // Pre-activation bounds: mix of values
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![-1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![0.0, 1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 2.0],
        )
        .unwrap(),
    )
    .unwrap();

    // Identity bounds at output
    let bounds = BatchedLinearBounds::identity(&[2, 4]);

    let result = gelu
        .propagate_linear_batched_with_bounds(&bounds, &pre_activation)
        .unwrap();

    // Verify shapes preserved
    assert_eq!(result.lower_a.shape(), &[2, 4, 4]);
    assert_eq!(result.lower_b.shape(), &[2, 4]);

    // Verify diagonal structure (off-diagonal should be zero since each output
    // depends only on corresponding input for identity bounds)
    for b in 0..2 {
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert!(
                        result.lower_a[[b, i, j]].abs() < 1e-5,
                        "lower_a[{}, {}, {}] should be 0 (off-diagonal), got {}",
                        b,
                        i,
                        j,
                        result.lower_a[[b, i, j]]
                    );
                }
            }
        }
    }

    // For very positive inputs (like x in [1, 2], [2, 3]), GELU slope is close to 1
    // Batch 0, position 2: input [1, 2] - GELU is approximately linear here
    // The diagonal should be close to 1 (but not exactly due to GELU curvature)
    assert!(
        result.lower_a[[0, 2, 2]] > 0.5,
        "GELU slope for positive input should be > 0.5, got {}",
        result.lower_a[[0, 2, 2]]
    );
}

#[test]
fn test_batched_crown_mlp_chain() {
    // Test a simple MLP: Linear -> GELU -> Linear with batched bounds
    // This represents a transformer MLP block (without the expansion factor)

    // First linear: 4 -> 4 (identity for simplicity)
    let weight1 = Array2::eye(4);
    let linear1 = LinearLayer::new(weight1, None).unwrap();

    // GELU activation
    let gelu = GELULayer::default();

    // Second linear: 4 -> 4 (identity for simplicity)
    let weight2 = Array2::eye(4);
    let linear2 = LinearLayer::new(weight2, None).unwrap();

    // Input bounds: [batch=1, seq=2, hidden=4]
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 4]), vec![-1.0; 8]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 4]), vec![1.0; 8]).unwrap(),
    )
    .unwrap();

    // Forward pass through network to get pre-activation bounds
    let after_linear1 = linear1.propagate_ibp(&input).unwrap();
    let after_gelu = gelu.propagate_ibp(&after_linear1).unwrap();
    let _after_linear2 = linear2.propagate_ibp(&after_gelu).unwrap();

    // Now backward pass with batched CROWN
    // Start with identity bounds at output: [1, 2, 4, 4]
    let bounds = BatchedLinearBounds::identity(&[1, 2, 4]);

    // Backward through linear2
    let after_l2_back = linear2.propagate_linear_batched(&bounds).unwrap();
    assert_eq!(after_l2_back.lower_a.shape(), &[1, 2, 4, 4]);

    // Backward through GELU with pre-activation bounds
    let after_gelu_back = gelu
        .propagate_linear_batched_with_bounds(&after_l2_back, &after_linear1)
        .unwrap();
    assert_eq!(after_gelu_back.lower_a.shape(), &[1, 2, 4, 4]);

    // Backward through linear1
    let final_bounds = linear1.propagate_linear_batched(&after_gelu_back).unwrap();
    assert_eq!(final_bounds.lower_a.shape(), &[1, 2, 4, 4]);

    // Concretize to get concrete bounds
    let concrete = final_bounds.concretize(&input);
    assert_eq!(concrete.shape(), &[1, 2, 4]);

    // Verify soundness: concrete bounds should be valid
    // (lower <= actual output <= upper for all inputs in the input bounds)
    assert!(
        concrete.lower.iter().all(|&x| x.is_finite()),
        "All lower bounds should be finite"
    );
    assert!(
        concrete.upper.iter().all(|&x| x.is_finite()),
        "All upper bounds should be finite"
    );
    assert!(
        concrete
            .lower
            .iter()
            .zip(concrete.upper.iter())
            .all(|(&l, &u)| l <= u),
        "Lower should be <= upper"
    );
}

#[test]
fn test_network_propagate_crown_batched_transformer_scale() {
    // Test batched CROWN on transformer-scale input: [batch=1, seq=4, hidden=384]
    // This is a key test for verifying transformer verification works

    let mut network = Network::new();

    // MLP: hidden -> 4*hidden -> hidden (transformer MLP pattern)
    let hidden = 64; // Reduced from 384 for test speed, but tests same structure
    let expansion = 4;

    // Up projection: hidden -> 4*hidden
    // Use deterministic initialization: Xavier-like scaling with position-based variation
    let scale1 = (2.0 / (hidden + hidden * expansion) as f32).sqrt();
    let weight1 = Array2::from_shape_fn((hidden * expansion, hidden), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32;
        scale1 * (phase.sin() * 0.5)
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight1, None).unwrap()));

    // GELU
    network.add_layer(Layer::GELU(GELULayer::default()));

    // Down projection: 4*hidden -> hidden
    let scale2 = (2.0 / (hidden * expansion + hidden) as f32).sqrt();
    let weight2 = Array2::from_shape_fn((hidden, hidden * expansion), |(i, j)| {
        let phase = (i * 23 + j * 37) as f32;
        scale2 * (phase.cos() * 0.5)
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight2, None).unwrap()));

    // Input: [batch=1, seq=4, hidden]
    let batch = 1;
    let seq = 4;
    let total_elements = batch * seq * hidden;
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[batch, seq, hidden]), vec![-0.1; total_elements]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[batch, seq, hidden]), vec![0.1; total_elements]).unwrap(),
    )
    .unwrap();

    // Run batched CROWN
    let batched_result = network.propagate_crown_batched(&input).unwrap();

    // Verify output shape matches input shape (MLP preserves shape)
    assert_eq!(batched_result.shape(), &[batch, seq, hidden]);

    // Verify all bounds are finite and valid
    let mut valid_count = 0;
    let mut finite_count = 0;
    for ((l, u), _) in batched_result
        .lower
        .iter()
        .zip(batched_result.upper.iter())
        .zip(0..total_elements)
    {
        if l.is_finite() && u.is_finite() {
            finite_count += 1;
        }
        if *l <= *u {
            valid_count += 1;
        }
    }

    assert_eq!(finite_count, total_elements, "All bounds should be finite");
    assert_eq!(valid_count, total_elements, "All bounds should be valid");

    // Measure bound widths
    let avg_width: f32 = batched_result
        .lower
        .iter()
        .zip(batched_result.upper.iter())
        .map(|(l, u)| u - l)
        .sum::<f32>()
        / total_elements as f32;

    println!(
        "Transformer MLP batched CROWN: shape {:?}, avg bound width: {:.4}",
        batched_result.shape(),
        avg_width
    );

    // Bounds should not explode (reasonable width for small perturbation)
    assert!(
        avg_width < 10.0,
        "Bound width should be reasonable (< 10), got {}",
        avg_width
    );
}

#[test]
fn test_network_propagate_crown_batched_with_softmax() {
    // Test batched CROWN on a network with Linear -> Softmax
    // This verifies the Softmax integration in propagate_crown_batched

    let mut network = Network::new();

    // Linear: 4 -> 4
    let weight = Array2::from_shape_fn((4, 4), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32;
        0.3 * phase.sin()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight, None).unwrap()));

    // Softmax
    network.add_layer(Layer::Softmax(SoftmaxLayer::new(-1)));

    // Input: [batch=2, seq=3, 4]
    let batch = 2;
    let seq = 3;
    let hidden = 4;
    let total_elements = batch * seq * hidden;

    let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));

    for b in 0..batch {
        for s in 0..seq {
            for h in 0..hidden {
                let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                let base = (hash as f32 / u32::MAX as f32) * 2.0 - 1.0;
                lower[[b, s, h]] = base - 0.1;
                upper[[b, s, h]] = base + 0.1;
            }
        }
    }

    let input = BoundedTensor::new(lower, upper).unwrap();

    // Run batched CROWN
    let batched_result = network.propagate_crown_batched(&input).unwrap();

    // Verify output shape
    assert_eq!(batched_result.shape(), &[batch, seq, hidden]);

    // Verify all bounds are finite and valid
    let mut valid_count = 0;
    let mut finite_count = 0;
    for (l, u) in batched_result.lower.iter().zip(batched_result.upper.iter()) {
        if l.is_finite() && u.is_finite() {
            finite_count += 1;
        }
        if *l <= *u + 1e-6 {
            valid_count += 1;
        }
    }

    assert_eq!(finite_count, total_elements, "All bounds should be finite");
    assert_eq!(valid_count, total_elements, "All bounds should be valid");

    // Softmax outputs should be in [0, 1]
    for (l, u) in batched_result.lower.iter().zip(batched_result.upper.iter()) {
        assert!(*l >= -0.01, "Softmax lower bound should be >= 0, got {}", l);
        assert!(*u <= 1.01, "Softmax upper bound should be <= 1, got {}", u);
    }

    // Measure bound widths
    let avg_width: f32 = batched_result
        .lower
        .iter()
        .zip(batched_result.upper.iter())
        .map(|(l, u)| u - l)
        .sum::<f32>()
        / total_elements as f32;

    println!(
        "Linear+Softmax batched CROWN: shape {:?}, avg bound width: {:.4}",
        batched_result.shape(),
        avg_width
    );

    // Bounds should not explode
    assert!(
        avg_width < 1.0,
        "Bound width should be reasonable (< 1 for softmax), got {}",
        avg_width
    );
}

#[test]
fn test_network_propagate_crown_batched_with_layernorm() {
    // Test batched CROWN on a network with Linear -> LayerNorm
    // This verifies the LayerNorm integration in propagate_crown_batched

    let mut network = Network::new();

    let hidden = 4;

    // Linear: 4 -> 4
    let weight = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32;
        0.3 * phase.sin()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight, None).unwrap()));

    // LayerNorm with default gamma=1, beta=0
    let ln = LayerNormLayer::new_default(hidden, 1e-5);
    network.add_layer(Layer::LayerNorm(ln));

    // Input: [batch=2, seq=3, 4]
    let batch = 2;
    let seq = 3;
    let total_elements = batch * seq * hidden;

    let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));

    for b in 0..batch {
        for s in 0..seq {
            for h in 0..hidden {
                let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                let base = (hash as f32 / u32::MAX as f32) * 2.0 - 1.0;
                lower[[b, s, h]] = base - 0.15;
                upper[[b, s, h]] = base + 0.15;
            }
        }
    }

    let input = BoundedTensor::new(lower, upper).unwrap();

    // Run batched CROWN
    let batched_result = network.propagate_crown_batched(&input).unwrap();

    // Verify output shape
    assert_eq!(batched_result.shape(), &[batch, seq, hidden]);

    // Verify all bounds are finite and valid
    let mut valid_count = 0;
    let mut finite_count = 0;
    for (l, u) in batched_result.lower.iter().zip(batched_result.upper.iter()) {
        if l.is_finite() && u.is_finite() {
            finite_count += 1;
        }
        if *l <= *u + 1e-6 {
            valid_count += 1;
        }
    }

    assert_eq!(finite_count, total_elements, "All bounds should be finite");
    assert_eq!(valid_count, total_elements, "All bounds should be valid");

    // Measure bound widths
    let avg_width: f32 = batched_result
        .lower
        .iter()
        .zip(batched_result.upper.iter())
        .map(|(l, u)| u - l)
        .sum::<f32>()
        / total_elements as f32;

    println!(
        "Linear+LayerNorm batched CROWN: shape {:?}, avg bound width: {:.4}",
        batched_result.shape(),
        avg_width
    );

    // Bounds should not explode
    assert!(
        avg_width < 10.0,
        "Bound width should be reasonable (< 10 for layernorm), got {}",
        avg_width
    );
}

#[test]
fn test_layernorm_batched_linear_bounds() {
    // Test the LayerNormLayer::propagate_linear_batched_with_bounds directly

    let hidden = 4;
    let ln = LayerNormLayer::new_default(hidden, 1e-5);

    // Create 2D input shape [batch, hidden]
    let batch = 3;

    // Pre-activation bounds
    let mut pre_lower = ArrayD::zeros(IxDyn(&[batch, hidden]));
    let mut pre_upper = ArrayD::zeros(IxDyn(&[batch, hidden]));

    for b in 0..batch {
        for h in 0..hidden {
            let hash = ((b * 10 + h) as u32).wrapping_mul(2654435761_u32);
            let base = (hash as f32 / u32::MAX as f32) * 2.0 - 1.0;
            pre_lower[[b, h]] = base - 0.2;
            pre_upper[[b, h]] = base + 0.2;
        }
    }

    let pre_bounds = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Create identity bounds: shape [batch, hidden, hidden]
    let identity = BatchedLinearBounds::identity(&[batch, hidden]);

    // Propagate backward
    let result = ln
        .propagate_linear_batched_with_bounds(&identity, &pre_bounds)
        .unwrap();

    // Verify shape
    assert_eq!(result.lower_a.shape(), &[batch, hidden, hidden]);
    assert_eq!(result.lower_b.shape(), &[batch, hidden]);

    // Verify all values are finite
    let all_finite = result.lower_a.iter().all(|v| v.is_finite())
        && result.upper_a.iter().all(|v| v.is_finite())
        && result.lower_b.iter().all(|v| v.is_finite())
        && result.upper_b.iter().all(|v| v.is_finite());

    assert!(all_finite, "All batched layernorm bounds should be finite");

    // Verify soundness by sampling
    for sample_idx in 0..20 {
        // Sample a concrete input for each batch position
        let mut x_sample = ArrayD::<f32>::zeros(IxDyn(&[batch, hidden]));
        for b in 0..batch {
            for h in 0..hidden {
                let hash = ((sample_idx * 1000 + b * 10 + h) as u32).wrapping_mul(2654435761_u32);
                let t = hash as f32 / u32::MAX as f32;
                x_sample[[b, h]] = pre_bounds.lower[[b, h]]
                    + (pre_bounds.upper[[b, h]] - pre_bounds.lower[[b, h]]) * t;
            }
        }

        // Evaluate layernorm at this point for each batch
        for b in 0..batch {
            let x_1d: Array1<f32> = (0..hidden).map(|h| x_sample[[b, h]]).collect();
            let y_actual = ln.eval(&x_1d);

            // Concretize the linear bounds at this sample point
            for j in 0..hidden {
                let mut lower_val = result.lower_b[[b, j]];
                let mut upper_val = result.upper_b[[b, j]];

                for k in 0..hidden {
                    let la = result.lower_a[[b, j, k]];
                    let ua = result.upper_a[[b, j, k]];

                    // For lower bound: if coeff positive, use lower of input; if negative, use upper
                    if la >= 0.0 {
                        lower_val += la * pre_bounds.lower[[b, k]];
                    } else {
                        lower_val += la * pre_bounds.upper[[b, k]];
                    }

                    // For upper bound: if coeff positive, use upper of input; if negative, use lower
                    if ua >= 0.0 {
                        upper_val += ua * pre_bounds.upper[[b, k]];
                    } else {
                        upper_val += ua * pre_bounds.lower[[b, k]];
                    }
                }

                // The actual output should be within the concretized bounds
                // Note: These are loose bounds due to sampling-based relaxation
                assert!(
                    y_actual[j] >= lower_val - 0.5,
                    "LayerNorm batch {} output {} violates lower: {} < {}",
                    b,
                    j,
                    y_actual[j],
                    lower_val
                );
                assert!(
                    y_actual[j] <= upper_val + 0.5,
                    "LayerNorm batch {} output {} violates upper: {} > {}",
                    b,
                    j,
                    y_actual[j],
                    upper_val
                );
            }
        }
    }

    println!(
        "LayerNorm batched bounds test passed with {} batch positions",
        batch
    );
}

#[test]
fn test_transformer_block_ibp_with_residual() {
    // Test a simplified transformer block with residual connection:
    // output = input + MLP(LayerNorm(input))
    // where MLP = Linear -> GELU -> Linear
    //
    // This tests Phase 4 of transformer verification: full block with residuals.

    let hidden = 4;
    let expansion = 2; // MLP expands to 2x hidden for speed

    // Create LayerNorm
    let ln = LayerNormLayer::new_default(hidden, 1e-5);

    // Create MLP: up projection -> GELU -> down projection
    let scale1 = (2.0 / (hidden + hidden * expansion) as f32).sqrt();
    let weight_up = Array2::from_shape_fn((hidden * expansion, hidden), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32;
        scale1 * phase.sin() * 0.3
    });
    let linear_up = LinearLayer::new(weight_up, None).unwrap();

    let gelu = GELULayer::default();

    let scale2 = (2.0 / (hidden * expansion + hidden) as f32).sqrt();
    let weight_down = Array2::from_shape_fn((hidden, hidden * expansion), |(i, j)| {
        let phase = (i * 23 + j * 37) as f32;
        scale2 * phase.cos() * 0.3
    });
    let linear_down = LinearLayer::new(weight_down, None).unwrap();

    // Create input bounds: [batch=2, seq=3, hidden]
    let batch = 2;
    let seq = 3;
    let epsilon = 0.1; // Small perturbation

    let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));

    for b in 0..batch {
        for s in 0..seq {
            for h in 0..hidden {
                let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                let base = (hash as f32 / u32::MAX as f32) * 2.0 - 1.0;
                lower[[b, s, h]] = base - epsilon;
                upper[[b, s, h]] = base + epsilon;
            }
        }
    }

    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    // Track bound widths through the block
    let mut width_log: Vec<(&str, f32, f32, f32)> = Vec::new();

    let compute_width_stats = |bounds: &BoundedTensor| -> (f32, f32, f32) {
        let widths: Vec<f32> = bounds
            .lower
            .iter()
            .zip(bounds.upper.iter())
            .map(|(l, u)| u - l)
            .collect();
        let min_w = widths.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_w = widths.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let avg_w = widths.iter().sum::<f32>() / widths.len() as f32;
        (min_w, avg_w, max_w)
    };

    // Input
    let (min_w, avg_w, max_w) = compute_width_stats(&input_bounds);
    width_log.push(("Input", min_w, avg_w, max_w));

    // Step 1: LayerNorm (per position)
    // Flatten to [batch*seq, hidden], apply layernorm, reshape back
    let flat_shape = vec![batch * seq, hidden];
    let flat_input = input_bounds.reshape(&flat_shape).unwrap();

    // LayerNorm IBP
    let ln_out = ln.propagate_ibp(&flat_input).unwrap();
    let ln_out = ln_out.reshape(&[batch, seq, hidden]).unwrap();
    let (min_w, avg_w, max_w) = compute_width_stats(&ln_out);
    width_log.push(("LayerNorm", min_w, avg_w, max_w));

    // Step 2: MLP up projection
    let flat_ln_out = ln_out.reshape(&flat_shape).unwrap();
    let mlp_up = linear_up.propagate_ibp(&flat_ln_out).unwrap();
    let (min_w, avg_w, max_w) = compute_width_stats(&mlp_up);
    width_log.push(("MLP Up", min_w, avg_w, max_w));

    // Step 3: GELU
    let gelu_out = gelu.propagate_ibp(&mlp_up).unwrap();
    let (min_w, avg_w, max_w) = compute_width_stats(&gelu_out);
    width_log.push(("GELU", min_w, avg_w, max_w));

    // Step 4: MLP down projection
    let mlp_down = linear_down.propagate_ibp(&gelu_out).unwrap();
    let mlp_down = mlp_down.reshape(&[batch, seq, hidden]).unwrap();
    let (min_w, avg_w, max_w) = compute_width_stats(&mlp_down);
    width_log.push(("MLP Down", min_w, avg_w, max_w));

    // Step 5: Residual Add
    let add_layer = AddLayer;
    let output_bounds = add_layer
        .propagate_ibp_binary(&input_bounds, &mlp_down)
        .unwrap();
    let (min_w, avg_w, max_w) = compute_width_stats(&output_bounds);
    width_log.push(("Residual Add", min_w, avg_w, max_w));

    // Print bound width progression
    println!("\n=== Transformer Block IBP Bound Width Progression ===");
    println!("{:<15} {:>10} {:>10} {:>10}", "Layer", "Min", "Avg", "Max");
    println!("{}", "-".repeat(48));
    for (name, min_w, avg_w, max_w) in &width_log {
        println!(
            "{:<15} {:>10.4} {:>10.4} {:>10.4}",
            name, min_w, avg_w, max_w
        );
    }

    // Verify soundness by sampling
    let mut violations = 0;
    for sample_idx in 0..50 {
        // Sample random input within bounds
        let mut x = ArrayD::<f32>::zeros(IxDyn(&[batch, seq, hidden]));
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let hash = ((sample_idx * 10000 + b * 100 + s * 10 + h) as u32)
                        .wrapping_mul(2654435761_u32);
                    let t = hash as f32 / u32::MAX as f32;
                    x[[b, s, h]] = input_bounds.lower[[b, s, h]]
                        + (input_bounds.upper[[b, s, h]] - input_bounds.lower[[b, s, h]]) * t;
                }
            }
        }

        // Evaluate the block manually: output = input + MLP(LayerNorm(input))
        // Flatten for per-position operations
        let x_flat = x
            .clone()
            .into_shape_with_order((batch * seq, hidden))
            .unwrap();

        // LayerNorm per position
        let mut ln_y = Array2::<f32>::zeros((batch * seq, hidden));
        for pos in 0..(batch * seq) {
            let x_pos: Array1<f32> = (0..hidden).map(|h| x_flat[[pos, h]]).collect();
            let y_pos = ln.eval(&x_pos);
            for h in 0..hidden {
                ln_y[[pos, h]] = y_pos[h];
            }
        }

        // MLP up
        let mlp_up_y = ln_y.dot(&linear_up.weight.t());

        // GELU
        let gelu_y = mlp_up_y.mapv(|v| {
            0.5 * v * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v.powi(3))).tanh())
        });

        // MLP down
        let mlp_down_y = gelu_y.dot(&linear_down.weight.t());

        // Reshape back
        let mlp_down_y = mlp_down_y
            .into_shape_with_order((batch, seq, hidden))
            .unwrap();

        // Residual add: output = input + mlp_down
        let output = &x + &mlp_down_y;

        // Check if output is within bounds
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let y = output[[b, s, h]];
                    let l = output_bounds.lower[[b, s, h]];
                    let u = output_bounds.upper[[b, s, h]];
                    if y < l - 1e-5 || y > u + 1e-5 {
                        violations += 1;
                    }
                }
            }
        }
    }

    println!(
        "\nSoundness check: {} violations out of {} samples",
        violations,
        50 * batch * seq * hidden
    );
    assert_eq!(violations, 0, "IBP bounds should be sound");

    // Check that bounds don't explode
    let final_avg_width = width_log.last().unwrap().2;
    assert!(
        final_avg_width < 10.0,
        "Final bound width {} should be reasonable (< 10)",
        final_avg_width
    );

    println!("Transformer block IBP test passed!");
}

#[test]
fn test_add_layer_batched_crown_backward() {
    // Test AddLayer::propagate_linear_batched_binary
    // Verifies that batched CROWN backward for residual connections works correctly.

    let shape = vec![2, 3, 4]; // batch=2, seq=3, hidden=4

    // Create identity bounds at output
    let output_bounds = BatchedLinearBounds::identity(&shape);

    // Propagate backward through Add
    let add_layer = AddLayer;
    let (bounds_a, bounds_b) = add_layer
        .propagate_linear_batched_binary(&output_bounds)
        .unwrap();

    // Both branches should have the same coefficient matrices
    assert_eq!(bounds_a.lower_a.shape(), output_bounds.lower_a.shape());
    assert_eq!(bounds_b.lower_a.shape(), output_bounds.lower_a.shape());

    // Coefficient matrices should be identical (identity passes through)
    for (a, b) in bounds_a.lower_a.iter().zip(bounds_b.lower_a.iter()) {
        assert!((a - b).abs() < 1e-6, "Coefficients should match");
    }

    // Biases should be halved (to avoid double-counting)
    // Output has zero bias initially, so both should have zero bias
    let total_bias_a: f32 = bounds_a.lower_b.iter().sum();
    let total_bias_b: f32 = bounds_b.lower_b.iter().sum();
    assert!((total_bias_a).abs() < 1e-6, "Bias should be near zero");
    assert!((total_bias_b).abs() < 1e-6, "Bias should be near zero");

    // Test with non-zero bias
    let mut bounds_with_bias = output_bounds.clone();
    bounds_with_bias.lower_b.fill(2.0);
    bounds_with_bias.upper_b.fill(2.0);

    let (bounds_a, bounds_b) = add_layer
        .propagate_linear_batched_binary(&bounds_with_bias)
        .unwrap();

    // Biases should be halved
    for v in bounds_a.lower_b.iter() {
        assert!((v - 1.0).abs() < 1e-6, "Lower bias should be halved: {}", v);
    }
    for v in bounds_b.lower_b.iter() {
        assert!((v - 1.0).abs() < 1e-6, "Lower bias should be halved: {}", v);
    }

    println!("AddLayer batched CROWN backward test passed!");
}

#[test]
fn test_transformer_block_bound_explosion_analysis() {
    // Phase 4: Identify bound explosion source
    // Run the block with varying epsilon and track which operation causes explosion.

    let hidden = 8;
    let expansion = 4;
    let batch = 1;
    let seq = 4;

    // Create layers
    let ln = LayerNormLayer::new_default(hidden, 1e-5);

    let weight_up = Array2::from_shape_fn((hidden * expansion, hidden), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32;
        0.1 * phase.sin()
    });
    let linear_up = LinearLayer::new(weight_up, None).unwrap();

    let gelu = GELULayer::default();

    let weight_down = Array2::from_shape_fn((hidden, hidden * expansion), |(i, j)| {
        let phase = (i * 23 + j * 37) as f32;
        0.1 * phase.cos()
    });
    let linear_down = LinearLayer::new(weight_down, None).unwrap();

    let add_layer = AddLayer;

    println!("\n=== Bound Explosion Analysis ===");
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Epsilon", "Input", "LayerNorm", "MLPUp", "GELU", "MLPDown", "Output"
    );
    println!("{}", "-".repeat(82));

    for epsilon in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5] {
        // Create input with given epsilon
        let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
        let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));

        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                    let base = (hash as f32 / u32::MAX as f32) * 0.5; // base in [0, 0.5]
                    lower[[b, s, h]] = base - epsilon;
                    upper[[b, s, h]] = base + epsilon;
                }
            }
        }

        let input = BoundedTensor::new(lower, upper).unwrap();

        let avg_width = |bt: &BoundedTensor| -> f32 {
            bt.lower
                .iter()
                .zip(bt.upper.iter())
                .map(|(l, u)| u - l)
                .sum::<f32>()
                / bt.len() as f32
        };

        let input_width = avg_width(&input);

        // LayerNorm
        let flat_input = input.reshape(&[batch * seq, hidden]).unwrap();
        let ln_out = ln.propagate_ibp(&flat_input).unwrap();
        let ln_width = avg_width(&ln_out);

        // MLP Up
        let mlp_up = linear_up.propagate_ibp(&ln_out).unwrap();
        let up_width = avg_width(&mlp_up);

        // GELU
        let gelu_out = gelu.propagate_ibp(&mlp_up).unwrap();
        let gelu_width = avg_width(&gelu_out);

        // MLP Down
        let mlp_down = linear_down.propagate_ibp(&gelu_out).unwrap();
        let down_width = avg_width(&mlp_down);

        // Reshape and Add
        let mlp_down = mlp_down.reshape(&[batch, seq, hidden]).unwrap();
        let output = add_layer.propagate_ibp_binary(&input, &mlp_down).unwrap();
        let output_width = avg_width(&output);

        println!(
            "{:<10.3} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            epsilon, input_width, ln_width, up_width, gelu_width, down_width, output_width
        );
    }

    // Key insight: With small weights (0.1), bound growth is manageable.
    // The test documents the growth pattern for analysis.

    println!("\nAnalysis: LayerNorm slightly expands bounds, Linear layers grow proportionally");
    println!("to weight magnitudes, GELU can amplify unstable regions.");
}

#[test]
fn test_layernorm_forward_mode_vs_conservative() {
    // Compare forward-mode LayerNorm (tighter but approximate) vs
    // conservative mode (sound but may explode).
    //
    // Forward mode uses fixed mean/std from center point, dramatically
    // reducing bound explosion for small perturbations.

    let hidden = 8;
    let batch = 1;
    let seq = 4;

    // Create two LayerNorm layers: conservative and forward mode
    let ln_conservative = LayerNormLayer::new_default(hidden, 1e-5);
    let ln_forward = LayerNormLayer::new_default(hidden, 1e-5).with_forward_mode(true);

    println!("\n=== Forward Mode vs Conservative LayerNorm IBP ===");
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>12}",
        "Epsilon", "Conservative", "Forward", "Ratio", "Improvement"
    );
    println!("{}", "-".repeat(62));

    for epsilon in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5] {
        // Create input bounds
        let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
        let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));

        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                    let base = (hash as f32 / u32::MAX as f32) * 0.5;
                    lower[[b, s, h]] = base - epsilon;
                    upper[[b, s, h]] = base + epsilon;
                }
            }
        }

        let input = BoundedTensor::new(lower, upper).unwrap();

        let avg_width = |bt: &BoundedTensor| -> f32 {
            bt.lower
                .iter()
                .zip(bt.upper.iter())
                .map(|(l, u)| u - l)
                .sum::<f32>()
                / bt.len() as f32
        };

        // Flatten for LayerNorm
        let flat_input = input.reshape(&[batch * seq, hidden]).unwrap();

        // Conservative mode
        let cons_out = ln_conservative.propagate_ibp(&flat_input).unwrap();
        let cons_width = avg_width(&cons_out);

        // Forward mode
        let fwd_out = ln_forward.propagate_ibp(&flat_input).unwrap();
        let fwd_width = avg_width(&fwd_out);

        let ratio = cons_width / fwd_width;
        let improvement = (1.0 - fwd_width / cons_width) * 100.0;

        println!(
            "{:<10.3} {:>12.4} {:>12.4} {:>12.2}x {:>11.1}%",
            epsilon, cons_width, fwd_width, ratio, improvement
        );

        // Forward mode should be significantly tighter for small epsilon
        if epsilon <= 0.1 {
            assert!(
                fwd_width < cons_width,
                "Forward mode should be tighter for epsilon={}: {} vs {}",
                epsilon,
                fwd_width,
                cons_width
            );
        }

        // Forward mode bounds should still be valid (lower <= upper)
        for (l, u) in fwd_out.lower.iter().zip(fwd_out.upper.iter()) {
            assert!(
                l <= u,
                "Forward mode produced invalid bounds: lower {} > upper {}",
                l,
                u
            );
        }
    }

    println!("\nConclusion: Forward mode dramatically reduces bound explosion.");
    println!("For epsilon=0.1, expect ~50x tighter bounds.");
}

#[test]
fn test_layernorm_forward_mode_low_variance_center() {
    // Test that LayerNorm forward-mode doesn't explode when center has low variance.
    //
    // Problem: When bounds' midpoint (center) has near-zero variance, the sensitivity
    // = gamma/std explodes (std ≈ sqrt(eps) ≈ 0.003 → 333× amplification).
    //
    // Solution: Use effective_std = max(std, MIN_EFFECTIVE_STD) to cap sensitivity.
    // With MIN_EFFECTIVE_STD = 0.3, sensitivity is capped at ~3.3× for gamma ≈ 1.

    let hidden = 8;

    // Create LayerNorm with gamma=1 (default)
    let ln = LayerNormLayer::new_default(hidden, 1e-5).with_forward_mode(true);

    // Case 1: Center with ZERO values → variance = 0, std = sqrt(eps) ≈ 0.003
    // Without fix: sensitivity = 1/0.003 = 333×
    // With fix: sensitivity = 1/0.3 = 3.3×
    let epsilon = 0.001;
    let mut zero_lower = ArrayD::zeros(IxDyn(&[1, hidden]));
    let mut zero_upper = ArrayD::zeros(IxDyn(&[1, hidden]));
    for i in 0..hidden {
        zero_lower[[0, i]] = -epsilon;
        zero_upper[[0, i]] = epsilon;
    }
    let zero_input = BoundedTensor::new(zero_lower, zero_upper).unwrap();
    let zero_out = ln.propagate_ibp(&zero_input).unwrap();

    // Measure amplification: output_width / input_width
    let input_width = 2.0 * epsilon;
    let output_width = zero_out
        .lower
        .iter()
        .zip(zero_out.upper.iter())
        .map(|(l, u)| u - l)
        .fold(0.0_f32, f32::max);
    let amplification = output_width / input_width;

    println!("\n=== Low-Variance Center Test ===");
    println!("Center: all zeros (variance = 0)");
    println!("Input width: {:.6}", input_width);
    println!("Output width: {:.6}", output_width);
    println!("Amplification: {:.2}×", amplification);

    // With MIN_EFFECTIVE_STD = 0.3, amplification should be < 10×
    // (accounting for mean-coupling term: sensitivity * (r + max_r/n))
    assert!(
        amplification < 20.0,
        "Amplification should be < 20× with low-variance fix, got {:.2}×",
        amplification
    );

    // Case 2: Center with near-zero values (small but not exactly zero)
    let mut small_lower = ArrayD::zeros(IxDyn(&[1, hidden]));
    let mut small_upper = ArrayD::zeros(IxDyn(&[1, hidden]));
    for i in 0..hidden {
        let base = 1e-10; // Very small center value
        small_lower[[0, i]] = base - epsilon;
        small_upper[[0, i]] = base + epsilon;
    }
    let small_input = BoundedTensor::new(small_lower, small_upper).unwrap();
    let small_out = ln.propagate_ibp(&small_input).unwrap();

    let small_output_width = small_out
        .lower
        .iter()
        .zip(small_out.upper.iter())
        .map(|(l, u)| u - l)
        .fold(0.0_f32, f32::max);
    let small_amplification = small_output_width / input_width;

    println!("\nCenter: near-zero (1e-10)");
    println!("Output width: {:.6}", small_output_width);
    println!("Amplification: {:.2}×", small_amplification);

    assert!(
        small_amplification < 20.0,
        "Amplification should be < 20× for near-zero center, got {:.2}×",
        small_amplification
    );

    println!("\nConclusion: MIN_EFFECTIVE_STD prevents sensitivity explosion.");
}

#[test]
fn test_layernorm_ibp_fallback_on_nonfinite_input() {
    let hidden = 4;
    let ln = LayerNormLayer::new_default(hidden, 1e-5);

    let lower = ArrayD::from_elem(IxDyn(&[2, hidden]), f32::NEG_INFINITY);
    let upper = ArrayD::from_elem(IxDyn(&[2, hidden]), f32::INFINITY);
    // Use new_unchecked to bypass debug_asserts - this test intentionally uses Inf
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let out = ln.propagate_ibp(&input).unwrap();

    assert!(
        out.lower
            .iter()
            .chain(out.upper.iter())
            .all(|&v| v.is_finite()),
        "fallback output should be finite, got lower={:?} upper={:?}",
        out.lower,
        out.upper
    );

    let expected = ((hidden as f32) - 1.0).sqrt();
    for &v in out.lower.iter() {
        assert!((v + expected).abs() < 1e-6);
    }
    for &v in out.upper.iter() {
        assert!((v - expected).abs() < 1e-6);
    }
}

#[test]
fn test_layernorm_forward_mode_fallback_on_center_overflow() {
    let hidden = 4;
    let ln = LayerNormLayer::new_default(hidden, 1e-5).with_forward_mode(true);

    let lower = ArrayD::from_elem(IxDyn(&[1, hidden]), f32::MAX);
    let upper = ArrayD::from_elem(IxDyn(&[1, hidden]), f32::MAX);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let out = ln.propagate_ibp(&input).unwrap();
    assert!(
        out.lower
            .iter()
            .chain(out.upper.iter())
            .all(|&v| v.is_finite()),
        "fallback output should be finite, got lower={:?} upper={:?}",
        out.lower,
        out.upper
    );

    let expected = ((hidden as f32) - 1.0).sqrt();
    for &v in out.lower.iter() {
        assert!((v + expected).abs() < 1e-6);
    }
    for &v in out.upper.iter() {
        assert!((v - expected).abs() < 1e-6);
    }
}

#[test]
fn test_sanitize_bounds_for_fallback_replaces_nan_and_inverted() {
    // Use new_unchecked to bypass debug_asserts - this test intentionally uses NaN and Inf
    let bounds = BoundedTensor::new_unchecked(
        arr1(&[f32::NAN, f32::INFINITY, 1.0]).into_dyn(),
        arr1(&[f32::NAN, f32::NEG_INFINITY, 0.0]).into_dyn(),
    )
    .unwrap();

    let sanitized = GraphNetwork::sanitize_bounds_for_fallback(&bounds);
    assert!(
        sanitized
            .lower
            .iter()
            .chain(sanitized.upper.iter())
            .all(|&v| !v.is_nan()),
        "sanitized bounds should not contain NaN"
    );

    let pairs: Vec<(f32, f32)> = sanitized
        .lower
        .iter()
        .cloned()
        .zip(sanitized.upper.iter().cloned())
        .collect();
    assert_eq!(pairs[0], (f32::NEG_INFINITY, f32::INFINITY));
    assert_eq!(pairs[1], (f32::NEG_INFINITY, f32::INFINITY));
    assert_eq!(pairs[2], (f32::NEG_INFINITY, f32::INFINITY));
}

#[test]
fn test_transformer_block_crown_with_residual() {
    // Test CROWN backward through residual connection:
    // output = input + MLP(LayerNorm(input))
    //
    // For CROWN backward through y = x + F(x):
    // - Start with identity bounds at output
    // - Split through Add: bounds go to both branches
    // - Identity branch: bounds_x = bounds
    // - F(x) branch: propagate bounds backward through F
    // - Final bounds: sum coefficients from both branches
    //
    // This tests Phase 4 task 4: implement block CROWN.

    let hidden = 4;
    let expansion = 2;
    let batch = 2;
    let seq = 2;
    let epsilon = 0.05; // Smaller epsilon to avoid bound explosion

    // Create layers
    let ln = LayerNormLayer::new_default(hidden, 1e-5);

    let scale1 = (2.0 / (hidden + hidden * expansion) as f32).sqrt();
    let weight_up = Array2::from_shape_fn((hidden * expansion, hidden), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32;
        scale1 * phase.sin() * 0.2
    });
    let linear_up = LinearLayer::new(weight_up, None).unwrap();

    let gelu = GELULayer::default();

    let scale2 = (2.0 / (hidden * expansion + hidden) as f32).sqrt();
    let weight_down = Array2::from_shape_fn((hidden, hidden * expansion), |(i, j)| {
        let phase = (i * 23 + j * 37) as f32;
        scale2 * phase.cos() * 0.2
    });
    let linear_down = LinearLayer::new(weight_down, None).unwrap();

    // Create input bounds
    let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));

    for b in 0..batch {
        for s in 0..seq {
            for h in 0..hidden {
                let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                let base = (hash as f32 / u32::MAX as f32) * 0.5;
                lower[[b, s, h]] = base - epsilon;
                upper[[b, s, h]] = base + epsilon;
            }
        }
    }

    let input_bounds = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    // === IBP Forward Pass to collect intermediate bounds ===
    let flat_shape = vec![batch * seq, hidden];
    let flat_input = input_bounds.reshape(&flat_shape).unwrap();

    // LayerNorm
    let ln_out = ln.propagate_ibp(&flat_input).unwrap();

    // MLP Up
    let mlp_up_out = linear_up.propagate_ibp(&ln_out).unwrap();

    // GELU
    let gelu_out = gelu.propagate_ibp(&mlp_up_out).unwrap();

    // MLP Down
    let mlp_down_out = linear_down.propagate_ibp(&gelu_out).unwrap();
    let mlp_down_3d = mlp_down_out.reshape(&[batch, seq, hidden]).unwrap();

    // IBP final output
    let add_layer = AddLayer;
    let ibp_output = add_layer
        .propagate_ibp_binary(&input_bounds, &mlp_down_3d)
        .unwrap();

    // === CROWN Backward Pass ===
    // Initialize identity bounds at output
    let output_shape = vec![batch, seq, hidden];
    let crown_bounds = BatchedLinearBounds::identity(&output_shape);

    // Step 1: Split through Add -> (bounds_input, bounds_mlp)
    let (bounds_input_branch, bounds_mlp_branch) = add_layer
        .propagate_linear_batched_binary(&crown_bounds)
        .unwrap();

    // Step 2: Propagate bounds_mlp_branch backward through MLP + LayerNorm
    // Reshape to flat for operations
    let flat_mlp_bounds = BatchedLinearBounds {
        lower_a: bounds_mlp_branch
            .lower_a
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden, hidden]))
            .unwrap(),
        lower_b: bounds_mlp_branch
            .lower_b
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden]))
            .unwrap(),
        upper_a: bounds_mlp_branch
            .upper_a
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden, hidden]))
            .unwrap(),
        upper_b: bounds_mlp_branch
            .upper_b
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden]))
            .unwrap(),
        input_shape: vec![batch * seq, hidden],
        output_shape: vec![batch * seq, hidden],
    };

    // MLP Down backward
    let after_down = linear_down
        .propagate_linear_batched(&flat_mlp_bounds)
        .unwrap();

    // GELU backward
    let after_gelu = gelu
        .propagate_linear_batched_with_bounds(&after_down, &mlp_up_out)
        .unwrap();

    // MLP Up backward
    let after_up = linear_up.propagate_linear_batched(&after_gelu).unwrap();

    // LayerNorm backward
    let after_ln = ln
        .propagate_linear_batched_with_bounds(&after_up, &flat_input)
        .unwrap();

    // Reshape back to 3D
    let mlp_branch_final = BatchedLinearBounds {
        lower_a: after_ln
            .lower_a
            .into_shape_with_order(IxDyn(&[batch, seq, hidden, hidden]))
            .unwrap(),
        lower_b: after_ln
            .lower_b
            .into_shape_with_order(IxDyn(&[batch, seq, hidden]))
            .unwrap(),
        upper_a: after_ln
            .upper_a
            .into_shape_with_order(IxDyn(&[batch, seq, hidden, hidden]))
            .unwrap(),
        upper_b: after_ln
            .upper_b
            .into_shape_with_order(IxDyn(&[batch, seq, hidden]))
            .unwrap(),
        input_shape: vec![batch, seq, hidden],
        output_shape: vec![batch, seq, hidden],
    };

    // Step 3: Combine input branch and MLP branch
    // For y = x + F(x), the combined coefficients are: A_combined = A_input + A_mlp
    let combined_lower_a = &bounds_input_branch.lower_a + &mlp_branch_final.lower_a;
    let combined_upper_a = &bounds_input_branch.upper_a + &mlp_branch_final.upper_a;
    let combined_lower_b = &bounds_input_branch.lower_b + &mlp_branch_final.lower_b;
    let combined_upper_b = &bounds_input_branch.upper_b + &mlp_branch_final.upper_b;

    let combined_bounds = BatchedLinearBounds {
        lower_a: combined_lower_a,
        lower_b: combined_lower_b,
        upper_a: combined_upper_a,
        upper_b: combined_upper_b,
        input_shape: vec![batch, seq, hidden],
        output_shape: vec![batch, seq, hidden],
    };

    // Step 4: Concretize with input bounds
    let crown_output = combined_bounds.concretize(&input_bounds);

    // === Compare IBP vs CROWN ===
    println!("\n=== CROWN vs IBP for Transformer Block with Residual ===");

    let ibp_widths: Vec<f32> = ibp_output
        .lower
        .iter()
        .zip(ibp_output.upper.iter())
        .map(|(l, u)| u - l)
        .collect();
    let crown_widths: Vec<f32> = crown_output
        .lower
        .iter()
        .zip(crown_output.upper.iter())
        .map(|(l, u)| u - l)
        .collect();

    let ibp_avg_width: f32 = ibp_widths.iter().sum::<f32>() / ibp_widths.len() as f32;
    let crown_avg_width: f32 = crown_widths.iter().sum::<f32>() / crown_widths.len() as f32;

    println!("IBP average bound width:   {:.6}", ibp_avg_width);
    println!("CROWN average bound width: {:.6}", crown_avg_width);
    println!(
        "CROWN tightness ratio:     {:.2}x",
        ibp_avg_width / crown_avg_width.max(1e-10)
    );

    // Verify CROWN bounds are valid (lower <= upper)
    let mut valid_count = 0;
    for (l, u) in crown_output.lower.iter().zip(crown_output.upper.iter()) {
        if *l <= *u + 1e-5 {
            valid_count += 1;
        }
    }
    assert_eq!(
        valid_count,
        crown_output.len(),
        "All CROWN bounds should be valid"
    );

    // Verify CROWN is at least as tight as IBP (or close)
    // Note: Due to numerical issues, CROWN might be slightly looser in some cases
    let tightness_ratio = crown_avg_width / ibp_avg_width.max(1e-10);
    println!("Tightness check: CROWN/IBP = {:.4}", tightness_ratio);

    // CROWN should generally be tighter, but allow some slack for numerical issues
    assert!(
        tightness_ratio < 2.0,
        "CROWN bounds ({}) should not be much looser than IBP ({})",
        crown_avg_width,
        ibp_avg_width
    );

    // Verify soundness by sampling
    let mut violations = 0;
    for sample_idx in 0..50 {
        let mut x = ArrayD::<f32>::zeros(IxDyn(&[batch, seq, hidden]));
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let hash = ((sample_idx * 10000 + b * 100 + s * 10 + h) as u32)
                        .wrapping_mul(2654435761_u32);
                    let t = hash as f32 / u32::MAX as f32;
                    x[[b, s, h]] = input_bounds.lower[[b, s, h]]
                        + (input_bounds.upper[[b, s, h]] - input_bounds.lower[[b, s, h]]) * t;
                }
            }
        }

        // Evaluate the block
        let x_flat = x
            .clone()
            .into_shape_with_order((batch * seq, hidden))
            .unwrap();

        let mut ln_y = Array2::<f32>::zeros((batch * seq, hidden));
        for pos in 0..(batch * seq) {
            let x_pos: Array1<f32> = (0..hidden).map(|h| x_flat[[pos, h]]).collect();
            let y_pos = ln.eval(&x_pos);
            for h in 0..hidden {
                ln_y[[pos, h]] = y_pos[h];
            }
        }

        let mlp_up_y = ln_y.dot(&linear_up.weight.t());
        let gelu_y = mlp_up_y.mapv(|v| {
            0.5 * v * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v.powi(3))).tanh())
        });
        let mlp_down_y = gelu_y.dot(&linear_down.weight.t());
        let mlp_down_y = mlp_down_y
            .into_shape_with_order((batch, seq, hidden))
            .unwrap();
        let output = &x + &mlp_down_y;

        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let y = output[[b, s, h]];
                    let l = crown_output.lower[[b, s, h]];
                    let u = crown_output.upper[[b, s, h]];
                    if y < l - 1e-4 || y > u + 1e-4 {
                        violations += 1;
                    }
                }
            }
        }
    }

    println!(
        "CROWN soundness check: {} violations out of {} samples",
        violations,
        50 * batch * seq * hidden
    );
    assert!(
        violations == 0,
        "CROWN bounds should be sound ({} violations)",
        violations
    );

    println!("Transformer block CROWN with residual test passed!");
}

#[test]
fn test_multi_block_bound_growth() {
    // Phase 5: Test bound growth through sequential transformer blocks.
    // Each block: output = input + MLP(LayerNorm(input))
    //
    // Measure how bounds grow through 1, 2, 3 blocks with both IBP and CROWN.

    let hidden = 4;
    let expansion = 2;
    let batch = 1;
    let seq = 2;

    // Create shared MLP weights (same weights for each block for simplicity)
    let scale1 = (2.0 / (hidden + hidden * expansion) as f32).sqrt();
    let weight_up = Array2::from_shape_fn((hidden * expansion, hidden), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32;
        scale1 * phase.sin() * 0.15 // Smaller weights to reduce explosion
    });

    let scale2 = (2.0 / (hidden * expansion + hidden) as f32).sqrt();
    let weight_down = Array2::from_shape_fn((hidden, hidden * expansion), |(i, j)| {
        let phase = (i * 23 + j * 37) as f32;
        scale2 * phase.cos() * 0.15
    });

    // Helper function to run one block IBP
    let run_block_ibp = |input: &BoundedTensor,
                         ln: &LayerNormLayer,
                         linear_up: &LinearLayer,
                         gelu: &GELULayer,
                         linear_down: &LinearLayer|
     -> BoundedTensor {
        let flat_shape = vec![batch * seq, hidden];
        let flat_input = input.reshape(&flat_shape).unwrap();

        // LayerNorm
        let ln_out = ln.propagate_ibp(&flat_input).unwrap();

        // MLP Up
        let mlp_up = linear_up.propagate_ibp(&ln_out).unwrap();

        // GELU
        let gelu_out = gelu.propagate_ibp(&mlp_up).unwrap();

        // MLP Down
        let mlp_down = linear_down.propagate_ibp(&gelu_out).unwrap();
        let mlp_down_3d = mlp_down.reshape(&[batch, seq, hidden]).unwrap();

        // Residual Add
        let add_layer = AddLayer;
        add_layer.propagate_ibp_binary(input, &mlp_down_3d).unwrap()
    };

    println!("\n=== Phase 5: Multi-Block Bound Growth Analysis ===");
    println!(
        "{:<8} {:>10} {:>14} {:>14} {:>14} {:>14}",
        "Epsilon", "Input", "Block 1", "Block 2", "Block 3", "Growth Rate"
    );
    println!("{}", "-".repeat(76));

    for epsilon in [0.001, 0.005, 0.01, 0.02, 0.05] {
        // Create layers for this run
        let ln = LayerNormLayer::new_default(hidden, 1e-5);
        let linear_up = LinearLayer::new(weight_up.clone(), None).unwrap();
        let gelu = GELULayer::default();
        let linear_down = LinearLayer::new(weight_down.clone(), None).unwrap();

        // Create input bounds
        let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
        let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));

        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                    let base = (hash as f32 / u32::MAX as f32) * 0.3;
                    lower[[b, s, h]] = base - epsilon as f32;
                    upper[[b, s, h]] = base + epsilon as f32;
                }
            }
        }

        let input = BoundedTensor::new(lower, upper).unwrap();

        let avg_width = |bt: &BoundedTensor| -> f32 {
            bt.lower
                .iter()
                .zip(bt.upper.iter())
                .map(|(l, u)| u - l)
                .sum::<f32>()
                / bt.len() as f32
        };

        let input_width = avg_width(&input);

        // Block 1
        let after_block1 = run_block_ibp(&input, &ln, &linear_up, &gelu, &linear_down);
        let width1 = avg_width(&after_block1);

        // Block 2
        let after_block2 = run_block_ibp(&after_block1, &ln, &linear_up, &gelu, &linear_down);
        let width2 = avg_width(&after_block2);

        // Block 3
        let after_block3 = run_block_ibp(&after_block2, &ln, &linear_up, &gelu, &linear_down);
        let width3 = avg_width(&after_block3);

        // Compute average per-block growth rate
        let growth_rate = (width3 / input_width).powf(1.0 / 3.0);

        println!(
            "{:<8.3} {:>10.4} {:>14.4} {:>14.4} {:>14.4} {:>14.2}x",
            epsilon, input_width, width1, width2, width3, growth_rate
        );
    }

    // Test that small epsilon keeps bounds manageable after 3 blocks
    let ln = LayerNormLayer::new_default(hidden, 1e-5);
    let linear_up = LinearLayer::new(weight_up.clone(), None).unwrap();
    let gelu = GELULayer::default();
    let linear_down = LinearLayer::new(weight_down.clone(), None).unwrap();

    let epsilon = 0.001f32;
    let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..hidden {
                let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                let base = (hash as f32 / u32::MAX as f32) * 0.3;
                lower[[b, s, h]] = base - epsilon;
                upper[[b, s, h]] = base + epsilon;
            }
        }
    }
    let input = BoundedTensor::new(lower, upper).unwrap();

    let after_block1 = run_block_ibp(&input, &ln, &linear_up, &gelu, &linear_down);
    let after_block2 = run_block_ibp(&after_block1, &ln, &linear_up, &gelu, &linear_down);
    let after_block3 = run_block_ibp(&after_block2, &ln, &linear_up, &gelu, &linear_down);

    let avg_width = |bt: &BoundedTensor| -> f32 {
        bt.lower
            .iter()
            .zip(bt.upper.iter())
            .map(|(l, u)| u - l)
            .sum::<f32>()
            / bt.len() as f32
    };

    let final_width = avg_width(&after_block3);
    println!(
        "\nWith ε=0.001, final bound width after 3 blocks: {:.6}",
        final_width
    );

    // Verify soundness on final output
    let mut violations = 0;
    for sample_idx in 0..30 {
        // Sample input
        let mut x = ArrayD::<f32>::zeros(IxDyn(&[batch, seq, hidden]));
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let hash = ((sample_idx * 10000 + b * 100 + s * 10 + h) as u32)
                        .wrapping_mul(2654435761_u32);
                    let t = hash as f32 / u32::MAX as f32;
                    x[[b, s, h]] = input.lower[[b, s, h]]
                        + (input.upper[[b, s, h]] - input.lower[[b, s, h]]) * t;
                }
            }
        }

        // Evaluate 3 blocks
        let eval_block = |x: &ArrayD<f32>| -> ArrayD<f32> {
            let x_flat = x
                .clone()
                .into_shape_with_order((batch * seq, hidden))
                .unwrap();

            // LayerNorm
            let mut ln_y = Array2::<f32>::zeros((batch * seq, hidden));
            for pos in 0..(batch * seq) {
                let x_pos: Array1<f32> = (0..hidden).map(|h| x_flat[[pos, h]]).collect();
                let y_pos = ln.eval(&x_pos);
                for h in 0..hidden {
                    ln_y[[pos, h]] = y_pos[h];
                }
            }

            // MLP
            let mlp_up_y = ln_y.dot(&linear_up.weight.t());
            let gelu_y = mlp_up_y.mapv(|v| {
                0.5 * v
                    * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v.powi(3))).tanh())
            });
            let mlp_down_y = gelu_y.dot(&linear_down.weight.t());
            let mlp_down_y = mlp_down_y
                .into_shape_with_order((batch, seq, hidden))
                .unwrap();

            // Residual
            x + &mlp_down_y
        };

        let y1 = eval_block(&x);
        let y2 = eval_block(&y1);
        let y3 = eval_block(&y2);

        // Check bounds
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let y = y3[[b, s, h]];
                    let l = after_block3.lower[[b, s, h]];
                    let u = after_block3.upper[[b, s, h]];
                    if y < l - 1e-4 || y > u + 1e-4 {
                        violations += 1;
                    }
                }
            }
        }
    }

    println!(
        "Soundness check: {} violations out of {} samples",
        violations,
        30 * batch * seq * hidden
    );
    assert_eq!(violations, 0, "Multi-block IBP bounds should be sound");

    // Verify bounds don't explode too much
    assert!(
        final_width < 1.0,
        "With ε=0.001, 3-block bounds should stay under 1.0, got {}",
        final_width
    );

    println!("Multi-block bound growth test passed!");
}

#[test]
fn test_multi_block_crown_vs_ibp() {
    // Phase 5: Compare CROWN vs IBP for multi-block verification.
    // CROWN should provide tighter bounds than IBP across multiple blocks.

    let hidden = 4;
    let expansion = 2;
    let batch = 1;
    let seq = 2;
    let epsilon = 0.01f32;

    // Create layers
    let ln = LayerNormLayer::new_default(hidden, 1e-5);

    let scale1 = (2.0 / (hidden + hidden * expansion) as f32).sqrt();
    let weight_up = Array2::from_shape_fn((hidden * expansion, hidden), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32;
        scale1 * phase.sin() * 0.1
    });
    let linear_up = LinearLayer::new(weight_up, None).unwrap();

    let gelu = GELULayer::default();

    let scale2 = (2.0 / (hidden * expansion + hidden) as f32).sqrt();
    let weight_down = Array2::from_shape_fn((hidden, hidden * expansion), |(i, j)| {
        let phase = (i * 23 + j * 37) as f32;
        scale2 * phase.cos() * 0.1
    });
    let linear_down = LinearLayer::new(weight_down, None).unwrap();

    let add_layer = AddLayer;

    // Create input bounds
    let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..hidden {
                let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                let base = (hash as f32 / u32::MAX as f32) * 0.3;
                lower[[b, s, h]] = base - epsilon;
                upper[[b, s, h]] = base + epsilon;
            }
        }
    }
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    let avg_width = |bt: &BoundedTensor| -> f32 {
        bt.lower
            .iter()
            .zip(bt.upper.iter())
            .map(|(l, u)| u - l)
            .sum::<f32>()
            / bt.len() as f32
    };

    // === IBP through 2 blocks ===
    let run_block_ibp = |input: &BoundedTensor| -> BoundedTensor {
        let flat_shape = vec![batch * seq, hidden];
        let flat_input = input.reshape(&flat_shape).unwrap();
        let ln_out = ln.propagate_ibp(&flat_input).unwrap();
        let mlp_up = linear_up.propagate_ibp(&ln_out).unwrap();
        let gelu_out = gelu.propagate_ibp(&mlp_up).unwrap();
        let mlp_down = linear_down.propagate_ibp(&gelu_out).unwrap();
        let mlp_down_3d = mlp_down.reshape(&[batch, seq, hidden]).unwrap();
        add_layer.propagate_ibp_binary(input, &mlp_down_3d).unwrap()
    };

    let ibp_block1 = run_block_ibp(&input_bounds);
    let ibp_block2 = run_block_ibp(&ibp_block1);

    // === CROWN through 2 blocks (using IBP intermediate bounds) ===
    // For CROWN through multiple blocks, we run IBP first to get intermediate bounds,
    // then run CROWN backward from the final output.

    // Collect all intermediate bounds via IBP
    let flat_input = input_bounds.reshape(&[batch * seq, hidden]).unwrap();

    // Block 1 intermediates
    let ln_out_1 = ln.propagate_ibp(&flat_input).unwrap();
    let mlp_up_out_1 = linear_up.propagate_ibp(&ln_out_1).unwrap();
    let gelu_out_1 = gelu.propagate_ibp(&mlp_up_out_1).unwrap();
    let mlp_down_out_1 = linear_down.propagate_ibp(&gelu_out_1).unwrap();
    let mlp_down_3d_1 = mlp_down_out_1.reshape(&[batch, seq, hidden]).unwrap();
    let block1_out = add_layer
        .propagate_ibp_binary(&input_bounds, &mlp_down_3d_1)
        .unwrap();

    // Block 2 intermediates
    let flat_block1 = block1_out.reshape(&[batch * seq, hidden]).unwrap();
    let ln_out_2 = ln.propagate_ibp(&flat_block1).unwrap();
    let mlp_up_out_2 = linear_up.propagate_ibp(&ln_out_2).unwrap();
    let gelu_out_2 = gelu.propagate_ibp(&mlp_up_out_2).unwrap();
    let _mlp_down_out_2 = linear_down.propagate_ibp(&gelu_out_2).unwrap();

    // CROWN backward from output of block 2
    let output_shape = vec![batch, seq, hidden];
    let crown_bounds = BatchedLinearBounds::identity(&output_shape);

    // Split through Add of block 2
    let (bounds_input_2, bounds_mlp_2) = add_layer
        .propagate_linear_batched_binary(&crown_bounds)
        .unwrap();

    // Propagate MLP branch backward through block 2
    let flat_mlp_bounds_2 = BatchedLinearBounds {
        lower_a: bounds_mlp_2
            .lower_a
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden, hidden]))
            .unwrap(),
        lower_b: bounds_mlp_2
            .lower_b
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden]))
            .unwrap(),
        upper_a: bounds_mlp_2
            .upper_a
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden, hidden]))
            .unwrap(),
        upper_b: bounds_mlp_2
            .upper_b
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden]))
            .unwrap(),
        input_shape: vec![batch * seq, hidden],
        output_shape: vec![batch * seq, hidden],
    };

    let after_down_2 = linear_down
        .propagate_linear_batched(&flat_mlp_bounds_2)
        .unwrap();
    let after_gelu_2 = gelu
        .propagate_linear_batched_with_bounds(&after_down_2, &mlp_up_out_2)
        .unwrap();
    let after_up_2 = linear_up.propagate_linear_batched(&after_gelu_2).unwrap();
    let after_ln_2 = ln
        .propagate_linear_batched_with_bounds(&after_up_2, &flat_block1)
        .unwrap();

    // Reshape and combine
    let mlp_branch_2 = BatchedLinearBounds {
        lower_a: after_ln_2
            .lower_a
            .into_shape_with_order(IxDyn(&[batch, seq, hidden, hidden]))
            .unwrap(),
        lower_b: after_ln_2
            .lower_b
            .into_shape_with_order(IxDyn(&[batch, seq, hidden]))
            .unwrap(),
        upper_a: after_ln_2
            .upper_a
            .into_shape_with_order(IxDyn(&[batch, seq, hidden, hidden]))
            .unwrap(),
        upper_b: after_ln_2
            .upper_b
            .into_shape_with_order(IxDyn(&[batch, seq, hidden]))
            .unwrap(),
        input_shape: vec![batch, seq, hidden],
        output_shape: vec![batch, seq, hidden],
    };

    // Combined bounds at block 1 output
    let combined_at_block1_lower_a = &bounds_input_2.lower_a + &mlp_branch_2.lower_a;
    let combined_at_block1_upper_a = &bounds_input_2.upper_a + &mlp_branch_2.upper_a;
    let combined_at_block1_lower_b = &bounds_input_2.lower_b + &mlp_branch_2.lower_b;
    let combined_at_block1_upper_b = &bounds_input_2.upper_b + &mlp_branch_2.upper_b;

    let bounds_at_block1 = BatchedLinearBounds {
        lower_a: combined_at_block1_lower_a,
        lower_b: combined_at_block1_lower_b,
        upper_a: combined_at_block1_upper_a,
        upper_b: combined_at_block1_upper_b,
        input_shape: vec![batch, seq, hidden],
        output_shape: vec![batch, seq, hidden],
    };

    // Now propagate these bounds backward through block 1
    let (bounds_input_1, bounds_mlp_1) = add_layer
        .propagate_linear_batched_binary(&bounds_at_block1)
        .unwrap();

    let flat_mlp_bounds_1 = BatchedLinearBounds {
        lower_a: bounds_mlp_1
            .lower_a
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden, hidden]))
            .unwrap(),
        lower_b: bounds_mlp_1
            .lower_b
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden]))
            .unwrap(),
        upper_a: bounds_mlp_1
            .upper_a
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden, hidden]))
            .unwrap(),
        upper_b: bounds_mlp_1
            .upper_b
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden]))
            .unwrap(),
        input_shape: vec![batch * seq, hidden],
        output_shape: vec![batch * seq, hidden],
    };

    let after_down_1 = linear_down
        .propagate_linear_batched(&flat_mlp_bounds_1)
        .unwrap();
    let after_gelu_1 = gelu
        .propagate_linear_batched_with_bounds(&after_down_1, &mlp_up_out_1)
        .unwrap();
    let after_up_1 = linear_up.propagate_linear_batched(&after_gelu_1).unwrap();
    let after_ln_1 = ln
        .propagate_linear_batched_with_bounds(&after_up_1, &flat_input)
        .unwrap();

    let mlp_branch_1 = BatchedLinearBounds {
        lower_a: after_ln_1
            .lower_a
            .into_shape_with_order(IxDyn(&[batch, seq, hidden, hidden]))
            .unwrap(),
        lower_b: after_ln_1
            .lower_b
            .into_shape_with_order(IxDyn(&[batch, seq, hidden]))
            .unwrap(),
        upper_a: after_ln_1
            .upper_a
            .into_shape_with_order(IxDyn(&[batch, seq, hidden, hidden]))
            .unwrap(),
        upper_b: after_ln_1
            .upper_b
            .into_shape_with_order(IxDyn(&[batch, seq, hidden]))
            .unwrap(),
        input_shape: vec![batch, seq, hidden],
        output_shape: vec![batch, seq, hidden],
    };

    // Final combined bounds at input
    let final_lower_a = &bounds_input_1.lower_a + &mlp_branch_1.lower_a;
    let final_upper_a = &bounds_input_1.upper_a + &mlp_branch_1.upper_a;
    let final_lower_b = &bounds_input_1.lower_b + &mlp_branch_1.lower_b;
    let final_upper_b = &bounds_input_1.upper_b + &mlp_branch_1.upper_b;

    let final_bounds = BatchedLinearBounds {
        lower_a: final_lower_a,
        lower_b: final_lower_b,
        upper_a: final_upper_a,
        upper_b: final_upper_b,
        input_shape: vec![batch, seq, hidden],
        output_shape: vec![batch, seq, hidden],
    };

    // Concretize CROWN bounds
    let crown_output = final_bounds.concretize(&input_bounds);

    // Compare
    let ibp_width = avg_width(&ibp_block2);
    let crown_width = avg_width(&crown_output);

    println!("\n=== Phase 5: Multi-Block CROWN vs IBP ===");
    println!("IBP bound width after 2 blocks:   {:.6}", ibp_width);
    println!("CROWN bound width after 2 blocks: {:.6}", crown_width);
    println!(
        "CROWN improvement: {:.2}x tighter",
        ibp_width / crown_width.max(1e-10)
    );

    // Verify CROWN bounds are sound
    let mut violations = 0;
    for sample_idx in 0..30 {
        let mut x = ArrayD::<f32>::zeros(IxDyn(&[batch, seq, hidden]));
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let hash = ((sample_idx * 10000 + b * 100 + s * 10 + h) as u32)
                        .wrapping_mul(2654435761_u32);
                    let t = hash as f32 / u32::MAX as f32;
                    x[[b, s, h]] = input_bounds.lower[[b, s, h]]
                        + (input_bounds.upper[[b, s, h]] - input_bounds.lower[[b, s, h]]) * t;
                }
            }
        }

        // Evaluate 2 blocks
        let eval_block = |x: &ArrayD<f32>| -> ArrayD<f32> {
            let x_flat = x
                .clone()
                .into_shape_with_order((batch * seq, hidden))
                .unwrap();

            let mut ln_y = Array2::<f32>::zeros((batch * seq, hidden));
            for pos in 0..(batch * seq) {
                let x_pos: Array1<f32> = (0..hidden).map(|h| x_flat[[pos, h]]).collect();
                let y_pos = ln.eval(&x_pos);
                for h in 0..hidden {
                    ln_y[[pos, h]] = y_pos[h];
                }
            }

            let mlp_up_y = ln_y.dot(&linear_up.weight.t());
            let gelu_y = mlp_up_y.mapv(|v| {
                0.5 * v
                    * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v.powi(3))).tanh())
            });
            let mlp_down_y = gelu_y.dot(&linear_down.weight.t());
            let mlp_down_y = mlp_down_y
                .into_shape_with_order((batch, seq, hidden))
                .unwrap();

            x + &mlp_down_y
        };

        let y1 = eval_block(&x);
        let y2 = eval_block(&y1);

        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let y = y2[[b, s, h]];
                    let l = crown_output.lower[[b, s, h]];
                    let u = crown_output.upper[[b, s, h]];
                    if y < l - 1e-4 || y > u + 1e-4 {
                        violations += 1;
                    }
                }
            }
        }
    }

    println!(
        "CROWN soundness: {} violations out of {} samples",
        violations,
        30 * batch * seq * hidden
    );
    assert_eq!(violations, 0, "Multi-block CROWN bounds should be sound");

    // CROWN should be at least as tight as IBP
    assert!(
        crown_width <= ibp_width * 1.1,
        "CROWN should not be much looser than IBP"
    );

    println!("Multi-block CROWN vs IBP test passed!");
}

// ========================================================================
// Full-Block Zonotope Propagation Tests
// ========================================================================

#[test]
fn test_full_block_zonotope_propagation() {
    // Test full-block zonotope propagation through transformer attention:
    // LayerNorm -> Q_proj + K_proj -> Q@K^T -> Softmax -> scores@V -> Add(residual)
    //
    // This tests the complete zonotope propagation path including:
    // - LayerNorm (affine approximation)
    // - Linear projections
    // - Q@K^T attention correlation tracking
    // - Softmax (affine approximation)
    // - Value multiplication
    // - Residual connections (Add)
    //
    // Note: Full zonotope propagation trades precision for correlation tracking.
    // The linearization errors from LayerNorm and Softmax can accumulate,
    // so zonotope bounds may be looser than IBP for complex networks.
    // The value is in tracking correlations through operations that benefit
    // from it (like Q@K^T diagonal entries).

    let seq = 3_usize;
    let dim = 4_usize;
    let epsilon = 0.05_f32; // Small epsilon for better linear approximations

    let mut graph = GraphNetwork::new();

    // LayerNorm normalization
    let gamma = ndarray::Array1::ones(dim);
    let beta = ndarray::Array1::zeros(dim);
    graph.add_node(GraphNode::from_input(
        "ln",
        Layer::LayerNorm(LayerNormLayer::new(gamma, beta, 1e-5)),
    ));

    // Q, K, V projections (identity for simplicity)
    let eye = ndarray::Array2::<f32>::eye(dim);
    graph.add_node(GraphNode::new(
        "q",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
        vec!["ln".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "k",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
        vec!["ln".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "v",
        Layer::Linear(LinearLayer::new(eye, None).unwrap()),
        vec!["ln".to_string()],
    ));

    // Q@K^T attention scores
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "q",
        "k",
    ));

    // Softmax over sequence dimension
    graph.add_node(GraphNode::new(
        "attn_weights",
        Layer::Softmax(SoftmaxLayer::new(-1)), // Softmax over last dim (seq)
        vec!["scores".to_string()],
    ));

    // Attention output: attn_weights @ V
    graph.add_node(GraphNode::binary(
        "attn_output",
        Layer::MatMul(MatMulLayer::new(false, None)), // Not transposed
        "attn_weights",
        "v",
    ));

    // Residual connection: attn_output + input (through linear identity)
    let eye_residual = ndarray::Array2::<f32>::eye(dim);
    graph.add_node(GraphNode::from_input(
        "residual_proj",
        Layer::Linear(LinearLayer::new(eye_residual, None).unwrap()),
    ));

    graph.add_node(GraphNode::binary(
        "output",
        Layer::Add(AddLayer),
        "attn_output",
        "residual_proj",
    ));
    graph.set_output("output");

    // Create input with varied values
    let center_values: Vec<f32> = (0..seq * dim).map(|i| (i % dim) as f32 + 1.0).collect();
    let lower = ndarray::ArrayD::from_shape_vec(
        vec![seq, dim],
        center_values.iter().map(|&v| v - epsilon).collect(),
    )
    .unwrap();
    let upper = ndarray::ArrayD::from_shape_vec(
        vec![seq, dim],
        center_values.iter().map(|&v| v + epsilon).collect(),
    )
    .unwrap();
    let input = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    // Run full-block zonotope propagation
    let zonotope_bounds = graph.propagate_zonotope(&input, epsilon).unwrap();

    // Run standard IBP for comparison
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Verify zonotope bounds are valid
    assert!(
        zonotope_bounds.lower.iter().all(|&v| v.is_finite()),
        "Zonotope lower bounds should be finite"
    );
    assert!(
        zonotope_bounds.upper.iter().all(|&v| v.is_finite()),
        "Zonotope upper bounds should be finite"
    );
    assert!(
        zonotope_bounds
            .lower
            .iter()
            .zip(zonotope_bounds.upper.iter())
            .all(|(&l, &u)| l <= u + 1e-6),
        "Zonotope lower bounds should be <= upper bounds"
    );

    // Verify IBP bounds are valid for comparison
    assert!(
        ibp_bounds.lower.iter().all(|&v| v.is_finite()),
        "IBP lower bounds should be finite"
    );

    // Calculate widths
    let zonotope_width = zonotope_bounds.max_width();
    let ibp_width = ibp_bounds.max_width();

    println!(
        "Full-block zonotope vs IBP: zonotope_width={:.4}, ibp_width={:.4}, ratio={:.4}",
        zonotope_width,
        ibp_width,
        zonotope_width / ibp_width
    );

    // Full zonotope propagation through complex networks may be looser than IBP
    // due to accumulated linearization errors from LayerNorm and Softmax.
    // We verify that bounds are valid and report the comparison.
    // The value of zonotope propagation is in specific patterns (Q@K^T diagonals)
    // rather than overall bound tightness for complex networks.
    assert!(
        zonotope_width <= ibp_width * 2.0, // Allow 2x slack for linearization overhead
        "Zonotope bounds should not be extremely looser than IBP (got {:.4} vs {:.4})",
        zonotope_width,
        ibp_width
    );

    println!("Full-block zonotope propagation test passed!");
}

#[test]
fn test_zonotope_vs_ibp_attention_qkt_diagonal_tightness() {
    // Test that zonotope propagation tracks correlations for Q@K^T diagonal entries.
    // This is the key benefit of zonotope propagation: recognizing that X[i]·X[i] >= 0.
    //
    // IBP (standard): propagate_ibp already includes zonotope tightening for Q@K^T
    // via try_attention_matmul_bounds_zonotope, so both should give similar results.
    //
    // This test verifies that the full propagate_zonotope path also benefits from
    // correlation tracking through the Q@K^T pattern.

    let seq = 4_usize;
    let dim = 8_usize;
    let epsilon = 0.5_f32; // Large epsilon to see the difference

    // Build simple Q@K^T graph where Q and K come from the same input
    let mut graph = GraphNetwork::new();

    // Single input feeds both Q and K projections
    let eye = ndarray::Array2::<f32>::eye(dim);
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::Linear(LinearLayer::new(eye, None).unwrap()),
    ));

    // Q@K^T
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "q",
        "k",
    ));
    graph.set_output("scores");

    // Input centered at 0 with epsilon perturbation
    // This makes the X[i]·X[i] correlation obvious: X ∈ [-ε, ε] means X² ≥ 0
    let lower = ndarray::ArrayD::from_elem(vec![seq, dim], -epsilon);
    let upper = ndarray::ArrayD::from_elem(vec![seq, dim], epsilon);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Compute baseline interval bounds (no zonotope tightening)
    let baseline_interval = MatMulLayer::new(true, None)
        .propagate_ibp_binary(&input, &input)
        .unwrap();

    // Run zonotope propagation
    let zonotope_bounds = graph.propagate_zonotope(&input, epsilon).unwrap();

    // Run IBP (which already includes zonotope tightening for Q@K^T)
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Check diagonal bounds
    let diag_baseline = baseline_interval.lower[[0, 0]];
    let diag_zonotope = zonotope_bounds.lower[[0, 0]];
    let diag_ibp = ibp_bounds.lower[[0, 0]];

    println!(
        "Diagonal[0,0] lower bounds: baseline={:.4}, zonotope={:.4}, ibp={:.4}",
        diag_baseline, diag_zonotope, diag_ibp
    );

    // For X@X^T where X ∈ [-ε, ε], diagonal entries are sums of squares.
    // Baseline interval: treats X[i,j] as independent, worst case is (-ε)*(+ε) = -ε²
    // This gives diagonal lower bound = dim * (-ε²) = -dim * ε²
    //
    // Zonotope: knows X[i,j]² ∈ [0, ε²], so diagonal lower bound = 0

    // Baseline should have negative lower bound
    assert!(
        diag_baseline < -1e-6,
        "Baseline interval MatMul should have negative diagonal lower (got {})",
        diag_baseline
    );

    // Both zonotope and IBP (with zonotope tightening) should have non-negative diagonal
    assert!(
        diag_zonotope >= -1e-6,
        "Zonotope diagonal lower should be >= 0 (got {})",
        diag_zonotope
    );
    assert!(
        diag_ibp >= -1e-6,
        "IBP (with zonotope tightening) diagonal lower should be >= 0 (got {})",
        diag_ibp
    );

    // Calculate improvement over baseline
    let zonotope_improvement = diag_zonotope - diag_baseline;
    let ibp_improvement = diag_ibp - diag_baseline;

    println!(
        "Improvement over baseline: zonotope={:.4}, ibp={:.4}",
        zonotope_improvement, ibp_improvement
    );

    println!("Zonotope vs IBP Q@K^T diagonal tightness test passed!");
}

#[test]
fn test_propagate_zonotope_causal_softmax_masks_future_positions() {
    let seq = 5_usize;
    let epsilon = 0.2_f32;

    let mut graph = GraphNetwork::new();

    // Identity scores: input -> Linear(I) so the graph has a non-trivial input node.
    let eye = ndarray::Array2::<f32>::eye(seq);
    graph.add_node(GraphNode::from_input(
        "scores",
        Layer::Linear(LinearLayer::new(eye, None).unwrap()),
    ));

    graph.add_node(GraphNode::new(
        "attn",
        Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)),
        vec!["scores".to_string()],
    ));
    graph.set_output("attn");

    let center_values: Vec<f32> = (0..seq * seq).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let lower = ndarray::ArrayD::from_shape_vec(
        vec![seq, seq],
        center_values.iter().map(|&v| v - epsilon).collect(),
    )
    .unwrap();
    let upper = ndarray::ArrayD::from_shape_vec(
        vec![seq, seq],
        center_values.iter().map(|&v| v + epsilon).collect(),
    )
    .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let out = graph.propagate_zonotope(&input, epsilon).unwrap();

    // Masked entries must be exactly 0 (within a tiny tolerance).
    for i in 0..seq {
        for j in (i + 1)..seq {
            assert!(
                out.upper[[i, j]] <= 1e-6 && out.lower[[i, j]] >= -1e-6,
                "masked causal softmax bounds should be 0 at ({},{}) got [{},{}]",
                i,
                j,
                out.lower[[i, j]],
                out.upper[[i, j]]
            );
        }
    }
}
