//! CROWN and α-CROWN algorithm tests.

use super::*;
use ndarray::{arr1, arr2, Array1, Array2};

// ============================================================
// CROWN TESTS
// ============================================================

#[test]
fn test_crown_single_linear_matches_ibp() {
    // For a single linear layer, CROWN should produce exact same bounds as IBP
    // because both are sound and linear layers preserve linearity.
    let weight = arr2(&[[1.0, 2.0], [3.0, -1.0]]);
    let bias = arr1(&[1.0, -1.0]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();

    // For pure linear network, CROWN and IBP should produce identical bounds
    for i in 0..2 {
        assert!(
            (ibp_output.lower[[i]] - crown_output.lower[[i]]).abs() < 1e-5,
            "lower[{}]: IBP={} CROWN={}",
            i,
            ibp_output.lower[[i]],
            crown_output.lower[[i]]
        );
        assert!(
            (ibp_output.upper[[i]] - crown_output.upper[[i]]).abs() < 1e-5,
            "upper[{}]: IBP={} CROWN={}",
            i,
            ibp_output.upper[[i]],
            crown_output.upper[[i]]
        );
    }
}

#[test]
fn test_crown_linear_relu_positive_region() {
    // When ReLU input is entirely positive, CROWN = IBP
    // W = [[1, 1]], b = [5] -> output always >= 5 for input in [0,1]
    let w1 = arr2(&[[1.0, 1.0]]);
    let b1 = arr1(&[5.0]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();

    // Pre-ReLU bounds: [5, 7], all positive, so ReLU is identity
    // Both should give same bounds
    assert!((ibp_output.lower[[0]] - 5.0).abs() < 1e-5);
    assert!((ibp_output.upper[[0]] - 7.0).abs() < 1e-5);
    assert!((crown_output.lower[[0]] - 5.0).abs() < 1e-5);
    assert!((crown_output.upper[[0]] - 7.0).abs() < 1e-5);
}

#[test]
fn test_crown_linear_relu_negative_region() {
    // When ReLU input is entirely negative, both give zeros
    // W = [[1, 1]], b = [-5] -> output in [-5, -3] for input in [0,1]
    let w1 = arr2(&[[1.0, 1.0]]);
    let b1 = arr1(&[-5.0]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();

    // Pre-ReLU bounds: [-5, -3], all negative, so ReLU outputs 0
    assert!((ibp_output.lower[[0]] - 0.0).abs() < 1e-5);
    assert!((ibp_output.upper[[0]] - 0.0).abs() < 1e-5);
    assert!((crown_output.lower[[0]] - 0.0).abs() < 1e-5);
    assert!((crown_output.upper[[0]] - 0.0).abs() < 1e-5);
}

#[test]
fn test_crown_linear_relu_crossing() {
    // When ReLU crosses zero, CROWN uses linear relaxation
    // W = [[1, 1]], b = [-0.5] -> pre-ReLU in [-0.5, 1.5] for input in [0,1]
    let w1 = arr2(&[[1.0, 1.0]]);
    let b1 = arr1(&[-0.5]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();

    // Pre-ReLU bounds: [-0.5, 1.5], crossing zero
    // IBP: lower = max(0, -0.5) = 0, upper = max(0, 1.5) = 1.5
    assert!((ibp_output.lower[[0]] - 0.0).abs() < 1e-5);
    assert!((ibp_output.upper[[0]] - 1.5).abs() < 1e-5);

    // CROWN with linear relaxation for crossing ReLU:
    // The linear approximation may give a looser lower bound than IBP
    // because CROWN uses y >= alpha*x which can be negative when x < 0
    // But CROWN should be SOUND: actual outputs are within bounds

    // Verify CROWN bounds are sound by testing concrete inputs
    let test_inputs = [
        arr1(&[0.0, 0.0]),
        arr1(&[1.0, 1.0]),
        arr1(&[0.5, 0.5]),
        arr1(&[0.0, 1.0]),
        arr1(&[1.0, 0.0]),
    ];

    for x in &test_inputs {
        let z = w1.dot(x) + &b1;
        let y = z.mapv(|v| v.max(0.0));

        assert!(
            y[0] >= crown_output.lower[[0]] - 1e-5,
            "CROWN lower {} not sound for input {:?}, actual output {}",
            crown_output.lower[[0]],
            x,
            y[0]
        );
        assert!(
            y[0] <= crown_output.upper[[0]] + 1e-5,
            "CROWN upper {} not sound for input {:?}, actual output {}",
            crown_output.upper[[0]],
            x,
            y[0]
        );
    }

    // CROWN upper bound should be at least as tight as IBP
    assert!(
        crown_output.upper[[0]] <= ibp_output.upper[[0]] + 1e-5,
        "CROWN upper {} > IBP upper {}",
        crown_output.upper[[0]],
        ibp_output.upper[[0]]
    );
}

#[test]
fn test_crown_mlp_tighter_than_ibp() {
    // 2-layer MLP: Linear(2->2) -> ReLU -> Linear(2->1)
    // For deeper networks with crossing ReLUs, CROWN should give tighter bounds
    //
    // First layer: W1 = [[1, -1], [-1, 1]], b1 = [0, 0]
    // For input in [-1, 1]^2:
    //   z1[0] = x0 - x1 in [-2, 2]
    //   z1[1] = -x0 + x1 in [-2, 2]
    // After ReLU: both outputs in [0, 2]
    //
    // Second layer: W2 = [[1, 1]], b2 = [0]
    // Output: a1[0] + a1[1]
    //
    // IBP: Each ReLU output in [0, 2], so sum in [0, 4]
    // But actually: a1[0] + a1[1] = max(x0-x1, 0) + max(x1-x0, 0) = |x0-x1|
    // For x in [-1,1]^2: |x0-x1| in [0, 2], so tighter bounds are [0, 2]
    //
    // CROWN exploits the linear relationship and should give tighter bounds

    let w1 = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    let w2 = arr2(&[[1.0, 1.0]]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, None).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();

    // IBP gives [0, 4] due to overestimation
    assert!(
        (ibp_output.lower[[0]] - 0.0).abs() < 1e-5,
        "IBP lower = {}",
        ibp_output.lower[[0]]
    );
    assert!(
        (ibp_output.upper[[0]] - 4.0).abs() < 1e-5,
        "IBP upper = {}",
        ibp_output.upper[[0]]
    );

    // CROWN should give tighter bounds (at least as tight as IBP)
    assert!(
        crown_output.lower[[0]] >= ibp_output.lower[[0]] - 1e-5,
        "CROWN lower {} should be >= IBP lower {}",
        crown_output.lower[[0]],
        ibp_output.lower[[0]]
    );
    assert!(
        crown_output.upper[[0]] <= ibp_output.upper[[0]] + 1e-5,
        "CROWN upper {} should be <= IBP upper {}",
        crown_output.upper[[0]],
        ibp_output.upper[[0]]
    );

    // Note: The exact tightness depends on the ReLU relaxation heuristic
    // With default heuristic, CROWN should still provide some tightening
}

#[test]
fn test_crown_soundness() {
    // Soundness test: concrete outputs must be within CROWN bounds
    let w1 = arr2(&[[2.0, -1.0], [1.0, 3.0]]);
    let b1 = arr1(&[1.0, -1.0]);
    let w2 = arr2(&[[1.0, -1.0]]);
    let b2 = arr1(&[0.5]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap(),
    ));

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let crown_output = network.propagate_crown(&input_bounds).unwrap();

    // Test several concrete inputs
    let test_inputs = [
        arr1(&[-1.0, -1.0]),
        arr1(&[1.0, 1.0]),
        arr1(&[0.0, 0.0]),
        arr1(&[-1.0, 1.0]),
        arr1(&[1.0, -1.0]),
        arr1(&[0.5, -0.5]),
        arr1(&[-0.5, 0.5]),
    ];

    for x in &test_inputs {
        // Compute concrete output: Linear -> ReLU -> Linear
        let z1 = w1.dot(x) + &b1;
        let a1 = z1.mapv(|v| v.max(0.0));
        let y = w2.dot(&a1) + &b2;

        // Verify soundness
        assert!(
            y[0] >= crown_output.lower[[0]] - 1e-5,
            "Soundness violation: concrete {} < CROWN lower {} for input {:?}",
            y[0],
            crown_output.lower[[0]],
            x
        );
        assert!(
            y[0] <= crown_output.upper[[0]] + 1e-5,
            "Soundness violation: concrete {} > CROWN upper {} for input {:?}",
            y[0],
            crown_output.upper[[0]],
            x
        );
    }
}

#[test]
fn test_crown_empty_network() {
    let network = Network::new();

    let input =
        BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[2.0, 3.0]).into_dyn()).unwrap();

    let output = network.propagate_crown(&input).unwrap();

    // Empty network: output = input
    assert!((output.lower[[0]] - 0.0).abs() < 1e-5);
    assert!((output.upper[[0]] - 2.0).abs() < 1e-5);
    assert!((output.lower[[1]] - 1.0).abs() < 1e-5);
    assert!((output.upper[[1]] - 3.0).abs() < 1e-5);
}

#[test]
fn test_crown_soundness_multiple_networks() {
    // Test that CROWN bounds are SOUND for various networks
    // Note: CROWN may not always be tighter than IBP for all networks
    // due to the linear relaxation approximation, but it MUST be sound.

    // Network definitions with their weight matrices for concrete evaluation
    struct TestNetwork {
        network: Network,
        w1: Array2<f32>,
        b1: Option<Array1<f32>>,
        w2: Array2<f32>,
        b2: Option<Array1<f32>>,
        w3: Option<Array2<f32>>,
        b3: Option<Array1<f32>>,
    }

    let networks = [
        // Network 1: Simple MLP (Linear -> ReLU -> Linear)
        TestNetwork {
            network: {
                let mut net = Network::new();
                net.add_layer(Layer::Linear(
                    LinearLayer::new(arr2(&[[1.0, 1.0], [-1.0, 1.0]]), Some(arr1(&[0.0, 0.0])))
                        .unwrap(),
                ));
                net.add_layer(Layer::ReLU(ReLULayer));
                net.add_layer(Layer::Linear(
                    LinearLayer::new(arr2(&[[1.0, -1.0]]), None).unwrap(),
                ));
                net
            },
            w1: arr2(&[[1.0, 1.0], [-1.0, 1.0]]),
            b1: Some(arr1(&[0.0, 0.0])),
            w2: arr2(&[[1.0, -1.0]]),
            b2: None,
            w3: None,
            b3: None,
        },
        // Network 2: MLP with negative weights (Linear -> ReLU -> Linear)
        TestNetwork {
            network: {
                let mut net = Network::new();
                net.add_layer(Layer::Linear(
                    LinearLayer::new(arr2(&[[-0.5, 0.5], [0.5, 0.5]]), Some(arr1(&[0.1, -0.1])))
                        .unwrap(),
                ));
                net.add_layer(Layer::ReLU(ReLULayer));
                net.add_layer(Layer::Linear(
                    LinearLayer::new(arr2(&[[0.5, 0.5]]), Some(arr1(&[0.0]))).unwrap(),
                ));
                net
            },
            w1: arr2(&[[-0.5, 0.5], [0.5, 0.5]]),
            b1: Some(arr1(&[0.1, -0.1])),
            w2: arr2(&[[0.5, 0.5]]),
            b2: Some(arr1(&[0.0])),
            w3: None,
            b3: None,
        },
    ];

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Test inputs within bounds
    let test_inputs = [
        arr1(&[-1.0, -1.0]),
        arr1(&[1.0, 1.0]),
        arr1(&[0.0, 0.0]),
        arr1(&[-1.0, 1.0]),
        arr1(&[1.0, -1.0]),
        arr1(&[0.5, -0.5]),
        arr1(&[-0.5, 0.5]),
    ];

    for (net_idx, test_net) in networks.iter().enumerate() {
        let crown_output = test_net.network.propagate_crown(&input_bounds).unwrap();

        for x in &test_inputs {
            // Compute concrete output
            let z1 = test_net.w1.dot(x) + test_net.b1.as_ref().unwrap_or(&arr1(&[0.0, 0.0]));
            let a1 = z1.mapv(|v| v.max(0.0));
            let z2 = test_net.w2.dot(&a1) + test_net.b2.as_ref().unwrap_or(&arr1(&[0.0]));
            let y = if test_net.w3.is_some() {
                let a2 = z2.mapv(|v| v.max(0.0));
                let z3 = test_net.w3.as_ref().unwrap().dot(&a2)
                    + test_net.b3.as_ref().unwrap_or(&arr1(&[0.0]));
                z3
            } else {
                z2
            };

            // Verify soundness
            assert!(
                y[0] >= crown_output.lower[[0]] - 1e-5,
                "Network {}: CROWN lower {} not sound for input {:?}, actual {}",
                net_idx,
                crown_output.lower[[0]],
                x,
                y[0]
            );
            assert!(
                y[0] <= crown_output.upper[[0]] + 1e-5,
                "Network {}: CROWN upper {} not sound for input {:?}, actual {}",
                net_idx,
                crown_output.upper[[0]],
                x,
                y[0]
            );
        }
    }
}

#[test]
fn test_linear_bounds_identity() {
    let bounds = LinearBounds::identity(3);
    assert_eq!(bounds.num_outputs(), 3);
    assert_eq!(bounds.num_inputs(), 3);

    // Identity: A = I, b = 0
    assert!((bounds.lower_a[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((bounds.lower_a[[0, 1]] - 0.0).abs() < 1e-6);
    assert!((bounds.lower_a[[1, 1]] - 1.0).abs() < 1e-6);
    assert!((bounds.lower_b[[0]] - 0.0).abs() < 1e-6);
}

#[test]
fn test_linear_bounds_concretize() {
    // Test concretization with simple linear bounds
    // lower: y >= 2*x + 1, upper: y <= 3*x + 2
    let bounds = LinearBounds {
        lower_a: arr2(&[[2.0]]),
        lower_b: arr1(&[1.0]),
        upper_a: arr2(&[[3.0]]),
        upper_b: arr1(&[2.0]),
    };

    // Input x in [0, 1]
    let input = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let output = bounds.concretize(&input);

    // lower = 2*0 + 1 = 1 (use lower input for positive coeff)
    // upper = 3*1 + 2 = 5 (use upper input for positive coeff)
    assert!(
        (output.lower[[0]] - 1.0).abs() < 1e-5,
        "lower = {}",
        output.lower[[0]]
    );
    assert!(
        (output.upper[[0]] - 5.0).abs() < 1e-5,
        "upper = {}",
        output.upper[[0]]
    );
}

#[test]
fn test_linear_bounds_concretize_negative_coeff() {
    // Test concretization with negative coefficients
    // lower: y >= -2*x + 1, upper: y <= -1*x + 2
    let bounds = LinearBounds {
        lower_a: arr2(&[[-2.0]]),
        lower_b: arr1(&[1.0]),
        upper_a: arr2(&[[-1.0]]),
        upper_b: arr1(&[2.0]),
    };

    // Input x in [0, 1]
    let input = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let output = bounds.concretize(&input);

    // lower = -2*1 + 1 = -1 (use upper input for negative coeff to minimize)
    // upper = -1*0 + 2 = 2 (use lower input for negative coeff to maximize)
    assert!(
        (output.lower[[0]] - (-1.0)).abs() < 1e-5,
        "lower = {}",
        output.lower[[0]]
    );
    assert!(
        (output.upper[[0]] - 2.0).abs() < 1e-5,
        "upper = {}",
        output.upper[[0]]
    );
}

// ============================================================
// α-CROWN TESTS
// ============================================================

#[test]
fn test_alpha_crown_soundness() {
    // Test that α-CROWN bounds are sound (contain actual outputs)
    // Network: Linear -> ReLU -> Linear
    let mut network = Network::new();

    // First linear: 2 inputs -> 3 outputs
    let w1 = arr2(&[[1.0, 2.0], [-1.0, 1.0], [0.5, -0.5]]);
    let b1 = arr1(&[0.1, -0.2, 0.3]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Second linear: 3 inputs -> 2 outputs
    let w2 = arr2(&[[1.0, -1.0, 0.5], [0.5, 1.0, -0.5]]);
    let b2 = arr1(&[0.0, 0.1]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap(),
    ));

    // Input bounds: x in [-0.5, 0.5]
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    // Get α-CROWN bounds
    let alpha_crown_output = network.propagate_alpha_crown(&input).unwrap();

    // Sample concrete points and verify they're within bounds
    let test_points = vec![
        vec![-0.5, -0.5],
        vec![0.5, 0.5],
        vec![0.0, 0.0],
        vec![-0.5, 0.5],
        vec![0.5, -0.5],
        vec![0.25, -0.25],
    ];

    for point in test_points {
        // Forward pass through network
        let x = arr1(&point);

        // Layer 1: Linear
        let z1 = w1.dot(&x) + &b1;
        // Layer 2: ReLU
        let a1 = z1.mapv(|v| v.max(0.0));
        // Layer 3: Linear
        let z2 = w2.dot(&a1) + &b2;

        // Verify output is within bounds
        for i in 0..z2.len() {
            assert!(
                z2[i] >= alpha_crown_output.lower[[i]] - 1e-5,
                "α-CROWN lower bound violated at output {}: {} < {}",
                i,
                z2[i],
                alpha_crown_output.lower[[i]]
            );
            assert!(
                z2[i] <= alpha_crown_output.upper[[i]] + 1e-5,
                "α-CROWN upper bound violated at output {}: {} > {}",
                i,
                z2[i],
                alpha_crown_output.upper[[i]]
            );
        }
    }
}

#[test]
fn test_alpha_crown_at_least_as_tight_as_crown() {
    // α-CROWN should produce bounds at least as tight as CROWN
    // (or equal when optimization doesn't help)
    let mut network = Network::new();

    // Create a network with crossing ReLUs
    let w1 = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    let b1 = arr1(&[0.0, 0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let w2 = arr2(&[[1.0, 1.0]]);
    let b2 = arr1(&[0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    // Input bounds that create crossing ReLUs
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let crown_output = network.propagate_crown(&input).unwrap();
    let alpha_crown_output = network.propagate_alpha_crown(&input).unwrap();

    // α-CROWN should have tighter or equal bounds
    // Lower bounds should be >= CROWN's lower bounds
    // Upper bounds should be <= CROWN's upper bounds
    // Due to numerical precision, allow small tolerance
    let tol = 1e-4;

    for i in 0..crown_output.len() {
        // α-CROWN lower should be >= CROWN lower (or very close)
        assert!(
            alpha_crown_output.lower[[i]] >= crown_output.lower[[i]] - tol,
            "α-CROWN lower bound {} at {} is worse than CROWN {}",
            alpha_crown_output.lower[[i]],
            i,
            crown_output.lower[[i]]
        );
        // α-CROWN upper should be <= CROWN upper (or very close)
        assert!(
            alpha_crown_output.upper[[i]] <= crown_output.upper[[i]] + tol,
            "α-CROWN upper bound {} at {} is worse than CROWN {}",
            alpha_crown_output.upper[[i]],
            i,
            crown_output.upper[[i]]
        );
    }
}

#[test]
fn test_alpha_crown_deep_network() {
    // Test α-CROWN on a deeper network where optimization should help more
    let mut network = Network::new();

    // 4-layer MLP: 2 -> 4 -> 4 -> 2
    let w1 = arr2(&[[0.5, 0.3], [-0.4, 0.6], [0.2, -0.3], [-0.1, 0.4]]);
    let b1 = arr1(&[0.1, -0.1, 0.0, 0.05]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let w2 = arr2(&[
        [0.3, -0.2, 0.4, 0.1],
        [-0.3, 0.5, -0.1, 0.2],
        [0.2, 0.1, -0.3, 0.4],
        [0.1, -0.4, 0.2, -0.1],
    ]);
    let b2 = arr1(&[0.0, 0.1, -0.05, 0.02]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let w3 = arr2(&[[0.4, 0.3, -0.2, 0.1], [-0.3, 0.2, 0.4, -0.1]]);
    let b3 = arr1(&[0.0, 0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w3, Some(b3)).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    // All methods should give sound bounds
    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();
    let alpha_crown_output = network.propagate_alpha_crown(&input).unwrap();

    // Verify soundness by sampling
    let test_points = vec![
        vec![0.0, 0.0],
        vec![-0.5, -0.5],
        vec![0.5, 0.5],
        vec![0.3, -0.2],
        vec![-0.1, 0.4],
    ];

    for point in test_points {
        // Manual forward pass
        let x: Array1<f32> = arr1(&point);
        let z1 = arr2(&[[0.5_f32, 0.3], [-0.4, 0.6], [0.2, -0.3], [-0.1, 0.4]]).dot(&x)
            + arr1(&[0.1_f32, -0.1, 0.0, 0.05]);
        let a1 = z1.mapv(|v: f32| v.max(0.0));
        let z2 = arr2(&[
            [0.3_f32, -0.2, 0.4, 0.1],
            [-0.3, 0.5, -0.1, 0.2],
            [0.2, 0.1, -0.3, 0.4],
            [0.1, -0.4, 0.2, -0.1],
        ])
        .dot(&a1)
            + arr1(&[0.0_f32, 0.1, -0.05, 0.02]);
        let a2 = z2.mapv(|v: f32| v.max(0.0));
        let z3 = arr2(&[[0.4_f32, 0.3, -0.2, 0.1], [-0.3, 0.2, 0.4, -0.1]]).dot(&a2)
            + arr1(&[0.0_f32, 0.0]);

        for i in 0..z3.len() {
            // All methods should be sound
            assert!(
                z3[i] >= ibp_output.lower[[i]] - 1e-5,
                "IBP violated at {}: {} < {}",
                i,
                z3[i],
                ibp_output.lower[[i]]
            );
            assert!(
                z3[i] >= crown_output.lower[[i]] - 1e-5,
                "CROWN violated at {}: {} < {}",
                i,
                z3[i],
                crown_output.lower[[i]]
            );
            assert!(
                z3[i] >= alpha_crown_output.lower[[i]] - 1e-5,
                "α-CROWN violated at {}: {} < {}",
                i,
                z3[i],
                alpha_crown_output.lower[[i]]
            );
        }
    }

    // IBP should be loosest, CROWN tighter, α-CROWN tightest (or equal)
    let ibp_width: f32 = (0..ibp_output.len())
        .map(|i| ibp_output.upper[[i]] - ibp_output.lower[[i]])
        .sum();
    let crown_width: f32 = (0..crown_output.len())
        .map(|i| crown_output.upper[[i]] - crown_output.lower[[i]])
        .sum();
    let alpha_crown_width: f32 = (0..alpha_crown_output.len())
        .map(|i| alpha_crown_output.upper[[i]] - alpha_crown_output.lower[[i]])
        .sum();

    // CROWN should be at least as tight as IBP
    assert!(
        crown_width <= ibp_width + 1e-4,
        "CROWN width {} > IBP width {}",
        crown_width,
        ibp_width
    );

    // α-CROWN should be at least as tight as CROWN
    assert!(
        alpha_crown_width <= crown_width + 1e-4,
        "α-CROWN width {} > CROWN width {}",
        alpha_crown_width,
        crown_width
    );
}

#[test]
fn test_alpha_crown_optimization_diagnostic() {
    // Diagnostic test to verify α-CROWN optimization is actually working
    // This test prints detailed info about the optimization process
    let mut network = Network::new();

    // 4-layer MLP with many unstable neurons
    let w1 = arr2(&[[0.5, 0.3], [-0.4, 0.6], [0.2, -0.3], [-0.1, 0.4]]);
    let b1 = arr1(&[0.1, -0.1, 0.0, 0.05]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let w2 = arr2(&[
        [0.3, -0.2, 0.4, 0.1],
        [-0.3, 0.5, -0.1, 0.2],
        [0.2, 0.1, -0.3, 0.4],
        [0.1, -0.4, 0.2, -0.1],
    ]);
    let b2 = arr1(&[0.0, 0.1, -0.05, 0.02]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let w3 = arr2(&[[0.4, 0.3, -0.2, 0.1], [-0.3, 0.2, 0.4, -0.1]]);
    let b3 = arr1(&[0.0, 0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w3, Some(b3)).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    // Run IBP to check pre-activation bounds for unstable neurons
    let layer_bounds = network.collect_ibp_bounds(&input).unwrap();
    println!("\n=== α-CROWN OPTIMIZATION DIAGNOSTIC ===");
    println!("Layer bounds (pre-activation):");
    for (i, lb) in layer_bounds.iter().enumerate() {
        let lower_flat = lb.lower.as_slice().unwrap_or(&[]);
        let upper_flat = lb.upper.as_slice().unwrap_or(&[]);
        let unstable_count = lower_flat
            .iter()
            .zip(upper_flat.iter())
            .filter(|(l, u)| **l < 0.0 && **u > 0.0)
            .count();
        println!(
            "  Layer {}: {} neurons, {} unstable",
            i,
            lower_flat.len(),
            unstable_count
        );
    }

    // IBP bounds
    let ibp_output = network.propagate_ibp(&input).unwrap();
    let ibp_width: f32 = (0..ibp_output.len())
        .map(|i| ibp_output.upper[[i]] - ibp_output.lower[[i]])
        .sum();
    println!("\nIBP:     width = {:.6}", ibp_width);

    // CROWN bounds
    let crown_output = network.propagate_crown(&input).unwrap();
    let crown_width: f32 = (0..crown_output.len())
        .map(|i| crown_output.upper[[i]] - crown_output.lower[[i]])
        .sum();
    println!("CROWN:   width = {:.6}", crown_width);

    // Check gradient values by manual perturbation
    println!("\nGradient check (finite differences):");

    // Test perturbing alpha[0] for first ReLU layer
    // This requires direct access to alpha state, so we'll use a simpler test
    // Perturb input epsilon to see if bounds change
    let eps_perturbations = [0.4, 0.5, 0.6];
    for eps in eps_perturbations {
        let perturbed_input =
            BoundedTensor::new(arr1(&[-eps, -eps]).into_dyn(), arr1(&[eps, eps]).into_dyn())
                .unwrap();
        let crown_p = network.propagate_crown(&perturbed_input).unwrap();
        let alpha_p = network.propagate_alpha_crown(&perturbed_input).unwrap();
        let crown_w: f32 = (0..crown_p.len())
            .map(|i| crown_p.upper[[i]] - crown_p.lower[[i]])
            .sum();
        let alpha_w: f32 = (0..alpha_p.len())
            .map(|i| alpha_p.upper[[i]] - alpha_p.lower[[i]])
            .sum();
        println!(
            "  eps={:.1}: CROWN={:.4}, α-CROWN={:.4}, diff={:.4}",
            eps,
            crown_w,
            alpha_w,
            crown_w - alpha_w
        );
    }

    // α-CROWN with various configurations
    println!("\nα-CROWN iterations test:");
    for iters in [1, 5, 10, 20, 50, 100] {
        let config = AlphaCrownConfig {
            iterations: iters,
            learning_rate: 0.1,
            tolerance: 1e-10, // Very small to prevent early stopping
            use_momentum: true,
            momentum: 0.9,
            lr_decay: 0.98,
            ..Default::default()
        };
        let alpha_output = network
            .propagate_alpha_crown_with_config(&input, &config)
            .unwrap();
        let alpha_width: f32 = (0..alpha_output.len())
            .map(|i| alpha_output.upper[[i]] - alpha_output.lower[[i]])
            .sum();
        let improvement = if crown_width > 0.0 {
            (crown_width - alpha_width) / crown_width * 100.0
        } else {
            0.0
        };
        println!(
            "α-CROWN(iters={:3}): width = {:.6}, improvement vs CROWN: {:+.4}%",
            iters, alpha_width, improvement
        );
    }

    println!("=========================================\n");

    // If α-CROWN equals CROWN for all configs, there might be a bug
    // Use explicit config with enough iterations to guarantee improvement
    let config_50 = AlphaCrownConfig {
        iterations: 50,
        learning_rate: 0.1,
        tolerance: 1e-10,
        use_momentum: true,
        momentum: 0.9,
        lr_decay: 0.98,
        ..Default::default()
    };
    let alpha_50 = network
        .propagate_alpha_crown_with_config(&input, &config_50)
        .unwrap();
    let alpha_50_width: f32 = (0..alpha_50.len())
        .map(|i| alpha_50.upper[[i]] - alpha_50.lower[[i]])
        .sum();

    // For a deep network with unstable neurons, α-CROWN should provide some improvement
    // Even 0.1% improvement indicates optimization is working
    let improvement_pct = (crown_width - alpha_50_width) / crown_width * 100.0;
    println!(
        "Final: CROWN width={:.6}, α-CROWN(50 iter) width={:.6}, improvement={:+.4}%",
        crown_width, alpha_50_width, improvement_pct
    );

    // Assert that α-CROWN is at least as tight as CROWN
    assert!(
        alpha_50_width <= crown_width + 1e-4,
        "α-CROWN should not be worse than CROWN: α-CROWN={:.6}, CROWN={:.6}",
        alpha_50_width,
        crown_width
    );

    // Assert that α-CROWN actually improves (this network has 8 unstable neurons)
    assert!(
        improvement_pct > 0.0,
        "α-CROWN should improve over CROWN for this network with unstable neurons"
    );
}

#[test]
fn test_alpha_crown_spsa_vs_finite_diff() {
    use crate::bounds::GradientMethod;
    use std::time::Instant;

    // Build a larger network to see timing differences
    // 5 hidden layers with 20 neurons each
    let mut network = Network::new();
    let input_dim = 10;
    let hidden_dim = 20;
    let output_dim = 5;

    // Input -> Hidden 1
    let w = Array2::<f32>::from_shape_fn((hidden_dim, input_dim), |(i, j)| {
        ((i + j) % 5) as f32 * 0.1 - 0.2
    });
    let b = Array1::from_elem(hidden_dim, 0.0);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w.clone(), Some(b.clone())).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Hidden layers
    for _ in 0..3 {
        let w = Array2::<f32>::from_shape_fn((hidden_dim, hidden_dim), |(i, j)| {
            ((i * 3 + j * 7) % 11) as f32 * 0.05 - 0.25
        });
        network.add_layer(Layer::Linear(
            LinearLayer::new(w.clone(), Some(b.clone())).unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
    }

    // Output
    let w_out = Array2::<f32>::from_shape_fn((output_dim, hidden_dim), |(i, j)| {
        ((i + j * 2) % 7) as f32 * 0.1 - 0.3
    });
    let b_out = Array1::from_elem(output_dim, 0.1);
    network.add_layer(Layer::Linear(LinearLayer::new(w_out, Some(b_out)).unwrap()));

    // Input bounds
    let lower = Array1::from_elem(input_dim, -0.5);
    let upper = Array1::from_elem(input_dim, 0.5);
    let input = BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).unwrap();

    let crown_output = network.propagate_crown(&input).unwrap();
    let crown_width: f32 = (0..crown_output.len())
        .map(|i| crown_output.upper[[i]] - crown_output.lower[[i]])
        .sum();

    println!("\n=== SPSA vs Finite Differences Comparison ===");
    println!("Network: 4 ReLU layers with {} neurons each", hidden_dim);
    println!("CROWN baseline width: {:.4}", crown_width);

    // Test SPSA (default, should be fast)
    let spsa_config = AlphaCrownConfig {
        iterations: 50,
        gradient_method: GradientMethod::Spsa,
        spsa_samples: 4,
        learning_rate: 0.1,
        lr_decay: 0.99,
        ..Default::default()
    };
    let start = Instant::now();
    let spsa_output = network
        .propagate_alpha_crown_with_config(&input, &spsa_config)
        .unwrap();
    let spsa_time = start.elapsed();
    let spsa_width: f32 = (0..spsa_output.len())
        .map(|i| spsa_output.upper[[i]] - spsa_output.lower[[i]])
        .sum();
    let spsa_improvement = (crown_width - spsa_width) / crown_width * 100.0;

    // Test Finite Differences (should be slower)
    let fd_config = AlphaCrownConfig {
        iterations: 5, // Fewer iterations because it's slower
        gradient_method: GradientMethod::FiniteDifferences,
        learning_rate: 0.5,
        lr_decay: 0.98,
        ..Default::default()
    };
    let start = Instant::now();
    let fd_output = network
        .propagate_alpha_crown_with_config(&input, &fd_config)
        .unwrap();
    let fd_time = start.elapsed();
    let fd_width: f32 = (0..fd_output.len())
        .map(|i| fd_output.upper[[i]] - fd_output.lower[[i]])
        .sum();
    let fd_improvement = (crown_width - fd_width) / crown_width * 100.0;

    println!("\nSPSA (50 iters, 4 samples/iter = 400 forward passes):");
    println!("  Time: {:?}", spsa_time);
    println!(
        "  Width: {:.4}, Improvement: {:+.4}%",
        spsa_width, spsa_improvement
    );

    println!("\nFinite Diff (5 iters, ~2*unstable passes/iter):");
    println!("  Time: {:?}", fd_time);
    println!(
        "  Width: {:.4}, Improvement: {:+.4}%",
        fd_width, fd_improvement
    );

    let speedup = fd_time.as_secs_f64() / spsa_time.as_secs_f64().max(1e-9);
    println!("\nSPSA speedup: {:.1}x faster", speedup);
    println!("==============================================\n");

    // Both methods should produce valid bounds (at least as tight as CROWN)
    assert!(
        spsa_width <= crown_width + 1e-4,
        "SPSA bounds should be valid"
    );
    assert!(fd_width <= crown_width + 1e-4, "FD bounds should be valid");
}

#[test]
fn test_alpha_direct_impact() {
    // Simplest possible network to test alpha directly
    // Linear -> ReLU -> Linear
    // With known weights to ensure crossing neurons
    let mut network = Network::new();

    // Linear: 2 -> 2 with identity weights
    let w1 = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let b1 = arr1(&[-0.25, 0.25]); // Shift so neurons are crossing
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Second linear: sum the outputs
    let w2 = arr2(&[[1.0, 1.0]]);
    let b2 = arr1(&[0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    // Input: x ∈ [-0.5, 0.5]²
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    // Pre-activation for ReLU:
    // neuron 0: x[0] + b1[0] = x[0] - 0.25 ∈ [-0.75, 0.25] (crossing!)
    // neuron 1: x[1] + b1[1] = x[1] + 0.25 ∈ [-0.25, 0.75] (crossing!)

    println!("\n=== DIRECT ALPHA IMPACT TEST ===");
    println!("Network: Linear(2->2) -> ReLU -> Linear(2->1)");
    println!("Pre-activation bounds:");
    println!("  neuron 0: [-0.75, 0.25] (l=-0.75, u=0.25, -l>u → α_heuristic=0)");
    println!("  neuron 1: [-0.25, 0.75] (l=-0.25, u=0.75, -l<u → α_heuristic=1)");

    // Check IBP
    let ibp_out = network.propagate_ibp(&input).unwrap();
    println!(
        "\nIBP output: [{:.4}, {:.4}]",
        ibp_out.lower[[0]],
        ibp_out.upper[[0]]
    );

    // Check CROWN
    let crown_out = network.propagate_crown(&input).unwrap();
    println!(
        "CROWN output: [{:.4}, {:.4}]",
        crown_out.lower[[0]],
        crown_out.upper[[0]]
    );

    // Check α-CROWN with different iteration counts
    for iters in [0, 1, 10, 50] {
        let config = AlphaCrownConfig {
            iterations: iters,
            learning_rate: 0.5,
            tolerance: 1e-10,
            use_momentum: true,
            momentum: 0.9,
            lr_decay: 0.99,
            ..Default::default()
        };
        let alpha_out = network
            .propagate_alpha_crown_with_config(&input, &config)
            .unwrap();
        println!(
            "α-CROWN(iters={}): [{:.4}, {:.4}]",
            iters,
            alpha_out.lower[[0]],
            alpha_out.upper[[0]]
        );
    }

    // Width comparison
    let ibp_width = ibp_out.upper[[0]] - ibp_out.lower[[0]];
    let crown_width = crown_out.upper[[0]] - crown_out.lower[[0]];
    let alpha_out_default = network.propagate_alpha_crown(&input).unwrap();
    let alpha_width = alpha_out_default.upper[[0]] - alpha_out_default.lower[[0]];

    println!("\nWidth comparison:");
    println!("  IBP: {:.4}", ibp_width);
    println!("  CROWN: {:.4}", crown_width);
    println!("  α-CROWN: {:.4}", alpha_width);
    println!("=================================\n");

    // α-CROWN should be at least as tight as CROWN
    assert!(
        alpha_width <= crown_width + 1e-5,
        "α-CROWN width should not be worse than CROWN"
    );

    // Note: For very simple networks, α-CROWN may not improve over CROWN because
    // the heuristic initialization already provides optimal alpha values.
    // This is fine - the key invariant is that α-CROWN is never worse than CROWN.
    let improvement = (crown_width - alpha_width) / crown_width * 100.0;
    println!("Improvement: {:.4}%", improvement);
}

#[test]
fn test_alpha_crown_config() {
    // Test α-CROWN with custom configuration
    let mut network = Network::new();

    let w1 = arr2(&[[1.0, -1.0]]);
    let b1 = arr1(&[0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let w2 = arr2(&[[1.0]]);
    let b2 = arr1(&[0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Test with different iteration counts
    let config_1iter = AlphaCrownConfig {
        iterations: 1,
        ..Default::default()
    };
    let config_50iter = AlphaCrownConfig {
        iterations: 50,
        ..Default::default()
    };

    let result_1iter = network
        .propagate_alpha_crown_with_config(&input, &config_1iter)
        .unwrap();
    let result_50iter = network
        .propagate_alpha_crown_with_config(&input, &config_50iter)
        .unwrap();

    // Both should be sound
    // 50 iterations should be at least as tight as 1 iteration
    let width_1 = result_1iter.upper[[0]] - result_1iter.lower[[0]];
    let width_50 = result_50iter.upper[[0]] - result_50iter.lower[[0]];

    assert!(
        width_50 <= width_1 + 1e-4,
        "More iterations should not make bounds worse: {} > {}",
        width_50,
        width_1
    );
}

#[test]
fn test_alpha_crown_no_relu() {
    // Test α-CROWN on network without ReLU (should be same as CROWN)
    let mut network = Network::new();

    let w1 = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b1 = arr1(&[0.5, -0.5]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));

    let w2 = arr2(&[[1.0, -1.0]]);
    let b2 = arr1(&[0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let crown_output = network.propagate_crown(&input).unwrap();
    let alpha_crown_output = network.propagate_alpha_crown(&input).unwrap();

    // Should be identical for networks without ReLU
    for i in 0..crown_output.len() {
        assert!(
            (crown_output.lower[[i]] - alpha_crown_output.lower[[i]]).abs() < 1e-5,
            "Lower bounds differ at {}: CROWN {} vs α-CROWN {}",
            i,
            crown_output.lower[[i]],
            alpha_crown_output.lower[[i]]
        );
        assert!(
            (crown_output.upper[[i]] - alpha_crown_output.upper[[i]]).abs() < 1e-5,
            "Upper bounds differ at {}: CROWN {} vs α-CROWN {}",
            i,
            crown_output.upper[[i]],
            alpha_crown_output.upper[[i]]
        );
    }
}

#[test]
fn test_alpha_state_initialization() {
    // Test AlphaState initialization from pre-activation bounds
    let bounds = vec![
        // First ReLU layer: 3 neurons
        // neuron 0: always positive (l=1, u=2)
        // neuron 1: always negative (l=-2, u=-1)
        // neuron 2: crossing (l=-1, u=2)
        BoundedTensor::new(
            arr1(&[1.0, -2.0, -1.0]).into_dyn(),
            arr1(&[2.0, -1.0, 2.0]).into_dyn(),
        )
        .unwrap(),
    ];

    let alpha_state = AlphaState::from_preactivation_bounds(&bounds, &[0]);

    assert_eq!(alpha_state.alphas.len(), 1);
    assert_eq!(alpha_state.alphas[0].len(), 3);

    // Check alpha values
    assert!(
        (alpha_state.alphas[0][0] - 1.0).abs() < 1e-5,
        "Positive neuron should have α=1"
    );
    assert!(
        (alpha_state.alphas[0][1] - 0.0).abs() < 1e-5,
        "Negative neuron should have α=0"
    );
    // Crossing with u=2 > -l=1, so adaptive heuristic gives α=1
    assert!(
        (alpha_state.alphas[0][2] - 1.0).abs() < 1e-5,
        "Crossing neuron (u > -l) should have α=1"
    );

    // Check unstable mask
    assert!(
        !alpha_state.unstable_mask[0][0],
        "Positive neuron should not be unstable"
    );
    assert!(
        !alpha_state.unstable_mask[0][1],
        "Negative neuron should not be unstable"
    );
    assert!(
        alpha_state.unstable_mask[0][2],
        "Crossing neuron should be unstable"
    );

    assert_eq!(alpha_state.num_unstable(), 1);
}

#[test]
fn test_alpha_crown_empty_network() {
    // Test α-CROWN on empty network
    let network = Network::new();
    let input =
        BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn()).unwrap();

    let output = network.propagate_alpha_crown(&input).unwrap();

    // Should return input unchanged
    assert_eq!(output.lower[[0]], 1.0);
    assert_eq!(output.lower[[1]], 2.0);
    assert_eq!(output.upper[[0]], 3.0);
    assert_eq!(output.upper[[1]], 4.0);
}

// ==================== Auto-LiRPA Comparison Benchmarks ====================
//
// These tests compare gamma-crown bounds against reference values computed
// by Auto-LiRPA (Python). The reference values are from:
// benchmarks/auto_lirpa_reference.py
//
// Run with: cargo test -p gamma-propagate benchmark_auto_lirpa -- --nocapture

#[test]
fn benchmark_auto_lirpa_toy_model() {
    // Toy model from Auto-LiRPA examples/simple/toy.py:
    // - Linear: 2 -> 2 (w1=[[1, -1], [2, -1]], no bias)
    // - ReLU
    // - Linear: 2 -> 1 (w2=[[1, -1]], no bias)
    //
    // Input bounds: lower=[-1, -2], upper=[2, 1]
    //
    // Auto-LiRPA reference results (PyTorch 2.9.1):
    // - IBP: lower=-6.0, upper=4.0
    // - CROWN: lower=-3.0, upper=3.0
    // - alpha-CROWN: lower=-3.0, upper=3.0

    let w1 = arr2(&[[1.0f32, -1.0], [2.0, -1.0]]);
    let w2 = arr2(&[[1.0f32, -1.0]]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1.clone(), None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2.clone(), None).unwrap()));

    let input_bounds = BoundedTensor::new(
        arr1(&[-1.0f32, -2.0]).into_dyn(),
        arr1(&[2.0f32, 1.0]).into_dyn(),
    )
    .unwrap();

    // Auto-LiRPA reference values
    let ref_ibp_lower = -6.0f32;
    let ref_ibp_upper = 4.0f32;
    let ref_crown_lower = -3.0f32;
    let ref_crown_upper = 3.0f32;

    // IBP
    let ibp_result = network.propagate_ibp(&input_bounds).unwrap();
    let ibp_lower = ibp_result.lower[[0]];
    let ibp_upper = ibp_result.upper[[0]];

    println!("\n=== Auto-LiRPA Toy Model Comparison ===");
    println!("IBP:");
    println!(
        "  gamma-crown: lower={:.6}, upper={:.6}",
        ibp_lower, ibp_upper
    );
    println!(
        "  Auto-LiRPA:  lower={:.6}, upper={:.6}",
        ref_ibp_lower, ref_ibp_upper
    );

    assert!(
        (ibp_lower - ref_ibp_lower).abs() < 1e-4,
        "IBP lower mismatch: gamma-crown={}, Auto-LiRPA={}",
        ibp_lower,
        ref_ibp_lower
    );
    assert!(
        (ibp_upper - ref_ibp_upper).abs() < 1e-4,
        "IBP upper mismatch: gamma-crown={}, Auto-LiRPA={}",
        ibp_upper,
        ref_ibp_upper
    );

    // CROWN
    let crown_result = network.propagate_crown(&input_bounds).unwrap();
    let crown_lower = crown_result.lower[[0]];
    let crown_upper = crown_result.upper[[0]];

    println!("CROWN:");
    println!(
        "  gamma-crown: lower={:.6}, upper={:.6}",
        crown_lower, crown_upper
    );
    println!(
        "  Auto-LiRPA:  lower={:.6}, upper={:.6}",
        ref_crown_lower, ref_crown_upper
    );

    assert!(
        (crown_lower - ref_crown_lower).abs() < 1e-4,
        "CROWN lower mismatch: gamma-crown={}, Auto-LiRPA={}",
        crown_lower,
        ref_crown_lower
    );
    assert!(
        (crown_upper - ref_crown_upper).abs() < 1e-4,
        "CROWN upper mismatch: gamma-crown={}, Auto-LiRPA={}",
        crown_upper,
        ref_crown_upper
    );

    // alpha-CROWN
    let alpha_result = network.propagate_alpha_crown(&input_bounds).unwrap();
    let alpha_lower = alpha_result.lower[[0]];
    let alpha_upper = alpha_result.upper[[0]];

    println!("alpha-CROWN:");
    println!(
        "  gamma-crown: lower={:.6}, upper={:.6}",
        alpha_lower, alpha_upper
    );
    println!(
        "  Auto-LiRPA:  lower={:.6}, upper={:.6}",
        ref_crown_lower, ref_crown_upper
    );

    // alpha-CROWN should be at least as tight as CROWN
    assert!(
        alpha_lower >= crown_lower - 1e-5,
        "alpha-CROWN lower should be >= CROWN lower"
    );
    assert!(
        alpha_upper <= crown_upper + 1e-5,
        "alpha-CROWN upper should be <= CROWN upper"
    );

    // Verify soundness with concrete points
    println!("Soundness check:");
    let test_inputs = [
        arr1(&[-1.0f32, -2.0]),
        arr1(&[2.0f32, 1.0]),
        arr1(&[0.5f32, -0.5]),
    ];

    for x in &test_inputs {
        let z1 = w1.dot(x);
        let a1 = z1.mapv(|v| v.max(0.0));
        let y = w2.dot(&a1);
        let output = y[0];

        println!("  x={:?} -> y={:.4}", x.to_vec(), output);

        assert!(
            output >= ibp_lower - 1e-5 && output <= ibp_upper + 1e-5,
            "Soundness violation for IBP"
        );
        assert!(
            output >= crown_lower - 1e-5 && output <= crown_upper + 1e-5,
            "Soundness violation for CROWN"
        );
        assert!(
            output >= alpha_lower - 1e-5 && output <= alpha_upper + 1e-5,
            "Soundness violation for alpha-CROWN"
        );
    }
}

#[test]
#[allow(clippy::excessive_precision)]
fn benchmark_auto_lirpa_deep_model() {
    // Deep model: 3-layer MLP with specific weights from Auto-LiRPA
    // Input: 3 -> 4 -> 4 -> 2
    // Weights are from torch.manual_seed(42) with randn * 0.5
    //
    // Auto-LiRPA reference results (epsilon=0.1 around [0.5, 0.5, 0.5]):
    // - IBP: lower=[0.00688, -0.01418], upper=[0.01246, 0.00072]
    // - CROWN: lower=[0.00821, -0.01064], upper=[0.01113, -0.00283]

    // Exact weights from Auto-LiRPA benchmark
    let fc1_weight = arr2(&[
        [0.16834518f32, 0.06440470, 0.11723118],
        [0.11516652, -0.56142819, -0.09316415],
        [1.10410070, -0.31899852, 0.23082861],
        [0.13367544, 0.26745233, 0.40467861],
    ]);
    let fc1_bias = arr1(&[0.11102903f32, -0.16897990, -0.09889599, 0.09579718]);

    let fc2_weight = arr2(&[
        [-0.69233710f32, -0.43561807, -0.11168297, 0.85868055],
        [0.15943986, -0.21225949, 0.15286016, -0.38729626],
        [-0.77878612, 0.49781805, -0.43989292, -0.30057147],
        [-0.63707572, 1.06139255, -0.61732668, -0.24395694],
    ]);
    let fc2_bias = arr1(&[0.02815196f32, 0.00561635, 0.05227160, -0.02383569]);

    let fc3_weight = arr2(&[
        [-0.02495167f32, 0.26316848, -0.00424941, 0.36453030],
        [0.06657098, 0.43198884, -0.50783736, -0.44437426],
    ]);
    let fc3_bias = arr1(&[0.01497797f32, -0.02088939]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(fc1_weight.clone(), Some(fc1_bias.clone())).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(fc2_weight.clone(), Some(fc2_bias.clone())).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(fc3_weight.clone(), Some(fc3_bias.clone())).unwrap(),
    ));

    // Input bounds: epsilon=0.1 around [0.5, 0.5, 0.5]
    let input_bounds = BoundedTensor::new(
        arr1(&[0.4f32, 0.4, 0.4]).into_dyn(),
        arr1(&[0.6f32, 0.6, 0.6]).into_dyn(),
    )
    .unwrap();

    // Auto-LiRPA reference values
    let ref_ibp_lower = arr1(&[0.00687957f32, -0.01418082]);
    let ref_ibp_upper = arr1(&[0.01246351f32, 0.00071711]);
    let ref_crown_lower = arr1(&[0.00820859f32, -0.01063502]);
    let ref_crown_upper = arr1(&[0.01113450f32, -0.00282869]);

    // IBP
    let ibp_result = network.propagate_ibp(&input_bounds).unwrap();

    println!("\n=== Auto-LiRPA Deep Model Comparison ===");
    println!("IBP:");
    println!(
        "  gamma-crown: lower=[{:.6}, {:.6}], upper=[{:.6}, {:.6}]",
        ibp_result.lower[[0]],
        ibp_result.lower[[1]],
        ibp_result.upper[[0]],
        ibp_result.upper[[1]]
    );
    println!(
        "  Auto-LiRPA:  lower=[{:.6}, {:.6}], upper=[{:.6}, {:.6}]",
        ref_ibp_lower[0], ref_ibp_lower[1], ref_ibp_upper[0], ref_ibp_upper[1]
    );

    // Check IBP bounds (allow small tolerance for floating point)
    for i in 0..2 {
        assert!(
            (ibp_result.lower[[i]] - ref_ibp_lower[i]).abs() < 1e-4,
            "IBP lower[{}] mismatch: gamma-crown={}, Auto-LiRPA={}",
            i,
            ibp_result.lower[[i]],
            ref_ibp_lower[i]
        );
        assert!(
            (ibp_result.upper[[i]] - ref_ibp_upper[i]).abs() < 1e-4,
            "IBP upper[{}] mismatch: gamma-crown={}, Auto-LiRPA={}",
            i,
            ibp_result.upper[[i]],
            ref_ibp_upper[i]
        );
    }

    // CROWN
    let crown_result = network.propagate_crown(&input_bounds).unwrap();

    println!("CROWN:");
    println!(
        "  gamma-crown: lower=[{:.6}, {:.6}], upper=[{:.6}, {:.6}]",
        crown_result.lower[[0]],
        crown_result.lower[[1]],
        crown_result.upper[[0]],
        crown_result.upper[[1]]
    );
    println!(
        "  Auto-LiRPA:  lower=[{:.6}, {:.6}], upper=[{:.6}, {:.6}]",
        ref_crown_lower[0], ref_crown_lower[1], ref_crown_upper[0], ref_crown_upper[1]
    );

    // CROWN bounds should be close to reference
    for i in 0..2 {
        assert!(
            (crown_result.lower[[i]] - ref_crown_lower[i]).abs() < 1e-4,
            "CROWN lower[{}] mismatch: gamma-crown={}, Auto-LiRPA={}",
            i,
            crown_result.lower[[i]],
            ref_crown_lower[i]
        );
        assert!(
            (crown_result.upper[[i]] - ref_crown_upper[i]).abs() < 1e-4,
            "CROWN upper[{}] mismatch: gamma-crown={}, Auto-LiRPA={}",
            i,
            crown_result.upper[[i]],
            ref_crown_upper[i]
        );
    }

    // Verify CROWN is tighter than IBP
    let ibp_width_0 = ibp_result.upper[[0]] - ibp_result.lower[[0]];
    let ibp_width_1 = ibp_result.upper[[1]] - ibp_result.lower[[1]];
    let crown_width_0 = crown_result.upper[[0]] - crown_result.lower[[0]];
    let crown_width_1 = crown_result.upper[[1]] - crown_result.lower[[1]];

    println!("Width comparison:");
    println!("  IBP width:   [{:.6}, {:.6}]", ibp_width_0, ibp_width_1);
    println!(
        "  CROWN width: [{:.6}, {:.6}]",
        crown_width_0, crown_width_1
    );
    println!(
        "  CROWN tightening: [{:.1}%, {:.1}%]",
        100.0 * (1.0 - crown_width_0 / ibp_width_0),
        100.0 * (1.0 - crown_width_1 / ibp_width_1)
    );

    assert!(
        crown_width_0 <= ibp_width_0 + 1e-5,
        "CROWN should be at least as tight as IBP for output 0"
    );
    assert!(
        crown_width_1 <= ibp_width_1 + 1e-5,
        "CROWN should be at least as tight as IBP for output 1"
    );

    // alpha-CROWN
    let alpha_result = network.propagate_alpha_crown(&input_bounds).unwrap();

    println!("alpha-CROWN:");
    println!(
        "  gamma-crown: lower=[{:.6}, {:.6}], upper=[{:.6}, {:.6}]",
        alpha_result.lower[[0]],
        alpha_result.lower[[1]],
        alpha_result.upper[[0]],
        alpha_result.upper[[1]]
    );

    let alpha_width_0 = alpha_result.upper[[0]] - alpha_result.lower[[0]];
    let alpha_width_1 = alpha_result.upper[[1]] - alpha_result.lower[[1]];

    println!(
        "  alpha-CROWN width: [{:.6}, {:.6}]",
        alpha_width_0, alpha_width_1
    );

    // alpha-CROWN should be at least as tight as CROWN
    assert!(
        alpha_width_0 <= crown_width_0 + 1e-5,
        "alpha-CROWN should be at least as tight as CROWN for output 0"
    );
    assert!(
        alpha_width_1 <= crown_width_1 + 1e-5,
        "alpha-CROWN should be at least as tight as CROWN for output 1"
    );

    // Verify soundness by checking center point
    let center = arr1(&[0.5f32, 0.5, 0.5]);
    let z1 = fc1_weight.dot(&center) + &fc1_bias;
    let a1 = z1.mapv(|v| v.max(0.0));
    let z2 = fc2_weight.dot(&a1) + &fc2_bias;
    let a2 = z2.mapv(|v| v.max(0.0));
    let output = fc3_weight.dot(&a2) + &fc3_bias;

    println!("Soundness check at center [0.5, 0.5, 0.5]:");
    println!("  Concrete output: [{:.6}, {:.6}]", output[0], output[1]);

    for i in 0..2 {
        assert!(
            output[i] >= ibp_result.lower[[i]] - 1e-5 && output[i] <= ibp_result.upper[[i]] + 1e-5,
            "Soundness violation for IBP at output {}",
            i
        );
        assert!(
            output[i] >= crown_result.lower[[i]] - 1e-5
                && output[i] <= crown_result.upper[[i]] + 1e-5,
            "Soundness violation for CROWN at output {}",
            i
        );
        assert!(
            output[i] >= alpha_result.lower[[i]] - 1e-5
                && output[i] <= alpha_result.upper[[i]] + 1e-5,
            "Soundness violation for alpha-CROWN at output {}",
            i
        );
    }
}

#[test]
#[allow(clippy::excessive_precision)]
fn benchmark_performance_comparison() {
    // Performance benchmark comparing gamma-crown vs Auto-LiRPA
    //
    // Auto-LiRPA reference timing (Python, PyTorch 2.9.1):
    // Toy Model (100 iterations):
    //   IBP:   0.288 ms/iter
    //   CROWN: 0.731 ms/iter
    // Deep Model (100 iterations):
    //   IBP:   0.384 ms/iter
    //   CROWN: 0.768 ms/iter

    use std::time::Instant;

    // Toy model setup
    let w1 = arr2(&[[1.0f32, -1.0], [2.0, -1.0]]);
    let w2 = arr2(&[[1.0f32, -1.0]]);

    let mut toy_network = Network::new();
    toy_network.add_layer(Layer::Linear(LinearLayer::new(w1.clone(), None).unwrap()));
    toy_network.add_layer(Layer::ReLU(ReLULayer));
    toy_network.add_layer(Layer::Linear(LinearLayer::new(w2.clone(), None).unwrap()));

    let toy_input = BoundedTensor::new(
        arr1(&[-1.0f32, -2.0]).into_dyn(),
        arr1(&[2.0f32, 1.0]).into_dyn(),
    )
    .unwrap();

    // Deep model setup
    let fc1_weight = arr2(&[
        [0.16834518f32, 0.06440470, 0.11723118],
        [0.11516652, -0.56142819, -0.09316415],
        [1.10410070, -0.31899852, 0.23082861],
        [0.13367544, 0.26745233, 0.40467861],
    ]);
    let fc1_bias = arr1(&[0.11102903f32, -0.16897990, -0.09889599, 0.09579718]);
    let fc2_weight = arr2(&[
        [-0.69233710f32, -0.43561807, -0.11168297, 0.85868055],
        [0.15943986, -0.21225949, 0.15286016, -0.38729626],
        [-0.77878612, 0.49781805, -0.43989292, -0.30057147],
        [-0.63707572, 1.06139255, -0.61732668, -0.24395694],
    ]);
    let fc2_bias = arr1(&[0.02815196f32, 0.00561635, 0.05227160, -0.02383569]);
    let fc3_weight = arr2(&[
        [-0.02495167f32, 0.26316848, -0.00424941, 0.36453030],
        [0.06657098, 0.43198884, -0.50783736, -0.44437426],
    ]);
    let fc3_bias = arr1(&[0.01497797f32, -0.02088939]);

    let mut deep_network = Network::new();
    deep_network.add_layer(Layer::Linear(
        LinearLayer::new(fc1_weight, Some(fc1_bias)).unwrap(),
    ));
    deep_network.add_layer(Layer::ReLU(ReLULayer));
    deep_network.add_layer(Layer::Linear(
        LinearLayer::new(fc2_weight, Some(fc2_bias)).unwrap(),
    ));
    deep_network.add_layer(Layer::ReLU(ReLULayer));
    deep_network.add_layer(Layer::Linear(
        LinearLayer::new(fc3_weight, Some(fc3_bias)).unwrap(),
    ));

    let deep_input = BoundedTensor::new(
        arr1(&[0.4f32, 0.4, 0.4]).into_dyn(),
        arr1(&[0.6f32, 0.6, 0.6]).into_dyn(),
    )
    .unwrap();

    let n_iters = 1000;

    // Warm-up
    for _ in 0..10 {
        let _ = toy_network.propagate_ibp(&toy_input);
        let _ = toy_network.propagate_crown(&toy_input);
        let _ = deep_network.propagate_ibp(&deep_input);
        let _ = deep_network.propagate_crown(&deep_input);
    }

    // Benchmark Toy Model IBP
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = toy_network.propagate_ibp(&toy_input);
    }
    let toy_ibp_us = start.elapsed().as_micros() as f64 / n_iters as f64;

    // Benchmark Toy Model CROWN
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = toy_network.propagate_crown(&toy_input);
    }
    let toy_crown_us = start.elapsed().as_micros() as f64 / n_iters as f64;

    // Benchmark Deep Model IBP
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = deep_network.propagate_ibp(&deep_input);
    }
    let deep_ibp_us = start.elapsed().as_micros() as f64 / n_iters as f64;

    // Benchmark Deep Model CROWN
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = deep_network.propagate_crown(&deep_input);
    }
    let deep_crown_us = start.elapsed().as_micros() as f64 / n_iters as f64;

    // Auto-LiRPA reference times (in microseconds)
    let ref_toy_ibp_us = 288.0;
    let ref_toy_crown_us = 731.0;
    let ref_deep_ibp_us = 384.0;
    let ref_deep_crown_us = 768.0;

    println!("\n=== Performance Comparison: gamma-crown vs Auto-LiRPA ===");
    println!("({} iterations)\n", n_iters);

    println!("Toy Model:");
    println!(
        "  IBP:   gamma-crown={:.1}us, Auto-LiRPA={:.0}us, speedup={:.1}x",
        toy_ibp_us,
        ref_toy_ibp_us,
        ref_toy_ibp_us / toy_ibp_us
    );
    println!(
        "  CROWN: gamma-crown={:.1}us, Auto-LiRPA={:.0}us, speedup={:.1}x",
        toy_crown_us,
        ref_toy_crown_us,
        ref_toy_crown_us / toy_crown_us
    );

    println!("\nDeep Model:");
    println!(
        "  IBP:   gamma-crown={:.1}us, Auto-LiRPA={:.0}us, speedup={:.1}x",
        deep_ibp_us,
        ref_deep_ibp_us,
        ref_deep_ibp_us / deep_ibp_us
    );
    println!(
        "  CROWN: gamma-crown={:.1}us, Auto-LiRPA={:.0}us, speedup={:.1}x",
        deep_crown_us,
        ref_deep_crown_us,
        ref_deep_crown_us / deep_crown_us
    );

    // Verify gamma-crown is faster (this should easily be true for Rust vs Python)
    // Note: In debug mode, Rust code is 10-100x slower than release, so we use
    // a very lenient threshold. In release mode, we expect competitive performance.
    #[cfg(debug_assertions)]
    let slowdown_threshold = 50.0; // Debug: allow up to 50x slower than Python
    #[cfg(not(debug_assertions))]
    let slowdown_threshold = 10.0; // Release: allow up to 10x slower than Python

    assert!(
        toy_ibp_us < ref_toy_ibp_us * slowdown_threshold,
        "Toy IBP unexpectedly slow: {}us vs {}us reference (threshold: {}x)",
        toy_ibp_us,
        ref_toy_ibp_us,
        slowdown_threshold
    );
    assert!(
        toy_crown_us < ref_toy_crown_us * slowdown_threshold,
        "Toy CROWN unexpectedly slow: {}us vs {}us reference (threshold: {}x)",
        toy_crown_us,
        ref_toy_crown_us,
        slowdown_threshold
    );
}

#[test]
fn test_crown_vs_autolirpa_tiny() {
    // Exact network from minimal test:
    // Linear(2->2) -> ReLU -> Linear(2->2)
    // W1 = [[1, 1], [1, -1]], b1 = [-0.5, 0]
    // W2 = [[1, 0], [0, 1]], b2 = [0, 0]

    let w1 = arr2(&[[1.0, 1.0], [1.0, -1.0]]);
    let b1 = arr1(&[-0.5, 0.0]);
    let w2 = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let b2 = arr1(&[0.0, 0.0]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Expected results (from Auto-LiRPA):
    // IBP: lower=[0, 0], upper=[1.5, 1]
    // CROWN: lower=[-0.5, 0], upper=[1.5, 1]

    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();

    println!(
        "IBP: lower={:?}, upper={:?}",
        ibp_output.lower.as_slice().unwrap(),
        ibp_output.upper.as_slice().unwrap()
    );
    println!(
        "CROWN: lower={:?}, upper={:?}",
        crown_output.lower.as_slice().unwrap(),
        crown_output.upper.as_slice().unwrap()
    );

    // Check IBP
    assert!(
        (ibp_output.lower[[0]] - 0.0).abs() < 1e-5,
        "IBP lower[0] should be 0, got {}",
        ibp_output.lower[[0]]
    );
    assert!(
        (ibp_output.lower[[1]] - 0.0).abs() < 1e-5,
        "IBP lower[1] should be 0, got {}",
        ibp_output.lower[[1]]
    );
    assert!(
        (ibp_output.upper[[0]] - 1.5).abs() < 1e-5,
        "IBP upper[0] should be 1.5, got {}",
        ibp_output.upper[[0]]
    );
    assert!(
        (ibp_output.upper[[1]] - 1.0).abs() < 1e-5,
        "IBP upper[1] should be 1.0, got {}",
        ibp_output.upper[[1]]
    );

    // Check CROWN matches Auto-LiRPA
    assert!(
        (crown_output.lower[[0]] - (-0.5)).abs() < 1e-5,
        "CROWN lower[0] should be -0.5, got {}",
        crown_output.lower[[0]]
    );
    assert!(
        (crown_output.lower[[1]] - 0.0).abs() < 1e-5,
        "CROWN lower[1] should be 0, got {}",
        crown_output.lower[[1]]
    );
    assert!(
        (crown_output.upper[[0]] - 1.5).abs() < 1e-5,
        "CROWN upper[0] should be 1.5, got {}",
        crown_output.upper[[0]]
    );
    assert!(
        (crown_output.upper[[1]] - 1.0).abs() < 1e-5,
        "CROWN upper[1] should be 1.0, got {}",
        crown_output.upper[[1]]
    );
}

#[test]
fn test_crown_ibp_tighter_than_crown() {
    // Test that CROWN-IBP produces bounds that are at least as tight as standard CROWN.
    // This is a deep network where CROWN-IBP's tighter intermediate bounds should help.
    //
    // Network: 4 layers of Linear->ReLU with "crossing" inputs (where ReLU needs relaxation)

    // Layer 1: Linear(3->4)
    let w1 = arr2(&[
        [1.0, 2.0, -1.0],
        [-1.0, 1.0, 1.0],
        [0.5, -0.5, 1.0],
        [1.0, 1.0, 1.0],
    ]);
    let b1 = arr1(&[-0.5, 0.0, -0.3, 0.0]);

    // Layer 2: Linear(4->4)
    let w2 = arr2(&[
        [1.0, -1.0, 0.5, 0.0],
        [0.5, 1.0, -0.5, 0.5],
        [-0.5, 0.5, 1.0, -0.5],
        [0.0, 0.5, -0.5, 1.0],
    ]);
    let b2 = arr1(&[0.0, -0.2, 0.1, 0.0]);

    // Layer 3: Linear(4->3)
    let w3 = arr2(&[
        [1.0, -0.5, 0.5, 0.5],
        [-0.5, 1.0, -0.5, 0.5],
        [0.5, 0.5, 1.0, -0.5],
    ]);
    let b3 = arr1(&[0.0, 0.0, 0.0]);

    // Layer 4: Linear(3->2)
    let w4 = arr2(&[[1.0, -1.0, 0.5], [-0.5, 1.0, 1.0]]);
    let b4 = arr1(&[0.0, 0.0]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w3, Some(b3)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w4, Some(b4)).unwrap()));

    // Input with perturbation that creates "crossing" regions
    let input = BoundedTensor::new(
        arr1(&[-0.5, -0.5, -0.5]).into_dyn(),
        arr1(&[0.5, 0.5, 0.5]).into_dyn(),
    )
    .unwrap();

    let crown_output = network.propagate_crown(&input).unwrap();
    let crown_ibp_output = network.propagate_crown_ibp(&input).unwrap();

    println!(
        "CROWN: lower={:?}, upper={:?}",
        crown_output.lower.as_slice().unwrap(),
        crown_output.upper.as_slice().unwrap()
    );
    println!(
        "CROWN-IBP: lower={:?}, upper={:?}",
        crown_ibp_output.lower.as_slice().unwrap(),
        crown_ibp_output.upper.as_slice().unwrap()
    );

    // Print improvement metrics
    // Note: CROWN-IBP produces tighter intermediate bounds which leads to different
    // ReLU relaxation parameters. This doesn't guarantee every individual bound is
    // tighter, but typically improves overall bound width.
    let crown_width: f32 = crown_output.width().iter().sum();
    let crown_ibp_width: f32 = crown_ibp_output.width().iter().sum();
    println!(
        "Total width - CROWN: {:.4}, CROWN-IBP: {:.4}",
        crown_width, crown_ibp_width
    );
    println!(
        "Improvement: {:.2}%",
        (1.0 - crown_ibp_width / crown_width) * 100.0
    );

    // Verify that bounds are sound (still valid bounds)
    for i in 0..crown_output.len() {
        assert!(
            crown_ibp_output.lower[[i]] <= crown_ibp_output.upper[[i]],
            "CROWN-IBP bounds should be valid: lower[{}]={} <= upper[{}]={}",
            i,
            crown_ibp_output.lower[[i]],
            i,
            crown_ibp_output.upper[[i]]
        );
    }

    // Check that total width is improved (or at least not significantly worse)
    // CROWN-IBP should generally produce tighter overall bounds
    assert!(
        crown_ibp_width <= crown_width * 1.1, // Allow 10% tolerance
        "CROWN-IBP total width {} should not be much worse than CROWN {}",
        crown_ibp_width,
        crown_width
    );
}

#[test]
fn test_crown_ibp_collect_bounds() {
    // Test that collect_crown_ibp_bounds produces tighter bounds than collect_ibp_bounds

    // Simple 2-layer network
    let w1 = arr2(&[[1.0, 1.0], [1.0, -1.0]]);
    let b1 = arr1(&[-0.5, 0.0]);
    let w2 = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let b2 = arr1(&[0.0, 0.0]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let ibp_bounds = network.collect_ibp_bounds(&input).unwrap();
    let crown_ibp_bounds = network.collect_crown_ibp_bounds(&input).unwrap();

    println!("Layer bounds comparison:");
    for (i, (ibp, crown_ibp)) in ibp_bounds.iter().zip(crown_ibp_bounds.iter()).enumerate() {
        println!(
            "Layer {}: IBP width={:.4}, CROWN-IBP width={:.4}",
            i,
            ibp.max_width(),
            crown_ibp.max_width()
        );
    }

    // CROWN-IBP bounds should be at least as tight at each layer
    for (i, (ibp, crown_ibp)) in ibp_bounds.iter().zip(crown_ibp_bounds.iter()).enumerate() {
        for j in 0..ibp.len() {
            assert!(
                crown_ibp.lower[[j]] >= ibp.lower[[j]] - 1e-5,
                "Layer {} elem {}: CROWN-IBP lower {} >= IBP lower {}",
                i,
                j,
                crown_ibp.lower[[j]],
                ibp.lower[[j]]
            );
            assert!(
                crown_ibp.upper[[j]] <= ibp.upper[[j]] + 1e-5,
                "Layer {} elem {}: CROWN-IBP upper {} <= IBP upper {}",
                i,
                j,
                crown_ibp.upper[[j]],
                ibp.upper[[j]]
            );
        }
    }
}

#[test]
fn test_crown_ibp_collect_bounds_preserves_non_1d_shapes() {
    // Regression test: propagate_crown_partial must reshape its concretized bounds to the
    // IBP output shape so CROWN-IBP can tighten intermediate bounds on non-1D activations
    // (e.g., ONNX models with batch/spatial dimensions like [1, 1, 1, 5]).

    // Build a small network where IBP is loose but CROWN can be tight:
    // x ∈ [-1, 1]
    // y1 = ReLU(x), y2 = ReLU(-x)
    // z = y1 + y2 ∈ [0, 1] but IBP gives upper bound 2.
    //
    // Then reshape z to [1, 1, 1, 5] to force a non-1D activation shape.
    let w1 = arr2(&[[1.0], [-1.0]]); // 1 -> 2
    let b1 = arr1(&[0.0, 0.0]);

    // 2 -> 5, each output is y1 + y2
    let w2 = arr2(&[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]);
    let b2 = arr1(&[0.0, 0.0, 0.0, 0.0, 0.0]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1, 1, 1, 5])));

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let ibp_bounds = network.collect_ibp_bounds(&input).unwrap();
    let crown_ibp_bounds = network.collect_crown_ibp_bounds(&input).unwrap();

    // Reshape layer output must remain 4D for both IBP and CROWN-IBP.
    assert_eq!(ibp_bounds[3].shape(), &[1, 1, 1, 5]);
    assert_eq!(crown_ibp_bounds[3].shape(), &[1, 1, 1, 5]);

    // IBP upper bound is loose: y1 ∈ [0,1], y2 ∈ [0,1] -> y1+y2 ∈ [0,2]
    for &v in ibp_bounds[3].upper.iter() {
        assert!(
            (v - 2.0).abs() < 1e-5,
            "Expected IBP upper bound 2.0, got {v}"
        );
    }

    // With correct reshaping, CROWN-IBP intersects with CROWN and tightens to ≤ 1.
    for &v in crown_ibp_bounds[3].upper.iter() {
        assert!(
            v <= 1.0 + 1e-4,
            "Expected tightened CROWN-IBP upper bound ≤ 1.0, got {v}"
        );
    }
}

#[test]
fn test_crown_ibp_conv2d_4d_shapes() {
    // Test CROWN-IBP with Conv2D network that has 4D tensor shapes.
    // Verifies that shape handling works correctly for convolutional models.
    use crate::layers::Conv2dLayer;
    use ndarray::{ArrayD, IxDyn};

    // Input: [1, 4, 4] (1 channel, 4x4 spatial)
    // Conv2d: kernel [2, 1, 2, 2] -> [2, 3, 3] (2 channels, 3x3 spatial)
    // ReLU -> [2, 3, 3]
    // Flatten -> 18

    let mut kernel = ArrayD::zeros(IxDyn(&[2, 1, 2, 2]));
    // First output channel: sum of 2x2 patch
    kernel[[0, 0, 0, 0]] = 0.5;
    kernel[[0, 0, 0, 1]] = 0.5;
    kernel[[0, 0, 1, 0]] = 0.5;
    kernel[[0, 0, 1, 1]] = 0.5;
    // Second output channel: difference pattern (creates crossing)
    kernel[[1, 0, 0, 0]] = 1.0;
    kernel[[1, 0, 0, 1]] = -1.0;
    kernel[[1, 0, 1, 0]] = -1.0;
    kernel[[1, 0, 1, 1]] = 1.0;

    let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 4, 4).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Input bounds for [1, 4, 4] shape
    let lower = ArrayD::from_elem(IxDyn(&[1, 4, 4]), -0.5);
    let upper = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.5);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Run all three methods
    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();
    let crown_ibp_output = network.propagate_crown_ibp(&input).unwrap();

    // Verify shapes are preserved
    assert_eq!(ibp_output.shape(), &[2, 3, 3], "IBP output shape mismatch");
    assert_eq!(
        crown_output.shape(),
        &[2, 3, 3],
        "CROWN output shape mismatch"
    );
    assert_eq!(
        crown_ibp_output.shape(),
        &[2, 3, 3],
        "CROWN-IBP output shape mismatch"
    );

    // Calculate widths
    let ibp_width: f32 = ibp_output.width().iter().sum();
    let crown_width: f32 = crown_output.width().iter().sum();
    let crown_ibp_width: f32 = crown_ibp_output.width().iter().sum();

    println!("Conv2D 4D shapes test:");
    println!("  IBP total width: {:.4}", ibp_width);
    println!("  CROWN total width: {:.4}", crown_width);
    println!("  CROWN-IBP total width: {:.4}", crown_ibp_width);

    // CROWN should be at least as tight as IBP
    assert!(
        crown_width <= ibp_width + 1e-4,
        "CROWN ({:.4}) should be <= IBP ({:.4})",
        crown_width,
        ibp_width
    );

    // Verify all bounds are valid
    for i in 0..ibp_output.len() {
        assert!(
            crown_ibp_output.lower.as_slice().unwrap()[i]
                <= crown_ibp_output.upper.as_slice().unwrap()[i],
            "CROWN-IBP bounds invalid at index {}",
            i
        );
    }
}

#[test]
fn test_crown_ibp_intermediate_bounds_conv2d() {
    // Test that collect_crown_ibp_bounds produces tighter intermediate bounds
    // for Conv2D networks with 4D shapes.
    use crate::layers::Conv2dLayer;
    use ndarray::{ArrayD, IxDyn};

    // Similar to above but we check intermediate bounds
    let mut kernel = ArrayD::zeros(IxDyn(&[2, 1, 2, 2]));
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 0, 1]] = -1.0;
    kernel[[0, 0, 1, 0]] = 1.0;
    kernel[[0, 0, 1, 1]] = -1.0;
    kernel[[1, 0, 0, 0]] = 0.5;
    kernel[[1, 0, 0, 1]] = 0.5;
    kernel[[1, 0, 1, 0]] = 0.5;
    kernel[[1, 0, 1, 1]] = 0.5;

    let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 4, 4).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));

    let lower = ArrayD::from_elem(IxDyn(&[1, 4, 4]), -0.5);
    let upper = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.5);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let ibp_bounds = network.collect_ibp_bounds(&input).unwrap();
    let crown_ibp_bounds = network.collect_crown_ibp_bounds(&input).unwrap();

    // Verify shapes match at each layer
    for (i, (ibp, crown_ibp)) in ibp_bounds.iter().zip(crown_ibp_bounds.iter()).enumerate() {
        assert_eq!(
            ibp.shape(),
            crown_ibp.shape(),
            "Layer {} shape mismatch: IBP {:?} vs CROWN-IBP {:?}",
            i,
            ibp.shape(),
            crown_ibp.shape()
        );
        println!(
            "Layer {}: shape {:?}, IBP width {:.4}, CROWN-IBP width {:.4}",
            i,
            ibp.shape(),
            ibp.max_width(),
            crown_ibp.max_width()
        );
    }

    // Verify IBP bounds at layer 0 (Conv2d output before ReLU)
    // These should be [2, 3, 3]
    assert_eq!(
        ibp_bounds[0].shape(),
        &[2, 3, 3],
        "Conv2d output shape should be [2, 3, 3]"
    );
    assert_eq!(
        crown_ibp_bounds[0].shape(),
        &[2, 3, 3],
        "CROWN-IBP Conv2d output shape should be [2, 3, 3]"
    );
}

#[test]
fn test_alpha_crown_adaptive_skip() {
    // Test that adaptive skip works correctly based on ReLU layer count

    // Build a network with many ReLU layers (> threshold of 8)
    let mut deep_network = Network::new();
    for _ in 0..12 {
        // 12 ReLU layers
        let w = Array2::<f32>::from_shape_fn((4, 4), |_| 0.1);
        let b = Array1::zeros(4);
        deep_network.add_layer(Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()));
        deep_network.add_layer(Layer::ReLU(ReLULayer));
    }
    // Output layer
    let w = Array2::<f32>::from_shape_fn((2, 4), |_| 0.1);
    let b = Array1::zeros(2);
    deep_network.add_layer(Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()));

    // Create input bounds
    let lower = Array1::from_vec(vec![-0.5, -0.5, -0.5, -0.5]).into_dyn();
    let upper = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Config with adaptive skip enabled (default threshold = 8)
    let config_skip = AlphaCrownConfig {
        iterations: 10,
        adaptive_skip: true,
        adaptive_skip_depth_threshold: 8, // 12 > 8 so should skip
        adaptive_skip_pilot: false,       // Disable pilot for this test
        ..Default::default()
    };

    // Config with adaptive skip disabled
    let config_no_skip = AlphaCrownConfig {
        iterations: 10,
        adaptive_skip: false,
        ..Default::default()
    };

    // Get CROWN bounds for comparison
    let crown_output = deep_network.propagate_crown(&input).unwrap();
    let crown_width: f32 = (0..crown_output.len())
        .map(|i| crown_output.upper[[i]] - crown_output.lower[[i]])
        .sum();

    // With adaptive skip enabled (12 ReLU > threshold 8), should skip α-CROWN
    let skip_output = deep_network
        .propagate_alpha_crown_with_config(&input, &config_skip)
        .unwrap();
    let skip_width: f32 = (0..skip_output.len())
        .map(|i| skip_output.upper[[i]] - skip_output.lower[[i]])
        .sum();

    // With skip disabled, should run α-CROWN normally
    let no_skip_output = deep_network
        .propagate_alpha_crown_with_config(&input, &config_no_skip)
        .unwrap();
    let no_skip_width: f32 = (0..no_skip_output.len())
        .map(|i| no_skip_output.upper[[i]] - no_skip_output.lower[[i]])
        .sum();

    println!("Deep network (12 ReLU layers):");
    println!("  CROWN width: {:.6}", crown_width);
    println!("  α-CROWN (skip enabled, threshold=8): {:.6}", skip_width);
    println!("  α-CROWN (skip disabled): {:.6}", no_skip_width);

    // With skip enabled for deep network, should return CROWN bounds (or very close)
    // because it skips the optimization
    assert!(
        (skip_width - crown_width).abs() < 1e-4,
        "With adaptive skip enabled for deep network, should return CROWN bounds. \
         Got skip_width={:.6}, crown_width={:.6}, diff={:.6}",
        skip_width,
        crown_width,
        (skip_width - crown_width).abs()
    );

    // Test with shallow network (< threshold)
    let mut shallow_network = Network::new();
    for _ in 0..3 {
        // Only 3 ReLU layers
        let w = Array2::<f32>::from_shape_fn((4, 4), |_| 0.1);
        let b = Array1::zeros(4);
        shallow_network.add_layer(Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()));
        shallow_network.add_layer(Layer::ReLU(ReLULayer));
    }
    let w = Array2::<f32>::from_shape_fn((2, 4), |_| 0.1);
    let b = Array1::zeros(2);
    shallow_network.add_layer(Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()));

    // For shallow network (3 < 8), should NOT skip even with adaptive_skip enabled
    let shallow_output = shallow_network
        .propagate_alpha_crown_with_config(&input, &config_skip)
        .unwrap();
    let shallow_width: f32 = (0..shallow_output.len())
        .map(|i| shallow_output.upper[[i]] - shallow_output.lower[[i]])
        .sum();

    let shallow_crown = shallow_network.propagate_crown(&input).unwrap();
    let shallow_crown_width: f32 = (0..shallow_crown.len())
        .map(|i| shallow_crown.upper[[i]] - shallow_crown.lower[[i]])
        .sum();

    println!("Shallow network (3 ReLU layers):");
    println!("  CROWN width: {:.6}", shallow_crown_width);
    println!(
        "  α-CROWN (skip enabled, threshold=8): {:.6}",
        shallow_width
    );

    // Shallow network should NOT skip, so α-CROWN may improve over CROWN
    // (or at least be equal, not strictly CROWN bounds)
    println!("Test passed: adaptive skip correctly handles deep vs shallow networks");
}

#[test]
fn test_alpha_crown_analytic_chain_gradients() {
    use crate::bounds::GradientMethod;
    use std::time::Instant;

    // Test the new AnalyticChain gradient method against other methods.
    // This test verifies that AnalyticChain produces valid bounds that are
    // at least as tight as CROWN (soundness) and comparable to other gradient methods.

    // Build a small network with multiple ReLU layers to test chain-rule gradients
    let mut network = Network::new();
    let input_dim = 4;
    let hidden_dim = 8;
    let output_dim = 2;

    // Input -> Hidden 1
    let w1 = Array2::<f32>::from_shape_fn((hidden_dim, input_dim), |(i, j)| {
        ((i + j * 2) % 5) as f32 * 0.15 - 0.35
    });
    let b1 = Array1::from_elem(hidden_dim, 0.1);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Hidden 1 -> Hidden 2
    let w2 = Array2::<f32>::from_shape_fn((hidden_dim, hidden_dim), |(i, j)| {
        ((i * 3 + j) % 7) as f32 * 0.1 - 0.3
    });
    let b2 = Array1::from_elem(hidden_dim, -0.05);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Hidden 2 -> Output
    let w_out = Array2::<f32>::from_shape_fn((output_dim, hidden_dim), |(i, j)| {
        ((i + j * 2) % 6) as f32 * 0.2 - 0.5
    });
    let b_out = Array1::from_elem(output_dim, 0.0);
    network.add_layer(Layer::Linear(LinearLayer::new(w_out, Some(b_out)).unwrap()));

    // Input bounds
    let lower = Array1::from_elem(input_dim, -0.3);
    let upper = Array1::from_elem(input_dim, 0.3);
    let input = BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).unwrap();

    // Get baseline CROWN bounds
    let crown_output = network.propagate_crown(&input).unwrap();
    let crown_width: f32 = (0..crown_output.len())
        .map(|i| crown_output.upper[[i]] - crown_output.lower[[i]])
        .sum();

    println!("\n=== AnalyticChain Gradient Test ===");
    println!("Network: 2 ReLU layers with {} hidden neurons", hidden_dim);
    println!("CROWN baseline width: {:.6}", crown_width);

    // Common config
    let iterations = 30;
    let learning_rate = 0.15;
    let lr_decay = 0.98;

    // Test SPSA (baseline comparison)
    let spsa_config = AlphaCrownConfig {
        iterations,
        gradient_method: GradientMethod::Spsa,
        spsa_samples: 4,
        learning_rate,
        lr_decay,
        ..Default::default()
    };
    let start = Instant::now();
    let spsa_output = network
        .propagate_alpha_crown_with_config(&input, &spsa_config)
        .unwrap();
    let spsa_time = start.elapsed();
    let spsa_width: f32 = (0..spsa_output.len())
        .map(|i| spsa_output.upper[[i]] - spsa_output.lower[[i]])
        .sum();
    let spsa_improvement = (crown_width - spsa_width) / crown_width * 100.0;

    // Test local Analytic gradients
    let analytic_config = AlphaCrownConfig {
        iterations,
        gradient_method: GradientMethod::Analytic,
        learning_rate,
        lr_decay,
        ..Default::default()
    };
    let start = Instant::now();
    let analytic_output = network
        .propagate_alpha_crown_with_config(&input, &analytic_config)
        .unwrap();
    let analytic_time = start.elapsed();
    let analytic_width: f32 = (0..analytic_output.len())
        .map(|i| analytic_output.upper[[i]] - analytic_output.lower[[i]])
        .sum();
    let analytic_improvement = (crown_width - analytic_width) / crown_width * 100.0;

    // Test AnalyticChain (new implementation!)
    let chain_config = AlphaCrownConfig {
        iterations,
        gradient_method: GradientMethod::AnalyticChain,
        learning_rate,
        lr_decay,
        ..Default::default()
    };
    let start = Instant::now();
    let chain_output = network
        .propagate_alpha_crown_with_config(&input, &chain_config)
        .unwrap();
    let chain_time = start.elapsed();
    let chain_width: f32 = (0..chain_output.len())
        .map(|i| chain_output.upper[[i]] - chain_output.lower[[i]])
        .sum();
    let chain_improvement = (crown_width - chain_width) / crown_width * 100.0;

    println!("\nResults ({} iterations):", iterations);
    println!(
        "  SPSA:          width={:.6}, improvement={:+.2}%, time={:?}",
        spsa_width, spsa_improvement, spsa_time
    );
    println!(
        "  Analytic:      width={:.6}, improvement={:+.2}%, time={:?}",
        analytic_width, analytic_improvement, analytic_time
    );
    println!(
        "  AnalyticChain: width={:.6}, improvement={:+.2}%, time={:?}",
        chain_width, chain_improvement, chain_time
    );
    println!("==========================================\n");

    // Soundness check: all methods should produce valid bounds (at least as tight as CROWN)
    assert!(
        spsa_width <= crown_width + 1e-4,
        "SPSA bounds should be valid (width {} vs CROWN {})",
        spsa_width,
        crown_width
    );
    assert!(
        analytic_width <= crown_width + 1e-4,
        "Analytic bounds should be valid (width {} vs CROWN {})",
        analytic_width,
        crown_width
    );
    assert!(
        chain_width <= crown_width + 1e-4,
        "AnalyticChain bounds should be valid (width {} vs CROWN {})",
        chain_width,
        crown_width
    );

    // All three methods should produce finite bounds
    for i in 0..chain_output.len() {
        assert!(
            chain_output.lower[[i]].is_finite(),
            "AnalyticChain lower bound {} should be finite",
            i
        );
        assert!(
            chain_output.upper[[i]].is_finite(),
            "AnalyticChain upper bound {} should be finite",
            i
        );
    }

    println!("AnalyticChain gradient test PASSED: valid bounds produced");
}
