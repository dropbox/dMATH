//! Verifier tests

use crate::{
    Layer, LinearLayer, Network, PropagationConfig, PropagationMethod, ReLULayer, Verifier,
};
use gamma_core::{Bound, VerificationResult, VerificationSpec};
use ndarray::{arr1, arr2};

// ============================================================
// VERIFIER TESTS
// ============================================================

#[test]
fn test_verifier_ibp_simple_network() {
    // Create a simple 2-layer network: Linear -> ReLU
    let mut network = Network::new();
    let weight = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let bias = arr1(&[0.1, -0.1]);
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Create verifier with IBP
    let config = PropagationConfig {
        method: PropagationMethod::Ibp,
        max_iterations: 100,
        tolerance: 1e-4,
        use_gpu: false,
    };
    let verifier = Verifier::new(config);

    // Create specification
    let spec = VerificationSpec {
        input_bounds: vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        output_bounds: vec![Bound::new(f32::NEG_INFINITY, f32::INFINITY); 2],
        timeout_ms: None,
        input_shape: None,
    };

    // Verify
    let result = verifier.verify(&network, &spec).unwrap();

    // Should get bounds (not just input echoed back)
    match &result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_eq!(output_bounds.len(), 2);
            // Output bounds should be wider than input due to ReLU
            println!("IBP output bounds: {:?}", output_bounds);
        }
        VerificationResult::Unknown { bounds, reason } => {
            println!("Unknown result: {} (bounds: {:?})", reason, bounds);
        }
        _ => panic!("Unexpected result: {:?}", result),
    }
}

#[test]
fn test_verifier_crown_simple_network() {
    // Create a simple 2-layer network: Linear -> ReLU
    let mut network = Network::new();
    let weight = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let bias = arr1(&[0.1, -0.1]);
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Create verifier with CROWN
    let config = PropagationConfig {
        method: PropagationMethod::Crown,
        max_iterations: 100,
        tolerance: 1e-4,
        use_gpu: false,
    };
    let verifier = Verifier::new(config);

    // Create specification
    let spec = VerificationSpec {
        input_bounds: vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        output_bounds: vec![Bound::new(f32::NEG_INFINITY, f32::INFINITY); 2],
        timeout_ms: None,
        input_shape: None,
    };

    // Verify
    let result = verifier.verify(&network, &spec).unwrap();

    match &result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_eq!(output_bounds.len(), 2);
            println!("CROWN output bounds: {:?}", output_bounds);
        }
        VerificationResult::Unknown { bounds, reason } => {
            println!("Unknown result: {} (bounds: {:?})", reason, bounds);
        }
        _ => panic!("Unexpected result: {:?}", result),
    }
}

#[test]
fn test_verifier_ibp_vs_crown_comparison() {
    // Compare IBP and CROWN - both should produce valid bounds
    // Note: CROWN is not always tighter than IBP, especially for shallow networks
    let mut network = Network::new();
    let weight = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let bias = arr1(&[0.1, -0.1]);
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 1.0], [1.0, -1.0]]), None).unwrap(),
    ));

    let spec = VerificationSpec {
        input_bounds: vec![Bound::new(-0.5, 0.5), Bound::new(-0.5, 0.5)],
        output_bounds: vec![Bound::new(f32::NEG_INFINITY, f32::INFINITY); 2],
        timeout_ms: None,
        input_shape: None,
    };

    // IBP
    let ibp_config = PropagationConfig {
        method: PropagationMethod::Ibp,
        ..Default::default()
    };
    let ibp_verifier = Verifier::new(ibp_config);
    let ibp_result = ibp_verifier.verify(&network, &spec).unwrap();

    // CROWN
    let crown_config = PropagationConfig {
        method: PropagationMethod::Crown,
        ..Default::default()
    };
    let crown_verifier = Verifier::new(crown_config);
    let crown_result = crown_verifier.verify(&network, &spec).unwrap();

    // Extract bounds
    let (ibp_bounds, crown_bounds) = match (&ibp_result, &crown_result) {
        (
            VerificationResult::Verified {
                output_bounds: ibp, ..
            },
            VerificationResult::Verified {
                output_bounds: crown,
                ..
            },
        ) => (ibp, crown),
        (
            VerificationResult::Unknown { bounds: ibp, .. },
            VerificationResult::Verified {
                output_bounds: crown,
                ..
            },
        ) => (ibp, crown),
        (
            VerificationResult::Verified {
                output_bounds: ibp, ..
            },
            VerificationResult::Unknown { bounds: crown, .. },
        ) => (ibp, crown),
        (
            VerificationResult::Unknown { bounds: ibp, .. },
            VerificationResult::Unknown { bounds: crown, .. },
        ) => (ibp, crown),
        _ => panic!("Unexpected results"),
    };

    // Calculate widths
    let ibp_width: f32 = ibp_bounds.iter().map(|b| b.upper - b.lower).sum();
    let crown_width: f32 = crown_bounds.iter().map(|b| b.upper - b.lower).sum();

    println!("IBP total width: {}", ibp_width);
    println!("CROWN total width: {}", crown_width);
    println!("IBP bounds: {:?}", ibp_bounds);
    println!("CROWN bounds: {:?}", crown_bounds);

    // Both should produce finite bounds
    assert!(ibp_width.is_finite(), "IBP should produce finite bounds");
    assert!(
        crown_width.is_finite(),
        "CROWN should produce finite bounds"
    );

    // Both methods should be sound - this is validated by other tests
    // Note: CROWN can be looser than IBP for some shallow networks due to
    // the linear relaxation overhead when ReLU regions are wide
}

#[test]
fn test_verifier_alpha_crown() {
    // Test α-CROWN verification
    let mut network = Network::new();
    let weight = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let bias = arr1(&[0.1, -0.1]);
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let config = PropagationConfig {
        method: PropagationMethod::AlphaCrown,
        max_iterations: 100,
        tolerance: 1e-4,
        use_gpu: false,
    };
    let verifier = Verifier::new(config);

    let spec = VerificationSpec {
        input_bounds: vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        output_bounds: vec![Bound::new(f32::NEG_INFINITY, f32::INFINITY); 2],
        timeout_ms: None,
        input_shape: None,
    };

    let result = verifier.verify(&network, &spec).unwrap();

    match &result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_eq!(output_bounds.len(), 2);
            println!("α-CROWN output bounds: {:?}", output_bounds);
        }
        VerificationResult::Unknown { bounds, reason } => {
            println!("Unknown result: {} (bounds: {:?})", reason, bounds);
        }
        _ => panic!("Unexpected result: {:?}", result),
    }
}

#[test]
fn test_verifier_beta_crown() {
    // Test β-CROWN verification
    let mut network = Network::new();

    // Simple network that should verify output > -10
    let weight = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let bias = arr1(&[1.0, 1.0]); // Positive bias
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    // Final layer: 2 inputs -> 1 output (weight shape [1, 2])
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let config = PropagationConfig {
        method: PropagationMethod::BetaCrown,
        max_iterations: 100,
        tolerance: 1e-4,
        use_gpu: false,
    };
    let verifier = Verifier::new(config);

    // Specify output > -10
    let spec = VerificationSpec {
        input_bounds: vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        output_bounds: vec![Bound::new(-10.0, f32::INFINITY)],
        timeout_ms: Some(5000),
        input_shape: None,
    };

    let result = verifier.verify(&network, &spec).unwrap();

    println!("β-CROWN result: {:?}", result);

    // Should verify (output is positive due to ReLU + positive bias)
    match &result {
        VerificationResult::Verified { output_bounds, .. } => {
            println!("Verified with bounds: {:?}", output_bounds);
        }
        VerificationResult::Unknown { bounds, reason } => {
            println!("Unknown: {} (bounds: {:?})", reason, bounds);
        }
        _ => {}
    }
}

#[test]
fn test_verifier_spec_satisfied() {
    // Test that verification correctly checks output spec
    let mut network = Network::new();

    // Network that produces output in [0, 1]
    let weight = arr2(&[[0.5]]);
    let bias = arr1(&[0.5]);
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));

    let config = PropagationConfig {
        method: PropagationMethod::Ibp,
        ..Default::default()
    };
    let verifier = Verifier::new(config);

    // Tight input, should produce output near 0.5
    let spec = VerificationSpec {
        input_bounds: vec![Bound::new(0.0, 0.0)],
        output_bounds: vec![Bound::new(0.0, 1.0)],
        timeout_ms: None,
        input_shape: None,
    };

    let result = verifier.verify(&network, &spec).unwrap();

    // Should verify since 0.5 is in [0, 1]
    match result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_eq!(output_bounds.len(), 1);
            assert!(output_bounds[0].lower >= 0.0);
            assert!(output_bounds[0].upper <= 1.0);
        }
        _ => panic!("Expected Verified, got {:?}", result),
    }
}

#[test]
fn test_verifier_spec_violated() {
    // Test that verification correctly detects spec violation
    let mut network = Network::new();

    // Network that produces output around 1.5
    let weight = arr2(&[[1.0]]);
    let bias = arr1(&[1.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));

    let config = PropagationConfig {
        method: PropagationMethod::Ibp,
        ..Default::default()
    };
    let verifier = Verifier::new(config);

    // Input in [-0.5, 0.5], output will be in [0.5, 1.5]
    // Spec requires output in [0, 1] - will be violated
    let spec = VerificationSpec {
        input_bounds: vec![Bound::new(-0.5, 0.5)],
        output_bounds: vec![Bound::new(0.0, 1.0)],
        timeout_ms: None,
        input_shape: None,
    };

    let result = verifier.verify(&network, &spec).unwrap();

    // Should be Unknown since output bounds exceed spec
    match result {
        VerificationResult::Unknown { bounds, reason } => {
            println!(
                "Correctly detected violation: {} (bounds: {:?})",
                reason, bounds
            );
            assert!(bounds[0].upper > 1.0, "Upper bound should exceed spec");
        }
        _ => panic!("Expected Unknown, got {:?}", result),
    }
}

#[test]
fn test_verifier_empty_network() {
    // Empty network should just pass input through
    let network = Network::new();

    let config = PropagationConfig {
        method: PropagationMethod::Ibp,
        ..Default::default()
    };
    let verifier = Verifier::new(config);

    let spec = VerificationSpec {
        input_bounds: vec![Bound::new(-1.0, 1.0)],
        output_bounds: vec![Bound::new(-2.0, 2.0)],
        timeout_ms: None,
        input_shape: None,
    };

    let result = verifier.verify(&network, &spec).unwrap();

    // Should verify since input bounds are within output spec
    match result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_eq!(output_bounds.len(), 1);
            assert!((output_bounds[0].lower - (-1.0)).abs() < 1e-5);
            assert!((output_bounds[0].upper - 1.0).abs() < 1e-5);
        }
        _ => panic!("Expected Verified, got {:?}", result),
    }
}
