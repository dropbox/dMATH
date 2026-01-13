//! Integration tests for neural network verification using gamma-smt + gamma-onnx.
//!
//! Tests the IntegratedVerifier on NNet benchmark models.
//!
//! Fast tests use minimal_relu.nnet (2->3->1 network).
//! Slow tests use acasxu_1_1.nnet (5->50x6->5 network) and are #[ignore].

use gamma_core::{Bound, VerificationResult};
use gamma_onnx::nnet::load_nnet;
use gamma_smt::{BoundMethod, IntegratedVerifier, IntegratedVerifierConfig};
use gamma_tensor::BoundedTensor;
use ndarray::{ArrayD, IxDyn};
use std::time::Instant;

/// Get path to test models directory.
fn test_models_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/models")
}

// =============================================================================
// Minimal ReLU Network Tests (Fast - 2->3->1 architecture)
// =============================================================================

#[test]
fn test_load_minimal_relu_network() {
    let model_path = test_models_dir().join("minimal_relu.nnet");
    if !model_path.exists() {
        eprintln!("Skipping: minimal_relu.nnet not found at {:?}", model_path);
        return;
    }

    let nnet = load_nnet(&model_path).expect("Failed to load minimal_relu.nnet");

    // Verify model structure: 2 layers, 2 inputs, 3 hidden, 1 output
    assert_eq!(nnet.num_layers, 2);
    assert_eq!(nnet.input_size, 2);
    assert_eq!(nnet.output_size, 1);
    assert_eq!(nnet.layer_sizes, vec![2, 3, 1]);

    // Convert to PropNetwork
    let network = nnet
        .to_prop_network()
        .expect("Failed to convert to PropNetwork");

    // 2 linear layers + 1 ReLU layer = 3 layers
    assert_eq!(network.layers.len(), 3);
}

#[test]
fn test_integrated_verification_minimal_relu() {
    let model_path = test_models_dir().join("minimal_relu.nnet");
    if !model_path.exists() {
        eprintln!("Skipping: minimal_relu.nnet not found at {:?}", model_path);
        return;
    }

    let nnet = load_nnet(&model_path).expect("Failed to load model");
    let network = nnet.to_prop_network().expect("Failed to convert");

    // Create input bounds
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    )
    .unwrap();

    // Run IBP to get output bounds
    let ibp_result = network.propagate_ibp(&input).expect("IBP failed");
    println!(
        "IBP output bounds: [{:.4}, {:.4}]",
        ibp_result.lower[[0]],
        ibp_result.upper[[0]]
    );

    // Create verifier and test with loose bounds
    let verifier = IntegratedVerifier::new();
    let loose_bounds = vec![Bound::new(-100.0, 100.0)];

    let start = Instant::now();
    let result = verifier.verify(&network, &input, &loose_bounds);
    let elapsed = start.elapsed();

    println!("SMT verification time: {:.3}s", elapsed.as_secs_f64());
    println!("Result: {:?}", result);

    assert!(result.is_ok(), "Verification failed: {:?}", result.err());
    assert!(
        result.unwrap().is_verified(),
        "Expected verified with loose bounds"
    );
}

#[test]
fn test_minimal_relu_violated_property() {
    let model_path = test_models_dir().join("minimal_relu.nnet");
    if !model_path.exists() {
        eprintln!("Skipping: minimal_relu.nnet not found");
        return;
    }

    let nnet = load_nnet(&model_path).expect("Failed to load model");
    let network = nnet.to_prop_network().expect("Failed to convert");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    )
    .unwrap();

    // Get IBP bounds
    let ibp_result = network.propagate_ibp(&input).expect("IBP failed");

    // Create impossible property: output must be far outside IBP bounds
    let verifier = IntegratedVerifier::new();
    let impossible_bounds = vec![Bound::new(
        ibp_result.upper[[0]] + 100.0,
        ibp_result.upper[[0]] + 200.0,
    )];

    let result = verifier.verify(&network, &input, &impossible_bounds);
    println!("Result for impossible property: {:?}", result);

    // Should be violated (SAT - found counterexample)
    assert!(result.is_ok(), "Verification error: {:?}", result.err());
    assert!(
        matches!(result.unwrap(), VerificationResult::Violated { .. }),
        "Expected violated for impossible property"
    );
}

#[test]
fn test_minimal_relu_tight_bounds() {
    let model_path = test_models_dir().join("minimal_relu.nnet");
    if !model_path.exists() {
        eprintln!("Skipping: minimal_relu.nnet not found");
        return;
    }

    let nnet = load_nnet(&model_path).expect("Failed to load model");
    let network = nnet.to_prop_network().expect("Failed to convert");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    )
    .unwrap();

    // Get IBP bounds with 1% margin
    let ibp_result = network.propagate_ibp(&input).expect("IBP failed");
    let width = ibp_result.upper[[0]] - ibp_result.lower[[0]];
    let margin = 0.01 * width;

    let verifier = IntegratedVerifier::new();
    let tight_bounds = vec![Bound::new(
        ibp_result.lower[[0]] - margin,
        ibp_result.upper[[0]] + margin,
    )];

    let start = Instant::now();
    let result = verifier.verify(&network, &input, &tight_bounds);
    let elapsed = start.elapsed();

    println!(
        "IBP bounds: [{:.4}, {:.4}]",
        ibp_result.lower[[0]],
        ibp_result.upper[[0]]
    );
    println!(
        "Tight bounds (1% margin): [{:.4}, {:.4}]",
        ibp_result.lower[[0]] - margin,
        ibp_result.upper[[0]] + margin
    );
    println!("SMT verification time: {:.3}s", elapsed.as_secs_f64());
    println!("Result: {:?}", result);

    assert!(result.is_ok(), "Verification error: {:?}", result.err());
    assert!(
        result.unwrap().is_verified(),
        "Expected verified with tight IBP-based bounds"
    );
}

#[test]
fn test_ibp_vs_crown_ibp_minimal() {
    let model_path = test_models_dir().join("minimal_relu.nnet");
    if !model_path.exists() {
        eprintln!("Skipping: minimal_relu.nnet not found");
        return;
    }

    let nnet = load_nnet(&model_path).expect("Failed to load model");
    let network = nnet.to_prop_network().expect("Failed to convert");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    )
    .unwrap();

    let loose_bounds = vec![Bound::new(-100.0, 100.0)];

    // IBP config
    let ibp_config = IntegratedVerifierConfig {
        bound_method: BoundMethod::Ibp,
        use_bigm: false,
        big_m: 1e6,
        ..Default::default()
    };
    let ibp_verifier = IntegratedVerifier::with_config(ibp_config);

    // CROWN-IBP config
    let crown_ibp_config = IntegratedVerifierConfig {
        bound_method: BoundMethod::CrownIbp,
        use_bigm: false,
        big_m: 1e6,
        ..Default::default()
    };
    let crown_ibp_verifier = IntegratedVerifier::with_config(crown_ibp_config);

    let start = Instant::now();
    let ibp_result = ibp_verifier.verify(&network, &input, &loose_bounds);
    let ibp_time = start.elapsed();

    let start = Instant::now();
    let crown_ibp_result = crown_ibp_verifier.verify(&network, &input, &loose_bounds);
    let crown_ibp_time = start.elapsed();

    println!("\n=== IBP vs CROWN-IBP (minimal network) ===");
    println!("IBP time: {:.3}s", ibp_time.as_secs_f64());
    println!("CROWN-IBP time: {:.3}s", crown_ibp_time.as_secs_f64());

    assert!(ibp_result.is_ok(), "IBP failed: {:?}", ibp_result);
    assert!(
        crown_ibp_result.is_ok(),
        "CROWN-IBP failed: {:?}",
        crown_ibp_result
    );
}

// =============================================================================
// ACAS-Xu Network Tests (Bound Propagation Only - Fast)
// =============================================================================

/// Create input bounds for ACAS-Xu Property 2.
fn acasxu_property2_input() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.6, -0.5, -0.5, 0.45, -0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.6798578, 0.5, 0.5, 0.5, -0.45]).unwrap(),
    )
    .unwrap()
}

/// Create a small perturbation input for faster testing.
fn acasxu_small_input() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.6, -0.01, -0.01, 0.475, -0.475]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.62, 0.01, 0.01, 0.48, -0.47]).unwrap(),
    )
    .unwrap()
}

#[test]
fn test_load_acasxu_to_prop_network() {
    let model_path = test_models_dir().join("acasxu_1_1.nnet");
    if !model_path.exists() {
        eprintln!("Skipping: ACAS-Xu model not found at {:?}", model_path);
        return;
    }

    let nnet = load_nnet(&model_path).expect("Failed to load ACAS-Xu model");

    // Verify model structure
    assert_eq!(nnet.num_layers, 7);
    assert_eq!(nnet.input_size, 5);
    assert_eq!(nnet.output_size, 5);
    assert_eq!(nnet.layer_sizes, vec![5, 50, 50, 50, 50, 50, 50, 5]);

    // Convert to PropNetwork
    let network = nnet
        .to_prop_network()
        .expect("Failed to convert to PropNetwork");

    // 7 linear layers + 6 ReLU layers = 13 layers
    assert_eq!(network.layers.len(), 13);

    // Test IBP propagation
    let input = acasxu_small_input();
    let result = network.propagate_ibp(&input).expect("IBP failed");

    assert_eq!(result.shape(), &[5]);
    for (l, u) in result.lower.iter().zip(result.upper.iter()) {
        assert!(l <= u, "Invalid bounds: {} > {}", l, u);
    }
}

#[test]
fn test_acasxu_bound_propagation() {
    let model_path = test_models_dir().join("acasxu_1_1.nnet");
    if !model_path.exists() {
        eprintln!("Skipping: ACAS-Xu model not found");
        return;
    }

    let nnet = load_nnet(&model_path).expect("Failed to load model");
    let network = nnet.to_prop_network().expect("Failed to convert");
    let input = acasxu_property2_input();

    // Compare IBP vs CROWN
    let start = Instant::now();
    let ibp_result = network.propagate_ibp(&input).expect("IBP failed");
    let ibp_time = start.elapsed();

    let start = Instant::now();
    let crown_result = network.propagate_crown(&input).expect("CROWN failed");
    let crown_time = start.elapsed();

    println!("\n=== ACAS-Xu Property 2 Bound Propagation ===");
    println!("Input region: X0=[0.6, 0.68], X1=[-0.5, 0.5], X2=[-0.5, 0.5], X3=[0.45, 0.5], X4=[-0.5, -0.45]\n");

    println!("IBP bounds (time: {:.3}s):", ibp_time.as_secs_f64());
    for i in 0..5 {
        println!(
            "  Y{}: [{:.2}, {:.2}] (width: {:.2})",
            i,
            ibp_result.lower[[i]],
            ibp_result.upper[[i]],
            ibp_result.upper[[i]] - ibp_result.lower[[i]]
        );
    }

    println!("\nCROWN bounds (time: {:.3}s):", crown_time.as_secs_f64());
    for i in 0..5 {
        println!(
            "  Y{}: [{:.2}, {:.2}] (width: {:.2})",
            i,
            crown_result.lower[[i]],
            crown_result.upper[[i]],
            crown_result.upper[[i]] - crown_result.lower[[i]]
        );
    }

    // Compute improvement
    let ibp_total_width: f32 = (0..5)
        .map(|i| ibp_result.upper[[i]] - ibp_result.lower[[i]])
        .sum();
    let crown_total_width: f32 = (0..5)
        .map(|i| crown_result.upper[[i]] - crown_result.lower[[i]])
        .sum();

    println!("\nTotal bound widths:");
    println!("  IBP: {:.2}", ibp_total_width);
    println!("  CROWN: {:.2}", crown_total_width);
    println!(
        "  CROWN improvement: {:.1}%",
        (1.0 - crown_total_width / ibp_total_width) * 100.0
    );

    // CROWN should be at least as tight as IBP
    assert!(
        crown_total_width <= ibp_total_width * 1.01, // Allow 1% tolerance for numerical issues
        "CROWN should be tighter than IBP"
    );
}

#[test]
fn test_acasxu_property2_bounds_analysis() {
    let model_path = test_models_dir().join("acasxu_1_1.nnet");
    if !model_path.exists() {
        eprintln!("Skipping: ACAS-Xu model not found");
        return;
    }

    let nnet = load_nnet(&model_path).expect("Failed to load model");
    let network = nnet.to_prop_network().expect("Failed to convert");
    let input = acasxu_property2_input();

    let crown_result = network.propagate_crown(&input).expect("CROWN failed");

    // Property 2: "If input conditions hold, then COC is not maximal"
    // This means Y0 should NOT be >= all other Yi.
    // If upper(Y0) < lower(Yi) for any i, Property 2 is verified by bounds alone.
    println!("\n=== ACAS-Xu Property 2 Analysis ===");
    println!("Property: COC (Y0) should not be maximal");
    println!("Condition: Y0 < max(Y1, Y2, Y3, Y4) for all inputs in region\n");

    let y0_upper = crown_result.upper[[0]];
    println!("Y0 upper bound (CROWN): {:.2}", y0_upper);

    let mut can_verify_by_bounds = false;
    for i in 1..5 {
        let yi_lower = crown_result.lower[[i]];
        println!("Y{} lower bound (CROWN): {:.2}", i, yi_lower);
        if y0_upper < yi_lower {
            println!(
                "  -> Y0 < Y{} always holds -> Property 2 VERIFIED by bounds",
                i
            );
            can_verify_by_bounds = true;
            break;
        }
    }

    if !can_verify_by_bounds {
        println!("\nProperty 2 cannot be verified by bounds alone - SMT required");
    }
}

// =============================================================================
// ACAS-Xu SMT Tests (Slow - marked with #[ignore])
// =============================================================================

/// Slow test: Full SMT verification on ACAS-Xu.
/// Run with: cargo test --test acasxu_verification -- --ignored
#[test]
#[ignore]
fn test_acasxu_smt_verification_small() {
    let model_path = test_models_dir().join("acasxu_1_1.nnet");
    if !model_path.exists() {
        eprintln!("Skipping: ACAS-Xu model not found");
        return;
    }

    let nnet = load_nnet(&model_path).expect("Failed to load model");
    let network = nnet.to_prop_network().expect("Failed to convert");
    let input = acasxu_small_input();

    let ibp_result = network.propagate_ibp(&input).expect("IBP failed");
    println!("IBP output bounds:");
    for i in 0..5 {
        println!(
            "  Y{}: [{:.4}, {:.4}]",
            i,
            ibp_result.lower[[i]],
            ibp_result.upper[[i]]
        );
    }

    let verifier = IntegratedVerifier::new();
    let loose_bounds: Vec<Bound> = (0..5).map(|_| Bound::new(-1000.0, 1000.0)).collect();

    println!("\nStarting SMT verification (may take several minutes)...");
    let start = Instant::now();
    let result = verifier.verify(&network, &input, &loose_bounds);
    let elapsed = start.elapsed();

    println!("SMT verification time: {:.1}s", elapsed.as_secs_f64());
    println!("Result: {:?}", result);

    assert!(result.is_ok(), "Verification error: {:?}", result.err());
}

// =============================================================================
// Lazy Verifier Tests (CEGAR-style)
// =============================================================================

/// Test lazy verifier on minimal ReLU network.
#[test]
fn test_lazy_verifier_minimal_relu() {
    use gamma_smt::{LazyVerifier, LazyVerifierConfig};

    let model_path = test_models_dir().join("minimal_relu.nnet");
    if !model_path.exists() {
        eprintln!("Skipping: minimal_relu.nnet not found");
        return;
    }

    let nnet = load_nnet(&model_path).expect("Failed to load model");

    // Extract weights and biases from NNet (convert f32 to f64 for SMT)
    let weights: Vec<Vec<f64>> = nnet
        .weights
        .iter()
        .map(|w| w.iter().map(|&v| v as f64).collect())
        .collect();
    let biases: Vec<Vec<f64>> = nnet
        .biases
        .iter()
        .map(|b| b.iter().map(|&v| v as f64).collect())
        .collect();
    let layer_dims: Vec<usize> = nnet.layer_sizes.clone();

    // Compute intermediate bounds using the PropNetwork
    let network = nnet.to_prop_network().expect("Failed to convert");

    // Create tight input region
    let lower = ArrayD::from_elem(IxDyn(&[2]), -0.5f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 0.5f32);
    let input = BoundedTensor::new(lower, upper).expect("Failed to create input");

    // Get intermediate bounds via IBP
    let ibp_bounds = network.collect_ibp_bounds(&input).expect("IBP failed");

    // Convert to Bound format
    let input_bounds = vec![Bound::new(-0.5, 0.5), Bound::new(-0.5, 0.5)];

    // Intermediate bounds for the hidden layer (after first linear, before ReLU)
    // We need bounds on the pre-ReLU activations
    let intermediate_bounds: Vec<Vec<Bound>> = if ibp_bounds.len() >= 3 {
        // After first linear layer (index 1 in ibp_bounds)
        let hidden_bounds = &ibp_bounds[1];
        let hidden_len = hidden_bounds.lower.len();
        vec![(0..hidden_len)
            .map(|i| Bound::new(hidden_bounds.lower[[i]], hidden_bounds.upper[[i]]))
            .collect()]
    } else {
        vec![]
    };

    // Loose output bounds - should be verified
    let output_bounds = vec![Bound::new(-100.0f32, 100.0f32)];

    let config = LazyVerifierConfig {
        max_iterations: 50,
        big_m: 1e6,
        relu_tolerance: 1e-6,
        timeout_ms: Some(10_000),
    };
    let verifier = LazyVerifier::with_config(config);

    let start = Instant::now();
    let result = verifier
        .verify_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &output_bounds,
            &intermediate_bounds,
        )
        .expect("Lazy verification failed");
    let elapsed = start.elapsed();

    println!("\nLazy verifier result: {:?}", result);
    println!("Time: {:.3}ms", elapsed.as_secs_f64() * 1000.0);

    assert!(
        matches!(result, VerificationResult::Verified { .. }),
        "Expected verified for loose bounds"
    );
}

/// Test lazy verifier detects violation on minimal network.
#[test]
fn test_lazy_verifier_minimal_relu_violated() {
    use gamma_smt::{LazyVerifier, LazyVerifierConfig};

    let model_path = test_models_dir().join("minimal_relu.nnet");
    if !model_path.exists() {
        eprintln!("Skipping: minimal_relu.nnet not found");
        return;
    }

    let nnet = load_nnet(&model_path).expect("Failed to load model");

    let weights: Vec<Vec<f64>> = nnet
        .weights
        .iter()
        .map(|w| w.iter().map(|&v| v as f64).collect())
        .collect();
    let biases: Vec<Vec<f64>> = nnet
        .biases
        .iter()
        .map(|b| b.iter().map(|&v| v as f64).collect())
        .collect();
    let layer_dims: Vec<usize> = nnet.layer_sizes.clone();

    let network = nnet.to_prop_network().expect("Failed to convert");

    // Create input region
    let lower = ArrayD::from_elem(IxDyn(&[2]), -0.5f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 0.5f32);
    let input = BoundedTensor::new(lower, upper).expect("Failed to create input");

    // Get true output bounds via IBP
    let ibp_bounds = network.collect_ibp_bounds(&input).expect("IBP failed");
    let output_ibp = ibp_bounds.last().expect("No output bounds");

    let input_bounds = vec![Bound::new(-0.5, 0.5), Bound::new(-0.5, 0.5)];

    let intermediate_bounds: Vec<Vec<Bound>> = if ibp_bounds.len() >= 3 {
        let hidden_bounds = &ibp_bounds[1];
        let hidden_len = hidden_bounds.lower.len();
        vec![(0..hidden_len)
            .map(|i| Bound::new(hidden_bounds.lower[[i]], hidden_bounds.upper[[i]]))
            .collect()]
    } else {
        vec![]
    };

    // Tight bounds that exclude part of the output range -> should be violated
    // Use bounds that are clearly narrower than the true range
    let output_lower = output_ibp.lower[[0]];
    let output_upper = output_ibp.upper[[0]];
    let mid = (output_lower + output_upper) / 2.0;
    let tight_output_bounds = vec![Bound::new(mid - 0.01, mid + 0.01)];

    println!(
        "True output range: [{:.4}, {:.4}]",
        output_lower, output_upper
    );
    println!("Tight bounds: [{:.4}, {:.4}]", mid - 0.01, mid + 0.01);

    let config = LazyVerifierConfig::default();
    let verifier = LazyVerifier::with_config(config);

    let result = verifier
        .verify_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &tight_output_bounds,
            &intermediate_bounds,
        )
        .expect("Lazy verification failed");

    println!("Lazy verifier result: {:?}", result);

    assert!(
        matches!(result, VerificationResult::Violated { .. }),
        "Expected violation for tight bounds, got {:?}",
        result
    );
}
