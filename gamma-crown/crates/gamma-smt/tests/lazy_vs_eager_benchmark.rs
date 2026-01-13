//! Benchmark comparing lazy (CEGAR) vs eager SMT verification.
//!
//! This test compares:
//! - IntegratedVerifier (eager): Encodes all ReLU constraints upfront
//! - LazyVerifier (CEGAR): Starts with relaxation, refines on-demand
//!
//! Run with: cargo test -p gamma-smt --test lazy_vs_eager_benchmark -- --nocapture

use gamma_core::{Bound, VerificationResult};
use gamma_onnx::nnet::load_nnet;
use gamma_propagate::layers::{Layer, LinearLayer, ReLULayer};
use gamma_propagate::Network;
use gamma_smt::{
    BoundMethod, IntegratedVerifier, IntegratedVerifierConfig, LazyVerifier, LazyVerifierConfig,
};
use gamma_tensor::BoundedTensor;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use std::time::{Duration, Instant};

/// Get path to test models directory.
fn test_models_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/models")
}

/// Result of a single benchmark run.
#[derive(Debug)]
struct BenchmarkResult {
    elapsed: Duration,
    result: String,
    #[allow(dead_code)]
    refinements: Option<usize>,
}

impl BenchmarkResult {
    fn new(_name: &str, elapsed: Duration, result: &VerificationResult) -> Self {
        let result_str = match result {
            VerificationResult::Verified { .. } => "Verified".to_string(),
            VerificationResult::Violated { .. } => "Violated".to_string(),
            VerificationResult::Unknown { reason, .. } => format!("Unknown: {}", reason),
            VerificationResult::Timeout { .. } => "Timeout".to_string(),
        };
        Self {
            elapsed,
            result: result_str,
            refinements: None,
        }
    }
}

/// Create a multi-layer network for benchmarking.
/// Architecture: input_dim -> hidden_dim (x num_hidden) -> output_dim
fn create_benchmark_network(
    input_dim: usize,
    hidden_dim: usize,
    num_hidden: usize,
    output_dim: usize,
) -> Network {
    let mut network = Network::new();

    // First hidden layer
    let weight1 = Array2::from_shape_fn((hidden_dim, input_dim), |(i, j)| {
        let val = ((i * input_dim + j) % 17) as f32 / 17.0 - 0.5;
        val * 2.0 // Scale to [-1, 1]
    });
    let bias1 = Array1::from_elem(hidden_dim, 0.1f32);
    let layer1 = LinearLayer::new(weight1, Some(bias1)).unwrap();
    network.add_layer(Layer::Linear(layer1));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Additional hidden layers
    for h in 1..num_hidden {
        let weight = Array2::from_shape_fn((hidden_dim, hidden_dim), |(i, j)| {
            let val = ((h * hidden_dim * hidden_dim + i * hidden_dim + j) % 19) as f32 / 19.0 - 0.5;
            val * 2.0
        });
        let bias = Array1::from_elem(hidden_dim, 0.05f32);
        let layer = LinearLayer::new(weight, Some(bias)).unwrap();
        network.add_layer(Layer::Linear(layer));
        network.add_layer(Layer::ReLU(ReLULayer));
    }

    // Output layer
    let weight_out = Array2::from_shape_fn((output_dim, hidden_dim), |(i, j)| {
        let val = ((i * hidden_dim + j) % 13) as f32 / 13.0 - 0.5;
        val * 2.0
    });
    let bias_out = Array1::from_elem(output_dim, 0.0f32);
    let layer_out = LinearLayer::new(weight_out, Some(bias_out)).unwrap();
    network.add_layer(Layer::Linear(layer_out));

    network
}

/// Extract network parameters for lazy verifier.
fn extract_params(network: &Network) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>) {
    let mut weights: Vec<Vec<f64>> = Vec::new();
    let mut biases: Vec<Vec<f64>> = Vec::new();
    let mut layer_dims: Vec<usize> = Vec::new();
    let mut first_layer = true;

    for layer in &network.layers {
        if let Layer::Linear(linear) = layer {
            let (out_dim, in_dim) = linear.weight.dim();
            if first_layer {
                layer_dims.push(in_dim);
                first_layer = false;
            }
            layer_dims.push(out_dim);
            weights.push(linear.weight.iter().map(|&x| x as f64).collect());
            biases.push(
                linear
                    .bias
                    .as_ref()
                    .map(|b| b.iter().map(|&x| x as f64).collect())
                    .unwrap_or_else(|| vec![0.0; out_dim]),
            );
        }
    }

    (weights, biases, layer_dims)
}

/// Get intermediate bounds for lazy verifier.
fn get_intermediate_bounds(network: &Network, input: &BoundedTensor) -> Vec<Vec<Bound>> {
    let ibp_bounds = network.collect_ibp_bounds(input).expect("IBP failed");

    // Count linear layers followed by ReLU
    let num_linear = network
        .layers
        .iter()
        .filter(|l| matches!(l, Layer::Linear(_)))
        .count();

    let mut intermediate_bounds = Vec::new();
    let mut linear_idx = 0;

    for (i, layer) in network.layers.iter().enumerate() {
        if matches!(layer, Layer::Linear(_)) {
            let has_relu = network
                .layers
                .get(i + 1)
                .map(|l| matches!(l, Layer::ReLU(_)))
                .unwrap_or(false);

            // Record bounds for layers with ReLU (except last linear)
            if has_relu && linear_idx < num_linear - 1 && linear_idx < ibp_bounds.len() {
                let bounds = &ibp_bounds[linear_idx];
                let bound_vec: Vec<Bound> = bounds
                    .lower
                    .iter()
                    .zip(bounds.upper.iter())
                    .map(|(&l, &u)| Bound::new(l, u))
                    .collect();
                intermediate_bounds.push(bound_vec);
            }
            linear_idx += 1;
        }
    }

    intermediate_bounds
}

/// Run benchmark on a network with both verifiers.
fn benchmark_network(
    name: &str,
    network: &Network,
    input: &BoundedTensor,
    output_bounds: &[Bound],
) -> (BenchmarkResult, BenchmarkResult) {
    // Extract network parameters
    let (weights, biases, layer_dims) = extract_params(network);
    let input_bounds: Vec<Bound> = input
        .lower
        .iter()
        .zip(input.upper.iter())
        .map(|(&l, &u)| Bound::new(l, u))
        .collect();
    let intermediate_bounds = get_intermediate_bounds(network, input);

    // Eager verifier
    let eager_config = IntegratedVerifierConfig {
        bound_method: BoundMethod::Ibp,
        use_bigm: false,
        big_m: 1e6,
        ..Default::default()
    };
    let eager_verifier = IntegratedVerifier::with_config(eager_config);

    let start = Instant::now();
    let eager_result = eager_verifier
        .verify(network, input, output_bounds)
        .expect("Eager verification failed");
    let eager_elapsed = start.elapsed();
    let eager_bench =
        BenchmarkResult::new(&format!("{} (eager)", name), eager_elapsed, &eager_result);

    // Lazy verifier
    let lazy_config = LazyVerifierConfig {
        max_iterations: 100,
        big_m: 1e6,
        relu_tolerance: 1e-6,
        timeout_ms: Some(30_000),
    };
    let lazy_verifier = LazyVerifier::with_config(lazy_config);

    let start = Instant::now();
    let lazy_result = lazy_verifier
        .verify_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            output_bounds,
            &intermediate_bounds,
        )
        .expect("Lazy verification failed");
    let lazy_elapsed = start.elapsed();
    let mut lazy_bench =
        BenchmarkResult::new(&format!("{} (lazy)", name), lazy_elapsed, &lazy_result);

    // Note: We don't have access to refinement count from outside, could add if needed
    lazy_bench.refinements = Some(0);

    (eager_bench, lazy_bench)
}

// =============================================================================
// Benchmark Tests
// =============================================================================

#[test]
fn benchmark_minimal_relu() {
    let model_path = test_models_dir().join("minimal_relu.nnet");
    if !model_path.exists() {
        eprintln!("Skipping: minimal_relu.nnet not found at {:?}", model_path);
        return;
    }

    let nnet = load_nnet(&model_path).expect("Failed to load model");
    let network = nnet.to_prop_network().expect("Failed to convert");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    )
    .unwrap();

    let output_bounds = vec![Bound::new(-100.0, 100.0)];

    println!("\n=== Benchmark: minimal_relu.nnet (2->3->1) ===");
    let (eager, lazy) = benchmark_network("minimal_relu", &network, &input, &output_bounds);

    println!(
        "  Eager: {:.3}ms - {}",
        eager.elapsed.as_secs_f64() * 1000.0,
        eager.result
    );
    println!(
        "  Lazy:  {:.3}ms - {}",
        lazy.elapsed.as_secs_f64() * 1000.0,
        lazy.result
    );

    let speedup = eager.elapsed.as_secs_f64() / lazy.elapsed.as_secs_f64();
    println!("  Speedup: {:.2}x", speedup);
}

#[test]
fn benchmark_synthetic_small() {
    // 2 inputs -> 4 hidden -> 1 output (1 hidden layer)
    let network = create_benchmark_network(2, 4, 1, 1);

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -0.5f32),
        ArrayD::from_elem(IxDyn(&[2]), 0.5f32),
    )
    .unwrap();

    let output_bounds = vec![Bound::new(-100.0, 100.0)];

    println!("\n=== Benchmark: Synthetic 2->4->1 (1 hidden layer, 4 ReLUs) ===");
    let (eager, lazy) = benchmark_network("synthetic_small", &network, &input, &output_bounds);

    println!(
        "  Eager: {:.3}ms - {}",
        eager.elapsed.as_secs_f64() * 1000.0,
        eager.result
    );
    println!(
        "  Lazy:  {:.3}ms - {}",
        lazy.elapsed.as_secs_f64() * 1000.0,
        lazy.result
    );

    let speedup = eager.elapsed.as_secs_f64() / lazy.elapsed.as_secs_f64();
    println!("  Speedup: {:.2}x", speedup);
}

#[test]
#[ignore]
fn benchmark_synthetic_medium() {
    // 2 inputs -> 6 hidden x2 -> 1 output (2 hidden layers, 12 ReLUs)
    let network = create_benchmark_network(2, 6, 2, 1);

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -0.5f32),
        ArrayD::from_elem(IxDyn(&[2]), 0.5f32),
    )
    .unwrap();

    let output_bounds = vec![Bound::new(-100.0, 100.0)];

    println!("\n=== Benchmark: Synthetic 2->6->6->1 (2 hidden layers, 12 ReLUs) ===");
    let (eager, lazy) = benchmark_network("synthetic_medium", &network, &input, &output_bounds);

    println!(
        "  Eager: {:.3}ms - {}",
        eager.elapsed.as_secs_f64() * 1000.0,
        eager.result
    );
    println!(
        "  Lazy:  {:.3}ms - {}",
        lazy.elapsed.as_secs_f64() * 1000.0,
        lazy.result
    );

    let speedup = eager.elapsed.as_secs_f64() / lazy.elapsed.as_secs_f64();
    println!("  Speedup: {:.2}x", speedup);
}

#[test]
#[ignore]
fn benchmark_synthetic_larger() {
    // 3 inputs -> 8 hidden x2 -> 2 outputs (2 hidden layers, 16 ReLUs)
    let network = create_benchmark_network(3, 8, 2, 2);

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3]), -0.3f32),
        ArrayD::from_elem(IxDyn(&[3]), 0.3f32),
    )
    .unwrap();

    let output_bounds: Vec<Bound> = (0..2).map(|_| Bound::new(-100.0, 100.0)).collect();

    println!("\n=== Benchmark: Synthetic 3->8->8->2 (2 hidden layers, 16 ReLUs) ===");
    let (eager, lazy) = benchmark_network("synthetic_larger", &network, &input, &output_bounds);

    println!(
        "  Eager: {:.3}ms - {}",
        eager.elapsed.as_secs_f64() * 1000.0,
        eager.result
    );
    println!(
        "  Lazy:  {:.3}ms - {}",
        lazy.elapsed.as_secs_f64() * 1000.0,
        lazy.result
    );

    let speedup = eager.elapsed.as_secs_f64() / lazy.elapsed.as_secs_f64();
    println!("  Speedup: {:.2}x", speedup);
}

/// Test with tight bounds that may require refinement.
#[test]
fn benchmark_tight_bounds() {
    let network = create_benchmark_network(2, 4, 1, 1);

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -0.1f32),
        ArrayD::from_elem(IxDyn(&[2]), 0.1f32),
    )
    .unwrap();

    // Get IBP bounds first
    let ibp_result = network.propagate_ibp(&input).expect("IBP failed");

    // Create tight bounds that may require refinement
    // Use bounds 10% wider than IBP (should still verify)
    let output_bounds: Vec<Bound> = ibp_result
        .lower
        .iter()
        .zip(ibp_result.upper.iter())
        .map(|(&l, &u)| {
            let margin = (u - l) * 0.1;
            Bound::new(l - margin, u + margin)
        })
        .collect();

    println!("\n=== Benchmark: Tight bounds (may require refinement) ===");
    println!(
        "  IBP bounds: [{:.4}, {:.4}]",
        ibp_result.lower[[0]],
        ibp_result.upper[[0]]
    );

    let (eager, lazy) = benchmark_network("tight_bounds", &network, &input, &output_bounds);

    println!(
        "  Eager: {:.3}ms - {}",
        eager.elapsed.as_secs_f64() * 1000.0,
        eager.result
    );
    println!(
        "  Lazy:  {:.3}ms - {}",
        lazy.elapsed.as_secs_f64() * 1000.0,
        lazy.result
    );

    let speedup = eager.elapsed.as_secs_f64() / lazy.elapsed.as_secs_f64();
    println!("  Speedup: {:.2}x", speedup);
}

/// Run all benchmarks and print summary.
#[test]
#[ignore]
fn benchmark_summary() {
    println!("\n");
    println!("=================================================================");
    println!("          Lazy vs Eager SMT Verification Benchmark              ");
    println!("=================================================================");
    println!();
    println!("Comparing:");
    println!("  - Eager: Encodes all ReLU constraints upfront (triangle relaxation)");
    println!("  - Lazy (CEGAR): Starts with relaxation, refines violated neurons");
    println!();
    println!("Theoretical advantage of lazy:");
    println!("  - For networks with many uncertain ReLUs, only refines needed neurons");
    println!("  - Can verify properties without full ReLU encoding");
    println!("  - Trade-off: Multiple solver calls vs one large encoding");
    println!();

    // Minimal network
    let model_path = test_models_dir().join("minimal_relu.nnet");
    if model_path.exists() {
        let nnet = load_nnet(&model_path).expect("Failed to load model");
        let network = nnet.to_prop_network().expect("Failed to convert");
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
        )
        .unwrap();
        let output_bounds = vec![Bound::new(-100.0, 100.0)];

        let (eager, lazy) = benchmark_network("minimal_relu", &network, &input, &output_bounds);
        let speedup = eager.elapsed.as_secs_f64() / lazy.elapsed.as_secs_f64();

        println!("Network: minimal_relu.nnet (2->3->1, 3 ReLUs)");
        println!(
            "  Eager: {:.3}ms, Lazy: {:.3}ms, Speedup: {:.2}x",
            eager.elapsed.as_secs_f64() * 1000.0,
            lazy.elapsed.as_secs_f64() * 1000.0,
            speedup
        );
        println!();
    }

    // Synthetic networks of increasing size (kept small for reasonable runtime)
    let configs = [
        (2, 4, 1, 1, "2->4->1"),
        (2, 6, 2, 1, "2->6x2->1"),
        (3, 8, 2, 2, "3->8x2->2"),
    ];

    for (input_dim, hidden_dim, num_hidden, output_dim, desc) in configs {
        let network = create_benchmark_network(input_dim, hidden_dim, num_hidden, output_dim);
        let num_relus = hidden_dim * num_hidden;

        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[input_dim]), -0.5f32),
            ArrayD::from_elem(IxDyn(&[input_dim]), 0.5f32),
        )
        .unwrap();

        let output_bounds: Vec<Bound> =
            (0..output_dim).map(|_| Bound::new(-100.0, 100.0)).collect();

        let (eager, lazy) = benchmark_network(desc, &network, &input, &output_bounds);
        let speedup = eager.elapsed.as_secs_f64() / lazy.elapsed.as_secs_f64();

        println!("Network: {} ({} ReLUs)", desc, num_relus);
        println!(
            "  Eager: {:.3}ms, Lazy: {:.3}ms, Speedup: {:.2}x",
            eager.elapsed.as_secs_f64() * 1000.0,
            lazy.elapsed.as_secs_f64() * 1000.0,
            speedup
        );
        println!();
    }

    println!("=================================================================");
    println!("Note: Speedup > 1 means lazy is faster than eager");
    println!("For loose bounds with few uncertain ReLUs, lazy should be faster");
    println!("For tight bounds requiring refinement, eager may be faster");
    println!("=================================================================");
}
