use super::*;
use crate::network::{GraphNetwork, GraphNode};
use ndarray::{arr1, arr2};

#[test]
fn test_sdp_crown_relu_offset_zero_center_matches_closed_form() {
    // Example from SDP-CROWN paper (Figure 1):
    // f(x) = -ReLU(x1) - ReLU(x2) on B2(0, 1).
    //
    // Standard LiRPA/CROWN on the enclosing box produces g = [-0.5, -0.5] and offset -1.
    // SDP-CROWN improves the offset to -sqrt(0.5) ≈ -0.7071 (sqrt(2) tighter).
    let c = [-1.0f32, -1.0f32];
    let g = [-0.5f32, -0.5f32];
    let x_hat = [0.0f32, 0.0f32];
    let rho = 1.0f32;

    let h = crate::sdp_crown::relu_sdp_offset_opt(&c, &g, &x_hat, rho).unwrap();
    let expected = -(0.5f32).sqrt();
    assert!(
        (h - expected).abs() < 1e-4,
        "h={}, expected={}",
        h,
        expected
    );
}

#[test]
fn test_network_sdp_crown_matches_paper_example_bounds() {
    // Network: y = [-1, -1] · ReLU(x), input set x ∈ B2(0, 1).
    //
    // True range:
    // - min occurs at x = (1/sqrt(2), 1/sqrt(2)) => y = -sqrt(2)
    // - max occurs for x <= 0 => y = 0
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));

    let w = arr2(&[[-1.0, -1.0]]);
    let linear = LinearLayer::new(w, None).unwrap();
    net.add_layer(Layer::Linear(linear));

    // Provide the enclosing box x_hat ± rho for IBP slope selection.
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let x_hat = arr1(&[0.0, 0.0]);
    let rho = 1.0f32;

    let out = net.propagate_sdp_crown(&input, &x_hat, rho).unwrap();
    let lo = out.lower.as_slice().unwrap()[0];
    let up = out.upper.as_slice().unwrap()[0];

    let expected_lo = -(2.0f32).sqrt();
    let expected_up = 0.0f32;
    assert!(
        (lo - expected_lo).abs() < 1e-3,
        "lo={}, expected={}",
        lo,
        expected_lo
    );
    assert!(
        (up - expected_up).abs() < 1e-6,
        "up={}, expected={}",
        up,
        expected_up
    );
}

#[test]
fn test_graphnetwork_try_to_sequential_linear_relu_succeeds() {
    // Build a sequential Linear -> ReLU -> Linear graph
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0, 0.5], [0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    graph.add_node(GraphNode {
        name: "linear1".to_string(),
        layer: Layer::Linear(linear1),
        inputs: vec!["_input".to_string()],
    });

    graph.add_node(GraphNode {
        name: "relu".to_string(),
        layer: Layer::ReLU(ReLULayer),
        inputs: vec!["linear1".to_string()],
    });

    let w2 = arr2(&[[1.0, -1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode {
        name: "linear2".to_string(),
        layer: Layer::Linear(linear2),
        inputs: vec!["relu".to_string()],
    });

    graph.set_output("linear2");

    // Should successfully convert to Network
    let network = graph.try_to_sequential_network();
    assert!(
        network.is_some(),
        "Sequential Linear/ReLU graph should convert to Network"
    );

    let net = network.unwrap();
    assert_eq!(net.layers.len(), 3);
}

#[test]
fn test_graphnetwork_try_to_sequential_with_gelu_fails() {
    // Build a sequential Linear -> GELU graph (GELU not supported for SDP-CROWN)
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0, 0.5], [0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    graph.add_node(GraphNode {
        name: "linear1".to_string(),
        layer: Layer::Linear(linear1),
        inputs: vec!["_input".to_string()],
    });

    graph.add_node(GraphNode {
        name: "gelu".to_string(),
        layer: Layer::GELU(GELULayer::default()),
        inputs: vec!["linear1".to_string()],
    });

    graph.set_output("gelu");

    // Should return None since GELU is not supported
    let network = graph.try_to_sequential_network();
    assert!(
        network.is_none(),
        "Graph with GELU should not convert to Network for SDP-CROWN"
    );
}

#[test]
fn test_graphnetwork_try_to_sequential_with_branch_fails() {
    // Build a graph with a branch (Add layer needs two inputs)
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0], [0.5]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    graph.add_node(GraphNode {
        name: "linear1".to_string(),
        layer: Layer::Linear(linear1),
        inputs: vec!["_input".to_string()],
    });

    let w2 = arr2(&[[0.5], [1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode {
        name: "linear2".to_string(),
        layer: Layer::Linear(linear2),
        inputs: vec!["_input".to_string()],
    });

    graph.add_node(GraphNode {
        name: "add".to_string(),
        layer: Layer::Add(AddLayer),
        inputs: vec!["linear1".to_string(), "linear2".to_string()],
    });

    graph.set_output("add");

    // Should return None due to branch (Add is binary)
    let network = graph.try_to_sequential_network();
    assert!(
        network.is_none(),
        "Graph with branch should not convert to Network"
    );
}

#[test]
fn test_graphnetwork_sdp_crown_via_verifier() {
    use crate::types::{PropagationConfig, PropagationMethod};
    use crate::verifier::Verifier;
    use gamma_core::{Bound, VerificationResult, VerificationSpec};

    // Build a sequential Linear -> ReLU -> Linear graph
    let mut graph = GraphNetwork::new();

    // Input: 2D, Output after Linear1: 2D
    let w1 = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    graph.add_node(GraphNode {
        name: "linear1".to_string(),
        layer: Layer::Linear(linear1),
        inputs: vec!["_input".to_string()],
    });

    graph.add_node(GraphNode {
        name: "relu".to_string(),
        layer: Layer::ReLU(ReLULayer),
        inputs: vec!["linear1".to_string()],
    });

    // Output: 1D
    let w2 = arr2(&[[-1.0, -1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode {
        name: "linear2".to_string(),
        layer: Layer::Linear(linear2),
        inputs: vec!["relu".to_string()],
    });

    graph.set_output("linear2");

    // Verification spec with uniform epsilon box (required for SDP-CROWN)
    let spec = VerificationSpec {
        input_bounds: vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        output_bounds: vec![Bound::new(-10.0, 10.0)],
        input_shape: Some(vec![2]),
        timeout_ms: None,
    };

    let config = PropagationConfig {
        method: PropagationMethod::SdpCrown,
        max_iterations: 100,
        tolerance: 1e-4,
        use_gpu: false,
    };
    let verifier = Verifier::new(config);

    let result = verifier.verify_graph(&graph, &spec);
    assert!(
        result.is_ok(),
        "SDP-CROWN verification should succeed: {:?}",
        result.err()
    );

    let vr = result.unwrap();
    // The output bounds should be sound: y = -ReLU(x1) - ReLU(x2)
    // For x in B2(0, 1), min y = -sqrt(2), max y = 0
    match vr {
        VerificationResult::Verified { output_bounds, .. } => {
            assert!(
                output_bounds[0].lower <= 0.0,
                "Lower bound should be <= 0, got {:?}",
                output_bounds[0].lower
            );
            assert!(
                output_bounds[0].upper >= -std::f32::consts::SQRT_2,
                "Upper bound should be >= -sqrt(2), got {:?}",
                output_bounds[0].upper
            );
        }
        VerificationResult::Unknown { bounds, reason } => {
            assert!(
                bounds[0].lower <= 0.0,
                "Lower bound should be <= 0, got {:?} (reason: {})",
                bounds[0].lower,
                reason
            );
        }
        other => panic!("Unexpected result: {:?}", other),
    }
}
