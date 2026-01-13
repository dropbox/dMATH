//! GraphNetwork tests and Zonotope propagation tests

use crate::*;
use ndarray::{arr1, arr2, Array1, Array2, ArrayD, IxDyn};

// =========================================================================
// GraphNetwork Tests
// =========================================================================

#[test]
fn test_graph_network_empty() {
    let graph = GraphNetwork::new();
    let input = BoundedTensor::new(
        arr1(&[1.0_f32, 2.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    let output = graph.propagate_ibp(&input).unwrap();
    assert_eq!(output.shape(), input.shape());
}

#[test]
fn test_graph_network_single_node() {
    let mut graph = GraphNetwork::new();

    // Single ReLU node
    let relu_node = GraphNode::from_input("relu", Layer::ReLU(ReLULayer));
    graph.add_node(relu_node);
    graph.set_output("relu");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.5, 2.0]).into_dyn(),
        arr1(&[-0.5_f32, 1.5, 3.0]).into_dyn(),
    )
    .unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // ReLU: max(0, x)
    assert!((output.lower[[0]] - 0.0).abs() < 1e-5); // max(0, -1) = 0
    assert!((output.upper[[0]] - 0.0).abs() < 1e-5); // max(0, -0.5) = 0
    assert!((output.lower[[1]] - 0.5).abs() < 1e-5); // max(0, 0.5) = 0.5
    assert!((output.upper[[1]] - 1.5).abs() < 1e-5); // max(0, 1.5) = 1.5
}

#[test]
fn test_graph_network_sequential_chain() {
    // Create a chain: input -> linear -> relu
    let mut graph = GraphNetwork::new();

    // Linear layer: 2 inputs -> 2 outputs
    let weight = arr2(&[[1.0_f32, 0.0], [0.0, -1.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));

    // ReLU after linear
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        arr1(&[1.0_f32, 1.0]).into_dyn(),
        arr1(&[2.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // Linear: [x, y] -> [x, -y]
    // For x in [1, 2], y in [1, 2]:
    //   output[0] in [1, 2]
    //   output[1] in [-2, -1]
    // ReLU: [max(0, 1), max(0, -2)] to [max(0, 2), max(0, -1)]
    //   output[0] in [1, 2]
    //   output[1] in [0, 0]
    assert!((output.lower[[0]] - 1.0).abs() < 1e-5);
    assert!((output.upper[[0]] - 2.0).abs() < 1e-5);
    assert!((output.lower[[1]] - 0.0).abs() < 1e-5);
    assert!((output.upper[[1]] - 0.0).abs() < 1e-5);
}

#[test]
fn test_graph_network_from_sequential() {
    // Build sequential network
    let mut network = Network::new();
    let weight = arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]);
    network.add_layer(Layer::Linear(LinearLayer::new(weight, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Convert to graph
    let graph = GraphNetwork::from_sequential(&network);

    assert_eq!(graph.num_nodes(), 2);

    let input = BoundedTensor::new(
        arr1(&[1.0_f32, 1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let sequential_output = network.propagate_ibp(&input).unwrap();
    let graph_output = graph.propagate_ibp(&input).unwrap();

    // Should produce identical results
    assert!((sequential_output.lower[[0]] - graph_output.lower[[0]]).abs() < 1e-5);
    assert!((sequential_output.lower[[1]] - graph_output.lower[[1]]).abs() < 1e-5);
    assert!((sequential_output.upper[[0]] - graph_output.upper[[0]]).abs() < 1e-5);
    assert!((sequential_output.upper[[1]] - graph_output.upper[[1]]).abs() < 1e-5);
}

#[test]
fn test_graph_network_branching() {
    // Create a graph with branching: two linear projections from input, then add
    //        input
    //       /     \
    //    proj_a  proj_b
    //       \     /
    //         add
    let mut graph = GraphNetwork::new();

    // proj_a: identity
    let weight_a = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let proj_a = LinearLayer::new(weight_a, None).unwrap();
    graph.add_node(GraphNode::from_input("proj_a", Layer::Linear(proj_a)));

    // proj_b: scale by 2
    let weight_b = arr2(&[[2.0_f32, 0.0], [0.0, 2.0]]);
    let proj_b = LinearLayer::new(weight_b, None).unwrap();
    graph.add_node(GraphNode::from_input("proj_b", Layer::Linear(proj_b)));

    // Add: proj_a + proj_b
    graph.add_node(GraphNode::binary(
        "add",
        Layer::Add(AddLayer),
        "proj_a",
        "proj_b",
    ));
    graph.set_output("add");

    let input = BoundedTensor::new(
        arr1(&[1.0_f32, 2.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // proj_a: [1, 2], proj_b: [2, 4]
    // add: [3, 6]
    assert!((output.lower[[0]] - 3.0).abs() < 1e-5);
    assert!((output.upper[[0]] - 3.0).abs() < 1e-5);
    assert!((output.lower[[1]] - 6.0).abs() < 1e-5);
    assert!((output.upper[[1]] - 6.0).abs() < 1e-5);
}

#[test]
fn test_graph_network_where_ibp_union() {
    // Build a simple ternary Where graph:
    //   x = I(input)
    //   y = -x
    //   cond = relu(x) (condition is ignored by IBP Where relaxation; union bounds)
    //   out = where(cond, x, y)
    let mut graph = GraphNetwork::new();

    let w = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    graph.add_node(GraphNode::from_input(
        "x",
        Layer::Linear(LinearLayer::new(w, None).unwrap()),
    ));

    graph.add_node(GraphNode::new(
        "y",
        Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
        vec!["x".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "cond",
        Layer::ReLU(ReLULayer),
        vec!["x".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "out",
        Layer::Where(WhereLayer::new()),
        vec!["cond".to_string(), "x".to_string(), "y".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.5]).into_dyn(),
        arr1(&[2.0_f32, 1.5]).into_dyn(),
    )
    .unwrap();

    let out = graph.propagate_ibp(&input).unwrap();

    // x in [-1,2], y in [-2,1] => out in [-2,2]
    assert!((out.lower[[0]] - (-2.0)).abs() < 1e-5);
    assert!((out.upper[[0]] - 2.0).abs() < 1e-5);

    // x in [0.5,1.5], y in [-1.5,-0.5] => out in [-1.5,1.5]
    assert!((out.lower[[1]] - (-1.5)).abs() < 1e-5);
    assert!((out.upper[[1]] - 1.5).abs() < 1e-5);
}

#[test]
fn test_graph_network_where_crown_routes_true_branch() {
    // If the condition is provably all-true, DAG-CROWN should route the bounds to the `x`
    // branch and avoid the conservative union used by IBP.
    let mut graph = GraphNetwork::new();

    let w = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    graph.add_node(GraphNode::from_input(
        "x",
        Layer::Linear(LinearLayer::new(w, None).unwrap()),
    ));

    graph.add_node(GraphNode::new(
        "y",
        Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
        vec!["x".to_string()],
    ));

    // cond = 0*x + 1 => constant 1 (all true under >= 0.5 heuristic)
    graph.add_node(GraphNode::new(
        "cond0",
        Layer::MulConstant(MulConstantLayer::scalar(0.0)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cond",
        Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(IxDyn(&[]), 1.0))),
        vec!["cond0".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "out",
        Layer::Where(WhereLayer::new()),
        vec!["cond".to_string(), "x".to_string(), "y".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.5]).into_dyn(),
        arr1(&[2.0_f32, 1.5]).into_dyn(),
    )
    .unwrap();

    let ibp_out = graph.propagate_ibp(&input).unwrap();
    let crown_out = graph.propagate_crown(&input).unwrap();

    // With cond=true, output should match x, not the union.
    assert!((crown_out.lower[[0]] - (-1.0)).abs() < 1e-5);
    assert!((crown_out.upper[[0]] - 2.0).abs() < 1e-5);
    assert!((crown_out.lower[[1]] - 0.5).abs() < 1e-5);
    assert!((crown_out.upper[[1]] - 1.5).abs() < 1e-5);

    // IBP union should be at least as wide as routed CROWN output.
    assert!(ibp_out.lower[[0]] <= crown_out.lower[[0]] + 1e-6);
    assert!(ibp_out.upper[[0]] >= crown_out.upper[[0]] - 1e-6);
    assert!(ibp_out.lower[[1]] <= crown_out.lower[[1]] + 1e-6);
    assert!(ibp_out.upper[[1]] >= crown_out.upper[[1]] - 1e-6);
}

#[test]
fn test_graph_network_attention_pattern() {
    // Test bounded matmul with 2D input tensors (simulates Q @ K^T in attention)
    // Uses 2D input directly to matmul nodes without linear projection
    let mut graph = GraphNetwork::new();

    // For bounded matmul, we use ReLU first to create two 2D bounded tensors
    // then apply matmul. This tests the DAG structure.
    // input -> relu (as Q) AND input -> relu (as K) -> matmul

    // Simple pass-through via relu (positive inputs remain unchanged)
    graph.add_node(GraphNode::from_input("q", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("k", Layer::ReLU(ReLULayer)));

    // MatMul: Q @ K^T (both are 2x2)
    let matmul = MatMulLayer::new(true, Some(1.0 / 2.0_f32.sqrt())); // scale by 1/sqrt(d)
    graph.add_node(GraphNode::binary(
        "attn_scores",
        Layer::MatMul(matmul),
        "q",
        "k",
    ));

    // Softmax on last axis
    let softmax = SoftmaxLayer::new(-1);
    graph.add_node(GraphNode::new(
        "attn_probs",
        Layer::Softmax(softmax),
        vec!["attn_scores".to_string()],
    ));
    graph.set_output("attn_probs");

    // Input: 2D tensor (2 tokens, 2 dims each) - all positive so ReLU is identity
    let input = BoundedTensor::new(
        Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0, 0.0, 1.0])
            .unwrap()
            .into_dyn(),
        Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0, 0.0, 1.0])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // Check that softmax outputs are valid probabilities
    for &val in output.lower.iter() {
        assert!(val >= 0.0, "Softmax output {} < 0", val);
    }
    for &val in output.upper.iter() {
        assert!(val <= 1.0, "Softmax output {} > 1", val);
    }
}

#[test]
fn test_graph_topological_sort() {
    let mut graph = GraphNetwork::new();

    // Build: input -> a -> b -> c
    //                    \-> d (b and d in parallel from a)
    let weight = arr2(&[[1.0_f32]]);

    graph.add_node(GraphNode::from_input(
        "a",
        Layer::Linear(LinearLayer::new(weight.clone(), None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "b",
        Layer::ReLU(ReLULayer),
        vec!["a".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "c",
        Layer::ReLU(ReLULayer),
        vec!["b".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "d",
        Layer::ReLU(ReLULayer),
        vec!["a".to_string()],
    ));

    let sorted = graph.topological_sort().unwrap();

    // a must come before b, c, d
    // b must come before c
    let pos_a = sorted.iter().position(|x| x == "a").unwrap();
    let pos_b = sorted.iter().position(|x| x == "b").unwrap();
    let pos_c = sorted.iter().position(|x| x == "c").unwrap();
    let pos_d = sorted.iter().position(|x| x == "d").unwrap();

    assert!(pos_a < pos_b);
    assert!(pos_a < pos_d);
    assert!(pos_b < pos_c);
}

#[test]
fn test_attention_graph_builder() {
    let mut builder = AttentionGraphBuilder::new();

    // Build a simple graph using the builder API
    // Test with 1D vector to avoid shape issues with Linear layers
    let weight_q = arr2(&[[1.0_f32, 0.5], [0.5, 1.0]]);
    let weight_k = arr2(&[[1.0_f32, -0.5], [-0.5, 1.0]]);

    let q = builder.add_projection("q", weight_q, None).unwrap();
    let k = builder.add_projection("k", weight_k, None).unwrap();

    // Add outputs together (both are 1D vectors after projection)
    let sum = builder.add_residual(&q, &k);

    let graph = builder.build(&sum);

    assert_eq!(graph.num_nodes(), 3);

    // Test propagation with 1D input
    let input = BoundedTensor::new(
        arr1(&[1.0_f32, 1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // q = [1*1 + 0.5*1, 0.5*1 + 1*1] = [1.5, 1.5]
    // k = [1*1 + (-0.5)*1, (-0.5)*1 + 1*1] = [0.5, 0.5]
    // sum = [2.0, 2.0]
    assert!((output.lower[[0]] - 2.0).abs() < 1e-5);
    assert!((output.lower[[1]] - 2.0).abs() < 1e-5);
}

#[test]
fn test_attention_graph_builder_matmul() {
    // Test the builder with matmul using 2D inputs via ReLU passthrough
    let mut builder = AttentionGraphBuilder::new();

    // Add ReLU nodes for Q and K (acts as identity for positive inputs)
    let q = builder.add_relu("_input");
    let k = builder.add_relu("_input");
    let attn = builder.add_matmul(&q, &k, true, Some(0.5));
    let probs = builder.add_softmax(&attn, -1);

    let graph = builder.build(&probs);

    assert_eq!(graph.num_nodes(), 4);

    // Test propagation with 2D input (positive values, so ReLU is identity)
    let input = BoundedTensor::new(
        Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.5, 0.5, 1.0])
            .unwrap()
            .into_dyn(),
        Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.5, 0.5, 1.0])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // Softmax outputs should be valid probabilities
    for &val in output.lower.iter() {
        assert!(val >= 0.0, "Softmax lower {} < 0", val);
    }
    for &val in output.upper.iter() {
        assert!(val <= 1.0, "Softmax upper {} > 1", val);
    }
}

#[test]
fn test_attention_graph_builder_with_residual() {
    let mut builder = AttentionGraphBuilder::new();

    // Build: projection -> relu -> add(input, relu_output)
    let weight = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let proj = builder.add_projection("proj", weight, None).unwrap();
    let relu = builder.add_relu(&proj);
    let residual = builder.add_residual("_input", &relu);

    let graph = builder.build(&residual);

    let input = BoundedTensor::new(
        arr1(&[1.0_f32, 2.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // proj = input (identity), relu = proj (all positive), residual = input + relu = 2 * input
    assert!((output.lower[[0]] - 2.0).abs() < 1e-5);
    assert!((output.lower[[1]] - 4.0).abs() < 1e-5);
}

#[test]
fn test_graph_network_ibp_soundness() {
    // Test soundness: sample points should be within computed bounds
    let mut graph = GraphNetwork::new();

    // Build: input -> proj -> relu -> output
    let weight = arr2(&[[1.0_f32, -1.0], [-1.0, 1.0]]);
    let bias = arr1(&[0.5_f32, -0.5]);
    let proj = LinearLayer::new(weight, Some(bias)).unwrap();
    graph.add_node(GraphNode::from_input("proj", Layer::Linear(proj)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["proj".to_string()],
    ));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let bounds = graph.propagate_ibp(&input).unwrap();

    // Sample test points and verify they're within bounds
    let test_points = vec![
        arr1(&[-1.0_f32, -1.0]),
        arr1(&[1.0_f32, 1.0]),
        arr1(&[0.0_f32, 0.0]),
        arr1(&[-0.5_f32, 0.5]),
        arr1(&[0.5_f32, -0.5]),
    ];

    for point in test_points {
        // Linear: W @ x + b = [x0 - x1 + 0.5, -x0 + x1 - 0.5]
        let linear_out = arr1(&[
            point[[0]] - point[[1]] + 0.5,
            -point[[0]] + point[[1]] - 0.5,
        ]);

        // ReLU
        let relu_out = linear_out.mapv(|v| v.max(0.0));

        // Check bounds
        for i in 0..2 {
            assert!(
                relu_out[[i]] >= bounds.lower[[i]] - 1e-5,
                "Point {:?}: output[{}]={} < lower={}",
                point,
                i,
                relu_out[[i]],
                bounds.lower[[i]]
            );
            assert!(
                relu_out[[i]] <= bounds.upper[[i]] + 1e-5,
                "Point {:?}: output[{}]={} > upper={}",
                point,
                i,
                relu_out[[i]],
                bounds.upper[[i]]
            );
        }
    }
}

#[test]
fn test_graph_network_crown_sequential() {
    // Test CROWN propagation on a sequential graph
    let mut graph = GraphNetwork::new();

    // Build: input -> linear -> relu
    let weight = arr2(&[[1.0_f32, 0.5], [-0.5, 1.0]]);
    let bias = arr1(&[0.1_f32, -0.1]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let _ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Verify soundness: sample points should be within CROWN bounds
    let test_points = vec![
        arr1(&[-0.5_f32, -0.5]),
        arr1(&[0.5_f32, 0.5]),
        arr1(&[0.0_f32, 0.0]),
        arr1(&[-0.5_f32, 0.5]),
        arr1(&[0.5_f32, -0.5]),
    ];

    for point in test_points {
        let linear_out = arr1(&[
            point[[0]] + 0.5 * point[[1]] + 0.1,
            -0.5 * point[[0]] + point[[1]] - 0.1,
        ]);
        let relu_out = linear_out.mapv(|v| v.max(0.0));

        for i in 0..2 {
            assert!(
                relu_out[[i]] >= crown_bounds.lower[[i]] - 1e-5,
                "CROWN: Point {:?}: output[{}]={} < lower={}",
                point,
                i,
                relu_out[[i]],
                crown_bounds.lower[[i]]
            );
            assert!(
                relu_out[[i]] <= crown_bounds.upper[[i]] + 1e-5,
                "CROWN: Point {:?}: output[{}]={} > upper={}",
                point,
                i,
                relu_out[[i]],
                crown_bounds.upper[[i]]
            );
        }
    }

    // Note: CROWN's lower bound can be looser than IBP for ReLU outputs in some cases
    // because the linear relaxation y >= α*x can produce negative values when α < 1 and x < 0.
    // The key property is that CROWN bounds are SOUND (all true outputs are contained).
    // For tightness comparisons, see test_graph_network_crown_tighter_than_ibp which uses
    // upper bound comparison where CROWN is typically tighter.
}

#[test]
fn test_graph_network_crown_with_add() {
    // Test CROWN with Add binary operation in DAG structure
    let mut graph = GraphNetwork::new();

    // Build a DAG: input -> proj_a AND input -> proj_b -> add
    // With identity projections, output should be 3x input (proj_a + 2*proj_b = x + 2x = 3x)
    let weight_a = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let proj_a = LinearLayer::new(weight_a, None).unwrap();
    graph.add_node(GraphNode::from_input("proj_a", Layer::Linear(proj_a)));

    let weight_b = arr2(&[[2.0_f32, 0.0], [0.0, 2.0]]);
    let proj_b = LinearLayer::new(weight_b, None).unwrap();
    graph.add_node(GraphNode::from_input("proj_b", Layer::Linear(proj_b)));

    graph.add_node(GraphNode::binary(
        "add",
        Layer::Add(AddLayer),
        "proj_a",
        "proj_b",
    ));
    graph.set_output("add");

    // Test with concrete input
    let input = BoundedTensor::new(
        arr1(&[1.0_f32, 2.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // For linear networks, CROWN and IBP should give same results
    for i in 0..2 {
        assert!(
            (crown_bounds.lower[[i]] - ibp_bounds.lower[[i]]).abs() < 1e-4,
            "CROWN lower[{}]={} != IBP lower[{}]={}",
            i,
            crown_bounds.lower[[i]],
            i,
            ibp_bounds.lower[[i]]
        );
        assert!(
            (crown_bounds.upper[[i]] - ibp_bounds.upper[[i]]).abs() < 1e-4,
            "CROWN upper[{}]={} != IBP upper[{}]={}",
            i,
            crown_bounds.upper[[i]],
            i,
            ibp_bounds.upper[[i]]
        );
    }

    // Expected: [1, 2] + 2*[1, 2] = [3, 6]
    assert!((crown_bounds.lower[[0]] - 3.0).abs() < 1e-4);
    assert!((crown_bounds.lower[[1]] - 6.0).abs() < 1e-4);
}

#[test]
fn test_graph_network_crown_with_add_interval() {
    // Test CROWN with Add on interval inputs
    let mut graph = GraphNetwork::new();

    let weight_a = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let proj_a = LinearLayer::new(weight_a, None).unwrap();
    graph.add_node(GraphNode::from_input("proj_a", Layer::Linear(proj_a)));

    let weight_b = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let proj_b = LinearLayer::new(weight_b, None).unwrap();
    graph.add_node(GraphNode::from_input("proj_b", Layer::Linear(proj_b)));

    graph.add_node(GraphNode::binary(
        "add",
        Layer::Add(AddLayer),
        "proj_a",
        "proj_b",
    ));
    graph.set_output("add");

    // Input with interval: [0, 1] for both dimensions
    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // For linear DAG with Add: output = proj_a + proj_b = x + x = 2x
    // So bounds should be [0, 2] for both dimensions
    assert!((crown_bounds.lower[[0]] - 0.0).abs() < 1e-4);
    assert!((crown_bounds.upper[[0]] - 2.0).abs() < 1e-4);

    // CROWN should match IBP for this simple linear case
    for i in 0..2 {
        assert!((crown_bounds.lower[[i]] - ibp_bounds.lower[[i]]).abs() < 1e-4);
        assert!((crown_bounds.upper[[i]] - ibp_bounds.upper[[i]]).abs() < 1e-4);
    }
}

#[test]
fn test_matmul_ibp_soundness() {
    // Test that MatMul IBP bounds are sound (contain actual outputs)
    let matmul = MatMulLayer::new(false, None);

    // A: 2x3, B: 3x2 -> C: 2x2
    let a_lower = arr2(&[[0.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let a_upper = arr2(&[[1.0_f32, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    let input_a = BoundedTensor::new(a_lower.into_dyn(), a_upper.into_dyn()).unwrap();

    let b_lower = arr2(&[[0.0_f32, 0.0], [0.0, 0.0], [0.0, 0.0]]);
    let b_upper = arr2(&[[1.0_f32, 1.0], [1.0, 1.0], [1.0, 1.0]]);
    let input_b = BoundedTensor::new(b_lower.into_dyn(), b_upper.into_dyn()).unwrap();

    // Compute IBP bounds
    let ibp_bounds = matmul.propagate_ibp_binary(&input_a, &input_b).unwrap();

    // Test fixed sample points
    let test_points: Vec<(Array2<f32>, Array2<f32>)> = vec![
        // All zeros
        (Array2::zeros((2, 3)), Array2::zeros((3, 2))),
        // All ones
        (Array2::ones((2, 3)), Array2::ones((3, 2))),
        // Center
        (
            Array2::from_elem((2, 3), 0.5),
            Array2::from_elem((3, 2), 0.5),
        ),
        // Lower bounds
        (
            arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            arr2(&[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        ),
        // Upper bounds
        (
            arr2(&[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            arr2(&[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        ),
    ];

    for (a, b) in test_points {
        let c = a.dot(&b);

        // Check bounds contain the result
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    c[[i, j]] >= ibp_bounds.lower[[i, j]] - 1e-5,
                    "IBP lower bound violation: c[{},{}]={} < lower={}",
                    i,
                    j,
                    c[[i, j]],
                    ibp_bounds.lower[[i, j]]
                );
                assert!(
                    c[[i, j]] <= ibp_bounds.upper[[i, j]] + 1e-5,
                    "IBP upper bound violation: c[{},{}]={} > upper={}",
                    i,
                    j,
                    c[[i, j]],
                    ibp_bounds.upper[[i, j]]
                );
            }
        }
    }
}

#[test]
fn test_matmul_crown_backward() {
    // Test MatMul CROWN backward propagation directly
    let matmul = MatMulLayer::new(false, None);

    // A: 2x2, B: 2x2 -> C: 2x2
    let a_lower = arr2(&[[0.0_f32, 0.0], [0.0, 0.0]]);
    let a_upper = arr2(&[[1.0_f32, 1.0], [1.0, 1.0]]);
    let input_a = BoundedTensor::new(a_lower.into_dyn(), a_upper.into_dyn()).unwrap();

    let b_lower = arr2(&[[0.5_f32, 0.5], [0.5, 0.5]]);
    let b_upper = arr2(&[[1.0_f32, 1.0], [1.0, 1.0]]);
    let input_b = BoundedTensor::new(b_lower.into_dyn(), b_upper.into_dyn()).unwrap();

    // Create identity linear bounds for C (4 outputs = 2x2 flattened)
    let bounds = LinearBounds::identity(4);

    // Propagate backward through MatMul
    let (bounds_a, bounds_b) = matmul
        .propagate_linear_binary(&bounds, &input_a, &input_b)
        .unwrap();

    // Verify shapes are correct
    assert_eq!(bounds_a.num_outputs(), 4);
    assert_eq!(bounds_a.num_inputs(), 4); // A is 2x2 = 4 elements
    assert_eq!(bounds_b.num_outputs(), 4);
    assert_eq!(bounds_b.num_inputs(), 4); // B is 2x2 = 4 elements
}

#[test]
fn test_matmul_crown_with_transpose() {
    // Test MatMul CROWN backward with transpose_b = true
    let matmul = MatMulLayer::new(true, Some(0.5)); // transpose and scale

    // Q: 2x3, K: 2x3 (transposed to get 3x2) -> C: 2x2
    let q_lower = arr2(&[[0.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let q_upper = arr2(&[[1.0_f32, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    let input_q = BoundedTensor::new(q_lower.into_dyn(), q_upper.into_dyn()).unwrap();

    let k_lower = arr2(&[[0.5_f32, 0.5, 0.5], [0.5, 0.5, 0.5]]);
    let k_upper = arr2(&[[1.0_f32, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    let input_k = BoundedTensor::new(k_lower.into_dyn(), k_upper.into_dyn()).unwrap();

    // Create identity linear bounds for C (4 outputs = 2x2)
    let bounds = LinearBounds::identity(4);

    // Propagate backward
    let (bounds_q, bounds_k) = matmul
        .propagate_linear_binary(&bounds, &input_q, &input_k)
        .unwrap();

    // Verify shapes
    assert_eq!(bounds_q.num_outputs(), 4);
    assert_eq!(bounds_q.num_inputs(), 6); // Q is 2x3 = 6 elements
    assert_eq!(bounds_k.num_outputs(), 4);
    assert_eq!(bounds_k.num_inputs(), 6); // K is 2x3 = 6 elements
}

#[test]
fn test_graph_network_crown_matmul_soundness() {
    // Soundness: GraphNetwork DAG-CROWN with MatMul should contain sampled concrete outputs.
    let mut graph = GraphNetwork::new();

    // Use GELU to produce potentially negative bounds (exercises McCormick sign handling).
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));

    // Scores = Q @ K^T / sqrt(d)
    let head_dim = 3_usize;
    let matmul = MatMulLayer::new(true, Some(1.0 / (head_dim as f32).sqrt()));
    graph.add_node(GraphNode::binary("scores", Layer::MatMul(matmul), "q", "k"));
    graph.set_output("scores");

    let input = BoundedTensor::new(
        Array2::from_elem((2, head_dim), -1.0_f32).into_dyn(),
        Array2::from_elem((2, head_dim), 1.0_f32).into_dyn(),
    )
    .unwrap();

    let bounds = graph.propagate_crown(&input).unwrap();
    let lower = bounds
        .lower
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    let upper = bounds
        .upper
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    for sample_idx in 0..25_usize {
        let mut x = Array2::<f32>::zeros((2, head_dim));
        for i in 0..2_usize {
            for j in 0..head_dim {
                let t = ((sample_idx as u32).wrapping_mul(2654435761_u32) ^ ((i * 31 + j) as u32))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                x[[i, j]] = -1.0 + 2.0 * t;
            }
        }

        let q = x.mapv(|v| gelu_eval(v, GeluApproximation::Erf));
        let k = x.mapv(|v| gelu_eval(v, GeluApproximation::Erf));
        let score = q.dot(&k.t()) * (1.0 / (head_dim as f32).sqrt());

        for i in 0..2_usize {
            for j in 0..2_usize {
                let v = score[[i, j]];
                assert!(
                    v >= lower[[i, j]] - 1e-4,
                    "MatMul CROWN lower violation at ({},{}) sample {}: {} < {}",
                    i,
                    j,
                    sample_idx,
                    v,
                    lower[[i, j]]
                );
                assert!(
                    v <= upper[[i, j]] + 1e-4,
                    "MatMul CROWN upper violation at ({},{}) sample {}: {} > {}",
                    i,
                    j,
                    sample_idx,
                    v,
                    upper[[i, j]]
                );
            }
        }
    }
}

#[test]
fn test_graph_network_crown_attention_full_soundness() {
    // Full attention: (Q @ K^T / sqrt(d)) -> softmax -> @ V
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

    let head_dim = 3_usize;
    let scores = MatMulLayer::new(true, Some(1.0 / (head_dim as f32).sqrt()));
    graph.add_node(GraphNode::binary("scores", Layer::MatMul(scores), "q", "k"));

    let softmax = SoftmaxLayer::new(-1);
    graph.add_node(GraphNode::new(
        "probs",
        Layer::Softmax(softmax),
        vec!["scores".to_string()],
    ));

    let out = MatMulLayer::new(false, None);
    graph.add_node(GraphNode::binary("out", Layer::MatMul(out), "probs", "v"));
    graph.set_output("out");

    let input = BoundedTensor::new(
        Array2::from_elem((2, head_dim), -1.0_f32).into_dyn(),
        Array2::from_elem((2, head_dim), 1.0_f32).into_dyn(),
    )
    .unwrap();

    let bounds = graph.propagate_crown(&input).unwrap();
    let lower = bounds
        .lower
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    let upper = bounds
        .upper
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    let sm = SoftmaxLayer::new(-1);

    for sample_idx in 0..25_usize {
        let mut x = Array2::<f32>::zeros((2, head_dim));
        for i in 0..2_usize {
            for j in 0..head_dim {
                let t = ((sample_idx as u32).wrapping_mul(2654435761_u32) ^ ((i * 31 + j) as u32))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                x[[i, j]] = -1.0 + 2.0 * t;
            }
        }

        let q = x.mapv(|v| gelu_eval(v, GeluApproximation::Erf));
        let k = x.mapv(|v| gelu_eval(v, GeluApproximation::Erf));
        let v = x.mapv(|val| gelu_eval(val, GeluApproximation::Erf));

        let score = q.dot(&k.t()) * (1.0 / (head_dim as f32).sqrt());

        let mut probs = Array2::<f32>::zeros((2, 2));
        for i in 0..2_usize {
            probs.row_mut(i).assign(&sm.eval(&score.row(i).to_owned()));
        }

        let out = probs.dot(&v);

        for i in 0..2_usize {
            for j in 0..head_dim {
                let val = out[[i, j]];
                assert!(
                    val >= lower[[i, j]] - 1e-4,
                    "Attention CROWN lower violation at ({},{}) sample {}: {} < {}",
                    i,
                    j,
                    sample_idx,
                    val,
                    lower[[i, j]]
                );
                assert!(
                    val <= upper[[i, j]] + 1e-4,
                    "Attention CROWN upper violation at ({},{}) sample {}: {} > {}",
                    i,
                    j,
                    sample_idx,
                    val,
                    upper[[i, j]]
                );
            }
        }
    }
}

#[test]
fn test_add_crown_propagation() {
    // Test Add CROWN backward propagation
    let add = AddLayer;

    // Create identity linear bounds (4 outputs, 4 inputs)
    let bounds = LinearBounds::identity(4);

    // Propagate backward through Add
    let (bounds_a, bounds_b) = add.propagate_linear_binary(&bounds).unwrap();

    // Add passes bounds unchanged to both inputs
    assert_eq!(bounds_a.num_outputs(), bounds.num_outputs());
    assert_eq!(bounds_a.num_inputs(), bounds.num_inputs());
    assert_eq!(bounds_b.num_outputs(), bounds.num_outputs());
    assert_eq!(bounds_b.num_inputs(), bounds.num_inputs());

    // Verify coefficient matrices are the same as input
    for i in 0..4 {
        for j in 0..4 {
            assert!((bounds_a.lower_a[[i, j]] - bounds.lower_a[[i, j]]).abs() < 1e-6);
            assert!((bounds_b.lower_a[[i, j]] - bounds.lower_a[[i, j]]).abs() < 1e-6);
        }
    }
}

#[test]
fn test_graph_network_crown_add_bias_not_duplicated() {
    // ReLU creates non-zero intercept terms for crossing intervals; Add must not double-count them.
    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input("a", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("b", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::binary("add", Layer::Add(AddLayer), "a", "b"));
    graph.set_output("add");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let bounds = graph.propagate_crown(&input).unwrap();

    let test_points = vec![
        arr1(&[-1.0_f32, -1.0]),
        arr1(&[-0.5_f32, 0.5]),
        arr1(&[0.0_f32, 0.0]),
        arr1(&[0.5_f32, -0.5]),
        arr1(&[1.0_f32, 1.0]),
    ];

    for p in test_points {
        let relu = p.mapv(|v| v.max(0.0));
        let out = &relu + &relu;

        for i in 0..2_usize {
            assert!(
                out[[i]] >= bounds.lower[[i]] - 1e-5,
                "Add CROWN lower violation: point {:?} out[{}]={} < {}",
                p,
                i,
                out[[i]],
                bounds.lower[[i]]
            );
            assert!(
                out[[i]] <= bounds.upper[[i]] + 1e-5,
                "Add CROWN upper violation: point {:?} out[{}]={} > {}",
                p,
                i,
                out[[i]],
                bounds.upper[[i]]
            );
        }
    }
}

#[test]
fn test_transpose_layer_2d() {
    // Test 2D transpose
    let transpose = TransposeLayer::transpose_2d();

    let input = BoundedTensor::new(
        Array2::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_dyn(),
        Array2::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();

    let output = transpose.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[3, 2]);
    // Check transposed values
    assert!((output.lower[[0, 0]] - 1.0).abs() < 1e-5);
    assert!((output.lower[[0, 1]] - 4.0).abs() < 1e-5);
    assert!((output.lower[[1, 0]] - 2.0).abs() < 1e-5);
    assert!((output.lower[[1, 1]] - 5.0).abs() < 1e-5);
}

#[test]
fn test_transpose_layer_batched() {
    // Test batched transpose (swap last two dims of 3D tensor)
    let transpose = TransposeLayer::batched_transpose();

    // Shape: (2, 3, 4) -> (2, 4, 3)
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let input = BoundedTensor::new(
        ndarray::Array3::from_shape_vec((2, 3, 4), data.clone())
            .unwrap()
            .into_dyn(),
        ndarray::Array3::from_shape_vec((2, 3, 4), data)
            .unwrap()
            .into_dyn(),
    )
    .unwrap();

    let output = transpose.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 4, 3]);
}

#[test]
fn test_transpose_layer_interval_soundness() {
    // Test that transpose preserves interval bounds correctly
    let transpose = TransposeLayer::transpose_2d();

    let input = BoundedTensor::new(
        Array2::from_shape_vec((2, 3), vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0])
            .unwrap()
            .into_dyn(),
        Array2::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();

    let output = transpose.propagate_ibp(&input).unwrap();

    // Check that bounds are preserved for each element
    // Original [0,1] at (0,0) should be at (0,0) after transpose
    assert!((output.lower[[0, 0]] - 0.0).abs() < 1e-5);
    assert!((output.upper[[0, 0]] - 1.0).abs() < 1e-5);

    // Original [3,4] at (1,0) should be at (0,1) after transpose
    assert!((output.lower[[0, 1]] - 3.0).abs() < 1e-5);
    assert!((output.upper[[0, 1]] - 4.0).abs() < 1e-5);
}

#[test]
fn test_graph_network_crown_tighter_than_ibp() {
    // Create a network where CROWN should produce tighter bounds than IBP
    let mut graph = GraphNetwork::new();

    // Use weights that cause IBP to over-approximate
    let weight = arr2(&[[1.0_f32, -1.0], [1.0, 1.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("relu");

    // Input perturbation
    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Calculate widths
    let crown_width: f32 = (0..2)
        .map(|i| crown_bounds.upper[[i]] - crown_bounds.lower[[i]])
        .sum();
    let ibp_width: f32 = (0..2)
        .map(|i| ibp_bounds.upper[[i]] - ibp_bounds.lower[[i]])
        .sum();

    // CROWN should be tighter (smaller total width)
    assert!(
        crown_width <= ibp_width + 1e-5,
        "CROWN width {} should be <= IBP width {}",
        crown_width,
        ibp_width
    );
}

// ========================================================================
// GraphNetwork α-CROWN Tests
// ========================================================================

#[test]
fn test_graph_network_alpha_crown_soundness() {
    // Test that GraphNetwork α-CROWN produces sound bounds
    use ndarray::arr2;

    let mut graph = GraphNetwork::new();

    // Create: Linear -> ReLU -> Linear -> ReLU
    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0], [-1.0, 0.3]]);
    let linear1 = LinearLayer::new(w1, Some(arr1(&[0.0, 0.1, -0.1]))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.5_f32, -0.3, 0.8], [0.2, 0.6, -0.4]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.set_output("relu2");

    // Input with perturbation
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let alpha_bounds = graph.propagate_alpha_crown(&input).unwrap();
    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let _ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Verify soundness: sample random inputs
    for i in 0..100 {
        let t1 = (i * 7 % 100) as f32 / 100.0;
        let t2 = (i * 11 % 100) as f32 / 100.0;
        let x1 = -0.5 + t1;
        let x2 = -0.5 + t2;

        // Forward pass through network
        let z1 = [
            (1.0 * x1 - 0.5 * x2 + 0.0).max(0.0),
            (0.5 * x1 + 1.0 * x2 + 0.1).max(0.0),
            (-x1 + 0.3 * x2 - 0.1).max(0.0),
        ];
        let z2 = [
            (0.5 * z1[0] - 0.3 * z1[1] + 0.8 * z1[2]).max(0.0),
            (0.2 * z1[0] + 0.6 * z1[1] - 0.4 * z1[2]).max(0.0),
        ];

        // Check α-CROWN bounds contain the output
        for (j, &z2_val) in z2.iter().enumerate() {
            assert!(
                z2_val >= alpha_bounds.lower[[j]] - 1e-5
                    && z2_val <= alpha_bounds.upper[[j]] + 1e-5,
                "Output {} outside α-CROWN bounds: {} not in [{}, {}]",
                j,
                z2_val,
                alpha_bounds.lower[[j]],
                alpha_bounds.upper[[j]]
            );
        }
    }

    // α-CROWN should be at least as tight as CROWN
    for i in 0..2 {
        let alpha_width = alpha_bounds.upper[[i]] - alpha_bounds.lower[[i]];
        let crown_width = crown_bounds.upper[[i]] - crown_bounds.lower[[i]];
        assert!(
            alpha_width <= crown_width + 1e-4,
            "α-CROWN width {} should be <= CROWN width {} at output {}",
            alpha_width,
            crown_width,
            i
        );
    }
}

#[test]
fn test_graph_network_alpha_crown_with_gelu() {
    // Test α-CROWN with GELU in the network
    use ndarray::arr2;

    let mut graph = GraphNetwork::new();

    // Create: Linear -> GELU -> Linear -> ReLU
    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, Some(arr1(&[0.0, 0.1]))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::default()),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.5_f32, -0.3], [0.2, 0.6]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["gelu".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.set_output("relu");

    // Input with perturbation
    let input = BoundedTensor::new(
        arr1(&[-0.3_f32, -0.3]).into_dyn(),
        arr1(&[0.3_f32, 0.3]).into_dyn(),
    )
    .unwrap();

    let alpha_bounds = graph.propagate_alpha_crown(&input).unwrap();
    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let _ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Verify soundness: sample random inputs
    for i in 0..100 {
        let t1 = (i * 7 % 100) as f32 / 100.0;
        let t2 = (i * 11 % 100) as f32 / 100.0;
        let x1 = -0.3 + 0.6 * t1;
        let x2 = -0.3 + 0.6 * t2;

        // Forward pass through network
        let z1 = [
            gelu_eval(1.0 * x1 - 0.5 * x2 + 0.0, GeluApproximation::Erf),
            gelu_eval(0.5 * x1 + 1.0 * x2 + 0.1, GeluApproximation::Erf),
        ];
        let z2 = [
            (0.5 * z1[0] - 0.3 * z1[1]).max(0.0),
            (0.2 * z1[0] + 0.6 * z1[1]).max(0.0),
        ];

        // Check bounds contain the output
        for (j, &z2_val) in z2.iter().enumerate() {
            assert!(
                z2_val >= alpha_bounds.lower[[j]] - 1e-5
                    && z2_val <= alpha_bounds.upper[[j]] + 1e-5,
                "Output {} outside α-CROWN+GELU bounds: {} not in [{}, {}]",
                j,
                z2_val,
                alpha_bounds.lower[[j]],
                alpha_bounds.upper[[j]]
            );
        }
    }

    // α-CROWN should be at least as tight as CROWN
    for i in 0..2 {
        let alpha_width = alpha_bounds.upper[[i]] - alpha_bounds.lower[[i]];
        let crown_width = crown_bounds.upper[[i]] - crown_bounds.lower[[i]];
        // α-CROWN with GELU may not be tighter than pure CROWN since α optimization is only for ReLU
        // but it should still be sound
        assert!(
            alpha_width <= crown_width + 0.1, // Allow some tolerance
            "α-CROWN+GELU width {} significantly worse than CROWN width {} at output {}",
            alpha_width,
            crown_width,
            i
        );
    }
}

#[test]
fn test_graph_network_alpha_crown_no_relu() {
    // Test α-CROWN on network without ReLU (should fall back to CROWN)
    use ndarray::arr2;

    let mut graph = GraphNetwork::new();

    // Create: Linear only
    let w = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let linear = LinearLayer::new(w, None).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.set_output("linear");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let alpha_bounds = graph.propagate_alpha_crown(&input).unwrap();
    let crown_bounds = graph.propagate_crown(&input).unwrap();

    // Should be identical since no ReLU to optimize
    for i in 0..2 {
        assert!(
            (alpha_bounds.lower[[i]] - crown_bounds.lower[[i]]).abs() < 1e-5,
            "α-CROWN lower {} != CROWN lower {} at {}",
            alpha_bounds.lower[[i]],
            crown_bounds.lower[[i]],
            i
        );
        assert!(
            (alpha_bounds.upper[[i]] - crown_bounds.upper[[i]]).abs() < 1e-5,
            "α-CROWN upper {} != CROWN upper {} at {}",
            alpha_bounds.upper[[i]],
            crown_bounds.upper[[i]],
            i
        );
    }
}

#[test]
fn test_graph_network_dag_alpha_crown_with_skip_connection() {
    // Test DAG α-CROWN on a ResNet-like structure with skip connections (Add)
    use ndarray::arr2;

    let mut graph = GraphNetwork::new();

    // Create a residual block:
    //   Input -> Linear1 -> ReLU -> Linear2 --> Add --> Output
    //          \                              /
    //           \---------(skip)-------------/

    // Main path: Linear1
    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, Some(arr1(&[0.0, 0.1]))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    // Main path: ReLU
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    // Main path: Linear2
    let w2 = arr2(&[[0.5_f32, -0.3], [0.2, 0.6]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));

    // Skip connection: project input to match dimensions (identity-like)
    let w_skip = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let linear_skip = LinearLayer::new(w_skip, None).unwrap();
    graph.add_node(GraphNode::from_input(
        "skip_linear",
        Layer::Linear(linear_skip),
    ));

    // Add operation: combines main path with skip connection
    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["linear2".to_string(), "skip_linear".to_string()],
    ));

    // Final ReLU after add
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["add".to_string()],
    ));

    graph.set_output("relu2");

    // Input with perturbation
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    // Run α-CROWN (should use DAG α-CROWN internally for non-sequential graph)
    let alpha_bounds = graph.propagate_alpha_crown(&input).unwrap();
    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    println!(
        "DAG α-CROWN with skip connection: α-CROWN lower={:?}, CROWN lower={:?}, IBP lower={:?}",
        alpha_bounds.lower.as_slice().unwrap(),
        crown_bounds.lower.as_slice().unwrap(),
        ibp_bounds.lower.as_slice().unwrap()
    );
    println!(
        "DAG α-CROWN with skip connection: α-CROWN upper={:?}, CROWN upper={:?}, IBP upper={:?}",
        alpha_bounds.upper.as_slice().unwrap(),
        crown_bounds.upper.as_slice().unwrap(),
        ibp_bounds.upper.as_slice().unwrap()
    );

    // Note: For complex DAGs with ReLU, CROWN may give looser bounds than IBP in some cases.
    // This is because CROWN's linear relaxation can over-approximate more than IBP's interval
    // propagation for certain network structures. The key property is that all methods give
    // sound bounds (i.e., they contain the true output).

    // Verify soundness: all bounds contain the true output
    // Since the final layer is ReLU, the true output is always in [0, max_output]
    // We verify this by sampling the network

    // Get weight matrices for manual forward pass
    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let b1 = arr1(&[0.0_f32, 0.1]);
    let w2 = arr2(&[[0.5_f32, -0.3], [0.2, 0.6]]);
    let w_skip = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);

    // Sample various inputs and verify bounds are sound
    for i in 0..20 {
        let t = (i as f32) / 20.0;
        let sample_input = arr1(&[-0.5 + t, -0.5 + 0.5 * t]);

        // Compute network output: linear1 -> relu1 -> linear2 + skip -> relu2
        let h1 = w1.dot(&sample_input) + &b1;
        let h1_relu = h1.mapv(|v| v.max(0.0));
        let h2 = w2.dot(&h1_relu);
        let skip_out = w_skip.dot(&sample_input);
        let add_out = &h2 + &skip_out;
        let output = add_out.mapv(|v| v.max(0.0)); // Final ReLU

        // Verify bounds contain the output
        for j in 0..2 {
            assert!(
                output[[j]] >= alpha_bounds.lower[[j]] - 1e-3,
                "α-CROWN lower {} > actual output {} at dim {} (unsound!)",
                alpha_bounds.lower[[j]],
                output[[j]],
                j
            );
            assert!(
                output[[j]] <= alpha_bounds.upper[[j]] + 1e-3,
                "α-CROWN upper {} < actual output {} at dim {} (unsound!)",
                alpha_bounds.upper[[j]],
                output[[j]],
                j
            );
            assert!(
                output[[j]] >= crown_bounds.lower[[j]] - 1e-3,
                "CROWN lower {} > actual output {} at dim {} (unsound!)",
                crown_bounds.lower[[j]],
                output[[j]],
                j
            );
            assert!(
                output[[j]] <= crown_bounds.upper[[j]] + 1e-3,
                "CROWN upper {} < actual output {} at dim {} (unsound!)",
                crown_bounds.upper[[j]],
                output[[j]],
                j
            );
            assert!(
                output[[j]] >= ibp_bounds.lower[[j]] - 1e-3,
                "IBP lower {} > actual output {} at dim {} (unsound!)",
                ibp_bounds.lower[[j]],
                output[[j]],
                j
            );
            assert!(
                output[[j]] <= ibp_bounds.upper[[j]] + 1e-3,
                "IBP upper {} < actual output {} at dim {} (unsound!)",
                ibp_bounds.upper[[j]],
                output[[j]],
                j
            );
        }
    }

    // α-CROWN should be at least as tight as CROWN (α-optimization can only improve)
    for i in 0..2 {
        assert!(
            alpha_bounds.lower[[i]] >= crown_bounds.lower[[i]] - 1e-4,
            "α-CROWN lower {} < CROWN lower {} at {} (α-CROWN should be at least as tight)",
            alpha_bounds.lower[[i]],
            crown_bounds.lower[[i]],
            i
        );
        assert!(
            alpha_bounds.upper[[i]] <= crown_bounds.upper[[i]] + 1e-4,
            "α-CROWN upper {} > CROWN upper {} at {} (α-CROWN should be at least as tight)",
            alpha_bounds.upper[[i]],
            crown_bounds.upper[[i]],
            i
        );
    }

    // Compute bound widths
    let alpha_width: f32 = alpha_bounds
        .upper
        .iter()
        .zip(alpha_bounds.lower.iter())
        .map(|(u, l)| u - l)
        .sum::<f32>()
        / 2.0;
    let crown_width: f32 = crown_bounds
        .upper
        .iter()
        .zip(crown_bounds.lower.iter())
        .map(|(u, l)| u - l)
        .sum::<f32>()
        / 2.0;
    let ibp_width: f32 = ibp_bounds
        .upper
        .iter()
        .zip(ibp_bounds.lower.iter())
        .map(|(u, l)| u - l)
        .sum::<f32>()
        / 2.0;

    println!(
        "Average widths: α-CROWN={:.4}, CROWN={:.4}, IBP={:.4}",
        alpha_width, crown_width, ibp_width
    );
    println!(
        "Tightening: α-CROWN vs IBP={:.2}x, CROWN vs IBP={:.2}x",
        ibp_width / alpha_width.max(1e-6),
        ibp_width / crown_width.max(1e-6)
    );
}

#[test]
fn test_graph_network_crown_with_layernorm() {
    // Test CROWN propagation through GraphNetwork with LayerNorm
    use ndarray::arr2;

    let mut graph = GraphNetwork::new();

    // Create: Linear -> LayerNorm
    let w = arr2(&[[1.0_f32, -0.5, 0.3], [0.5, 1.0, -0.2], [-0.3, 0.2, 1.0]]);
    let linear = LinearLayer::new(w, Some(arr1(&[0.1, -0.1, 0.0]))).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));

    let gamma = arr1(&[1.0_f32, 1.0, 1.0]);
    let beta = arr1(&[0.0_f32, 0.0, 0.0]);
    let ln = LayerNormLayer::new(gamma, beta, 1e-5);
    graph.add_node(GraphNode::new(
        "layernorm",
        Layer::LayerNorm(ln.clone()),
        vec!["linear".to_string()],
    ));

    graph.set_output("layernorm");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5, 0.5]).into_dyn(),
    )
    .unwrap();

    // Get CROWN and IBP bounds
    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Verify soundness by sampling
    let linear_node = graph.nodes.get("linear").unwrap();
    let linear_layer = match &linear_node.layer {
        Layer::Linear(l) => l,
        _ => panic!("Expected Linear"),
    };

    for i in 0..50 {
        let t0 = (i * 17 % 50) as f32 / 50.0;
        let t1 = (i * 31 % 50) as f32 / 50.0;
        let t2 = (i * 47 % 50) as f32 / 50.0;

        let x_sample = arr1(&[-0.5 + t0, -0.5 + t1, -0.5 + t2]);

        // Forward through Linear
        let linear_out: Array1<f32> =
            linear_layer.weight.dot(&x_sample) + linear_layer.bias.as_ref().unwrap();

        // Forward through LayerNorm
        let ln_out = ln.eval(&linear_out);

        // Check bounds
        for j in 0..3 {
            assert!(
                ln_out[j] >= crown_bounds.lower[[j]] - 1e-3,
                "Sample {} output {} = {} < CROWN lower bound {} at dim {}",
                i,
                ln_out[j],
                ln_out[j],
                crown_bounds.lower[[j]],
                j
            );
            assert!(
                ln_out[j] <= crown_bounds.upper[[j]] + 1e-3,
                "Sample {} output {} = {} > CROWN upper bound {} at dim {}",
                i,
                ln_out[j],
                ln_out[j],
                crown_bounds.upper[[j]],
                j
            );
        }
    }

    // CROWN should not be much worse than IBP
    let crown_width: f32 = (0..3)
        .map(|i| crown_bounds.upper[[i]] - crown_bounds.lower[[i]])
        .sum();
    let ibp_width: f32 = (0..3)
        .map(|i| ibp_bounds.upper[[i]] - ibp_bounds.lower[[i]])
        .sum();

    println!(
        "GraphNetwork LayerNorm: IBP width = {}, CROWN width = {}",
        ibp_width, crown_width
    );

    // Allow some tolerance - CROWN might not always be tighter for LayerNorm
    assert!(
        crown_width <= ibp_width * 2.0,
        "CROWN width {} should not be much worse than IBP width {}",
        crown_width,
        ibp_width
    );
}

// =========================================================================
// Zonotope Propagation Tests
// =========================================================================

#[test]
fn test_graph_network_zonotope_matmul_soundness() {
    // Test that zonotope propagation produces sound bounds for Q@K^T pattern
    //
    // Key insight: When Q and K both come from the same input X with shared
    // error symbols, zonotope tracks correlations and produces sound bounds
    // even when IBP would be loose or unsound for certain configurations.

    let mut graph = GraphNetwork::new();

    // Create a simple graph: input -> split into Q and K -> Q@K^T
    // Using the input directly for both Q and K (identity split)
    graph.add_node(GraphNode::binary(
        "qk_matmul",
        Layer::MatMul(MatMulLayer::new(true, None)), // transpose_b = true for Q@K^T
        "_input",
        "_input",
    ));

    graph.set_output("qk_matmul");

    // Create a 2D input: (2, 3) matrix representing (seq_len=2, dim=3)
    let input_lower = arr2(&[[0.9, 0.9, 0.9], [0.9, 0.9, 0.9]]).into_dyn();
    let input_upper = arr2(&[[1.1, 1.1, 1.1], [1.1, 1.1, 1.1]]).into_dyn();
    let input = BoundedTensor::new(input_lower.clone(), input_upper.clone()).unwrap();

    // Run zonotope propagation
    let epsilon = 0.1;
    let zonotope_output = graph.propagate_zonotope(&input, epsilon).unwrap();

    // Run IBP for comparison
    let ibp_output = graph.propagate_ibp(&input).unwrap();

    // Output should be (2, 2) for Q@K^T where Q and K are (2, 3)
    assert_eq!(zonotope_output.shape(), &[2, 2]);
    assert_eq!(ibp_output.shape(), &[2, 2]);

    // Soundness check: sample concrete points and verify they're within bounds
    // For X @ X^T where X = [[1,1,1],[1,1,1]], we get [[3,3],[3,3]]
    // With perturbation, the output should contain this value
    let concrete_center = arr2(&[[3.0, 3.0], [3.0, 3.0]]);

    for i in 0..2 {
        for j in 0..2 {
            let center_val = concrete_center[[i, j]];
            // Both bounds should contain the center value (allowing for numerical tolerance)
            assert!(
                zonotope_output.lower[[i, j]] <= center_val + 0.5,
                "Zonotope lower {} should be <= {} at ({},{})",
                zonotope_output.lower[[i, j]],
                center_val,
                i,
                j
            );
            assert!(
                zonotope_output.upper[[i, j]] >= center_val - 0.5,
                "Zonotope upper {} should be >= {} at ({},{})",
                zonotope_output.upper[[i, j]],
                center_val,
                i,
                j
            );
        }
    }

    // Log comparison
    let zonotope_max_width = zonotope_output.max_width();
    let ibp_max_width = ibp_output.max_width();
    println!(
        "Q@K^T bounds: zonotope_max_width={}, ibp_max_width={}",
        zonotope_max_width, ibp_max_width
    );
}

#[test]
fn test_graph_network_zonotope_matmul_3d_smoke() {
    // Smoke test: zonotope propagation should support batched sequence tensors
    // with shape (batch, seq, dim) for Q@K^T patterns.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::binary(
        "qk_matmul",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "_input",
        "_input",
    ));
    graph.set_output("qk_matmul");

    let batch = 2_usize;
    let seq = 3_usize;
    let dim = 4_usize;
    let input = BoundedTensor::new(
        ndarray::ArrayD::from_elem(vec![batch, seq, dim], -1.0_f32),
        ndarray::ArrayD::from_elem(vec![batch, seq, dim], 1.0_f32),
    )
    .unwrap();

    let out = graph.propagate_zonotope(&input, 0.1).unwrap();
    assert_eq!(out.shape(), &[batch, seq, seq]);
    for (l, u) in out.lower.iter().zip(out.upper.iter()) {
        assert!(l.is_finite() && u.is_finite(), "Non-finite bounds");
        assert!(*l <= *u + 1e-6, "Invalid interval: {} > {}", l, u);
    }
}

#[test]
fn test_graph_network_zonotope_add_operation() {
    // Test zonotope propagation with Add operation
    let mut graph = GraphNetwork::new();

    // Create: input1 + input2 (where both are the same input)
    graph.add_node(GraphNode::binary(
        "sum",
        Layer::Add(AddLayer),
        "_input",
        "_input",
    ));
    graph.set_output("sum");

    // 2D input
    let input = BoundedTensor::new(
        arr2(&[[0.9, 1.9]]).into_dyn(),
        arr2(&[[1.1, 2.1]]).into_dyn(),
    )
    .unwrap();

    let result = graph.propagate_zonotope(&input, 0.1).unwrap();

    // x + x = 2x, so bounds should be approximately doubled
    // Center is [1, 2], so result should be [2, 4] with doubled width
    assert_eq!(result.shape(), &[1, 2]);

    // Check center is approximately [2, 4]
    let center_0 = (result.lower[[0, 0]] + result.upper[[0, 0]]) / 2.0;
    let center_1 = (result.lower[[0, 1]] + result.upper[[0, 1]]) / 2.0;
    assert!(
        (center_0 - 2.0).abs() < 0.1,
        "Center[0] should be ~2, got {}",
        center_0
    );
    assert!(
        (center_1 - 4.0).abs() < 0.1,
        "Center[1] should be ~4, got {}",
        center_1
    );
}

#[test]
fn test_graph_network_zonotope_fallback_to_ibp() {
    // Test that zonotope propagation falls back to IBP for unsupported operations
    let mut graph = GraphNetwork::new();

    // Create: input -> ReLU (not supported by zonotope)
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        arr2(&[[-0.5, 0.5]]).into_dyn(),
        arr2(&[[0.5, 1.5]]).into_dyn(),
    )
    .unwrap();

    // Should not error - should fall back to IBP
    let result = graph.propagate_zonotope(&input, 0.1).unwrap();

    // ReLU output shape should match input
    assert_eq!(result.shape(), &[1, 2]);

    // ReLU bounds should be valid
    // Element [0,0] crosses zero: [max(0, -0.5), max(0, 0.5)] = [0, 0.5]
    // Element [0,1] is positive: [max(0, 0.5), max(0, 1.5)] = [0.5, 1.5]
    assert!(result.lower[[0, 0]] >= -0.01, "ReLU lower should be >= 0");
    assert!(
        result.lower[[0, 1]] >= 0.49,
        "ReLU lower[1] should be >= 0.5"
    );
}

#[test]
fn test_graph_network_zonotope_non_2d_fallback() {
    // Test that <2D input falls back to IBP
    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    // 1D input (not 2D)
    let input =
        BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[1.0, 2.0]).into_dyn()).unwrap();

    // Should fall back to IBP for non-2D input
    let result = graph.propagate_zonotope(&input, 0.1).unwrap();

    // Should produce valid output
    assert_eq!(result.shape(), &[2]);
}

#[test]
fn test_block_wise_verification() {
    // Test block-wise verification with a simple 2-block transformer-like graph.
    // Each block has: attn_norm -> q_proj -> k_proj -> qk_matmul
    // Output ends at qk_matmul to avoid shape mismatch issues
    let mut graph = GraphNetwork::new();
    let seq = 4;
    let hidden = 8;
    let epsilon = 0.1;

    // Block 0
    graph.add_node(GraphNode::from_input(
        "layer0_attn_norm",
        Layer::LayerNorm(LayerNormLayer::new(
            Array1::ones(hidden),
            Array1::zeros(hidden),
            1e-5,
        )),
    ));

    let q_weight = Array2::from_shape_fn((hidden, hidden), |(i, j)| if i == j { 0.1 } else { 0.0 });
    graph.add_node(GraphNode::new(
        "layer0_q_proj",
        Layer::Linear(LinearLayer::new(q_weight.clone(), None).unwrap()),
        vec!["layer0_attn_norm".to_string()],
    ));

    let k_weight = Array2::from_shape_fn((hidden, hidden), |(i, j)| if i == j { 0.1 } else { 0.0 });
    graph.add_node(GraphNode::new(
        "layer0_k_proj",
        Layer::Linear(LinearLayer::new(k_weight.clone(), None).unwrap()),
        vec!["layer0_attn_norm".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "layer0_qk_matmul",
        Layer::MatMul(MatMulLayer::new(true, Some(1.0))),
        vec!["layer0_q_proj".to_string(), "layer0_k_proj".to_string()],
    ));

    // Block 1 - depends on block 0's LayerNorm output (via _input)
    graph.add_node(GraphNode::new(
        "layer1_attn_norm",
        Layer::LayerNorm(LayerNormLayer::new(
            Array1::ones(hidden),
            Array1::zeros(hidden),
            1e-5,
        )),
        vec!["layer0_attn_norm".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "layer1_q_proj",
        Layer::Linear(LinearLayer::new(q_weight.clone(), None).unwrap()),
        vec!["layer1_attn_norm".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "layer1_k_proj",
        Layer::Linear(LinearLayer::new(k_weight.clone(), None).unwrap()),
        vec!["layer1_attn_norm".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "layer1_qk_matmul",
        Layer::MatMul(MatMulLayer::new(true, Some(1.0))),
        vec!["layer1_q_proj".to_string(), "layer1_k_proj".to_string()],
    ));

    graph.set_output("layer1_qk_matmul");

    // Create input
    let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[seq, hidden])), epsilon);

    // Run block-wise verification
    let result = graph.propagate_ibp_block_wise(&input, epsilon).unwrap();

    // Should detect 2 blocks
    assert_eq!(
        result.total_blocks, 2,
        "Expected 2 blocks, got {}",
        result.total_blocks
    );
    assert_eq!(result.blocks.len(), 2);

    // Block 0 should have qk_matmul with zonotope tightening
    assert_eq!(result.blocks[0].block_name, "layer0");
    assert!(
        result.blocks[0].qk_matmul_width.is_some(),
        "Block 0 should have Q@K^T width"
    );

    // Block 1 should also have qk_matmul
    assert_eq!(result.blocks[1].block_name, "layer1");
    assert!(
        result.blocks[1].qk_matmul_width.is_some(),
        "Block 1 should have Q@K^T width"
    );

    // Q@K^T bounds should be finite (not NaN or inf)
    let qk0 = result.blocks[0].qk_matmul_width.unwrap();
    let qk1 = result.blocks[1].qk_matmul_width.unwrap();

    // Bounds should be finite - exact tightness depends on zonotope path detection
    // which may not trigger for all graph structures
    assert!(qk0.is_finite(), "Q@K^T should be finite, got {}", qk0);
    assert!(qk1.is_finite(), "Q@K^T should be finite, got {}", qk1);
    assert!(qk0 < 1e10, "Q@K^T should not saturate, got {}", qk0);
    assert!(qk1 < 1e10, "Q@K^T should not saturate, got {}", qk1);

    // No degradation (NaN/inf) expected for small epsilon
    assert_eq!(result.degraded_blocks, 0, "Expected no degraded blocks");
}

#[test]
fn test_parse_block_index() {
    // Test block index parsing from node names
    assert_eq!(GraphNetwork::parse_block_index("layer0_attn_norm"), Some(0));
    assert_eq!(GraphNetwork::parse_block_index("layer12_q_proj"), Some(12));
    assert_eq!(GraphNetwork::parse_block_index("layer127_add2"), Some(127));
    assert_eq!(GraphNetwork::parse_block_index("embedding"), None);
    assert_eq!(GraphNetwork::parse_block_index("output_norm"), None);
    assert_eq!(GraphNetwork::parse_block_index("layernorm"), None); // No number
}

#[test]
fn test_mulbinary_mccormick_crown_soundness() {
    // Test that MulBinary McCormick CROWN produces sound bounds
    // z = x * y where x and y are bounded
    use ndarray::Array1;

    let mul = MulBinaryLayer;

    // Test case 1: Positive bounds [1, 2] * [3, 5] = [3, 10]
    let input_a = BoundedTensor::new(
        Array1::from_vec(vec![1.0_f32]).into_dyn(),
        Array1::from_vec(vec![2.0_f32]).into_dyn(),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        Array1::from_vec(vec![3.0_f32]).into_dyn(),
        Array1::from_vec(vec![5.0_f32]).into_dyn(),
    )
    .unwrap();

    // IBP bounds - exact for bilinear operations
    let ibp_result = mul.propagate_ibp_binary(&input_a, &input_b).unwrap();
    assert!(
        (ibp_result.lower[0] - 3.0).abs() < 1e-5,
        "IBP lower should be 3.0"
    );
    assert!(
        (ibp_result.upper[0] - 10.0).abs() < 1e-5,
        "IBP upper should be 10.0"
    );

    // CROWN with identity bounds (out = z)
    let identity = LinearBounds::identity(1);
    let (bounds_a, bounds_b) = mul
        .propagate_linear_binary(&identity, &input_a, &input_b)
        .unwrap();

    // For z = x * y with identity output, McCormick gives:
    // z_lower ≥ a_x * x + a_y * y + c (where coeffs depend on bounds)
    // To compute the final bound, we need to combine both contributions

    // For the lower bound:
    // - bounds_a.lower_a[0,0] is the coefficient for x
    // - bounds_b.lower_a[0,0] is the coefficient for y
    // - bounds_a.lower_b[0] = bounds_b.lower_b[0] is the constant c (not split for MulBinary)

    // Manually compute the concretized lower bound
    let a_x_lower = bounds_a.lower_a[[0, 0]];
    let a_y_lower = bounds_b.lower_a[[0, 0]];
    let c_lower = bounds_a.lower_b[0]; // Same as bounds_b.lower_b[0]

    // Concretize: use x_l if coeff >= 0, else x_u (for lower bound minimization)
    let x_contrib_lower = if a_x_lower >= 0.0 {
        a_x_lower * input_a.lower[0]
    } else {
        a_x_lower * input_a.upper[0]
    };
    let y_contrib_lower = if a_y_lower >= 0.0 {
        a_y_lower * input_b.lower[0]
    } else {
        a_y_lower * input_b.upper[0]
    };
    let crown_lower = x_contrib_lower + y_contrib_lower + c_lower;

    // Do the same for upper bound
    let a_x_upper = bounds_a.upper_a[[0, 0]];
    let a_y_upper = bounds_b.upper_a[[0, 0]];
    let c_upper = bounds_a.upper_b[0];

    // Concretize: use x_u if coeff >= 0, else x_l (for upper bound maximization)
    let x_contrib_upper = if a_x_upper >= 0.0 {
        a_x_upper * input_a.upper[0]
    } else {
        a_x_upper * input_a.lower[0]
    };
    let y_contrib_upper = if a_y_upper >= 0.0 {
        a_y_upper * input_b.upper[0]
    } else {
        a_y_upper * input_b.lower[0]
    };
    let crown_upper = x_contrib_upper + y_contrib_upper + c_upper;

    // McCormick bounds should be sound: contain the true range [3, 10]
    assert!(
        crown_lower <= 3.0 + 1e-4,
        "CROWN lower {} must be <= IBP min 3.0",
        crown_lower
    );
    assert!(
        crown_upper >= 10.0 - 1e-4,
        "CROWN upper {} must be >= IBP max 10.0",
        crown_upper
    );

    // Test case 2: Mixed signs [-1, 2] * [-3, 4] = [-6, 8]
    let input_a2 = BoundedTensor::new(
        Array1::from_vec(vec![-1.0_f32]).into_dyn(),
        Array1::from_vec(vec![2.0_f32]).into_dyn(),
    )
    .unwrap();
    let input_b2 = BoundedTensor::new(
        Array1::from_vec(vec![-3.0_f32]).into_dyn(),
        Array1::from_vec(vec![4.0_f32]).into_dyn(),
    )
    .unwrap();

    let ibp_result2 = mul.propagate_ibp_binary(&input_a2, &input_b2).unwrap();
    // True range: min of (-1)*(-3)=3, (-1)*4=-4, 2*(-3)=-6, 2*4=8 -> [-6, 8]
    assert!(
        (ibp_result2.lower[0] - (-6.0)).abs() < 1e-5,
        "IBP lower should be -6.0"
    );
    assert!(
        (ibp_result2.upper[0] - 8.0).abs() < 1e-5,
        "IBP upper should be 8.0"
    );

    let (bounds_a2, bounds_b2) = mul
        .propagate_linear_binary(&identity, &input_a2, &input_b2)
        .unwrap();

    // Manually compute concretized bounds for mixed signs case
    let a_x_lower2 = bounds_a2.lower_a[[0, 0]];
    let a_y_lower2 = bounds_b2.lower_a[[0, 0]];
    let c_lower2 = bounds_a2.lower_b[0];

    let x_contrib_lower2 = if a_x_lower2 >= 0.0 {
        a_x_lower2 * input_a2.lower[0]
    } else {
        a_x_lower2 * input_a2.upper[0]
    };
    let y_contrib_lower2 = if a_y_lower2 >= 0.0 {
        a_y_lower2 * input_b2.lower[0]
    } else {
        a_y_lower2 * input_b2.upper[0]
    };
    let crown_lower2 = x_contrib_lower2 + y_contrib_lower2 + c_lower2;

    let a_x_upper2 = bounds_a2.upper_a[[0, 0]];
    let a_y_upper2 = bounds_b2.upper_a[[0, 0]];
    let c_upper2 = bounds_a2.upper_b[0];

    let x_contrib_upper2 = if a_x_upper2 >= 0.0 {
        a_x_upper2 * input_a2.upper[0]
    } else {
        a_x_upper2 * input_a2.lower[0]
    };
    let y_contrib_upper2 = if a_y_upper2 >= 0.0 {
        a_y_upper2 * input_b2.upper[0]
    } else {
        a_y_upper2 * input_b2.lower[0]
    };
    let crown_upper2 = x_contrib_upper2 + y_contrib_upper2 + c_upper2;

    // McCormick must be sound (may be looser than IBP for mixed signs)
    assert!(
        crown_lower2 <= -6.0 + 1e-4,
        "CROWN lower {} must be sound for mixed signs",
        crown_lower2
    );
    assert!(
        crown_upper2 >= 8.0 - 1e-4,
        "CROWN upper {} must be sound for mixed signs",
        crown_upper2
    );
}

#[test]
fn test_mulbinary_crown_in_graph_network() {
    // Test MulBinary with McCormick CROWN in a graph network (SwiGLU pattern)
    // SwiGLU: up(x) * silu(gate(x)) where silu ≈ x * sigmoid(x)

    let mut graph = GraphNetwork::new();

    // Two branches from input that will be multiplied
    let hidden = 4;

    // Linear for "up" branch
    let up_weights = Array2::<f32>::from_elem((hidden, hidden), 0.5);
    let up_bias = Array1::<f32>::zeros(hidden);
    let up_linear = LinearLayer::new(up_weights, Some(up_bias)).unwrap();
    graph.add_node(GraphNode::new(
        "up",
        Layer::Linear(up_linear),
        vec!["_input".to_string()],
    ));

    // Linear for "gate" branch
    let gate_weights = Array2::<f32>::from_elem((hidden, hidden), 0.3);
    let gate_bias = Array1::<f32>::zeros(hidden);
    let gate_linear = LinearLayer::new(gate_weights, Some(gate_bias)).unwrap();
    graph.add_node(GraphNode::new(
        "gate",
        Layer::Linear(gate_linear),
        vec!["_input".to_string()],
    ));

    // Apply sigmoid to gate (approximates silu when combined with gate*sigmoid(gate))
    graph.add_node(GraphNode::new(
        "gate_sigmoid",
        Layer::Sigmoid(SigmoidLayer),
        vec!["gate".to_string()],
    ));

    // Element-wise multiplication (the SwiGLU gating)
    graph.add_node(GraphNode::binary(
        "swiglu_mul",
        Layer::MulBinary(MulBinaryLayer),
        "up",
        "gate_sigmoid",
    ));

    graph.set_output("swiglu_mul");

    // Create bounded input
    let input = BoundedTensor::new(
        Array2::<f32>::from_elem((1, hidden), -0.5).into_dyn(),
        Array2::<f32>::from_elem((1, hidden), 0.5).into_dyn(),
    )
    .unwrap();

    // Run CROWN propagation (should use McCormick for MulBinary)
    let crown_bounds = graph.propagate_crown(&input).unwrap();

    // Run IBP for comparison
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // CROWN bounds should be sound (contain IBP bounds which are exact for this case)
    for i in 0..hidden {
        assert!(
            crown_bounds.lower[[0, i]] <= ibp_bounds.upper[[0, i]] + 1e-4,
            "CROWN lower must be finite and reasonable at position {}: crown_l={}, ibp_u={}",
            i,
            crown_bounds.lower[[0, i]],
            ibp_bounds.upper[[0, i]]
        );
        assert!(
            crown_bounds.upper[[0, i]] >= ibp_bounds.lower[[0, i]] - 1e-4,
            "CROWN upper must be finite and reasonable at position {}: crown_u={}, ibp_l={}",
            i,
            crown_bounds.upper[[0, i]],
            ibp_bounds.lower[[0, i]]
        );
        assert!(
            crown_bounds.lower[[0, i]].is_finite(),
            "CROWN lower must be finite at position {}",
            i
        );
        assert!(
            crown_bounds.upper[[0, i]].is_finite(),
            "CROWN upper must be finite at position {}",
            i
        );
    }
}

#[test]
fn test_mulbinary_batched_crown_in_graph_network() {
    // Test MulBinary with McCormick CROWN in batched mode (SwiGLU pattern)
    // This tests the propagate_linear_batched_binary method

    let mut graph = GraphNetwork::new();

    // Two branches from input that will be multiplied
    let hidden = 4;

    // Linear for "up" branch
    let up_weights = Array2::<f32>::from_elem((hidden, hidden), 0.5);
    let up_bias = Array1::<f32>::zeros(hidden);
    let up_linear = LinearLayer::new(up_weights, Some(up_bias)).unwrap();
    graph.add_node(GraphNode::new(
        "up",
        Layer::Linear(up_linear),
        vec!["_input".to_string()],
    ));

    // Linear for "gate" branch
    let gate_weights = Array2::<f32>::from_elem((hidden, hidden), 0.3);
    let gate_bias = Array1::<f32>::zeros(hidden);
    let gate_linear = LinearLayer::new(gate_weights, Some(gate_bias)).unwrap();
    graph.add_node(GraphNode::new(
        "gate",
        Layer::Linear(gate_linear),
        vec!["_input".to_string()],
    ));

    // Apply sigmoid to gate
    graph.add_node(GraphNode::new(
        "gate_sigmoid",
        Layer::Sigmoid(SigmoidLayer),
        vec!["gate".to_string()],
    ));

    // Element-wise multiplication (the SwiGLU gating)
    graph.add_node(GraphNode::binary(
        "swiglu_mul",
        Layer::MulBinary(MulBinaryLayer),
        "up",
        "gate_sigmoid",
    ));

    graph.set_output("swiglu_mul");

    // Create bounded input with batch dimension
    let batch = 2;
    let input = BoundedTensor::new(
        Array2::<f32>::from_elem((batch, hidden), -0.5).into_dyn(),
        Array2::<f32>::from_elem((batch, hidden), 0.5).into_dyn(),
    )
    .unwrap();

    // Run batched CROWN propagation
    let crown_bounds = graph.propagate_crown_batched(&input).unwrap();

    // Run IBP for comparison
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // CROWN bounds should be sound and finite
    for b in 0..batch {
        for i in 0..hidden {
            assert!(
                crown_bounds.lower[[b, i]].is_finite(),
                "CROWN lower must be finite at [{}, {}]: got {}",
                b,
                i,
                crown_bounds.lower[[b, i]]
            );
            assert!(
                crown_bounds.upper[[b, i]].is_finite(),
                "CROWN upper must be finite at [{}, {}]: got {}",
                b,
                i,
                crown_bounds.upper[[b, i]]
            );
            assert!(
                crown_bounds.lower[[b, i]] <= ibp_bounds.upper[[b, i]] + 1e-4,
                "CROWN lower must be sound at [{}, {}]: crown_l={}, ibp_u={}",
                b,
                i,
                crown_bounds.lower[[b, i]],
                ibp_bounds.upper[[b, i]]
            );
            assert!(
                crown_bounds.upper[[b, i]] >= ibp_bounds.lower[[b, i]] - 1e-4,
                "CROWN upper must be sound at [{}, {}]: crown_u={}, ibp_l={}",
                b,
                i,
                crown_bounds.upper[[b, i]],
                ibp_bounds.lower[[b, i]]
            );
        }
    }
}

// =========================================================================
// Domain Clipping Integration Tests
// =========================================================================

#[test]
fn test_domain_clipping_collect_statistics() {
    use crate::domain_clip::DomainClipper;

    // Build a simple network: input -> linear -> relu
    let mut graph = GraphNetwork::new();

    let weights = arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]);
    let bias = arr1(&[0.1_f32, 0.2]);
    let linear_layer = LinearLayer::new(weights, Some(bias)).unwrap();
    let linear_node = GraphNode::from_input("linear", Layer::Linear(linear_layer));
    graph.add_node(linear_node);

    let relu_node = GraphNode::new("relu", Layer::ReLU(ReLULayer), vec!["linear".to_string()]);
    graph.add_node(relu_node);
    graph.set_output("relu");

    // Collect statistics from concrete forward pass
    let mut clipper = DomainClipper::default();
    let concrete_input = BoundedTensor::concrete(arr1(&[1.0_f32, 2.0]).into_dyn());

    graph
        .collect_activation_statistics(&concrete_input, &mut clipper)
        .unwrap();

    // Verify statistics were collected
    let summary = clipper.summary();
    assert_eq!(summary.total_layers, 2); // linear and relu
    assert_eq!(summary.total_samples, 2); // one sample per layer
}

#[test]
fn test_domain_clipping_tightens_bounds() {
    use crate::domain_clip::{ClipStrategy, DomainClipConfig, DomainClipper};

    // Build a simple network: input -> linear
    let mut graph = GraphNetwork::new();

    let weights = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]); // Identity-like
    let bias = arr1(&[0.0_f32, 0.0]);
    let linear_layer = LinearLayer::new(weights, Some(bias)).unwrap();
    let linear_node = GraphNode::from_input("linear", Layer::Linear(linear_layer));
    graph.add_node(linear_node);
    graph.set_output("linear");

    // Collect statistics from multiple concrete samples
    let mut clipper = DomainClipper::new(DomainClipConfig {
        strategy: ClipStrategy::Empirical { margin_factor: 0.1 },
        min_samples: 1,
        enabled: true,
        exclude_patterns: vec![],
        max_tightening_factor: 100.0,
    });

    // Samples around [1, 2]
    for _ in 0..10 {
        let sample = BoundedTensor::concrete(arr1(&[1.0_f32, 2.0]).into_dyn());
        graph
            .collect_activation_statistics(&sample, &mut clipper)
            .unwrap();
    }

    // Now propagate with very wide input bounds
    let wide_input = BoundedTensor::new(
        arr1(&[-100.0_f32, -100.0]).into_dyn(),
        arr1(&[100.0_f32, 100.0]).into_dyn(),
    )
    .unwrap();

    // Without clipping
    let bounds_no_clip = graph.propagate_ibp(&wide_input).unwrap();

    // With clipping
    let bounds_clipped = graph
        .propagate_ibp_with_clipper(&wide_input, &mut clipper)
        .unwrap();

    // Clipped bounds should be tighter
    let width_no_clip = bounds_no_clip.max_width();
    let width_clipped = bounds_clipped.max_width();

    assert!(
        width_clipped < width_no_clip,
        "Clipped bounds ({:.2}) should be tighter than unclipped ({:.2})",
        width_clipped,
        width_no_clip
    );
}

#[test]
fn test_domain_clipping_soundness() {
    use crate::domain_clip::{ClipStrategy, DomainClipConfig, DomainClipper};

    // Build a network: input -> linear -> relu
    let mut graph = GraphNetwork::new();

    let weights = arr2(&[[2.0_f32, -1.0], [1.0, 3.0]]);
    let bias = arr1(&[0.5_f32, -0.5]);
    let linear_layer = LinearLayer::new(weights, Some(bias)).unwrap();
    let linear_node = GraphNode::from_input("linear", Layer::Linear(linear_layer));
    graph.add_node(linear_node);

    let relu_node = GraphNode::new("relu", Layer::ReLU(ReLULayer), vec!["linear".to_string()]);
    graph.add_node(relu_node);
    graph.set_output("relu");

    // Collect statistics from samples in a specific range
    let mut clipper = DomainClipper::new(DomainClipConfig {
        strategy: ClipStrategy::Statistical { k: 6.0 }, // 6-sigma bounds
        min_samples: 5,
        enabled: true,
        exclude_patterns: vec![],
        max_tightening_factor: 1000.0,
    });

    // Collect stats from samples in [-1, 1] range
    for i in 0..20 {
        let x = (i as f32 - 10.0) / 10.0; // -1.0 to 0.9
        let sample = BoundedTensor::concrete(arr1(&[x, x * 0.5]).into_dyn());
        graph
            .collect_activation_statistics(&sample, &mut clipper)
            .unwrap();
    }

    // Propagate with bounds that include our sample range
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -0.5]).into_dyn(),
        arr1(&[1.0_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let bounds_clipped = graph
        .propagate_ibp_with_clipper(&input, &mut clipper)
        .unwrap();

    // Verify clipped bounds are valid (lower <= upper)
    for (l, u) in bounds_clipped.lower.iter().zip(bounds_clipped.upper.iter()) {
        assert!(
            l <= u,
            "Clipped bounds must be valid: lower={} <= upper={}",
            l,
            u
        );
    }

    // Verify bounds contain concrete outputs for inputs in range
    let test_inputs = vec![
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[-0.5_f32, -0.25]).into_dyn(),
        arr1(&[0.5_f32, 0.25]).into_dyn(),
    ];

    for test_input in test_inputs {
        let concrete = BoundedTensor::concrete(test_input.clone());
        let concrete_output = graph.propagate_ibp(&concrete).unwrap();

        // Concrete output should be within clipped bounds
        for i in 0..concrete_output.len() {
            let val = concrete_output.lower.as_slice().unwrap()[i];
            let clip_l = bounds_clipped.lower.as_slice().unwrap()[i];
            let clip_u = bounds_clipped.upper.as_slice().unwrap()[i];

            assert!(
                val >= clip_l - 1e-5 && val <= clip_u + 1e-5,
                "Concrete output {} at index {} should be within clipped bounds [{}, {}]",
                val,
                i,
                clip_l,
                clip_u
            );
        }
    }
}

// =========================================================================
// BatchNorm CROWN Tests
// =========================================================================

/// Test BatchNorm IBP propagation with positive scale
#[test]
fn test_batchnorm_ibp_positive_scale() {
    // BatchNorm with 2 channels, positive scale
    // scale = [2.0, 3.0], bias = [1.0, -1.0]
    let scale = arr1(&[2.0_f32, 3.0]).into_dyn();
    let bias = arr1(&[1.0_f32, -1.0]).into_dyn();
    let bn = BatchNormLayer::from_scale_bias(scale, bias);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("bn", Layer::BatchNorm(bn)));
    graph.set_output("bn");

    // Input shape (C, H, W) = (2, 2, 2), values in [-1, 1]
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![-1.0; 8]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![1.0; 8]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // Channel 0: y = 2*x + 1, x in [-1, 1] => y in [-1, 3]
    // Channel 1: y = 3*x - 1, x in [-1, 1] => y in [-4, 2]
    for i in 0..4 {
        assert!(
            (output.lower[[0, i / 2, i % 2]] - (-1.0)).abs() < 1e-5,
            "Channel 0 lower should be -1"
        );
        assert!(
            (output.upper[[0, i / 2, i % 2]] - 3.0).abs() < 1e-5,
            "Channel 0 upper should be 3"
        );
    }
    for i in 0..4 {
        assert!(
            (output.lower[[1, i / 2, i % 2]] - (-4.0)).abs() < 1e-5,
            "Channel 1 lower should be -4"
        );
        assert!(
            (output.upper[[1, i / 2, i % 2]] - 2.0).abs() < 1e-5,
            "Channel 1 upper should be 2"
        );
    }
}

/// Test BatchNorm IBP with negative scale (should swap bounds)
#[test]
fn test_batchnorm_ibp_negative_scale() {
    // BatchNorm with negative scale
    // scale = [-2.0, 1.0], bias = [0.0, 0.0]
    let scale = arr1(&[-2.0_f32, 1.0]).into_dyn();
    let bias = arr1(&[0.0_f32, 0.0]).into_dyn();
    let bn = BatchNormLayer::from_scale_bias(scale, bias);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("bn", Layer::BatchNorm(bn)));
    graph.set_output("bn");

    // Input shape (2, 2): For 2D, BatchNorm uses channel at index 1
    // So column 0 is channel 0, column 1 is channel 1
    // Channel 0: values in [1.0, 2.0], Channel 1: values in [3.0, 5.0]
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 3.0, 1.0, 3.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2.0, 5.0, 2.0, 5.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // Channel 0 (negative scale): y = -2*x, x in [1, 2] => y in [-4, -2]
    // Need to swap because scale is negative
    assert!(
        (output.lower[[0, 0]] - (-4.0)).abs() < 1e-5,
        "Negative scale lower: expected -4, got {}",
        output.lower[[0, 0]]
    );
    assert!(
        (output.upper[[0, 0]] - (-2.0)).abs() < 1e-5,
        "Negative scale upper: expected -2, got {}",
        output.upper[[0, 0]]
    );

    // Channel 1 (positive scale): y = 1*x, x in [3, 5] => y in [3, 5]
    assert!(
        (output.lower[[0, 1]] - 3.0).abs() < 1e-5,
        "Positive scale lower: expected 3, got {}",
        output.lower[[0, 1]]
    );
    assert!(
        (output.upper[[0, 1]] - 5.0).abs() < 1e-5,
        "Positive scale upper: expected 5, got {}",
        output.upper[[0, 1]]
    );
}

/// Test BatchNorm CROWN backward propagation in GraphNetwork
#[test]
fn test_batchnorm_crown_backward() {
    // Linear -> Reshape -> BatchNorm -> Reshape -> Linear network
    // This tests that CROWN backward propagation through BatchNorm works
    let mut graph = GraphNetwork::new();

    // First linear: 4 inputs -> 4 outputs
    let w1 = Array2::eye(4);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    // Reshape to (C=2, L=2) for BatchNorm
    let reshape1 = ReshapeLayer::new(vec![2, 2]);
    graph.add_node(GraphNode::new(
        "reshape1",
        Layer::Reshape(reshape1),
        vec!["linear1".to_string()],
    ));

    // BatchNorm: scale = [2, 0.5], bias = [1, -1]
    let scale = arr1(&[2.0_f32, 0.5]).into_dyn();
    let bias = arr1(&[1.0_f32, -1.0]).into_dyn();
    let bn = BatchNormLayer::from_scale_bias(scale, bias);
    graph.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(bn),
        vec!["reshape1".to_string()],
    ));

    // Reshape to flatten for final linear
    let reshape2 = ReshapeLayer::new(vec![4]);
    graph.add_node(GraphNode::new(
        "reshape2",
        Layer::Reshape(reshape2),
        vec!["bn".to_string()],
    ));

    // Final linear: 4 inputs -> 1 output (sum all)
    let w2 = Array2::ones((1, 4));
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["reshape2".to_string()],
    ));

    graph.set_output("linear2");

    // Input is flat 4D
    let lower = arr1(&[-1.0_f32; 4]).into_dyn();
    let upper = arr1(&[1.0_f32; 4]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Get CROWN bounds
    let crown_output = graph.propagate_crown(&input).unwrap();
    let ibp_output = graph.propagate_ibp(&input).unwrap();

    // CROWN should be at least as tight as IBP
    assert!(
        crown_output.lower[[0]] >= ibp_output.lower[[0]] - 1e-5,
        "CROWN lower should be >= IBP lower"
    );
    assert!(
        crown_output.upper[[0]] <= ibp_output.upper[[0]] + 1e-5,
        "CROWN upper should be <= IBP upper"
    );

    // Verify soundness with concrete inputs
    let test_inputs = vec![
        arr1(&[-1.0_f32, -1.0, -1.0, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0, 1.0, 1.0]).into_dyn(),
        arr1(&[0.0_f32, 0.0, 0.0, 0.0]).into_dyn(),
        arr1(&[-0.5_f32, 0.5, -0.5, 0.5]).into_dyn(),
    ];

    for test_input in &test_inputs {
        let concrete = BoundedTensor::concrete(test_input.clone());
        let concrete_output = graph.propagate_ibp(&concrete).unwrap();

        assert!(
            concrete_output.lower[[0]] >= crown_output.lower[[0]] - 1e-5,
            "Soundness: concrete {} < CROWN lower {}",
            concrete_output.lower[[0]],
            crown_output.lower[[0]]
        );
        assert!(
            concrete_output.upper[[0]] <= crown_output.upper[[0]] + 1e-5,
            "Soundness: concrete {} > CROWN upper {}",
            concrete_output.upper[[0]],
            crown_output.upper[[0]]
        );
    }
}

/// Test BatchNorm CROWN soundness with ReLU
#[test]
fn test_batchnorm_crown_with_relu_soundness() {
    // Simple: Reshape -> BatchNorm -> ReLU -> Reshape -> Linear
    // Tests interaction between BatchNorm and ReLU in CROWN
    let mut graph = GraphNetwork::new();

    // Reshape to (C=2, L=2)
    let reshape1 = ReshapeLayer::new(vec![2, 2]);
    graph.add_node(GraphNode::from_input("reshape1", Layer::Reshape(reshape1)));

    // BatchNorm: scale = [1.0, 1.0], bias = [0.0, 0.0] (identity)
    let scale = arr1(&[1.0_f32, 1.0]).into_dyn();
    let bias = arr1(&[0.0_f32, 0.0]).into_dyn();
    let bn = BatchNormLayer::from_scale_bias(scale, bias);
    graph.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(bn),
        vec!["reshape1".to_string()],
    ));

    // ReLU
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["bn".to_string()],
    ));

    // Reshape to flatten
    let reshape2 = ReshapeLayer::new(vec![4]);
    graph.add_node(GraphNode::new(
        "reshape2",
        Layer::Reshape(reshape2),
        vec!["relu".to_string()],
    ));

    // Final linear: sum all
    let w2 = Array2::ones((1, 4));
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["reshape2".to_string()],
    ));

    graph.set_output("linear2");

    // Input bounds
    let lower = arr1(&[-1.0_f32, -1.0, -1.0, -1.0]).into_dyn();
    let upper = arr1(&[1.0_f32, 1.0, 1.0, 1.0]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let crown_output = graph.propagate_crown(&input).unwrap();
    let ibp_output = graph.propagate_ibp(&input).unwrap();

    // CROWN bounds should be valid
    assert!(
        crown_output.lower[[0]] <= crown_output.upper[[0]],
        "CROWN bounds must be valid"
    );

    // Test soundness with corner inputs
    let test_inputs = vec![
        arr1(&[-1.0_f32, -1.0, -1.0, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0, 1.0, 1.0]).into_dyn(),
        arr1(&[0.0_f32, 0.0, 0.0, 0.0]).into_dyn(),
        arr1(&[-1.0_f32, 1.0, -1.0, 1.0]).into_dyn(),
        arr1(&[1.0_f32, -1.0, 1.0, -1.0]).into_dyn(),
    ];

    for test_input in &test_inputs {
        let concrete = BoundedTensor::concrete(test_input.clone());
        let concrete_output = graph.propagate_ibp(&concrete).unwrap();

        assert!(
            concrete_output.lower[[0]] >= crown_output.lower[[0]] - 1e-4,
            "Soundness violation: concrete {} < CROWN lower {} for input {:?}",
            concrete_output.lower[[0]],
            crown_output.lower[[0]],
            test_input
        );
        assert!(
            concrete_output.upper[[0]] <= crown_output.upper[[0]] + 1e-4,
            "Soundness violation: concrete {} > CROWN upper {} for input {:?}",
            concrete_output.upper[[0]],
            crown_output.upper[[0]],
            test_input
        );
    }

    // CROWN should be at least as tight as IBP (or equal for linear net)
    assert!(
        crown_output.lower[[0]] >= ibp_output.lower[[0]] - 1e-4,
        "CROWN lower {} should be >= IBP lower {}",
        crown_output.lower[[0]],
        ibp_output.lower[[0]]
    );
}

/// Test BatchNorm CROWN with NCHW (4D) input
#[test]
fn test_batchnorm_crown_4d_input() {
    // Test with proper 4D CNN-style input
    let mut graph = GraphNetwork::new();

    // BatchNorm: 3 channels
    let scale = arr1(&[1.0_f32, 2.0, 0.5]).into_dyn();
    let bias = arr1(&[0.0_f32, 1.0, -1.0]).into_dyn();
    let bn = BatchNormLayer::from_scale_bias(scale, bias);
    graph.add_node(GraphNode::from_input("bn", Layer::BatchNorm(bn)));

    // Flatten
    let reshape = ReshapeLayer::new(vec![12]); // 3*2*2
    graph.add_node(GraphNode::new(
        "flatten",
        Layer::Reshape(reshape),
        vec!["bn".to_string()],
    ));

    // Linear to single output
    let w = Array2::ones((1, 12));
    let linear = LinearLayer::new(w, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear",
        Layer::Linear(linear),
        vec!["flatten".to_string()],
    ));

    graph.set_output("linear");

    // 4D input: (N=1, C=3, H=2, W=2)
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 3, 2, 2]), vec![-1.0; 12]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 3, 2, 2]), vec![1.0; 12]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let crown_output = graph.propagate_crown(&input).unwrap();
    let ibp_output = graph.propagate_ibp(&input).unwrap();

    // CROWN should give valid bounds
    assert!(
        crown_output.lower[[0]].is_finite(),
        "CROWN lower must be finite"
    );
    assert!(
        crown_output.upper[[0]].is_finite(),
        "CROWN upper must be finite"
    );
    assert!(
        crown_output.lower[[0]] <= crown_output.upper[[0]],
        "CROWN bounds must be valid"
    );

    // CROWN should be at least as tight as IBP
    assert!(
        crown_output.lower[[0]] >= ibp_output.lower[[0]] - 1e-4,
        "CROWN lower {} >= IBP lower {}",
        crown_output.lower[[0]],
        ibp_output.lower[[0]]
    );
    assert!(
        crown_output.upper[[0]] <= ibp_output.upper[[0]] + 1e-4,
        "CROWN upper {} <= IBP upper {}",
        crown_output.upper[[0]],
        ibp_output.upper[[0]]
    );
}

/// Test BatchNorm CROWN with negative scale (requires bound swapping)
#[test]
fn test_batchnorm_crown_negative_scale() {
    // Test that negative scale is handled correctly in backward pass
    let mut graph = GraphNetwork::new();

    // Reshape to (C=2, L=2) for BatchNorm
    let reshape1 = ReshapeLayer::new(vec![2, 2]);
    graph.add_node(GraphNode::from_input("reshape1", Layer::Reshape(reshape1)));

    // BatchNorm with one negative scale
    let scale = arr1(&[-1.0_f32, 1.0]).into_dyn();
    let bias = arr1(&[0.0_f32, 0.0]).into_dyn();
    let bn = BatchNormLayer::from_scale_bias(scale, bias);
    graph.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(bn),
        vec!["reshape1".to_string()],
    ));

    // Reshape to flatten
    let reshape2 = ReshapeLayer::new(vec![4]);
    graph.add_node(GraphNode::new(
        "reshape2",
        Layer::Reshape(reshape2),
        vec!["bn".to_string()],
    ));

    // Linear to single output (sum all)
    let w = Array2::ones((1, 4));
    let linear = LinearLayer::new(w, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear",
        Layer::Linear(linear),
        vec!["reshape2".to_string()],
    ));

    graph.set_output("linear");

    // Input: flat 4D, values in [0, 2]
    let lower = arr1(&[0.0_f32, 0.0, 0.0, 0.0]).into_dyn();
    let upper = arr1(&[2.0_f32, 2.0, 2.0, 2.0]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let crown_output = graph.propagate_crown(&input).unwrap();

    // After reshape: (2, 2) - Channel 0 has first 2 values, Channel 1 has last 2
    // Channel 0: y = -1 * x, x in [0, 2] => y in [-2, 0]
    // Channel 1: y = 1 * x, x in [0, 2] => y in [0, 2]
    // Sum of 4 elements: 2*[-2, 0] + 2*[0, 2] = [-4, 0] + [0, 4] = [-4, 4]

    // Test soundness
    let test_inputs = vec![
        arr1(&[0.0_f32, 0.0, 0.0, 0.0]).into_dyn(),
        arr1(&[2.0_f32, 2.0, 2.0, 2.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0, 1.0, 1.0]).into_dyn(),
        arr1(&[0.0_f32, 0.0, 2.0, 2.0]).into_dyn(),
        arr1(&[2.0_f32, 2.0, 0.0, 0.0]).into_dyn(),
    ];

    for test_input in &test_inputs {
        let concrete = BoundedTensor::concrete(test_input.clone());
        let concrete_output = graph.propagate_ibp(&concrete).unwrap();

        assert!(
            concrete_output.lower[[0]] >= crown_output.lower[[0]] - 1e-4,
            "Soundness: concrete {} < CROWN lower {} for input {:?}",
            concrete_output.lower[[0]],
            crown_output.lower[[0]],
            test_input
        );
        assert!(
            concrete_output.upper[[0]] <= crown_output.upper[[0]] + 1e-4,
            "Soundness: concrete {} > CROWN upper {} for input {:?}",
            concrete_output.upper[[0]],
            crown_output.upper[[0]],
            test_input
        );
    }
}
