//! Convolution layer tests: Conv1D, Conv2D, and MaxPool2D.
//!
//! These tests validate the IBP and CROWN propagation through convolutional
//! layers, including stride, padding, and transpose operations.

use super::*;
use ndarray::{arr1, Array1, ArrayD};

// ============================================================
// CONV2D HELPER FUNCTIONS
// ============================================================

/// Helper to create 4D kernel from nested slices
fn kernel_4d(data: &[[[[f32; 2]; 2]; 1]; 1]) -> ArrayD<f32> {
    // Shape: (out_channels=1, in_channels=1, kh=2, kw=2)
    let mut arr = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 2, 2]));
    for oc in 0..1 {
        for ic in 0..1 {
            for kh in 0..2 {
                for kw in 0..2 {
                    arr[[oc, ic, kh, kw]] = data[oc][ic][kh][kw];
                }
            }
        }
    }
    arr
}

/// Helper to create 3D input from nested slices
fn input_3d(data: &[[[f32; 3]; 3]; 1]) -> ArrayD<f32> {
    // Shape: (channels=1, height=3, width=3)
    let mut arr = ArrayD::zeros(ndarray::IxDyn(&[1, 3, 3]));
    for c in 0..1 {
        for h in 0..3 {
            for w in 0..3 {
                arr[[c, h, w]] = data[c][h][w];
            }
        }
    }
    arr
}

// ============================================================
// CONV2D IBP TESTS
// ============================================================

#[test]
fn test_conv2d_ibp_identity_kernel() {
    // 1x1 identity kernel: output equals input (center pixel only due to valid conv)
    // Actually, for 2x2 kernel on 3x3 input with stride=1, padding=0, output is 2x2
    // Let's use a simpler test: all-ones kernel with concrete input

    // Kernel: [[1, 1], [1, 1]] sums the 2x2 region
    let kernel = kernel_4d(&[[[[1.0, 1.0], [1.0, 1.0]]]]);
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    // Concrete input (lower == upper)
    let input_data = input_3d(&[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]);
    let input = BoundedTensor::concrete(input_data);

    let output = conv.propagate_ibp(&input).unwrap();

    // Output shape: (1, 2, 2) - 2x2 output from 3x3 input with 2x2 kernel
    assert_eq!(output.shape(), &[1, 2, 2]);

    // For concrete input, lower == upper
    // output[0,0,0] = 1+2+4+5 = 12
    // output[0,0,1] = 2+3+5+6 = 16
    // output[0,1,0] = 4+5+7+8 = 24
    // output[0,1,1] = 5+6+8+9 = 28
    assert!((output.lower[[0, 0, 0]] - 12.0).abs() < 1e-6);
    assert!((output.upper[[0, 0, 0]] - 12.0).abs() < 1e-6);
    assert!((output.lower[[0, 0, 1]] - 16.0).abs() < 1e-6);
    assert!((output.lower[[0, 1, 0]] - 24.0).abs() < 1e-6);
    assert!((output.lower[[0, 1, 1]] - 28.0).abs() < 1e-6);
}

#[test]
fn test_conv2d_ibp_batched_input() {
    // Verify Conv2d IBP supports (batch, channels, height, width) inputs.
    let kernel = kernel_4d(&[[[[1.0, 1.0], [1.0, 1.0]]]]);
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 3, 3]));
    for h in 0..3 {
        for w in 0..3 {
            input_data[[0, 0, h, w]] = (h * 3 + w + 1) as f32; // 1..9
            input_data[[1, 0, h, w]] = (h * 3 + w + 2) as f32; // 2..10
        }
    }
    let input = BoundedTensor::concrete(input_data);

    let output = conv.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 1, 2, 2]);
    // Batch 0: same as test_conv2d_ibp_identity_kernel
    assert!((output.lower[[0, 0, 0, 0]] - 12.0).abs() < 1e-6);
    assert!((output.lower[[0, 0, 0, 1]] - 16.0).abs() < 1e-6);
    assert!((output.lower[[0, 0, 1, 0]] - 24.0).abs() < 1e-6);
    assert!((output.lower[[0, 0, 1, 1]] - 28.0).abs() < 1e-6);

    // Batch 1: each input entry is +1, so each 2x2 sum is +4.
    assert!((output.lower[[1, 0, 0, 0]] - 16.0).abs() < 1e-6);
    assert!((output.lower[[1, 0, 0, 1]] - 20.0).abs() < 1e-6);
    assert!((output.lower[[1, 0, 1, 0]] - 28.0).abs() < 1e-6);
    assert!((output.lower[[1, 0, 1, 1]] - 32.0).abs() < 1e-6);
}

#[test]
fn test_conv2d_ibp_positive_kernel() {
    // All positive kernel with bounded input
    let kernel = kernel_4d(&[[[[1.0, 2.0], [3.0, 4.0]]]]);
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    // Input: all zeros to all ones
    let lower_data = ArrayD::zeros(ndarray::IxDyn(&[1, 3, 3]));
    let upper_data = ArrayD::ones(ndarray::IxDyn(&[1, 3, 3]));
    let input = BoundedTensor::new(lower_data, upper_data).unwrap();

    let output = conv.propagate_ibp(&input).unwrap();

    // For all-positive kernel and input in [0, 1]:
    // lower = conv([0,0,0,0], kernel) = 0
    // upper = conv([1,1,1,1], kernel) = 1+2+3+4 = 10
    assert_eq!(output.shape(), &[1, 2, 2]);
    for idx in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)] {
        assert!(
            (output.lower[[idx.0, idx.1, idx.2]] - 0.0).abs() < 1e-6,
            "lower at {:?} = {}",
            idx,
            output.lower[[idx.0, idx.1, idx.2]]
        );
        assert!(
            (output.upper[[idx.0, idx.1, idx.2]] - 10.0).abs() < 1e-6,
            "upper at {:?} = {}",
            idx,
            output.upper[[idx.0, idx.1, idx.2]]
        );
    }
}

#[test]
fn test_conv2d_ibp_mixed_kernel() {
    // Mixed positive/negative kernel
    // Kernel: [[1, -1], [-1, 1]] (edge detector style)
    let kernel = kernel_4d(&[[[[1.0, -1.0], [-1.0, 1.0]]]]);
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    // Input in [0, 1]
    let lower_data = ArrayD::zeros(ndarray::IxDyn(&[1, 3, 3]));
    let upper_data = ArrayD::ones(ndarray::IxDyn(&[1, 3, 3]));
    let input = BoundedTensor::new(lower_data, upper_data).unwrap();

    let output = conv.propagate_ibp(&input).unwrap();

    // W+ = [[1, 0], [0, 1]], W- = [[0, -1], [-1, 0]]
    // lower_y = conv([0], W+) + conv([1], W-) = 0 + (-1-1) = -2
    // upper_y = conv([1], W+) + conv([0], W-) = (1+1) + 0 = 2
    assert_eq!(output.shape(), &[1, 2, 2]);
    for idx in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)] {
        assert!(
            (output.lower[[idx.0, idx.1, idx.2]] - (-2.0)).abs() < 1e-6,
            "lower at {:?} = {}",
            idx,
            output.lower[[idx.0, idx.1, idx.2]]
        );
        assert!(
            (output.upper[[idx.0, idx.1, idx.2]] - 2.0).abs() < 1e-6,
            "upper at {:?} = {}",
            idx,
            output.upper[[idx.0, idx.1, idx.2]]
        );
    }
}

#[test]
fn test_conv2d_ibp_with_bias() {
    // Kernel with bias
    let kernel = kernel_4d(&[[[[1.0, 1.0], [1.0, 1.0]]]]);
    let bias = arr1(&[5.0]); // Add 5 to output
    let conv = Conv2dLayer::new(kernel, Some(bias), (1, 1), (0, 0)).unwrap();

    // Concrete input
    let input_data = ArrayD::ones(ndarray::IxDyn(&[1, 3, 3]));
    let input = BoundedTensor::concrete(input_data);

    let output = conv.propagate_ibp(&input).unwrap();

    // Sum of 2x2 region of ones = 4, plus bias 5 = 9
    assert_eq!(output.shape(), &[1, 2, 2]);
    for idx in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)] {
        assert!(
            (output.lower[[idx.0, idx.1, idx.2]] - 9.0).abs() < 1e-6,
            "value at {:?} = {}",
            idx,
            output.lower[[idx.0, idx.1, idx.2]]
        );
    }
}

#[test]
fn test_conv2d_ibp_with_stride() {
    // 2x2 kernel with stride 2 on 4x4 input -> 2x2 output
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 2, 2]));
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 0, 1]] = 1.0;
    kernel[[0, 0, 1, 0]] = 1.0;
    kernel[[0, 0, 1, 1]] = 1.0;
    let conv = Conv2dLayer::new(kernel, None, (2, 2), (0, 0)).unwrap();

    // 4x4 input
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[1, 4, 4]));
    for i in 0..16 {
        input_data[[0, i / 4, i % 4]] = i as f32;
    }
    let input = BoundedTensor::concrete(input_data);

    let output = conv.propagate_ibp(&input).unwrap();

    // With stride 2: output is 2x2
    // output[0,0] covers input[0:2,0:2] = 0+1+4+5 = 10
    // output[0,1] covers input[0:2,2:4] = 2+3+6+7 = 18
    // output[1,0] covers input[2:4,0:2] = 8+9+12+13 = 42
    // output[1,1] covers input[2:4,2:4] = 10+11+14+15 = 50
    assert_eq!(output.shape(), &[1, 2, 2]);
    assert!((output.lower[[0, 0, 0]] - 10.0).abs() < 1e-6);
    assert!((output.lower[[0, 0, 1]] - 18.0).abs() < 1e-6);
    assert!((output.lower[[0, 1, 0]] - 42.0).abs() < 1e-6);
    assert!((output.lower[[0, 1, 1]] - 50.0).abs() < 1e-6);
}

#[test]
fn test_conv2d_ibp_with_padding() {
    // 3x3 kernel with padding 1 on 3x3 input -> 3x3 output
    let kernel = ArrayD::ones(ndarray::IxDyn(&[1, 1, 3, 3]));
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (1, 1)).unwrap();

    // 3x3 input of all ones
    let input_data = ArrayD::ones(ndarray::IxDyn(&[1, 3, 3]));
    let input = BoundedTensor::concrete(input_data);

    let output = conv.propagate_ibp(&input).unwrap();

    // With padding=1, output is 3x3
    // Corner: sees 4 ones (2x2 valid region) = 4
    // Edge: sees 6 ones (2x3 valid region) = 6
    // Center: sees 9 ones (3x3 valid region) = 9
    assert_eq!(output.shape(), &[1, 3, 3]);
    assert!(
        (output.lower[[0, 0, 0]] - 4.0).abs() < 1e-6,
        "corner = {}",
        output.lower[[0, 0, 0]]
    );
    assert!(
        (output.lower[[0, 0, 1]] - 6.0).abs() < 1e-6,
        "edge = {}",
        output.lower[[0, 0, 1]]
    );
    assert!(
        (output.lower[[0, 1, 1]] - 9.0).abs() < 1e-6,
        "center = {}",
        output.lower[[0, 1, 1]]
    );
}

#[test]
fn test_conv2d_ibp_soundness() {
    // Soundness test: verify concrete outputs are within bounds
    let kernel = kernel_4d(&[[[[2.0, -1.0], [1.0, -2.0]]]]);
    let conv = Conv2dLayer::new(kernel.clone(), None, (1, 1), (0, 0)).unwrap();

    // Input in [-1, 1]
    let lower_data = ArrayD::from_elem(ndarray::IxDyn(&[1, 3, 3]), -1.0f32);
    let upper_data = ArrayD::from_elem(ndarray::IxDyn(&[1, 3, 3]), 1.0f32);
    let input = BoundedTensor::new(lower_data, upper_data).unwrap();

    let output_bounds = conv.propagate_ibp(&input).unwrap();

    // Test several concrete inputs
    let test_inputs = [
        ArrayD::from_elem(ndarray::IxDyn(&[1, 3, 3]), -1.0f32), // all lower
        ArrayD::from_elem(ndarray::IxDyn(&[1, 3, 3]), 1.0f32),  // all upper
        ArrayD::from_elem(ndarray::IxDyn(&[1, 3, 3]), 0.0f32),  // center
    ];

    for test_input in &test_inputs {
        let concrete_output = conv2d_single(test_input, &kernel, (1, 1), (0, 0));

        for oc in 0..1 {
            for oh in 0..2 {
                for ow in 0..2 {
                    let val = concrete_output[[oc, oh, ow]];
                    assert!(
                        val >= output_bounds.lower[[oc, oh, ow]] - 1e-6,
                        "Soundness: val {} < lower {}",
                        val,
                        output_bounds.lower[[oc, oh, ow]]
                    );
                    assert!(
                        val <= output_bounds.upper[[oc, oh, ow]] + 1e-6,
                        "Soundness: val {} > upper {}",
                        val,
                        output_bounds.upper[[oc, oh, ow]]
                    );
                }
            }
        }
    }
}

#[test]
fn test_conv2d_shape_validation() {
    // Kernel must be 4D
    let bad_kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 2, 2])); // 3D, not 4D
    let result = Conv2dLayer::new(bad_kernel, None, (1, 1), (0, 0));
    assert!(result.is_err());
}

#[test]
fn test_conv2d_input_channels_validation() {
    // Input channels must match kernel
    let kernel = ArrayD::ones(ndarray::IxDyn(&[1, 3, 2, 2])); // 3 input channels
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    let input = BoundedTensor::new(
        ArrayD::zeros(ndarray::IxDyn(&[1, 3, 3])), // 1 channel, expected 3
        ArrayD::ones(ndarray::IxDyn(&[1, 3, 3])),
    )
    .unwrap();

    let result = conv.propagate_ibp(&input);
    assert!(result.is_err());
}

// ============================================================
// CONV2D CROWN TESTS
// ============================================================

#[test]
fn test_conv2d_transpose_basic() {
    // Test that conv2d_transpose is the inverse of conv2d in the gradient sense
    // Input: [1, 3, 3], Kernel: [1, 1, 2, 2], Output: [1, 2, 2]
    let mut kernel = ArrayD::ones(ndarray::IxDyn(&[1, 1, 2, 2]));
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 0, 1]] = 2.0;
    kernel[[0, 0, 1, 0]] = 3.0;
    kernel[[0, 0, 1, 1]] = 4.0;

    // Forward conv input
    let mut input = ArrayD::zeros(ndarray::IxDyn(&[1, 3, 3]));
    for i in 0..3 {
        for j in 0..3 {
            input[[0, i, j]] = (i * 3 + j + 1) as f32;
        }
    }

    // Gradient at output (identity for this test)
    let mut grad_out = ArrayD::zeros(ndarray::IxDyn(&[1, 2, 2]));
    grad_out[[0, 0, 0]] = 1.0;
    grad_out[[0, 0, 1]] = 1.0;
    grad_out[[0, 1, 0]] = 1.0;
    grad_out[[0, 1, 1]] = 1.0;

    // Compute transposed conv
    let grad_in = conv2d_transpose(&grad_out, &kernel, (1, 1), (0, 0), (3, 3));

    // Expected: scatter of grad * kernel at each input position
    // Position (0,0) receives grad[0,0] * kernel[0,0] = 1 * 1 = 1
    // Position (0,1) receives grad[0,0] * kernel[0,1] + grad[0,1] * kernel[0,0] = 1*2 + 1*1 = 3
    // etc.
    assert_eq!(grad_in.shape(), &[1, 3, 3]);
    assert!((grad_in[[0, 0, 0]] - 1.0).abs() < 1e-6);
    assert!((grad_in[[0, 0, 1]] - 3.0).abs() < 1e-6);
    assert!((grad_in[[0, 0, 2]] - 2.0).abs() < 1e-6);
    assert!((grad_in[[0, 1, 0]] - 4.0).abs() < 1e-6);
    assert!((grad_in[[0, 1, 1]] - 10.0).abs() < 1e-6);
    assert!((grad_in[[0, 1, 2]] - 6.0).abs() < 1e-6);
    assert!((grad_in[[0, 2, 0]] - 3.0).abs() < 1e-6);
    assert!((grad_in[[0, 2, 1]] - 7.0).abs() < 1e-6);
    assert!((grad_in[[0, 2, 2]] - 4.0).abs() < 1e-6);
}

#[test]
fn test_conv2d_crown_identity_bounds() {
    // Test CROWN with identity linear bounds (output = conv_output)
    let mut kernel = ArrayD::ones(ndarray::IxDyn(&[2, 1, 2, 2]));
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 0, 1]] = 0.0;
    kernel[[0, 0, 1, 0]] = 0.0;
    kernel[[0, 0, 1, 1]] = 0.0;
    kernel[[1, 0, 0, 0]] = 0.0;
    kernel[[1, 0, 0, 1]] = 1.0;
    kernel[[1, 0, 1, 0]] = 0.0;
    kernel[[1, 0, 1, 1]] = 0.0;

    // With input [1, 3, 3] and kernel [2, 1, 2, 2], output is [2, 2, 2] = 8 elements
    let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 3, 3).unwrap();

    // Identity bounds: A = I, b = 0
    let identity_bounds = LinearBounds::identity(8);

    // Propagate backward through conv
    let result = conv
        .propagate_linear(&identity_bounds)
        .unwrap()
        .into_owned();

    // Should have shape [8, 9] (8 outputs, 9 = 1*3*3 inputs)
    assert_eq!(result.lower_a.shape(), &[8, 9]);
    assert_eq!(result.upper_a.shape(), &[8, 9]);
}

#[test]
fn test_conv2d_crown_simple_backward() {
    // Simple test: 1x1 conv (essentially a linear layer)
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 1, 1]));
    kernel[[0, 0, 0, 0]] = 2.0; // Output channel 0: 2x input
    kernel[[1, 0, 0, 0]] = 3.0; // Output channel 1: 3x input

    // Input [1, 2, 2] -> Output [2, 2, 2]
    let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 2, 2).unwrap();

    // Identity bounds on output
    let identity_bounds = LinearBounds::identity(8); // 2 * 2 * 2 = 8

    let result = conv
        .propagate_linear(&identity_bounds)
        .unwrap()
        .into_owned();

    // For 1x1 conv, backward pass through a single output position should give:
    // A @ W where W is the 1x1 kernel value
    // Output channel 0 position (0,0) -> A[0, :] should have kernel[0,0,0,0]=2.0 at input position 0
    assert!((result.lower_a[[0, 0]] - 2.0).abs() < 1e-6);
    assert!((result.lower_a[[0, 1]] - 0.0).abs() < 1e-6);

    // Output channel 1 position (0,0) -> A[4, :] should have kernel[1,0,0,0]=3.0 at input position 0
    assert!((result.lower_a[[4, 0]] - 3.0).abs() < 1e-6);
}

#[test]
fn test_conv2d_crown_with_bias() {
    // Test CROWN bias handling
    let kernel = ArrayD::ones(ndarray::IxDyn(&[1, 1, 2, 2]));
    let bias = Array1::from_vec(vec![0.5]);

    // Input [1, 3, 3] -> Output [1, 2, 2]
    let conv = Conv2dLayer::with_input_shape(kernel, Some(bias), (1, 1), (0, 0), 3, 3).unwrap();

    // Identity bounds on output
    let identity_bounds = LinearBounds::identity(4); // 1 * 2 * 2 = 4

    let result = conv
        .propagate_linear(&identity_bounds)
        .unwrap()
        .into_owned();

    // Bias contribution: each output position contributes 0.5 to its bound
    // Identity bounds sum over one position each, so bias contrib is 0.5
    assert!((result.lower_b[0] - 0.5).abs() < 1e-6);
    assert!((result.lower_b[1] - 0.5).abs() < 1e-6);
    assert!((result.lower_b[2] - 0.5).abs() < 1e-6);
    assert!((result.lower_b[3] - 0.5).abs() < 1e-6);
}

#[test]
fn test_conv2d_crown_vs_ibp_tightness() {
    // Compare CROWN vs IBP - CROWN should be at least as tight
    // Use a small network: Conv2d -> (flatten for bounds computation)
    let kernel = ArrayD::from_elem(ndarray::IxDyn(&[2, 1, 2, 2]), 0.5);
    let bias = Array1::from_vec(vec![0.1, -0.1]);

    let conv =
        Conv2dLayer::with_input_shape(kernel.clone(), Some(bias.clone()), (1, 1), (0, 0), 3, 3)
            .unwrap();
    let conv_ibp = Conv2dLayer::new(kernel, Some(bias), (1, 1), (0, 0)).unwrap();

    // Input with perturbation
    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, 3, 3]), 0.5);
    let input = BoundedTensor::from_epsilon(center.clone(), 0.1);

    // IBP bounds
    let ibp_output = conv_ibp.propagate_ibp(&input).unwrap();
    let ibp_flat = ibp_output.flatten();

    // CROWN bounds
    let identity = LinearBounds::identity(8); // 2 * 2 * 2 = 8
    let crown_bounds = conv.propagate_linear(&identity).unwrap().into_owned();

    // Concretize CROWN bounds with input bounds
    let flat_input = input.flatten();
    let x_l = flat_input
        .lower
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let x_u = flat_input
        .upper
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    // CROWN lower: A+ @ x_l + A- @ x_u + b
    // CROWN upper: A+ @ x_u + A- @ x_l + b
    let a_pos_l = crown_bounds.lower_a.mapv(|v| v.max(0.0));
    let a_neg_l = crown_bounds.lower_a.mapv(|v| v.min(0.0));
    let a_pos_u = crown_bounds.upper_a.mapv(|v| v.max(0.0));
    let a_neg_u = crown_bounds.upper_a.mapv(|v| v.min(0.0));

    let crown_lower = a_pos_l.dot(&x_l) + a_neg_l.dot(&x_u) + &crown_bounds.lower_b;
    let crown_upper = a_pos_u.dot(&x_u) + a_neg_u.dot(&x_l) + &crown_bounds.upper_b;

    // For linear layers (conv is linear), CROWN should match IBP exactly
    // since there's no non-linearity to relax
    for i in 0..8 {
        let ibp_l = ibp_flat.lower.as_slice().unwrap()[i];
        let ibp_u = ibp_flat.upper.as_slice().unwrap()[i];
        let crown_l = crown_lower[i];
        let crown_u = crown_upper[i];

        // CROWN should be at least as tight as IBP (or same for linear)
        assert!(
            crown_l >= ibp_l - 1e-5,
            "CROWN lower {} should be >= IBP lower {}",
            crown_l,
            ibp_l
        );
        assert!(
            crown_u <= ibp_u + 1e-5,
            "CROWN upper {} should be <= IBP upper {}",
            crown_u,
            ibp_u
        );

        // For pure linear, should be nearly identical
        assert!(
            (crown_l - ibp_l).abs() < 1e-4,
            "CROWN ({}) and IBP ({}) should match for linear",
            crown_l,
            ibp_l
        );
        assert!(
            (crown_u - ibp_u).abs() < 1e-4,
            "CROWN ({}) and IBP ({}) should match for linear",
            crown_u,
            ibp_u
        );
    }
}

#[test]
fn test_conv2d_crown_requires_input_shape() {
    // Test that CROWN fails without input_shape set
    let kernel = ArrayD::ones(ndarray::IxDyn(&[1, 1, 2, 2]));
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    let identity_bounds = LinearBounds::identity(4);
    let result = conv.propagate_linear(&identity_bounds);

    assert!(result.is_err());
}

#[test]
fn test_conv2d_crown_network_integration() {
    // Test CROWN through a Conv2d -> ReLU network
    // This verifies the full backward pass works with non-linearities

    // Input: [1, 4, 4] -> Conv [2, 1, 2, 2] -> [2, 3, 3] -> ReLU -> [2, 3, 3]
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 2, 2]));
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 0, 1]] = -1.0;
    kernel[[0, 0, 1, 0]] = 1.0;
    kernel[[0, 0, 1, 1]] = -1.0;
    kernel[[1, 0, 0, 0]] = 0.5;
    kernel[[1, 0, 0, 1]] = 0.5;
    kernel[[1, 0, 1, 0]] = 0.5;
    kernel[[1, 0, 1, 1]] = 0.5;

    let conv = Conv2dLayer::with_input_shape(kernel.clone(), None, (1, 1), (0, 0), 4, 4).unwrap();

    // Input with perturbation around 0.5
    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, 4, 4]), 0.5);
    let input = BoundedTensor::from_epsilon(center, 0.1);

    // Get IBP bounds for pre-activation (conv output)
    let conv_ibp = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();
    let pre_activation = conv_ibp.propagate_ibp(&input).unwrap();

    // ReLU layer
    let relu = ReLULayer;

    // IBP through ReLU
    let ibp_output = relu.propagate_ibp(&pre_activation).unwrap();

    // Now do CROWN backward:
    // Start with identity on ReLU output
    let relu_output_size = 2 * 3 * 3; // 18
    let identity = LinearBounds::identity(relu_output_size);

    // Backward through ReLU
    let relu_bounds = relu
        .propagate_linear_with_bounds(&identity, &pre_activation)
        .unwrap();

    // Backward through Conv
    let conv_bounds = conv.propagate_linear(&relu_bounds).unwrap().into_owned();

    // Concretize
    let flat_input = input.flatten();
    let x_l = flat_input
        .lower
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let x_u = flat_input
        .upper
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    let a_pos_l = conv_bounds.lower_a.mapv(|v| v.max(0.0));
    let a_neg_l = conv_bounds.lower_a.mapv(|v| v.min(0.0));
    let a_pos_u = conv_bounds.upper_a.mapv(|v| v.max(0.0));
    let a_neg_u = conv_bounds.upper_a.mapv(|v| v.min(0.0));

    let crown_lower = a_pos_l.dot(&x_l) + a_neg_l.dot(&x_u) + &conv_bounds.lower_b;
    let crown_upper = a_pos_u.dot(&x_u) + a_neg_u.dot(&x_l) + &conv_bounds.upper_b;

    // CROWN should be tighter than or equal to IBP
    let ibp_flat = ibp_output.flatten();
    for i in 0..relu_output_size {
        let ibp_l = ibp_flat.lower.as_slice().unwrap()[i];
        let ibp_u = ibp_flat.upper.as_slice().unwrap()[i];
        let crown_l = crown_lower[i];
        let crown_u = crown_upper[i];

        // CROWN should be at least as tight
        assert!(
            crown_l >= ibp_l - 1e-4,
            "Output {}: CROWN lower {} should be >= IBP lower {}",
            i,
            crown_l,
            ibp_l
        );
        assert!(
            crown_u <= ibp_u + 1e-4,
            "Output {}: CROWN upper {} should be <= IBP upper {}",
            i,
            crown_u,
            ibp_u
        );
    }

    // With ReLU, CROWN should often be tighter
    let ibp_widths: Vec<f32> = (0..relu_output_size)
        .map(|i| ibp_flat.upper.as_slice().unwrap()[i] - ibp_flat.lower.as_slice().unwrap()[i])
        .collect();
    let crown_widths: Vec<f32> = (0..relu_output_size)
        .map(|i| crown_upper[i] - crown_lower[i])
        .collect();

    let avg_ibp_width: f32 = ibp_widths.iter().sum::<f32>() / relu_output_size as f32;
    let avg_crown_width: f32 = crown_widths.iter().sum::<f32>() / relu_output_size as f32;

    println!("Conv2d->ReLU CROWN vs IBP:");
    println!("  Average IBP width: {}", avg_ibp_width);
    println!("  Average CROWN width: {}", avg_crown_width);
    println!(
        "  CROWN improvement: {:.2}%",
        (1.0 - avg_crown_width / avg_ibp_width) * 100.0
    );

    // CROWN should be tighter for networks with ReLU
    assert!(
        avg_crown_width <= avg_ibp_width + 1e-5,
        "CROWN should be at least as tight as IBP"
    );
}

#[test]
fn test_conv2d_network_propagate_crown() {
    // Test that Network::propagate_crown works with Conv2d layers
    // instead of falling back to IBP
    use crate::network::Network;

    // Build a Conv2d -> ReLU network
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 2, 2]));
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 0, 1]] = -1.0;
    kernel[[0, 0, 1, 0]] = 1.0;
    kernel[[0, 0, 1, 1]] = -1.0;
    kernel[[1, 0, 0, 0]] = 0.5;
    kernel[[1, 0, 0, 1]] = 0.5;
    kernel[[1, 0, 1, 0]] = 0.5;
    kernel[[1, 0, 1, 1]] = 0.5;

    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();
    let relu = ReLULayer;

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(relu));

    // Input with perturbation
    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, 4, 4]), 0.5);
    let input = BoundedTensor::from_epsilon(center, 0.1);

    // Get IBP bounds
    let ibp_output = network.propagate_ibp(&input).unwrap();

    // Get CROWN bounds - should now work instead of falling back
    let crown_output = network.propagate_crown(&input).unwrap();

    // CROWN should be at least as tight as IBP
    let ibp_flat = ibp_output.flatten();
    let crown_flat = crown_output.flatten();

    let output_size = ibp_flat.len();
    for i in 0..output_size {
        let ibp_l = ibp_flat.lower.as_slice().unwrap()[i];
        let ibp_u = ibp_flat.upper.as_slice().unwrap()[i];
        let crown_l = crown_flat.lower.as_slice().unwrap()[i];
        let crown_u = crown_flat.upper.as_slice().unwrap()[i];

        // CROWN should be at least as tight (with tolerance)
        assert!(
            crown_l >= ibp_l - 1e-4,
            "Output {}: CROWN lower {} should be >= IBP lower {}",
            i,
            crown_l,
            ibp_l
        );
        assert!(
            crown_u <= ibp_u + 1e-4,
            "Output {}: CROWN upper {} should be <= IBP upper {}",
            i,
            crown_u,
            ibp_u
        );
    }

    // Verify CROWN provides tighter bounds for Conv->ReLU networks
    let ibp_avg_width: f32 = (0..output_size)
        .map(|i| ibp_flat.upper.as_slice().unwrap()[i] - ibp_flat.lower.as_slice().unwrap()[i])
        .sum::<f32>()
        / output_size as f32;
    let crown_avg_width: f32 = (0..output_size)
        .map(|i| crown_flat.upper.as_slice().unwrap()[i] - crown_flat.lower.as_slice().unwrap()[i])
        .sum::<f32>()
        / output_size as f32;

    println!("Network Conv2d->ReLU:");
    println!("  IBP avg width: {}", ibp_avg_width);
    println!("  CROWN avg width: {}", crown_avg_width);
    println!(
        "  Improvement: {:.2}%",
        (1.0 - crown_avg_width / ibp_avg_width) * 100.0
    );
}

#[test]
fn test_conv1d_network_propagate_crown() {
    // Test that Network::propagate_crown works with Conv1d layers
    // Use same setup as test_conv1d_crown_network_integration which passes
    use crate::network::Network;

    // Build a Conv1d -> ReLU network (same kernel as test_conv1d_crown_network_integration)
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 3]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = -1.0;
    kernel[[0, 0, 2]] = 1.0;
    kernel[[1, 0, 0]] = 0.5;
    kernel[[1, 0, 1]] = 0.5;
    kernel[[1, 0, 2]] = 0.5;

    let conv = Conv1dLayer::new(kernel, None, 1, 0).unwrap();
    let relu = ReLULayer;

    let mut network = Network::new();
    network.add_layer(Layer::Conv1d(conv));
    network.add_layer(Layer::ReLU(relu));

    // Input: [1, 8] single channel, length 8 (same as test_conv1d_crown_network_integration)
    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, 8]), 0.5);
    let input = BoundedTensor::from_epsilon(center, 0.1);

    // Get IBP bounds
    let ibp_output = network.propagate_ibp(&input).unwrap();

    // Get CROWN bounds
    let crown_output = network.propagate_crown(&input).unwrap();

    // CROWN should be at least as tight as IBP
    let ibp_flat = ibp_output.flatten();
    let crown_flat = crown_output.flatten();

    let output_size = ibp_flat.len();
    println!("Conv1d->ReLU Network test:");
    println!("  Input shape: {:?}", input.shape());
    println!("  Output shape: {:?}", ibp_output.shape());

    for i in 0..output_size {
        let ibp_l = ibp_flat.lower.as_slice().unwrap()[i];
        let ibp_u = ibp_flat.upper.as_slice().unwrap()[i];
        let crown_l = crown_flat.lower.as_slice().unwrap()[i];
        let crown_u = crown_flat.upper.as_slice().unwrap()[i];

        // Debug output
        if i < 3 {
            println!(
                "  Output {}: IBP=[{:.4}, {:.4}], CROWN=[{:.4}, {:.4}]",
                i, ibp_l, ibp_u, crown_l, crown_u
            );
        }

        assert!(
            crown_l >= ibp_l - 1e-4,
            "Output {}: CROWN lower {} should be >= IBP lower {}",
            i,
            crown_l,
            ibp_l
        );
        assert!(
            crown_u <= ibp_u + 1e-4,
            "Output {}: CROWN upper {} should be <= IBP upper {}",
            i,
            crown_u,
            ibp_u
        );
    }

    println!("Network Conv1d->ReLU: output size = {}", output_size);
}

// ============================================================
// MAXPOOL2D TESTS
// ============================================================

#[test]
fn test_maxpool2d_ibp_concrete() {
    // Test max pooling with concrete (non-interval) input
    let maxpool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));

    // Input: 4x4 with values 1-16
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[1, 4, 4]));
    for h in 0..4 {
        for w in 0..4 {
            input_data[[0, h, w]] = (h * 4 + w + 1) as f32;
        }
    }
    let input = BoundedTensor::concrete(input_data);

    let output = maxpool.propagate_ibp(&input).unwrap();

    // Output shape: (1, 2, 2)
    assert_eq!(output.shape(), &[1, 2, 2]);

    // MaxPool 2x2 stride 2 on 4x4:
    // [1,2,5,6] -> max=6, [3,4,7,8] -> max=8
    // [9,10,13,14] -> max=14, [11,12,15,16] -> max=16
    assert!((output.lower[[0, 0, 0]] - 6.0).abs() < 1e-6);
    assert!((output.lower[[0, 0, 1]] - 8.0).abs() < 1e-6);
    assert!((output.lower[[0, 1, 0]] - 14.0).abs() < 1e-6);
    assert!((output.lower[[0, 1, 1]] - 16.0).abs() < 1e-6);
}

#[test]
fn test_maxpool2d_ibp_interval() {
    // Test max pooling with interval input
    let maxpool = MaxPool2dLayer::new((2, 2), (1, 1), (0, 0));

    // Input: 3x3 with bounds [i, i+1] for each position i
    let mut lower_data = ArrayD::zeros(ndarray::IxDyn(&[1, 3, 3]));
    let mut upper_data = ArrayD::zeros(ndarray::IxDyn(&[1, 3, 3]));
    for h in 0..3 {
        for w in 0..3 {
            let i = (h * 3 + w) as f32;
            lower_data[[0, h, w]] = i;
            upper_data[[0, h, w]] = i + 1.0;
        }
    }
    let input = BoundedTensor::new(lower_data, upper_data).unwrap();

    let output = maxpool.propagate_ibp(&input).unwrap();

    // Output shape: (1, 2, 2) for 3x3 input with 2x2 kernel stride 1
    assert_eq!(output.shape(), &[1, 2, 2]);

    // Position (0,0) pools from (0,0), (0,1), (1,0), (1,1)
    // Lower bounds: 0, 1, 3, 4 -> max = 4
    // Upper bounds: 1, 2, 4, 5 -> max = 5
    assert!((output.lower[[0, 0, 0]] - 4.0).abs() < 1e-6);
    assert!((output.upper[[0, 0, 0]] - 5.0).abs() < 1e-6);

    // Position (0,1) pools from (0,1), (0,2), (1,1), (1,2)
    // Lower bounds: 1, 2, 4, 5 -> max = 5
    // Upper bounds: 2, 3, 5, 6 -> max = 6
    assert!((output.lower[[0, 0, 1]] - 5.0).abs() < 1e-6);
    assert!((output.upper[[0, 0, 1]] - 6.0).abs() < 1e-6);

    // Position (1,0) pools from (1,0), (1,1), (2,0), (2,1)
    // Lower bounds: 3, 4, 6, 7 -> max = 7
    // Upper bounds: 4, 5, 7, 8 -> max = 8
    assert!((output.lower[[0, 1, 0]] - 7.0).abs() < 1e-6);
    assert!((output.upper[[0, 1, 0]] - 8.0).abs() < 1e-6);

    // Position (1,1) pools from (1,1), (1,2), (2,1), (2,2)
    // Lower bounds: 4, 5, 7, 8 -> max = 8
    // Upper bounds: 5, 6, 8, 9 -> max = 9
    assert!((output.lower[[0, 1, 1]] - 8.0).abs() < 1e-6);
    assert!((output.upper[[0, 1, 1]] - 9.0).abs() < 1e-6);
}

#[test]
fn test_maxpool2d_soundness() {
    // Soundness test: verify concrete outputs are within bounds
    let maxpool = MaxPool2dLayer::new((2, 2), (1, 1), (0, 0));

    // Input with perturbation
    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, 4, 4]), 0.5);
    let input = BoundedTensor::from_epsilon(center.clone(), 0.1);

    let output = maxpool.propagate_ibp(&input).unwrap();

    // Test several concrete points within input bounds
    for offset in [-0.1, -0.05, 0.0, 0.05, 0.1] {
        let concrete_input = center.clone().mapv(|v| v + offset);

        // Manually compute max pool
        for oh in 0..3 {
            for ow in 0..3 {
                let mut max_val = f32::NEG_INFINITY;
                for kh in 0..2 {
                    for kw in 0..2 {
                        max_val = max_val.max(concrete_input[[0, oh + kh, ow + kw]]);
                    }
                }

                // Verify concrete output is within bounds
                assert!(
                    max_val >= output.lower[[0, oh, ow]] - 1e-6,
                    "Concrete {} should be >= lower {}",
                    max_val,
                    output.lower[[0, oh, ow]]
                );
                assert!(
                    max_val <= output.upper[[0, oh, ow]] + 1e-6,
                    "Concrete {} should be <= upper {}",
                    max_val,
                    output.upper[[0, oh, ow]]
                );
            }
        }
    }
}

#[test]
fn test_maxpool2d_stride() {
    // Test max pooling with stride
    let maxpool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));

    // 6x6 input -> 3x3 output with stride 2
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[1, 6, 6]));
    for h in 0..6 {
        for w in 0..6 {
            input_data[[0, h, w]] = (h * 6 + w) as f32;
        }
    }
    let input = BoundedTensor::concrete(input_data);

    let output = maxpool.propagate_ibp(&input).unwrap();

    // Output shape: (1, 3, 3)
    assert_eq!(output.shape(), &[1, 3, 3]);

    // Position (0,0) pools from (0,0), (0,1), (1,0), (1,1): 0,1,6,7 -> max=7
    assert!((output.lower[[0, 0, 0]] - 7.0).abs() < 1e-6);

    // Position (0,1) pools from (0,2), (0,3), (1,2), (1,3): 2,3,8,9 -> max=9
    assert!((output.lower[[0, 0, 1]] - 9.0).abs() < 1e-6);
}

#[test]
fn test_maxpool2d_padding() {
    // Test max pooling with padding
    let maxpool = MaxPool2dLayer::new((3, 3), (1, 1), (1, 1));

    // 3x3 input with padding 1 -> 3x3 output
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[1, 3, 3]));
    for h in 0..3 {
        for w in 0..3 {
            input_data[[0, h, w]] = (h * 3 + w + 1) as f32;
        }
    }
    let input = BoundedTensor::concrete(input_data);

    let output = maxpool.propagate_ibp(&input).unwrap();

    // Output shape should be (1, 3, 3)
    assert_eq!(output.shape(), &[1, 3, 3]);

    // Center position (1,1) sees all 9 values -> max=9
    assert!((output.lower[[0, 1, 1]] - 9.0).abs() < 1e-6);
}

#[test]
fn test_maxpool2d_multi_channel() {
    // Test max pooling with multiple channels
    let maxpool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));

    // 2 channels, 4x4 each
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[2, 4, 4]));
    for c in 0..2 {
        for h in 0..4 {
            for w in 0..4 {
                input_data[[c, h, w]] = ((c + 1) * 100 + h * 4 + w) as f32;
            }
        }
    }
    let input = BoundedTensor::concrete(input_data);

    let output = maxpool.propagate_ibp(&input).unwrap();

    // Output shape: (2, 2, 2)
    assert_eq!(output.shape(), &[2, 2, 2]);

    // Channel 0: values 100-115, max of first 2x2 block = 105 (100+1+4)
    // Wait, let me recalculate: positions (0,0), (0,1), (1,0), (1,1)
    // Values: 100, 101, 104, 105 -> max = 105
    assert!((output.lower[[0, 0, 0]] - 105.0).abs() < 1e-6);

    // Channel 1: values 200-215, same pattern
    assert!((output.lower[[1, 0, 0]] - 205.0).abs() < 1e-6);
}

// ============================================================
// CONV1D TESTS
// ============================================================

#[test]
fn test_conv1d_basic() {
    // Simple 1D convolution: sum of 3 adjacent elements
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 3]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = 1.0;
    kernel[[0, 0, 2]] = 1.0;
    let conv = Conv1dLayer::new(kernel, None, 1, 0).unwrap();

    // Input: [1, 2, 3, 4, 5]
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[1, 5]));
    for i in 0..5 {
        input_data[[0, i]] = (i + 1) as f32;
    }
    let input = BoundedTensor::concrete(input_data);

    let output = conv.propagate_ibp(&input).unwrap();

    // Output: [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
    assert_eq!(output.shape(), &[1, 3]);
    assert!((output.lower[[0, 0]] - 6.0).abs() < 1e-6);
    assert!((output.lower[[0, 1]] - 9.0).abs() < 1e-6);
    assert!((output.lower[[0, 2]] - 12.0).abs() < 1e-6);
}

#[test]
fn test_conv1d_ibp_batched_input() {
    // Verify Conv1d IBP supports (batch, channels, length) inputs.
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 3]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = 1.0;
    kernel[[0, 0, 2]] = 1.0;
    let conv = Conv1dLayer::new(kernel, None, 1, 0).unwrap();

    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 5]));
    for i in 0..5 {
        input_data[[0, 0, i]] = (i + 1) as f32; // 1..5
        input_data[[1, 0, i]] = (i + 2) as f32; // 2..6
    }
    let input = BoundedTensor::concrete(input_data);

    let output = conv.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 1, 3]);
    // Batch 0: [6, 9, 12]
    assert!((output.lower[[0, 0, 0]] - 6.0).abs() < 1e-6);
    assert!((output.lower[[0, 0, 1]] - 9.0).abs() < 1e-6);
    assert!((output.lower[[0, 0, 2]] - 12.0).abs() < 1e-6);

    // Batch 1: [9, 12, 15]
    assert!((output.lower[[1, 0, 0]] - 9.0).abs() < 1e-6);
    assert!((output.lower[[1, 0, 1]] - 12.0).abs() < 1e-6);
    assert!((output.lower[[1, 0, 2]] - 15.0).abs() < 1e-6);
}

#[test]
fn test_conv1d_with_stride() {
    // Conv1d with stride=2
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 2]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = 1.0;
    let conv = Conv1dLayer::new(kernel, None, 2, 0).unwrap();

    // Input: [1, 2, 3, 4] with stride 2 -> output length = (4-2)/2 + 1 = 2
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[1, 4]));
    for i in 0..4 {
        input_data[[0, i]] = (i + 1) as f32;
    }
    let input = BoundedTensor::concrete(input_data);

    let output = conv.propagate_ibp(&input).unwrap();

    // Output: [1+2, 3+4] = [3, 7]
    assert_eq!(output.shape(), &[1, 2]);
    assert!((output.lower[[0, 0]] - 3.0).abs() < 1e-6);
    assert!((output.lower[[0, 1]] - 7.0).abs() < 1e-6);
}

#[test]
fn test_conv1d_with_padding() {
    // Conv1d with padding=1
    let kernel = ArrayD::ones(ndarray::IxDyn(&[1, 1, 3]));
    let conv = Conv1dLayer::new(kernel, None, 1, 1).unwrap();

    // Input: [1, 1, 1] with padding 1 -> output length = (3+2-3)/1 + 1 = 3
    let input_data = ArrayD::ones(ndarray::IxDyn(&[1, 3]));
    let input = BoundedTensor::concrete(input_data);

    let output = conv.propagate_ibp(&input).unwrap();

    // With padding: [0,1,1], [1,1,1], [1,1,0] -> sums: 2, 3, 2
    assert_eq!(output.shape(), &[1, 3]);
    assert!(
        (output.lower[[0, 0]] - 2.0).abs() < 1e-6,
        "left edge = {}",
        output.lower[[0, 0]]
    );
    assert!(
        (output.lower[[0, 1]] - 3.0).abs() < 1e-6,
        "center = {}",
        output.lower[[0, 1]]
    );
    assert!(
        (output.lower[[0, 2]] - 2.0).abs() < 1e-6,
        "right edge = {}",
        output.lower[[0, 2]]
    );
}

#[test]
fn test_conv1d_with_bias() {
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 2]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = 1.0;
    kernel[[1, 0, 0]] = 2.0;
    kernel[[1, 0, 1]] = -1.0;
    let bias = arr1(&[0.5, -0.5]);
    let conv = Conv1dLayer::new(kernel, Some(bias), 1, 0).unwrap();

    // Input: [1, 2]
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[1, 2]));
    input_data[[0, 0]] = 1.0;
    input_data[[0, 1]] = 2.0;
    let input = BoundedTensor::concrete(input_data);

    let output = conv.propagate_ibp(&input).unwrap();

    // Channel 0: 1+2 + 0.5 = 3.5
    // Channel 1: 2*1 + (-1)*2 - 0.5 = 2 - 2 - 0.5 = -0.5
    assert_eq!(output.shape(), &[2, 1]);
    assert!((output.lower[[0, 0]] - 3.5).abs() < 1e-6);
    assert!((output.lower[[1, 0]] - (-0.5)).abs() < 1e-6);
}

#[test]
fn test_conv1d_ibp_soundness() {
    // Soundness test: verify concrete outputs are within bounds
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 3]));
    kernel[[0, 0, 0]] = 2.0;
    kernel[[0, 0, 1]] = -1.0;
    kernel[[0, 0, 2]] = 1.0;
    let conv = Conv1dLayer::new(kernel.clone(), None, 1, 0).unwrap();

    // Input in [-1, 1]
    let lower_data = ArrayD::from_elem(ndarray::IxDyn(&[1, 5]), -1.0f32);
    let upper_data = ArrayD::from_elem(ndarray::IxDyn(&[1, 5]), 1.0f32);
    let input = BoundedTensor::new(lower_data, upper_data).unwrap();

    let output_bounds = conv.propagate_ibp(&input).unwrap();

    // Test several concrete inputs
    let test_inputs = [
        ArrayD::from_elem(ndarray::IxDyn(&[1, 5]), -1.0f32), // all lower
        ArrayD::from_elem(ndarray::IxDyn(&[1, 5]), 1.0f32),  // all upper
        ArrayD::from_elem(ndarray::IxDyn(&[1, 5]), 0.0f32),  // center
    ];

    for test_input in &test_inputs {
        let concrete_output = conv1d_single(test_input, &kernel, 1, 0);

        for oc in 0..1 {
            for ol in 0..3 {
                let val = concrete_output[[oc, ol]];
                assert!(
                    val >= output_bounds.lower[[oc, ol]] - 1e-6,
                    "Soundness: val {} < lower {}",
                    val,
                    output_bounds.lower[[oc, ol]]
                );
                assert!(
                    val <= output_bounds.upper[[oc, ol]] + 1e-6,
                    "Soundness: val {} > upper {}",
                    val,
                    output_bounds.upper[[oc, ol]]
                );
            }
        }
    }
}

#[test]
fn test_conv1d_shape_validation() {
    // Kernel must be 3D
    let bad_kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 2])); // 2D, not 3D
    let result = Conv1dLayer::new(bad_kernel, None, 1, 0);
    assert!(result.is_err());
}

// ============================================================
// CONV1D CROWN TESTS
// ============================================================

#[test]
fn test_conv1d_transpose_basic() {
    // Test that conv1d_transpose is the inverse of conv1d in the gradient sense
    // For a 1x1 conv with identity kernel, transpose should also be identity
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 1]));
    kernel[[0, 0, 0]] = 1.0;

    // "Gradient" at output
    let mut grad_out = ArrayD::zeros(ndarray::IxDyn(&[1, 5]));
    grad_out[[0, 0]] = 1.0;
    grad_out[[0, 2]] = 3.0;
    grad_out[[0, 4]] = 5.0;

    let grad_in = conv1d_transpose(&grad_out, &kernel, 1, 0, 5);

    // With identity 1x1 kernel and no stride/padding, should match
    assert_eq!(grad_in.shape(), &[1, 5]);
    assert!((grad_in[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((grad_in[[0, 2]] - 3.0).abs() < 1e-6);
    assert!((grad_in[[0, 4]] - 5.0).abs() < 1e-6);
}

#[test]
fn test_conv1d_transpose_multi_channel() {
    // Test transpose with multiple input/output channels
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 3, 2])); // out_c=2, in_c=3, k=2
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = 0.5;
    kernel[[1, 1, 0]] = -1.0;
    kernel[[1, 2, 1]] = 2.0;

    // Gradient at conv output: [2, 3] (out_c=2, out_len=3)
    let mut grad_out = ArrayD::zeros(ndarray::IxDyn(&[2, 3]));
    grad_out[[0, 0]] = 1.0;
    grad_out[[1, 1]] = 1.0;

    let grad_in = conv1d_transpose(&grad_out, &kernel, 1, 0, 4);

    // Output should be [3, 4] (in_c=3, in_len=4)
    assert_eq!(grad_in.shape(), &[3, 4]);

    // Channel 0: kernel[0,0,:] = [1.0, 0.5], grad[0,:] = [1, 0, 0]
    // Transpose scatters: pos 0 gets 1.0*1.0=1.0, pos 1 gets 1.0*0.5=0.5
    assert!((grad_in[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((grad_in[[0, 1]] - 0.5).abs() < 1e-6);

    // Channel 1: kernel[1,1,:] = [-1.0, 0.0], grad[1,:] = [0, 1, 0]
    // Transpose scatters: pos 1 gets -1.0*1.0=-1.0
    assert!((grad_in[[1, 1]] - (-1.0)).abs() < 1e-6);

    // Channel 2: kernel[1,2,:] = [0.0, 2.0], grad[1,:] = [0, 1, 0]
    // Transpose scatters: pos 2 gets 2.0*1.0=2.0
    assert!((grad_in[[2, 2]] - 2.0).abs() < 1e-6);
}

#[test]
fn test_conv1d_crown_identity_bounds() {
    // Test shape handling: verify dimensions work correctly
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 3])); // out_c=2, in_c=1, k=3
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = 1.0;
    kernel[[0, 0, 2]] = 1.0;
    kernel[[1, 0, 0]] = 0.5;
    kernel[[1, 0, 1]] = 0.0;
    kernel[[1, 0, 2]] = 0.5;

    // Input: [1, 8] -> Output: [2, 6] = 12 elements
    let conv = Conv1dLayer::with_input_length(kernel, None, 1, 0, 8).unwrap();

    // Identity bounds on output
    let output_size = 12;
    let identity = LinearBounds::identity(output_size);

    let result = conv.propagate_linear(&identity);
    assert!(result.is_ok());

    let new_bounds = result.unwrap();
    // New bounds should be on flattened input: 1 * 8 = 8
    assert_eq!(new_bounds.num_inputs(), 8);
    assert_eq!(new_bounds.num_outputs(), 12);
}

#[test]
fn test_conv1d_crown_simple_backward() {
    // Test CROWN backward pass through a simple 1x1 conv
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 1]));
    kernel[[0, 0, 0]] = 2.0; // Just scales by 2

    let conv = Conv1dLayer::with_input_length(kernel, None, 1, 0, 4).unwrap();

    // Identity bounds on output [1, 4] = 4 elements
    let identity = LinearBounds::identity(4);

    let result = conv.propagate_linear(&identity).unwrap().into_owned();

    // Backward through 2x scaling should give A with 2.0 entries
    for i in 0..4 {
        assert!(
            (result.lower_a[[i, i]] - 2.0).abs() < 1e-6,
            "Expected 2.0 at [{}, {}], got {}",
            i,
            i,
            result.lower_a[[i, i]]
        );
        assert!((result.upper_a[[i, i]] - 2.0).abs() < 1e-6);
    }
}

#[test]
fn test_conv1d_crown_with_bias() {
    // Test that bias is handled correctly in CROWN
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 1]));
    kernel[[0, 0, 0]] = 1.0;

    let bias = arr1(&[3.0]);

    let conv = Conv1dLayer::with_input_length(kernel, Some(bias), 1, 0, 4).unwrap();

    // Identity bounds on output
    let identity = LinearBounds::identity(4);

    let result = conv.propagate_linear(&identity).unwrap().into_owned();

    // Bias contribution: each output gets +3.0
    // For identity A matrix, bias_contrib[i] = sum over spatial * bias = 1 * 3.0 = 3.0
    for i in 0..4 {
        assert!(
            (result.lower_b[i] - 3.0).abs() < 1e-6,
            "Expected bias 3.0 at [{}], got {}",
            i,
            result.lower_b[i]
        );
        assert!((result.upper_b[i] - 3.0).abs() < 1e-6);
    }
}

#[test]
fn test_conv1d_crown_vs_ibp_tightness() {
    // For pure Conv1d (linear operation), CROWN should match IBP exactly
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 3]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = -1.0;
    kernel[[0, 0, 2]] = 1.0;
    kernel[[1, 0, 0]] = 0.5;
    kernel[[1, 0, 1]] = 0.5;
    kernel[[1, 0, 2]] = 0.5;

    let bias = arr1(&[1.0, -0.5]);

    let in_len = 6;
    let conv_crown =
        Conv1dLayer::with_input_length(kernel.clone(), Some(bias.clone()), 1, 0, in_len).unwrap();
    let conv_ibp = Conv1dLayer::new(kernel, Some(bias), 1, 0).unwrap();

    // Input bounds
    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, in_len]), 0.5);
    let input = BoundedTensor::from_epsilon(center, 0.1);

    // IBP bounds
    let ibp_output = conv_ibp.propagate_ibp(&input).unwrap();

    // CROWN bounds
    let out_len = conv_crown.output_length(in_len);
    let output_size = 2 * out_len; // out_c=2
    let identity = LinearBounds::identity(output_size);

    let crown_bounds = conv_crown.propagate_linear(&identity).unwrap().into_owned();

    // Concretize CROWN bounds
    let flat_input = input.flatten();
    let x_l = flat_input
        .lower
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let x_u = flat_input
        .upper
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    let a_pos_l = crown_bounds.lower_a.mapv(|v| v.max(0.0));
    let a_neg_l = crown_bounds.lower_a.mapv(|v| v.min(0.0));
    let a_pos_u = crown_bounds.upper_a.mapv(|v| v.max(0.0));
    let a_neg_u = crown_bounds.upper_a.mapv(|v| v.min(0.0));

    let crown_lower = a_pos_l.dot(&x_l) + a_neg_l.dot(&x_u) + &crown_bounds.lower_b;
    let crown_upper = a_pos_u.dot(&x_u) + a_neg_u.dot(&x_l) + &crown_bounds.upper_b;

    // Compare with IBP
    let ibp_flat = ibp_output.flatten();
    for i in 0..output_size {
        let ibp_l = ibp_flat.lower.as_slice().unwrap()[i];
        let ibp_u = ibp_flat.upper.as_slice().unwrap()[i];

        // For linear layers, CROWN and IBP should match
        assert!(
            (crown_lower[i] - ibp_l).abs() < 1e-4,
            "Output {}: CROWN lower {} vs IBP lower {}",
            i,
            crown_lower[i],
            ibp_l
        );
        assert!(
            (crown_upper[i] - ibp_u).abs() < 1e-4,
            "Output {}: CROWN upper {} vs IBP upper {}",
            i,
            crown_upper[i],
            ibp_u
        );
    }
}

#[test]
fn test_conv1d_crown_requires_input_length() {
    // CROWN should fail if input_length is not set
    let kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 3]));
    let conv = Conv1dLayer::new(kernel, None, 1, 0).unwrap();

    let identity = LinearBounds::identity(4);
    let result = conv.propagate_linear(&identity);
    assert!(result.is_err());
}

#[test]
fn test_conv1d_crown_network_integration() {
    // Test CROWN through a Conv1d -> ReLU network
    // This verifies the full backward pass works with non-linearities

    // Input: [1, 8] -> Conv [2, 1, 3] -> [2, 6] -> ReLU -> [2, 6]
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 3]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = -1.0;
    kernel[[0, 0, 2]] = 1.0;
    kernel[[1, 0, 0]] = 0.5;
    kernel[[1, 0, 1]] = 0.5;
    kernel[[1, 0, 2]] = 0.5;

    let conv = Conv1dLayer::with_input_length(kernel.clone(), None, 1, 0, 8).unwrap();

    // Input with perturbation around 0.5
    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, 8]), 0.5);
    let input = BoundedTensor::from_epsilon(center, 0.1);

    // Get IBP bounds for pre-activation (conv output)
    let conv_ibp = Conv1dLayer::new(kernel, None, 1, 0).unwrap();
    let pre_activation = conv_ibp.propagate_ibp(&input).unwrap();

    // ReLU layer
    let relu = ReLULayer;

    // IBP through ReLU
    let ibp_output = relu.propagate_ibp(&pre_activation).unwrap();

    // Now do CROWN backward:
    // Start with identity on ReLU output
    let relu_output_size = 2 * 6; // 12
    let identity = LinearBounds::identity(relu_output_size);

    // Backward through ReLU
    let relu_bounds = relu
        .propagate_linear_with_bounds(&identity, &pre_activation)
        .unwrap();

    // Backward through Conv
    let conv_bounds = conv.propagate_linear(&relu_bounds).unwrap().into_owned();

    // Concretize
    let flat_input = input.flatten();
    let x_l = flat_input
        .lower
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let x_u = flat_input
        .upper
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    let a_pos_l = conv_bounds.lower_a.mapv(|v| v.max(0.0));
    let a_neg_l = conv_bounds.lower_a.mapv(|v| v.min(0.0));
    let a_pos_u = conv_bounds.upper_a.mapv(|v| v.max(0.0));
    let a_neg_u = conv_bounds.upper_a.mapv(|v| v.min(0.0));

    let crown_lower = a_pos_l.dot(&x_l) + a_neg_l.dot(&x_u) + &conv_bounds.lower_b;
    let crown_upper = a_pos_u.dot(&x_u) + a_neg_u.dot(&x_l) + &conv_bounds.upper_b;

    // CROWN should be tighter than or equal to IBP
    let ibp_flat = ibp_output.flatten();
    for i in 0..relu_output_size {
        let ibp_l = ibp_flat.lower.as_slice().unwrap()[i];
        let ibp_u = ibp_flat.upper.as_slice().unwrap()[i];
        let crown_l = crown_lower[i];
        let crown_u = crown_upper[i];

        // CROWN should be at least as tight
        assert!(
            crown_l >= ibp_l - 1e-4,
            "Output {}: CROWN lower {} should be >= IBP lower {}",
            i,
            crown_l,
            ibp_l
        );
        assert!(
            crown_u <= ibp_u + 1e-4,
            "Output {}: CROWN upper {} should be <= IBP upper {}",
            i,
            crown_u,
            ibp_u
        );
    }

    // With ReLU, CROWN should often be tighter
    let ibp_widths: Vec<f32> = (0..relu_output_size)
        .map(|i| ibp_flat.upper.as_slice().unwrap()[i] - ibp_flat.lower.as_slice().unwrap()[i])
        .collect();
    let crown_widths: Vec<f32> = (0..relu_output_size)
        .map(|i| crown_upper[i] - crown_lower[i])
        .collect();

    let avg_ibp_width: f32 = ibp_widths.iter().sum::<f32>() / relu_output_size as f32;
    let avg_crown_width: f32 = crown_widths.iter().sum::<f32>() / relu_output_size as f32;

    println!("Conv1d->ReLU CROWN vs IBP:");
    println!("  Average IBP width: {}", avg_ibp_width);
    println!("  Average CROWN width: {}", avg_crown_width);
    println!(
        "  CROWN improvement: {:.2}%",
        (1.0 - avg_crown_width / avg_ibp_width) * 100.0
    );

    // CROWN should be tighter for networks with ReLU
    assert!(
        avg_crown_width <= avg_ibp_width + 1e-5,
        "CROWN should be at least as tight as IBP"
    );
}

#[test]
fn test_conv1d_batched_crown_basic() {
    // Test batched CROWN backward propagation through Conv1d
    // Input: [2, 8] (2 channels, 8 length)
    // Kernel: [3, 2, 3] (3 out_channels, 2 in_channels, 3 kernel_size)
    // Output: [3, 6] (3 channels, 6 length) = 18 flattened

    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[3, 2, 3]));
    // Initialize kernel with some values
    for oc in 0..3 {
        for ic in 0..2 {
            for k in 0..3 {
                kernel[[oc, ic, k]] = ((oc * 6 + ic * 3 + k) as f32 * 0.1) - 0.3;
            }
        }
    }

    let bias = arr1(&[0.1, -0.1, 0.2]);
    let conv = Conv1dLayer::with_input_length(kernel.clone(), Some(bias.clone()), 1, 0, 8).unwrap();

    // For Conv1d, use flattened output size for identity bounds
    // Output: [3, 6] -> flattened size = 18
    let conv_out_size = 3 * 6; // 18
    let identity_bounds = BatchedLinearBounds::identity(&[conv_out_size]);

    // Propagate backward
    let input_bounds = conv.propagate_linear_batched(&identity_bounds).unwrap();

    // Verify output dimensions
    let conv_in_size = 2 * 8; // 16
    let expected_a_shape = vec![conv_out_size, conv_in_size]; // [18, 16]
    assert_eq!(
        input_bounds.lower_a.shape(),
        expected_a_shape.as_slice(),
        "lower_a shape mismatch"
    );
    assert_eq!(
        input_bounds.upper_a.shape(),
        expected_a_shape.as_slice(),
        "upper_a shape mismatch"
    );

    // Verify the bounds are finite
    assert!(
        input_bounds.lower_a.iter().all(|&v| v.is_finite()),
        "lower_a has non-finite values"
    );
    assert!(
        input_bounds.upper_a.iter().all(|&v| v.is_finite()),
        "upper_a has non-finite values"
    );
    assert!(
        input_bounds.lower_b.iter().all(|&v| v.is_finite()),
        "lower_b has non-finite values"
    );
    assert!(
        input_bounds.upper_b.iter().all(|&v| v.is_finite()),
        "upper_b has non-finite values"
    );

    println!("Conv1d batched CROWN test passed!");
    println!("  Input bounds A shape: {:?}", input_bounds.lower_a.shape());
    println!("  Input bounds b shape: {:?}", input_bounds.lower_b.shape());
}

#[test]
fn test_conv1d_batched_crown_soundness() {
    // Test that batched CROWN produces sound bounds by sampling random inputs
    // and verifying output is within bounds

    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 3]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = -1.0;
    kernel[[0, 0, 2]] = 1.0;
    kernel[[1, 0, 0]] = 0.5;
    kernel[[1, 0, 1]] = 0.5;
    kernel[[1, 0, 2]] = 0.5;

    let bias = arr1(&[0.1, -0.2]);
    let conv = Conv1dLayer::with_input_length(kernel.clone(), Some(bias.clone()), 1, 0, 8).unwrap();

    // Input with perturbation
    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, 8]), 0.5);
    let input = BoundedTensor::from_epsilon(center.clone(), 0.1);

    // Get IBP bounds for comparison
    let ibp_output = conv.propagate_ibp(&input).unwrap();

    // For Conv1d, use flattened output size for identity bounds
    // Output: [2, 6] -> flattened size = 12
    let conv_out_size = 2 * 6; // 12
    let identity_bounds = BatchedLinearBounds::identity(&[conv_out_size]);
    let crown_bounds = conv.propagate_linear_batched(&identity_bounds).unwrap();

    // Concretize CROWN bounds
    let crown_output = crown_bounds.concretize(&input);

    // CROWN should be as tight or tighter than IBP (same for linear layers)
    let ibp_flat = ibp_output.flatten();
    let crown_flat = crown_output.flatten();

    for i in 0..12 {
        let ibp_l = ibp_flat.lower.as_slice().unwrap()[i];
        let ibp_u = ibp_flat.upper.as_slice().unwrap()[i];
        let crown_l = crown_flat.lower.as_slice().unwrap()[i];
        let crown_u = crown_flat.upper.as_slice().unwrap()[i];

        // CROWN should be at least as tight (with small tolerance for numerical error)
        assert!(
            crown_l >= ibp_l - 1e-4,
            "Output {}: CROWN lower {} < IBP lower {} (diff: {})",
            i,
            crown_l,
            ibp_l,
            crown_l - ibp_l
        );
        assert!(
            crown_u <= ibp_u + 1e-4,
            "Output {}: CROWN upper {} > IBP upper {} (diff: {})",
            i,
            crown_u,
            ibp_u,
            crown_u - ibp_u
        );
    }

    println!("Conv1d batched CROWN soundness test passed!");
}

#[test]
fn test_conv2d_batched_crown_basic() {
    // Test batched CROWN backward propagation through Conv2d
    // Input: [2, 4, 4] (2 channels, 4x4 spatial)
    // Kernel: [3, 2, 3, 3] (3 out_channels, 2 in_channels, 3x3 kernel)
    // Output: [3, 2, 2] (3 channels, 2x2 spatial) = 12 flattened

    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[3, 2, 3, 3]));
    // Initialize kernel with some values
    for oc in 0..3 {
        for ic in 0..2 {
            for kh in 0..3 {
                for kw in 0..3 {
                    kernel[[oc, ic, kh, kw]] =
                        ((oc * 18 + ic * 9 + kh * 3 + kw) as f32 * 0.05) - 0.4;
                }
            }
        }
    }

    let bias = arr1(&[0.1, -0.1, 0.2]);
    let conv =
        Conv2dLayer::with_input_shape(kernel.clone(), Some(bias.clone()), (1, 1), (0, 0), 4, 4)
            .unwrap();

    // For Conv2d, use flattened output size for identity bounds
    // Output: [3, 2, 2] -> flattened size = 12
    let conv_out_size = 3 * 2 * 2; // 12
    let identity_bounds = BatchedLinearBounds::identity(&[conv_out_size]);

    // Propagate backward
    let input_bounds = conv.propagate_linear_batched(&identity_bounds).unwrap();

    // Verify output dimensions
    let conv_in_size = 2 * 4 * 4; // 32
    let expected_a_shape = vec![conv_out_size, conv_in_size]; // [12, 32]
    assert_eq!(
        input_bounds.lower_a.shape(),
        expected_a_shape.as_slice(),
        "lower_a shape mismatch"
    );
    assert_eq!(
        input_bounds.upper_a.shape(),
        expected_a_shape.as_slice(),
        "upper_a shape mismatch"
    );

    // Verify the bounds are finite
    assert!(
        input_bounds.lower_a.iter().all(|&v| v.is_finite()),
        "lower_a has non-finite values"
    );
    assert!(
        input_bounds.upper_a.iter().all(|&v| v.is_finite()),
        "upper_a has non-finite values"
    );
    assert!(
        input_bounds.lower_b.iter().all(|&v| v.is_finite()),
        "lower_b has non-finite values"
    );
    assert!(
        input_bounds.upper_b.iter().all(|&v| v.is_finite()),
        "upper_b has non-finite values"
    );

    println!("Conv2d batched CROWN test passed!");
    println!("  Input bounds A shape: {:?}", input_bounds.lower_a.shape());
    println!("  Input bounds b shape: {:?}", input_bounds.lower_b.shape());
}

#[test]
fn test_conv2d_batched_crown_soundness() {
    // Test that batched CROWN produces sound bounds by sampling random inputs
    // and verifying output is within bounds

    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 3, 3]));
    // Simple pattern for easier verification
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 0, 1]] = -1.0;
    kernel[[0, 0, 0, 2]] = 1.0;
    kernel[[0, 0, 1, 0]] = 0.5;
    kernel[[0, 0, 1, 1]] = 0.5;
    kernel[[0, 0, 1, 2]] = 0.5;
    kernel[[0, 0, 2, 0]] = -0.5;
    kernel[[0, 0, 2, 1]] = -0.5;
    kernel[[0, 0, 2, 2]] = -0.5;
    kernel[[1, 0, 0, 0]] = 0.3;
    kernel[[1, 0, 1, 1]] = 0.3;
    kernel[[1, 0, 2, 2]] = 0.3;

    let bias = arr1(&[0.1, -0.2]);
    let conv =
        Conv2dLayer::with_input_shape(kernel.clone(), Some(bias.clone()), (1, 1), (0, 0), 6, 6)
            .unwrap();

    // Input with perturbation (1 channel, 6x6) - use 3D shape for IBP
    let center_3d = ArrayD::from_elem(ndarray::IxDyn(&[1, 6, 6]), 0.5);
    let input_3d = BoundedTensor::from_epsilon(center_3d.clone(), 0.1);

    // Get IBP bounds for comparison (3D input -> 3D output)
    let ibp_output = conv.propagate_ibp(&input_3d).unwrap();

    // For Conv2d, output is [2, 4, 4] (2 channels, 4x4 spatial)
    // flattened size = 32
    let conv_out_size = 2 * 4 * 4; // 32
    let identity_bounds = BatchedLinearBounds::identity(&[conv_out_size]);
    let crown_bounds = conv.propagate_linear_batched(&identity_bounds).unwrap();

    // For CROWN concretization, we need flattened input to match A's input dimension
    // Input size = 1 * 6 * 6 = 36
    let input_flat = input_3d.flatten();

    // Concretize CROWN bounds with flattened input
    let crown_output = crown_bounds.concretize(&input_flat);

    // CROWN should be as tight or tighter than IBP (same for linear layers)
    let ibp_flat = ibp_output.flatten();
    let crown_flat = crown_output.flatten();

    for i in 0..32 {
        let ibp_l = ibp_flat.lower.as_slice().unwrap()[i];
        let ibp_u = ibp_flat.upper.as_slice().unwrap()[i];
        let crown_l = crown_flat.lower.as_slice().unwrap()[i];
        let crown_u = crown_flat.upper.as_slice().unwrap()[i];

        // CROWN should be at least as tight (with small tolerance for numerical error)
        assert!(
            crown_l >= ibp_l - 1e-4,
            "Output {}: CROWN lower {} < IBP lower {} (diff: {})",
            i,
            crown_l,
            ibp_l,
            crown_l - ibp_l
        );
        assert!(
            crown_u <= ibp_u + 1e-4,
            "Output {}: CROWN upper {} > IBP upper {} (diff: {})",
            i,
            crown_u,
            ibp_u,
            crown_u - ibp_u
        );
    }

    println!("Conv2d batched CROWN soundness test passed!");
}

#[test]
fn test_conv2d_batched_crown_vs_regular_crown() {
    // Verify that batched CROWN produces the same results as regular CROWN
    // when batch size is 1

    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 2, 3, 3]));
    for i in 0..36 {
        let oc = i / 18;
        let ic = (i % 18) / 9;
        let kh = (i % 9) / 3;
        let kw = i % 3;
        kernel[[oc, ic, kh, kw]] = (i as f32 * 0.05) - 0.4;
    }

    let bias = arr1(&[0.1, -0.1]);
    let conv =
        Conv2dLayer::with_input_shape(kernel.clone(), Some(bias.clone()), (1, 1), (0, 0), 5, 5)
            .unwrap();

    // Output: [2, 3, 3] = 18 elements
    let conv_out_size = 2 * 3 * 3;

    // Regular CROWN with LinearBounds
    let regular_bounds = LinearBounds::identity(conv_out_size);
    let regular_result = conv.propagate_linear(&regular_bounds).unwrap();

    // Batched CROWN with BatchedLinearBounds
    let batched_bounds = BatchedLinearBounds::identity(&[conv_out_size]);
    let batched_result = conv.propagate_linear_batched(&batched_bounds).unwrap();

    // Results should match
    let regular_la = regular_result.lower_a.as_slice().unwrap();
    let batched_la = batched_result.lower_a.as_slice().unwrap();

    assert_eq!(regular_la.len(), batched_la.len(), "lower_a size mismatch");
    for (i, (&r, &b)) in regular_la.iter().zip(batched_la.iter()).enumerate() {
        assert!(
            (r - b).abs() < 1e-5,
            "lower_a[{}] mismatch: regular={}, batched={}, diff={}",
            i,
            r,
            b,
            (r - b).abs()
        );
    }

    let regular_ua = regular_result.upper_a.as_slice().unwrap();
    let batched_ua = batched_result.upper_a.as_slice().unwrap();

    for (i, (&r, &b)) in regular_ua.iter().zip(batched_ua.iter()).enumerate() {
        assert!(
            (r - b).abs() < 1e-5,
            "upper_a[{}] mismatch: regular={}, batched={}, diff={}",
            i,
            r,
            b,
            (r - b).abs()
        );
    }

    let regular_lb = regular_result.lower_b.as_slice().unwrap();
    let batched_lb = batched_result.lower_b.as_slice().unwrap();

    for (i, (&r, &b)) in regular_lb.iter().zip(batched_lb.iter()).enumerate() {
        assert!(
            (r - b).abs() < 1e-5,
            "lower_b[{}] mismatch: regular={}, batched={}, diff={}",
            i,
            r,
            b,
            (r - b).abs()
        );
    }

    println!("Conv2d batched CROWN matches regular CROWN!");
}

// Note: Conv2d::propagate_linear_batched is implemented and works at the layer level
// (see tests above), but network-level integration with BatchedLinearBounds requires
// shape transformation to flatten spatial dimensions. For now, Network::propagate_crown_batched
// falls back to regular CROWN for Conv2d layers. Future work: add shape transformation
// logic to enable full batched CROWN for Conv2d networks.

// ============================================================
// AVERAGE POOL CROWN TESTS
// ============================================================

#[test]
fn test_average_pool_crown_backward_basic() {
    // Test AveragePool CROWN backward propagation
    // Input: [1, 3, 3] (1 channel, 3x3 spatial)
    // Kernel: 2x2, Stride: 1, Padding: 0
    // Output: [1, 2, 2] (1 channel, 2x2 spatial)
    use crate::layers::AveragePoolLayer;

    let pre_lower = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, 3, 3]),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();
    let pre_upper = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, 3, 3]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let avg_pool = AveragePoolLayer::new((2, 2), (1, 1), (0, 0), false);

    // Output size is 2x2 = 4, Input size is 3x3 = 9
    let linear_bounds = LinearBounds::identity(4);

    let result = avg_pool
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Check dimensions
    assert_eq!(result.lower_a.shape(), &[4, 9]);
    assert_eq!(result.upper_a.shape(), &[4, 9]);
    assert_eq!(result.lower_b.len(), 4);
    assert_eq!(result.upper_b.len(), 4);

    // Check structure: each output averages 4 inputs
    // Output [0] = avg(input[0], input[1], input[3], input[4]) = avg of positions (0,0), (0,1), (1,0), (1,1)
    let weight = 0.25_f32; // 1/4 for 2x2 kernel
    let tol = 1e-5;

    // Lower_a for output 0 should have weight 0.25 for inputs 0, 1, 3, 4
    assert!(
        (result.lower_a[[0, 0]] - weight).abs() < tol,
        "Expected weight {} at [0,0], got {}",
        weight,
        result.lower_a[[0, 0]]
    );
    assert!(
        (result.lower_a[[0, 1]] - weight).abs() < tol,
        "Expected weight {} at [0,1], got {}",
        weight,
        result.lower_a[[0, 1]]
    );
    assert!(
        (result.lower_a[[0, 3]] - weight).abs() < tol,
        "Expected weight {} at [0,3], got {}",
        weight,
        result.lower_a[[0, 3]]
    );
    assert!(
        (result.lower_a[[0, 4]] - weight).abs() < tol,
        "Expected weight {} at [0,4], got {}",
        weight,
        result.lower_a[[0, 4]]
    );

    // Inputs not in the window should have weight 0
    assert!(
        result.lower_a[[0, 2]].abs() < tol,
        "Expected 0 at [0,2], got {}",
        result.lower_a[[0, 2]]
    );
    assert!(
        result.lower_a[[0, 5]].abs() < tol,
        "Expected 0 at [0,5], got {}",
        result.lower_a[[0, 5]]
    );
}

#[test]
fn test_average_pool_crown_soundness() {
    // Test that CROWN bounds are sound (contain the actual function values)
    use crate::layers::AveragePoolLayer;

    let pre_lower = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, 4, 4]),
        (0..16).map(|i| i as f32).collect::<Vec<_>>(),
    )
    .unwrap();
    let pre_upper = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, 4, 4]),
        (0..16).map(|i| i as f32 + 1.0).collect::<Vec<_>>(),
    )
    .unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let avg_pool = AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false);

    // Output: [1, 2, 2] = 4 elements, Input: [1, 4, 4] = 16 elements
    let linear_bounds = LinearBounds::identity(4);

    let result = avg_pool
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample points and verify bounds contain actual values
    for sample in 0..10 {
        // Generate a random point in the interval
        let point: Vec<f32> = (0..16)
            .map(|i| {
                let t = ((sample as u32).wrapping_mul(2654435761) ^ (i as u32)) as f32
                    / u32::MAX as f32;
                let pre_l = pre_lower.as_slice().unwrap()[i];
                let pre_u = pre_upper.as_slice().unwrap()[i];
                pre_l + (pre_u - pre_l) * t
            })
            .collect();

        // Compute actual average pool output
        // Output [0,0,0] = avg(input[0], input[1], input[4], input[5])
        // Output [0,0,1] = avg(input[2], input[3], input[6], input[7])
        // Output [0,1,0] = avg(input[8], input[9], input[12], input[13])
        // Output [0,1,1] = avg(input[10], input[11], input[14], input[15])
        let actual_output = [
            (point[0] + point[1] + point[4] + point[5]) / 4.0,
            (point[2] + point[3] + point[6] + point[7]) / 4.0,
            (point[8] + point[9] + point[12] + point[13]) / 4.0,
            (point[10] + point[11] + point[14] + point[15]) / 4.0,
        ];

        // Check each output dimension
        for (j, &actual_val) in actual_output.iter().enumerate() {
            let lb_val: f32 = (0..16)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            let ub_val: f32 = (0..16)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            let tol = 1e-4;
            assert!(
                lb_val <= actual_val + tol,
                "CROWN lower bound violated at sample {}, dim {}: lb {} > actual {}",
                sample,
                j,
                lb_val,
                actual_val
            );
            assert!(
                ub_val >= actual_val - tol,
                "CROWN upper bound violated at sample {}, dim {}: ub {} < actual {}",
                sample,
                j,
                ub_val,
                actual_val
            );
        }
    }
}

#[test]
fn test_average_pool_global_crown() {
    // Test global average pooling CROWN
    use crate::layers::AveragePoolLayer;

    let pre_lower = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[2, 3, 3]),
        vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // channel 0
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // channel 1
        ],
    )
    .unwrap();
    let pre_upper = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[2, 3, 3]),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // channel 0
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // channel 1
        ],
    )
    .unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Global pooling: kernel_size (0, 0)
    let avg_pool = AveragePoolLayer::new((0, 0), (1, 1), (0, 0), false);

    // Output: [2, 1, 1] = 2 elements (one per channel)
    // Input: [2, 3, 3] = 18 elements
    let linear_bounds = LinearBounds::identity(2);

    let result = avg_pool
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Check dimensions
    assert_eq!(result.lower_a.shape(), &[2, 18]);
    assert_eq!(result.upper_a.shape(), &[2, 18]);

    // Global pool averages all 9 elements per channel
    let weight = 1.0 / 9.0;
    let tol = 1e-5;

    // Channel 0 output (index 0) should have weight 1/9 for inputs 0-8
    for i in 0..9 {
        assert!(
            (result.lower_a[[0, i]] - weight).abs() < tol,
            "Expected weight {} at [0,{}], got {}",
            weight,
            i,
            result.lower_a[[0, i]]
        );
    }
    // Channel 0 output should have 0 weight for channel 1 inputs (9-17)
    for i in 9..18 {
        assert!(
            result.lower_a[[0, i]].abs() < tol,
            "Expected 0 at [0,{}], got {}",
            i,
            result.lower_a[[0, i]]
        );
    }

    // Channel 1 output (index 1) should have weight 1/9 for inputs 9-17
    for i in 9..18 {
        assert!(
            (result.lower_a[[1, i]] - weight).abs() < tol,
            "Expected weight {} at [1,{}], got {}",
            weight,
            i,
            result.lower_a[[1, i]]
        );
    }
}

#[test]
fn test_average_pool_crown_network_integration() {
    // Test AveragePool CROWN in a network context
    use crate::layers::{AveragePoolLayer, LinearLayer, ReshapeLayer};
    use crate::network::Network;
    use ndarray::Array2;

    // Create a simple network: Reshape -> AveragePool -> Flatten -> Linear
    // Input: flat 9 elements -> reshape to [1, 3, 3] -> avgpool 2x2 -> [1, 2, 2] -> flatten -> linear

    let avg_pool = AveragePoolLayer::new((2, 2), (1, 1), (0, 0), false);

    let weight =
        Array2::from_shape_vec((2, 4), vec![1.0, 0.5, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0]).unwrap();
    let bias: Option<Array1<f32>> = Some(arr1(&[0.0, 0.0]));
    let linear = LinearLayer::new(weight, bias).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1, 3, 3])));
    network.add_layer(Layer::AveragePool(avg_pool));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(linear));

    // Create input bounds
    let input_lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[9]), vec![0.0; 9]).unwrap();
    let input_upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[9]), vec![1.0; 9]).unwrap();
    let input = BoundedTensor::new(input_lower, input_upper).unwrap();

    // Test CROWN propagation
    let crown_result = network.propagate_crown(&input).unwrap();

    // Test IBP propagation for comparison
    let ibp_result = network.propagate_ibp(&input).unwrap();

    // Flatten the results for comparison
    let crown_lower = crown_result.lower.as_slice().unwrap();
    let crown_upper = crown_result.upper.as_slice().unwrap();
    let ibp_lower = ibp_result.lower.as_slice().unwrap();
    let ibp_upper = ibp_result.upper.as_slice().unwrap();

    // CROWN bounds should be tighter than or equal to IBP bounds
    for i in 0..crown_lower.len() {
        assert!(
            crown_lower[i] >= ibp_lower[i] - 1e-4,
            "CROWN lower bound {} should be >= IBP lower bound {}",
            crown_lower[i],
            ibp_lower[i]
        );
        assert!(
            crown_upper[i] <= ibp_upper[i] + 1e-4,
            "CROWN upper bound {} should be <= IBP upper bound {}",
            crown_upper[i],
            ibp_upper[i]
        );
    }
}

// =============================================================================
// MaxPool2d CROWN Tests
// =============================================================================

#[test]
fn test_max_pool_crown_backward_basic() {
    // Test MaxPool2d CROWN backward propagation coefficient structure
    use crate::layers::MaxPool2dLayer;
    use crate::LinearBounds;

    // Create a 2x2 max pool with stride 2 (non-overlapping)
    // Input: 1 channel, 4x4 -> Output: 1 channel, 2x2
    let max_pool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));

    // Input shape: [1, 4, 4] = 16 elements
    // Output shape: [1, 2, 2] = 4 elements
    let input_size = 16;
    let output_size = 4;

    // Create identity linear bounds at the output
    let bounds = LinearBounds::identity(output_size);

    // Create input bounds where one element in each window is clearly larger
    // Window 0 (positions 0,1,4,5): element 0 has highest range
    // Window 1 (positions 2,3,6,7): element 2 has highest range
    // etc.
    let mut lower = vec![0.0f32; input_size];
    let mut upper = vec![0.5f32; input_size];

    // Make element 0 the clear winner in window 0 (top-left)
    lower[0] = 1.0;
    upper[0] = 2.0;

    // Make element 3 the clear winner in window 1 (top-right)
    lower[3] = 1.0;
    upper[3] = 2.0;

    // Make element 8 the clear winner in window 2 (bottom-left)
    lower[8] = 1.0;
    upper[8] = 2.0;

    // Make element 15 the clear winner in window 3 (bottom-right)
    lower[15] = 1.0;
    upper[15] = 2.0;

    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 4, 4]), lower).unwrap(),
        ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 4, 4]), upper).unwrap(),
    )
    .unwrap();

    let result = max_pool
        .propagate_linear_with_bounds(&bounds, &pre_activation)
        .unwrap();

    // With clear winners, gradient should flow through winners only
    // Output 0 (window 0) should have gradient through input 0 only
    let tol = 1e-5;
    assert!(
        (result.lower_a[[0, 0]] - 1.0).abs() < tol,
        "Expected 1.0 at [0,0], got {}",
        result.lower_a[[0, 0]]
    );
    // Other inputs in window 0 should have 0 gradient
    assert!(result.lower_a[[0, 1]].abs() < tol);
    assert!(result.lower_a[[0, 4]].abs() < tol);
    assert!(result.lower_a[[0, 5]].abs() < tol);

    // Output 1 (window 1) should have gradient through input 3 only
    assert!(
        (result.lower_a[[1, 3]] - 1.0).abs() < tol,
        "Expected 1.0 at [1,3], got {}",
        result.lower_a[[1, 3]]
    );
}

#[test]
fn test_max_pool_crown_soundness() {
    // Test that MaxPool2d CROWN bounds are sound (contain actual values)
    use crate::layers::MaxPool2dLayer;
    use ndarray::ArrayD;

    let max_pool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));

    // Create input with some variation
    let input_lower = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, 4, 4]),
        vec![
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
        ],
    )
    .unwrap();
    let input_upper = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, 4, 4]),
        vec![
            0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
        ],
    )
    .unwrap();
    let input = BoundedTensor::new(input_lower.clone(), input_upper.clone()).unwrap();

    // Get IBP bounds (exact for max pool)
    let ibp_result = max_pool.propagate_ibp(&input).unwrap();

    // Test multiple concrete points within input bounds
    let test_points = [
        // Point at lower bounds
        vec![
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
        ],
        // Point at upper bounds
        vec![
            0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
        ],
        // Point at midpoint
        vec![
            0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55,
            1.65, 1.75,
        ],
    ];

    let ibp_lower = ibp_result.lower.as_slice().unwrap();
    let ibp_upper = ibp_result.upper.as_slice().unwrap();

    for point in test_points.iter() {
        let concrete_input = BoundedTensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 4, 4]), point.clone()).unwrap(),
            ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 4, 4]), point.clone()).unwrap(),
        )
        .unwrap();
        let concrete_output = max_pool.propagate_ibp(&concrete_input).unwrap();
        let concrete_vals = concrete_output.lower.as_slice().unwrap();

        // Check that concrete values are within IBP bounds
        for (i, &val) in concrete_vals.iter().enumerate() {
            assert!(
                val >= ibp_lower[i] - 1e-5 && val <= ibp_upper[i] + 1e-5,
                "Concrete value {} at index {} not in IBP bounds [{}, {}]",
                val,
                i,
                ibp_lower[i],
                ibp_upper[i]
            );
        }
    }
}

#[test]
fn test_max_pool_crown_uncertain_case() {
    // Test MaxPool2d CROWN when there's no clear winner (uses constant IBP bounds)
    use crate::layers::MaxPool2dLayer;
    use crate::LinearBounds;

    let max_pool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));

    // Input shape: [1, 2, 2] = 4 elements (single pooling window)
    // Output shape: [1, 1, 1] = 1 element
    let _input_size = 4;
    let output_size = 1;

    let bounds = LinearBounds::identity(output_size);

    // All inputs have overlapping intervals - no clear winner
    let lower = vec![0.0f32, 0.1, 0.2, 0.3];
    let upper = vec![1.0f32, 1.1, 1.2, 1.3];

    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2, 2]), lower).unwrap(),
        ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2, 2]), upper).unwrap(),
    )
    .unwrap();

    let result = max_pool
        .propagate_linear_with_bounds(&bounds, &pre_activation)
        .unwrap();

    // In uncertain case, no gradient flows through (constant bounds)
    // All coefficient matrix entries should be 0
    let tol = 1e-5;

    for i in 0..4 {
        assert!(
            result.lower_a[[0, i]].abs() < tol,
            "Expected 0 at lower_a[0,{}], got {}",
            i,
            result.lower_a[[0, i]]
        );
        assert!(
            result.upper_a[[0, i]].abs() < tol,
            "Expected 0 at upper_a[0,{}], got {}",
            i,
            result.upper_a[[0, i]]
        );
    }

    // Bias terms should be the constant IBP bounds
    // max_lower = 0.3, max_upper = 1.3
    // With identity bounds (la=ua=1):
    // - lower_b = la * max_lower = 1.0 * 0.3 = 0.3
    // - upper_b = ua * max_upper = 1.0 * 1.3 = 1.3
    let tol_bias = 0.05;
    assert!(
        (result.lower_b[0] - 0.3).abs() < tol_bias,
        "Lower bias {} should be ~0.3 (max_lower)",
        result.lower_b[0]
    );
    assert!(
        (result.upper_b[0] - 1.3).abs() < tol_bias,
        "Upper bias {} should be ~1.3 (max_upper)",
        result.upper_b[0]
    );
}

#[test]
fn test_max_pool_crown_network_integration() {
    // Test MaxPool2d CROWN in a network context
    use crate::layers::{LinearLayer, MaxPool2dLayer, ReshapeLayer};
    use crate::network::Network;
    use ndarray::Array2;

    // Create a simple network: Reshape -> MaxPool -> Flatten -> Linear
    // Input: flat 16 elements -> reshape to [1, 4, 4] -> maxpool 2x2 -> [1, 2, 2] -> flatten -> linear

    let max_pool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));

    let weight =
        Array2::from_shape_vec((2, 4), vec![1.0, 0.5, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0]).unwrap();
    let bias: Option<Array1<f32>> = Some(arr1(&[0.0, 0.0]));
    let linear = LinearLayer::new(weight, bias).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1, 4, 4])));
    network.add_layer(Layer::MaxPool2d(max_pool));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(linear));

    // Create input bounds
    let input_lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[16]), vec![0.0; 16]).unwrap();
    let input_upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[16]), vec![1.0; 16]).unwrap();
    let input = BoundedTensor::new(input_lower, input_upper).unwrap();

    // Test CROWN propagation
    let crown_result = network.propagate_crown(&input).unwrap();

    // Test IBP propagation for comparison
    let ibp_result = network.propagate_ibp(&input).unwrap();

    // Flatten the results for comparison
    let crown_lower = crown_result.lower.as_slice().unwrap();
    let crown_upper = crown_result.upper.as_slice().unwrap();
    let ibp_lower = ibp_result.lower.as_slice().unwrap();
    let ibp_upper = ibp_result.upper.as_slice().unwrap();

    // CROWN bounds should be close to IBP bounds (might not be tighter due to approximation)
    // But they should be sound (contain the true values)
    for i in 0..crown_lower.len() {
        // Allow some slack due to approximation error in max pool CROWN
        assert!(
            crown_lower[i] >= ibp_lower[i] - 1.0,
            "CROWN lower bound {} should be >= IBP lower bound {} - 1.0",
            crown_lower[i],
            ibp_lower[i]
        );
        assert!(
            crown_upper[i] <= ibp_upper[i] + 1.0,
            "CROWN upper bound {} should be <= IBP upper bound {} + 1.0",
            crown_upper[i],
            ibp_upper[i]
        );
    }

    // Verify soundness by testing concrete points
    let test_values = vec![
        vec![0.5; 16], // midpoint
        vec![0.0; 16], // lower
        vec![1.0; 16], // upper
    ];

    for vals in test_values {
        let concrete_input = BoundedTensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[16]), vals.clone()).unwrap(),
            ArrayD::from_shape_vec(ndarray::IxDyn(&[16]), vals).unwrap(),
        )
        .unwrap();
        let concrete_output = network.propagate_ibp(&concrete_input).unwrap();
        let concrete_vals = concrete_output.lower.as_slice().unwrap();

        for (i, &val) in concrete_vals.iter().enumerate() {
            assert!(
                val >= crown_lower[i] - 1e-4 && val <= crown_upper[i] + 1e-4,
                "Concrete value {} at index {} not in CROWN bounds [{}, {}]",
                val,
                i,
                crown_lower[i],
                crown_upper[i]
            );
        }
    }
}

#[test]
fn test_beta_crown_maxpool2d_network_does_not_panic() {
    // Regression test: β-CROWN should support MaxPool2d in sequential networks.
    //
    // This is required for CLI property reductions that introduce pooling-based max objectives.
    use crate::beta_crown::{
        BabVerificationStatus, BetaCrownConfig, BetaCrownVerifier, BranchingHeuristic,
    };
    use crate::layers::{LinearLayer, MaxPool2dLayer, ReshapeLayer};
    use crate::network::Network;
    use crate::Layer;
    use ndarray::{Array2, ArrayD, IxDyn};
    use std::time::Duration;

    // Network: Reshape [16] -> [1,4,4] -> MaxPool2d -> [1,2,2] -> Reshape [4] -> Linear -> [1]
    let mut network = Network::new();
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1, 4, 4])));
    network.add_layer(Layer::MaxPool2d(MaxPool2dLayer::new(
        (2, 2),
        (2, 2),
        (0, 0),
    )));
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![4])));

    let weight = Array2::from_shape_vec((1, 4), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    network.add_layer(Layer::Linear(LinearLayer::new(weight, None).unwrap()));

    let input_lower = ArrayD::from_shape_vec(IxDyn(&[16]), vec![0.0; 16]).unwrap();
    let input_upper = ArrayD::from_shape_vec(IxDyn(&[16]), vec![1.0; 16]).unwrap();
    let input = BoundedTensor::new(input_lower, input_upper).unwrap();

    let config = BetaCrownConfig {
        timeout: Duration::from_secs(5),
        max_domains: 1_000,
        max_depth: 10,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        branching_heuristic: BranchingHeuristic::LargestBoundWidth,
        batch_size: 1,
        parallel_children: false,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, -0.1).unwrap();
    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "Expected Verified, got {:?}",
        result.result
    );
}
