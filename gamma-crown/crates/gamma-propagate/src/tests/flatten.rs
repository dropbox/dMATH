//! Tests for Flatten layer IBP propagation.
//!
//! This module tests the FlattenLayer implementation, including:
//! - Basic axis handling (axis 0, 1, -1)
//! - Interval bounds preservation
//! - Edge cases (1D input)
//! - CNN use case simulation

use super::*;
use ndarray::ArrayD;

#[test]
fn test_flatten_axis0() {
    // Flatten with axis=0: (C, H, W) -> (1, C*H*W)
    let flatten = FlattenLayer::new(0);

    // Input: (2, 3, 4) = 24 elements
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[2, 3, 4]));
    for c in 0..2 {
        for h in 0..3 {
            for w in 0..4 {
                input_data[[c, h, w]] = (c * 12 + h * 4 + w) as f32;
            }
        }
    }
    let input = BoundedTensor::concrete(input_data.clone());

    let output = flatten.propagate_ibp(&input).unwrap();

    // Output shape: (1, 24)
    assert_eq!(output.shape(), &[1, 24]);
    assert_eq!(output.lower.len(), 24);

    // Values should be preserved (in row-major order)
    let expected_flat = input_data.as_slice().unwrap();
    let actual_flat = output.lower.as_slice().unwrap();
    for (i, (&expected, &actual)) in expected_flat.iter().zip(actual_flat).enumerate() {
        assert!(
            (expected - actual).abs() < 1e-6,
            "Mismatch at index {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_flatten_axis1() {
    // Flatten with axis=1: (batch, C, H, W) -> (batch, C*H*W)
    // This is the common CNN -> Linear transition
    let flatten = FlattenLayer::new(1);

    // Input: (2, 3, 4) - treated as batch=2, rest=12
    let input_data = ArrayD::ones(ndarray::IxDyn(&[2, 3, 4]));
    let input = BoundedTensor::concrete(input_data);

    let output = flatten.propagate_ibp(&input).unwrap();

    // Output shape: (2, 12)
    assert_eq!(output.shape(), &[2, 12]);
}

#[test]
fn test_flatten_axis_negative() {
    // Flatten with axis=-1: should flatten the last dimension only
    let flatten = FlattenLayer::new(-1);

    // Input: (2, 3, 4)
    let input_data = ArrayD::ones(ndarray::IxDyn(&[2, 3, 4]));
    let input = BoundedTensor::concrete(input_data);

    let output = flatten.propagate_ibp(&input).unwrap();

    // axis=-1 in 3D means axis=2
    // Output shape: (2*3, 4) = (6, 4)
    assert_eq!(output.shape(), &[6, 4]);
}

#[test]
fn test_flatten_interval_bounds() {
    // Test that interval bounds are preserved correctly
    let flatten = FlattenLayer::new(0);

    // Create tensor with different lower and upper bounds
    let lower_data = ArrayD::zeros(ndarray::IxDyn(&[2, 3]));
    let upper_data = ArrayD::ones(ndarray::IxDyn(&[2, 3]));
    let input = BoundedTensor::new(lower_data, upper_data).unwrap();

    let output = flatten.propagate_ibp(&input).unwrap();

    // Output shape: (1, 6)
    assert_eq!(output.shape(), &[1, 6]);

    // All lower bounds should be 0, all upper bounds should be 1
    for &l in output.lower.iter() {
        assert!((l - 0.0).abs() < 1e-6);
    }
    for &u in output.upper.iter() {
        assert!((u - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_flatten_1d_input() {
    // Edge case: 1D input
    let flatten = FlattenLayer::new(0);

    let input_data = ArrayD::ones(ndarray::IxDyn(&[5]));
    let input = BoundedTensor::concrete(input_data);

    let output = flatten.propagate_ibp(&input).unwrap();

    // (5,) with axis=0 -> (1, 5)
    assert_eq!(output.shape(), &[1, 5]);
}

#[test]
fn test_flatten_cnn_use_case() {
    // Simulate CNN output: (out_channels, height, width) = (32, 8, 8)
    // Flatten with axis=0 for fully-connected layer
    let flatten = FlattenLayer::new(0);

    // Random-ish input simulating MaxPool output
    let mut lower_data = ArrayD::zeros(ndarray::IxDyn(&[32, 8, 8]));
    let mut upper_data = ArrayD::zeros(ndarray::IxDyn(&[32, 8, 8]));
    for c in 0..32 {
        for h in 0..8 {
            for w in 0..8 {
                lower_data[[c, h, w]] = (c as f32 * 0.1) - 0.5;
                upper_data[[c, h, w]] = (c as f32 * 0.1) + 0.5;
            }
        }
    }
    let input = BoundedTensor::new(lower_data, upper_data).unwrap();

    let output = flatten.propagate_ibp(&input).unwrap();

    // Output shape: (1, 32*8*8) = (1, 2048)
    assert_eq!(output.shape(), &[1, 2048]);

    // Verify bound widths are preserved
    let input_width = output.upper[[0, 0]] - output.lower[[0, 0]];
    assert!((input_width - 1.0).abs() < 1e-6, "Width should be 1.0");
}
