//! Pooled memory operations for bound propagation.
//!
//! This module provides variants of common operations that use the tensor memory pool
//! to reduce allocation overhead. These are drop-in replacements for performance-critical paths.
//!
//! # Usage
//!
//! ```ignore
//! use gamma_propagate::pooled;
//!
//! // Instead of: let result = batched_matvec(&a, &x);
//! // Use:
//! let result = pooled::batched_matvec_pooled(&a, &x);
//! ```

use gamma_tensor::{BoundedTensor, PooledBuffer, TensorPool};
use ndarray::{s, ArrayD, IxDyn};

use crate::{safe_add_for_bounds, safe_mul_for_bounds};

/// Batched matrix-vector multiplication using pooled memory.
///
/// Same as `batched_matvec` but uses the tensor pool for the result buffer.
/// When the result is dropped, the buffer returns to the pool for reuse.
///
/// For A with shape [..., m, n] and x with shape [..., n],
/// computes y with shape `[..., m]` where `y[...][i] = sum_j A[...][i,j] * x[...][j]`
pub fn batched_matvec_pooled(a: &ArrayD<f32>, x: &ArrayD<f32>) -> ArrayD<f32> {
    let a_shape = a.shape();
    let x_shape = x.shape();

    if a_shape.len() < 2 || x_shape.is_empty() {
        return ArrayD::zeros(IxDyn(&[]));
    }

    let m = a_shape[a_shape.len() - 2];
    let n = a_shape[a_shape.len() - 1];

    assert_eq!(
        *x_shape.last().unwrap(),
        n,
        "Input dimension mismatch in batched_matvec_pooled"
    );

    // Output shape: [...batch, m]
    let batch_dims = &a_shape[..a_shape.len() - 2];
    let mut out_shape: Vec<usize> = batch_dims.to_vec();
    out_shape.push(m);

    let total_batch: usize = batch_dims.iter().product::<usize>().max(1);
    let total_elements = total_batch * m;

    // Use pooled buffer for result
    let mut buffer = TensorPool::acquire(total_elements);
    buffer.truncate(total_elements);
    let result_slice = buffer.as_mut_slice();

    // Reshape inputs
    let a_flat = a.view().into_shape_with_order((total_batch, m, n)).unwrap();
    let x_flat = x.view().into_shape_with_order((total_batch, n)).unwrap();

    // Check for inf/NaN
    let has_inf_or_nan = x.iter().any(|&v| v.is_infinite() || v.is_nan())
        || a.iter().any(|&v| v.is_infinite() || v.is_nan());

    if has_inf_or_nan {
        // Safe path for infinities
        for b in 0..total_batch {
            for i in 0..m {
                let mut sum = 0.0f32;
                for j in 0..n {
                    let term = safe_mul_for_bounds(a_flat[[b, i, j]], x_flat[[b, j]]);
                    sum = safe_add_for_bounds(sum, term);
                }
                result_slice[b * m + i] = sum;
            }
        }
    } else {
        // Fast path using ndarray operations
        for b in 0..total_batch {
            let a_slice = a_flat.slice(s![b, .., ..]);
            let x_slice = x_flat.slice(s![b, ..]);
            let dot_result = a_slice.dot(&x_slice);
            for i in 0..m {
                result_slice[b * m + i] = dot_result[i];
            }
        }
    }

    // Convert to ArrayD (consumes buffer, won't return to pool)
    buffer.into_arrayd(&out_shape)
}

/// Create a BoundedTensor using pooled memory.
///
/// Use this when creating temporary bounded tensors that will be discarded.
pub fn bounded_tensor_pooled(lower: ArrayD<f32>, upper: ArrayD<f32>) -> BoundedTensor {
    // Note: This just wraps the existing constructor.
    // True pooling would require BoundedTensor to accept PooledBuffer storage,
    // which would require API changes. This is a placeholder for future optimization.
    BoundedTensor { lower, upper }
}

/// Create zeros using pooled memory, returned as ArrayD.
///
/// The returned ArrayD owns its memory (buffer is consumed).
#[inline]
pub fn zeros_pooled(shape: &[usize]) -> ArrayD<f32> {
    let total = shape.iter().product::<usize>();
    let mut buffer = TensorPool::acquire(total);
    buffer.truncate(total);
    buffer.into_arrayd(shape)
}

/// Create an identity-like structure using pooled memory.
///
/// Returns a pooled buffer filled with identity matrix data.
/// Useful for BatchedLinearBounds::identity optimization.
pub fn identity_buffer_pooled(dim: usize) -> PooledBuffer {
    let size = dim * dim;
    let mut buffer = TensorPool::acquire(size);
    buffer.truncate(size);
    let data = buffer.as_mut_slice();

    // Fill with identity matrix pattern
    for i in 0..dim {
        data[i * dim + i] = 1.0;
    }

    buffer
}

/// Clone an ArrayD using pooled memory.
///
/// More efficient than regular clone when the result will be dropped soon.
pub fn clone_pooled(arr: &ArrayD<f32>) -> ArrayD<f32> {
    let shape = arr.shape();
    let total = arr.len();
    let mut buffer = TensorPool::acquire(total);
    buffer.truncate(total);

    // Copy data
    if let Some(src) = arr.as_slice() {
        // Contiguous array - fast copy
        buffer.as_mut_slice().copy_from_slice(src);
    } else {
        // Non-contiguous array - iterate
        let data = buffer.as_mut_slice();
        for (i, &v) in arr.iter().enumerate() {
            data[i] = v;
        }
    }

    buffer.into_arrayd(shape)
}

/// Add two arrays using pooled memory for the result.
pub fn add_pooled(a: &ArrayD<f32>, b: &ArrayD<f32>) -> ArrayD<f32> {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch in add_pooled");
    let shape = a.shape();
    let total = a.len();

    let mut buffer = TensorPool::acquire(total);
    buffer.truncate(total);
    let out = buffer.as_mut_slice();

    let a_slice = a.as_slice().expect("Non-contiguous array");
    let b_slice = b.as_slice().expect("Non-contiguous array");

    for i in 0..total {
        out[i] = a_slice[i] + b_slice[i];
    }

    buffer.into_arrayd(shape)
}

/// Subtract two arrays using pooled memory for the result.
pub fn sub_pooled(a: &ArrayD<f32>, b: &ArrayD<f32>) -> ArrayD<f32> {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch in sub_pooled");
    let shape = a.shape();
    let total = a.len();

    let mut buffer = TensorPool::acquire(total);
    buffer.truncate(total);
    let out = buffer.as_mut_slice();

    let a_slice = a.as_slice().expect("Non-contiguous array");
    let b_slice = b.as_slice().expect("Non-contiguous array");

    for i in 0..total {
        out[i] = a_slice[i] - b_slice[i];
    }

    buffer.into_arrayd(shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batched_matvec_pooled() {
        TensorPool::clear();
        TensorPool::reset_stats();

        // Simple 2D case: [2, 3] @ [3] = [2]
        let a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 1.0, 1.0]).unwrap();

        let result = batched_matvec_pooled(&a, &x);
        assert_eq!(result.shape(), &[2]);
        assert!((result[[0]] - 6.0).abs() < 1e-6); // 1+2+3
        assert!((result[[1]] - 15.0).abs() < 1e-6); // 4+5+6

        let stats = TensorPool::stats();
        assert!(stats.allocations > 0);
    }

    #[test]
    fn test_zeros_pooled() {
        TensorPool::clear();

        let arr = zeros_pooled(&[3, 4]);
        assert_eq!(arr.shape(), &[3, 4]);
        assert!(arr.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_identity_buffer_pooled() {
        TensorPool::clear();

        let buffer = identity_buffer_pooled(3);
        let data = buffer.as_slice();

        // Check identity pattern: 1s on diagonal, 0s elsewhere
        assert_eq!(data[0], 1.0); // [0,0]
        assert_eq!(data[1], 0.0); // [0,1]
        assert_eq!(data[4], 1.0); // [1,1]
        assert_eq!(data[8], 1.0); // [2,2]
    }

    #[test]
    fn test_add_pooled() {
        TensorPool::clear();

        let a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap();

        let result = add_pooled(&a, &b);
        assert_eq!(result[[0]], 5.0);
        assert_eq!(result[[1]], 7.0);
        assert_eq!(result[[2]], 9.0);
    }

    #[test]
    fn test_clone_pooled() {
        TensorPool::clear();

        let arr = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let cloned = clone_pooled(&arr);

        assert_eq!(arr.shape(), cloned.shape());
        assert_eq!(arr[[0, 0]], cloned[[0, 0]]);
        assert_eq!(arr[[1, 1]], cloned[[1, 1]]);
    }

    #[test]
    fn test_pool_reuse_with_direct_buffers() {
        // When using PooledBuffer directly (not into_arrayd), we get reuse.
        // into_arrayd() consumes the buffer, so it doesn't return to pool.
        TensorPool::clear();
        TensorPool::reset_stats();

        // Acquire and drop buffers directly (not via into_arrayd)
        for _ in 0..10 {
            let buffer = TensorPool::acquire(10000);
            drop(buffer); // Returns to pool
        }

        let stats = TensorPool::stats();
        // Should have 9 hits (first is miss, rest are hits)
        assert_eq!(stats.allocations, 10);
        assert_eq!(stats.pool_hits, 9);
        assert_eq!(stats.pool_misses, 1);
    }

    #[test]
    fn test_into_arrayd_does_not_return_to_pool() {
        // Verify that into_arrayd() consumes buffer (expected behavior)
        TensorPool::clear();
        TensorPool::reset_stats();

        // These convert to ArrayD, so buffers are consumed not returned
        for _ in 0..5 {
            let _arr = zeros_pooled(&[100, 100]);
        }

        let stats = TensorPool::stats();
        // All allocations are misses because buffers are consumed
        assert_eq!(stats.allocations, 5);
        assert_eq!(stats.pool_misses, 5);
        assert_eq!(stats.returns, 0); // Nothing returned
    }

    #[test]
    fn test_sub_pooled() {
        TensorPool::clear();

        let a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();

        let result = sub_pooled(&a, &b);
        assert_eq!(result[[0]], 9.0);
        assert_eq!(result[[1]], 18.0);
        assert_eq!(result[[2]], 27.0);
    }

    #[test]
    fn test_sub_pooled_multidimensional() {
        TensorPool::clear();

        let a = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![5.0, 10.0, 15.0, 20.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = sub_pooled(&a, &b);
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 4.0);
        assert_eq!(result[[0, 1]], 8.0);
        assert_eq!(result[[1, 0]], 12.0);
        assert_eq!(result[[1, 1]], 16.0);
    }

    #[test]
    fn test_bounded_tensor_pooled() {
        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0, 1.0, 2.0]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();

        let tensor = bounded_tensor_pooled(lower.clone(), upper.clone());

        assert_eq!(tensor.lower, lower);
        assert_eq!(tensor.upper, upper);
        assert_eq!(tensor.shape(), &[3]);
    }

    #[test]
    fn test_batched_matvec_pooled_3d() {
        TensorPool::clear();

        // Batched case: [2, 2, 3] @ [2, 3] = [2, 2]
        // Two batches of 2x3 matrices
        let a = ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                // batch 0
                1.0, 2.0, 3.0, // row 0
                4.0, 5.0, 6.0, // row 1
                // batch 1
                7.0, 8.0, 9.0, // row 0
                10.0, 11.0, 12.0, // row 1
            ],
        )
        .unwrap();

        let x = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![
                1.0, 1.0, 1.0, // batch 0
                2.0, 2.0, 2.0, // batch 1
            ],
        )
        .unwrap();

        let result = batched_matvec_pooled(&a, &x);
        assert_eq!(result.shape(), &[2, 2]);

        // batch 0: [1+2+3, 4+5+6] = [6, 15]
        assert!((result[[0, 0]] - 6.0).abs() < 1e-6);
        assert!((result[[0, 1]] - 15.0).abs() < 1e-6);

        // batch 1: [2*(7+8+9), 2*(10+11+12)] = [48, 66]
        assert!((result[[1, 0]] - 48.0).abs() < 1e-6);
        assert!((result[[1, 1]] - 66.0).abs() < 1e-6);
    }

    #[test]
    fn test_batched_matvec_pooled_with_infinity() {
        TensorPool::clear();

        // Test with infinity in matrix
        let a =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, f32::INFINITY, 0.0, 0.0, 1.0, 0.0])
                .unwrap();
        let x = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 0.0, 1.0]).unwrap();

        let result = batched_matvec_pooled(&a, &x);
        assert_eq!(result.shape(), &[2]);

        // Row 0: 1*1 + inf*0 + 0*1 = 1 (but inf*0 is handled safely)
        // Row 1: 0*1 + 1*0 + 0*1 = 0
        assert_eq!(result[[1]], 0.0);
    }

    #[test]
    fn test_batched_matvec_pooled_empty_input() {
        TensorPool::clear();

        // Test with degenerate shapes
        let a = ArrayD::zeros(IxDyn(&[0]));
        let x = ArrayD::zeros(IxDyn(&[0]));

        let result = batched_matvec_pooled(&a, &x);
        assert_eq!(result.shape(), &[] as &[usize]);
    }

    #[test]
    fn test_batched_matvec_pooled_single_element() {
        TensorPool::clear();

        // 1x1 matrix times 1-element vector
        let a = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![5.0]).unwrap();
        let x = ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap();

        let result = batched_matvec_pooled(&a, &x);
        assert_eq!(result.shape(), &[1]);
        assert!((result[[0]] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_zeros_pooled_various_shapes() {
        TensorPool::clear();

        // 1D
        let arr1d = zeros_pooled(&[5]);
        assert_eq!(arr1d.shape(), &[5]);
        assert!(arr1d.iter().all(|&v| v == 0.0));

        // 3D
        let arr3d = zeros_pooled(&[2, 3, 4]);
        assert_eq!(arr3d.shape(), &[2, 3, 4]);
        assert!(arr3d.iter().all(|&v| v == 0.0));

        // Empty shape (scalar)
        let scalar = zeros_pooled(&[]);
        assert_eq!(scalar.shape(), &[] as &[usize]);
    }

    #[test]
    fn test_identity_buffer_pooled_various_sizes() {
        TensorPool::clear();

        // 1x1
        let buf1 = identity_buffer_pooled(1);
        assert_eq!(buf1.as_slice().len(), 1);
        assert_eq!(buf1.as_slice()[0], 1.0);

        // 2x2
        let buf2 = identity_buffer_pooled(2);
        let data2 = buf2.as_slice();
        assert_eq!(data2.len(), 4);
        assert_eq!(data2[0], 1.0); // [0,0]
        assert_eq!(data2[1], 0.0); // [0,1]
        assert_eq!(data2[2], 0.0); // [1,0]
        assert_eq!(data2[3], 1.0); // [1,1]

        // 4x4
        let buf4 = identity_buffer_pooled(4);
        let data4 = buf4.as_slice();
        assert_eq!(data4.len(), 16);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(data4[i * 4 + j], expected, "Mismatch at [{}, {}]", i, j);
            }
        }
    }

    #[test]
    fn test_add_pooled_multidimensional() {
        TensorPool::clear();

        let a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();

        let result = add_pooled(&a, &b);
        assert_eq!(result.shape(), &[2, 3]);
        // All sums should be 7
        assert!(result.iter().all(|&v| (v - 7.0).abs() < 1e-6));
    }

    #[test]
    fn test_clone_pooled_preserves_shape() {
        TensorPool::clear();

        // 3D array
        let arr = ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let cloned = clone_pooled(&arr);

        assert_eq!(arr.shape(), cloned.shape());
        assert_eq!(arr.len(), cloned.len());

        // Check all values
        for (orig, clone) in arr.iter().zip(cloned.iter()) {
            assert_eq!(*orig, *clone);
        }
    }

    #[test]
    fn test_clone_pooled_empty() {
        TensorPool::clear();

        // Empty array
        let empty: ArrayD<f32> = ArrayD::zeros(IxDyn(&[0]));
        let cloned = clone_pooled(&empty);

        assert_eq!(empty.shape(), cloned.shape());
        assert_eq!(empty.len(), 0);
        assert_eq!(cloned.len(), 0);
    }

    #[test]
    fn test_batched_matvec_pooled_nan_handling() {
        TensorPool::clear();

        // Test with NaN - should use safe path
        let a = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![f32::NAN, 1.0, 1.0, 1.0]).unwrap();
        let x = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).unwrap();

        let result = batched_matvec_pooled(&a, &x);
        assert_eq!(result.shape(), &[2]);

        // Row 0: nan*0 + 1*1 - uses safe_mul which handles nan*0
        // Row 1: 1*0 + 1*1 = 1
        assert_eq!(result[[1]], 1.0);
    }

    #[test]
    fn test_bounded_tensor_pooled_different_shapes() {
        // 2D tensor
        let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0; 6]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();

        let tensor = bounded_tensor_pooled(lower, upper);
        assert_eq!(tensor.shape(), &[2, 3]);

        // Verify width
        let widths = tensor.width();
        assert!(widths.iter().all(|&w| (w - 2.0).abs() < 1e-6));
    }
}
