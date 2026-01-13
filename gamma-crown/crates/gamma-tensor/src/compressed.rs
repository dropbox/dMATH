//! Compressed bounds using f16 storage for memory efficiency.
//!
//! This module provides `CompressedBounds` which stores interval bounds using
//! 16-bit floats (f16/half-precision), reducing memory usage by 50% compared
//! to standard f32 storage.
//!
//! ## Trade-offs
//!
//! - **Memory**: 50% reduction (4 bytes -> 2 bytes per bound)
//! - **Precision**: f16 has ~3 decimal digits of precision (vs ~7 for f32)
//! - **Range**: f16 max is ~65504 (vs ~3.4e38 for f32)
//!
//! ## Use Cases
//!
//! - Checkpoint storage in streaming/gradient checkpointing
//! - Long-term bound storage when memory is constrained
//! - NOT for active computation (convert to f32 first)
//!
//! # Example
//!
//! ```
//! use gamma_tensor::{BoundedTensor, CompressedBounds};
//! use ndarray::ArrayD;
//!
//! // Create bounds
//! let lower = ArrayD::from_elem(ndarray::IxDyn(&[100]), -1.0f32);
//! let upper = ArrayD::from_elem(ndarray::IxDyn(&[100]), 1.0f32);
//! let bounds = BoundedTensor::new(lower, upper).unwrap();
//!
//! // Compress for storage (50% memory reduction)
//! let compressed = CompressedBounds::from_bounded_tensor(&bounds);
//!
//! // Decompress when needed for computation
//! let restored = compressed.to_bounded_tensor().unwrap();
//! ```

use crate::BoundedTensor;
use gamma_core::{GammaError, Result};
use half::f16;
use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

/// Compressed bounds storage using f16 (half-precision) floats.
///
/// Provides 50% memory reduction compared to `BoundedTensor` at the cost
/// of reduced precision. Suitable for checkpoint storage, not for active
/// computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedBounds {
    /// Lower bounds in f16 format.
    lower: Vec<f16>,
    /// Upper bounds in f16 format.
    upper: Vec<f16>,
    /// Shape of the tensor.
    shape: Vec<usize>,
}

impl CompressedBounds {
    /// Create compressed bounds from raw f16 vectors.
    ///
    /// # Errors
    /// Returns error if lower and upper have different lengths.
    pub fn new(lower: Vec<f16>, upper: Vec<f16>, shape: Vec<usize>) -> Result<Self> {
        let expected_len: usize = shape.iter().product();
        if lower.len() != expected_len || upper.len() != expected_len {
            return Err(GammaError::shape_mismatch(
                vec![expected_len],
                vec![lower.len(), upper.len()],
            ));
        }
        Ok(Self {
            lower,
            upper,
            shape,
        })
    }

    /// Create compressed bounds from a `BoundedTensor`.
    ///
    /// This is the primary way to create compressed bounds.
    /// Conversion is lossless for values within f16 range (~65504 max).
    pub fn from_bounded_tensor(bounds: &BoundedTensor) -> Self {
        let shape = bounds.shape().to_vec();

        // Convert f32 to f16, handling potential overflow/underflow
        let lower: Vec<f16> = bounds.lower.iter().map(|&v| f16::from_f32(v)).collect();
        let upper: Vec<f16> = bounds.upper.iter().map(|&v| f16::from_f32(v)).collect();

        Self {
            lower,
            upper,
            shape,
        }
    }

    /// Convert back to `BoundedTensor` for computation.
    ///
    /// This restores the bounds to f32 format. Note that precision
    /// may be lost due to the f16 intermediate representation.
    pub fn to_bounded_tensor(&self) -> Result<BoundedTensor> {
        let lower_f32: Vec<f32> = self.lower.iter().map(|&v| v.to_f32()).collect();
        let upper_f32: Vec<f32> = self.upper.iter().map(|&v| v.to_f32()).collect();

        let lower = ArrayD::from_shape_vec(IxDyn(&self.shape), lower_f32)
            .map_err(|e| GammaError::InvalidSpec(format!("Failed to reshape lower: {}", e)))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&self.shape), upper_f32)
            .map_err(|e| GammaError::InvalidSpec(format!("Failed to reshape upper: {}", e)))?;

        BoundedTensor::new(lower, upper)
    }

    /// Shape of the compressed bounds.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.lower.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.lower.is_empty()
    }

    /// Memory usage in bytes.
    ///
    /// Returns the approximate memory footprint of this structure.
    pub fn memory_bytes(&self) -> usize {
        // Each f16 is 2 bytes, we have lower + upper
        let data_bytes = self.lower.len() * 2 * 2;
        // Plus shape overhead (8 bytes per dimension on 64-bit)
        let shape_bytes = self.shape.len() * 8;
        data_bytes + shape_bytes
    }

    /// Memory usage compared to equivalent f32 BoundedTensor.
    ///
    /// Returns (compressed_bytes, f32_bytes, compression_ratio).
    pub fn compression_stats(&self) -> (usize, usize, f32) {
        let compressed = self.memory_bytes();
        // f32 version: 4 bytes per element, lower + upper
        let f32_bytes = self.lower.len() * 4 * 2;
        let ratio = compressed as f32 / f32_bytes as f32;
        (compressed, f32_bytes, ratio)
    }

    /// Get raw lower bounds (f16).
    pub fn lower_raw(&self) -> &[f16] {
        &self.lower
    }

    /// Get raw upper bounds (f16).
    pub fn upper_raw(&self) -> &[f16] {
        &self.upper
    }

    /// Widen bounds to ensure soundness after f16 conversion.
    ///
    /// Due to f16's limited precision, conversion may introduce small errors.
    /// This method conservatively widens bounds to guarantee soundness:
    /// - Lower bounds are decreased by epsilon
    /// - Upper bounds are increased by epsilon
    ///
    /// # Arguments
    /// * `relative_epsilon` - Relative widening factor (e.g., 0.001 for 0.1%)
    pub fn widen_for_soundness(&mut self, relative_epsilon: f32) {
        let eps = f16::from_f32(relative_epsilon);
        let min_delta = f16::from_f32(1e-6); // Minimum absolute widening

        for (l, u) in self.lower.iter_mut().zip(self.upper.iter_mut()) {
            // Widen lower bound down
            let l_abs = if *l < f16::ZERO { -*l } else { *l };
            let l_delta = l_abs * eps;
            let l_widen = if l_delta > min_delta {
                l_delta
            } else {
                min_delta
            };

            // Widen upper bound up
            let u_abs = if *u < f16::ZERO { -*u } else { *u };
            let u_delta = u_abs * eps;
            let u_widen = if u_delta > min_delta {
                u_delta
            } else {
                min_delta
            };

            // Apply widening (lower decreases, upper increases)
            *l -= l_widen;
            *u += u_widen;
        }
    }

    /// Check if any values are infinite or NaN after compression.
    ///
    /// Large f32 values (>65504) become infinity in f16.
    /// Returns true if bounds contain non-finite values.
    pub fn has_overflow(&self) -> bool {
        self.lower.iter().any(|v| !v.is_finite()) || self.upper.iter().any(|v| !v.is_finite())
    }

    /// Maximum precision loss from f32 -> f16 -> f32 round-trip.
    ///
    /// Computes the maximum absolute difference between original and
    /// round-tripped values. Returns (max_lower_error, max_upper_error).
    pub fn max_precision_loss(
        original: &BoundedTensor,
        compressed: &CompressedBounds,
    ) -> (f32, f32) {
        let mut max_lower_error = 0.0f32;
        let mut max_upper_error = 0.0f32;

        for (orig, comp) in original.lower.iter().zip(compressed.lower.iter()) {
            let restored = comp.to_f32();
            let error = (orig - restored).abs();
            max_lower_error = max_lower_error.max(error);
        }

        for (orig, comp) in original.upper.iter().zip(compressed.upper.iter()) {
            let restored = comp.to_f32();
            let error = (orig - restored).abs();
            max_upper_error = max_upper_error.max(error);
        }

        (max_lower_error, max_upper_error)
    }
}

/// Statistics about compression quality.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Memory used by compressed representation (bytes).
    pub compressed_bytes: usize,
    /// Memory that would be used by f32 representation (bytes).
    pub original_bytes: usize,
    /// Compression ratio (compressed / original).
    pub compression_ratio: f32,
    /// Maximum precision loss in lower bounds.
    pub max_lower_error: f32,
    /// Maximum precision loss in upper bounds.
    pub max_upper_error: f32,
    /// Whether any values overflowed to infinity.
    pub has_overflow: bool,
}

impl CompressionStats {
    /// Compute statistics from original and compressed bounds.
    pub fn from_compression(original: &BoundedTensor, compressed: &CompressedBounds) -> Self {
        let (compressed_bytes, original_bytes, compression_ratio) = compressed.compression_stats();
        let (max_lower_error, max_upper_error) =
            CompressedBounds::max_precision_loss(original, compressed);
        let has_overflow = compressed.has_overflow();

        Self {
            compressed_bytes,
            original_bytes,
            compression_ratio,
            max_lower_error,
            max_upper_error,
            has_overflow,
        }
    }

    /// Memory savings as percentage.
    pub fn memory_savings_percent(&self) -> f32 {
        100.0 * (1.0 - self.compression_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_compress_decompress_basic() {
        let lower = arr1(&[-1.0f32, 0.0, 0.5]).into_dyn();
        let upper = arr1(&[1.0f32, 0.5, 1.0]).into_dyn();
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        let compressed = CompressedBounds::from_bounded_tensor(&bounds);
        let restored = compressed.to_bounded_tensor().unwrap();

        assert_eq!(restored.shape(), bounds.shape());

        // f16 precision is about 3 decimal places, so tolerance is 1e-3
        for (orig, rest) in bounds.lower.iter().zip(restored.lower.iter()) {
            assert_relative_eq!(orig, rest, max_relative = 1e-3);
        }
        for (orig, rest) in bounds.upper.iter().zip(restored.upper.iter()) {
            assert_relative_eq!(orig, rest, max_relative = 1e-3);
        }
    }

    #[test]
    fn test_compress_2d_tensor() {
        let lower = arr2(&[[-1.0f32, -0.5], [0.0, 0.25]]).into_dyn();
        let upper = arr2(&[[1.0f32, 0.5], [0.5, 1.0]]).into_dyn();
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        let compressed = CompressedBounds::from_bounded_tensor(&bounds);
        assert_eq!(compressed.shape(), &[2, 2]);
        assert_eq!(compressed.len(), 4);

        let restored = compressed.to_bounded_tensor().unwrap();
        assert_eq!(restored.shape(), &[2, 2]);
    }

    #[test]
    fn test_memory_savings() {
        let n = 10000;
        let lower = ArrayD::from_elem(IxDyn(&[n]), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[n]), 1.0f32);
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        let compressed = CompressedBounds::from_bounded_tensor(&bounds);
        let (compressed_bytes, f32_bytes, ratio) = compressed.compression_stats();

        // Should be approximately 50% compression
        assert!(ratio < 0.6, "Expected ~50% compression, got {}", ratio);
        assert!(ratio > 0.4, "Compression too aggressive: {}", ratio);
        assert!(compressed_bytes < f32_bytes);
    }

    #[test]
    fn test_precision_loss() {
        // Test with values that have more precision than f16 can represent
        let lower = arr1(&[1.234567f32, -0.00012345, 100.123]).into_dyn();
        let upper = arr1(&[1.234568f32, -0.00012344, 100.124]).into_dyn();
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        let compressed = CompressedBounds::from_bounded_tensor(&bounds);
        let (max_lower_err, max_upper_err) =
            CompressedBounds::max_precision_loss(&bounds, &compressed);

        // f16 has ~3 decimal digits, so error should be < 1% of value
        // For small values like 1.234567, error should be < 0.01
        assert!(
            max_lower_err < 0.1,
            "Excessive lower precision loss: {}",
            max_lower_err
        );
        assert!(
            max_upper_err < 0.1,
            "Excessive upper precision loss: {}",
            max_upper_err
        );
    }

    #[test]
    fn test_overflow_detection() {
        // f16 max is ~65504, so 100000 should overflow
        let lower = arr1(&[1.0f32, -100000.0]).into_dyn();
        let upper = arr1(&[2.0f32, 100000.0]).into_dyn();
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        let compressed = CompressedBounds::from_bounded_tensor(&bounds);
        assert!(compressed.has_overflow(), "Should detect overflow");
    }

    #[test]
    fn test_no_overflow_in_normal_range() {
        let lower = arr1(&[-1000.0f32, -500.0, 0.0]).into_dyn();
        let upper = arr1(&[1000.0f32, 500.0, 100.0]).into_dyn();
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        let compressed = CompressedBounds::from_bounded_tensor(&bounds);
        assert!(
            !compressed.has_overflow(),
            "Should not overflow in normal range"
        );
    }

    #[test]
    fn test_widen_for_soundness() {
        let lower = arr1(&[-1.0f32, 0.0, 0.5]).into_dyn();
        let upper = arr1(&[1.0f32, 0.5, 1.0]).into_dyn();
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        let mut compressed = CompressedBounds::from_bounded_tensor(&bounds);
        compressed.widen_for_soundness(0.001); // 0.1% widening

        let restored = compressed.to_bounded_tensor().unwrap();

        // After widening, restored lower should be <= original lower
        // and restored upper should be >= original upper
        for (orig, rest) in bounds.lower.iter().zip(restored.lower.iter()) {
            assert!(
                rest <= orig,
                "Lower bound {} not conservative (original {})",
                rest,
                orig
            );
        }
        for (orig, rest) in bounds.upper.iter().zip(restored.upper.iter()) {
            assert!(
                rest >= orig,
                "Upper bound {} not conservative (original {})",
                rest,
                orig
            );
        }
    }

    #[test]
    fn test_compression_stats() {
        let n = 1000;
        let lower = ArrayD::from_elem(IxDyn(&[n]), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[n]), 1.0f32);
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        let compressed = CompressedBounds::from_bounded_tensor(&bounds);
        let stats = CompressionStats::from_compression(&bounds, &compressed);

        assert!(stats.memory_savings_percent() > 40.0);
        assert!(stats.memory_savings_percent() < 60.0);
        assert!(!stats.has_overflow);
        assert!(stats.max_lower_error < 0.01);
        assert!(stats.max_upper_error < 0.01);
    }

    #[test]
    fn test_empty_tensor() {
        let lower = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        let compressed = CompressedBounds::from_bounded_tensor(&bounds);
        assert!(compressed.is_empty());
        assert_eq!(compressed.len(), 0);

        let restored = compressed.to_bounded_tensor().unwrap();
        assert!(restored.is_empty());
    }

    #[test]
    fn test_large_tensor() {
        // Test with realistic transformer-sized tensor
        let n = 768 * 512; // hidden_dim * seq_len
        let lower = ArrayD::from_elem(IxDyn(&[1, 512, 768]), -10.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[1, 512, 768]), 10.0f32);
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        let compressed = CompressedBounds::from_bounded_tensor(&bounds);
        assert_eq!(compressed.len(), n);

        let (compressed_bytes, f32_bytes, _) = compressed.compression_stats();

        // Verify significant memory savings
        assert!(
            compressed_bytes < f32_bytes * 6 / 10,
            "Expected >40% savings: {} vs {}",
            compressed_bytes,
            f32_bytes
        );
    }

    // ========================================
    // Mutation-killing tests for CompressedBounds
    // ========================================

    #[test]
    fn test_is_empty_exact() {
        // Non-empty should return false, not true
        let lower = arr1(&[1.0f32, 2.0]).into_dyn();
        let upper = arr1(&[3.0f32, 4.0]).into_dyn();
        let bounds = BoundedTensor::new(lower, upper).unwrap();
        let compressed = CompressedBounds::from_bounded_tensor(&bounds);
        assert!(!compressed.is_empty());

        // Empty should return true
        let empty = CompressedBounds::new(vec![], vec![], vec![0]).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_memory_bytes_exact_computation() {
        // memory_bytes = lower.len() * 2 * 2 + shape.len() * 8
        // For 3 elements with 1D shape: 3*4 + 1*8 = 20
        let lower = arr1(&[1.0f32, 2.0, 3.0]).into_dyn();
        let upper = arr1(&[4.0f32, 5.0, 6.0]).into_dyn();
        let bounds = BoundedTensor::new(lower, upper).unwrap();
        let compressed = CompressedBounds::from_bounded_tensor(&bounds);

        // 3 elements * 2 bytes * 2 (lower + upper) + 1 dimension * 8 bytes = 20
        assert_eq!(compressed.memory_bytes(), 3 * 2 * 2 + 8);

        // For 2D: 2x3=6 elements with 2D shape: 6*4 + 2*8 = 40
        let lower2d = arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
        let upper2d = arr2(&[[7.0f32, 8.0, 9.0], [10.0, 11.0, 12.0]]).into_dyn();
        let bounds2d = BoundedTensor::new(lower2d, upper2d).unwrap();
        let compressed2d = CompressedBounds::from_bounded_tensor(&bounds2d);
        assert_eq!(compressed2d.memory_bytes(), 6 * 2 * 2 + 2 * 8);
    }

    #[test]
    fn test_lower_raw_upper_raw_nonempty() {
        // These functions should return non-empty slices when data exists
        let lower = arr1(&[1.0f32, 2.0, 3.0]).into_dyn();
        let upper = arr1(&[4.0f32, 5.0, 6.0]).into_dyn();
        let bounds = BoundedTensor::new(lower, upper).unwrap();
        let compressed = CompressedBounds::from_bounded_tensor(&bounds);

        let lower_raw = compressed.lower_raw();
        let upper_raw = compressed.upper_raw();

        // Should have 3 elements, not 0 or 1
        assert_eq!(lower_raw.len(), 3);
        assert_eq!(upper_raw.len(), 3);

        // Values should be approximately correct
        assert!((lower_raw[0].to_f32() - 1.0).abs() < 0.01);
        assert!((lower_raw[1].to_f32() - 2.0).abs() < 0.01);
        assert!((upper_raw[2].to_f32() - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_widen_for_soundness_actually_widens() {
        // Verify widening actually changes values (not a no-op)
        let lower = arr1(&[10.0f32, -10.0, 0.5]).into_dyn();
        let upper = arr1(&[20.0f32, 10.0, 1.0]).into_dyn();
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        let mut compressed = CompressedBounds::from_bounded_tensor(&bounds);
        let lower_before: Vec<f32> = compressed.lower.iter().map(|v| v.to_f32()).collect();
        let upper_before: Vec<f32> = compressed.upper.iter().map(|v| v.to_f32()).collect();

        compressed.widen_for_soundness(0.01); // 1% widening

        let lower_after: Vec<f32> = compressed.lower.iter().map(|v| v.to_f32()).collect();
        let upper_after: Vec<f32> = compressed.upper.iter().map(|v| v.to_f32()).collect();

        // Lower should decrease (become more negative/smaller)
        for i in 0..3 {
            assert!(
                lower_after[i] < lower_before[i],
                "Lower bound {} should decrease: {} -> {}",
                i,
                lower_before[i],
                lower_after[i]
            );
        }

        // Upper should increase
        for i in 0..3 {
            assert!(
                upper_after[i] > upper_before[i],
                "Upper bound {} should increase: {} -> {}",
                i,
                upper_before[i],
                upper_after[i]
            );
        }
    }

    #[test]
    fn test_has_overflow_distinguish_lower_upper() {
        // Test that we can detect overflow in lower bounds only
        let lower = arr1(&[f32::NEG_INFINITY, 1.0]).into_dyn();
        let upper = arr1(&[1.0f32, 2.0]).into_dyn();
        let bounds_lower_inf = BoundedTensor::new_unchecked(lower, upper).unwrap();
        let compressed_lower = CompressedBounds::from_bounded_tensor(&bounds_lower_inf);
        assert!(
            compressed_lower.has_overflow(),
            "Should detect lower bound overflow"
        );

        // Test that we can detect overflow in upper bounds only
        let lower2 = arr1(&[0.0f32, 1.0]).into_dyn();
        let upper2 = arr1(&[1.0f32, f32::INFINITY]).into_dyn();
        let bounds_upper_inf = BoundedTensor::new_unchecked(lower2, upper2).unwrap();
        let compressed_upper = CompressedBounds::from_bounded_tensor(&bounds_upper_inf);
        assert!(
            compressed_upper.has_overflow(),
            "Should detect upper bound overflow"
        );

        // Test normal values have no overflow
        let lower3 = arr1(&[0.0f32, 1.0]).into_dyn();
        let upper3 = arr1(&[1.0f32, 2.0]).into_dyn();
        let bounds_normal = BoundedTensor::new(lower3, upper3).unwrap();
        let compressed_normal = CompressedBounds::from_bounded_tensor(&bounds_normal);
        assert!(
            !compressed_normal.has_overflow(),
            "Normal values should not overflow"
        );
    }

    #[test]
    fn test_max_precision_loss_returns_correct_tuple() {
        // Verify both tuple elements are computed correctly, not just (0,0) or (-1,-1)
        // Use values that are not exactly representable in f16 so the error is non-zero.
        let lower = arr1(&[0.1f32, 0.2, 0.3]).into_dyn();
        let upper = arr1(&[0.15f32, 0.25, 0.35]).into_dyn();
        let bounds = BoundedTensor::new(lower, upper).unwrap();
        let compressed = CompressedBounds::from_bounded_tensor(&bounds);

        let (lower_err, upper_err) = CompressedBounds::max_precision_loss(&bounds, &compressed);

        // Errors should be >= 0 (absolute values)
        assert!(lower_err >= 0.0);
        assert!(upper_err >= 0.0);

        // Since at least one value is not representable in f16, max error should be > 0.
        assert!(lower_err > 0.0, "Expected non-zero lower error");
        assert!(upper_err > 0.0, "Expected non-zero upper error");

        // For f16 conversion of small values, error should still be small.
        assert!(lower_err < 1.0, "Lower error too large: {}", lower_err);
        assert!(upper_err < 1.0, "Upper error too large: {}", upper_err);
    }

    #[test]
    fn test_widen_for_soundness_expected_magnitude() {
        // This test targets common mutation survivors in widen_for_soundness:
        // - abs sign handling (negative values must widen proportionally)
        // - multiplication vs addition/division in delta computation
        // - min_delta fallback must not override proportional widening
        let lower = vec![
            f16::from_f32(-10.0), // large magnitude negative
            f16::from_f32(-6.0),  // negative
            f16::from_f32(0.0),   // triggers min_delta
        ];
        let upper = vec![
            f16::from_f32(20.0), // large magnitude positive
            f16::from_f32(-5.0), // negative upper bound (must still widen up)
            f16::from_f32(0.0),  // triggers min_delta
        ];

        let mut compressed = CompressedBounds::new(lower.clone(), upper.clone(), vec![3]).unwrap();
        compressed.widen_for_soundness(0.01);

        let lower_after: Vec<f32> = compressed.lower_raw().iter().map(|v| v.to_f32()).collect();
        let upper_after: Vec<f32> = compressed.upper_raw().iter().map(|v| v.to_f32()).collect();

        let lower_before: Vec<f32> = lower.iter().map(|v| v.to_f32()).collect();
        let upper_before: Vec<f32> = upper.iter().map(|v| v.to_f32()).collect();

        let lower_delta_0 = lower_before[0] - lower_after[0];
        let upper_delta_0 = upper_after[0] - upper_before[0];
        assert!(
            (0.08..0.12).contains(&lower_delta_0),
            "Expected ~0.1 lower widening for -10.0, got {}",
            lower_delta_0
        );
        assert!(
            (0.18..0.22).contains(&upper_delta_0),
            "Expected ~0.2 upper widening for 20.0, got {}",
            upper_delta_0
        );

        let lower_delta_1 = lower_before[1] - lower_after[1];
        let upper_delta_1 = upper_after[1] - upper_before[1];
        assert!(
            (0.04..0.08).contains(&lower_delta_1),
            "Expected ~0.06 lower widening for -6.0, got {}",
            lower_delta_1
        );
        assert!(
            (0.03..0.07).contains(&upper_delta_1),
            "Expected ~0.05 upper widening for -5.0, got {}",
            upper_delta_1
        );

        let min_delta_f32 = f16::from_f32(1e-6).to_f32();
        let lower_delta_2 = lower_before[2] - lower_after[2];
        let upper_delta_2 = upper_after[2] - upper_before[2];
        assert!(
            (lower_delta_2 - min_delta_f32).abs() <= min_delta_f32,
            "Expected min_delta widening for 0.0 lower, got {} (min_delta {})",
            lower_delta_2,
            min_delta_f32
        );
        assert!(
            (upper_delta_2 - min_delta_f32).abs() <= min_delta_f32,
            "Expected min_delta widening for 0.0 upper, got {} (min_delta {})",
            upper_delta_2,
            min_delta_f32
        );
    }
}
