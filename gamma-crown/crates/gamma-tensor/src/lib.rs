//! Bounded tensor types with interval arithmetic.
//!
//! This crate provides tensor types where each element has lower and upper bounds,
//! supporting the bound propagation algorithms in γ-CROWN.
//!
//! # Bound Representations
//!
//! - [`BoundedTensor`]: Interval bounds [lower, upper]. Simple but loose.
//! - [`zonotope::ZonotopeTensor`]: Correlation-aware bounds. Tighter for attention.
//! - [`compressed::CompressedBounds`]: f16 storage for memory efficiency.
//!
//! # Memory Pooling
//!
//! - [`pool::TensorPool`]: Thread-local memory pool for buffer reuse.
//! - [`pool::PooledBuffer`]: Auto-returning buffer handle.
//!
//! # SIMD Acceleration
//!
//! - [`simd`]: Vectorized interval arithmetic operations for hot paths.

use gamma_core::{Bound, GammaError, Result};
use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

pub mod compressed;
pub mod pool;
pub mod simd;
pub mod zonotope;

pub use compressed::{CompressedBounds, CompressionStats};
pub use pool::{PoolStats, PooledBuffer, TensorPool};
pub use zonotope::ZonotopeTensor;

#[inline]
fn next_up_f32(x: f32) -> f32 {
    if x.is_nan() || x == f32::INFINITY {
        return x;
    }
    if x == 0.0 {
        // Smallest positive subnormal.
        return f32::from_bits(1);
    }

    let bits = x.to_bits();
    if x.is_sign_positive() {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

#[inline]
fn next_down_f32(x: f32) -> f32 {
    if x.is_nan() || x == f32::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        // Smallest negative subnormal.
        return f32::from_bits(0x8000_0001);
    }

    let bits = x.to_bits();
    if x.is_sign_positive() {
        f32::from_bits(bits - 1)
    } else {
        f32::from_bits(bits + 1)
    }
}

/// A tensor where each element has certified lower and upper bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedTensor {
    /// Lower bounds for each element.
    pub lower: ArrayD<f32>,
    /// Upper bounds for each element.
    pub upper: ArrayD<f32>,
}

impl BoundedTensor {
    /// Check if array contains NaN or Inf values.
    /// Used by debug_assert to catch numerical issues early.
    #[inline]
    fn has_nan_or_inf(arr: &ArrayD<f32>) -> bool {
        arr.iter().any(|&v| v.is_nan() || v.is_infinite())
    }

    /// Create a bounded tensor from lower and upper bound arrays.
    ///
    /// # Debug Assertions
    /// In debug builds, this function asserts that:
    /// - Neither array contains NaN or infinite values
    /// - All lower bounds are <= upper bounds
    ///
    /// # Soundness
    /// NaN/Inf inputs indicate numerical issues upstream that would
    /// invalidate verification results. Catching them early aids debugging.
    #[inline]
    pub fn new(lower: ArrayD<f32>, upper: ArrayD<f32>) -> Result<Self> {
        if lower.shape() != upper.shape() {
            return Err(GammaError::shape_mismatch(
                lower.shape().to_vec(),
                upper.shape().to_vec(),
            ));
        }

        // Debug-only checks for numerical soundness
        debug_assert!(
            !Self::has_nan_or_inf(&lower),
            "BoundedTensor::new: lower bounds contain NaN or Inf"
        );
        debug_assert!(
            !Self::has_nan_or_inf(&upper),
            "BoundedTensor::new: upper bounds contain NaN or Inf"
        );
        debug_assert!(
            ndarray::Zip::from(&lower)
                .and(&upper)
                .all(|&l, &u| l <= u || l.is_nan() || u.is_nan()),
            "BoundedTensor::new: found lower > upper (inverted bounds)"
        );

        Ok(Self { lower, upper })
    }

    /// Create a concrete tensor (lower == upper).
    ///
    /// # Debug Assertions
    /// In debug builds, asserts that values contain no NaN or Inf.
    pub fn concrete(values: ArrayD<f32>) -> Self {
        debug_assert!(
            !Self::has_nan_or_inf(&values),
            "BoundedTensor::concrete: values contain NaN or Inf"
        );
        Self {
            lower: values.clone(),
            upper: values,
        }
    }

    /// Create a bounded tensor from a concrete value and perturbation epsilon.
    ///
    /// # Debug Assertions
    /// In debug builds, asserts that:
    /// - Values contain no NaN or Inf
    /// - Epsilon is non-negative and finite
    pub fn from_epsilon(values: ArrayD<f32>, epsilon: f32) -> Self {
        debug_assert!(
            !Self::has_nan_or_inf(&values),
            "BoundedTensor::from_epsilon: values contain NaN or Inf"
        );
        debug_assert!(
            epsilon >= 0.0 && epsilon.is_finite(),
            "BoundedTensor::from_epsilon: epsilon must be non-negative and finite, got {}",
            epsilon
        );
        Self {
            lower: values.mapv(|v| v - epsilon),
            upper: values.mapv(|v| v + epsilon),
        }
    }

    /// Create a bounded tensor without debug assertions.
    ///
    /// # Safety Note
    /// This bypasses NaN/Inf/inverted-bounds checks. Only use for:
    /// - Testing sanitization/fallback code that must handle bad inputs
    /// - Performance-critical paths where inputs are already validated
    ///
    /// For normal use, prefer [`Self::new`] which catches numerical issues early.
    #[inline]
    pub fn new_unchecked(lower: ArrayD<f32>, upper: ArrayD<f32>) -> Result<Self> {
        if lower.shape() != upper.shape() {
            return Err(GammaError::shape_mismatch(
                lower.shape().to_vec(),
                upper.shape().to_vec(),
            ));
        }
        Ok(Self { lower, upper })
    }

    /// Create a bounded tensor with NaN/Inf values clamped.
    ///
    /// This sanitizes bounds by replacing:
    /// - NaN in lower → -clamp_val
    /// - NaN in upper → +clamp_val
    /// - +Inf → +clamp_val
    /// - -Inf → -clamp_val
    /// - Values outside [-clamp_val, clamp_val] → clamped
    ///
    /// Use this when `continue_after_overflow` mode is enabled and you need to
    /// handle explosive bounds that would otherwise cause debug assertion panics.
    ///
    /// The clamped tensor passes all debug assertions and can be safely used
    /// in subsequent propagation operations.
    #[inline]
    pub fn new_sanitized(lower: ArrayD<f32>, upper: ArrayD<f32>, clamp_val: f32) -> Result<Self> {
        if lower.shape() != upper.shape() {
            return Err(GammaError::shape_mismatch(
                lower.shape().to_vec(),
                upper.shape().to_vec(),
            ));
        }

        let original_shape = lower.shape().to_vec();

        let lower = lower.mapv(|x| {
            if x.is_nan() {
                -clamp_val
            } else if x.is_infinite() {
                if x > 0.0 {
                    clamp_val
                } else {
                    -clamp_val
                }
            } else {
                x.clamp(-clamp_val, clamp_val)
            }
        });

        let upper = upper.mapv(|x| {
            if x.is_nan() {
                clamp_val
            } else if x.is_infinite() {
                if x > 0.0 {
                    clamp_val
                } else {
                    -clamp_val
                }
            } else {
                x.clamp(-clamp_val, clamp_val)
            }
        });

        // Ensure lower <= upper (NaN replacement may have inverted them)
        // Use par_map_inplace for efficiency on large tensors
        let mut result_lower = lower;
        let mut result_upper = upper;
        ndarray::Zip::from(&mut result_lower)
            .and(&mut result_upper)
            .for_each(|l, u| {
                if *l > *u {
                    std::mem::swap(l, u);
                }
            });

        debug_assert_eq!(result_lower.shape(), &original_shape[..]);
        Ok(Self {
            lower: result_lower,
            upper: result_upper,
        })
    }

    /// Sanitize this tensor by clamping NaN/Inf values.
    ///
    /// Returns a new tensor with the same shape where all NaN/Inf values
    /// have been replaced with clamped finite values.
    ///
    /// See [`Self::new_sanitized`] for details on the clamping behavior.
    #[inline]
    pub fn sanitize(&self, clamp_val: f32) -> Self {
        // Safe to unwrap: shapes already match (self is valid)
        Self::new_sanitized(self.lower.clone(), self.upper.clone(), clamp_val).unwrap()
    }

    /// Check if this tensor contains any NaN or Inf values.
    #[inline]
    pub fn has_overflow(&self) -> bool {
        Self::has_nan_or_inf(&self.lower) || Self::has_nan_or_inf(&self.upper)
    }

    /// Shape of the tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.lower.shape()
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.lower.ndim()
    }

    /// Total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.lower.len()
    }

    /// Check if tensor is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lower.is_empty()
    }

    /// Apply directed rounding for mathematically sound interval arithmetic.
    ///
    /// IEEE 754 floating-point operations round to nearest-even by default.
    /// For strict interval arithmetic soundness, lower bounds should round DOWN
    /// (toward -∞) and upper bounds should round UP (toward +∞).
    ///
    /// This method widens bounds by taking the next representable float toward
    /// -∞ for lower bounds and toward +∞ for upper bounds (at most 1 ULP per element).
    ///
    /// # Use Cases
    /// - Apply after critical propagation steps to ensure containment
    /// - Use for final verification bounds when strict soundness is required
    ///
    /// # Performance
    /// Adds ~1 ULP of looseness per application. For typical verification
    /// with 100+ layers, this accumulates to ~100 ULPs, which is negligible
    /// compared to relaxation approximation errors (typically 10^3 - 10^6 ULPs).
    #[inline]
    pub fn round_for_soundness(&self) -> Self {
        Self {
            lower: self.lower.mapv(next_down_f32),
            upper: self.upper.mapv(next_up_f32),
        }
    }

    /// Apply directed rounding in place.
    ///
    /// Modifies this tensor to widen bounds by 1 ULP for soundness.
    /// See [`Self::round_for_soundness`] for details.
    #[inline]
    pub fn round_for_soundness_inplace(&mut self) {
        self.lower.mapv_inplace(next_down_f32);
        self.upper.mapv_inplace(next_up_f32);
    }

    /// Get bounds for a specific element.
    #[inline]
    pub fn get(&self, index: &[usize]) -> Bound {
        Bound::new(self.lower[index], self.upper[index])
    }

    /// Compute the width (upper - lower) for each element.
    #[inline]
    pub fn width(&self) -> ArrayD<f32> {
        &self.upper - &self.lower
    }

    /// Maximum width across all elements.
    pub fn max_width(&self) -> f32 {
        self.width().iter().cloned().fold(0.0_f32, f32::max)
    }

    /// Check if any bounds have exploded.
    pub fn has_unbounded(&self) -> bool {
        self.lower.iter().any(|v| v.is_infinite()) || self.upper.iter().any(|v| v.is_infinite())
    }

    /// Compute the intersection of two bounded tensors.
    ///
    /// For each element, the intersection is [max(l1, l2), min(u1, u2)].
    /// This gives the tightest bounds that are valid for both inputs.
    ///
    /// # Panics
    /// Panics in debug mode if shapes don't match.
    pub fn intersection(&self, other: &Self) -> Self {
        debug_assert_eq!(
            self.shape(),
            other.shape(),
            "intersection requires same shapes"
        );

        // Element-wise max of lower bounds, min of upper bounds
        let lower = ndarray::Zip::from(&self.lower)
            .and(&other.lower)
            .map_collect(|&a, &b| a.max(b));

        let upper = ndarray::Zip::from(&self.upper)
            .and(&other.upper)
            .map_collect(|&a, &b| a.min(b));

        Self { lower, upper }
    }

    /// Reshape the tensor.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let current_shape = self.lower.shape();
        let current_size: usize = current_shape.iter().product();
        let target_size: usize = shape.iter().product();

        // Check if shapes are identical - no reshape needed
        if current_shape == shape {
            return Ok(Self {
                lower: self.lower.clone(),
                upper: self.upper.clone(),
            });
        }

        // Check element count before attempting reshape
        if current_size != target_size {
            return Err(GammaError::shape_mismatch(
                shape.to_vec(),
                current_shape.to_vec(),
            ));
        }

        // Make arrays contiguous before reshaping to avoid layout issues
        let lower_contiguous = if self.lower.is_standard_layout() {
            self.lower.clone()
        } else {
            self.lower.as_standard_layout().to_owned()
        };
        let upper_contiguous = if self.upper.is_standard_layout() {
            self.upper.clone()
        } else {
            self.upper.as_standard_layout().to_owned()
        };

        let lower = lower_contiguous
            .into_shape_with_order(IxDyn(shape))
            .map_err(|e| {
                // Debug: print detailed info when reshape fails
                eprintln!(
                    "BoundedTensor reshape failed: current={:?}, target={:?}, error={:?}",
                    current_shape, shape, e
                );
                GammaError::shape_mismatch(shape.to_vec(), current_shape.to_vec())
            })?;
        let upper = upper_contiguous
            .into_shape_with_order(IxDyn(shape))
            .map_err(|e| {
                eprintln!(
                    "BoundedTensor reshape failed (upper): current={:?}, target={:?}, error={:?}",
                    current_shape, shape, e
                );
                GammaError::shape_mismatch(shape.to_vec(), current_shape.to_vec())
            })?;
        Ok(Self { lower, upper })
    }

    /// Flatten the tensor.
    pub fn flatten(&self) -> Self {
        let n = self.len();
        Self {
            lower: self
                .lower
                .clone()
                .into_shape_with_order(IxDyn(&[n]))
                .unwrap(),
            upper: self
                .upper
                .clone()
                .into_shape_with_order(IxDyn(&[n]))
                .unwrap(),
        }
    }

    /// Compute the center point of the bounded tensor: (lower + upper) / 2.
    pub fn center(&self) -> ArrayD<f32> {
        (&self.lower + &self.upper) / 2.0
    }

    /// Extract a single slice along the specified axis.
    ///
    /// Returns a tensor with the specified axis removed (not kept as size-1).
    ///
    /// # Arguments
    /// * `axis` - The axis to slice along
    /// * `index` - The index to select
    ///
    /// # Example
    /// ```ignore
    /// // Input shape: [batch, seq, hidden] = [1, 512, 768]
    /// let pos_0 = tensor.slice_axis(1, 0)?;  // [1, 768]
    /// let pos_1 = tensor.slice_axis(1, 1)?;  // [1, 768]
    /// ```
    pub fn slice_axis(&self, axis: usize, index: usize) -> Result<BoundedTensor> {
        use ndarray::Axis;

        let shape = self.shape();
        if axis >= shape.len() {
            return Err(GammaError::InvalidSpec(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                shape.len()
            )));
        }
        if index >= shape[axis] {
            return Err(GammaError::InvalidSpec(format!(
                "Index {} out of bounds for axis {} with size {}",
                index, axis, shape[axis]
            )));
        }

        let lower = self.lower.index_axis(Axis(axis), index).to_owned();
        let upper = self.upper.index_axis(Axis(axis), index).to_owned();

        BoundedTensor::new(lower, upper)
    }

    /// Extract a range of slices along the specified axis.
    ///
    /// Returns a tensor with reduced size along the specified axis.
    ///
    /// # Arguments
    /// * `axis` - The axis to slice along
    /// * `start` - Starting index (inclusive)
    /// * `end` - Ending index (exclusive)
    ///
    /// # Example
    /// ```ignore
    /// // Input shape: [batch, seq, hidden] = [1, 512, 768]
    /// let first_half = tensor.slice_axis_range(1, 0, 256)?;  // [1, 256, 768]
    /// ```
    pub fn slice_axis_range(&self, axis: usize, start: usize, end: usize) -> Result<BoundedTensor> {
        use ndarray::{Axis, Slice};

        let shape = self.shape();
        if axis >= shape.len() {
            return Err(GammaError::InvalidSpec(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                shape.len()
            )));
        }
        if end > shape[axis] {
            return Err(GammaError::InvalidSpec(format!(
                "End index {} out of bounds for axis {} with size {}",
                end, axis, shape[axis]
            )));
        }
        if start >= end {
            return Err(GammaError::InvalidSpec(format!(
                "Invalid range: start {} >= end {}",
                start, end
            )));
        }

        // Use slice_axis which is cleaner than building SliceInfo manually
        let slice_spec = Slice::from(start..end);
        let lower = self
            .lower
            .slice_axis(Axis(axis), slice_spec)
            .as_standard_layout()
            .into_owned();
        let upper = self
            .upper
            .slice_axis(Axis(axis), slice_spec)
            .as_standard_layout()
            .into_owned();

        BoundedTensor::new(lower, upper)
    }

    /// Insert a size-1 axis at the specified position.
    ///
    /// # Arguments
    /// * `axis` - Where to insert the new axis
    ///
    /// # Example
    /// ```ignore
    /// // Input shape: [batch, hidden] = [1, 768]
    /// let expanded = tensor.expand_axis(1)?;  // [1, 1, 768]
    /// ```
    pub fn expand_axis(&self, axis: usize) -> Result<BoundedTensor> {
        let shape = self.shape();
        if axis > shape.len() {
            return Err(GammaError::InvalidSpec(format!(
                "Axis {} out of bounds for inserting into tensor with {} dimensions",
                axis,
                shape.len()
            )));
        }

        let lower = self.lower.clone().insert_axis(ndarray::Axis(axis));
        let upper = self.upper.clone().insert_axis(ndarray::Axis(axis));

        BoundedTensor::new(lower, upper)
    }

    /// Concatenate multiple bounded tensors along an axis.
    ///
    /// # Arguments
    /// * `tensors` - Slice of tensors to concatenate
    /// * `axis` - Axis along which to concatenate
    ///
    /// # Example
    /// ```ignore
    /// // Each position has shape [batch, hidden] = [1, 768]
    /// let pos_0 = tensor.slice_axis(1, 0)?;
    /// let pos_1 = tensor.slice_axis(1, 1)?;
    /// // After expand_axis, each is [1, 1, 768]
    /// let combined = BoundedTensor::concat(&[pos_0.expand_axis(1)?, pos_1.expand_axis(1)?], 1)?;
    /// // Result: [1, 2, 768]
    /// ```
    pub fn concat(tensors: &[BoundedTensor], axis: usize) -> Result<BoundedTensor> {
        use ndarray::Axis;

        if tensors.is_empty() {
            return Err(GammaError::InvalidSpec(
                "Cannot concatenate empty tensor list".to_string(),
            ));
        }

        let first_shape = tensors[0].shape();
        if axis >= first_shape.len() {
            return Err(GammaError::InvalidSpec(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                first_shape.len()
            )));
        }

        // Validate shapes (all must match except along concat axis)
        for (i, t) in tensors.iter().enumerate().skip(1) {
            let shape = t.shape();
            if shape.len() != first_shape.len() {
                return Err(GammaError::shape_mismatch(
                    first_shape.to_vec(),
                    shape.to_vec(),
                ));
            }
            for (d, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if d != axis && s1 != s2 {
                    return Err(GammaError::InvalidSpec(format!(
                        "Shape mismatch at tensor {}: dimension {} is {} but expected {}",
                        i, d, s2, s1
                    )));
                }
            }
        }

        let lower_views: Vec<_> = tensors.iter().map(|t| t.lower.view()).collect();
        let upper_views: Vec<_> = tensors.iter().map(|t| t.upper.view()).collect();

        let lower = ndarray::concatenate(Axis(axis), &lower_views)
            .map_err(|e| GammaError::InvalidSpec(format!("Concatenation failed: {}", e)))?;
        let upper = ndarray::concatenate(Axis(axis), &upper_views)
            .map_err(|e| GammaError::InvalidSpec(format!("Concatenation failed: {}", e)))?;

        BoundedTensor::new(lower, upper)
    }

    /// Stack multiple bounded tensors along a new axis.
    ///
    /// Creates a new axis and stacks tensors along it.
    ///
    /// # Arguments
    /// * `tensors` - Slice of tensors to stack (all must have same shape)
    /// * `axis` - Where to insert the new stacking axis
    ///
    /// # Example
    /// ```ignore
    /// // Each position has shape [batch, hidden] = [1, 768]
    /// let positions = vec![pos_0, pos_1, pos_2];
    /// let stacked = BoundedTensor::stack(&positions, 1)?;  // [1, 3, 768]
    /// ```
    pub fn stack(tensors: &[BoundedTensor], axis: usize) -> Result<BoundedTensor> {
        use ndarray::Axis;

        if tensors.is_empty() {
            return Err(GammaError::InvalidSpec(
                "Cannot stack empty tensor list".to_string(),
            ));
        }

        let first_shape = tensors[0].shape();
        if axis > first_shape.len() {
            return Err(GammaError::InvalidSpec(format!(
                "Axis {} out of bounds for stacking into tensor with {} dimensions",
                axis,
                first_shape.len()
            )));
        }

        // Validate all shapes match exactly
        for t in tensors.iter().skip(1) {
            if t.shape() != first_shape {
                return Err(GammaError::shape_mismatch(
                    first_shape.to_vec(),
                    t.shape().to_vec(),
                ));
            }
        }

        let lower_views: Vec<_> = tensors.iter().map(|t| t.lower.view()).collect();
        let upper_views: Vec<_> = tensors.iter().map(|t| t.upper.view()).collect();

        let lower = ndarray::stack(Axis(axis), &lower_views)
            .map_err(|e| GammaError::InvalidSpec(format!("Stacking failed: {}", e)))?;
        let upper = ndarray::stack(Axis(axis), &upper_views)
            .map_err(|e| GammaError::InvalidSpec(format!("Stacking failed: {}", e)))?;

        BoundedTensor::new(lower, upper)
    }
}

/// Interval arithmetic operations on bounded tensors.
impl BoundedTensor {
    /// Element-wise addition of two bounded tensors.
    #[inline]
    pub fn add(&self, other: &BoundedTensor) -> Result<BoundedTensor> {
        if self.shape() != other.shape() {
            return Err(GammaError::shape_mismatch(
                self.shape().to_vec(),
                other.shape().to_vec(),
            ));
        }
        Ok(BoundedTensor {
            lower: &self.lower + &other.lower,
            upper: &self.upper + &other.upper,
        })
    }

    /// Element-wise multiplication (interval multiplication).
    ///
    /// Uses SIMD-accelerated interval arithmetic when arrays are contiguous.
    /// Interval multiplication: `[a,b] * [c,d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]`
    #[inline]
    pub fn mul(&self, other: &BoundedTensor) -> Result<BoundedTensor> {
        if self.shape() != other.shape() {
            return Err(GammaError::shape_mismatch(
                self.shape().to_vec(),
                other.shape().to_vec(),
            ));
        }

        let shape = self.shape().to_vec();
        let n = self.len();

        // Try to use SIMD path if arrays are contiguous
        let a_lower_slice = self.lower.as_slice();
        let a_upper_slice = self.upper.as_slice();
        let b_lower_slice = other.lower.as_slice();
        let b_upper_slice = other.upper.as_slice();

        if let (Some(al), Some(au), Some(bl), Some(bu)) =
            (a_lower_slice, a_upper_slice, b_lower_slice, b_upper_slice)
        {
            // SIMD fast path: contiguous memory
            let mut out_lower = vec![0.0f32; n];
            let mut out_upper = vec![0.0f32; n];

            simd::interval_mul(al, au, bl, bu, &mut out_lower, &mut out_upper);

            let lower = ArrayD::from_shape_vec(IxDyn(&shape), out_lower).unwrap();
            let upper = ArrayD::from_shape_vec(IxDyn(&shape), out_upper).unwrap();

            Ok(BoundedTensor { lower, upper })
        } else {
            // Fallback: non-contiguous arrays use ndarray operations
            let a = &self.lower;
            let b = &self.upper;
            let c = &other.lower;
            let d = &other.upper;

            let ac = a * c;
            let ad = a * d;
            let bc = b * c;
            let bd = b * d;

            let mut lower = ac.clone();
            let mut upper = ac.clone();

            ndarray::Zip::from(&mut lower)
                .and(&ac)
                .and(&ad)
                .and(&bc)
                .and(&bd)
                .for_each(|l, &ac, &ad, &bc, &bd| {
                    *l = ac.min(ad).min(bc).min(bd);
                });

            ndarray::Zip::from(&mut upper)
                .and(&ac)
                .and(&ad)
                .and(&bc)
                .and(&bd)
                .for_each(|u, &ac, &ad, &bc, &bd| {
                    *u = ac.max(ad).max(bc).max(bd);
                });

            Ok(BoundedTensor { lower, upper })
        }
    }

    /// Scalar multiplication.
    pub fn scale(&self, scalar: f32) -> BoundedTensor {
        if scalar >= 0.0 {
            BoundedTensor {
                lower: self.lower.mapv(|v| v * scalar),
                upper: self.upper.mapv(|v| v * scalar),
            }
        } else {
            // Negative scalar swaps bounds
            BoundedTensor {
                lower: self.upper.mapv(|v| v * scalar),
                upper: self.lower.mapv(|v| v * scalar),
            }
        }
    }

    /// Scalar addition.
    pub fn shift(&self, scalar: f32) -> BoundedTensor {
        BoundedTensor {
            lower: self.lower.mapv(|v| v + scalar),
            upper: self.upper.mapv(|v| v + scalar),
        }
    }

    /// General transpose with arbitrary permutation.
    ///
    /// # Arguments
    /// * `perm` - Permutation of dimensions. E.g., [0, 2, 1, 3] swaps dims 1 and 2.
    ///
    /// # Example
    /// ```ignore
    /// // [batch, seq, heads, dim] -> [batch, heads, seq, dim]
    /// let transposed = tensor.transpose(&[0, 2, 1, 3])?;
    /// ```
    pub fn transpose(&self, perm: &[usize]) -> Result<BoundedTensor> {
        let shape = self.shape();
        let ndim = shape.len();

        if perm.len() != ndim {
            return Err(GammaError::InvalidSpec(format!(
                "Permutation length {} doesn't match tensor ndim {}",
                perm.len(),
                ndim
            )));
        }

        // Validate permutation
        let mut sorted_perm = perm.to_vec();
        sorted_perm.sort();
        let expected: Vec<usize> = (0..ndim).collect();
        if sorted_perm != expected {
            return Err(GammaError::InvalidSpec(format!(
                "Invalid permutation {:?}, expected a permutation of 0..{}",
                perm, ndim
            )));
        }

        let lower = self.lower.clone().permuted_axes(IxDyn(perm));
        let upper = self.upper.clone().permuted_axes(IxDyn(perm));

        // Ensure contiguous memory layout
        let lower = lower.as_standard_layout().into_owned();
        let upper = upper.as_standard_layout().into_owned();

        BoundedTensor::new(lower, upper)
    }

    /// Transpose the last two dimensions.
    ///
    /// For a tensor of shape [..., M, N], returns shape [..., N, M].
    /// This is commonly used for attention: K^T = K.transpose_last_two()
    pub fn transpose_last_two(&self) -> Result<BoundedTensor> {
        let shape = self.shape();
        let ndim = shape.len();

        if ndim < 2 {
            return Err(GammaError::InvalidSpec(
                "Cannot transpose tensor with fewer than 2 dimensions".to_string(),
            ));
        }

        // Build the permutation: [..., ndim-1, ndim-2]
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.swap(ndim - 2, ndim - 1);

        let lower = self.lower.clone().permuted_axes(IxDyn(&perm));
        let upper = self.upper.clone().permuted_axes(IxDyn(&perm));

        // Ensure contiguous memory layout
        let lower = lower.as_standard_layout().into_owned();
        let upper = upper.as_standard_layout().into_owned();

        BoundedTensor::new(lower, upper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, arr3};

    #[test]
    fn test_concrete_tensor() {
        let values = arr1(&[1.0, 2.0, 3.0]).into_dyn();
        let t = BoundedTensor::concrete(values);
        assert_eq!(t.max_width(), 0.0);
    }

    #[test]
    fn test_epsilon_perturbation() {
        let values = arr1(&[0.0, 1.0]).into_dyn();
        let t = BoundedTensor::from_epsilon(values, 0.1);
        assert!((t.lower[[0]] - (-0.1)).abs() < 1e-6);
        assert!((t.upper[[0]] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_addition() {
        let a =
            BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[1.0, 2.0]).into_dyn()).unwrap();
        let b =
            BoundedTensor::new(arr1(&[0.5, 0.5]).into_dyn(), arr1(&[1.5, 1.5]).into_dyn()).unwrap();

        let c = a.add(&b).unwrap();
        assert_eq!(c.lower[[0]], 0.5);
        assert_eq!(c.upper[[0]], 2.5);
    }

    #[test]
    fn test_slice_axis_basic() {
        // Shape: [2, 3]
        let lower = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
        let upper = arr2(&[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]).into_dyn();
        let t = BoundedTensor::new(lower, upper).unwrap();

        // Slice along axis 0 (select row)
        let row0 = t.slice_axis(0, 0).unwrap();
        assert_eq!(row0.shape(), &[3]);
        assert_eq!(row0.lower[[0]], 1.0);
        assert_eq!(row0.lower[[2]], 3.0);

        let row1 = t.slice_axis(0, 1).unwrap();
        assert_eq!(row1.shape(), &[3]);
        assert_eq!(row1.lower[[0]], 4.0);

        // Slice along axis 1 (select column)
        let col1 = t.slice_axis(1, 1).unwrap();
        assert_eq!(col1.shape(), &[2]);
        assert_eq!(col1.lower[[0]], 2.0);
        assert_eq!(col1.lower[[1]], 5.0);
    }

    #[test]
    fn test_slice_axis_3d() {
        // Shape: [batch=1, seq=4, hidden=3]
        let lower = arr3(&[[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]])
        .into_dyn();
        let upper = lower.mapv(|x| x + 0.5);
        let t = BoundedTensor::new(lower, upper).unwrap();

        // Slice position 2 along seq axis (axis=1)
        let pos2 = t.slice_axis(1, 2).unwrap();
        assert_eq!(pos2.shape(), &[1, 3]); // [batch, hidden]
        assert_eq!(pos2.lower[[0, 0]], 7.0);
        assert_eq!(pos2.lower[[0, 2]], 9.0);
    }

    #[test]
    fn test_slice_axis_range() {
        // Shape: [batch=1, seq=4, hidden=3]
        let lower = arr3(&[[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]])
        .into_dyn();
        let upper = lower.mapv(|x| x + 0.5);
        let t = BoundedTensor::new(lower, upper).unwrap();

        // Slice range [1, 3) along seq axis
        let mid = t.slice_axis_range(1, 1, 3).unwrap();
        assert_eq!(mid.shape(), &[1, 2, 3]); // [batch, seq=2, hidden]
        assert_eq!(mid.lower[[0, 0, 0]], 4.0); // Was position 1
        assert_eq!(mid.lower[[0, 1, 0]], 7.0); // Was position 2
    }

    #[test]
    fn test_expand_axis() {
        // Shape: [3]
        let lower = arr1(&[1.0, 2.0, 3.0]).into_dyn();
        let upper = arr1(&[1.5, 2.5, 3.5]).into_dyn();
        let t = BoundedTensor::new(lower, upper).unwrap();

        // Expand at axis 0
        let expanded = t.expand_axis(0).unwrap();
        assert_eq!(expanded.shape(), &[1, 3]);
        assert_eq!(expanded.lower[[0, 1]], 2.0);

        // Expand at axis 1
        let expanded2 = t.expand_axis(1).unwrap();
        assert_eq!(expanded2.shape(), &[3, 1]);
        assert_eq!(expanded2.lower[[1, 0]], 2.0);
    }

    #[test]
    fn test_stack_positions() {
        // Simulate stacking position outputs back together
        // Each position has shape [batch=1, hidden=3]
        let pos0 = BoundedTensor::new(
            arr2(&[[1.0, 2.0, 3.0]]).into_dyn(),
            arr2(&[[1.5, 2.5, 3.5]]).into_dyn(),
        )
        .unwrap();

        let pos1 = BoundedTensor::new(
            arr2(&[[4.0, 5.0, 6.0]]).into_dyn(),
            arr2(&[[4.5, 5.5, 6.5]]).into_dyn(),
        )
        .unwrap();

        let pos2 = BoundedTensor::new(
            arr2(&[[7.0, 8.0, 9.0]]).into_dyn(),
            arr2(&[[7.5, 8.5, 9.5]]).into_dyn(),
        )
        .unwrap();

        // Stack along axis 1 to get [batch=1, seq=3, hidden=3]
        let stacked = BoundedTensor::stack(&[pos0, pos1, pos2], 1).unwrap();
        assert_eq!(stacked.shape(), &[1, 3, 3]);
        assert_eq!(stacked.lower[[0, 0, 0]], 1.0);
        assert_eq!(stacked.lower[[0, 1, 0]], 4.0);
        assert_eq!(stacked.lower[[0, 2, 0]], 7.0);
    }

    #[test]
    fn test_concat_positions() {
        // Simulate concatenating position outputs with expand_axis
        // Each position has shape [batch=1, hidden=3]
        let pos0 = BoundedTensor::new(
            arr2(&[[1.0, 2.0, 3.0]]).into_dyn(),
            arr2(&[[1.5, 2.5, 3.5]]).into_dyn(),
        )
        .unwrap();

        let pos1 = BoundedTensor::new(
            arr2(&[[4.0, 5.0, 6.0]]).into_dyn(),
            arr2(&[[4.5, 5.5, 6.5]]).into_dyn(),
        )
        .unwrap();

        // First expand to [batch=1, seq=1, hidden=3]
        let pos0_exp = pos0.expand_axis(1).unwrap();
        let pos1_exp = pos1.expand_axis(1).unwrap();

        // Then concat along axis 1
        let combined = BoundedTensor::concat(&[pos0_exp, pos1_exp], 1).unwrap();
        assert_eq!(combined.shape(), &[1, 2, 3]);
        assert_eq!(combined.lower[[0, 0, 0]], 1.0);
        assert_eq!(combined.lower[[0, 1, 0]], 4.0);
    }

    #[test]
    fn test_slice_and_stack_roundtrip() {
        // Start with [batch=1, seq=4, hidden=3]
        let lower = arr3(&[[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]])
        .into_dyn();
        let upper = lower.mapv(|x| x + 0.5);
        let original = BoundedTensor::new(lower, upper).unwrap();

        // Slice into individual positions
        let positions: Vec<_> = (0..4).map(|i| original.slice_axis(1, i).unwrap()).collect();

        // Stack them back together
        let reconstructed = BoundedTensor::stack(&positions, 1).unwrap();

        // Should be identical
        assert_eq!(reconstructed.shape(), original.shape());
        assert!(reconstructed
            .lower
            .iter()
            .zip(original.lower.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn test_slice_axis_errors() {
        let t = BoundedTensor::concrete(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());

        // Axis out of bounds
        assert!(t.slice_axis(5, 0).is_err());

        // Index out of bounds
        assert!(t.slice_axis(0, 10).is_err());
    }

    // === NaN/Inf Detection Tests ===

    #[test]
    fn test_has_nan_or_inf_normal_values() {
        let arr = arr1(&[1.0, 2.0, 3.0, -1.0, 0.0]).into_dyn();
        assert!(!BoundedTensor::has_nan_or_inf(&arr));
    }

    #[test]
    fn test_has_nan_or_inf_with_nan() {
        let arr = arr1(&[1.0, f32::NAN, 3.0]).into_dyn();
        assert!(BoundedTensor::has_nan_or_inf(&arr));
    }

    #[test]
    fn test_has_nan_or_inf_with_pos_inf() {
        let arr = arr1(&[1.0, f32::INFINITY, 3.0]).into_dyn();
        assert!(BoundedTensor::has_nan_or_inf(&arr));
    }

    #[test]
    fn test_has_nan_or_inf_with_neg_inf() {
        let arr = arr1(&[1.0, f32::NEG_INFINITY, 3.0]).into_dyn();
        assert!(BoundedTensor::has_nan_or_inf(&arr));
    }

    // === Edge Case Tests (f32::MAX, denormals, negative zero) ===

    #[test]
    fn test_f32_max_bounds() {
        // f32::MAX is a valid bound value (not infinity)
        let lower = arr1(&[f32::MIN, -f32::MAX, 0.0]).into_dyn();
        let upper = arr1(&[0.0, 0.0, f32::MAX]).into_dyn();
        let t = BoundedTensor::new(lower, upper).unwrap();
        assert_eq!(t.upper[[2]], f32::MAX);
        assert_eq!(t.lower[[0]], f32::MIN);
    }

    #[test]
    fn test_denormal_values() {
        // Denormal (subnormal) numbers are valid - smallest positive f32
        let tiny = f32::MIN_POSITIVE / 2.0; // Subnormal value
        let lower = arr1(&[-tiny, 0.0]).into_dyn();
        let upper = arr1(&[0.0, tiny]).into_dyn();
        let t = BoundedTensor::new(lower, upper).unwrap();
        assert!(t.upper[[1]] > 0.0 && t.upper[[1]] < f32::MIN_POSITIVE);
    }

    #[test]
    fn test_negative_zero() {
        // -0.0 is valid and equals 0.0 for comparison
        let lower = arr1(&[-0.0, -1.0]).into_dyn();
        let upper = arr1(&[0.0, 1.0]).into_dyn();
        let t = BoundedTensor::new(lower, upper).unwrap();
        // -0.0 == 0.0 in IEEE 754
        assert_eq!(t.lower[[0]], 0.0);
        assert_eq!(t.lower[[0]], -0.0);
    }

    #[test]
    fn test_very_small_epsilon() {
        // Very small but non-denormal epsilon
        let values = arr1(&[0.0, 1.0, -1.0]).into_dyn();
        let epsilon = f32::EPSILON;
        let t = BoundedTensor::from_epsilon(values, epsilon);
        assert!((t.lower[[0]] - (-f32::EPSILON)).abs() < 1e-10);
        assert!((t.upper[[0]] - f32::EPSILON).abs() < 1e-10);
    }

    #[test]
    fn test_zero_width_bounds() {
        // Zero-width bounds (concrete values)
        let lower = arr1(&[1.0, 2.0, 3.0]).into_dyn();
        let upper = arr1(&[1.0, 2.0, 3.0]).into_dyn();
        let t = BoundedTensor::new(lower, upper).unwrap();
        assert_eq!(t.max_width(), 0.0);
    }

    #[test]
    fn test_large_width_bounds() {
        // Wide bounds approaching but not reaching infinity
        let lower = arr1(&[-f32::MAX / 2.0]).into_dyn();
        let upper = arr1(&[f32::MAX / 2.0]).into_dyn();
        let t = BoundedTensor::new(lower, upper).unwrap();
        assert!(t.max_width().is_finite());
    }

    // === Directed Rounding Tests ===

    #[test]
    fn test_round_for_soundness_widens_bounds() {
        // Directed rounding should widen bounds by 1 ULP
        let lower = arr1(&[1.0_f32, 0.1, -1.0]).into_dyn();
        let upper = arr1(&[1.0_f32, 0.1, -1.0]).into_dyn();
        let t = BoundedTensor::new(lower, upper).unwrap();

        let rounded = t.round_for_soundness();

        // Each lower should be < original, each upper should be > original
        for i in 0..3 {
            assert!(
                rounded.lower[[i]] < t.lower[[i]],
                "Lower bound should decrease at index {}: {} < {}",
                i,
                rounded.lower[[i]],
                t.lower[[i]]
            );
            assert!(
                rounded.upper[[i]] > t.upper[[i]],
                "Upper bound should increase at index {}: {} > {}",
                i,
                rounded.upper[[i]],
                t.upper[[i]]
            );
        }

        // Width should increase by exactly 2 ULPs (1 down + 1 up)
        assert!(rounded.max_width() > 0.0);
    }

    #[test]
    fn test_round_for_soundness_1ulp_difference() {
        // Test that bounds widen by exactly 1 ULP
        let val = 1.0_f32;
        let lower = arr1(&[val]).into_dyn();
        let upper = arr1(&[val]).into_dyn();
        let t = BoundedTensor::new(lower, upper).unwrap();

        let rounded = t.round_for_soundness();

        // next_down(1.0) and next_up(1.0) should differ by exactly 1 ULP each
        assert_eq!(rounded.lower[[0]], next_down_f32(1.0));
        assert_eq!(rounded.upper[[0]], next_up_f32(1.0));
        assert_eq!(rounded.upper[[0]], 1.0 + f32::EPSILON);
    }

    #[test]
    fn test_round_for_soundness_preserves_infinity() {
        // Infinity should stay infinity (no ULP beyond infinity)
        let lower = arr1(&[f32::NEG_INFINITY]).into_dyn();
        let upper = arr1(&[f32::INFINITY]).into_dyn();
        let t = BoundedTensor::new_unchecked(lower, upper).unwrap();

        let rounded = t.round_for_soundness();

        assert_eq!(rounded.lower[[0]], f32::NEG_INFINITY);
        assert_eq!(rounded.upper[[0]], f32::INFINITY);
    }

    #[test]
    fn test_round_for_soundness_inplace() {
        // Test in-place rounding
        let lower = arr1(&[1.0_f32]).into_dyn();
        let upper = arr1(&[2.0_f32]).into_dyn();
        let mut t = BoundedTensor::new(lower, upper).unwrap();

        t.round_for_soundness_inplace();

        assert!(t.lower[[0]] < 1.0);
        assert!(t.upper[[0]] > 2.0);
    }
}

/// Property-based soundness tests using proptest
#[cfg(test)]
mod proptest_soundness {
    use super::*;
    use ndarray::{arr1, arr2, Array, Array2};
    use proptest::prelude::*;

    /// Strategy to generate valid interval bounds [lower, upper] where lower <= upper
    fn valid_interval() -> impl Strategy<Value = (f32, f32)> {
        // Use ranges that avoid infinity and extreme values
        (-1000.0f32..1000.0f32)
            .prop_flat_map(|a| (-1000.0f32..1000.0f32).prop_map(move |b| (a.min(b), a.max(b))))
    }

    /// Sample points within an interval for verification.
    /// Uses clamping to ensure FP rounding doesn't produce out-of-bounds samples.
    fn sample_points(lower: f32, upper: f32, num_samples: usize) -> Vec<f32> {
        if lower == upper {
            return vec![lower];
        }
        (0..=num_samples)
            .map(|i| {
                let t = i as f32 / num_samples as f32;
                let sample = lower + (upper - lower) * t;
                // Clamp to handle FP rounding that could exceed bounds
                sample.clamp(lower, upper)
            })
            .collect()
    }

    proptest! {
        /// Interval addition soundness: [a,b] + [c,d] must contain a+c and b+d
        #[test]
        fn soundness_interval_add(
            (a, b) in valid_interval(),
            (c, d) in valid_interval()
        ) {
            let t1 = BoundedTensor::new(
                arr1(&[a]).into_dyn(),
                arr1(&[b]).into_dyn()
            ).unwrap();
            let t2 = BoundedTensor::new(
                arr1(&[c]).into_dyn(),
                arr1(&[d]).into_dyn()
            ).unwrap();

            let result = t1.add(&t2).unwrap();

            // Test that for any x1 in [a,b] and x2 in [c,d], x1+x2 is in result bounds
            for x1 in sample_points(a, b, 10) {
                for x2 in sample_points(c, d, 10) {
                    let sum = x1 + x2;
                    prop_assert!(
                        result.lower[[0]] <= sum && sum <= result.upper[[0]],
                        "Addition unsound: {}+{}={} not in [{}, {}]",
                        x1, x2, sum, result.lower[[0]], result.upper[[0]]
                    );
                }
            }
        }

        /// Interval multiplication soundness:
        /// [a,b] * [c,d] must contain all products x*y for x in [a,b], y in [c,d]
        #[test]
        fn soundness_interval_mul(
            (a, b) in valid_interval(),
            (c, d) in valid_interval()
        ) {
            let t1 = BoundedTensor::new(
                arr1(&[a]).into_dyn(),
                arr1(&[b]).into_dyn()
            ).unwrap();
            let t2 = BoundedTensor::new(
                arr1(&[c]).into_dyn(),
                arr1(&[d]).into_dyn()
            ).unwrap();

            let result = t1.mul(&t2).unwrap();

            // Test that for any x1 in [a,b] and x2 in [c,d], x1*x2 is in result bounds
            for x1 in sample_points(a, b, 10) {
                for x2 in sample_points(c, d, 10) {
                    let product = x1 * x2;
                    prop_assert!(
                        result.lower[[0]] <= product && product <= result.upper[[0]],
                        "Multiplication unsound: {}*{}={} not in [{}, {}]",
                        x1, x2, product, result.lower[[0]], result.upper[[0]]
                    );
                }
            }
        }

        /// Scalar multiplication soundness: s * [a,b] must contain s*x for all x in [a,b]
        #[test]
        fn soundness_scalar_mul(
            (a, b) in valid_interval(),
            s in -100.0f32..100.0f32
        ) {
            let t = BoundedTensor::new(
                arr1(&[a]).into_dyn(),
                arr1(&[b]).into_dyn()
            ).unwrap();

            let result = t.scale(s);

            // Test that for any x in [a,b], s*x is in result bounds
            for x in sample_points(a, b, 20) {
                let scaled = s * x;
                prop_assert!(
                    result.lower[[0]] <= scaled && scaled <= result.upper[[0]],
                    "Scalar mul unsound: {}*{}={} not in [{}, {}]",
                    s, x, scaled, result.lower[[0]], result.upper[[0]]
                );
            }
        }

        /// Scalar addition soundness: [a,b] + s must contain x+s for all x in [a,b]
        #[test]
        fn soundness_scalar_add(
            (a, b) in valid_interval(),
            s in -100.0f32..100.0f32
        ) {
            let t = BoundedTensor::new(
                arr1(&[a]).into_dyn(),
                arr1(&[b]).into_dyn()
            ).unwrap();

            let result = t.shift(s);

            // Test that for any x in [a,b], x+s is in result bounds
            for x in sample_points(a, b, 20) {
                let shifted = x + s;
                prop_assert!(
                    result.lower[[0]] <= shifted && shifted <= result.upper[[0]],
                    "Scalar add unsound: {}+{}={} not in [{}, {}]",
                    x, s, shifted, result.lower[[0]], result.upper[[0]]
                );
            }
        }

        /// Directed rounding must always widen bounds (for finite values)
        #[test]
        fn soundness_directed_rounding(
            (a, b) in valid_interval()
        ) {
            let t = BoundedTensor::new(
                arr1(&[a]).into_dyn(),
                arr1(&[b]).into_dyn()
            ).unwrap();

            let rounded = t.round_for_soundness();

            // Rounded bounds must be wider or equal (contain original)
            prop_assert!(
                rounded.lower[[0]] <= t.lower[[0]],
                "Directed rounding should decrease lower: {} > {}",
                rounded.lower[[0]], t.lower[[0]]
            );
            prop_assert!(
                rounded.upper[[0]] >= t.upper[[0]],
                "Directed rounding should increase upper: {} < {}",
                rounded.upper[[0]], t.upper[[0]]
            );
        }

        /// Bounds should remain valid (lower <= upper) after operations
        #[test]
        fn validity_bounds_ordering(
            (a, b) in valid_interval(),
            (c, d) in valid_interval(),
            s in -100.0f32..100.0f32
        ) {
            let t1 = BoundedTensor::new(
                arr1(&[a]).into_dyn(),
                arr1(&[b]).into_dyn()
            ).unwrap();
            let t2 = BoundedTensor::new(
                arr1(&[c]).into_dyn(),
                arr1(&[d]).into_dyn()
            ).unwrap();

            // After add
            let added = t1.add(&t2).unwrap();
            prop_assert!(added.lower[[0]] <= added.upper[[0]], "Add produced inverted bounds");

            // After mul
            let multed = t1.mul(&t2).unwrap();
            prop_assert!(multed.lower[[0]] <= multed.upper[[0]], "Mul produced inverted bounds");

            // After scale
            let scaled = t1.scale(s);
            prop_assert!(scaled.lower[[0]] <= scaled.upper[[0]], "Scale produced inverted bounds");

            // After shift
            let shifted = t1.shift(s);
            prop_assert!(shifted.lower[[0]] <= shifted.upper[[0]], "Shift produced inverted bounds");

            // After rounding
            let rounded = t1.round_for_soundness();
            prop_assert!(rounded.lower[[0]] <= rounded.upper[[0]], "Rounding produced inverted bounds");
        }
    }

    #[test]
    fn test_new_sanitized_with_nan() {
        let lower = arr1(&[1.0, f32::NAN, -1.0]).into_dyn();
        let upper = arr1(&[2.0, f32::NAN, 0.0]).into_dyn();
        let clamp_val = 100.0;

        let t = BoundedTensor::new_sanitized(lower, upper, clamp_val).unwrap();

        // NaN in lower should become -clamp_val
        assert_eq!(t.lower[[1]], -clamp_val);
        // NaN in upper should become +clamp_val
        assert_eq!(t.upper[[1]], clamp_val);
        // Finite values should be unchanged
        assert_eq!(t.lower[[0]], 1.0);
        assert_eq!(t.upper[[0]], 2.0);
    }

    #[test]
    fn test_new_sanitized_with_infinity() {
        let lower = arr1(&[f32::NEG_INFINITY, 0.0, f32::INFINITY]).into_dyn();
        let upper = arr1(&[0.0, f32::INFINITY, f32::INFINITY]).into_dyn();
        let clamp_val = 1e10;

        let t = BoundedTensor::new_sanitized(lower, upper, clamp_val).unwrap();

        // -Inf should become -clamp_val
        assert_eq!(t.lower[[0]], -clamp_val);
        // +Inf should become +clamp_val
        assert_eq!(t.upper[[1]], clamp_val);
        // +Inf in lower should become +clamp_val (then possibly swapped)
        assert!(t.lower[[2]] <= t.upper[[2]]);
    }

    #[test]
    fn test_new_sanitized_ensures_ordering() {
        // Simulate case where sanitization might invert bounds
        // (e.g., NaN replaced differently in lower vs upper)
        let lower = arr1(&[f32::NAN]).into_dyn();
        let upper = arr1(&[f32::NAN]).into_dyn();
        let clamp_val = 100.0;

        let t = BoundedTensor::new_sanitized(lower, upper, clamp_val).unwrap();

        // Even after NaN replacement, lower <= upper must hold
        assert!(t.lower[[0]] <= t.upper[[0]]);
    }

    #[test]
    fn test_new_sanitized_preserves_shape() {
        let lower = arr2(&[[1.0, 2.0], [3.0, f32::NAN]]).into_dyn();
        let upper = arr2(&[[2.0, 3.0], [4.0, f32::INFINITY]]).into_dyn();
        let clamp_val = 100.0;

        let t = BoundedTensor::new_sanitized(lower.clone(), upper.clone(), clamp_val).unwrap();

        assert_eq!(t.shape(), lower.shape());
    }

    #[test]
    fn test_sanitize_method() {
        let lower = arr1(&[f32::NAN, 1.0, f32::NEG_INFINITY]).into_dyn();
        let upper = arr1(&[f32::INFINITY, 2.0, f32::NAN]).into_dyn();

        // Use new_unchecked to create tensor with NaN/Inf (bypasses assertions)
        let t = BoundedTensor::new_unchecked(lower, upper).unwrap();

        assert!(t.has_overflow());

        let sanitized = t.sanitize(1000.0);

        assert!(!sanitized.has_overflow());
        assert!(sanitized.lower[[0]] <= sanitized.upper[[0]]);
        assert!(sanitized.lower[[1]] <= sanitized.upper[[1]]);
        assert!(sanitized.lower[[2]] <= sanitized.upper[[2]]);
    }

    #[test]
    fn test_has_overflow() {
        // Normal tensor should not have overflow
        let normal =
            BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn()).unwrap();
        assert!(!normal.has_overflow());

        // Tensor with Inf should have overflow
        let with_inf = BoundedTensor::new_unchecked(
            arr1(&[1.0, f32::NEG_INFINITY]).into_dyn(),
            arr1(&[3.0, f32::INFINITY]).into_dyn(),
        )
        .unwrap();
        assert!(with_inf.has_overflow());

        // Tensor with NaN should have overflow
        let with_nan = BoundedTensor::new_unchecked(
            arr1(&[f32::NAN, 2.0]).into_dyn(),
            arr1(&[3.0, 4.0]).into_dyn(),
        )
        .unwrap();
        assert!(with_nan.has_overflow());
    }

    // ========================================
    // Mutation-killing tests for BoundedTensor
    // ========================================

    #[test]
    fn test_ndim_exact_values() {
        // Test that ndim returns exact dimension count, not 0 or 1
        let t1d = BoundedTensor::concrete(arr1(&[1.0, 2.0, 3.0]).into_dyn());
        assert_eq!(t1d.ndim(), 1);

        let t2d = BoundedTensor::concrete(
            Array2::from_shape_vec((2, 3), vec![1.0; 6])
                .unwrap()
                .into_dyn(),
        );
        assert_eq!(t2d.ndim(), 2);

        let t3d = BoundedTensor::concrete(
            Array::from_shape_vec(IxDyn(&[2, 3, 4]), vec![1.0; 24]).unwrap(),
        );
        assert_eq!(t3d.ndim(), 3);

        let t4d = BoundedTensor::concrete(
            Array::from_shape_vec(IxDyn(&[2, 3, 4, 5]), vec![1.0; 120]).unwrap(),
        );
        assert_eq!(t4d.ndim(), 4);
    }

    #[test]
    fn test_len_exact_values() {
        // Test that len returns exact element count, not 1
        let t1 = BoundedTensor::concrete(arr1(&[1.0]).into_dyn());
        assert_eq!(t1.len(), 1);

        let t6 = BoundedTensor::concrete(arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).into_dyn());
        assert_eq!(t6.len(), 6);

        let t12 = BoundedTensor::concrete(
            Array2::from_shape_vec((3, 4), vec![1.0; 12])
                .unwrap()
                .into_dyn(),
        );
        assert_eq!(t12.len(), 12);
    }

    #[test]
    fn test_is_empty_exact() {
        // Non-empty tensor should return false, not true
        let non_empty = BoundedTensor::concrete(arr1(&[1.0]).into_dyn());
        assert!(!non_empty.is_empty());

        // Empty tensor should return true
        let empty = BoundedTensor::concrete(Array::from_shape_vec(IxDyn(&[0]), vec![]).unwrap());
        assert!(empty.is_empty());
    }

    #[test]
    fn test_has_unbounded_distinguish_bounds() {
        // Test all combinations to distinguish || vs && and false vs true

        // Neither bound infinite - should return false
        let finite = BoundedTensor::new_unchecked(
            arr1(&[1.0, 2.0]).into_dyn(),
            arr1(&[3.0, 4.0]).into_dyn(),
        )
        .unwrap();
        assert!(!finite.has_unbounded());

        // Only lower bound infinite - should return true (|| catches this)
        let lower_inf = BoundedTensor::new_unchecked(
            arr1(&[f32::NEG_INFINITY, 2.0]).into_dyn(),
            arr1(&[3.0, 4.0]).into_dyn(),
        )
        .unwrap();
        assert!(lower_inf.has_unbounded());

        // Only upper bound infinite - should return true (|| catches this)
        let upper_inf = BoundedTensor::new_unchecked(
            arr1(&[1.0, 2.0]).into_dyn(),
            arr1(&[3.0, f32::INFINITY]).into_dyn(),
        )
        .unwrap();
        assert!(upper_inf.has_unbounded());

        // Both bounds infinite - should return true
        let both_inf = BoundedTensor::new_unchecked(
            arr1(&[f32::NEG_INFINITY]).into_dyn(),
            arr1(&[f32::INFINITY]).into_dyn(),
        )
        .unwrap();
        assert!(both_inf.has_unbounded());
    }

    #[test]
    fn test_center_exact_computation() {
        // Test that center is exactly (lower + upper) / 2
        // Using values where mutations would give different results
        let t = BoundedTensor::new(
            arr1(&[0.0, 2.0, 10.0]).into_dyn(),
            arr1(&[4.0, 6.0, 20.0]).into_dyn(),
        )
        .unwrap();
        let center = t.center();

        // (0+4)/2 = 2, (2+6)/2 = 4, (10+20)/2 = 15
        assert_eq!(center[[0]], 2.0); // If + replaced with -, would get -2.0
        assert_eq!(center[[1]], 4.0); // If + replaced with *, would get 12.0
        assert_eq!(center[[2]], 15.0); // If / replaced with *, would get 30.0
    }

    #[test]
    fn test_mul_exact_interval_bounds() {
        // Test element-wise multiplication computes correct interval bounds
        // For intervals [a,b] * [c,d], result is [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]

        // Positive * Positive: [2,3] * [4,5] = [8, 15]
        let a = BoundedTensor::new(arr1(&[2.0]).into_dyn(), arr1(&[3.0]).into_dyn()).unwrap();
        let b = BoundedTensor::new(arr1(&[4.0]).into_dyn(), arr1(&[5.0]).into_dyn()).unwrap();
        let result = a.mul(&b).unwrap();
        assert_eq!(result.lower[[0]], 8.0); // min(8, 10, 12, 15) = 8
        assert_eq!(result.upper[[0]], 15.0); // max(8, 10, 12, 15) = 15

        // Mixed signs: [-1,2] * [-3,4] = [-8, 8]
        // ac=-1*-3=3, ad=-1*4=-4, bc=2*-3=-6, bd=2*4=8
        let c = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();
        let d = BoundedTensor::new(arr1(&[-3.0]).into_dyn(), arr1(&[4.0]).into_dyn()).unwrap();
        let result2 = c.mul(&d).unwrap();
        assert_eq!(result2.lower[[0]], -6.0); // min(3, -4, -6, 8) = -6
        assert_eq!(result2.upper[[0]], 8.0); // max(3, -4, -6, 8) = 8
    }

    #[test]
    fn test_new_sanitized_boundary_conditions() {
        // Test that > vs >= distinctions matter in new_sanitized
        // The function clamps values using: if x > 0.0 / if x > 0.0

        // Exactly 0.0 should be treated differently from > 0.0
        let clamp = 10.0;
        let result = BoundedTensor::new_sanitized(
            arr1(&[f32::NEG_INFINITY, 0.0, f32::INFINITY]).into_dyn(),
            arr1(&[f32::NEG_INFINITY, 0.0, f32::INFINITY]).into_dyn(),
            clamp,
        )
        .unwrap();

        // NEG_INFINITY on lower should become -clamp_val
        assert_eq!(result.lower[[0]], -clamp);
        // INFINITY on upper should become clamp_val
        assert_eq!(result.upper[[2]], clamp);
        // 0.0 should remain 0.0 (clamped to valid range)
        assert_eq!(result.lower[[1]], 0.0);
        assert_eq!(result.upper[[1]], 0.0);
    }

    #[test]
    fn test_slice_axis_range_boundary() {
        // Test boundary condition in slice_axis_range: end > dim[axis]
        let t = BoundedTensor::concrete(
            Array::from_shape_vec(IxDyn(&[2, 5, 3]), (0..30).map(|x| x as f32).collect()).unwrap(),
        );

        // Valid slice
        let slice = t.slice_axis_range(1, 1, 4).unwrap();
        assert_eq!(slice.shape(), &[2, 3, 3]);

        // End == dim should work (boundary case)
        let slice2 = t.slice_axis_range(1, 0, 5).unwrap();
        assert_eq!(slice2.shape(), &[2, 5, 3]);

        // End > dim should fail
        let err = t.slice_axis_range(1, 0, 6);
        assert!(err.is_err());
    }

    #[test]
    fn test_concat_shape_validation() {
        // Test that concat validates shapes correctly (!= vs ==)
        let t1 = BoundedTensor::concrete(Array2::zeros((2, 3)).into_dyn());
        let t2 = BoundedTensor::concrete(Array2::zeros((2, 3)).into_dyn());
        let t3 = BoundedTensor::concrete(Array2::zeros((2, 4)).into_dyn()); // Different shape

        // Same shapes should concat fine
        let result = BoundedTensor::concat(&[t1.clone(), t2], 0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().shape(), &[4, 3]);

        // Different shapes should fail (except on concat axis)
        let err = BoundedTensor::concat(&[t1, t3], 0);
        assert!(err.is_err());
    }

    #[test]
    fn test_stack_ndim_boundary() {
        // Test that stack validates ndim > 0
        let t1 = BoundedTensor::concrete(arr1(&[1.0, 2.0]).into_dyn());
        let t2 = BoundedTensor::concrete(arr1(&[3.0, 4.0]).into_dyn());

        let result = BoundedTensor::stack(&[t1, t2], 0).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.ndim(), 2);
    }

    #[test]
    fn test_transpose_validation() {
        // Test transpose axes validation (shape[i] != shape[axes[i]])
        let t = BoundedTensor::concrete(
            Array::from_shape_vec(IxDyn(&[2, 3, 4]), vec![1.0; 24]).unwrap(),
        );

        // Valid transpose
        let result = t.transpose(&[2, 0, 1]).unwrap();
        assert_eq!(result.shape(), &[4, 2, 3]);

        // Wrong number of axes should fail
        let err = t.transpose(&[0, 1]);
        assert!(err.is_err());
    }

    #[test]
    fn test_transpose_last_two_minimum_dims() {
        // Test that transpose_last_two requires at least 2 dimensions
        let t1d = BoundedTensor::concrete(arr1(&[1.0, 2.0]).into_dyn());
        let err = t1d.transpose_last_two();
        assert!(err.is_err());

        let t2d = BoundedTensor::concrete(
            Array2::from_shape_vec((2, 3), vec![1.0; 6])
                .unwrap()
                .into_dyn(),
        );
        let result = t2d.transpose_last_two().unwrap();
        assert_eq!(result.shape(), &[3, 2]);
    }

    // ========== Mutation-killing tests for lib.rs ==========

    #[test]
    fn test_new_sanitized_positive_infinity_exact_values() {
        // Target: lines 214, 224 - `if x > 0.0` mutations
        // Tests that +INFINITY becomes +clamp_val (not -clamp_val)
        let clamp = 1e6;

        // Positive infinity in LOWER bound should become +clamp_val
        let result = BoundedTensor::new_sanitized(
            arr1(&[f32::INFINITY]).into_dyn(),
            arr1(&[f32::INFINITY]).into_dyn(),
            clamp,
        )
        .unwrap();
        // If `> 0.0` mutated to `== 0.0`, INFINITY would give -clamp_val
        assert_eq!(
            result.lower[[0]],
            clamp,
            "INFINITY in lower must become +clamp_val"
        );
        assert_eq!(
            result.upper[[0]],
            clamp,
            "INFINITY in upper must become +clamp_val"
        );

        // Positive infinity in lower with finite upper
        let result2 = BoundedTensor::new_sanitized(
            arr1(&[f32::INFINITY]).into_dyn(),
            arr1(&[2e6]).into_dyn(), // Larger than clamp
            clamp,
        )
        .unwrap();
        // Lower should be clamped to +clamp_val, then may be swapped if needed
        // After sanitization, lower should be clamp (1e6), upper clamped to clamp (1e6)
        assert!(result2.lower[[0]] <= result2.upper[[0]]);
        assert_eq!(result2.lower[[0]], clamp);
    }

    #[test]
    fn test_new_sanitized_negative_infinity_upper() {
        // Target: line 224:49 - `delete -` in upper sanitization
        // Tests that -INFINITY in upper becomes -clamp_val
        let clamp = 500.0;

        let result = BoundedTensor::new_sanitized(
            arr1(&[f32::NEG_INFINITY]).into_dyn(),
            arr1(&[f32::NEG_INFINITY]).into_dyn(),
            clamp,
        )
        .unwrap();
        // Both should become -clamp_val
        // If `delete -` mutation, NEG_INFINITY would give +clamp_val (wrong)
        assert_eq!(
            result.lower[[0]],
            -clamp,
            "NEG_INFINITY in lower must become -clamp_val"
        );
        assert_eq!(
            result.upper[[0]],
            -clamp,
            "NEG_INFINITY in upper must become -clamp_val"
        );
    }

    #[test]
    fn test_new_sanitized_swap_only_when_inverted() {
        // Target: line 237 - `if *l > *u` swap condition
        // Ensure swap only happens when lower > upper (not when equal)
        let clamp = 100.0;

        // Case: lower == upper (should NOT swap, values should stay as-is)
        let result = BoundedTensor::new_sanitized(
            arr1(&[5.0, 10.0]).into_dyn(),
            arr1(&[5.0, 10.0]).into_dyn(),
            clamp,
        )
        .unwrap();
        assert_eq!(result.lower[[0]], 5.0);
        assert_eq!(result.upper[[0]], 5.0);
        assert_eq!(result.lower[[1]], 10.0);
        assert_eq!(result.upper[[1]], 10.0);

        // Case: lower > upper (should swap)
        let result2 =
            BoundedTensor::new_sanitized(arr1(&[20.0]).into_dyn(), arr1(&[10.0]).into_dyn(), clamp)
                .unwrap();
        assert_eq!(result2.lower[[0]], 10.0, "After swap, lower should be 10");
        assert_eq!(result2.upper[[0]], 20.0, "After swap, upper should be 20");

        // Case: lower < upper (should NOT swap)
        let result3 =
            BoundedTensor::new_sanitized(arr1(&[5.0]).into_dyn(), arr1(&[15.0]).into_dyn(), clamp)
                .unwrap();
        assert_eq!(result3.lower[[0]], 5.0);
        assert_eq!(result3.upper[[0]], 15.0);
    }

    #[test]
    fn test_concat_validates_all_non_concat_dimensions() {
        // Target: line 575 - `d != axis` vs `d == axis`
        // Ensure concat validates ALL dimensions except concat axis

        // Create tensors with matching first dim but different second dim
        let t1 = BoundedTensor::concrete(Array2::zeros((3, 4)).into_dyn());
        let t2 = BoundedTensor::concrete(Array2::zeros((3, 5)).into_dyn()); // dim 1 differs

        // Concat on axis 0: dimension 1 differs (4 vs 5), should fail
        let err = BoundedTensor::concat(&[t1.clone(), t2.clone()], 0);
        assert!(
            err.is_err(),
            "Concat should fail when non-concat dimension differs"
        );

        // But concat on axis 1 should work (dimension 0 matches)
        let result = BoundedTensor::concat(&[t1, t2], 1);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().shape(), &[3, 9]); // 4 + 5 = 9
    }

    #[test]
    fn test_stack_axis_at_ndim_boundary() {
        // Target: line 619 - `axis > first_shape.len()` vs `>=`
        // Ensure stack allows axis == ndim (insert at end)

        let t1 = BoundedTensor::concrete(arr1(&[1.0, 2.0]).into_dyn()); // shape [2]
        let t2 = BoundedTensor::concrete(arr1(&[3.0, 4.0]).into_dyn());

        // axis = 0: new axis at beginning -> [2, 2]
        let r0 = BoundedTensor::stack(&[t1.clone(), t2.clone()], 0).unwrap();
        assert_eq!(r0.shape(), &[2, 2]);

        // axis = 1 = ndim: new axis at end -> [2, 2]
        // With `> ndim`, this should work. With `>= ndim`, it would fail.
        let r1 = BoundedTensor::stack(&[t1.clone(), t2.clone()], 1).unwrap();
        assert_eq!(r1.shape(), &[2, 2]);

        // axis = 2 > ndim: should fail
        let err = BoundedTensor::stack(&[t1, t2], 2);
        assert!(err.is_err(), "Stack axis beyond ndim should fail");
    }

    #[test]
    fn test_mul_non_contiguous_arrays() {
        // Target: lines 708-711 - `* with +` or `* with /` in fallback mul path
        // Force non-contiguous arrays by using permuted_axes without as_standard_layout

        // Test case 1: ad (a*d) is the minimum, catches line 709 mutation
        // a=-2, b=1, c=2, d=3
        // Products: ac=-4, ad=-6, bc=2, bd=3
        // With *: min=-6, max=3
        // With + on ad: ad=-2+3=1, min would be -4 (DIFFERENT!)

        // Create non-contiguous arrays using permuted_axes on owned 2D arrays
        // permuted_axes on owned Array2 reorders axes but keeps same data buffer
        let arr1_lower =
            Array2::<f32>::from_shape_vec((2, 3), vec![-2.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let arr1_upper =
            Array2::<f32>::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        // Transpose: [2,3] -> [3,2], makes it non-contiguous
        let lower1_t = arr1_lower.permuted_axes([1, 0]).into_dyn();
        let upper1_t = arr1_upper.permuted_axes([1, 0]).into_dyn();

        // Verify these are actually non-contiguous (as_slice returns None)
        assert!(
            lower1_t.as_slice().is_none(),
            "Array should be non-contiguous after permute"
        );

        let t1 = BoundedTensor::new(lower1_t, upper1_t).unwrap();

        let arr2_lower =
            Array2::<f32>::from_shape_vec((2, 3), vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let arr2_upper =
            Array2::<f32>::from_shape_vec((2, 3), vec![3.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let lower2_t = arr2_lower.permuted_axes([1, 0]).into_dyn();
        let upper2_t = arr2_upper.permuted_axes([1, 0]).into_dyn();
        let t2 = BoundedTensor::new(lower2_t, upper2_t).unwrap();

        let result = t1.mul(&t2).unwrap();
        assert_eq!(result.lower[[0, 0]], -6.0, "min should be ad=-6 (a*d=-2*3)");
        assert_eq!(result.upper[[0, 0]], 3.0, "max should be bd=3 (b*d=1*3)");

        // Test case 2: bc (b*c) is the minimum, catches line 710 mutation
        // a=-1, b=2, c=-3, d=4
        // Products: ac=3, ad=-4, bc=-6, bd=8
        // With *: min=-6, max=8
        // With + on bc: bc=2+-3=-1, min would be -4 (DIFFERENT!)
        // Must use non-contiguous arrays to hit the fallback path (lines 702-731)
        let arr3_lower =
            Array2::<f32>::from_shape_vec((2, 3), vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let arr3_upper =
            Array2::<f32>::from_shape_vec((2, 3), vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let lower3_t = arr3_lower.permuted_axes([1, 0]).into_dyn();
        let upper3_t = arr3_upper.permuted_axes([1, 0]).into_dyn();
        let t3 = BoundedTensor::new(lower3_t, upper3_t).unwrap();

        let arr4_lower =
            Array2::<f32>::from_shape_vec((2, 3), vec![-3.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let arr4_upper =
            Array2::<f32>::from_shape_vec((2, 3), vec![4.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let lower4_t = arr4_lower.permuted_axes([1, 0]).into_dyn();
        let upper4_t = arr4_upper.permuted_axes([1, 0]).into_dyn();
        let t4 = BoundedTensor::new(lower4_t, upper4_t).unwrap();

        let result2 = t3.mul(&t4).unwrap();
        assert_eq!(
            result2.lower[[0, 0]],
            -6.0,
            "min should be bc=-6 (b*c=2*-3)"
        );
        assert_eq!(result2.upper[[0, 0]], 8.0, "max should be bd=8 (b*d=2*4)");

        // Test case 3: ac (a*c) is the maximum, catches line 708 mutation
        // a=-3, b=-1, c=-2, d=-1
        // Products: ac=6, ad=3, bc=2, bd=1
        // With *: min=1, max=6
        // With + on ac: ac=-3+-2=-5, products are [-5, 3, 2, 1], max=3 (DIFFERENT!)
        let arr5_lower =
            Array2::<f32>::from_shape_vec((2, 3), vec![-3.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let arr5_upper =
            Array2::<f32>::from_shape_vec((2, 3), vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let lower5_t = arr5_lower.permuted_axes([1, 0]).into_dyn();
        let upper5_t = arr5_upper.permuted_axes([1, 0]).into_dyn();
        let t5 = BoundedTensor::new(lower5_t, upper5_t).unwrap();

        let arr6_lower =
            Array2::<f32>::from_shape_vec((2, 3), vec![-2.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let arr6_upper =
            Array2::<f32>::from_shape_vec((2, 3), vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let lower6_t = arr6_lower.permuted_axes([1, 0]).into_dyn();
        let upper6_t = arr6_upper.permuted_axes([1, 0]).into_dyn();
        let t6 = BoundedTensor::new(lower6_t, upper6_t).unwrap();

        let result3 = t5.mul(&t6).unwrap();
        assert_eq!(result3.lower[[0, 0]], 1.0, "min should be bd=1 (b*d=-1*-1)");
        assert_eq!(result3.upper[[0, 0]], 6.0, "max should be ac=6 (a*c=-3*-2)");
    }

    #[test]
    fn test_mul_exact_products_negative_intervals() {
        // Additional mul test with negative values to catch * with + or / mutations
        // For intervals with negatives, the products differ significantly from sums/quotients

        // [-2, -1] * [3, 4] should give [-8, -3]
        // ac=-2*3=-6, ad=-2*4=-8, bc=-1*3=-3, bd=-1*4=-4
        // min=-8, max=-3
        let a = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[-1.0]).into_dyn()).unwrap();
        let b = BoundedTensor::new(arr1(&[3.0]).into_dyn(), arr1(&[4.0]).into_dyn()).unwrap();
        let result = a.mul(&b).unwrap();

        // With +: ac=-2+3=1, ad=-2+4=2, bc=-1+3=2, bd=-1+4=3 → [1, 3] (wrong sign!)
        // With /: ac=-2/3=-0.67, ad=-2/4=-0.5, bc=-1/3=-0.33, bd=-1/4=-0.25 → [-0.67, -0.25] (wrong magnitude!)
        assert_eq!(
            result.lower[[0]],
            -8.0,
            "min of products with negative interval"
        );
        assert_eq!(
            result.upper[[0]],
            -3.0,
            "max of products with negative interval"
        );

        // [0, 1] * [0, 1] should give [0, 1]
        // Products: 0*0=0, 0*1=0, 1*0=0, 1*1=1
        let c = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
        let result2 = c.mul(&c.clone()).unwrap();
        // With +: 0+0=0, 0+1=1, 1+0=1, 1+1=2 → [0, 2] (wrong upper!)
        assert_eq!(result2.lower[[0]], 0.0);
        assert_eq!(
            result2.upper[[0]],
            1.0,
            "[0,1]*[0,1] upper must be 1, not 2"
        );
    }
}
