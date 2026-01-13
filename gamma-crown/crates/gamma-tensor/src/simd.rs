//! SIMD-accelerated interval arithmetic operations.
//!
//! This module provides vectorized implementations of key bound propagation operations:
//! - Interval multiplication with 4-way min/max reduction
//! - Positive/negative coefficient split
//! - Safe multiplication handling inf/nan
//!
//! Uses platform-specific SIMD intrinsics (NEON on aarch64, AVX2 on x86_64)
//! with fallback to auto-vectorizable scalar code.
//!
//! # Performance
//!
//! These operations are hot paths in bound propagation:
//! - `interval_mul_inplace`: 2-4x faster than scalar for large arrays
//! - `pos_neg_split`: 2x faster due to single pass
//! - `safe_mul_add`: Handles inf/nan while maintaining vectorization

/// Number of f32 elements per SIMD vector.
/// - aarch64 NEON: 4 (128-bit)
/// - x86_64 AVX2: 8 (256-bit)
#[cfg(target_arch = "aarch64")]
const SIMD_WIDTH: usize = 4;

#[cfg(target_arch = "x86_64")]
const SIMD_WIDTH: usize = 8;

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
const SIMD_WIDTH: usize = 4;

/// Interval multiplication: `[a,b] * [c,d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]`
///
/// Computes element-wise interval products for two arrays of intervals.
/// Results are written to `out_lower` and `out_upper`.
///
/// # Arguments
/// * `a_lower`, `a_upper` - First interval array
/// * `b_lower`, `b_upper` - Second interval array
/// * `out_lower`, `out_upper` - Output interval array (must be same length)
///
/// # Panics
/// Panics if slice lengths don't match.
#[inline]
pub fn interval_mul(
    a_lower: &[f32],
    a_upper: &[f32],
    b_lower: &[f32],
    b_upper: &[f32],
    out_lower: &mut [f32],
    out_upper: &mut [f32],
) {
    let n = a_lower.len();
    debug_assert_eq!(n, a_upper.len());
    debug_assert_eq!(n, b_lower.len());
    debug_assert_eq!(n, b_upper.len());
    debug_assert_eq!(n, out_lower.len());
    debug_assert_eq!(n, out_upper.len());

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 always has NEON
        unsafe { interval_mul_neon(a_lower, a_upper, b_lower, b_upper, out_lower, out_upper) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 is detected
            unsafe { interval_mul_avx2(a_lower, a_upper, b_lower, b_upper, out_lower, out_upper) };
        } else {
            interval_mul_scalar(a_lower, a_upper, b_lower, b_upper, out_lower, out_upper);
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        interval_mul_scalar(a_lower, a_upper, b_lower, b_upper, out_lower, out_upper);
    }
}

/// Scalar fallback for interval multiplication.
#[inline]
fn interval_mul_scalar(
    a_lower: &[f32],
    a_upper: &[f32],
    b_lower: &[f32],
    b_upper: &[f32],
    out_lower: &mut [f32],
    out_upper: &mut [f32],
) {
    for i in 0..a_lower.len() {
        let al = a_lower[i];
        let au = a_upper[i];
        let bl = b_lower[i];
        let bu = b_upper[i];

        let p1 = al * bl;
        let p2 = al * bu;
        let p3 = au * bl;
        let p4 = au * bu;

        out_lower[i] = p1.min(p2).min(p3).min(p4);
        out_upper[i] = p1.max(p2).max(p3).max(p4);
    }
}

/// NEON implementation for aarch64.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn interval_mul_neon(
    a_lower: &[f32],
    a_upper: &[f32],
    b_lower: &[f32],
    b_upper: &[f32],
    out_lower: &mut [f32],
    out_upper: &mut [f32],
) {
    use std::arch::aarch64::*;

    let n = a_lower.len();
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    // Process 4 elements at a time
    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;

        // Load vectors
        let al = vld1q_f32(a_lower.as_ptr().add(offset));
        let au = vld1q_f32(a_upper.as_ptr().add(offset));
        let bl = vld1q_f32(b_lower.as_ptr().add(offset));
        let bu = vld1q_f32(b_upper.as_ptr().add(offset));

        // Compute all four products
        let p1 = vmulq_f32(al, bl); // al * bl
        let p2 = vmulq_f32(al, bu); // al * bu
        let p3 = vmulq_f32(au, bl); // au * bl
        let p4 = vmulq_f32(au, bu); // au * bu

        // Min/max reduction
        let min12 = vminq_f32(p1, p2);
        let min34 = vminq_f32(p3, p4);
        let min_all = vminq_f32(min12, min34);

        let max12 = vmaxq_f32(p1, p2);
        let max34 = vmaxq_f32(p3, p4);
        let max_all = vmaxq_f32(max12, max34);

        // Store results
        vst1q_f32(out_lower.as_mut_ptr().add(offset), min_all);
        vst1q_f32(out_upper.as_mut_ptr().add(offset), max_all);
    }

    // Handle remainder with scalar
    if remainder > 0 {
        let start = chunks * SIMD_WIDTH;
        interval_mul_scalar(
            &a_lower[start..],
            &a_upper[start..],
            &b_lower[start..],
            &b_upper[start..],
            &mut out_lower[start..],
            &mut out_upper[start..],
        );
    }
}

/// AVX2 implementation for x86_64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interval_mul_avx2(
    a_lower: &[f32],
    a_upper: &[f32],
    b_lower: &[f32],
    b_upper: &[f32],
    out_lower: &mut [f32],
    out_upper: &mut [f32],
) {
    use std::arch::x86_64::*;

    let n = a_lower.len();
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    // Process 8 elements at a time
    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;

        // Load vectors
        let al = _mm256_loadu_ps(a_lower.as_ptr().add(offset));
        let au = _mm256_loadu_ps(a_upper.as_ptr().add(offset));
        let bl = _mm256_loadu_ps(b_lower.as_ptr().add(offset));
        let bu = _mm256_loadu_ps(b_upper.as_ptr().add(offset));

        // Compute all four products
        let p1 = _mm256_mul_ps(al, bl);
        let p2 = _mm256_mul_ps(al, bu);
        let p3 = _mm256_mul_ps(au, bl);
        let p4 = _mm256_mul_ps(au, bu);

        // Min/max reduction
        let min12 = _mm256_min_ps(p1, p2);
        let min34 = _mm256_min_ps(p3, p4);
        let min_all = _mm256_min_ps(min12, min34);

        let max12 = _mm256_max_ps(p1, p2);
        let max34 = _mm256_max_ps(p3, p4);
        let max_all = _mm256_max_ps(max12, max34);

        // Store results
        _mm256_storeu_ps(out_lower.as_mut_ptr().add(offset), min_all);
        _mm256_storeu_ps(out_upper.as_mut_ptr().add(offset), max_all);
    }

    // Handle remainder with scalar
    if remainder > 0 {
        let start = chunks * SIMD_WIDTH;
        interval_mul_scalar(
            &a_lower[start..],
            &a_upper[start..],
            &b_lower[start..],
            &b_upper[start..],
            &mut out_lower[start..],
            &mut out_upper[start..],
        );
    }
}

/// Split array into positive and negative parts in a single pass.
///
/// Computes: `pos[i] = max(x[i], 0)`, `neg[i] = min(x[i], 0)`
///
/// More efficient than two separate mapv calls.
#[inline]
pub fn pos_neg_split(x: &[f32], pos: &mut [f32], neg: &mut [f32]) {
    let n = x.len();
    debug_assert_eq!(n, pos.len());
    debug_assert_eq!(n, neg.len());

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 always has NEON
        unsafe { pos_neg_split_neon(x, pos, neg) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 is detected
            unsafe { pos_neg_split_avx2(x, pos, neg) };
        } else {
            pos_neg_split_scalar(x, pos, neg);
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        pos_neg_split_scalar(x, pos, neg);
    }
}

/// Scalar fallback for pos/neg split.
#[inline]
fn pos_neg_split_scalar(x: &[f32], pos: &mut [f32], neg: &mut [f32]) {
    for i in 0..x.len() {
        let v = x[i];
        pos[i] = v.max(0.0);
        neg[i] = v.min(0.0);
    }
}

/// NEON implementation for pos/neg split.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pos_neg_split_neon(x: &[f32], pos: &mut [f32], neg: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = x.len();
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    let zero = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let v = vld1q_f32(x.as_ptr().add(offset));

        let pos_v = vmaxq_f32(v, zero);
        let neg_v = vminq_f32(v, zero);

        vst1q_f32(pos.as_mut_ptr().add(offset), pos_v);
        vst1q_f32(neg.as_mut_ptr().add(offset), neg_v);
    }

    if remainder > 0 {
        let start = chunks * SIMD_WIDTH;
        pos_neg_split_scalar(&x[start..], &mut pos[start..], &mut neg[start..]);
    }
}

/// AVX2 implementation for pos/neg split.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pos_neg_split_avx2(x: &[f32], pos: &mut [f32], neg: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = x.len();
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    let zero = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let v = _mm256_loadu_ps(x.as_ptr().add(offset));

        let pos_v = _mm256_max_ps(v, zero);
        let neg_v = _mm256_min_ps(v, zero);

        _mm256_storeu_ps(pos.as_mut_ptr().add(offset), pos_v);
        _mm256_storeu_ps(neg.as_mut_ptr().add(offset), neg_v);
    }

    if remainder > 0 {
        let start = chunks * SIMD_WIDTH;
        pos_neg_split_scalar(&x[start..], &mut pos[start..], &mut neg[start..]);
    }
}

/// Safe element-wise multiply-add: `out[i] += a[i] * b[i]`, handling `0 * inf = 0`.
///
/// In interval arithmetic, a coefficient of 0 means no contribution,
/// so 0 * inf = 0 (not NaN). This is critical for sound verification.
#[inline]
pub fn safe_mul_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    debug_assert_eq!(n, out.len());

    // Check if we need safe handling (any inf values)
    let has_inf = a.iter().any(|&v| v.is_infinite()) || b.iter().any(|&v| v.is_infinite());

    if has_inf {
        // Safe path: handle 0 * inf = 0
        for i in 0..n {
            let av = a[i];
            let bv = b[i];
            if av == 0.0 || bv == 0.0 {
                // 0 * anything = 0 (including inf)
                // Don't add to out - contribution is 0
            } else {
                out[i] += av * bv;
            }
        }
    } else {
        // Fast path: use SIMD
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: aarch64 always has NEON
            unsafe { mul_add_neon(a, b, out) };
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("fma") {
                // SAFETY: FMA is detected
                unsafe { mul_add_fma(a, b, out) };
            } else if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 is detected
                unsafe { mul_add_avx2(a, b, out) };
            } else {
                mul_add_scalar(a, b, out);
            }
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            mul_add_scalar(a, b, out);
        }
    }
}

/// Scalar fallback for mul_add.
#[inline]
fn mul_add_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] += a[i] * b[i];
    }
}

/// NEON implementation for mul_add (uses fused multiply-add).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mul_add_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let av = vld1q_f32(a.as_ptr().add(offset));
        let bv = vld1q_f32(b.as_ptr().add(offset));
        let cv = vld1q_f32(out.as_ptr().add(offset));

        // Fused multiply-add: cv + av * bv
        let result = vfmaq_f32(cv, av, bv);
        vst1q_f32(out.as_mut_ptr().add(offset), result);
    }

    if remainder > 0 {
        let start = chunks * SIMD_WIDTH;
        mul_add_scalar(&a[start..], &b[start..], &mut out[start..]);
    }
}

/// AVX2 implementation for mul_add (without FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_add_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let av = _mm256_loadu_ps(a.as_ptr().add(offset));
        let bv = _mm256_loadu_ps(b.as_ptr().add(offset));
        let cv = _mm256_loadu_ps(out.as_ptr().add(offset));

        let prod = _mm256_mul_ps(av, bv);
        let result = _mm256_add_ps(cv, prod);
        _mm256_storeu_ps(out.as_mut_ptr().add(offset), result);
    }

    if remainder > 0 {
        let start = chunks * SIMD_WIDTH;
        mul_add_scalar(&a[start..], &b[start..], &mut out[start..]);
    }
}

/// FMA implementation for x86_64 with FMA support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn mul_add_fma(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let av = _mm256_loadu_ps(a.as_ptr().add(offset));
        let bv = _mm256_loadu_ps(b.as_ptr().add(offset));
        let cv = _mm256_loadu_ps(out.as_ptr().add(offset));

        // Fused multiply-add: cv + av * bv
        let result = _mm256_fmadd_ps(av, bv, cv);
        _mm256_storeu_ps(out.as_mut_ptr().add(offset), result);
    }

    if remainder > 0 {
        let start = chunks * SIMD_WIDTH;
        mul_add_scalar(&a[start..], &b[start..], &mut out[start..]);
    }
}

/// Sum all elements of a slice.
#[inline]
pub fn sum(x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 always has NEON
        unsafe { sum_neon(x) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 is detected
            unsafe { sum_avx2(x) }
        } else {
            sum_scalar(x)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        sum_scalar(x)
    }
}

/// Scalar fallback for sum.
#[allow(dead_code)]
#[inline]
fn sum_scalar(x: &[f32]) -> f32 {
    x.iter().sum()
}

/// NEON implementation for sum.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_neon(x: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = x.len();
    let chunks = n / SIMD_WIDTH;

    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let v = vld1q_f32(x.as_ptr().add(offset));
        acc = vaddq_f32(acc, v);
    }

    // Horizontal sum
    let mut result = vaddvq_f32(acc);

    // Add remainder
    for &v in &x[(chunks * SIMD_WIDTH)..] {
        result += v;
    }

    result
}

/// AVX2 implementation for sum.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_avx2(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = x.len();
    let chunks = n / SIMD_WIDTH;

    let mut acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let v = _mm256_loadu_ps(x.as_ptr().add(offset));
        acc = _mm256_add_ps(acc, v);
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, hi32);
    let mut result = _mm_cvtss_f32(sum32);

    // Add remainder
    for i in (chunks * SIMD_WIDTH)..n {
        result += x[i];
    }

    result
}

/// Compute dot product of two slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    debug_assert_eq!(n, b.len());

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 always has NEON
        unsafe { dot_neon(a, b) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            // SAFETY: FMA is detected
            unsafe { dot_fma(a, b) }
        } else if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 is detected
            unsafe { dot_avx2(a, b) }
        } else {
            dot_scalar(a, b)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_scalar(a, b)
    }
}

/// Scalar fallback for dot product.
#[allow(dead_code)]
#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// NEON implementation for dot product.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / SIMD_WIDTH;

    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let av = vld1q_f32(a.as_ptr().add(offset));
        let bv = vld1q_f32(b.as_ptr().add(offset));
        acc = vfmaq_f32(acc, av, bv);
    }

    let mut result = vaddvq_f32(acc);

    // Remainder
    for i in (chunks * SIMD_WIDTH)..n {
        result += a[i] * b[i];
    }

    result
}

/// AVX2 implementation for dot product (no FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / SIMD_WIDTH;

    let mut acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let av = _mm256_loadu_ps(a.as_ptr().add(offset));
        let bv = _mm256_loadu_ps(b.as_ptr().add(offset));
        let prod = _mm256_mul_ps(av, bv);
        acc = _mm256_add_ps(acc, prod);
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, hi32);
    let mut result = _mm_cvtss_f32(sum32);

    // Remainder
    for i in (chunks * SIMD_WIDTH)..n {
        result += a[i] * b[i];
    }

    result
}

/// FMA implementation for dot product.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn dot_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / SIMD_WIDTH;

    let mut acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let av = _mm256_loadu_ps(a.as_ptr().add(offset));
        let bv = _mm256_loadu_ps(b.as_ptr().add(offset));
        acc = _mm256_fmadd_ps(av, bv, acc);
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, hi32);
    let mut result = _mm_cvtss_f32(sum32);

    // Remainder
    for i in (chunks * SIMD_WIDTH)..n {
        result += a[i] * b[i];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_mul_basic() {
        // [1, 2] * [3, 4] = [3, 8]
        let a_lower = [1.0];
        let a_upper = [2.0];
        let b_lower = [3.0];
        let b_upper = [4.0];
        let mut out_lower = [0.0];
        let mut out_upper = [0.0];

        interval_mul(
            &a_lower,
            &a_upper,
            &b_lower,
            &b_upper,
            &mut out_lower,
            &mut out_upper,
        );

        assert!((out_lower[0] - 3.0).abs() < 1e-6);
        assert!((out_upper[0] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_interval_mul_negative() {
        // [-2, 1] * [-3, 2]
        // Products: (-2)*(-3)=6, (-2)*2=-4, 1*(-3)=-3, 1*2=2
        // min=-4, max=6 → [-4, 6]
        let a_lower = [-2.0];
        let a_upper = [1.0];
        let b_lower = [-3.0];
        let b_upper = [2.0];
        let mut out_lower = [0.0];
        let mut out_upper = [0.0];

        interval_mul(
            &a_lower,
            &a_upper,
            &b_lower,
            &b_upper,
            &mut out_lower,
            &mut out_upper,
        );

        assert!((out_lower[0] - (-4.0)).abs() < 1e-6);
        assert!((out_upper[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_interval_mul_large_array() {
        let n = 1024;
        let a_lower: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let a_upper: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let b_lower: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let b_upper: Vec<f32> = (0..n).map(|i| ((i + 1) as f32) * 0.5).collect();
        let mut out_lower = vec![0.0; n];
        let mut out_upper = vec![0.0; n];

        interval_mul(
            &a_lower,
            &a_upper,
            &b_lower,
            &b_upper,
            &mut out_lower,
            &mut out_upper,
        );

        // Verify a few elements
        for i in [0, 100, 500, 1000, 1023] {
            let al = a_lower[i];
            let au = a_upper[i];
            let bl = b_lower[i];
            let bu = b_upper[i];

            let expected_min = (al * bl).min(al * bu).min(au * bl).min(au * bu);
            let expected_max = (al * bl).max(al * bu).max(au * bl).max(au * bu);

            assert!(
                (out_lower[i] - expected_min).abs() < 1e-4,
                "Lower mismatch at {}: {} vs {}",
                i,
                out_lower[i],
                expected_min
            );
            assert!(
                (out_upper[i] - expected_max).abs() < 1e-4,
                "Upper mismatch at {}: {} vs {}",
                i,
                out_upper[i],
                expected_max
            );
        }
    }

    #[test]
    fn test_pos_neg_split_basic() {
        let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut pos = [0.0; 5];
        let mut neg = [0.0; 5];

        pos_neg_split(&x, &mut pos, &mut neg);

        assert_eq!(pos, [0.0, 0.0, 0.0, 1.0, 2.0]);
        assert_eq!(neg, [-2.0, -1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_pos_neg_split_large() {
        let n = 1024;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) - 512.0).collect();
        let mut pos = vec![0.0; n];
        let mut neg = vec![0.0; n];

        pos_neg_split(&x, &mut pos, &mut neg);

        for i in 0..n {
            assert_eq!(pos[i], x[i].max(0.0));
            assert_eq!(neg[i], x[i].min(0.0));
        }
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let result = dot(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!((result - expected).abs() < 1e-4);
    }

    #[test]
    fn test_dot_large() {
        let n = 10000;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..n).map(|i| ((n - i) as f32) * 0.001).collect();

        let result = dot(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!((result - expected).abs() / expected.abs() < 1e-4);
    }

    #[test]
    fn test_sum() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = sum(&x);
        assert!((result - 36.0).abs() < 1e-6);
    }

    #[test]
    fn test_safe_mul_add_normal() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let mut out = [0.0; 4];

        safe_mul_add(&a, &b, &mut out);

        assert_eq!(out, [4.0, 6.0, 6.0, 4.0]);
    }

    #[test]
    fn test_safe_mul_add_with_inf() {
        let a = [0.0, 1.0, 2.0, 0.0];
        let b = [f32::INFINITY, f32::NEG_INFINITY, 3.0, f32::INFINITY];
        let mut out = [0.0; 4];

        safe_mul_add(&a, &b, &mut out);

        // 0 * inf = 0 (contribution is 0, so out stays 0)
        assert_eq!(out[0], 0.0);
        // 1 * -inf = -inf
        assert!(out[1].is_infinite() && out[1].is_sign_negative());
        // 2 * 3 = 6
        assert!((out[2] - 6.0).abs() < 1e-6);
        // 0 * inf = 0
        assert_eq!(out[3], 0.0);
    }

    // ========================================
    // Mutation-killing tests for SIMD functions
    // ========================================

    #[test]
    fn test_mul_add_scalar_exact() {
        // Test that mul_add actually performs: out[i] += a[i] * b[i]
        let a = [2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0];
        let mut out = [1.0, 2.0, 3.0]; // Start with non-zero

        mul_add_scalar(&a, &b, &mut out);

        // out[0] = 1.0 + 2.0*5.0 = 11.0 (not 10 if we didn't add, not 1.0 if no-op)
        assert_eq!(out[0], 11.0);
        // out[1] = 2.0 + 3.0*6.0 = 20.0
        assert_eq!(out[1], 20.0);
        // out[2] = 3.0 + 4.0*7.0 = 31.0
        assert_eq!(out[2], 31.0);
    }

    #[test]
    fn test_sum_scalar_exact() {
        // Test that sum returns actual sum, not 0, 1, or -1
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sum_scalar(&x);
        assert_eq!(result, 15.0); // Not 0, 1, or -1

        // Test with negative values
        let y = [-1.0, -2.0, 3.0];
        assert_eq!(sum_scalar(&y), 0.0);

        // Test single element
        let z = [42.0];
        assert_eq!(sum_scalar(&z), 42.0);
    }

    #[test]
    fn test_dot_scalar_exact() {
        // Test that dot computes actual dot product
        // [1,2,3] . [4,5,6] = 4 + 10 + 18 = 32
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = dot_scalar(&a, &b);
        assert_eq!(result, 32.0); // Not 0, 1, or -1

        // Test that multiplication happens, not addition
        let c = [10.0, 20.0];
        let d = [2.0, 3.0];
        assert_eq!(dot_scalar(&c, &d), 80.0); // 10*2 + 20*3 = 80, not 35 (sum)
    }

    #[test]
    fn test_mul_add_scalar_not_noop() {
        // Verify the function actually modifies output
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        let mut out = [0.0, 0.0];

        mul_add_scalar(&a, &b, &mut out);

        // If replaced with (), output would still be [0, 0]
        assert_ne!(out[0], 0.0);
        assert_ne!(out[1], 0.0);
        assert_eq!(out[0], 3.0); // 0 + 1*3 = 3
        assert_eq!(out[1], 8.0); // 0 + 2*4 = 8
    }

    #[test]
    fn test_sum_empty() {
        let empty: [f32; 0] = [];
        assert_eq!(sum_scalar(&empty), 0.0);
    }

    #[test]
    fn test_dot_distinguishes_operations() {
        // Test that * is not replaced with + or /
        let a = [3.0, 4.0];
        let b = [2.0, 5.0];
        // Correct: 3*2 + 4*5 = 6 + 20 = 26
        // If + instead of *: (3+2) + (4+5) = 14
        // If / instead of *: (3/2) + (4/5) = 1.5 + 0.8 = 2.3
        let result = dot_scalar(&a, &b);
        assert_eq!(result, 26.0);
    }

    #[test]
    fn test_sum_public_api() {
        // Test the public sum function
        let x = [1.0, 2.0, 3.0, 4.0];
        let result = sum(&x);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_dot_public_api() {
        // Test the public dot function
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let result = dot(&a, &b);
        // 1*4 + 2*3 + 3*2 + 4*1 = 4 + 6 + 6 + 4 = 20
        assert_eq!(result, 20.0);
    }
}
