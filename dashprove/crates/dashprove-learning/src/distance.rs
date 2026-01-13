//! SIMD-accelerated distance kernels with portable fallbacks.
//!
//! The SIMD paths use runtime feature detection (AVX2 on x86/x86_64, NEON on
//! aarch64) and fall back to a scalar implementation when unavailable.
//!
//! Provides:
//! - `euclidean_distance_sq` / `euclidean_distance` - L2 distance for f32 vectors
//! - `dot_product` - Inner product for f32 vectors (used by cosine similarity)
//! - `vector_norm_sq` / `vector_norm` - L2 norm (magnitude) for f32 vectors
//! - `hamming_distance` - Hamming distance for byte slices (binary codes)

/// Squared Euclidean distance (`L2^2`) between two vectors.
#[inline]
pub fn euclidean_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        // Prefer AVX2 when available, otherwise use SSE2.
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { euclidean_distance_sq_avx2(a, b) };
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 is guaranteed on x86_64.
            return unsafe { euclidean_distance_sq_sse2(a, b) };
        }

        return euclidean_distance_sq_scalar(a, b);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is guaranteed on aarch64 targets.
        unsafe { euclidean_distance_sq_neon(a, b) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        euclidean_distance_sq_scalar(a, b)
    }
}

/// Euclidean distance (`L2`) between two vectors.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    euclidean_distance_sq(a, b).sqrt()
}

/// Dot product (inner product) of two vectors.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { dot_product_avx2(a, b) };
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { dot_product_sse2(a, b) };
        }
        return dot_product_scalar(a, b);
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_product_neon(a, b) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        dot_product_scalar(a, b)
    }
}

/// Scalar fallback for dot product.
#[inline]
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Squared L2 norm (magnitude squared) of a vector.
#[inline]
pub fn vector_norm_sq(v: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { vector_norm_sq_avx2(v) };
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { vector_norm_sq_sse2(v) };
        }
        return vector_norm_sq_scalar(v);
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { vector_norm_sq_neon(v) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        vector_norm_sq_scalar(v)
    }
}

/// L2 norm (magnitude) of a vector.
#[inline]
pub fn vector_norm(v: &[f32]) -> f32 {
    vector_norm_sq(v).sqrt()
}

/// Scalar fallback for squared vector norm.
#[inline]
pub fn vector_norm_sq_scalar(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum()
}

/// Scalar fallback for squared Euclidean distance.
#[inline]
pub fn euclidean_distance_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ============================= x86/x86_64 =============================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn euclidean_distance_sq_sse2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut sum = _mm_setzero_ps();
    let mut i = 0usize;

    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let diff = _mm_sub_ps(va, vb);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        i += 4;
    }

    let mut buf = [0.0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), sum);
    let mut total = buf.iter().sum();

    while i < len {
        let d = *a.get_unchecked(i) - *b.get_unchecked(i);
        total += d * d;
        i += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn euclidean_distance_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut sum = _mm256_setzero_ps();
    let mut i = 0usize;

    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        i += 8;
    }

    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), sum);
    let mut total: f32 = buf.iter().sum();

    // Remainder handled with SSE2 for better throughput than scalar tail.
    if i < len {
        total += euclidean_distance_sq_sse2(a.get_unchecked(i..len), b.get_unchecked(i..len));
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn dot_product_sse2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut sum = _mm_setzero_ps();
    let mut i = 0usize;

    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
        i += 4;
    }

    let mut buf = [0.0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), sum);
    let mut total: f32 = buf.iter().sum();

    while i < len {
        total += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut sum = _mm256_setzero_ps();
    let mut i = 0usize;

    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum = _mm256_fmadd_ps(va, vb, sum);
        i += 8;
    }

    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), sum);
    let mut total: f32 = buf.iter().sum();

    if i < len {
        total += dot_product_sse2(a.get_unchecked(i..len), b.get_unchecked(i..len));
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn vector_norm_sq_sse2(v: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = v.len();
    let mut sum = _mm_setzero_ps();
    let mut i = 0usize;

    while i + 4 <= len {
        let vv = _mm_loadu_ps(v.as_ptr().add(i));
        sum = _mm_add_ps(sum, _mm_mul_ps(vv, vv));
        i += 4;
    }

    let mut buf = [0.0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), sum);
    let mut total: f32 = buf.iter().sum();

    while i < len {
        let x = *v.get_unchecked(i);
        total += x * x;
        i += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vector_norm_sq_avx2(v: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = v.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0usize;

    while i + 8 <= len {
        let vv = _mm256_loadu_ps(v.as_ptr().add(i));
        sum = _mm256_fmadd_ps(vv, vv, sum);
        i += 8;
    }

    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), sum);
    let mut total: f32 = buf.iter().sum();

    if i < len {
        total += vector_norm_sq_sse2(v.get_unchecked(i..len));
    }

    total
}

// ================================ NEON ================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn euclidean_distance_sq_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len().min(b.len());
    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0usize;

    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff);
        i += 4;
    }

    let mut buf = [0.0f32; 4];
    vst1q_f32(buf.as_mut_ptr(), sum);
    let mut total: f32 = buf.iter().sum();

    while i < len {
        let d = *a.get_unchecked(i) - *b.get_unchecked(i);
        total += d * d;
        i += 1;
    }

    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len().min(b.len());
    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0usize;

    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        sum = vfmaq_f32(sum, va, vb);
        i += 4;
    }

    let mut buf = [0.0f32; 4];
    vst1q_f32(buf.as_mut_ptr(), sum);
    let mut total: f32 = buf.iter().sum();

    while i < len {
        total += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }

    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vector_norm_sq_neon(v: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = v.len();
    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0usize;

    while i + 4 <= len {
        let vv = vld1q_f32(v.as_ptr().add(i));
        sum = vfmaq_f32(sum, vv, vv);
        i += 4;
    }

    let mut buf = [0.0f32; 4];
    vst1q_f32(buf.as_mut_ptr(), sum);
    let mut total: f32 = buf.iter().sum();

    while i < len {
        let x = *v.get_unchecked(i);
        total += x * x;
        i += 1;
    }

    total
}

// ========================= Hamming Distance ==========================

/// Hamming distance between two byte slices (binary codes).
///
/// Uses POPCNT instruction on x86_64 when available, otherwise falls back
/// to Rust's `count_ones()` which the compiler can optimize.
#[inline]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("popcnt") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { hamming_distance_popcnt(a, b) };
        }
    }

    hamming_distance_scalar(a, b)
}

/// Scalar Hamming distance using 64-bit chunks and `count_ones()`.
#[inline]
pub fn hamming_distance_scalar(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len().min(b.len());
    let chunks = len / 8;
    let mut total = 0u32;

    // Process 8 bytes at a time using u64 popcount
    for i in 0..chunks {
        let offset = i * 8;
        let a_chunk = u64::from_ne_bytes([
            a[offset],
            a[offset + 1],
            a[offset + 2],
            a[offset + 3],
            a[offset + 4],
            a[offset + 5],
            a[offset + 6],
            a[offset + 7],
        ]);
        let b_chunk = u64::from_ne_bytes([
            b[offset],
            b[offset + 1],
            b[offset + 2],
            b[offset + 3],
            b[offset + 4],
            b[offset + 5],
            b[offset + 6],
            b[offset + 7],
        ]);
        total += (a_chunk ^ b_chunk).count_ones();
    }

    // Handle remaining bytes
    for i in (chunks * 8)..len {
        total += (a[i] ^ b[i]).count_ones();
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn hamming_distance_popcnt(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::_popcnt64;

    let len = a.len().min(b.len());
    let chunks = len / 8;
    let mut total = 0i64;

    // Process 8 bytes at a time using hardware popcnt
    for i in 0..chunks {
        let offset = i * 8;
        let a_ptr = a.as_ptr().add(offset) as *const i64;
        let b_ptr = b.as_ptr().add(offset) as *const i64;
        let a_val = std::ptr::read_unaligned(a_ptr);
        let b_val = std::ptr::read_unaligned(b_ptr);
        total += _popcnt64(a_val ^ b_val);
    }

    // Handle remaining bytes with scalar
    let mut tail = 0u32;
    for i in (chunks * 8)..len {
        tail += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones();
    }

    (total as u32) + tail
}

// ========================= Matrix-Vector Multiply ==========================

/// Matrix-vector multiplication: result = M * v (M is row-major)
///
/// Uses SIMD dot_product for each row computation.
///
/// # Arguments
/// * `matrix` - Row-major matrix of size dim x dim
/// * `vector` - Input vector of length dim
/// * `dim` - Dimension (number of rows and columns)
///
/// # Returns
/// Result vector of length dim
#[inline]
pub fn matrix_vector_multiply(matrix: &[f32], vector: &[f32], dim: usize) -> Vec<f32> {
    debug_assert_eq!(matrix.len(), dim * dim);
    debug_assert_eq!(vector.len(), dim);

    (0..dim)
        .map(|i| {
            let row_start = i * dim;
            dot_product(&matrix[row_start..row_start + dim], vector)
        })
        .collect()
}

/// Transpose matrix-vector multiply: result = M^T * v (M is row-major)
///
/// Uses SIMD acceleration for the inner products.
///
/// # Arguments
/// * `matrix` - Row-major matrix of size dim x dim
/// * `vector` - Input vector of length dim
/// * `dim` - Dimension (number of rows and columns)
///
/// # Returns
/// Result vector of length dim
#[inline]
pub fn transpose_matrix_vector_multiply(matrix: &[f32], vector: &[f32], dim: usize) -> Vec<f32> {
    debug_assert_eq!(matrix.len(), dim * dim);
    debug_assert_eq!(vector.len(), dim);

    // M^T * v: result[j] = sum_i(M[i,j] * v[i])
    // This is less cache-friendly than M*v but we can still SIMD the accumulation

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            let mut result = vec![0.0f32; dim];
            // SAFETY: guarded by runtime feature detection
            unsafe {
                transpose_matvec_avx2(matrix, vector, &mut result, dim);
            }
            return result;
        }
        // Scalar fallback for x86_64 without AVX2
        return transpose_matvec_scalar(matrix, vector, dim);
    }

    #[cfg(target_arch = "aarch64")]
    {
        let mut result = vec![0.0f32; dim];
        // SAFETY: NEON is guaranteed on aarch64
        unsafe {
            transpose_matvec_neon(matrix, vector, &mut result, dim);
        }
        result
    }

    // Scalar fallback for other architectures
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        transpose_matvec_scalar(matrix, vector, dim)
    }
}

/// Scalar fallback for transpose matrix-vector multiply
#[inline]
#[allow(dead_code)]
fn transpose_matvec_scalar(matrix: &[f32], vector: &[f32], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim];
    for j in 0..dim {
        for i in 0..dim {
            result[j] += matrix[i * dim + j] * vector[i];
        }
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_matvec_avx2(matrix: &[f32], vector: &[f32], result: &mut [f32], dim: usize) {
    use std::arch::x86_64::*;

    // For each column j, compute result[j] = sum_i(M[i,j] * v[i])
    // We process 8 columns at a time with AVX2

    let mut j = 0usize;
    while j + 8 <= dim {
        let mut sum = _mm256_setzero_ps();

        for i in 0..dim {
            let v_broadcast = _mm256_set1_ps(*vector.get_unchecked(i));
            let m_row = _mm256_loadu_ps(matrix.as_ptr().add(i * dim + j));
            sum = _mm256_fmadd_ps(m_row, v_broadcast, sum);
        }

        _mm256_storeu_ps(result.as_mut_ptr().add(j), sum);
        j += 8;
    }

    // Handle remaining columns
    while j < dim {
        let mut s = 0.0f32;
        for i in 0..dim {
            s += *matrix.get_unchecked(i * dim + j) * *vector.get_unchecked(i);
        }
        *result.get_unchecked_mut(j) = s;
        j += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn transpose_matvec_neon(matrix: &[f32], vector: &[f32], result: &mut [f32], dim: usize) {
    use std::arch::aarch64::*;

    // Process 4 columns at a time with NEON
    let mut j = 0usize;
    while j + 4 <= dim {
        let mut sum = vdupq_n_f32(0.0);

        for i in 0..dim {
            let v_broadcast = vdupq_n_f32(*vector.get_unchecked(i));
            let m_row = vld1q_f32(matrix.as_ptr().add(i * dim + j));
            sum = vfmaq_f32(sum, m_row, v_broadcast);
        }

        vst1q_f32(result.as_mut_ptr().add(j), sum);
        j += 4;
    }

    // Handle remaining columns
    while j < dim {
        let mut s = 0.0f32;
        for i in 0..dim {
            s += *matrix.get_unchecked(i * dim + j) * *vector.get_unchecked(i);
        }
        *result.get_unchecked_mut(j) = s;
        j += 1;
    }
}

// ========================= Vector Accumulation ==========================

/// Accumulate vector b into vector a: a[i] += b[i] for all i
///
/// Uses SIMD acceleration for the addition operation.
/// This is useful for k-means centroid updates where many vectors
/// are accumulated into centroid sums.
///
/// # Arguments
/// * `a` - Mutable accumulator vector
/// * `b` - Vector to add to accumulator
///
/// # Panics
/// Debug assert if vectors have different lengths
#[inline]
pub fn vector_add_accumulate(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime feature detection
            unsafe {
                vector_add_accumulate_avx2(a, b);
            }
            return;
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 is guaranteed on x86_64
            unsafe {
                vector_add_accumulate_sse2(a, b);
            }
            return;
        }
        vector_add_accumulate_scalar(a, b);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is guaranteed on aarch64
        unsafe {
            vector_add_accumulate_neon(a, b);
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        vector_add_accumulate_scalar(a, b);
    }
}

/// Scalar fallback for vector accumulation
#[inline]
pub fn vector_add_accumulate_scalar(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += *y;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn vector_add_accumulate_sse2(a: &mut [f32], b: &[f32]) {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut i = 0usize;

    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let sum = _mm_add_ps(va, vb);
        _mm_storeu_ps(a.as_mut_ptr().add(i), sum);
        i += 4;
    }

    // Handle remaining elements
    while i < len {
        *a.get_unchecked_mut(i) += *b.get_unchecked(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vector_add_accumulate_avx2(a: &mut [f32], b: &[f32]) {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut i = 0usize;

    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let sum = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(a.as_mut_ptr().add(i), sum);
        i += 8;
    }

    // Handle remaining with SSE2
    if i < len {
        vector_add_accumulate_sse2(a.get_unchecked_mut(i..len), b.get_unchecked(i..len));
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vector_add_accumulate_neon(a: &mut [f32], b: &[f32]) {
    use std::arch::aarch64::*;

    let len = a.len().min(b.len());
    let mut i = 0usize;

    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let sum = vaddq_f32(va, vb);
        vst1q_f32(a.as_mut_ptr().add(i), sum);
        i += 4;
    }

    // Handle remaining elements
    while i < len {
        *a.get_unchecked_mut(i) += *b.get_unchecked(i);
        i += 1;
    }
}

/// Scale a vector in place: a[i] *= scale for all i
///
/// Uses SIMD acceleration for the multiplication operation.
/// This is useful for k-means centroid normalization after accumulation.
///
/// # Arguments
/// * `a` - Mutable vector to scale
/// * `scale` - Scalar multiplier
#[inline]
pub fn vector_scale_inplace(a: &mut [f32], scale: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime feature detection
            unsafe {
                vector_scale_inplace_avx2(a, scale);
            }
            return;
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 is guaranteed on x86_64
            unsafe {
                vector_scale_inplace_sse2(a, scale);
            }
            return;
        }
        vector_scale_inplace_scalar(a, scale);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is guaranteed on aarch64
        unsafe {
            vector_scale_inplace_neon(a, scale);
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        vector_scale_inplace_scalar(a, scale);
    }
}

/// Scalar fallback for vector scaling
#[inline]
pub fn vector_scale_inplace_scalar(a: &mut [f32], scale: f32) {
    for x in a.iter_mut() {
        *x *= scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn vector_scale_inplace_sse2(a: &mut [f32], scale: f32) {
    use std::arch::x86_64::*;

    let len = a.len();
    let scale_vec = _mm_set1_ps(scale);
    let mut i = 0usize;

    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let scaled = _mm_mul_ps(va, scale_vec);
        _mm_storeu_ps(a.as_mut_ptr().add(i), scaled);
        i += 4;
    }

    // Handle remaining elements
    while i < len {
        *a.get_unchecked_mut(i) *= scale;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vector_scale_inplace_avx2(a: &mut [f32], scale: f32) {
    use std::arch::x86_64::*;

    let len = a.len();
    let scale_vec = _mm256_set1_ps(scale);
    let mut i = 0usize;

    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let scaled = _mm256_mul_ps(va, scale_vec);
        _mm256_storeu_ps(a.as_mut_ptr().add(i), scaled);
        i += 8;
    }

    // Handle remaining with SSE2
    if i < len {
        vector_scale_inplace_sse2(a.get_unchecked_mut(i..len), scale);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vector_scale_inplace_neon(a: &mut [f32], scale: f32) {
    use std::arch::aarch64::*;

    let len = a.len();
    let scale_vec = vdupq_n_f32(scale);
    let mut i = 0usize;

    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let scaled = vmulq_f32(va, scale_vec);
        vst1q_f32(a.as_mut_ptr().add(i), scaled);
        i += 4;
    }

    // Handle remaining elements
    while i < len {
        *a.get_unchecked_mut(i) *= scale;
        i += 1;
    }
}

// ================================ Tests ================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vector(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                // Simple LCG for deterministic values
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state % 100_000) as f32 / 10_000.0
            })
            .collect()
    }

    #[test]
    fn distance_matches_scalar_small() {
        let a = make_vector(12, 1);
        let b = make_vector(12, 2);
        let simd = euclidean_distance_sq(&a, &b);
        let scalar = euclidean_distance_sq_scalar(&a, &b);
        let tol = (scalar.abs() * 1e-4).max(1e-5);
        assert!(
            (simd - scalar).abs() < tol,
            "simd {} vs scalar {} (tol {})",
            simd,
            scalar,
            tol
        );
    }

    #[test]
    fn distance_matches_scalar_large_and_unaligned() {
        let a = make_vector(257, 42);
        let b = make_vector(257, 99);
        let simd = euclidean_distance_sq(&a, &b);
        let scalar = euclidean_distance_sq_scalar(&a, &b);
        let tol = (scalar.abs() * 1e-4).max(1e-4);
        assert!(
            (simd - scalar).abs() < tol,
            "simd {} vs scalar {} (tol {})",
            simd,
            scalar,
            tol
        );
    }

    #[test]
    fn distance_is_zero_for_identical_vectors() {
        let a = make_vector(64, 123);
        let simd = euclidean_distance_sq(&a, &a);
        assert!(simd.abs() < 1e-6);
        assert!(euclidean_distance(&a, &a).abs() < 1e-6);
    }

    fn make_bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn hamming_matches_scalar_small() {
        let a = make_bytes(12, 1);
        let b = make_bytes(12, 2);
        let fast = hamming_distance(&a, &b);
        let scalar = hamming_distance_scalar(&a, &b);
        assert_eq!(fast, scalar);
    }

    #[test]
    fn hamming_matches_scalar_large_and_unaligned() {
        let a = make_bytes(127, 42);
        let b = make_bytes(127, 99);
        let fast = hamming_distance(&a, &b);
        let scalar = hamming_distance_scalar(&a, &b);
        assert_eq!(fast, scalar);
    }

    #[test]
    fn hamming_is_zero_for_identical_codes() {
        let a = make_bytes(64, 123);
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn hamming_is_max_for_inverted_codes() {
        let a = make_bytes(8, 1);
        let b: Vec<u8> = a.iter().map(|x| !x).collect();
        // All 64 bits should differ
        assert_eq!(hamming_distance(&a, &b), 64);
    }

    // ================== Dot Product Tests ==================

    #[test]
    fn dot_product_matches_scalar_small() {
        let a = make_vector(12, 1);
        let b = make_vector(12, 2);
        let simd = dot_product(&a, &b);
        let scalar = dot_product_scalar(&a, &b);
        let tol = (scalar.abs() * 1e-4).max(1e-5);
        assert!(
            (simd - scalar).abs() < tol,
            "simd {} vs scalar {} (tol {})",
            simd,
            scalar,
            tol
        );
    }

    #[test]
    fn dot_product_matches_scalar_large_and_unaligned() {
        let a = make_vector(257, 42);
        let b = make_vector(257, 99);
        let simd = dot_product(&a, &b);
        let scalar = dot_product_scalar(&a, &b);
        let tol = (scalar.abs() * 1e-4).max(1e-4);
        assert!(
            (simd - scalar).abs() < tol,
            "simd {} vs scalar {} (tol {})",
            simd,
            scalar,
            tol
        );
    }

    #[test]
    fn dot_product_of_orthogonal_vectors_is_zero() {
        // Create two orthogonal vectors
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        assert!(dot_product(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn dot_product_of_parallel_vectors() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 4.0, 6.0, 8.0]; // 2 * a
        let expected = 2.0 * (1.0 + 4.0 + 9.0 + 16.0); // 2 * |a|^2 = 60
        let result = dot_product(&a, &b);
        assert!(
            (result - expected).abs() < 1e-5,
            "got {} expected {}",
            result,
            expected
        );
    }

    // ================== Vector Norm Tests ==================

    #[test]
    fn vector_norm_sq_matches_scalar_small() {
        let v = make_vector(12, 1);
        let simd = vector_norm_sq(&v);
        let scalar = vector_norm_sq_scalar(&v);
        let tol = (scalar.abs() * 1e-4).max(1e-5);
        assert!(
            (simd - scalar).abs() < tol,
            "simd {} vs scalar {} (tol {})",
            simd,
            scalar,
            tol
        );
    }

    #[test]
    fn vector_norm_sq_matches_scalar_large_and_unaligned() {
        let v = make_vector(257, 42);
        let simd = vector_norm_sq(&v);
        let scalar = vector_norm_sq_scalar(&v);
        let tol = (scalar.abs() * 1e-4).max(1e-4);
        assert!(
            (simd - scalar).abs() < tol,
            "simd {} vs scalar {} (tol {})",
            simd,
            scalar,
            tol
        );
    }

    #[test]
    fn vector_norm_of_unit_vector_is_one() {
        let v = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!((vector_norm(&v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vector_norm_of_zero_vector_is_zero() {
        let v = vec![0.0; 16];
        assert!(vector_norm(&v).abs() < 1e-6);
        assert!(vector_norm_sq(&v).abs() < 1e-6);
    }

    #[test]
    fn vector_norm_sq_equals_dot_product_with_self() {
        let v = make_vector(96, 123);
        let norm_sq = vector_norm_sq(&v);
        let dot_self = dot_product(&v, &v);
        let tol = (norm_sq.abs() * 1e-4).max(1e-5);
        assert!(
            (norm_sq - dot_self).abs() < tol,
            "norm_sq {} vs dot(v,v) {} (tol {})",
            norm_sq,
            dot_self,
            tol
        );
    }

    // ================== Matrix-Vector Multiply Tests ==================

    fn make_matrix(dim: usize, seed: u64) -> Vec<f32> {
        make_vector(dim * dim, seed)
    }

    fn matvec_scalar(m: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; dim];
        for i in 0..dim {
            for j in 0..dim {
                result[i] += m[i * dim + j] * v[j];
            }
        }
        result
    }

    #[test]
    fn matrix_vector_multiply_matches_scalar_small() {
        let m = make_matrix(8, 1);
        let v = make_vector(8, 2);
        let simd = matrix_vector_multiply(&m, &v, 8);
        let scalar = matvec_scalar(&m, &v, 8);
        for i in 0..8 {
            let tol = (scalar[i].abs() * 1e-4).max(1e-5);
            assert!(
                (simd[i] - scalar[i]).abs() < tol,
                "simd[{}]={} vs scalar[{}]={} (tol {})",
                i,
                simd[i],
                i,
                scalar[i],
                tol
            );
        }
    }

    #[test]
    fn matrix_vector_multiply_matches_scalar_large() {
        let m = make_matrix(64, 42);
        let v = make_vector(64, 99);
        let simd = matrix_vector_multiply(&m, &v, 64);
        let scalar = matvec_scalar(&m, &v, 64);
        for i in 0..64 {
            let tol = (scalar[i].abs() * 1e-4).max(1e-4);
            assert!(
                (simd[i] - scalar[i]).abs() < tol,
                "simd[{}]={} vs scalar[{}]={} (tol {})",
                i,
                simd[i],
                i,
                scalar[i],
                tol
            );
        }
    }

    #[test]
    fn matrix_vector_multiply_identity() {
        // Identity matrix should return the same vector
        let dim = 16;
        let mut identity = vec![0.0f32; dim * dim];
        for i in 0..dim {
            identity[i * dim + i] = 1.0;
        }
        let v = make_vector(dim, 123);
        let result = matrix_vector_multiply(&identity, &v, dim);
        for i in 0..dim {
            assert!(
                (result[i] - v[i]).abs() < 1e-6,
                "result[{}]={} vs v[{}]={}",
                i,
                result[i],
                i,
                v[i]
            );
        }
    }

    fn transpose_matvec_scalar_test(m: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; dim];
        for j in 0..dim {
            for i in 0..dim {
                result[j] += m[i * dim + j] * v[i];
            }
        }
        result
    }

    #[test]
    fn transpose_matrix_vector_multiply_matches_scalar_small() {
        let m = make_matrix(8, 1);
        let v = make_vector(8, 2);
        let simd = transpose_matrix_vector_multiply(&m, &v, 8);
        let scalar = transpose_matvec_scalar_test(&m, &v, 8);
        for i in 0..8 {
            let tol = (scalar[i].abs() * 1e-4).max(1e-5);
            assert!(
                (simd[i] - scalar[i]).abs() < tol,
                "simd[{}]={} vs scalar[{}]={} (tol {})",
                i,
                simd[i],
                i,
                scalar[i],
                tol
            );
        }
    }

    #[test]
    fn transpose_matrix_vector_multiply_matches_scalar_large() {
        let m = make_matrix(64, 42);
        let v = make_vector(64, 99);
        let simd = transpose_matrix_vector_multiply(&m, &v, 64);
        let scalar = transpose_matvec_scalar_test(&m, &v, 64);
        for i in 0..64 {
            let tol = (scalar[i].abs() * 1e-4).max(1e-4);
            assert!(
                (simd[i] - scalar[i]).abs() < tol,
                "simd[{}]={} vs scalar[{}]={} (tol {})",
                i,
                simd[i],
                i,
                scalar[i],
                tol
            );
        }
    }

    #[test]
    fn transpose_matrix_vector_multiply_unaligned() {
        // Test with dimension not divisible by SIMD width
        let dim = 13;
        let m = make_matrix(dim, 42);
        let v = make_vector(dim, 99);
        let simd = transpose_matrix_vector_multiply(&m, &v, dim);
        let scalar = transpose_matvec_scalar_test(&m, &v, dim);
        for i in 0..dim {
            let tol = (scalar[i].abs() * 1e-4).max(1e-4);
            assert!(
                (simd[i] - scalar[i]).abs() < tol,
                "simd[{}]={} vs scalar[{}]={} (tol {})",
                i,
                simd[i],
                i,
                scalar[i],
                tol
            );
        }
    }

    #[test]
    fn matvec_transpose_relationship() {
        // (M * v)^T = v^T * M^T, but for vectors: M * v and M^T * v are different
        // Test that M^T * (M * v) produces a valid result
        let dim = 16;
        let m = make_matrix(dim, 42);
        let v = make_vector(dim, 99);
        let mv = matrix_vector_multiply(&m, &v, dim);
        let mtmv = transpose_matrix_vector_multiply(&m, &mv, dim);
        // Just check it produces valid output (not NaN/inf)
        for x in &mtmv {
            assert!(x.is_finite(), "result contains non-finite value: {}", x);
        }
    }

    // ================== Vector Accumulation Tests ==================

    #[test]
    fn vector_add_accumulate_matches_scalar_small() {
        let mut a_simd = make_vector(12, 1);
        let mut a_scalar = a_simd.clone();
        let b = make_vector(12, 2);

        vector_add_accumulate(&mut a_simd, &b);
        vector_add_accumulate_scalar(&mut a_scalar, &b);

        for i in 0..12 {
            let tol = (a_scalar[i].abs() * 1e-5).max(1e-6);
            assert!(
                (a_simd[i] - a_scalar[i]).abs() < tol,
                "simd[{}]={} vs scalar[{}]={} (tol {})",
                i,
                a_simd[i],
                i,
                a_scalar[i],
                tol
            );
        }
    }

    #[test]
    fn vector_add_accumulate_matches_scalar_large_and_unaligned() {
        let mut a_simd = make_vector(257, 42);
        let mut a_scalar = a_simd.clone();
        let b = make_vector(257, 99);

        vector_add_accumulate(&mut a_simd, &b);
        vector_add_accumulate_scalar(&mut a_scalar, &b);

        for i in 0..257 {
            let tol = (a_scalar[i].abs() * 1e-5).max(1e-6);
            assert!(
                (a_simd[i] - a_scalar[i]).abs() < tol,
                "simd[{}]={} vs scalar[{}]={} (tol {})",
                i,
                a_simd[i],
                i,
                a_scalar[i],
                tol
            );
        }
    }

    #[test]
    fn vector_add_accumulate_with_zeros() {
        let mut a = make_vector(64, 123);
        let original = a.clone();
        let zeros = vec![0.0f32; 64];

        vector_add_accumulate(&mut a, &zeros);

        for i in 0..64 {
            assert!(
                (a[i] - original[i]).abs() < 1e-6,
                "Adding zeros changed value at {}: {} -> {}",
                i,
                original[i],
                a[i]
            );
        }
    }

    #[test]
    fn vector_add_accumulate_commutative_effect() {
        let a_orig = make_vector(32, 1);
        let b = make_vector(32, 2);

        // a + b
        let mut result1 = a_orig.clone();
        vector_add_accumulate(&mut result1, &b);

        // Manually compute expected
        let expected: Vec<f32> = a_orig.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        for i in 0..32 {
            let tol = (expected[i].abs() * 1e-5).max(1e-6);
            assert!(
                (result1[i] - expected[i]).abs() < tol,
                "result[{}]={} vs expected[{}]={}",
                i,
                result1[i],
                i,
                expected[i]
            );
        }
    }

    // ================== Vector Scale Tests ==================

    #[test]
    fn vector_scale_inplace_matches_scalar_small() {
        let mut a_simd = make_vector(12, 1);
        let mut a_scalar = a_simd.clone();
        let scale = 2.5f32;

        vector_scale_inplace(&mut a_simd, scale);
        vector_scale_inplace_scalar(&mut a_scalar, scale);

        for i in 0..12 {
            let tol = (a_scalar[i].abs() * 1e-5).max(1e-6);
            assert!(
                (a_simd[i] - a_scalar[i]).abs() < tol,
                "simd[{}]={} vs scalar[{}]={} (tol {})",
                i,
                a_simd[i],
                i,
                a_scalar[i],
                tol
            );
        }
    }

    #[test]
    fn vector_scale_inplace_matches_scalar_large_and_unaligned() {
        let mut a_simd = make_vector(257, 42);
        let mut a_scalar = a_simd.clone();
        let scale = 0.333f32;

        vector_scale_inplace(&mut a_simd, scale);
        vector_scale_inplace_scalar(&mut a_scalar, scale);

        for i in 0..257 {
            let tol = (a_scalar[i].abs() * 1e-5).max(1e-6);
            assert!(
                (a_simd[i] - a_scalar[i]).abs() < tol,
                "simd[{}]={} vs scalar[{}]={} (tol {})",
                i,
                a_simd[i],
                i,
                a_scalar[i],
                tol
            );
        }
    }

    #[test]
    fn vector_scale_inplace_by_one() {
        let mut a = make_vector(64, 123);
        let original = a.clone();

        vector_scale_inplace(&mut a, 1.0);

        for i in 0..64 {
            assert!(
                (a[i] - original[i]).abs() < 1e-6,
                "Scaling by 1 changed value at {}: {} -> {}",
                i,
                original[i],
                a[i]
            );
        }
    }

    #[test]
    fn vector_scale_inplace_by_zero() {
        let mut a = make_vector(64, 123);

        vector_scale_inplace(&mut a, 0.0);

        for (i, &val) in a.iter().enumerate() {
            assert!(
                val.abs() < 1e-6,
                "Scaling by 0 should produce zeros, got {} at {}",
                val,
                i
            );
        }
    }

    #[test]
    fn vector_scale_inplace_inverse() {
        let original = make_vector(32, 42);
        let mut a = original.clone();
        let scale = 3.0f32;

        // Scale up then scale down should return to original (within tolerance)
        vector_scale_inplace(&mut a, scale);
        vector_scale_inplace(&mut a, 1.0 / scale);

        for i in 0..32 {
            let tol = (original[i].abs() * 1e-5).max(1e-6);
            assert!(
                (a[i] - original[i]).abs() < tol,
                "Scale then inverse at {}: {} vs {}",
                i,
                a[i],
                original[i]
            );
        }
    }

    #[test]
    fn vector_accumulate_then_scale_for_mean() {
        // Simulate k-means centroid update: accumulate 4 vectors, then divide by 4
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v2 = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let v3 = vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0];
        let v4 = vec![4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0];

        let mut sum = vec![0.0f32; 8];
        vector_add_accumulate(&mut sum, &v1);
        vector_add_accumulate(&mut sum, &v2);
        vector_add_accumulate(&mut sum, &v3);
        vector_add_accumulate(&mut sum, &v4);
        vector_scale_inplace(&mut sum, 1.0 / 4.0);

        // Expected mean: [(1+2+3+4)/4, (2+4+6+8)/4, ...]
        let expected = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0];

        for i in 0..8 {
            assert!(
                (sum[i] - expected[i]).abs() < 1e-5,
                "Mean at {}: {} vs expected {}",
                i,
                sum[i],
                expected[i]
            );
        }
    }
}
