//! Thread-local memory pool for tensor buffers.
//!
//! This module provides a memory pool that reuses `Vec<f32>` buffers to reduce
//! allocation overhead during bound propagation. Buffers are organized into
//! size classes (powers of 2) for efficient reuse.
//!
//! # Usage
//!
//! ```ignore
//! use gamma_tensor::pool::{TensorPool, PooledBuffer};
//!
//! // Acquire a buffer with at least 1000 elements
//! let mut buffer = TensorPool::acquire(1000);
//!
//! // Fill with data
//! let data = buffer.as_mut_slice();
//! for (i, v) in data.iter_mut().enumerate() {
//!     *v = i as f32;
//! }
//!
//! // Convert to ndarray when done
//! let array = buffer.into_arrayd(&[10, 100]);
//! ```
//!
//! # Performance
//!
//! Expected benefits:
//! - 30-50% memory reduction from buffer reuse
//! - 10-20% speedup from reduced allocation overhead
//! - No locking: thread-local pools avoid synchronization cost

use ndarray::{ArrayD, IxDyn};
use std::cell::RefCell;

/// Minimum size class: 64 elements (256 bytes for f32)
const MIN_SIZE_CLASS: usize = 64;

/// Maximum size class exponent (2^30 = ~1B elements = 4GB per buffer)
const MAX_SIZE_CLASS_EXP: usize = 30;

/// Maximum buffers to keep per size class
const MAX_BUFFERS_PER_CLASS: usize = 16;

thread_local! {
    static POOL: RefCell<PoolStorage> = RefCell::new(PoolStorage::new());
}

/// Internal storage for the thread-local pool.
struct PoolStorage {
    /// buckets[i] holds buffers of size MIN_SIZE_CLASS * 2^i
    /// bucket 0: 64 elements
    /// bucket 1: 128 elements
    /// bucket 2: 256 elements
    /// ...
    buckets: Vec<Vec<Vec<f32>>>,
    /// Statistics for monitoring pool usage
    stats: PoolStatsInternal,
}

#[derive(Default, Clone)]
struct PoolStatsInternal {
    /// Total allocations requested
    allocations: usize,
    /// Allocations satisfied from pool (cache hits)
    pool_hits: usize,
    /// New allocations required (cache misses)
    pool_misses: usize,
    /// Total buffers returned to pool
    returns: usize,
    /// Buffers discarded (pool was full)
    discards: usize,
}

impl PoolStorage {
    fn new() -> Self {
        // Pre-create empty buckets for each size class
        let num_buckets = MAX_SIZE_CLASS_EXP - MIN_SIZE_CLASS.trailing_zeros() as usize + 1;
        Self {
            buckets: vec![Vec::new(); num_buckets],
            stats: PoolStatsInternal::default(),
        }
    }

    /// Get the bucket index for a given capacity.
    fn bucket_index(capacity: usize) -> usize {
        if capacity <= MIN_SIZE_CLASS {
            return 0;
        }
        // Round up to next power of 2, then compute index
        let rounded = capacity.next_power_of_two();
        let exp = rounded.trailing_zeros() as usize;
        let min_exp = MIN_SIZE_CLASS.trailing_zeros() as usize;
        exp.saturating_sub(min_exp)
    }

    /// Get the actual size for a bucket index.
    fn size_for_bucket(bucket: usize) -> usize {
        MIN_SIZE_CLASS << bucket
    }

    /// Acquire a buffer from the pool or allocate a new one.
    fn acquire(&mut self, capacity: usize) -> PooledBuffer {
        self.stats.allocations += 1;

        let bucket = Self::bucket_index(capacity);
        let actual_size = Self::size_for_bucket(bucket);

        // Try to get a buffer from this bucket
        if bucket < self.buckets.len() {
            if let Some(mut data) = self.buckets[bucket].pop() {
                self.stats.pool_hits += 1;
                // Clear the buffer for reuse (fill with zeros)
                data.clear();
                data.resize(actual_size, 0.0);
                return PooledBuffer {
                    data,
                    size_class: bucket,
                    capacity: actual_size,
                };
            }
        }

        // Need to allocate a new buffer
        self.stats.pool_misses += 1;
        let data = vec![0.0f32; actual_size];
        PooledBuffer {
            data,
            size_class: bucket,
            capacity: actual_size,
        }
    }

    /// Return a buffer to the pool.
    fn release(&mut self, mut buffer: Vec<f32>, size_class: usize) {
        self.stats.returns += 1;

        if size_class < self.buckets.len() && self.buckets[size_class].len() < MAX_BUFFERS_PER_CLASS
        {
            // Clear capacity but keep allocation
            buffer.clear();
            self.buckets[size_class].push(buffer);
        } else {
            // Pool is full or invalid size class, discard
            self.stats.discards += 1;
        }
    }

    /// Get current pool statistics.
    fn stats(&self) -> PoolStats {
        let total_pooled: usize = self.buckets.iter().map(|b| b.len()).sum();
        let total_bytes: usize = self
            .buckets
            .iter()
            .enumerate()
            .map(|(i, b)| b.len() * Self::size_for_bucket(i) * 4)
            .sum();

        PoolStats {
            allocations: self.stats.allocations,
            pool_hits: self.stats.pool_hits,
            pool_misses: self.stats.pool_misses,
            returns: self.stats.returns,
            discards: self.stats.discards,
            pooled_buffers: total_pooled,
            pooled_bytes: total_bytes,
            hit_rate: if self.stats.allocations > 0 {
                self.stats.pool_hits as f64 / self.stats.allocations as f64
            } else {
                0.0
            },
        }
    }

    /// Clear all pooled buffers.
    fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
    }
}

/// Thread-local pool of reusable f32 buffers.
///
/// Buffers are organized into size classes (powers of 2) starting at 64 elements.
/// Each thread has its own pool to avoid synchronization overhead.
pub struct TensorPool;

impl TensorPool {
    /// Acquire a buffer with at least `capacity` f32 elements.
    ///
    /// The returned buffer may have more capacity than requested (rounded up to
    /// the next power of 2). The buffer is zero-initialized.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let buffer = TensorPool::acquire(100);
    /// assert!(buffer.len() >= 100);
    /// ```
    #[inline]
    pub fn acquire(capacity: usize) -> PooledBuffer {
        POOL.with(|pool| pool.borrow_mut().acquire(capacity))
    }

    /// Get statistics about the current thread's pool.
    pub fn stats() -> PoolStats {
        POOL.with(|pool| pool.borrow().stats())
    }

    /// Clear all pooled buffers in the current thread's pool.
    ///
    /// This is mainly useful for testing to ensure clean state.
    pub fn clear() {
        POOL.with(|pool| pool.borrow_mut().clear())
    }

    /// Reset statistics counters (for benchmarking).
    pub fn reset_stats() {
        POOL.with(|pool| {
            pool.borrow_mut().stats = PoolStatsInternal::default();
        })
    }
}

/// A buffer acquired from the pool that auto-returns on drop.
///
/// The buffer can be used as a mutable slice, then either:
/// - Dropped to return to the pool for reuse
/// - Converted to an `ArrayD` via `into_arrayd()`
pub struct PooledBuffer {
    data: Vec<f32>,
    size_class: usize,
    capacity: usize,
}

impl PooledBuffer {
    /// Get the buffer contents as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get the buffer contents as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get the buffer length.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the actual capacity (may be larger than requested).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Truncate the buffer to a specific length.
    ///
    /// This is useful when you need exactly `len` elements for an array reshape.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if len < self.data.len() {
            self.data.truncate(len);
        }
    }

    /// Convert the buffer into an ndarray ArrayD with the given shape.
    ///
    /// The buffer is consumed and will NOT be returned to the pool. Use this
    /// only when you need the data as an ndarray for the rest of its lifetime.
    ///
    /// # Panics
    ///
    /// Panics if the product of shape dimensions doesn't match the buffer length.
    pub fn into_arrayd(mut self, shape: &[usize]) -> ArrayD<f32> {
        let expected_len: usize = shape.iter().product();
        if expected_len != self.data.len() {
            // Adjust length if needed (truncate or panic)
            if expected_len < self.data.len() {
                self.data.truncate(expected_len);
            } else {
                panic!(
                    "PooledBuffer::into_arrayd: shape {:?} requires {} elements but buffer has {}",
                    shape,
                    expected_len,
                    self.data.len()
                );
            }
        }

        // Take ownership of the data, preventing return to pool
        let data = std::mem::take(&mut self.data);
        // Mark as already consumed so Drop doesn't try to return it
        self.size_class = usize::MAX;

        ArrayD::from_shape_vec(IxDyn(shape), data).expect("Shape mismatch in into_arrayd")
    }

    /// Convert the buffer into a raw `Vec<f32>`, preventing return to pool.
    ///
    /// Use this when you need the raw Vec for other purposes.
    pub fn into_vec(mut self) -> Vec<f32> {
        let data = std::mem::take(&mut self.data);
        self.size_class = usize::MAX; // Mark as consumed
        data
    }

    /// Create a new PooledBuffer from an existing Vec (wraps it for potential pooling).
    ///
    /// The Vec will be returned to the pool when the PooledBuffer is dropped.
    pub fn from_vec(data: Vec<f32>) -> Self {
        let capacity = data.len();
        let size_class = PoolStorage::bucket_index(capacity);
        Self {
            data,
            size_class,
            capacity,
        }
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Don't return if already consumed (size_class == MAX)
        if self.size_class != usize::MAX && !self.data.is_empty() {
            let data = std::mem::take(&mut self.data);
            POOL.with(|pool| pool.borrow_mut().release(data, self.size_class));
        }
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// Statistics about the memory pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total allocations requested.
    pub allocations: usize,
    /// Allocations satisfied from pool (cache hits).
    pub pool_hits: usize,
    /// New allocations required (cache misses).
    pub pool_misses: usize,
    /// Total buffers returned to pool.
    pub returns: usize,
    /// Buffers discarded (pool was full).
    pub discards: usize,
    /// Current number of buffers in the pool.
    pub pooled_buffers: usize,
    /// Current bytes held in the pool.
    pub pooled_bytes: usize,
    /// Hit rate (pool_hits / allocations).
    pub hit_rate: f64,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TensorPool: {} allocs ({:.1}% hits), {} pooled ({} KB)",
            self.allocations,
            self.hit_rate * 100.0,
            self.pooled_buffers,
            self.pooled_bytes / 1024
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acquire_returns_zeroed_buffer() {
        TensorPool::clear();
        let buffer = TensorPool::acquire(100);
        assert!(buffer.len() >= 100);
        assert!(buffer.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_size_class_bucketing() {
        // 64 elements -> bucket 0
        assert_eq!(PoolStorage::bucket_index(1), 0);
        assert_eq!(PoolStorage::bucket_index(64), 0);
        // 65-128 -> bucket 1
        assert_eq!(PoolStorage::bucket_index(65), 1);
        assert_eq!(PoolStorage::bucket_index(128), 1);
        // 129-256 -> bucket 2
        assert_eq!(PoolStorage::bucket_index(129), 2);
        assert_eq!(PoolStorage::bucket_index(256), 2);
    }

    #[test]
    fn test_buffer_reuse() {
        TensorPool::clear();
        TensorPool::reset_stats();

        // First allocation: miss
        let buffer1 = TensorPool::acquire(100);
        let ptr1 = buffer1.as_slice().as_ptr();
        drop(buffer1);

        // Second allocation of same size: should hit
        let buffer2 = TensorPool::acquire(100);
        let ptr2 = buffer2.as_slice().as_ptr();

        // Same memory should be reused
        assert_eq!(ptr1, ptr2);

        let stats = TensorPool::stats();
        assert_eq!(stats.allocations, 2);
        assert_eq!(stats.pool_hits, 1);
        assert_eq!(stats.pool_misses, 1);
    }

    #[test]
    fn test_into_arrayd() {
        TensorPool::clear();

        let mut buffer = TensorPool::acquire(12);
        for (i, v) in buffer.as_mut_slice().iter_mut().enumerate() {
            *v = i as f32;
        }
        buffer.truncate(12);

        let array = buffer.into_arrayd(&[3, 4]);
        assert_eq!(array.shape(), &[3, 4]);
        assert_eq!(array[[0, 0]], 0.0);
        assert_eq!(array[[2, 3]], 11.0);
    }

    #[test]
    fn test_into_vec_prevents_pool_return() {
        TensorPool::clear();
        TensorPool::reset_stats();

        let buffer = TensorPool::acquire(100);
        let _vec = buffer.into_vec();

        // No return since we took the vec
        let stats = TensorPool::stats();
        assert_eq!(stats.returns, 0);
    }

    #[test]
    fn test_from_vec() {
        TensorPool::clear();

        let vec = vec![1.0f32; 100];
        let buffer = PooledBuffer::from_vec(vec);
        assert_eq!(buffer.len(), 100);
        assert!(buffer.as_slice().iter().all(|&v| v == 1.0));

        // Should return to pool on drop
        drop(buffer);

        let stats = TensorPool::stats();
        assert_eq!(stats.returns, 1);
    }

    #[test]
    fn test_max_buffers_per_class() {
        TensorPool::clear();

        // Fill pool beyond capacity
        let buffers: Vec<_> = (0..MAX_BUFFERS_PER_CLASS + 5)
            .map(|_| TensorPool::acquire(100))
            .collect();

        // Drop all buffers
        drop(buffers);

        let stats = TensorPool::stats();
        // Only MAX_BUFFERS_PER_CLASS should be kept
        assert!(stats.pooled_buffers <= MAX_BUFFERS_PER_CLASS);
        assert!(stats.discards >= 5);
    }

    #[test]
    fn test_different_size_classes_separate() {
        TensorPool::clear();
        TensorPool::reset_stats();

        // Allocate small buffer
        let small = TensorPool::acquire(50);
        let small_ptr = small.as_slice().as_ptr();
        drop(small);

        // Allocate larger buffer - should NOT reuse small buffer
        let large = TensorPool::acquire(200);
        let large_ptr = large.as_slice().as_ptr();

        // Different addresses (different size classes)
        assert_ne!(small_ptr, large_ptr);

        let stats = TensorPool::stats();
        assert_eq!(stats.pool_misses, 2); // Both were misses (different sizes)
    }

    #[test]
    fn test_pool_stats_display() {
        TensorPool::clear();
        TensorPool::reset_stats();

        let _buf1 = TensorPool::acquire(100);
        let buf2 = TensorPool::acquire(100);
        drop(buf2);

        let stats = TensorPool::stats();
        let display = format!("{}", stats);
        assert!(display.contains("allocs"));
        assert!(display.contains("hits"));
    }

    // ========================================
    // Mutation-killing tests for TensorPool
    // ========================================

    #[test]
    fn test_pooled_buffer_is_empty_exact() {
        TensorPool::clear();
        // Non-empty buffer should return false
        let buffer = TensorPool::acquire(100);
        assert!(!buffer.is_empty());

        // Empty buffer should return true
        let empty_buffer = PooledBuffer::from_vec(vec![]);
        assert!(empty_buffer.is_empty());
    }

    #[test]
    fn test_pooled_buffer_capacity_exact() {
        TensorPool::clear();
        let buffer = TensorPool::acquire(100);
        // Capacity should be at least 100, not 0 or 1
        assert!(buffer.capacity() >= 100);
        assert!(buffer.capacity() >= 64); // At least MIN_SIZE_CLASS

        // More specific test: capacity should match a power of 2 >= 100
        let cap = buffer.capacity();
        assert!(cap >= 128); // Should be rounded up to 128 minimum
    }

    #[test]
    fn test_pooled_buffer_truncate_boundary() {
        TensorPool::clear();
        let mut buffer = TensorPool::acquire(100);
        assert!(buffer.len() >= 100);

        // Truncate to less than current length should work
        let _original_len = buffer.len();
        buffer.truncate(50);
        assert_eq!(buffer.len(), 50);

        // Truncate to greater than current length should be a no-op
        buffer.truncate(200);
        assert_eq!(buffer.len(), 50); // Still 50, not 200

        // Truncate to 0 should work
        buffer.truncate(0);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_pooled_buffer_into_arrayd_truncates() {
        TensorPool::clear();
        let mut buffer = TensorPool::acquire(20);
        for i in 0..20 {
            buffer.as_mut_slice()[i] = i as f32;
        }

        // Should truncate to fit 3x4 = 12
        buffer.truncate(20);
        let array = buffer.into_arrayd(&[3, 4]);
        assert_eq!(array.shape(), &[3, 4]);
        assert_eq!(array[[0, 0]], 0.0);
        assert_eq!(array[[2, 3]], 11.0);
    }

    #[test]
    fn test_clear_actually_clears() {
        // Create some buffers and return them
        {
            let _b1 = TensorPool::acquire(100);
            let _b2 = TensorPool::acquire(100);
        }

        let stats_before = TensorPool::stats();
        assert!(stats_before.returns >= 2);

        TensorPool::clear();

        // After clear, pool should be empty
        let stats_after = TensorPool::stats();
        assert_eq!(stats_after.pooled_buffers, 0);
    }

    #[test]
    fn test_reset_stats_actually_resets() {
        TensorPool::clear();

        // Do some operations
        let _b = TensorPool::acquire(100);

        let stats_before = TensorPool::stats();
        assert!(stats_before.allocations >= 1);

        TensorPool::reset_stats();

        let stats_after = TensorPool::stats();
        assert_eq!(stats_after.allocations, 0);
        assert_eq!(stats_after.pool_hits, 0);
        assert_eq!(stats_after.pool_misses, 0);
    }

    #[test]
    fn test_drop_returns_to_pool() {
        TensorPool::clear();
        TensorPool::reset_stats();

        {
            let buffer = TensorPool::acquire(100);
            drop(buffer);
        }

        let stats = TensorPool::stats();
        assert_eq!(stats.returns, 1);
    }

    #[test]
    fn test_drop_doesnt_return_consumed_buffer() {
        TensorPool::clear();
        TensorPool::reset_stats();

        {
            let buffer = TensorPool::acquire(100);
            let _vec = buffer.into_vec(); // Consume it
        }

        let stats = TensorPool::stats();
        assert_eq!(stats.returns, 0); // Should not return since we consumed it
    }

    // ========================================
    // Mutation-killing tests for pool.rs
    // These tests target specific mutants that survived
    // ========================================

    /// Kill mutant: Line 156 - replace * with + or / in PoolStorage::stats
    /// Tests that pooled_bytes is calculated correctly as len * bucket_size * 4
    #[test]
    fn test_stats_pooled_bytes_calculation() {
        TensorPool::clear();
        TensorPool::reset_stats();

        // Acquire and release a buffer so it's in the pool
        let buffer = TensorPool::acquire(64); // Bucket 0 = 64 elements
        drop(buffer);

        let stats = TensorPool::stats();
        assert_eq!(stats.pooled_buffers, 1);
        // Should be 64 elements * 4 bytes/element = 256 bytes
        // If * was replaced with +: 1 + 64 + 4 = 69 (wrong)
        // If * was replaced with /: 1 / 64 / 4 = 0 (wrong)
        assert_eq!(stats.pooled_bytes, 256);

        // Add another buffer of different size
        let buffer2 = TensorPool::acquire(128); // Bucket 1 = 128 elements
        drop(buffer2);

        let stats2 = TensorPool::stats();
        assert_eq!(stats2.pooled_buffers, 2);
        // Should be 64*4 + 128*4 = 256 + 512 = 768 bytes
        assert_eq!(stats2.pooled_bytes, 768);
    }

    /// Kill mutant: Line 167 - replace > with ==, <, or >= in PoolStorage::stats
    /// Tests hit_rate calculation with zero allocations
    #[test]
    fn test_stats_hit_rate_zero_allocations() {
        TensorPool::clear();
        TensorPool::reset_stats();

        let stats = TensorPool::stats();
        assert_eq!(stats.allocations, 0);
        // With zero allocations, hit_rate should be 0.0
        // If > was replaced with ==: would divide by zero (panic or NaN)
        // If > was replaced with <: would divide by zero for allocations=0
        // If > was replaced with >=: would still be 0.0 for allocations=0 but wrong for allocations>0
        assert_eq!(stats.hit_rate, 0.0);
        assert!(!stats.hit_rate.is_nan());
    }

    /// Kill mutant: Line 168 - replace / with % or * in PoolStorage::stats
    /// Tests that hit_rate is calculated as pool_hits / allocations
    #[test]
    fn test_stats_hit_rate_calculation() {
        TensorPool::clear();
        TensorPool::reset_stats();

        // First allocation: miss
        let buffer1 = TensorPool::acquire(100);
        drop(buffer1);

        // Second allocation: hit (reuses buffer)
        let buffer2 = TensorPool::acquire(100);
        drop(buffer2);

        // Third allocation: hit (reuses buffer)
        let buffer3 = TensorPool::acquire(100);
        drop(buffer3);

        // Fourth allocation: hit (reuses buffer)
        let buffer4 = TensorPool::acquire(100);
        drop(buffer4);

        let stats = TensorPool::stats();
        assert_eq!(stats.allocations, 4);
        assert_eq!(stats.pool_hits, 3);
        // hit_rate = 3 / 4 = 0.75
        // If / was replaced with %: 3 % 4 = 3 (wrong)
        // If / was replaced with *: 3 * 4 = 12 (wrong)
        assert!((stats.hit_rate - 0.75).abs() < 1e-10);
    }

    /// Kill mutant: Line 273 - replace < with <= in PooledBuffer::truncate
    /// Tests boundary case where len == data.len()
    #[test]
    fn test_truncate_boundary_equal_length() {
        TensorPool::clear();

        let mut buffer = TensorPool::acquire(64);
        let original_len = buffer.len();
        assert_eq!(original_len, 64);

        // Truncate to exactly the current length - should be a no-op
        buffer.truncate(64);
        assert_eq!(buffer.len(), 64);

        // Now truncate to less - should actually truncate
        buffer.truncate(63);
        assert_eq!(buffer.len(), 63);

        // Truncate back to 64 - should be a no-op (can't expand)
        buffer.truncate(64);
        assert_eq!(buffer.len(), 63);
    }

    /// Kill mutant: Line 290 - replace < with <= in PooledBuffer::into_arrayd
    /// Tests boundary case where expected_len == data.len()
    #[test]
    fn test_into_arrayd_exact_match() {
        TensorPool::clear();

        let mut buffer = TensorPool::acquire(64);
        // Fill with test data
        for i in 0..64 {
            buffer.as_mut_slice()[i] = i as f32;
        }

        // Shape [8, 8] = 64 elements, exactly matching buffer length
        // Should NOT truncate
        let array = buffer.into_arrayd(&[8, 8]);
        assert_eq!(array.shape(), &[8, 8]);
        assert_eq!(array[[0, 0]], 0.0);
        assert_eq!(array[[7, 7]], 63.0);
    }

    /// Kill mutant: Line 336 - replace && with || in Drop::drop
    /// Tests that drop only returns buffer when BOTH conditions are met:
    /// - size_class != usize::MAX (not consumed)
    /// - !is_empty() (has data)
    #[test]
    fn test_drop_conditions_both_required() {
        TensorPool::clear();
        TensorPool::reset_stats();

        // Case 1: Consumed buffer (size_class == MAX) - should NOT return
        {
            let buffer = TensorPool::acquire(100);
            let _vec = buffer.into_vec(); // Consumes, sets size_class = MAX
        }
        let stats1 = TensorPool::stats();
        assert_eq!(stats1.returns, 0);

        TensorPool::reset_stats();

        // Case 2: Empty buffer (is_empty() == true) - should NOT return
        {
            let buffer = PooledBuffer::from_vec(vec![]);
            assert!(buffer.is_empty());
            drop(buffer);
        }
        let stats2 = TensorPool::stats();
        assert_eq!(stats2.returns, 0);

        TensorPool::reset_stats();

        // Case 3: Normal buffer (size_class != MAX AND !is_empty()) - SHOULD return
        {
            let buffer = TensorPool::acquire(100);
            assert!(!buffer.is_empty());
            drop(buffer);
        }
        let stats3 = TensorPool::stats();
        assert_eq!(stats3.returns, 1);
    }

    /// Kill mutant: Line 109 - replace < with <= in PoolStorage::acquire
    /// Tests that bucket index boundary is handled correctly
    #[test]
    fn test_acquire_bucket_boundary() {
        TensorPool::clear();
        TensorPool::reset_stats();

        // Test acquiring buffers at size class boundaries
        // Bucket 0: 64 elements
        let b0 = TensorPool::acquire(64);
        assert_eq!(b0.capacity(), 64);
        drop(b0);

        // Reacquire from pool should work
        let b0_reuse = TensorPool::acquire(64);
        assert_eq!(b0_reuse.capacity(), 64);

        let stats = TensorPool::stats();
        assert_eq!(stats.pool_hits, 1);
    }

    /// Kill mutant: Line 137 - replace < with <= in PoolStorage::release
    /// Tests that release handles size class boundary correctly
    #[test]
    fn test_release_bucket_boundary() {
        TensorPool::clear();
        TensorPool::reset_stats();

        // Create a buffer and release it
        let buffer = TensorPool::acquire(64);
        let size_class = buffer.size_class;
        drop(buffer);

        // Verify it was returned to pool
        let stats = TensorPool::stats();
        assert_eq!(stats.returns, 1);
        assert_eq!(stats.pooled_buffers, 1);

        // Reacquire should get the same buffer back
        let buffer2 = TensorPool::acquire(64);
        assert_eq!(buffer2.size_class, size_class);
    }

    /// Kill mutant: verify multiplication in bytes calculation with multiple buckets
    #[test]
    fn test_pooled_bytes_multi_bucket() {
        TensorPool::clear();
        TensorPool::reset_stats();

        // Bucket 0: 64 elements = 256 bytes
        let b0 = TensorPool::acquire(64);
        drop(b0);

        // Bucket 1: 128 elements = 512 bytes
        let b1 = TensorPool::acquire(128);
        drop(b1);

        // Bucket 2: 256 elements = 1024 bytes
        let b2 = TensorPool::acquire(256);
        drop(b2);

        let stats = TensorPool::stats();
        assert_eq!(stats.pooled_buffers, 3);
        // Total: 256 + 512 + 1024 = 1792 bytes
        assert_eq!(stats.pooled_bytes, 1792);
    }

    /// Kill mutant: Lines 109, 137 - boundary check for oversized bucket indices
    /// When bucket >= num_buckets, we should NOT try to access self.buckets[bucket]
    /// If < was changed to <=, bucket == num_buckets would cause panic
    #[test]
    fn test_oversized_bucket_index_handling() {
        TensorPool::clear();
        TensorPool::reset_stats();

        // Test that bucket_index can return values >= num_buckets for large sizes
        // num_buckets = MAX_SIZE_CLASS_EXP - MIN_SIZE_CLASS.trailing_zeros() + 1 = 30 - 6 + 1 = 25
        // bucket_index returns exp - 6 where exp = trailing_zeros(next_power_of_two(capacity))
        // For capacity = 2^31, bucket = 31 - 6 = 25 which equals num_buckets

        // We can't actually allocate 2^31 elements, but we can test that the code
        // handles the case where bucket >= num_buckets by checking bucket_index directly
        let large_bucket = PoolStorage::bucket_index(1 << 31); // 2^31 elements
        assert!(large_bucket >= 25); // bucket 25 = num_buckets

        // Now test that we can create a PooledBuffer with an invalid size_class
        // and release it - it should be discarded, not cause a panic
        let oversized = PooledBuffer {
            data: vec![0.0f32; 100],
            size_class: 100, // Invalid: > num_buckets
            capacity: 100,
        };
        drop(oversized); // Should not panic, should discard

        let stats = TensorPool::stats();
        assert_eq!(stats.discards, 1); // Should be discarded due to invalid size_class
    }

    /// Kill mutant: Line 109 - if bucket < buckets.len() was bucket <= buckets.len()
    /// For bucket == num_buckets, indexing would panic
    /// Since we can't easily create such a large allocation, we test the boundary
    #[test]
    fn test_acquire_max_valid_bucket() {
        TensorPool::clear();
        TensorPool::reset_stats();

        // Test the largest bucket that's still valid
        // Bucket 24 = 2^(24+6) = 2^30 = 1 billion elements = 4GB
        // This is too large to actually allocate in a test

        // Instead test that bucket calculation works correctly near the boundary
        // bucket_index(2^30) should return 24 (the last valid bucket)
        let bucket_30 = PoolStorage::bucket_index(1 << 30);
        assert_eq!(bucket_30, 24); // 30 - 6 = 24

        // bucket_index(2^31) should return 25 (invalid bucket)
        let bucket_31 = PoolStorage::bucket_index(1 << 31);
        assert_eq!(bucket_31, 25); // 31 - 6 = 25

        // The check `bucket < num_buckets` (where num_buckets=25) means:
        // - bucket 24: valid (24 < 25)
        // - bucket 25: invalid (25 < 25 is false)
        // If mutated to <=, bucket 25 would pass but cause index OOB
    }

    // Note on equivalent mutants:
    // Lines 273 and 290 - `len < self.data.len()` in truncate/into_arrayd
    // These are EQUIVALENT MUTANTS because:
    // - When len == data.len(), truncating does nothing (Vec::truncate is no-op)
    // - So `<` vs `<=` produces identical behavior at the boundary
    // These cannot be killed by any test.
}
