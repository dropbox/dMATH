//! Scalable fingerprint storage using memory-mapped files.
//!
//! This module provides `MmapFingerprintSet`, a fingerprint storage mechanism
//! that uses memory-mapped files to handle state spaces larger than available RAM.
//!
//! # Algorithm
//!
//! Uses open addressing with linear probing:
//! - Each slot stores a 64-bit fingerprint
//! - Empty slots contain the value `0` (fingerprint 0 is special-cased)
//! - On collision, probe linearly up to `MAX_PROBE` slots
//!
//! # Thread Safety
//!
//! The implementation uses atomic operations for concurrent access:
//! - `insert()` uses compare-and-swap to safely insert
//! - `contains()` uses atomic load with acquire ordering
//! - Multiple threads can safely insert/query simultaneously
//!
//! # Memory Mapping
//!
//! Storage can be either:
//! - **Anonymous**: In-memory only (no file, limited to RAM)
//! - **File-backed**: Uses a temporary file, allowing the OS to page to disk

use crate::state::Fingerprint;
use memmap2::MmapMut;
use rustc_hash::{FxHashSet, FxHasher};
use std::hash::BuildHasherDefault;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// FxHasher-based BuildHasher for faster hashing of Fingerprint keys.
type FxBuildHasher = BuildHasherDefault<FxHasher>;
use tempfile::NamedTempFile;

/// Maximum number of probes before giving up on insert.
/// If exceeded, the table is too full and should be resized.
const MAX_PROBE: usize = 1024;

/// Sentinel value for empty slots.
/// Fingerprint 0 is handled specially (stored as u64::MAX).
const EMPTY: u64 = 0;

/// Special encoding for fingerprint value 0 (since 0 is our empty sentinel).
const FP_ZERO_ENCODING: u64 = u64::MAX;

/// Encode a fingerprint for storage.
/// Fingerprint 0 is special-cased since 0 is our empty sentinel.
#[inline]
fn encode_fingerprint(fp: Fingerprint) -> u64 {
    if fp.0 == 0 {
        FP_ZERO_ENCODING
    } else {
        fp.0
    }
}

/// Decode a stored value back to its fingerprint.
/// Reverses the encoding done by `encode_fingerprint`.
#[inline]
fn decode_fingerprint(encoded: u64) -> u64 {
    if encoded == FP_ZERO_ENCODING {
        0
    } else {
        encoded
    }
}

/// Memory-mapped fingerprint set for scalable state space exploration.
///
/// This data structure stores fingerprints in a memory-mapped array,
/// allowing it to handle state spaces that exceed available RAM by
/// letting the OS page data to disk as needed.
///
/// # Example
///
/// ```ignore
/// use tla_check::storage::MmapFingerprintSet;
///
/// // Create a set with capacity for 1M fingerprints
/// let set = MmapFingerprintSet::new(1_000_000, None).unwrap();
///
/// let fp = 12345678u64;
/// assert!(!set.contains(fp));
/// assert!(set.insert(fp));  // Returns true - was new
/// assert!(set.contains(fp));
/// assert!(!set.insert(fp)); // Returns false - already present
/// ```
pub struct MmapFingerprintSet {
    /// The memory-mapped array of fingerprints
    mmap: MmapMut,
    /// Capacity (number of slots)
    capacity: usize,
    /// Number of entries (approximate, may lag slightly in concurrent use)
    count: AtomicUsize,
    /// Backing file (kept open to prevent deletion)
    #[allow(dead_code)]
    backing_file: Option<NamedTempFile>,
    /// Load factor threshold (when count/capacity exceeds this, table is "full")
    max_load_factor: f64,
    /// Error flag - set when an insert fails due to table overflow
    has_error: AtomicBool,
    /// Number of fingerprints that were dropped due to table overflow
    dropped_count: AtomicUsize,
}

impl MmapFingerprintSet {
    /// Create a new memory-mapped fingerprint set.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of fingerprints to store. This determines
    ///   the size of the memory-mapped region (capacity * 8 bytes).
    /// * `backing_dir` - If `Some(path)`, create a file-backed mapping in this
    ///   directory. If `None`, use anonymous mapping (in-memory only).
    ///
    /// # Returns
    ///
    /// Returns the new set, or an I/O error if mapping fails.
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0.
    pub fn new(capacity: usize, backing_dir: Option<PathBuf>) -> io::Result<Self> {
        assert!(capacity > 0, "capacity must be non-zero");

        let byte_size = capacity * 8; // 8 bytes per u64

        let (mmap, backing_file) = if let Some(dir) = backing_dir {
            // File-backed mapping
            let file = NamedTempFile::new_in(dir)?;
            file.as_file().set_len(byte_size as u64)?;
            let mmap = unsafe { MmapMut::map_mut(file.as_file())? };
            (mmap, Some(file))
        } else {
            // Anonymous mapping (in-memory)
            let mmap = MmapMut::map_anon(byte_size)?;
            (mmap, None)
        };

        Ok(MmapFingerprintSet {
            mmap,
            capacity,
            count: AtomicUsize::new(0),
            backing_file,
            max_load_factor: 0.75,
            has_error: AtomicBool::new(false),
            dropped_count: AtomicUsize::new(0),
        })
    }

    /// Create a new set with custom load factor.
    ///
    /// The load factor determines when the table is considered "full".
    /// Default is 0.75 (75% full).
    pub fn with_load_factor(
        capacity: usize,
        backing_dir: Option<PathBuf>,
        load_factor: f64,
    ) -> io::Result<Self> {
        let mut set = Self::new(capacity, backing_dir)?;
        set.max_load_factor = load_factor;
        Ok(set)
    }

    /// Get a reference to the slot at the given index as an atomic u64.
    ///
    /// # Safety
    ///
    /// The index must be in bounds (0..capacity).
    #[inline]
    fn slot(&self, index: usize) -> &AtomicU64 {
        debug_assert!(index < self.capacity);
        let ptr = self.mmap.as_ptr() as *const AtomicU64;
        // SAFETY: The mmap is properly aligned for u64, and index is in bounds.
        unsafe { &*ptr.add(index) }
    }

    /// Compute the primary hash index for a fingerprint.
    #[inline]
    fn hash_index(&self, fp: Fingerprint) -> usize {
        // Use the fingerprint itself as a hash (it's already a hash).
        // Apply a secondary scramble to improve distribution.
        let h = fp.0.wrapping_mul(0x9E3779B97F4A7C15); // Golden ratio constant
        (h as usize) % self.capacity
    }

    /// Insert a fingerprint into the set.
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if the fingerprint was newly inserted
    /// - `Ok(false)` if the fingerprint was already present
    /// - `Err(...)` if the table is too full (load factor exceeded)
    ///
    /// # Thread Safety
    ///
    /// This method is safe to call from multiple threads simultaneously.
    pub fn insert(&self, fp: Fingerprint) -> Result<bool, MmapError> {
        // Check load factor before inserting
        let current_count = self.count.load(Ordering::Relaxed);
        if current_count as f64 / self.capacity as f64 >= self.max_load_factor {
            return Err(MmapError::TableFull {
                count: current_count,
                capacity: self.capacity,
            });
        }

        let encoded = encode_fingerprint(fp);
        let start_index = self.hash_index(fp);

        for probe in 0..MAX_PROBE {
            let index = (start_index + probe) % self.capacity;
            let slot = self.slot(index);

            // Try to read the current value
            let current = slot.load(Ordering::Acquire);

            if current == encoded {
                // Already present
                return Ok(false);
            }

            if current == EMPTY {
                // Try to insert via CAS
                match slot.compare_exchange(EMPTY, encoded, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => {
                        // Successfully inserted
                        self.count.fetch_add(1, Ordering::Relaxed);
                        return Ok(true);
                    }
                    Err(actual) => {
                        // Someone else wrote here
                        if actual == encoded {
                            // They wrote the same fingerprint - already present
                            return Ok(false);
                        }
                        // Different fingerprint - continue probing
                    }
                }
            }
            // Slot occupied by different fingerprint - continue probing
        }

        // Exceeded max probes
        Err(MmapError::ProbeExceeded {
            fp,
            probes: MAX_PROBE,
        })
    }

    /// Check if a fingerprint is present in the set.
    ///
    /// # Thread Safety
    ///
    /// This method is safe to call from multiple threads simultaneously.
    pub fn contains(&self, fp: Fingerprint) -> bool {
        let encoded = encode_fingerprint(fp);
        let start_index = self.hash_index(fp);

        for probe in 0..MAX_PROBE {
            let index = (start_index + probe) % self.capacity;
            let slot = self.slot(index);
            let current = slot.load(Ordering::Acquire);

            if current == encoded {
                return true;
            }

            if current == EMPTY {
                // Empty slot - fingerprint not present
                return false;
            }
            // Different fingerprint - continue probing
        }

        // Exceeded max probes without finding - not present
        false
    }

    /// Return the number of fingerprints stored.
    ///
    /// Note: In concurrent usage, this may be slightly stale.
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the capacity (maximum number of fingerprints).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return the current load factor (count / capacity).
    pub fn load_factor(&self) -> f64 {
        self.len() as f64 / self.capacity as f64
    }

    /// Flush any pending writes to disk (for file-backed mappings).
    ///
    /// This is useful before checkpointing.
    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }

    /// Flush asynchronously (returns immediately).
    pub fn flush_async(&self) -> io::Result<()> {
        self.mmap.flush_async()
    }

    /// Check if any insert errors have occurred (table overflow).
    ///
    /// When this returns true, some fingerprints were not stored and the
    /// model checking results may be incomplete (states may have been dropped).
    pub fn has_errors(&self) -> bool {
        self.has_error.load(Ordering::Relaxed)
    }

    /// Get the count of dropped fingerprints due to table overflow.
    ///
    /// If this is non-zero, model checking results are unreliable.
    pub fn dropped_count(&self) -> usize {
        self.dropped_count.load(Ordering::Relaxed)
    }

    /// Record an insert error (table overflow or probe exceeded).
    fn record_error(&self) {
        self.has_error.store(true, Ordering::Relaxed);
        self.dropped_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Check the current capacity status.
    ///
    /// Returns `CapacityStatus::Normal` if plenty of space is available,
    /// `CapacityStatus::Warning` if approaching limit (>80% of load factor),
    /// or `CapacityStatus::Critical` if very close to limit (>95% of load factor).
    ///
    /// Use this to provide early warning to users when the fingerprint table
    /// is approaching capacity.
    pub fn capacity_status(&self) -> CapacityStatus {
        let count = self.count.load(Ordering::Relaxed);
        let usage = count as f64 / self.capacity as f64;
        let max_usage = self.max_load_factor;

        // Calculate how close we are to the load factor limit
        let usage_ratio = usage / max_usage;

        if usage_ratio >= 0.95 {
            CapacityStatus::Critical {
                count,
                capacity: self.capacity,
                usage,
            }
        } else if usage_ratio >= 0.80 {
            CapacityStatus::Warning {
                count,
                capacity: self.capacity,
                usage,
            }
        } else {
            CapacityStatus::Normal
        }
    }

    /// Get the max load factor for this set.
    pub fn max_load_factor(&self) -> f64 {
        self.max_load_factor
    }

    /// Collect all fingerprints currently stored in this set.
    ///
    /// This scans all slots and returns decoded fingerprint values.
    /// Time complexity is O(capacity), so use sparingly (e.g., during eviction).
    ///
    /// # Thread Safety
    ///
    /// This method provides a point-in-time snapshot. Concurrent inserts
    /// may or may not be included in the result.
    pub fn collect_all(&self) -> Vec<u64> {
        let mut result = Vec::with_capacity(self.count.load(Ordering::Relaxed));

        for i in 0..self.capacity {
            let slot = self.slot(i);
            let encoded = slot.load(Ordering::Acquire);

            if encoded != EMPTY {
                result.push(decode_fingerprint(encoded));
            }
        }

        result
    }

    /// Clear all entries from this set.
    ///
    /// This resets all slots to EMPTY and count to 0.
    /// Time complexity is O(capacity).
    ///
    /// # Thread Safety
    ///
    /// This method is NOT safe to call concurrently with insert/contains.
    /// Use only when exclusive access is guaranteed (e.g., during eviction
    /// when all workers are paused).
    pub fn clear(&self) {
        for i in 0..self.capacity {
            let slot = self.slot(i);
            slot.store(EMPTY, Ordering::Release);
        }
        self.count.store(0, Ordering::Relaxed);
        self.has_error.store(false, Ordering::Relaxed);
        self.dropped_count.store(0, Ordering::Relaxed);
    }
}

/// Errors that can occur during mmap fingerprint operations.
#[derive(Debug, Clone)]
pub enum MmapError {
    /// The table is too full (load factor exceeded).
    TableFull { count: usize, capacity: usize },
    /// Maximum probe distance exceeded.
    ProbeExceeded { fp: Fingerprint, probes: usize },
}

/// Capacity status for fingerprint storage.
///
/// Used to provide early warning when storage is approaching its limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CapacityStatus {
    /// Normal operation - plenty of space available.
    Normal,
    /// Warning - approaching capacity limit (>80% of load factor).
    /// Consider increasing capacity if more states are expected.
    Warning {
        /// Current number of entries.
        count: usize,
        /// Maximum capacity.
        capacity: usize,
        /// Current usage percentage (0.0 - 1.0).
        usage: f64,
    },
    /// Critical - very close to capacity limit (>95% of load factor).
    /// Inserts may start failing soon.
    Critical {
        /// Current number of entries.
        count: usize,
        /// Maximum capacity.
        capacity: usize,
        /// Current usage percentage (0.0 - 1.0).
        usage: f64,
    },
}

impl std::fmt::Display for MmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmapError::TableFull { count, capacity } => {
                write!(
                    f,
                    "fingerprint table full: {} entries, {} capacity",
                    count, capacity
                )
            }
            MmapError::ProbeExceeded { fp, probes } => {
                write!(f, "exceeded {} probes inserting fingerprint {}", probes, fp)
            }
        }
    }
}

impl std::error::Error for MmapError {}

/// A trait for fingerprint sets, allowing different implementations
/// (in-memory HashSet, DashSet, or MmapFingerprintSet).
pub trait FingerprintSet: Send + Sync {
    /// Insert a fingerprint, returning true if it was newly inserted.
    fn insert(&self, fp: Fingerprint) -> bool;

    /// Check if a fingerprint is present.
    fn contains(&self, fp: Fingerprint) -> bool;

    /// Return the number of fingerprints.
    fn len(&self) -> usize;

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if any insert errors have occurred (e.g., table overflow).
    ///
    /// When this returns true, some fingerprints may not have been stored
    /// and model checking results may be incomplete.
    ///
    /// Default implementation returns false (no errors possible).
    fn has_errors(&self) -> bool {
        false
    }

    /// Get the count of dropped fingerprints due to errors.
    ///
    /// If this is non-zero, model checking results are unreliable.
    ///
    /// Default implementation returns 0 (no errors possible).
    fn dropped_count(&self) -> usize {
        0
    }

    /// Check the current capacity status.
    ///
    /// Returns `CapacityStatus::Normal` for growable storage (like HashSet),
    /// or a warning/critical status for fixed-capacity storage approaching limits.
    ///
    /// Default implementation returns `Normal` (no capacity limit).
    fn capacity_status(&self) -> CapacityStatus {
        CapacityStatus::Normal
    }
}

impl FingerprintSet for MmapFingerprintSet {
    fn insert(&self, fp: Fingerprint) -> bool {
        match MmapFingerprintSet::insert(self, fp) {
            Ok(is_new) => is_new,
            Err(_) => {
                // Record the error so callers can detect dropped fingerprints
                self.record_error();
                false
            }
        }
    }

    fn contains(&self, fp: Fingerprint) -> bool {
        MmapFingerprintSet::contains(self, fp)
    }

    fn len(&self) -> usize {
        MmapFingerprintSet::len(self)
    }

    fn has_errors(&self) -> bool {
        MmapFingerprintSet::has_errors(self)
    }

    fn dropped_count(&self) -> usize {
        MmapFingerprintSet::dropped_count(self)
    }

    fn capacity_status(&self) -> CapacityStatus {
        MmapFingerprintSet::capacity_status(self)
    }
}

impl FingerprintSet for dashmap::DashSet<Fingerprint> {
    fn insert(&self, fp: Fingerprint) -> bool {
        dashmap::DashSet::insert(self, fp)
    }

    fn contains(&self, fp: Fingerprint) -> bool {
        dashmap::DashSet::contains(self, &fp)
    }

    fn len(&self) -> usize {
        dashmap::DashSet::len(self)
    }
}

/// FxHasher-based DashSet for faster fingerprint storage.
/// Since Fingerprint is already a 64-bit hash, FxHasher avoids redundant SipHash overhead.
impl FingerprintSet for dashmap::DashSet<Fingerprint, FxBuildHasher> {
    fn insert(&self, fp: Fingerprint) -> bool {
        dashmap::DashSet::insert(self, fp)
    }

    fn contains(&self, fp: Fingerprint) -> bool {
        dashmap::DashSet::contains(self, &fp)
    }

    fn len(&self) -> usize {
        dashmap::DashSet::len(self)
    }
}

/// Sharded fingerprint set for parallel model checking.
///
/// This implementation is similar to TLC's MultiFPSet - it partitions fingerprints
/// by their most significant bits into N shards (where N = 2^shard_bits).
/// Each shard is a separate FxHashSet protected by its own RwLock.
///
/// This provides:
/// - Good distribution: fingerprints are already hashes, so MSB distribution is uniform
/// - Low contention: with 64 shards and 8 workers, collision probability is ~12.5%
/// - Simple implementation: just an array of locked HashSets
/// - Cache-friendly: each shard can fit in L1/L2 cache when small
///
/// # Example
///
/// ```ignore
/// use tla_check::storage::ShardedFingerprintSet;
///
/// let set = ShardedFingerprintSet::new(6); // 2^6 = 64 shards
/// let fp = Fingerprint(12345678u64);
/// assert!(!set.contains(fp));
/// assert!(set.insert(fp));  // Returns true - was new
/// assert!(set.contains(fp));
/// assert!(!set.insert(fp)); // Returns false - already present
/// ```
pub struct ShardedFingerprintSet {
    /// The shards - each is a `RwLock<FxHashSet<Fingerprint>>`
    shards: Box<[parking_lot::RwLock<FxHashSet<Fingerprint>>]>,
    /// Number of bits used for sharding (log2 of shard count)
    shard_bits: u32,
    /// Mask for extracting shard index: (1 << shard_bits) - 1
    shard_mask: usize,
    /// Shift amount: 64 - shard_bits (to get MSB)
    shift: u32,
    /// Total count (approximate, for reporting)
    count: AtomicUsize,
}

impl ShardedFingerprintSet {
    /// Create a new sharded fingerprint set.
    ///
    /// # Arguments
    ///
    /// * `shard_bits` - Number of bits for sharding. Creates 2^shard_bits shards.
    ///   Recommended: 6 (64 shards) for 8-16 workers, 8 (256 shards) for 32+ workers.
    ///
    /// # Panics
    ///
    /// Panics if shard_bits is 0 or > 16.
    pub fn new(shard_bits: u32) -> Self {
        assert!(
            shard_bits > 0 && shard_bits <= 16,
            "shard_bits must be 1-16"
        );

        let num_shards = 1usize << shard_bits;
        let shards: Vec<_> = (0..num_shards)
            .map(|_| parking_lot::RwLock::new(FxHashSet::default()))
            .collect();

        ShardedFingerprintSet {
            shards: shards.into_boxed_slice(),
            shard_bits,
            shard_mask: num_shards - 1,
            shift: 64 - shard_bits,
            count: AtomicUsize::new(0),
        }
    }

    /// Create with a specific number of shards (must be power of 2).
    pub fn with_shards(num_shards: usize) -> Self {
        assert!(
            num_shards.is_power_of_two(),
            "num_shards must be power of 2"
        );
        let shard_bits = num_shards.trailing_zeros();
        Self::new(shard_bits)
    }

    /// Get the shard index for a fingerprint (using MSB like TLC's MultiFPSet).
    #[inline]
    fn shard_index(&self, fp: Fingerprint) -> usize {
        // Use MSB for partitioning (same as TLC: fp >>> fpbits)
        (fp.0 >> self.shift) as usize & self.shard_mask
    }

    /// Insert a fingerprint.
    ///
    /// Returns `true` if the fingerprint was newly inserted, `false` if already present.
    pub fn insert(&self, fp: Fingerprint) -> bool {
        let idx = self.shard_index(fp);
        // Typical TLC-style workloads generate many duplicate successors. Avoid taking an
        // exclusive write lock when the fingerprint is already present by checking under a
        // shared read lock first.
        {
            let shard = self.shards[idx].read();
            if shard.contains(&fp) {
                return false;
            }
        }

        // Racy with other inserters between the read and write locks.
        let mut shard = self.shards[idx].write();
        if shard.insert(fp) {
            self.count.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Check if a fingerprint is present.
    pub fn contains(&self, fp: Fingerprint) -> bool {
        let idx = self.shard_index(fp);
        let shard = self.shards[idx].read();
        shard.contains(&fp)
    }

    /// Return the total number of fingerprints (approximate in concurrent use).
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the number of shards.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Return the number of bits used for sharding.
    pub fn shard_bits(&self) -> u32 {
        self.shard_bits
    }
}

impl FingerprintSet for ShardedFingerprintSet {
    fn insert(&self, fp: Fingerprint) -> bool {
        ShardedFingerprintSet::insert(self, fp)
    }

    fn contains(&self, fp: Fingerprint) -> bool {
        ShardedFingerprintSet::contains(self, fp)
    }

    fn len(&self) -> usize {
        ShardedFingerprintSet::len(self)
    }
}

/// Fingerprint storage abstraction that supports different backends.
///
/// For sequential model checking, use `FingerprintStorage::InMemory`.
/// For large state spaces that exceed RAM, use `FingerprintStorage::Mmap`.
/// For billion-state specs, use `FingerprintStorage::Disk` (auto-eviction to disk).
pub enum FingerprintStorage {
    /// In-memory hash set with FxHasher (fast for u64 fingerprints, but limited to RAM).
    InMemory(std::sync::RwLock<FxHashSet<Fingerprint>>),
    /// Memory-mapped file (scales beyond RAM, slightly slower).
    Mmap(MmapFingerprintSet),
    /// Disk-backed storage with automatic eviction (for billion-state specs).
    Disk(DiskFingerprintSet),
}

impl FingerprintStorage {
    /// Create in-memory storage.
    pub fn in_memory() -> Self {
        FingerprintStorage::InMemory(std::sync::RwLock::new(FxHashSet::default()))
    }

    /// Create memory-mapped storage.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of fingerprints to store.
    /// * `backing_dir` - Optional directory for the backing file. If None,
    ///   uses anonymous mapping (in-memory, but allows using mmap semantics).
    pub fn mmap(capacity: usize, backing_dir: Option<PathBuf>) -> io::Result<Self> {
        Ok(FingerprintStorage::Mmap(MmapFingerprintSet::new(
            capacity,
            backing_dir,
        )?))
    }

    /// Create disk-backed storage with automatic eviction.
    ///
    /// # Arguments
    ///
    /// * `primary_capacity` - Size of the in-memory primary storage.
    ///   When this fills up, entries are evicted to disk.
    /// * `disk_dir` - Directory for the disk storage file.
    pub fn disk<P: Into<PathBuf>>(primary_capacity: usize, disk_dir: P) -> io::Result<Self> {
        Ok(FingerprintStorage::Disk(DiskFingerprintSet::new(
            primary_capacity,
            disk_dir,
        )?))
    }

    /// Insert a fingerprint, returning true if it was newly inserted.
    pub fn insert(&self, fp: Fingerprint) -> bool {
        match self {
            FingerprintStorage::InMemory(set) => set.write().unwrap().insert(fp),
            // Use the trait impl which tracks errors
            FingerprintStorage::Mmap(set) => FingerprintSet::insert(set, fp),
            FingerprintStorage::Disk(set) => set.insert(fp),
        }
    }

    /// Check if a fingerprint is present.
    pub fn contains(&self, fp: Fingerprint) -> bool {
        match self {
            FingerprintStorage::InMemory(set) => set.read().unwrap().contains(&fp),
            FingerprintStorage::Mmap(set) => set.contains(fp),
            FingerprintStorage::Disk(set) => set.contains(fp),
        }
    }

    /// Return the number of fingerprints.
    pub fn len(&self) -> usize {
        match self {
            FingerprintStorage::InMemory(set) => set.read().unwrap().len(),
            FingerprintStorage::Mmap(set) => set.len(),
            FingerprintStorage::Disk(set) => set.len(),
        }
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if any insert errors have occurred.
    pub fn has_errors(&self) -> bool {
        match self {
            FingerprintStorage::InMemory(_) => false,
            FingerprintStorage::Mmap(set) => set.has_errors(),
            // Disk storage handles errors via eviction, so no overflow possible
            FingerprintStorage::Disk(_) => false,
        }
    }

    /// Get the count of dropped fingerprints due to errors.
    pub fn dropped_count(&self) -> usize {
        match self {
            FingerprintStorage::InMemory(_) => 0,
            FingerprintStorage::Mmap(set) => set.dropped_count(),
            FingerprintStorage::Disk(_) => 0,
        }
    }

    /// Check the current capacity status.
    pub fn capacity_status(&self) -> CapacityStatus {
        match self {
            FingerprintStorage::InMemory(_) => CapacityStatus::Normal,
            FingerprintStorage::Mmap(set) => set.capacity_status(),
            // Disk storage is unlimited (auto-eviction), so always Normal
            FingerprintStorage::Disk(_) => CapacityStatus::Normal,
        }
    }
}

// Allow FingerprintStorage to be used where FingerprintSet is expected
impl FingerprintSet for FingerprintStorage {
    fn insert(&self, fp: Fingerprint) -> bool {
        FingerprintStorage::insert(self, fp)
    }

    fn contains(&self, fp: Fingerprint) -> bool {
        FingerprintStorage::contains(self, fp)
    }

    fn len(&self) -> usize {
        FingerprintStorage::len(self)
    }

    fn has_errors(&self) -> bool {
        FingerprintStorage::has_errors(self)
    }

    fn dropped_count(&self) -> usize {
        FingerprintStorage::dropped_count(self)
    }

    fn capacity_status(&self) -> CapacityStatus {
        FingerprintStorage::capacity_status(self)
    }
}

/// Entry size for trace locations: 16 bytes (8 bytes fingerprint + 8 bytes offset).
const TRACE_ENTRY_SIZE: usize = 16;

/// Memory-mapped trace location storage for scalable trace file mode.
///
/// This stores (fingerprint, offset) pairs for trace reconstruction.
/// Each entry maps a state's fingerprint to its location in the trace file.
///
/// # Design
///
/// Uses open addressing with linear probing:
/// - Each slot is 16 bytes: 8 bytes fingerprint + 8 bytes offset
/// - Empty slots have fingerprint = 0 (fingerprint 0 is special-cased)
/// - On collision, probe linearly up to `MAX_PROBE` slots
///
/// # Thread Safety
///
/// The implementation uses atomic operations for the fingerprint field
/// to allow concurrent reads. However, concurrent writes are not supported
/// (designed for sequential model checking mode).
///
/// # Example
///
/// ```ignore
/// use tla_check::storage::MmapTraceLocations;
/// use tla_check::state::Fingerprint;
///
/// // Create storage with capacity for 1M entries
/// let locs = MmapTraceLocations::new(1_000_000, None).unwrap();
///
/// let fp = Fingerprint(12345678);
/// assert!(!locs.contains(&fp));
/// locs.insert(fp, 1024);  // Map fingerprint to offset 1024
/// assert!(locs.contains(&fp));
/// assert_eq!(locs.get(&fp), Some(1024));
/// ```
pub struct MmapTraceLocations {
    /// The memory-mapped array of (fingerprint, offset) pairs
    mmap: MmapMut,
    /// Capacity (number of slots)
    capacity: usize,
    /// Number of entries
    count: AtomicUsize,
    /// Backing file (kept open to prevent deletion)
    #[allow(dead_code)]
    backing_file: Option<NamedTempFile>,
    /// Load factor threshold
    max_load_factor: f64,
}

impl MmapTraceLocations {
    /// Create a new memory-mapped trace location storage.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of entries. The storage will use
    ///   `capacity * 16` bytes.
    /// * `backing_dir` - If `Some(path)`, create a file-backed mapping.
    ///   If `None`, use anonymous mapping.
    ///
    /// # Returns
    ///
    /// Returns the new storage, or an I/O error if mapping fails.
    pub fn new(capacity: usize, backing_dir: Option<PathBuf>) -> io::Result<Self> {
        assert!(capacity > 0, "capacity must be non-zero");

        let byte_size = capacity * TRACE_ENTRY_SIZE;

        let (mmap, backing_file) = if let Some(dir) = backing_dir {
            // File-backed mapping
            let file = NamedTempFile::new_in(dir)?;
            file.as_file().set_len(byte_size as u64)?;
            let mmap = unsafe { MmapMut::map_mut(file.as_file())? };
            (mmap, Some(file))
        } else {
            // Anonymous mapping
            let mmap = MmapMut::map_anon(byte_size)?;
            (mmap, None)
        };

        Ok(MmapTraceLocations {
            mmap,
            capacity,
            count: AtomicUsize::new(0),
            backing_file,
            max_load_factor: 0.75,
        })
    }

    /// Create with custom load factor.
    pub fn with_load_factor(
        capacity: usize,
        backing_dir: Option<PathBuf>,
        load_factor: f64,
    ) -> io::Result<Self> {
        let mut locs = Self::new(capacity, backing_dir)?;
        locs.max_load_factor = load_factor;
        Ok(locs)
    }

    /// Get a reference to the fingerprint slot at the given index as AtomicU64.
    #[inline]
    fn fp_slot(&self, index: usize) -> &AtomicU64 {
        debug_assert!(index < self.capacity);
        let ptr = self.mmap.as_ptr() as *const AtomicU64;
        // Each entry is 16 bytes = 2 u64s. Fingerprint is first.
        unsafe { &*ptr.add(index * 2) }
    }

    /// Get a mutable reference to the offset at the given index.
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent writes to the same slot.
    #[inline]
    fn offset_slot(&self, index: usize) -> &AtomicU64 {
        debug_assert!(index < self.capacity);
        let ptr = self.mmap.as_ptr() as *const AtomicU64;
        // Each entry is 16 bytes = 2 u64s. Offset is second.
        unsafe { &*ptr.add(index * 2 + 1) }
    }

    /// Compute the primary hash index for a fingerprint.
    #[inline]
    fn hash_index(&self, fp: Fingerprint) -> usize {
        let h = fp.0.wrapping_mul(0x9E3779B97F4A7C15);
        (h as usize) % self.capacity
    }

    /// Insert a fingerprint-to-offset mapping.
    ///
    /// # Arguments
    ///
    /// * `fp` - The fingerprint to map
    /// * `offset` - The trace file offset for this fingerprint
    ///
    /// # Returns
    ///
    /// * `Ok(true)` if newly inserted
    /// * `Ok(false)` if fingerprint already present (offset is updated)
    /// * `Err(...)` if table is too full
    pub fn insert(&self, fp: Fingerprint, offset: u64) -> Result<bool, MmapError> {
        // Check load factor
        let current_count = self.count.load(Ordering::Relaxed);
        if current_count as f64 / self.capacity as f64 >= self.max_load_factor {
            return Err(MmapError::TableFull {
                count: current_count,
                capacity: self.capacity,
            });
        }

        let encoded = encode_fingerprint(fp);
        let start_index = self.hash_index(fp);

        for probe in 0..MAX_PROBE {
            let index = (start_index + probe) % self.capacity;
            let fp_slot = self.fp_slot(index);

            let current = fp_slot.load(Ordering::Acquire);

            if current == encoded {
                // Already present - update offset
                self.offset_slot(index).store(offset, Ordering::Release);
                return Ok(false);
            }

            if current == EMPTY {
                // Try to insert via CAS
                match fp_slot.compare_exchange(EMPTY, encoded, Ordering::AcqRel, Ordering::Acquire)
                {
                    Ok(_) => {
                        // Successfully claimed slot - write offset
                        self.offset_slot(index).store(offset, Ordering::Release);
                        self.count.fetch_add(1, Ordering::Relaxed);
                        return Ok(true);
                    }
                    Err(actual) => {
                        if actual == encoded {
                            // Someone else wrote same fingerprint - update offset
                            self.offset_slot(index).store(offset, Ordering::Release);
                            return Ok(false);
                        }
                        // Different fingerprint - continue probing
                    }
                }
            }
        }

        Err(MmapError::ProbeExceeded {
            fp,
            probes: MAX_PROBE,
        })
    }

    /// Get the offset for a fingerprint.
    ///
    /// # Returns
    ///
    /// `Some(offset)` if the fingerprint is present, `None` otherwise.
    pub fn get(&self, fp: &Fingerprint) -> Option<u64> {
        let encoded = encode_fingerprint(*fp);
        let start_index = self.hash_index(*fp);

        for probe in 0..MAX_PROBE {
            let index = (start_index + probe) % self.capacity;
            let fp_slot = self.fp_slot(index);
            let current = fp_slot.load(Ordering::Acquire);

            if current == encoded {
                return Some(self.offset_slot(index).load(Ordering::Acquire));
            }

            if current == EMPTY {
                return None;
            }
        }

        None
    }

    /// Check if a fingerprint is present.
    pub fn contains(&self, fp: &Fingerprint) -> bool {
        self.get(fp).is_some()
    }

    /// Return the number of entries.
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return the current load factor.
    pub fn load_factor(&self) -> f64 {
        self.len() as f64 / self.capacity as f64
    }

    /// Flush pending writes to disk.
    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }
}

/// Trait for trace location storage, allowing different implementations.
pub trait TraceLocationStorage: Send + Sync {
    /// Insert a fingerprint-to-offset mapping.
    fn insert(&self, fp: Fingerprint, offset: u64);

    /// Get the offset for a fingerprint.
    fn get(&self, fp: &Fingerprint) -> Option<u64>;

    /// Check if a fingerprint is present.
    fn contains(&self, fp: &Fingerprint) -> bool {
        self.get(fp).is_some()
    }

    /// Return the number of entries.
    fn len(&self) -> usize;

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl TraceLocationStorage for MmapTraceLocations {
    fn insert(&self, fp: Fingerprint, offset: u64) {
        let _ = MmapTraceLocations::insert(self, fp, offset);
    }

    fn get(&self, fp: &Fingerprint) -> Option<u64> {
        MmapTraceLocations::get(self, fp)
    }

    fn len(&self) -> usize {
        MmapTraceLocations::len(self)
    }
}

/// In-memory trace location storage using HashMap.
pub struct InMemoryTraceLocations {
    locations: std::sync::RwLock<std::collections::HashMap<Fingerprint, u64>>,
}

impl InMemoryTraceLocations {
    /// Create a new in-memory trace location storage.
    pub fn new() -> Self {
        InMemoryTraceLocations {
            locations: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }
}

impl Default for InMemoryTraceLocations {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceLocationStorage for InMemoryTraceLocations {
    fn insert(&self, fp: Fingerprint, offset: u64) {
        self.locations.write().unwrap().insert(fp, offset);
    }

    fn get(&self, fp: &Fingerprint) -> Option<u64> {
        self.locations.read().unwrap().get(fp).copied()
    }

    fn len(&self) -> usize {
        self.locations.read().unwrap().len()
    }
}

/// Trace location storage abstraction that supports both in-memory and scalable backends.
///
/// For small state spaces: Use `TraceLocationsStorage::InMemory`.
/// For large state spaces: Use `TraceLocationsStorage::Mmap`.
pub enum TraceLocationsStorage {
    /// In-memory hash map (fast, but limited to RAM).
    InMemory(InMemoryTraceLocations),
    /// Memory-mapped file (scales beyond RAM).
    Mmap(MmapTraceLocations),
}

impl TraceLocationsStorage {
    /// Create in-memory storage.
    pub fn in_memory() -> Self {
        TraceLocationsStorage::InMemory(InMemoryTraceLocations::new())
    }

    /// Create memory-mapped storage.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of entries.
    /// * `backing_dir` - Optional directory for the backing file.
    pub fn mmap(capacity: usize, backing_dir: Option<PathBuf>) -> io::Result<Self> {
        Ok(TraceLocationsStorage::Mmap(MmapTraceLocations::new(
            capacity,
            backing_dir,
        )?))
    }
}

impl TraceLocationStorage for TraceLocationsStorage {
    fn insert(&self, fp: Fingerprint, offset: u64) {
        match self {
            TraceLocationsStorage::InMemory(locs) => locs.insert(fp, offset),
            TraceLocationsStorage::Mmap(locs) => {
                let _ = locs.insert(fp, offset);
            }
        }
    }

    fn get(&self, fp: &Fingerprint) -> Option<u64> {
        match self {
            TraceLocationsStorage::InMemory(locs) => locs.get(fp),
            TraceLocationsStorage::Mmap(locs) => locs.get(fp),
        }
    }

    fn len(&self) -> usize {
        match self {
            TraceLocationsStorage::InMemory(locs) => locs.len(),
            TraceLocationsStorage::Mmap(locs) => locs.len(),
        }
    }
}

// ============================================================================
// DiskFingerprintSet - Two-tier storage with disk overflow
// ============================================================================

/// Number of fingerprints per disk page (8KB pages, 8 bytes per FP).
const FPS_PER_PAGE: usize = 1024;

/// Page size in bytes.
const PAGE_SIZE: usize = FPS_PER_PAGE * 8;

/// Disk-backed fingerprint set for billion-state model checking.
///
/// This implements the two-tier storage pattern from TLC's DiskFPSet:
/// - Primary: In-memory storage (MmapFingerprintSet or similar)
/// - Secondary: Sorted disk file with page index for binary search
///
/// When the primary storage fills up, fingerprints are evicted to disk
/// in sorted order. Lookups check both primary and secondary storage.
///
/// # Algorithm
///
/// **Insert (`put`):**
/// 1. Try to insert into primary storage
/// 2. If primary is full, trigger eviction
/// 3. Eviction: collect all primary FPs, sort, merge with disk, write new file
/// 4. Retry insert after eviction
///
/// **Lookup (`contains`):**
/// 1. Check primary storage
/// 2. If not found and disk file exists, do binary search on disk
///
/// # Thread Safety
///
/// The implementation supports concurrent reads and writes:
/// - Primary storage handles its own concurrency
/// - Disk reads are lock-free (immutable sorted file)
/// - Eviction uses a flag to coordinate pausing workers
///
/// # Example
///
/// ```ignore
/// use tla_check::storage::DiskFingerprintSet;
///
/// let set = DiskFingerprintSet::new(1_000_000, "/tmp/fp_storage")?;
/// set.insert(Fingerprint(12345));
/// assert!(set.contains(Fingerprint(12345)));
/// ```
pub struct DiskFingerprintSet {
    /// Primary: in-memory storage with fixed capacity.
    primary: MmapFingerprintSet,

    /// Secondary: sorted disk file (None until first eviction).
    disk_path: PathBuf,

    /// Page index: first fingerprint of each disk page for binary search.
    /// If empty, no disk file exists yet.
    page_index: parking_lot::RwLock<Vec<u64>>,

    /// Number of fingerprints stored on disk.
    disk_count: AtomicUsize,

    /// Flag indicating eviction is in progress.
    evicting: AtomicBool,

    /// Total count (primary + disk).
    total_count: AtomicUsize,

    /// Statistics: number of evictions performed.
    eviction_count: AtomicUsize,

    /// Statistics: number of disk lookups.
    disk_lookups: AtomicUsize,

    /// Statistics: number of disk hits.
    disk_hits: AtomicUsize,
}

impl DiskFingerprintSet {
    /// Create a new disk-backed fingerprint set.
    ///
    /// # Arguments
    ///
    /// * `primary_capacity` - Capacity of the in-memory primary storage.
    /// * `disk_dir` - Directory for the disk file.
    ///
    /// # Returns
    ///
    /// The new set, or an I/O error if initialization fails.
    pub fn new(primary_capacity: usize, disk_dir: impl Into<PathBuf>) -> io::Result<Self> {
        let disk_dir = disk_dir.into();
        std::fs::create_dir_all(&disk_dir)?;

        let disk_path = disk_dir.join("fingerprints.fp");

        // Use file-backed mmap for primary (allows OS paging)
        let primary = MmapFingerprintSet::new(primary_capacity, Some(disk_dir.clone()))?;

        Ok(DiskFingerprintSet {
            primary,
            disk_path,
            page_index: parking_lot::RwLock::new(Vec::new()),
            disk_count: AtomicUsize::new(0),
            evicting: AtomicBool::new(false),
            total_count: AtomicUsize::new(0),
            eviction_count: AtomicUsize::new(0),
            disk_lookups: AtomicUsize::new(0),
            disk_hits: AtomicUsize::new(0),
        })
    }

    /// Insert a fingerprint into the set.
    ///
    /// Returns `true` if the fingerprint was newly inserted, `false` if already present.
    pub fn insert(&self, fp: Fingerprint) -> bool {
        // Fast path: check if already present in primary or disk
        if self.primary.contains(fp) {
            return false;
        }
        if self.disk_lookup(fp) {
            return false;
        }

        // Try to insert into primary
        loop {
            match self.primary.insert(fp) {
                Ok(is_new) => {
                    if is_new {
                        self.total_count.fetch_add(1, Ordering::Relaxed);
                    }
                    return is_new;
                }
                Err(_) => {
                    // Primary is full - trigger eviction
                    self.evict();
                    // Retry after eviction
                }
            }
        }
    }

    /// Check if a fingerprint is present in the set.
    pub fn contains(&self, fp: Fingerprint) -> bool {
        // Check primary first (fast path)
        if self.primary.contains(fp) {
            return true;
        }

        // Check disk if we have a disk file
        self.disk_lookup(fp)
    }

    /// Lookup a fingerprint on disk using binary search.
    fn disk_lookup(&self, fp: Fingerprint) -> bool {
        let index = self.page_index.read();
        if index.is_empty() {
            return false;
        }

        self.disk_lookups.fetch_add(1, Ordering::Relaxed);

        // Binary search to find the page containing this fingerprint
        let page = match index.binary_search(&fp.0) {
            Ok(_) => {
                // Exact match on page boundary
                self.disk_hits.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            Err(0) => {
                // Before first page - not present
                return false;
            }
            Err(pos) => {
                // pos is insertion point, so page is pos - 1
                pos - 1
            }
        };

        // Check if past last page
        if page >= index.len() {
            return false;
        }

        // Read the page and search within it
        match self.search_disk_page(page, fp) {
            Ok(found) => {
                if found {
                    self.disk_hits.fetch_add(1, Ordering::Relaxed);
                }
                found
            }
            Err(_) => false,
        }
    }

    /// Search within a specific disk page for a fingerprint.
    fn search_disk_page(&self, page: usize, fp: Fingerprint) -> io::Result<bool> {
        use std::io::{Read, Seek, SeekFrom};

        let file = std::fs::File::open(&self.disk_path)?;
        let mut reader = std::io::BufReader::new(file);

        // Seek to page start
        let page_offset = (page * PAGE_SIZE) as u64;
        reader.seek(SeekFrom::Start(page_offset))?;

        // Read the page (or remaining data if last page)
        let mut buf = [0u8; PAGE_SIZE];
        let bytes_read = reader.read(&mut buf)?;
        let fps_in_page = bytes_read / 8;

        // Binary search within the page
        let fps: &[u64] = unsafe {
            std::slice::from_raw_parts(buf.as_ptr() as *const u64, fps_in_page)
        };

        Ok(fps.binary_search(&fp.0).is_ok())
    }

    /// Evict fingerprints from primary to disk.
    ///
    /// This is called when primary storage is full. It:
    /// 1. Collects all fingerprints from primary
    /// 2. Sorts them
    /// 3. Merges with existing disk file (if any)
    /// 4. Writes new sorted disk file
    /// 5. Rebuilds page index
    /// 6. Clears primary storage
    fn evict(&self) {
        // Use CAS to ensure only one thread does eviction
        if self
            .evicting
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            // Another thread is evicting - wait for it
            while self.evicting.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
            return;
        }

        // Perform eviction (under evicting flag)
        if let Err(e) = self.do_evict() {
            eprintln!("Warning: Eviction failed: {}", e);
        }

        self.evicting.store(false, Ordering::SeqCst);
    }

    /// Internal eviction logic.
    fn do_evict(&self) -> io::Result<()> {
        use std::io::{BufWriter, Write};

        self.eviction_count.fetch_add(1, Ordering::Relaxed);

        // Collect fingerprints from primary
        let mut fps = self.collect_primary_fps();
        if fps.is_empty() {
            return Ok(());
        }

        // Sort them
        fps.sort_unstable();

        // Merge with existing disk file if present
        let existing_disk_count = self.disk_count.load(Ordering::Relaxed);
        if existing_disk_count > 0 {
            fps = self.merge_with_disk(fps)?;
        }

        // Write new disk file
        let tmp_path = self.disk_path.with_extension("tmp");
        {
            let file = std::fs::File::create(&tmp_path)?;
            let mut writer = BufWriter::new(file);

            for &fp in &fps {
                writer.write_all(&fp.to_le_bytes())?;
            }
            writer.flush()?;
        }

        // Atomically replace old file
        std::fs::rename(&tmp_path, &self.disk_path)?;

        // Rebuild page index
        let mut new_index = Vec::with_capacity(fps.len().div_ceil(FPS_PER_PAGE));
        for (i, &fp) in fps.iter().enumerate() {
            if i % FPS_PER_PAGE == 0 {
                new_index.push(fp);
            }
        }

        // Update state
        *self.page_index.write() = new_index;
        self.disk_count.store(fps.len(), Ordering::Relaxed);

        // Clear primary (we've written everything to disk)
        // Note: This is safe because eviction holds an exclusive lock (evicting flag)
        self.primary.clear();

        Ok(())
    }

    /// Collect all fingerprints from primary storage.
    ///
    /// Note: This is O(capacity) scan - only used during eviction.
    fn collect_primary_fps(&self) -> Vec<u64> {
        self.primary.collect_all()
    }

    /// Merge fingerprints with existing disk file.
    fn merge_with_disk(&self, mut new_fps: Vec<u64>) -> io::Result<Vec<u64>> {
        use std::io::{BufReader, Read};

        let file = std::fs::File::open(&self.disk_path)?;
        let mut reader = BufReader::new(file);

        // Read all existing disk fingerprints
        let disk_count = self.disk_count.load(Ordering::Relaxed);
        let mut disk_fps = Vec::with_capacity(disk_count);

        let mut buf = [0u8; 8];
        while reader.read_exact(&mut buf).is_ok() {
            disk_fps.push(u64::from_le_bytes(buf));
        }

        // Merge sorted vectors
        new_fps.extend(disk_fps);
        new_fps.sort_unstable();
        new_fps.dedup(); // Remove duplicates

        Ok(new_fps)
    }

    /// Return the total number of fingerprints (primary + disk).
    pub fn len(&self) -> usize {
        self.total_count.load(Ordering::Relaxed)
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the number of fingerprints on disk.
    pub fn disk_count(&self) -> usize {
        self.disk_count.load(Ordering::Relaxed)
    }

    /// Return the number of evictions performed.
    pub fn eviction_count(&self) -> usize {
        self.eviction_count.load(Ordering::Relaxed)
    }

    /// Return statistics about disk operations.
    pub fn disk_stats(&self) -> (usize, usize) {
        (
            self.disk_lookups.load(Ordering::Relaxed),
            self.disk_hits.load(Ordering::Relaxed),
        )
    }
}

impl FingerprintSet for DiskFingerprintSet {
    fn insert(&self, fp: Fingerprint) -> bool {
        DiskFingerprintSet::insert(self, fp)
    }

    fn contains(&self, fp: Fingerprint) -> bool {
        DiskFingerprintSet::contains(self, fp)
    }

    fn len(&self) -> usize {
        DiskFingerprintSet::len(self)
    }

    fn has_errors(&self) -> bool {
        // Disk storage should not have errors (can always evict)
        false
    }

    fn dropped_count(&self) -> usize {
        0
    }

    fn capacity_status(&self) -> CapacityStatus {
        // Disk storage has effectively unlimited capacity
        CapacityStatus::Normal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(v: u64) -> Fingerprint {
        Fingerprint(v)
    }

    #[test]
    fn test_mmap_basic_operations() {
        let set = MmapFingerprintSet::new(1000, None).unwrap();

        // Initially empty
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);

        // Insert new fingerprint
        assert!(set.insert(fp(12345)).unwrap());
        assert_eq!(set.len(), 1);
        assert!(set.contains(fp(12345)));

        // Insert same fingerprint again
        assert!(!set.insert(fp(12345)).unwrap());
        assert_eq!(set.len(), 1);

        // Insert different fingerprint
        assert!(set.insert(fp(67890)).unwrap());
        assert_eq!(set.len(), 2);
        assert!(set.contains(fp(67890)));

        // Check non-existent
        assert!(!set.contains(fp(99999)));
    }

    #[test]
    fn test_mmap_fingerprint_zero() {
        // Fingerprint 0 is special-cased
        let set = MmapFingerprintSet::new(100, None).unwrap();

        assert!(!set.contains(fp(0)));
        assert!(set.insert(fp(0)).unwrap());
        assert!(set.contains(fp(0)));
        assert!(!set.insert(fp(0)).unwrap());
    }

    #[test]
    fn test_mmap_many_fingerprints() {
        let set = MmapFingerprintSet::new(10000, None).unwrap();

        // Insert 5000 fingerprints
        for i in 0..5000u64 {
            let v = i.wrapping_mul(0x12345678_9ABCDEF0); // Spread values
            assert!(set.insert(fp(v)).unwrap(), "failed to insert fp {}", i);
        }

        assert_eq!(set.len(), 5000);

        // Verify all present
        for i in 0..5000u64 {
            let v = i.wrapping_mul(0x12345678_9ABCDEF0);
            assert!(set.contains(fp(v)), "missing fp {}", i);
        }

        // Verify non-present
        for i in 5000..5100u64 {
            let v = i.wrapping_mul(0x12345678_9ABCDEF0);
            assert!(!set.contains(fp(v)), "unexpected fp {}", i);
        }
    }

    #[test]
    fn test_mmap_load_factor_limit() {
        // Small set with low load factor
        let set = MmapFingerprintSet::with_load_factor(100, None, 0.5).unwrap();

        // Insert 50 items (50% load factor)
        for i in 0..50u64 {
            set.insert(fp(i + 1)).unwrap(); // +1 to avoid 0
        }

        // Next insert should fail due to load factor
        let result = set.insert(fp(1000));
        assert!(
            result.is_err(),
            "expected TableFull error, got {:?}",
            result
        );

        match result {
            Err(MmapError::TableFull { count, capacity }) => {
                assert_eq!(count, 50);
                assert_eq!(capacity, 100);
            }
            _ => panic!("expected TableFull error"),
        }
    }

    #[test]
    fn test_mmap_concurrent_insert() {
        use std::sync::Arc;
        use std::thread;

        let set = Arc::new(MmapFingerprintSet::new(100000, None).unwrap());
        let num_threads = 8;
        let items_per_thread = 1000;

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let set = Arc::clone(&set);
                thread::spawn(move || {
                    for i in 0..items_per_thread {
                        let v = (t * items_per_thread + i + 1) as u64;
                        let _ = set.insert(fp(v));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(set.len(), num_threads * items_per_thread);
    }

    #[test]
    fn test_mmap_concurrent_contains() {
        use std::sync::Arc;
        use std::thread;

        let set = Arc::new(MmapFingerprintSet::new(10000, None).unwrap());

        // Pre-populate
        for i in 0..1000u64 {
            set.insert(fp(i + 1)).unwrap();
        }

        // Concurrent reads
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let set = Arc::clone(&set);
                thread::spawn(move || {
                    for i in 0..1000u64 {
                        assert!(set.contains(fp(i + 1)));
                    }
                    for i in 1000..2000u64 {
                        assert!(!set.contains(fp(i + 1)));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_mmap_file_backed() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let set = MmapFingerprintSet::new(1000, Some(tmp_dir.path().to_path_buf())).unwrap();

        assert!(set.insert(fp(12345)).unwrap());
        assert!(set.contains(fp(12345)));

        // Flush to ensure data is written
        set.flush().unwrap();
    }

    #[test]
    fn test_mmap_trait_impl() {
        let set: Box<dyn FingerprintSet> = Box::new(MmapFingerprintSet::new(1000, None).unwrap());

        assert!(set.insert(fp(12345)));
        assert!(set.contains(fp(12345)));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_dashset_trait_impl() {
        let set: Box<dyn FingerprintSet> = Box::new(dashmap::DashSet::<Fingerprint>::new());

        assert!(set.insert(fp(12345)));
        assert!(set.contains(fp(12345)));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_fingerprint_storage_in_memory() {
        let storage = FingerprintStorage::in_memory();

        assert!(storage.is_empty());
        assert!(storage.insert(fp(12345)));
        assert!(!storage.insert(fp(12345))); // duplicate
        assert!(storage.contains(fp(12345)));
        assert_eq!(storage.len(), 1);
    }

    #[test]
    fn test_fingerprint_storage_mmap() {
        let storage = FingerprintStorage::mmap(1000, None).unwrap();

        assert!(storage.is_empty());
        assert!(storage.insert(fp(12345)));
        assert!(!storage.insert(fp(12345))); // duplicate
        assert!(storage.contains(fp(12345)));
        assert_eq!(storage.len(), 1);
    }

    #[test]
    fn test_fingerprint_storage_disk() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let storage = FingerprintStorage::disk(100, tmp_dir.path()).unwrap();

        assert!(storage.is_empty());
        assert!(storage.insert(fp(12345)));
        assert!(!storage.insert(fp(12345))); // duplicate
        assert!(storage.contains(fp(12345)));
        assert_eq!(storage.len(), 1);

        // Insert more to trigger eviction
        for i in 1..=150 {
            storage.insert(fp(i));
        }

        // Verify all are accessible
        for i in 1..=150 {
            assert!(storage.contains(fp(i)), "Missing fingerprint {}", i);
        }

        // Disk storage reports Normal capacity (unlimited due to eviction)
        assert_eq!(storage.capacity_status(), CapacityStatus::Normal);
        assert!(!storage.has_errors());
    }

    #[test]
    fn test_fingerprint_storage_as_trait() {
        // Test that FingerprintStorage works as FingerprintSet
        let storage: Box<dyn FingerprintSet> =
            Box::new(FingerprintStorage::mmap(1000, None).unwrap());

        assert!(storage.insert(fp(111)));
        assert!(storage.contains(fp(111)));
        assert_eq!(storage.len(), 1);
    }

    // ========== MmapTraceLocations tests ==========

    #[test]
    fn test_trace_locations_basic() {
        let locs = MmapTraceLocations::new(1000, None).unwrap();

        // Initially empty
        assert!(locs.is_empty());
        assert_eq!(locs.len(), 0);

        // Insert new mapping
        assert!(locs.insert(fp(12345), 1024).unwrap());
        assert_eq!(locs.len(), 1);
        assert!(locs.contains(&fp(12345)));
        assert_eq!(locs.get(&fp(12345)), Some(1024));

        // Update existing mapping
        assert!(!locs.insert(fp(12345), 2048).unwrap());
        assert_eq!(locs.len(), 1);
        assert_eq!(locs.get(&fp(12345)), Some(2048));

        // Insert different fingerprint
        assert!(locs.insert(fp(67890), 4096).unwrap());
        assert_eq!(locs.len(), 2);
        assert_eq!(locs.get(&fp(67890)), Some(4096));

        // Check non-existent
        assert!(!locs.contains(&fp(99999)));
        assert_eq!(locs.get(&fp(99999)), None);
    }

    #[test]
    fn test_trace_locations_fingerprint_zero() {
        // Fingerprint 0 is special-cased
        let locs = MmapTraceLocations::new(100, None).unwrap();

        assert!(!locs.contains(&fp(0)));
        assert!(locs.insert(fp(0), 512).unwrap());
        assert!(locs.contains(&fp(0)));
        assert_eq!(locs.get(&fp(0)), Some(512));
        assert!(!locs.insert(fp(0), 1024).unwrap()); // update
        assert_eq!(locs.get(&fp(0)), Some(1024));
    }

    #[test]
    fn test_trace_locations_many_entries() {
        let locs = MmapTraceLocations::new(10000, None).unwrap();

        // Insert 5000 entries
        for i in 0..5000u64 {
            let v = i.wrapping_mul(0x12345678_9ABCDEF0);
            let offset = i * 16;
            assert!(
                locs.insert(fp(v), offset).unwrap(),
                "failed to insert fp {} at offset {}",
                v,
                offset
            );
        }

        assert_eq!(locs.len(), 5000);

        // Verify all present with correct offsets
        for i in 0..5000u64 {
            let v = i.wrapping_mul(0x12345678_9ABCDEF0);
            let offset = i * 16;
            assert_eq!(locs.get(&fp(v)), Some(offset), "wrong offset for fp {}", v);
        }

        // Verify non-present
        for i in 5000..5100u64 {
            let v = i.wrapping_mul(0x12345678_9ABCDEF0);
            assert!(!locs.contains(&fp(v)), "unexpected fp {}", v);
        }
    }

    #[test]
    fn test_trace_locations_load_factor() {
        let locs = MmapTraceLocations::with_load_factor(100, None, 0.5).unwrap();

        // Insert 50 entries (50% load factor)
        for i in 0..50u64 {
            locs.insert(fp(i + 1), i * 16).unwrap();
        }

        // Next insert should fail
        let result = locs.insert(fp(1000), 0);
        assert!(
            result.is_err(),
            "expected TableFull error, got {:?}",
            result
        );
    }

    #[test]
    fn test_trace_locations_file_backed() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let locs = MmapTraceLocations::new(1000, Some(tmp_dir.path().to_path_buf())).unwrap();

        assert!(locs.insert(fp(12345), 1024).unwrap());
        assert_eq!(locs.get(&fp(12345)), Some(1024));

        locs.flush().unwrap();
    }

    #[test]
    fn test_trace_locations_trait_impl() {
        let locs: Box<dyn TraceLocationStorage> =
            Box::new(MmapTraceLocations::new(1000, None).unwrap());

        locs.insert(fp(12345), 1024);
        assert!(locs.contains(&fp(12345)));
        assert_eq!(locs.get(&fp(12345)), Some(1024));
        assert_eq!(locs.len(), 1);
    }

    #[test]
    fn test_in_memory_trace_locations() {
        let locs = InMemoryTraceLocations::new();

        assert!(locs.is_empty());

        // Use trait method
        TraceLocationStorage::insert(&locs, fp(111), 500);
        assert_eq!(locs.get(&fp(111)), Some(500));
        assert!(locs.contains(&fp(111)));
        assert_eq!(locs.len(), 1);

        // Update
        TraceLocationStorage::insert(&locs, fp(111), 600);
        assert_eq!(locs.get(&fp(111)), Some(600));
        assert_eq!(locs.len(), 1);
    }

    #[test]
    fn test_in_memory_trace_locations_as_trait() {
        let locs: Box<dyn TraceLocationStorage> = Box::new(InMemoryTraceLocations::new());

        locs.insert(fp(222), 1000);
        assert_eq!(locs.get(&fp(222)), Some(1000));
        assert_eq!(locs.len(), 1);
    }

    #[test]
    fn test_mmap_error_tracking_no_errors() {
        // Test that a normal use case doesn't report errors
        let set = MmapFingerprintSet::new(1000, None).unwrap();

        assert!(!set.has_errors(), "New set should not have errors");
        assert_eq!(set.dropped_count(), 0, "New set should have 0 dropped");

        // Insert some fingerprints
        for i in 1..=100 {
            set.insert(fp(i)).unwrap();
        }

        assert!(!set.has_errors(), "Normal inserts should not cause errors");
        assert_eq!(set.dropped_count(), 0);
    }

    #[test]
    fn test_mmap_error_tracking_overflow() {
        // Create a very small table that will overflow quickly
        let set =
            MmapFingerprintSet::with_load_factor(10, None, 0.9).expect("Failed to create set");

        assert!(!set.has_errors(), "New set should not have errors");
        assert_eq!(set.dropped_count(), 0);

        // Fill the table to trigger overflow errors
        // With capacity 10 and 90% load factor, we can fit about 9 entries
        // After that, inserts should start failing
        let mut inserted = 0;
        let mut errors = 0;
        for i in 1..=50 {
            match set.insert(fp(i * 1000)) {
                // Use spaced values to avoid linear probing finding slots
                Ok(true) => inserted += 1,
                Ok(false) => {} // Already present (shouldn't happen with unique values)
                Err(_) => errors += 1,
            }
        }

        // We should have some successful inserts and some errors
        assert!(inserted > 0, "Should have some successful inserts");
        assert!(errors > 0, "Should have some errors from overflow");
    }

    #[test]
    fn test_mmap_error_tracking_via_trait() {
        // Test that error tracking works through the FingerprintSet trait
        let set =
            MmapFingerprintSet::with_load_factor(10, None, 0.9).expect("Failed to create set");

        let trait_obj: &dyn FingerprintSet = &set;

        assert!(!trait_obj.has_errors(), "New set should not have errors");
        assert_eq!(trait_obj.dropped_count(), 0);

        // Fill through the trait interface until we trigger errors
        for i in 1..=50 {
            trait_obj.insert(fp(i * 1000));
        }

        // Now errors should be tracked via the trait
        // Note: The trait impl calls record_error() on overflow
        assert!(
            trait_obj.has_errors(),
            "Trait should report errors after overflow"
        );
        assert!(
            trait_obj.dropped_count() > 0,
            "Trait should report dropped count"
        );
    }

    #[test]
    fn test_fingerprint_storage_error_tracking() {
        // Test that FingerprintStorage properly forwards error tracking
        let storage = FingerprintStorage::mmap(10, None).expect("Failed to create mmap storage");

        assert!(!storage.has_errors(), "New storage should not have errors");
        assert_eq!(storage.dropped_count(), 0);

        // In-memory storage should never have errors
        let in_mem = FingerprintStorage::in_memory();
        assert!(!in_mem.has_errors(), "In-memory never has errors");
        assert_eq!(in_mem.dropped_count(), 0);

        // Fill up in-memory (it can grow indefinitely)
        for i in 1..=100 {
            in_mem.insert(fp(i));
        }
        assert!(
            !in_mem.has_errors(),
            "In-memory should never report errors (can grow)"
        );
    }

    // ========== Capacity status tests ==========

    #[test]
    fn test_capacity_status_normal() {
        // Empty set should be Normal
        let set = MmapFingerprintSet::new(1000, None).unwrap();
        assert_eq!(set.capacity_status(), CapacityStatus::Normal);

        // Set at low usage should be Normal
        for i in 1..=100 {
            set.insert(fp(i)).unwrap();
        }
        // 100/1000 = 10% usage, well below warning threshold
        assert_eq!(
            set.capacity_status(),
            CapacityStatus::Normal,
            "10% usage should be Normal"
        );
    }

    #[test]
    fn test_capacity_status_warning() {
        // Create set with default 0.75 load factor
        // Warning threshold: 80% of 0.75 = 60% usage
        let set = MmapFingerprintSet::new(1000, None).unwrap();

        // Fill to just above warning threshold (60% = 600 entries)
        for i in 1..=610 {
            let _ = set.insert(fp(i));
        }

        match set.capacity_status() {
            CapacityStatus::Warning { usage, .. } => {
                assert!(usage >= 0.60, "Usage should be >= 60%");
                assert!(usage < 0.72, "Usage should be < 72% (critical threshold)");
            }
            other => panic!("Expected Warning status, got {:?}", other),
        }
    }

    #[test]
    fn test_capacity_status_critical() {
        // Create set with default 0.75 load factor
        // Critical threshold: 95% of 0.75 = 71.25% usage
        let set = MmapFingerprintSet::new(1000, None).unwrap();

        // Fill to just above critical threshold (72% = 720 entries)
        for i in 1..=720 {
            let _ = set.insert(fp(i));
        }

        match set.capacity_status() {
            CapacityStatus::Critical { usage, .. } => {
                assert!(usage >= 0.71, "Usage should be >= 71%");
            }
            other => panic!("Expected Critical status, got {:?}", other),
        }
    }

    #[test]
    fn test_capacity_status_via_trait() {
        // Test that capacity_status works through the FingerprintSet trait
        let set = MmapFingerprintSet::new(100, None).unwrap();
        let trait_obj: &dyn FingerprintSet = &set;

        assert_eq!(trait_obj.capacity_status(), CapacityStatus::Normal);

        // Fill to warning level (60% of 100 = 60 entries)
        for i in 1..=65 {
            trait_obj.insert(fp(i));
        }

        // Should now be Warning
        assert!(
            matches!(trait_obj.capacity_status(), CapacityStatus::Warning { .. }),
            "65% should trigger Warning status"
        );
    }

    #[test]
    fn test_capacity_status_fingerprint_storage() {
        // Test that FingerprintStorage properly forwards capacity_status
        let storage = FingerprintStorage::mmap(100, None).expect("Failed to create mmap storage");

        assert_eq!(
            storage.capacity_status(),
            CapacityStatus::Normal,
            "New mmap storage should be Normal"
        );

        // In-memory storage should always be Normal (growable)
        let in_mem = FingerprintStorage::in_memory();
        assert_eq!(
            in_mem.capacity_status(),
            CapacityStatus::Normal,
            "In-memory storage should always be Normal"
        );

        // Fill in-memory with many entries - should still be Normal
        for i in 1..=1000 {
            in_mem.insert(fp(i));
        }
        assert_eq!(
            in_mem.capacity_status(),
            CapacityStatus::Normal,
            "In-memory storage should still be Normal after many inserts"
        );
    }

    #[test]
    fn test_capacity_status_dashset_default() {
        // DashSet uses default implementation (always Normal)
        let set: Box<dyn FingerprintSet> = Box::new(dashmap::DashSet::<Fingerprint>::new());
        assert_eq!(
            set.capacity_status(),
            CapacityStatus::Normal,
            "DashSet should default to Normal (growable)"
        );

        // Fill with many entries
        for i in 1..=1000 {
            set.insert(fp(i));
        }
        assert_eq!(
            set.capacity_status(),
            CapacityStatus::Normal,
            "DashSet should still be Normal (growable)"
        );
    }

    // ========== ShardedFingerprintSet tests ==========

    #[test]
    fn test_sharded_basic_operations() {
        let set = ShardedFingerprintSet::new(4); // 16 shards

        // Initially empty
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);

        // Insert new fingerprint
        assert!(set.insert(fp(12345)));
        assert_eq!(set.len(), 1);
        assert!(set.contains(fp(12345)));

        // Insert same fingerprint again
        assert!(!set.insert(fp(12345)));
        assert_eq!(set.len(), 1);

        // Insert different fingerprint
        assert!(set.insert(fp(67890)));
        assert_eq!(set.len(), 2);
        assert!(set.contains(fp(67890)));

        // Check non-existent
        assert!(!set.contains(fp(99999)));
    }

    #[test]
    fn test_sharded_fingerprint_zero() {
        // Fingerprint 0 should work correctly
        let set = ShardedFingerprintSet::new(4);

        assert!(!set.contains(fp(0)));
        assert!(set.insert(fp(0)));
        assert!(set.contains(fp(0)));
        assert!(!set.insert(fp(0)));
    }

    #[test]
    fn test_sharded_many_fingerprints() {
        let set = ShardedFingerprintSet::new(6); // 64 shards

        // Insert 5000 fingerprints
        for i in 0..5000u64 {
            let v = i.wrapping_mul(0x12345678_9ABCDEF0);
            assert!(set.insert(fp(v)), "failed to insert fp {}", i);
        }

        assert_eq!(set.len(), 5000);

        // Verify all present
        for i in 0..5000u64 {
            let v = i.wrapping_mul(0x12345678_9ABCDEF0);
            assert!(set.contains(fp(v)), "missing fp {}", i);
        }

        // Verify non-present
        for i in 5000..5100u64 {
            let v = i.wrapping_mul(0x12345678_9ABCDEF0);
            assert!(!set.contains(fp(v)), "unexpected fp {}", i);
        }
    }

    #[test]
    fn test_sharded_concurrent_insert() {
        use std::sync::Arc;
        use std::thread;

        let set = Arc::new(ShardedFingerprintSet::new(6)); // 64 shards
        let num_threads = 8;
        let items_per_thread = 1000;

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let set = Arc::clone(&set);
                thread::spawn(move || {
                    for i in 0..items_per_thread {
                        let v = (t * items_per_thread + i + 1) as u64;
                        let _ = set.insert(fp(v));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(set.len(), num_threads * items_per_thread);
    }

    #[test]
    fn test_sharded_concurrent_contains() {
        use std::sync::Arc;
        use std::thread;

        let set = Arc::new(ShardedFingerprintSet::new(6));

        // Pre-populate
        for i in 0..1000u64 {
            set.insert(fp(i + 1));
        }

        // Concurrent reads
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let set = Arc::clone(&set);
                thread::spawn(move || {
                    for i in 0..1000u64 {
                        assert!(set.contains(fp(i + 1)));
                    }
                    for i in 1000..2000u64 {
                        assert!(!set.contains(fp(i + 1)));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_sharded_trait_impl() {
        let set: Box<dyn FingerprintSet> = Box::new(ShardedFingerprintSet::new(4));

        assert!(set.insert(fp(12345)));
        assert!(set.contains(fp(12345)));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_sharded_with_shards() {
        let set = ShardedFingerprintSet::with_shards(32);
        assert_eq!(set.num_shards(), 32);
        assert_eq!(set.shard_bits(), 5); // log2(32) = 5

        let set2 = ShardedFingerprintSet::with_shards(64);
        assert_eq!(set2.num_shards(), 64);
        assert_eq!(set2.shard_bits(), 6); // log2(64) = 6
    }

    #[test]
    #[should_panic(expected = "shard_bits must be 1-16")]
    fn test_sharded_invalid_shard_bits_zero() {
        ShardedFingerprintSet::new(0);
    }

    #[test]
    #[should_panic(expected = "shard_bits must be 1-16")]
    fn test_sharded_invalid_shard_bits_too_large() {
        ShardedFingerprintSet::new(17);
    }

    #[test]
    #[should_panic(expected = "num_shards must be power of 2")]
    fn test_sharded_with_shards_not_power_of_2() {
        ShardedFingerprintSet::with_shards(7);
    }

    // ========== DiskFingerprintSet tests ==========

    #[test]
    fn test_disk_fpset_basic_operations() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let set = DiskFingerprintSet::new(1000, tmp_dir.path()).unwrap();

        // Initially empty
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);

        // Insert new fingerprint
        assert!(set.insert(fp(12345)));
        assert_eq!(set.len(), 1);
        assert!(set.contains(fp(12345)));

        // Insert same fingerprint again
        assert!(!set.insert(fp(12345)));
        assert_eq!(set.len(), 1);

        // Insert different fingerprint
        assert!(set.insert(fp(67890)));
        assert_eq!(set.len(), 2);
        assert!(set.contains(fp(67890)));

        // Check non-existent
        assert!(!set.contains(fp(99999)));
    }

    #[test]
    fn test_disk_fpset_fingerprint_zero() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let set = DiskFingerprintSet::new(100, tmp_dir.path()).unwrap();

        assert!(!set.contains(fp(0)));
        assert!(set.insert(fp(0)));
        assert!(set.contains(fp(0)));
        assert!(!set.insert(fp(0)));
    }

    #[test]
    fn test_disk_fpset_many_fingerprints() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let set = DiskFingerprintSet::new(10000, tmp_dir.path()).unwrap();

        // Insert 5000 fingerprints
        for i in 0..5000u64 {
            let v = i.wrapping_mul(0x12345678_9ABCDEF0);
            assert!(set.insert(fp(v)), "failed to insert fp {}", i);
        }

        assert_eq!(set.len(), 5000);

        // Verify all present
        for i in 0..5000u64 {
            let v = i.wrapping_mul(0x12345678_9ABCDEF0);
            assert!(set.contains(fp(v)), "missing fp {}", i);
        }

        // Verify non-present
        for i in 5000..5100u64 {
            let v = i.wrapping_mul(0x12345678_9ABCDEF0);
            assert!(!set.contains(fp(v)), "unexpected fp {}", i);
        }
    }

    #[test]
    fn test_disk_fpset_trait_impl() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let set: Box<dyn FingerprintSet> =
            Box::new(DiskFingerprintSet::new(1000, tmp_dir.path()).unwrap());

        assert!(set.insert(fp(12345)));
        assert!(set.contains(fp(12345)));
        assert_eq!(set.len(), 1);

        // Disk storage should report Normal capacity (unlimited)
        assert_eq!(set.capacity_status(), CapacityStatus::Normal);
        assert!(!set.has_errors());
        assert_eq!(set.dropped_count(), 0);
    }

    #[test]
    fn test_disk_fpset_statistics() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let set = DiskFingerprintSet::new(100, tmp_dir.path()).unwrap();

        // Insert some fingerprints
        for i in 1..=50 {
            set.insert(fp(i));
        }

        // Check statistics
        assert_eq!(set.len(), 50);
        assert_eq!(set.disk_count(), 0); // Nothing evicted yet
        assert_eq!(set.eviction_count(), 0);

        let (lookups, hits) = set.disk_stats();
        // Disk stats should be 0 since we haven't triggered disk lookup
        assert_eq!(lookups, 0);
        assert_eq!(hits, 0);
    }

    #[test]
    fn test_disk_fpset_eviction() {
        let tmp_dir = tempfile::tempdir().unwrap();
        // Use a small primary capacity to trigger eviction quickly
        // Load factor is 0.75, so 100 capacity means eviction at ~75 entries
        let set = DiskFingerprintSet::new(100, tmp_dir.path()).unwrap();

        // Insert enough to trigger eviction (>75 entries)
        let num_entries = 150;
        for i in 1..=num_entries {
            set.insert(fp(i));
        }

        // Verify eviction occurred
        assert!(
            set.eviction_count() >= 1,
            "Expected at least 1 eviction, got {}",
            set.eviction_count()
        );
        assert!(
            set.disk_count() > 0,
            "Expected some entries on disk after eviction"
        );
        assert_eq!(
            set.len(),
            num_entries as usize,
            "Total count should include all entries"
        );

        // Verify all entries can still be found (either in primary or on disk)
        for i in 1..=num_entries {
            assert!(
                set.contains(fp(i)),
                "Fingerprint {} not found after eviction",
                i
            );
        }

        // Verify disk lookups are happening for evicted entries
        let (lookups, _hits) = set.disk_stats();
        assert!(
            lookups > 0,
            "Expected disk lookups when checking contains after eviction"
        );
    }

    #[test]
    fn test_mmap_collect_all() {
        // Test the collect_all method used during eviction
        let set = MmapFingerprintSet::new(100, None).unwrap();

        // Empty set should return empty
        assert!(set.collect_all().is_empty());

        // Insert some fingerprints
        let entries: Vec<u64> = vec![1, 42, 100, 12345];
        for &v in &entries {
            set.insert(fp(v)).unwrap();
        }

        // Collect and verify all are present
        let collected: FxHashSet<u64> = set.collect_all().into_iter().collect();
        assert_eq!(collected.len(), entries.len());
        for &v in &entries {
            assert!(collected.contains(&v), "Missing fingerprint {}", v);
        }
    }

    #[test]
    fn test_mmap_collect_all_with_fp_zero() {
        // Test that fingerprint 0 is correctly collected
        let set = MmapFingerprintSet::new(100, None).unwrap();

        set.insert(fp(0)).unwrap(); // Fingerprint zero (encoded as MAX)
        set.insert(fp(1)).unwrap();
        set.insert(fp(42)).unwrap();

        let collected: FxHashSet<u64> = set.collect_all().into_iter().collect();
        assert_eq!(collected.len(), 3);
        assert!(collected.contains(&0), "Missing fingerprint 0");
        assert!(collected.contains(&1), "Missing fingerprint 1");
        assert!(collected.contains(&42), "Missing fingerprint 42");
    }

    #[test]
    fn test_mmap_clear() {
        let set = MmapFingerprintSet::new(100, None).unwrap();

        // Insert some entries
        for i in 1..=50 {
            set.insert(fp(i)).unwrap();
        }
        assert_eq!(set.len(), 50);

        // Clear the set
        set.clear();

        // Verify it's empty
        assert_eq!(set.len(), 0);
        assert!(set.collect_all().is_empty());

        // Verify previous entries are no longer found
        for i in 1..=50 {
            assert!(!set.contains(fp(i)), "Fingerprint {} should not be present after clear", i);
        }

        // Verify we can insert again
        set.insert(fp(1)).unwrap();
        assert_eq!(set.len(), 1);
        assert!(set.contains(fp(1)));
    }
}
