//! TLA+ Value types
//!
//! This module defines the runtime values used during TLA+ expression evaluation.
//! Values are designed to be:
//! - Immutable: All values are immutable (functional style)
//! - Hashable: All values can be used as set elements or function domain elements
//! - Comparable: Total ordering for deterministic iteration
//!
//! # Value Types
//!
//! | TLA+ Type | Rust Type |
//! |-----------|-----------|
//! | BOOLEAN   | `Value::Bool(bool)` |
//! | Int       | `Value::Int(BigInt)` |
//! | STRING    | `Value::String(Arc<str>)` |
//! | Set       | `Value::Set(SortedSet)` |
//! | Function  | `Value::Func(FuncValue)` |
//! | Sequence  | `Value::Seq(Vec<Value>)` |
//! | Record    | `Value::Record(OrdMap<Arc<str>, Value>)` |
//! | Tuple     | `Value::Tuple(Vec<Value>)` |

use crate::error::{EvalError, EvalResult};
use dashmap::DashMap;
use im::{HashMap, OrdMap, OrdSet, Vector};
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Zero};
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap as StdHashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use tla_core::ast::{BoundVar, Expr};
use tla_core::Spanned;

// ============================================================================
// Memory Statistics (compile with --features memory-stats)
// ============================================================================

#[cfg(feature = "memory-stats")]
pub mod memory_stats {
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Total number of LazyFuncValue instances created
    pub static LAZY_FUNC_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Total number of memo entries across all LazyFuncValues
    pub static MEMO_ENTRY_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Total number of ArrayStates created
    pub static ARRAY_STATE_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Total bytes allocated for ArrayState values (Box<[Value]>)
    pub static ARRAY_STATE_BYTES: AtomicUsize = AtomicUsize::new(0);

    /// Total number of Values cloned
    pub static VALUE_CLONE_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Total number of FuncValue entries created
    pub static FUNC_ENTRY_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Total number of Sets created
    pub static SET_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Total number of set intern cache hits
    pub static SET_CACHE_HIT_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Total number of IntIntervalFunc instances created
    pub static INT_FUNC_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Total number of IntIntervalFunc entries (values)
    pub static INT_FUNC_ENTRY_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Total number of IntIntervalFunc except operations
    pub static INT_FUNC_EXCEPT_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Total number of IntIntervalFunc except clones (when Arc is shared)
    pub static INT_FUNC_EXCEPT_CLONE_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Total bytes allocated for Arc<Vec<Value>> in IntIntervalFunc
    pub static INT_FUNC_BYTES: AtomicUsize = AtomicUsize::new(0);

    /// Reset all counters (useful for benchmarks)
    pub fn reset() {
        LAZY_FUNC_COUNT.store(0, Ordering::Relaxed);
        MEMO_ENTRY_COUNT.store(0, Ordering::Relaxed);
        ARRAY_STATE_COUNT.store(0, Ordering::Relaxed);
        ARRAY_STATE_BYTES.store(0, Ordering::Relaxed);
        VALUE_CLONE_COUNT.store(0, Ordering::Relaxed);
        FUNC_ENTRY_COUNT.store(0, Ordering::Relaxed);
        SET_COUNT.store(0, Ordering::Relaxed);
        SET_CACHE_HIT_COUNT.store(0, Ordering::Relaxed);
        INT_FUNC_COUNT.store(0, Ordering::Relaxed);
        INT_FUNC_ENTRY_COUNT.store(0, Ordering::Relaxed);
        INT_FUNC_EXCEPT_COUNT.store(0, Ordering::Relaxed);
        INT_FUNC_EXCEPT_CLONE_COUNT.store(0, Ordering::Relaxed);
        INT_FUNC_BYTES.store(0, Ordering::Relaxed);
    }

    /// Format bytes in human-readable form
    fn format_bytes(bytes: usize) -> String {
        if bytes >= 1024 * 1024 * 1024 {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        } else if bytes >= 1024 * 1024 {
            format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
        } else if bytes >= 1024 {
            format!("{:.2} KB", bytes as f64 / 1024.0)
        } else {
            format!("{} bytes", bytes)
        }
    }

    /// Print current statistics
    pub fn print_stats() {
        eprintln!("=== Memory Statistics ===");
        eprintln!(
            "LazyFuncValue instances: {}",
            LAZY_FUNC_COUNT.load(Ordering::Relaxed)
        );
        eprintln!("Memo entries: {}", MEMO_ENTRY_COUNT.load(Ordering::Relaxed));
        let array_state_count = ARRAY_STATE_COUNT.load(Ordering::Relaxed);
        let array_state_bytes = ARRAY_STATE_BYTES.load(Ordering::Relaxed);
        eprintln!(
            "ArrayState instances: {} ({})",
            array_state_count,
            format_bytes(array_state_bytes)
        );
        if array_state_count > 0 {
            eprintln!(
                "  Avg bytes/ArrayState: {}",
                array_state_bytes / array_state_count
            );
        }
        eprintln!(
            "Value clones: {}",
            VALUE_CLONE_COUNT.load(Ordering::Relaxed)
        );
        eprintln!(
            "FuncValue entries: {}",
            FUNC_ENTRY_COUNT.load(Ordering::Relaxed)
        );
        eprintln!(
            "Set instances (calls): {}",
            SET_COUNT.load(Ordering::Relaxed)
        );
        eprintln!(
            "Set cache hits: {}",
            SET_CACHE_HIT_COUNT.load(Ordering::Relaxed)
        );
        let int_func_count = INT_FUNC_COUNT.load(Ordering::Relaxed);
        let int_func_bytes = INT_FUNC_BYTES.load(Ordering::Relaxed);
        eprintln!(
            "IntIntervalFunc instances: {} ({})",
            int_func_count,
            format_bytes(int_func_bytes)
        );
        eprintln!(
            "IntIntervalFunc entries: {}",
            INT_FUNC_ENTRY_COUNT.load(Ordering::Relaxed)
        );
        eprintln!(
            "IntIntervalFunc except ops: {}",
            INT_FUNC_EXCEPT_COUNT.load(Ordering::Relaxed)
        );
        eprintln!(
            "IntIntervalFunc except clones: {}",
            INT_FUNC_EXCEPT_CLONE_COUNT.load(Ordering::Relaxed)
        );
        // Show unique interned values
        if let Some(table) = super::SET_INTERN_TABLE.get() {
            eprintln!("Unique interned sets: {}", table.len());
        }
        if let Some(table) = super::INT_FUNC_INTERN_TABLE.get() {
            eprintln!("Unique interned IntFuncs: {}", table.len());
        }
        // Print size constants for reference
        eprintln!("--- Size constants ---");
        eprintln!(
            "sizeof(Value): {} bytes",
            std::mem::size_of::<super::Value>()
        );

        // Estimate total memory usage
        let array_state_bytes = ARRAY_STATE_BYTES.load(Ordering::Relaxed);
        let int_func_bytes = INT_FUNC_BYTES.load(Ordering::Relaxed);
        eprintln!("--- Memory Estimates ---");
        eprintln!(
            "ArrayState heap (Values): {}",
            format_bytes(array_state_bytes)
        );
        eprintln!("IntIntervalFunc heap: {}", format_bytes(int_func_bytes));
        eprintln!(
            "Total tracked: {}",
            format_bytes(array_state_bytes + int_func_bytes)
        );
        eprintln!("=========================");
    }

    #[inline]
    pub fn inc_int_func_except() {
        INT_FUNC_EXCEPT_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_int_func_except_clone() {
        INT_FUNC_EXCEPT_CLONE_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_lazy_func() {
        LAZY_FUNC_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_memo_entry() {
        MEMO_ENTRY_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_array_state() {
        ARRAY_STATE_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_array_state_bytes(num_vars: usize) {
        // Box<[Value]> + ArrayState overhead
        let value_size = std::mem::size_of::<super::Value>();
        let bytes = num_vars * value_size + 56; // 56 bytes for ArrayState struct overhead
        ARRAY_STATE_BYTES.fetch_add(bytes, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_value_clone() {
        VALUE_CLONE_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_func_entries(count: usize) {
        FUNC_ENTRY_COUNT.fetch_add(count, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_set() {
        SET_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_set_cache_hit() {
        SET_CACHE_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_int_func(entries: usize) {
        INT_FUNC_COUNT.fetch_add(1, Ordering::Relaxed);
        INT_FUNC_ENTRY_COUNT.fetch_add(entries, Ordering::Relaxed);
        // Track heap allocation: Arc header + Vec + Values
        let value_size = std::mem::size_of::<super::Value>();
        let bytes = 16 + 24 + entries * value_size; // Arc (16) + Vec header (24) + Values
        INT_FUNC_BYTES.fetch_add(bytes, Ordering::Relaxed);
    }
}

// ============================================================================
// String Interning for Fast Pointer-Based Equality
// ============================================================================

/// Global string intern table for O(1) pointer-based string equality.
///
/// When the same string literal is used multiple times (e.g., "ECHO", "V0"),
/// interning ensures they share the same Arc<str>, making comparisons
/// a simple pointer equality check instead of content comparison.
///
/// Uses DashMap for lock-free concurrent access.
static STRING_INTERN_TABLE: std::sync::OnceLock<DashMap<String, Arc<str>>> =
    std::sync::OnceLock::new();

/// Get the string intern table, initializing if needed.
#[inline]
fn get_intern_table() -> &'static DashMap<String, Arc<str>> {
    STRING_INTERN_TABLE.get_or_init(DashMap::new)
}

/// Intern a string, returning a shared `Arc<str>`.
///
/// If the string was previously interned, returns the existing Arc.
/// Otherwise, creates a new Arc and stores it for future reuse.
///
/// This function should be used for all TLA+ string values and model value names
/// to enable O(1) pointer-based equality comparisons.
#[inline]
pub fn intern_string(s: &str) -> Arc<str> {
    let table = get_intern_table();

    // Fast path: check if already interned
    if let Some(arc) = table.get(s) {
        return Arc::clone(arc.value());
    }

    // Slow path: insert new string
    // Use entry API to avoid race conditions
    table
        .entry(s.to_string())
        .or_insert_with(|| Arc::from(s))
        .clone()
}

/// Create an interned String value.
///
/// Uses the string intern table to ensure pointer equality for repeated strings.
#[inline]
pub fn interned_str_value(s: &str) -> Value {
    Value::String(intern_string(s))
}

/// Create an interned ModelValue.
///
/// Uses the string intern table to ensure pointer equality for repeated model values.
#[inline]
pub fn interned_model_value(s: &str) -> Value {
    Value::ModelValue(intern_string(s))
}

// ============================================================================
// Set Interning for Memory Efficiency
// ============================================================================

/// Global set intern table for deduplicating small sets.
///
/// Many TLA+ specs repeatedly create the same small sets (e.g., subsets of
/// {1, 2, 3} when processes are 1..3). Interning ensures we store each unique
/// set only once.
///
/// Key: FNV-1a hash of set elements (sorted)
/// Value: Arc<[Value]> - the interned set array
///
/// Only interning small sets (≤8 elements) to limit cache size and avoid
/// fingerprint collisions on large sets.
static SET_INTERN_TABLE: std::sync::OnceLock<DashMap<u64, Arc<[Value]>>> =
    std::sync::OnceLock::new();

/// Maximum set size for interning (larger sets are not cached)
const MAX_INTERN_SET_SIZE: usize = 8;

/// Get the set intern table.
#[inline]
fn get_set_intern_table() -> &'static DashMap<u64, Arc<[Value]>> {
    SET_INTERN_TABLE.get_or_init(DashMap::new)
}

/// Compute a fingerprint for a sorted set array.
#[inline]
fn set_fingerprint(elements: &[Value]) -> u64 {
    use std::hash::{Hash, Hasher};
    // Use FxHasher for speed (matching FxHashMap elsewhere)
    let mut hasher = rustc_hash::FxHasher::default();
    elements.hash(&mut hasher);
    hasher.finish()
}

/// Create an Arc<[Value]> for a set, using interning for small sets.
///
/// For sets with ≤8 elements, checks the global cache first.
/// Returns cached version if found, otherwise stores and returns new arc.
#[inline]
fn intern_set_array(elements: Vec<Value>) -> Arc<[Value]> {
    // Don't intern large sets
    if elements.len() > MAX_INTERN_SET_SIZE {
        return Arc::from(elements);
    }

    // Don't intern empty sets (handled by EMPTY_SET singleton)
    if elements.is_empty() {
        return Arc::from(elements);
    }

    let table = get_set_intern_table();
    let fp = set_fingerprint(&elements);

    // Fast path: check if already interned
    if let Some(arc) = table.get(&fp) {
        // Verify it's actually the same set (fingerprint collision check)
        if arc.len() == elements.len() && arc.iter().zip(elements.iter()).all(|(a, b)| a == b) {
            #[cfg(feature = "memory-stats")]
            memory_stats::inc_set_cache_hit();
            return Arc::clone(arc.value());
        }
    }

    // Slow path: create new arc and intern it
    let arc: Arc<[Value]> = Arc::from(elements);
    table.insert(fp, Arc::clone(&arc));
    arc
}

/// Clear the set intern table.
/// Call between model checking runs to free memory.
pub fn clear_set_intern_table() {
    if let Some(table) = SET_INTERN_TABLE.get() {
        table.clear();
    }
}

// ============================================================================
// IntIntervalFunc Interning for Memory Efficiency
// ============================================================================

/// Global intern table for IntIntervalFunc values.
/// Key: FNV-1a hash of (min, max, elements)
/// Value: Arc<Vec<Value>> - the interned values array
static INT_FUNC_INTERN_TABLE: std::sync::OnceLock<DashMap<u64, Arc<Vec<Value>>>> =
    std::sync::OnceLock::new();

/// Maximum IntIntervalFunc size for interning
const MAX_INTERN_INT_FUNC_SIZE: usize = 8;

/// Get the IntIntervalFunc intern table.
#[inline]
fn get_int_func_intern_table() -> &'static DashMap<u64, Arc<Vec<Value>>> {
    INT_FUNC_INTERN_TABLE.get_or_init(DashMap::new)
}

/// Compute a fingerprint for an IntIntervalFunc modification.
/// Computes what the fingerprint would be after setting values[arr_idx] = new_value.
#[inline]
fn int_func_modified_fingerprint(
    min: i64,
    max: i64,
    values: &[Value],
    arr_idx: usize,
    new_value: &Value,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    min.hash(&mut hasher);
    max.hash(&mut hasher);
    // Hash values with the modification applied
    for (i, v) in values.iter().enumerate() {
        if i == arr_idx {
            new_value.hash(&mut hasher);
        } else {
            v.hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Compute a fingerprint for an IntIntervalFunc.
#[inline]
fn int_func_fingerprint(min: i64, max: i64, values: &[Value]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    min.hash(&mut hasher);
    max.hash(&mut hasher);
    values.hash(&mut hasher);
    hasher.finish()
}

/// Try to find an interned IntIntervalFunc with a modification applied.
/// Returns the interned Arc if found, None if we need to create a new one.
#[inline]
fn try_get_interned_modified(
    min: i64,
    max: i64,
    values: &[Value],
    arr_idx: usize,
    new_value: &Value,
) -> Option<Arc<Vec<Value>>> {
    let table = get_int_func_intern_table();
    let fp = int_func_modified_fingerprint(min, max, values, arr_idx, new_value);

    if let Some(arc) = table.get(&fp) {
        // Verify the values match (collision check)
        if arc.len() == values.len() {
            let matches = arc.iter().enumerate().all(|(i, v)| {
                if i == arr_idx {
                    v == new_value
                } else {
                    v == &values[i]
                }
            });
            if matches {
                return Some(Arc::clone(arc.value()));
            }
        }
    }
    None
}

/// Intern an IntIntervalFunc's values array.
#[inline]
fn intern_int_func_array(min: i64, max: i64, values: Vec<Value>) -> Arc<Vec<Value>> {
    if values.len() > MAX_INTERN_INT_FUNC_SIZE {
        return Arc::new(values);
    }

    let table = get_int_func_intern_table();
    let fp = int_func_fingerprint(min, max, &values);

    // Fast path: check if already interned
    if let Some(arc) = table.get(&fp) {
        if arc.len() == values.len() && arc.iter().zip(values.iter()).all(|(a, b)| a == b) {
            return Arc::clone(arc.value());
        }
    }

    // Slow path: create and intern
    let arc = Arc::new(values);
    table.insert(fp, Arc::clone(&arc));
    arc
}

/// Clear the IntIntervalFunc intern table.
pub fn clear_int_func_intern_table() {
    if let Some(table) = INT_FUNC_INTERN_TABLE.get() {
        table.clear();
    }
}

// ============================================================================
// SortedSet - Array-based sorted set for performance
// ============================================================================

/// A sorted, deduplicated set of values stored in an Arc<[Value]>.
///
/// This is a performance-optimized set representation that uses a sorted array
/// instead of a persistent B-tree (OrdSet). Benefits:
/// - Single allocation for the entire set
/// - Excellent cache locality for iteration
/// - O(log n) membership via binary search
/// - O(n) merge operations for union/intersection/difference
/// - O(1) clone (Arc reference count increment)
///
/// Small sets (≤8 elements) are interned for memory deduplication.
///
/// Invariants:
/// - Elements are sorted in ascending order (by Value::cmp)
/// - No duplicate elements
pub struct SortedSet {
    elements: Arc<[Value]>,
    /// Cached `value_fingerprint()` result for this set value.
    ///
    /// This is a pure cache (doesn't affect semantic equality). It is copied on clone so
    /// cached fingerprints propagate through the model checking loop's value/state cloning.
    cached_value_fp: AtomicU64,
}

const SORTED_SET_FP_UNSET: u64 = u64::MAX;

/// Empty set singleton for reuse
static EMPTY_SET: std::sync::OnceLock<SortedSet> = std::sync::OnceLock::new();

impl SortedSet {
    #[inline]
    fn from_sorted_vec_uninterned(mut vec: Vec<Value>) -> Self {
        if vec.is_empty() {
            return Self::new();
        }
        vec.dedup();
        SortedSet {
            elements: Arc::from(vec),
            cached_value_fp: AtomicU64::new(SORTED_SET_FP_UNSET),
        }
    }

    /// Create an empty set
    #[inline]
    pub fn new() -> Self {
        EMPTY_SET
            .get_or_init(|| SortedSet {
                elements: Arc::from([]),
                cached_value_fp: AtomicU64::new(SORTED_SET_FP_UNSET),
            })
            .clone()
    }

    /// Create a set from a single element
    #[inline]
    pub fn unit(v: Value) -> Self {
        #[cfg(feature = "memory-stats")]
        memory_stats::inc_set();
        SortedSet {
            elements: intern_set_array(vec![v]),
            cached_value_fp: AtomicU64::new(SORTED_SET_FP_UNSET),
        }
    }

    /// Create a set from an iterator, sorting and deduplicating
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        let mut vec: Vec<Value> = iter.into_iter().collect();
        if vec.is_empty() {
            return Self::new();
        }
        vec.sort();
        vec.dedup();
        #[cfg(feature = "memory-stats")]
        memory_stats::inc_set();
        SortedSet {
            elements: intern_set_array(vec),
            cached_value_fp: AtomicU64::new(SORTED_SET_FP_UNSET),
        }
    }

    /// Create a set from a pre-sorted, deduplicated slice (unchecked)
    ///
    /// # Safety
    /// Caller must ensure the slice is sorted and contains no duplicates.
    #[inline]
    pub fn from_sorted_unchecked(slice: Arc<[Value]>) -> Self {
        SortedSet {
            elements: slice,
            cached_value_fp: AtomicU64::new(SORTED_SET_FP_UNSET),
        }
    }

    /// Create a set from a sorted Vec, without additional sorting
    /// Deduplicates but assumes input is already sorted.
    #[inline]
    pub fn from_sorted_vec(mut vec: Vec<Value>) -> Self {
        if vec.is_empty() {
            return Self::new();
        }
        vec.dedup();
        #[cfg(feature = "memory-stats")]
        memory_stats::inc_set();
        SortedSet {
            elements: intern_set_array(vec),
            cached_value_fp: AtomicU64::new(SORTED_SET_FP_UNSET),
        }
    }

    /// Merge two sorted slices into a new sorted set (O(n+m) merge, no re-sorting)
    ///
    /// Both slices must already be sorted. The result is sorted and deduplicated.
    /// This is faster than from_iter when both inputs are pre-sorted.
    #[inline]
    pub fn from_merged(a: &[Value], b: &[Value]) -> Self {
        if a.is_empty() && b.is_empty() {
            return Self::new();
        }
        if a.is_empty() {
            // b is already sorted, intern it
            #[cfg(feature = "memory-stats")]
            memory_stats::inc_set();
            return SortedSet {
                elements: intern_set_array(b.to_vec()),
                cached_value_fp: AtomicU64::new(SORTED_SET_FP_UNSET),
            };
        }
        if b.is_empty() {
            // a is already sorted, intern it
            #[cfg(feature = "memory-stats")]
            memory_stats::inc_set();
            return SortedSet {
                elements: intern_set_array(a.to_vec()),
                cached_value_fp: AtomicU64::new(SORTED_SET_FP_UNSET),
            };
        }

        // Merge two sorted sequences
        let mut result = Vec::with_capacity(a.len() + b.len());
        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => {
                    result.push(a[i].clone());
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push(b[j].clone());
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    // Skip duplicate
                    result.push(a[i].clone());
                    i += 1;
                    j += 1;
                }
            }
        }

        // Append remaining elements (already sorted)
        while i < a.len() {
            result.push(a[i].clone());
            i += 1;
        }
        while j < b.len() {
            result.push(b[j].clone());
            j += 1;
        }

        #[cfg(feature = "memory-stats")]
        memory_stats::inc_set();
        SortedSet {
            elements: intern_set_array(result),
            cached_value_fp: AtomicU64::new(SORTED_SET_FP_UNSET),
        }
    }

    /// Get the number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Check if the set contains a value (O(log n) binary search)
    #[inline]
    pub fn contains(&self, v: &Value) -> bool {
        self.elements.binary_search(v).is_ok()
    }

    /// Iterate over elements in sorted order
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, Value> {
        self.elements.iter()
    }

    /// Get the underlying slice
    #[inline]
    pub fn as_slice(&self) -> &[Value] {
        &self.elements
    }

    /// Get the underlying Arc
    #[inline]
    pub fn as_arc(&self) -> &Arc<[Value]> {
        &self.elements
    }

    /// Check pointer equality (fast path for identical sets)
    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.elements, &other.elements)
    }

    /// Insert a value, returning a new set (O(n))
    pub fn insert(&self, v: Value) -> Self {
        match self.elements.binary_search(&v) {
            Ok(_) => self.clone(), // Already present
            Err(pos) => {
                let mut vec = Vec::with_capacity(self.elements.len() + 1);
                vec.extend_from_slice(&self.elements[..pos]);
                vec.push(v);
                vec.extend_from_slice(&self.elements[pos..]);
                SortedSet::from_sorted_vec_uninterned(vec)
            }
        }
    }

    /// Remove a value, returning a new set (O(n))
    pub fn without(&self, v: &Value) -> Self {
        match self.elements.binary_search(v) {
            Ok(pos) => {
                let mut vec = Vec::with_capacity(self.elements.len() - 1);
                vec.extend_from_slice(&self.elements[..pos]);
                vec.extend_from_slice(&self.elements[pos + 1..]);
                if vec.is_empty() {
                    Self::new()
                } else {
                    SortedSet::from_sorted_vec_uninterned(vec)
                }
            }
            Err(_) => self.clone(), // Not present
        }
    }

    /// Remove a value (alias for without, for OrdSet API compatibility)
    #[inline]
    pub fn remove(&self, v: &Value) -> Self {
        self.without(v)
    }

    /// Set union (O(n + m) merge)
    pub fn union(&self, other: &Self) -> Self {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        if self.ptr_eq(other) {
            return self.clone();
        }

        let mut result = Vec::with_capacity(self.len() + other.len());
        let mut i = 0;
        let mut j = 0;
        let a = self.as_slice();
        let b = other.as_slice();

        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                Ordering::Less => {
                    result.push(a[i].clone());
                    i += 1;
                }
                Ordering::Greater => {
                    result.push(b[j].clone());
                    j += 1;
                }
                Ordering::Equal => {
                    result.push(a[i].clone());
                    i += 1;
                    j += 1;
                }
            }
        }
        result.extend_from_slice(&a[i..]);
        result.extend_from_slice(&b[j..]);

        SortedSet::from_sorted_vec_uninterned(result)
    }

    /// Set intersection (O(n + m) merge)
    pub fn intersection(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::new();
        }
        if self.ptr_eq(other) {
            return self.clone();
        }

        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;
        let a = self.as_slice();
        let b = other.as_slice();

        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    result.push(a[i].clone());
                    i += 1;
                    j += 1;
                }
            }
        }

        if result.is_empty() {
            Self::new()
        } else {
            SortedSet::from_sorted_vec_uninterned(result)
        }
    }

    /// Set difference (self \ other) (O(n + m) merge)
    pub fn difference(&self, other: &Self) -> Self {
        if self.is_empty() || self.ptr_eq(other) {
            return Self::new();
        }
        if other.is_empty() {
            return self.clone();
        }

        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;
        let a = self.as_slice();
        let b = other.as_slice();

        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                Ordering::Less => {
                    result.push(a[i].clone());
                    i += 1;
                }
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
            }
        }
        result.extend_from_slice(&a[i..]);

        if result.is_empty() {
            Self::new()
        } else {
            SortedSet::from_sorted_vec_uninterned(result)
        }
    }

    /// Check if self is a subset of other
    pub fn is_subset(&self, other: &Self) -> bool {
        if self.len() > other.len() {
            return false;
        }
        if self.is_empty() {
            return true;
        }
        if self.ptr_eq(other) {
            return true;
        }

        let mut j = 0;
        let b = other.as_slice();
        for v in self.iter() {
            while j < b.len() && b[j] < *v {
                j += 1;
            }
            if j >= b.len() || b[j] != *v {
                return false;
            }
            j += 1;
        }
        true
    }

    /// Get first element
    #[inline]
    pub fn first(&self) -> Option<&Value> {
        self.elements.first()
    }

    /// Get last element
    #[inline]
    pub fn last(&self) -> Option<&Value> {
        self.elements.last()
    }

    /// Convert to OrdSet (for compatibility during migration)
    pub fn to_ord_set(&self) -> OrdSet<Value> {
        self.iter().cloned().collect()
    }

    /// Create from OrdSet (for compatibility during migration)
    pub fn from_ord_set(set: &OrdSet<Value>) -> Self {
        // OrdSet is already sorted, so we can skip sorting
        if set.is_empty() {
            return Self::new();
        }
        let vec: Vec<Value> = set.iter().cloned().collect();
        SortedSet::from_sorted_vec_uninterned(vec)
    }

    /// Get cached `value_fingerprint()` result if available.
    #[inline]
    pub fn get_cached_value_fingerprint(&self) -> Option<u64> {
        let cached = self.cached_value_fp.load(AtomicOrdering::Relaxed);
        (cached != SORTED_SET_FP_UNSET).then_some(cached)
    }

    /// Cache a `value_fingerprint()` result.
    #[inline]
    pub fn cache_value_fingerprint(&self, fp: u64) -> u64 {
        let _ = self.cached_value_fp.compare_exchange(
            SORTED_SET_FP_UNSET,
            fp,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        );
        fp
    }
}

impl Clone for SortedSet {
    fn clone(&self) -> Self {
        SortedSet {
            elements: Arc::clone(&self.elements),
            cached_value_fp: AtomicU64::new(self.cached_value_fp.load(AtomicOrdering::Relaxed)),
        }
    }
}

impl Default for SortedSet {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for SortedSet {
    fn eq(&self, other: &Self) -> bool {
        if self.ptr_eq(other) {
            return true;
        }
        self.elements == other.elements
    }
}

impl Eq for SortedSet {}

impl PartialOrd for SortedSet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SortedSet {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.ptr_eq(other) {
            return Ordering::Equal;
        }
        self.elements.cmp(&other.elements)
    }
}

impl Hash for SortedSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.elements.hash(state);
    }
}

impl fmt::Debug for SortedSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<'a> IntoIterator for &'a SortedSet {
    type Item = &'a Value;
    type IntoIter = std::slice::Iter<'a, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl FromIterator<Value> for SortedSet {
    fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        SortedSet::from_iter(iter)
    }
}

// ============================================================================
// SetBuilder - Efficient set construction
// ============================================================================

/// Builder for constructing sets efficiently.
///
/// Collects values into a Vec, then sorts and deduplicates on build().
/// More efficient than repeated insert() calls when building from scratch.
#[derive(Default)]
pub struct SetBuilder(Vec<Value>);

impl SetBuilder {
    /// Create a new empty builder
    #[inline]
    pub fn new() -> Self {
        SetBuilder(Vec::new())
    }

    /// Create a builder with capacity
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        SetBuilder(Vec::with_capacity(cap))
    }

    /// Add a value
    #[inline]
    pub fn insert(&mut self, v: Value) {
        self.0.push(v);
    }

    /// Add multiple values
    #[inline]
    pub fn extend<I: IntoIterator<Item = Value>>(&mut self, iter: I) {
        self.0.extend(iter);
    }

    /// Build the final SortedSet
    #[inline]
    pub fn build(self) -> SortedSet {
        SortedSet::from_iter(self.0)
    }

    /// Build into a Value::Set
    #[inline]
    pub fn build_value(self) -> Value {
        Value::Set(self.build())
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get current length (may include duplicates)
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

// ============================================================================
// SeqValue - Sequence with cached fingerprint
// ============================================================================

/// Sentinel value indicating no cached fingerprint
const SEQ_VALUE_FP_UNSET: u64 = u64::MAX;

/// A TLA+ sequence value with cached fingerprint.
///
/// Similar to Tuple but with fingerprint caching for performance.
/// Sequences are 1-indexed in TLA+, but stored as 0-indexed internally.
pub struct SeqValue {
    elements: Vector<Value>,
    /// Cached `value_fingerprint()` result for this sequence value.
    ///
    /// This is a pure cache (doesn't affect semantic equality). It is copied on clone so
    /// cached fingerprints propagate through the model checking loop's value/state cloning.
    cached_fp: AtomicU64,
}

impl SeqValue {
    /// Create a new empty sequence
    #[inline]
    pub fn new() -> Self {
        SeqValue {
            elements: Vector::new(),
            cached_fp: AtomicU64::new(SEQ_VALUE_FP_UNSET),
        }
    }

    /// Create a sequence from a vector of values
    #[inline]
    pub fn from_vec(vec: Vec<Value>) -> Self {
        if vec.is_empty() {
            return Self::new();
        }
        SeqValue {
            elements: Vector::from(vec),
            cached_fp: AtomicU64::new(SEQ_VALUE_FP_UNSET),
        }
    }

    /// Create a sequence from an im::Vector
    #[inline]
    pub fn from_imvec(elements: Vector<Value>) -> Self {
        SeqValue {
            elements,
            cached_fp: AtomicU64::new(SEQ_VALUE_FP_UNSET),
        }
    }

    /// Convert to a Vec (for backward compatibility and operations that need slices)
    #[inline]
    pub fn to_vec(&self) -> Vec<Value> {
        self.elements.iter().cloned().collect()
    }

    /// Get the number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Get an element by 0-based index - O(log n)
    #[inline]
    pub fn get(&self, index: usize) -> Option<&Value> {
        self.elements.get(index)
    }

    /// Get the first element - O(log n)
    #[inline]
    pub fn first(&self) -> Option<&Value> {
        self.elements.front()
    }

    /// Get the last element - O(log n)
    #[inline]
    pub fn last(&self) -> Option<&Value> {
        self.elements.back()
    }

    /// Iterate over elements
    #[inline]
    pub fn iter(&self) -> im::vector::Iter<'_, Value> {
        self.elements.iter()
    }

    /// Get the underlying im::Vector reference
    #[inline]
    pub fn as_imvec(&self) -> &Vector<Value> {
        &self.elements
    }

    // ========================================================================
    // Efficient O(log n) operations for persistent sequences
    // ========================================================================

    /// Return a new sequence without the first element - O(log n)
    /// Equivalent to Tail in TLA+
    #[inline]
    pub fn tail(&self) -> Self {
        if self.elements.is_empty() {
            return Self::new();
        }
        SeqValue {
            elements: self.elements.clone().split_off(1),
            cached_fp: AtomicU64::new(SEQ_VALUE_FP_UNSET),
        }
    }

    /// Return a new sequence with element appended - O(log n)
    /// Equivalent to Append in TLA+
    #[inline]
    pub fn append(&self, elem: Value) -> Self {
        let mut new_elements = self.elements.clone();
        new_elements.push_back(elem);
        SeqValue {
            elements: new_elements,
            cached_fp: AtomicU64::new(SEQ_VALUE_FP_UNSET),
        }
    }

    /// Return a subsequence from start to end (0-indexed, exclusive end) - O(log n)
    /// Equivalent to SubSeq in TLA+ (adjusted for 0-indexing)
    #[inline]
    pub fn subseq(&self, start: usize, end: usize) -> Self {
        if start >= end || start >= self.elements.len() {
            return Self::new();
        }
        let end = end.min(self.elements.len());
        SeqValue {
            elements: self.elements.clone().slice(start..end),
            cached_fp: AtomicU64::new(SEQ_VALUE_FP_UNSET),
        }
    }

    /// Return all but the last element - O(log n)
    /// Equivalent to Front in TLA+ (SequencesExt)
    #[inline]
    pub fn front(&self) -> Self {
        if self.elements.is_empty() {
            return Self::new();
        }
        let len = self.elements.len();
        SeqValue {
            elements: self.elements.clone().slice(0..len - 1),
            cached_fp: AtomicU64::new(SEQ_VALUE_FP_UNSET),
        }
    }

    // ========================================================================
    // Fingerprint caching
    // ========================================================================

    /// Get cached fingerprint if available
    #[inline]
    pub fn get_cached_fingerprint(&self) -> Option<u64> {
        let cached = self.cached_fp.load(AtomicOrdering::Relaxed);
        (cached != SEQ_VALUE_FP_UNSET).then_some(cached)
    }

    /// Cache a fingerprint value.
    #[inline]
    pub fn cache_fingerprint(&self, fp: u64) -> u64 {
        let _ = self.cached_fp.compare_exchange(
            SEQ_VALUE_FP_UNSET,
            fp,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        );
        fp
    }

    /// Check if two SeqValues share the same underlying storage (pointer equality)
    /// Note: im::Vector uses structural sharing, so this checks if they share the same root
    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        std::ptr::eq(&self.elements, &other.elements)
    }
}

impl Clone for SeqValue {
    fn clone(&self) -> Self {
        // Clone the cached fingerprint as well (AtomicU64 doesn't implement Clone)
        // im::Vector clone is O(1) due to structural sharing
        let cached = self.cached_fp.load(AtomicOrdering::Relaxed);
        SeqValue {
            elements: self.elements.clone(),
            cached_fp: AtomicU64::new(cached),
        }
    }
}

impl Default for SeqValue {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Vec<Value>> for SeqValue {
    fn from(vec: Vec<Value>) -> Self {
        SeqValue::from_vec(vec)
    }
}

impl From<Vector<Value>> for SeqValue {
    fn from(elements: Vector<Value>) -> Self {
        SeqValue::from_imvec(elements)
    }
}

impl PartialEq for SeqValue {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.elements == other.elements
    }
}

impl Eq for SeqValue {}

impl PartialOrd for SeqValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SeqValue {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare element by element
        for (a, b) in self.elements.iter().zip(other.elements.iter()) {
            match a.cmp(b) {
                Ordering::Equal => continue,
                other => return other,
            }
        }
        self.elements.len().cmp(&other.elements.len())
    }
}

impl Hash for SeqValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash length first for better distribution
        self.elements.len().hash(state);
        for elem in &self.elements {
            elem.hash(state);
        }
    }
}

impl fmt::Debug for SeqValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a> IntoIterator for &'a SeqValue {
    type Item = &'a Value;
    type IntoIter = im::vector::Iter<'a, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl FromIterator<Value> for SeqValue {
    fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        SeqValue::from_vec(iter.into_iter().collect())
    }
}

impl std::ops::Index<usize> for SeqValue {
    type Output = Value;

    fn index(&self, index: usize) -> &Self::Output {
        self.elements.get(index).expect("SeqValue index out of bounds")
    }
}

// ============================================================================
// Value enum
// ============================================================================

/// A TLA+ runtime value
#[derive(Clone)]
pub enum Value {
    /// Boolean: TRUE or FALSE
    Bool(bool),
    /// Small integer (fits in i64) - fast path for common case
    SmallInt(i64),
    /// Arbitrary-precision integer (BigInt) - slow path for large numbers
    Int(BigInt),
    /// String value
    String(Arc<str>),
    /// Set of values (sorted array for performance)
    Set(SortedSet),
    /// Lazy integer interval (a..b) without allocating all elements
    Interval(IntervalValue),
    /// Lazy powerset (SUBSET S) without allocating all 2^|S| elements
    Subset(SubsetValue),
    /// Lazy function set ([S -> T]) without allocating all |T|^|S| functions
    FuncSet(FuncSetValue),
    /// Lazy record set ([a: S, b: T]) without allocating all |S|*|T| records
    RecordSet(RecordSetValue),
    /// Lazy tuple set (S1 \X S2 \X ...) without allocating all |S1|*|S2|*... tuples
    TupleSet(TupleSetValue),
    /// Lazy set union (S1 \cup S2) without eager enumeration
    SetCup(SetCupValue),
    /// Lazy set intersection (S1 \cap S2) without eager enumeration
    SetCap(SetCapValue),
    /// Lazy set difference (S1 \ S2) without eager enumeration
    SetDiff(SetDiffValue),
    /// Lazy set filter ({x \in S : P(x)}) without eager enumeration
    /// Membership requires evaluation context, so set_contains returns None
    SetPred(SetPredValue),
    /// Lazy k-subset set (Ksubsets(S, k)) without allocating all C(n,k) subsets
    KSubset(KSubsetValue),
    /// Lazy big union (UNION S) without allocating all elements
    BigUnion(UnionValue),
    /// Function (mapping from domain to range)
    Func(FuncValue),
    /// Array-backed function for small integer interval domains
    /// Much faster than Func for EXCEPT operations on int-indexed functions
    IntFunc(IntIntervalFunc),
    /// Lazy function for non-enumerable domains (e.g., Nat, Int)
    /// Boxed to reduce Value enum size from 112 to ~72 bytes
    LazyFunc(Box<LazyFuncValue>),
    /// Sequence (1-indexed) - uses Arc<[Value]> for O(1) clone
    Seq(SeqValue),
    /// Record with named fields
    /// Uses array-backed RecordValue for better cache locality and iteration performance
    Record(RecordValue),
    /// Tuple (heterogeneous, 1-indexed) - uses Arc<[Value]> for O(1) clone
    Tuple(Arc<[Value]>),
    /// Model value (for symmetry sets and state graph)
    ModelValue(Arc<str>),
    /// Closure for higher-order operator arguments (LAMBDA expressions)
    /// Stores the lambda parameters, body, and captured environment
    Closure(ClosureValue),
    /// The set of all strings (infinite, lazy)
    /// Used by the Strings module STRING constant
    StringSet,
    /// The set of all values (infinite, lazy)
    /// Used by TLC's AnySet module and TLC!Any operator
    AnySet,
    /// Lazy sequence set (Seq(S)) without allocating all finite sequences over S
    SeqSet(SeqSetValue),
}

/// A lazy interval value representing a..b without allocating all elements
///
/// This is a performance optimization for large integer ranges.
/// The interval is inclusive: [low, high].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IntervalValue {
    /// Lower bound (inclusive)
    pub low: BigInt,
    /// Upper bound (inclusive)
    pub high: BigInt,
}

impl IntervalValue {
    /// Create a new interval [low, high]
    pub fn new(low: BigInt, high: BigInt) -> Self {
        IntervalValue { low, high }
    }

    /// Check if the interval is empty (low > high)
    pub fn is_empty(&self) -> bool {
        self.low > self.high
    }

    /// Get the number of elements in the interval
    pub fn len(&self) -> BigInt {
        if self.is_empty() {
            BigInt::zero()
        } else {
            &self.high - &self.low + BigInt::one()
        }
    }

    /// Check if a value is contained in this interval
    pub fn contains(&self, v: &Value) -> bool {
        match v {
            Value::SmallInt(n) => {
                // Fast path: avoid BigInt allocation when bounds fit in i64.
                if let (Some(low), Some(high)) = (self.low.to_i64(), self.high.to_i64()) {
                    return *n >= low && *n <= high;
                }
                let n = BigInt::from(*n);
                n >= self.low && n <= self.high
            }
            Value::Int(n) => {
                // Fast path: compare in i64 space when possible.
                if let (Some(low), Some(high), Some(v)) =
                    (self.low.to_i64(), self.high.to_i64(), n.to_i64())
                {
                    return v >= low && v <= high;
                }
                n >= &self.low && n <= &self.high
            }
            _ => false,
        }
    }

    /// Iterate over all elements in the interval
    pub fn iter(&self) -> IntervalIterator {
        IntervalIterator {
            current: self.low.clone(),
            high: self.high.clone(),
        }
    }

    /// Iterate over all elements in the interval as `Value`.
    ///
    /// Uses an i64-backed iterator when bounds fit, avoiding per-element BigInt work.
    pub fn iter_values(&self) -> IntervalValueIter {
        match (self.low.to_i64(), self.high.to_i64()) {
            (Some(low), Some(high)) => IntervalValueIter::I64(low..=high),
            _ => IntervalValueIter::BigInt(self.iter()),
        }
    }

    /// Convert to an eager SortedSet (use sparingly for large intervals)
    pub fn to_sorted_set(&self) -> SortedSet {
        // Interval elements are already sorted (low to high), so skip sort
        let vec: Vec<Value> = self.iter_values().collect();
        SortedSet::from_sorted_vec(vec)
    }

    /// Convert to an eager OrdSet (for compatibility)
    pub fn to_ord_set(&self) -> OrdSet<Value> {
        self.iter_values().collect()
    }
}

/// Iterator over interval elements
pub struct IntervalIterator {
    current: BigInt,
    high: BigInt,
}

impl Iterator for IntervalIterator {
    type Item = BigInt;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current > self.high {
            None
        } else {
            let result = self.current.clone();
            self.current += 1;
            Some(result)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.current > self.high {
            (0, Some(0))
        } else {
            let len = (&self.high - &self.current + BigInt::one())
                .to_usize()
                .unwrap_or(usize::MAX);
            (len, Some(len))
        }
    }
}

/// Iterator over interval elements as `Value`.
///
/// For small ranges (bounds fit in i64), this avoids BigInt allocation and arithmetic.
pub enum IntervalValueIter {
    I64(std::ops::RangeInclusive<i64>),
    BigInt(IntervalIterator),
}

impl Iterator for IntervalValueIter {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IntervalValueIter::I64(r) => r.next().map(Value::SmallInt),
            IntervalValueIter::BigInt(it) => it.next().map(Value::Int),
        }
    }
}

impl Ord for IntervalValue {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.low.cmp(&other.low) {
            Ordering::Equal => self.high.cmp(&other.high),
            ord => ord,
        }
    }
}

impl PartialOrd for IntervalValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A lazy powerset value representing SUBSET S without allocating all 2^|S| elements
///
/// This is a performance optimization for large powersets.
/// Membership check: x \in SUBSET S <==> x \subseteq S
#[derive(Clone, Debug)]
pub struct SubsetValue {
    /// The base set S (can be Set, Interval, or Subset)
    pub base: Box<Value>,
}

impl SubsetValue {
    /// Create a new powerset of the given set
    pub fn new(base: Value) -> Self {
        SubsetValue {
            base: Box::new(base),
        }
    }

    /// Check if a value is contained in this powerset (i.e., is a subset of base)
    pub fn contains(&self, v: &Value) -> bool {
        // v \in SUBSET S <==> v is a set AND v \subseteq S
        if let Some(iter) = v.iter_set() {
            for elem in iter {
                if !self.base.set_contains(&elem).unwrap_or(false) {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    /// Get the cardinality of the powerset: 2^|base|
    pub fn len(&self) -> Option<BigInt> {
        let base_len = self.base.set_len()?;
        // 2^base_len - be careful about very large sets
        base_len.to_u32().map(|n| BigInt::from(1u64) << n)
    }

    /// Check if the powerset is empty (only if base is invalid)
    pub fn is_empty(&self) -> bool {
        // SUBSET S always contains {} (the empty set)
        false
    }

    /// Convert to an eager SortedSet (use sparingly for large powersets)
    pub fn to_sorted_set(&self) -> Option<SortedSet> {
        let base_set = self.base.to_sorted_set()?;
        Some(powerset_eager(&base_set))
    }

    /// Convert to an eager OrdSet (for compatibility)
    pub fn to_ord_set(&self) -> Option<OrdSet<Value>> {
        self.to_sorted_set().map(|s| s.to_ord_set())
    }

    /// Iterate over all elements of the powerset
    pub fn iter(&self) -> Option<impl Iterator<Item = Value> + '_> {
        let base_set = self.base.to_sorted_set()?;
        Some(SubsetIterator::new(base_set))
    }
}

/// Iterator over powerset elements
pub struct SubsetIterator {
    elements: Vec<Value>,
    counter: u64,
    max: u64,
}

impl SubsetIterator {
    /// Create a new SubsetIterator over all subsets of the given base set.
    pub fn new(base: SortedSet) -> Self {
        let elements: Vec<Value> = base.iter().cloned().collect();
        let len = elements.len();
        let max = if len >= 64 { u64::MAX } else { 1u64 << len };
        SubsetIterator {
            elements,
            counter: 0,
            max,
        }
    }

    /// Returns the total number of subsets (2^n for n elements).
    #[inline]
    pub fn len(&self) -> u64 {
        self.max
    }

    /// Check if there are no subsets (always false for valid sets).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.max == 0
    }
}

impl Iterator for SubsetIterator {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        if self.counter >= self.max {
            return None;
        }
        // Generate subset corresponding to counter's bit pattern
        let mut subset = SetBuilder::new();
        for (i, elem) in self.elements.iter().enumerate() {
            if self.counter & (1u64 << i) != 0 {
                subset.insert(elem.clone());
            }
        }
        self.counter += 1;
        Some(subset.build_value())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.max - self.counter) as usize;
        (remaining, Some(remaining))
    }
}

impl PartialEq for SubsetValue {
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
    }
}

impl Eq for SubsetValue {}

impl Hash for SubsetValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.base.hash(state);
    }
}

impl Ord for SubsetValue {
    fn cmp(&self, other: &Self) -> Ordering {
        self.base.cmp(&other.base)
    }
}

impl PartialOrd for SubsetValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Helper function to compute powerset eagerly (for SubsetValue::to_ord_set)
fn powerset_eager(set: &SortedSet) -> SortedSet {
    let mut result = SetBuilder::new();
    let elements: Vec<_> = set.iter().cloned().collect();
    let n = elements.len();
    let count = if n >= 64 { u64::MAX } else { 1u64 << n };

    for i in 0..count {
        let mut subset = SetBuilder::new();
        for (j, elem) in elements.iter().enumerate() {
            if i & (1u64 << j) != 0 {
                subset.insert(elem.clone());
            }
        }
        result.insert(subset.build_value());
    }
    result.build()
}

/// A lazy function set value representing [S -> T] without allocating all |T|^|S| functions
///
/// This is a performance optimization for large function sets.
/// Membership check: f \in [S -> T] <==> f is a function with DOMAIN f = S and range(f) \subseteq T
#[derive(Clone, Debug)]
pub struct FuncSetValue {
    /// The domain set S
    pub domain: Box<Value>,
    /// The codomain set T
    pub codomain: Box<Value>,
}

impl FuncSetValue {
    /// Create a new function set [domain -> codomain]
    pub fn new(domain: Value, codomain: Value) -> Self {
        FuncSetValue {
            domain: Box::new(domain),
            codomain: Box::new(codomain),
        }
    }

    /// Check if a value is contained in this function set
    /// f \in [S -> T] iff f is a function with DOMAIN f = S and range(f) \subseteq T
    pub fn contains(&self, v: &Value) -> bool {
        match v {
            Value::Func(f) => {
                // Check domain equality
                let domain_set = match self.domain.to_ord_set() {
                    Some(s) => s,
                    None => return false,
                };
                if f.domain_as_ord_set() != domain_set {
                    return false;
                }
                // Check range subset
                for val in f.mapping_values() {
                    if !self.codomain.set_contains(val).unwrap_or(false) {
                        return false;
                    }
                }
                true
            }
            // IntFunc is a function with integer interval domain
            Value::IntFunc(f) => {
                // Check domain equality: function set domain must equal min..max
                let domain_set = match self.domain.to_ord_set() {
                    Some(s) => s,
                    None => return false,
                };
                let expected_domain: OrdSet<Value> = (f.min..=f.max).map(Value::SmallInt).collect();
                if domain_set != expected_domain {
                    return false;
                }
                // Check range subset: all values must be in codomain
                for val in f.values.iter() {
                    if !self.codomain.set_contains(val).unwrap_or(false) {
                        return false;
                    }
                }
                true
            }
            // Tuples/Seqs are functions with domain 1..n
            Value::Tuple(elems) => {
                // Check domain equality: domain must be 1..n
                let expected_domain = if elems.is_empty() {
                    OrdSet::new()
                } else {
                    (1..=elems.len())
                        .map(|i| Value::Int(BigInt::from(i)))
                        .collect()
                };
                let domain_set = match self.domain.to_ord_set() {
                    Some(s) => s,
                    None => return false,
                };
                if domain_set != expected_domain {
                    return false;
                }
                // Check range subset: all elements must be in codomain
                for elem in elems.iter() {
                    if !self.codomain.set_contains(elem).unwrap_or(false) {
                        return false;
                    }
                }
                true
            }
            Value::Seq(seq) => {
                // Check domain equality: domain must be 1..n
                let expected_domain = if seq.is_empty() {
                    OrdSet::new()
                } else {
                    (1..=seq.len())
                        .map(|i| Value::Int(BigInt::from(i)))
                        .collect()
                };
                let domain_set = match self.domain.to_ord_set() {
                    Some(s) => s,
                    None => return false,
                };
                if domain_set != expected_domain {
                    return false;
                }
                // Check range subset: all elements must be in codomain
                for elem in seq.iter() {
                    if !self.codomain.set_contains(elem).unwrap_or(false) {
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// Get the cardinality of the function set: |T|^|S|
    pub fn len(&self) -> Option<BigInt> {
        let domain_len = self.domain.set_len()?;
        let codomain_len = self.codomain.set_len()?;
        // |T|^|S| - be careful about very large sets but use BigInt for safety
        // TLC handles this so we should too - just return the BigInt result
        if let (Some(d), Some(c)) = (domain_len.to_u32(), codomain_len.to_u64()) {
            // Limit to reasonable exponent to prevent long computation
            // 2^30 is about 1 billion, which is a reasonable upper bound
            if d <= 30 {
                Some(BigInt::from(c).pow(d))
            } else {
                None // Exponent too large
            }
        } else {
            None
        }
    }

    /// Check if the function set is empty
    pub fn is_empty(&self) -> bool {
        // [S -> T] is empty iff S is non-empty and T is empty
        let domain_empty = self.domain.set_len().map_or(true, |n| n.is_zero());
        let codomain_empty = self.codomain.set_len().map_or(true, |n| n.is_zero());
        !domain_empty && codomain_empty
    }

    /// Convert to an eager OrdSet (use sparingly for large function sets)
    pub fn to_ord_set(&self) -> Option<OrdSet<Value>> {
        let domain_set = self.domain.to_ord_set()?;
        let codomain_set = self.codomain.to_ord_set()?;
        Some(func_set_eager(&domain_set, &codomain_set))
    }

    /// Convert to an eager SortedSet (use sparingly for large function sets)
    pub fn to_sorted_set(&self) -> Option<SortedSet> {
        self.to_ord_set().map(|s| SortedSet::from_ord_set(&s))
    }

    /// Iterate over all functions in the function set
    pub fn iter(&self) -> Option<impl Iterator<Item = Value> + '_> {
        let domain_set = self.domain.to_ord_set()?;
        let codomain_set = self.codomain.to_ord_set()?;
        Some(FuncSetIterator::new(domain_set, codomain_set))
    }
}

/// Iterator over function set elements
pub struct FuncSetIterator {
    domain_elems: Vec<Value>,
    codomain_elems: Vec<Value>,
    indices: Vec<usize>,
    done: bool,
}

impl FuncSetIterator {
    fn new(domain: OrdSet<Value>, codomain: OrdSet<Value>) -> Self {
        let domain_elems: Vec<Value> = domain.into_iter().collect();
        let codomain_elems: Vec<Value> = codomain.into_iter().collect();
        let n = domain_elems.len();
        let done = codomain_elems.is_empty() && !domain_elems.is_empty();
        FuncSetIterator {
            indices: vec![0; n],
            domain_elems,
            codomain_elems,
            done,
        }
    }

    /// Check if domain elements form a consecutive integer interval.
    /// Returns Some((min, max)) if so, None otherwise.
    fn is_int_interval_domain(&self) -> Option<(i64, i64)> {
        if self.domain_elems.is_empty() {
            return None;
        }

        // All elements must be integers
        let mut ints: Vec<i64> = Vec::with_capacity(self.domain_elems.len());
        for elem in &self.domain_elems {
            match elem {
                Value::SmallInt(n) => ints.push(*n),
                Value::Int(n) => {
                    if let Some(i) = n.to_i64() {
                        ints.push(i);
                    } else {
                        return None; // Out of i64 range
                    }
                }
                _ => return None, // Non-integer domain element
            }
        }

        // Must be sorted (domain_elems comes from OrdSet which is sorted)
        // Check if consecutive integers: min, min+1, min+2, ..., max
        let min = ints[0];
        let max = ints[ints.len() - 1];
        if (max - min + 1) as usize != ints.len() {
            return None; // Not consecutive
        }

        // Verify all are consecutive (sorted + size check should be enough,
        // but let's be explicit)
        for (i, &n) in ints.iter().enumerate() {
            if n != min + i as i64 {
                return None;
            }
        }

        Some((min, max))
    }
}

impl Iterator for FuncSetIterator {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Handle empty domain case: [{}->T] = {[]}
        if self.domain_elems.is_empty() {
            self.done = true;
            return Some(Value::Func(FuncValue::new(OrdSet::new(), OrdMap::new())));
        }

        // Check if domain is a consecutive integer sequence starting at some min
        // If so, use IntFunc for better EXCEPT performance
        // IMPORTANT: If domain is 1..n, create Seq instead (functions 1..n are sequences in TLA+)
        let func = if let Some((min, max)) = self.is_int_interval_domain() {
            // Build IntFunc/Seq with array of values
            let values: Vec<Value> = (0..self.domain_elems.len())
                .map(|i| self.codomain_elems[self.indices[i]].clone())
                .collect();
            // If domain is 1..n, this is semantically a sequence
            if min == 1 {
                Value::Seq(values.into())
            } else {
                Value::IntFunc(IntIntervalFunc::new(min, max, values))
            }
        } else {
            // Build standard FuncValue with OrdMap
            let mut mapping = OrdMap::new();
            for (i, d) in self.domain_elems.iter().enumerate() {
                mapping.insert(d.clone(), self.codomain_elems[self.indices[i]].clone());
            }
            let domain_set: OrdSet<Value> = self.domain_elems.iter().cloned().collect();
            Value::Func(FuncValue::new(domain_set, mapping))
        };

        // Increment indices (like counting in base |T|)
        let mut carry = true;
        for i in 0..self.indices.len() {
            if carry {
                self.indices[i] += 1;
                if self.indices[i] >= self.codomain_elems.len() {
                    self.indices[i] = 0;
                } else {
                    carry = false;
                }
            }
        }
        if carry {
            self.done = true;
        }

        Some(func)
    }
}

impl PartialEq for FuncSetValue {
    fn eq(&self, other: &Self) -> bool {
        self.domain == other.domain && self.codomain == other.codomain
    }
}

impl Eq for FuncSetValue {}

impl Hash for FuncSetValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.domain.hash(state);
        self.codomain.hash(state);
    }
}

impl Ord for FuncSetValue {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.domain.cmp(&other.domain) {
            Ordering::Equal => self.codomain.cmp(&other.codomain),
            ord => ord,
        }
    }
}

impl PartialOrd for FuncSetValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Helper function to compute function set eagerly (for FuncSetValue::to_ord_set)
fn func_set_eager(domain: &OrdSet<Value>, codomain: &OrdSet<Value>) -> OrdSet<Value> {
    let domain_vec: Vec<_> = domain.iter().cloned().collect();
    let codomain_vec: Vec<_> = codomain.iter().cloned().collect();
    let n = domain_vec.len();
    let m = codomain_vec.len();

    // Handle edge cases
    if n == 0 {
        // Empty domain: only the empty function
        let mut result = OrdSet::new();
        result.insert(Value::Func(FuncValue::new(OrdSet::new(), OrdMap::new())));
        return result;
    }
    if m == 0 {
        // Empty codomain with non-empty domain: no functions
        return OrdSet::new();
    }

    let mut result = OrdSet::new();
    let mut indices = vec![0usize; n];
    let mut done = false;

    while !done {
        // Build function from current indices
        let mut mapping = OrdMap::new();
        for (i, d) in domain_vec.iter().enumerate() {
            mapping.insert(d.clone(), codomain_vec[indices[i]].clone());
        }
        result.insert(Value::Func(FuncValue::new(domain.clone(), mapping)));

        // Increment indices
        let mut carry = true;
        for idx in indices.iter_mut().take(n) {
            if carry {
                *idx += 1;
                if *idx >= m {
                    *idx = 0;
                } else {
                    carry = false;
                }
            }
        }
        if carry {
            done = true;
        }
    }

    result
}

/// A lazy record set value representing [a: S, b: T, ...] without allocating all records.
///
/// This is a performance optimization for large record sets.
/// Membership check: r \in [a: S, b: T] <==> r is a record with exactly those fields and
/// for each field name k, `r[k] \in fields[k]`.
#[derive(Clone, Debug)]
pub struct RecordSetValue {
    /// Field name -> allowed values set
    pub fields: OrdMap<Arc<str>, Box<Value>>,
}

/// A lazy tuple set value representing S1 \X S2 \X ... without allocating all tuples.
///
/// This is a performance optimization for large cartesian products.
/// Membership check: t \in S1 \X S2 \X ... <==> t is a tuple of the right length and
/// for each index i, `t[i] \in components[i]`.
#[derive(Clone, Debug)]
pub struct TupleSetValue {
    /// Component sets in order
    pub components: Vec<Box<Value>>,
}

impl RecordSetValue {
    /// Create a new record set from (field_name, field_set) pairs.
    pub fn new(fields: impl IntoIterator<Item = (Arc<str>, Value)>) -> Self {
        let mut map = OrdMap::new();
        for (name, set) in fields {
            map.insert(name, Box::new(set));
        }
        RecordSetValue { fields: map }
    }

    /// Check if the record set is definitely empty.
    ///
    /// `[ ]` is **not** empty: it contains the empty record.
    /// Otherwise, the record set is empty if any field domain is known to be empty.
    pub fn is_empty(&self) -> bool {
        if self.fields.is_empty() {
            return false;
        }

        self.fields
            .iter()
            .any(|(_field, set)| set.set_len().is_some_and(|n| n.is_zero()))
    }

    /// Check if a value is contained in this record set.
    pub fn contains(&self, v: &Value) -> bool {
        let Value::Record(rec) = v else {
            return false;
        };

        if rec.len() != self.fields.len() {
            return false;
        }

        for (field, set) in self.fields.iter() {
            let Some(field_val) = rec.get(field.as_ref()) else {
                return false;
            };
            match set.set_contains(field_val) {
                Some(true) => {}
                Some(false) | None => return false,
            }
        }

        true
    }

    /// Get the cardinality of the record set: Π |Si| for each field i.
    ///
    /// If any field set cardinality is unknown, returns None.
    pub fn len(&self) -> Option<BigInt> {
        let mut total = BigInt::one();
        for (_field, set) in self.fields.iter() {
            let n = set.set_len()?;
            total *= n;
        }
        Some(total)
    }

    /// Convert to an eager OrdSet (use sparingly for large record sets).
    pub fn to_ord_set(&self) -> Option<OrdSet<Value>> {
        let iter = self.iter()?;
        Some(iter.collect())
    }

    /// Convert to an eager SortedSet (use sparingly for large record sets).
    pub fn to_sorted_set(&self) -> Option<SortedSet> {
        self.to_ord_set().map(|s| SortedSet::from_ord_set(&s))
    }

    /// Iterate over all records in the record set.
    pub fn iter(&self) -> Option<RecordSetIterator<'_>> {
        RecordSetIterator::new(self)
    }
}

impl Ord for RecordSetValue {
    fn cmp(&self, other: &Self) -> Ordering {
        // Two empty record sets are equal regardless of their field structure.
        // e.g., [a: {1}, b: {}] == [c: {2}, d: {}] == {}
        let self_empty = self.is_empty();
        let other_empty = other.is_empty();
        match (self_empty, other_empty) {
            (true, true) => return Ordering::Equal,
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            (false, false) => {}
        }

        self.fields.cmp(&other.fields)
    }
}

impl PartialOrd for RecordSetValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for RecordSetValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for RecordSetValue {}

/// Iterator over record set elements, generated in lexicographic order by field name then value.
pub struct RecordSetIterator<'a> {
    fields: Vec<(Arc<str>, &'a Value)>,
    iters: Vec<Box<dyn Iterator<Item = Value> + 'a>>,
    current: Vec<Value>,
    done: bool,
}

impl<'a> RecordSetIterator<'a> {
    fn new(record_set: &'a RecordSetValue) -> Option<Self> {
        let fields: Vec<(Arc<str>, &'a Value)> = record_set
            .fields
            .iter()
            .map(|(k, v)| (k.clone(), v.as_ref()))
            .collect();

        if fields.is_empty() {
            return Some(RecordSetIterator {
                fields,
                iters: Vec::new(),
                current: Vec::new(),
                done: false,
            });
        }

        let mut iters = Vec::with_capacity(fields.len());
        let mut current = Vec::with_capacity(fields.len());

        for (_name, set) in &fields {
            let mut iter = set.iter_set()?;
            match iter.next() {
                Some(first) => {
                    current.push(first);
                    iters.push(iter);
                }
                None => {
                    // Empty field set => empty record set.
                    return Some(RecordSetIterator {
                        fields,
                        iters: Vec::new(),
                        current: Vec::new(),
                        done: true,
                    });
                }
            }
        }

        Some(RecordSetIterator {
            fields,
            iters,
            current,
            done: false,
        })
    }
}

impl<'a> Iterator for RecordSetIterator<'a> {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Special case: [ ] record set has one element: the empty record.
        if self.fields.is_empty() {
            self.done = true;
            return Some(Value::Record(RecordValue::new()));
        }

        // Build the current record.
        let mut builder = RecordBuilder::with_capacity(self.fields.len());
        for (idx, (name, _)) in self.fields.iter().enumerate() {
            builder.insert(name.clone(), self.current[idx].clone());
        }
        let out = Value::Record(builder.build());

        // Advance the mixed-radix counter (last field changes fastest).
        for idx in (0..self.fields.len()).rev() {
            if let Some(next_val) = self.iters.get_mut(idx).and_then(|it| it.next()) {
                self.current[idx] = next_val;

                // Reset all less-significant fields to their first element.
                for j in (idx + 1)..self.fields.len() {
                    let Some(mut iter) = self.fields[j].1.iter_set() else {
                        self.done = true;
                        return Some(out);
                    };
                    let Some(first) = iter.next() else {
                        self.done = true;
                        return Some(out);
                    };
                    self.current[j] = first;
                    self.iters[j] = iter;
                }

                return Some(out);
            }

            // Carry: reset this field to its first element and continue to the next field.
            let Some(mut iter) = self.fields[idx].1.iter_set() else {
                self.done = true;
                return Some(out);
            };
            let Some(first) = iter.next() else {
                self.done = true;
                return Some(out);
            };
            self.current[idx] = first;
            self.iters[idx] = iter;
        }

        // Exhausted all fields: this was the last element.
        self.done = true;
        Some(out)
    }
}

impl TupleSetValue {
    /// Create a new tuple set (cartesian product) from component sets.
    pub fn new(components: impl IntoIterator<Item = Value>) -> Self {
        TupleSetValue {
            components: components.into_iter().map(Box::new).collect(),
        }
    }

    /// Check if a value is contained in this tuple set.
    /// `t \in S1 \X S2 \X ...` iff t is a tuple of the right length and `t[i] \in components[i]`.
    pub fn contains(&self, v: &Value) -> bool {
        let Value::Tuple(elems) = v else {
            return false;
        };

        if elems.len() != self.components.len() {
            return false;
        }

        for (elem, set) in elems.iter().zip(self.components.iter()) {
            match set.set_contains(elem) {
                Some(true) => {}
                Some(false) | None => return false,
            }
        }

        true
    }

    /// Get the cardinality of the tuple set: Π |Si| for each component i.
    pub fn len(&self) -> Option<BigInt> {
        let mut total = BigInt::one();
        for set in &self.components {
            let n = set.set_len()?;
            total *= n;
        }
        Some(total)
    }

    /// Check if the tuple set is empty.
    pub fn is_empty(&self) -> bool {
        // Empty if any component is empty
        self.components
            .iter()
            .any(|c| c.set_len().map_or(true, |n| n.is_zero()))
    }

    /// Convert to an eager OrdSet (use sparingly for large tuple sets).
    pub fn to_ord_set(&self) -> Option<OrdSet<Value>> {
        let iter = self.iter()?;
        Some(iter.collect())
    }

    /// Convert to an eager SortedSet.
    pub fn to_sorted_set(&self) -> Option<SortedSet> {
        self.to_ord_set().map(|s| SortedSet::from_ord_set(&s))
    }

    /// Iterate over all tuples in the tuple set.
    pub fn iter(&self) -> Option<TupleSetIterator<'_>> {
        TupleSetIterator::new(self)
    }
}

impl Ord for TupleSetValue {
    fn cmp(&self, other: &Self) -> Ordering {
        // Two empty tuple sets are equal regardless of their component structure
        // e.g., {} \X {1,2} == {1,2} \X {} == {}
        let self_empty = self.is_empty();
        let other_empty = other.is_empty();
        match (self_empty, other_empty) {
            (true, true) => return Ordering::Equal,
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            (false, false) => {}
        }

        // Compare by length first, then element-wise
        match self.components.len().cmp(&other.components.len()) {
            Ordering::Equal => {
                for (a, b) in self.components.iter().zip(other.components.iter()) {
                    match a.cmp(b) {
                        Ordering::Equal => continue,
                        ord => return ord,
                    }
                }
                Ordering::Equal
            }
            ord => ord,
        }
    }
}

impl PartialOrd for TupleSetValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for TupleSetValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for TupleSetValue {}

impl Hash for TupleSetValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.components.len().hash(state);
        for c in &self.components {
            c.hash(state);
        }
    }
}

/// Iterator over tuple set elements, generated in lexicographic order.
pub struct TupleSetIterator<'a> {
    components: &'a [Box<Value>],
    iters: Vec<Box<dyn Iterator<Item = Value> + 'a>>,
    current: Vec<Value>,
    done: bool,
}

impl<'a> TupleSetIterator<'a> {
    fn new(tuple_set: &'a TupleSetValue) -> Option<Self> {
        if tuple_set.components.is_empty() {
            // Empty product: single empty tuple
            return Some(TupleSetIterator {
                components: &tuple_set.components,
                iters: Vec::new(),
                current: Vec::new(),
                done: false,
            });
        }

        let mut iters = Vec::with_capacity(tuple_set.components.len());
        let mut current = Vec::with_capacity(tuple_set.components.len());

        for set in &tuple_set.components {
            let mut iter = set.iter_set()?;
            match iter.next() {
                Some(first) => {
                    current.push(first);
                    iters.push(iter);
                }
                None => {
                    // Empty component set => empty tuple set.
                    return Some(TupleSetIterator {
                        components: &tuple_set.components,
                        iters: Vec::new(),
                        current: Vec::new(),
                        done: true,
                    });
                }
            }
        }

        Some(TupleSetIterator {
            components: &tuple_set.components,
            iters,
            current,
            done: false,
        })
    }
}

impl<'a> Iterator for TupleSetIterator<'a> {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Special case: empty component list produces one empty tuple.
        if self.components.is_empty() {
            self.done = true;
            return Some(Value::Tuple(Vec::new().into()));
        }

        // Build the current tuple.
        let out = Value::Tuple(self.current.clone().into());

        // Advance the mixed-radix counter (last component changes fastest).
        for idx in (0..self.components.len()).rev() {
            if let Some(next_val) = self.iters.get_mut(idx).and_then(|it| it.next()) {
                self.current[idx] = next_val;

                // Reset all less-significant components to their first element.
                for j in (idx + 1)..self.components.len() {
                    let Some(mut iter) = self.components[j].iter_set() else {
                        self.done = true;
                        return Some(out);
                    };
                    let Some(first) = iter.next() else {
                        self.done = true;
                        return Some(out);
                    };
                    self.current[j] = first;
                    self.iters[j] = iter;
                }

                return Some(out);
            }

            // Carry: reset this component to its first element and continue.
            let Some(mut iter) = self.components[idx].iter_set() else {
                self.done = true;
                return Some(out);
            };
            let Some(first) = iter.next() else {
                self.done = true;
                return Some(out);
            };
            self.current[idx] = first;
            self.iters[idx] = iter;
        }

        // Exhausted all components: this was the last element.
        self.done = true;
        Some(out)
    }
}

// === Lazy Set Operation Types ===
//
// These types represent set operations (union, intersection, difference) lazily,
// allowing operations on non-enumerable sets like STRING and AnySet.

/// Lazy set union (S1 \cup S2)
///
/// Membership is computed lazily: v \in S1 \cup S2 iff v \in S1 OR v \in S2
/// Enumeration only happens when both operands are enumerable.
#[derive(Clone)]
pub struct SetCupValue {
    pub set1: Box<Value>,
    pub set2: Box<Value>,
}

impl SetCupValue {
    /// Create a new lazy set union
    pub fn new(set1: Value, set2: Value) -> Self {
        SetCupValue {
            set1: Box::new(set1),
            set2: Box::new(set2),
        }
    }

    /// Check if a value is in this union set
    /// v \in S1 \cup S2 iff v \in S1 OR v \in S2
    pub fn contains(&self, v: &Value) -> bool {
        let in1 = self.set1.set_contains(v).unwrap_or(false);
        if in1 {
            return true;
        }
        self.set2.set_contains(v).unwrap_or(false)
    }

    /// Check if the union is enumerable (both operands must be enumerable)
    pub fn is_enumerable(&self) -> bool {
        self.set1.iter_set().is_some() && self.set2.iter_set().is_some()
    }

    /// Get the cardinality (only if enumerable)
    pub fn len(&self) -> Option<BigInt> {
        let set = self.to_ord_set()?;
        Some(BigInt::from(set.len()))
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        // Empty iff both operands are empty
        let e1 = self.set1.set_len().is_some_and(|n| n.is_zero());
        let e2 = self.set2.set_len().is_some_and(|n| n.is_zero());
        e1 && e2
    }

    /// Convert to an eager OrdSet (only if both operands are enumerable)
    pub fn to_ord_set(&self) -> Option<OrdSet<Value>> {
        let s1 = self.set1.to_ord_set()?;
        let s2 = self.set2.to_ord_set()?;
        Some(s1.union(s2))
    }

    /// Convert to an eager SortedSet (only if both operands are enumerable)
    pub fn to_sorted_set(&self) -> Option<SortedSet> {
        self.to_ord_set().map(|s| SortedSet::from_ord_set(&s))
    }

    /// Iterate over all elements (only if enumerable)
    pub fn iter(&self) -> Option<impl Iterator<Item = Value> + '_> {
        let set = self.to_ord_set()?;
        Some(set.into_iter())
    }
}

impl fmt::Debug for SetCupValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SetCup({:?}, {:?})", self.set1, self.set2)
    }
}

impl Ord for SetCupValue {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.set1.cmp(&other.set1) {
            Ordering::Equal => self.set2.cmp(&other.set2),
            ord => ord,
        }
    }
}

impl PartialOrd for SetCupValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SetCupValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for SetCupValue {}

impl Hash for SetCupValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        "SetCup".hash(state);
        self.set1.hash(state);
        self.set2.hash(state);
    }
}

/// Lazy set intersection (S1 \cap S2)
///
/// Membership is computed lazily: v \in S1 \cap S2 iff v \in S1 AND v \in S2
/// Enumeration happens by iterating the smaller enumerable set (if any) and filtering.
#[derive(Clone)]
pub struct SetCapValue {
    pub set1: Box<Value>,
    pub set2: Box<Value>,
}

impl SetCapValue {
    /// Create a new lazy set intersection
    pub fn new(set1: Value, set2: Value) -> Self {
        SetCapValue {
            set1: Box::new(set1),
            set2: Box::new(set2),
        }
    }

    /// Check if a value is in this intersection
    /// v \in S1 \cap S2 iff v \in S1 AND v \in S2
    pub fn contains(&self, v: &Value) -> bool {
        let in1 = self.set1.set_contains(v).unwrap_or(false);
        if !in1 {
            return false;
        }
        self.set2.set_contains(v).unwrap_or(false)
    }

    /// Check if the intersection is enumerable (at least one operand must be enumerable)
    pub fn is_enumerable(&self) -> bool {
        self.set1.iter_set().is_some() || self.set2.iter_set().is_some()
    }

    /// Get the cardinality (only if enumerable)
    pub fn len(&self) -> Option<BigInt> {
        let set = self.to_ord_set()?;
        Some(BigInt::from(set.len()))
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        // Check if either operand is empty (which makes intersection empty)
        let e1 = self.set1.set_len().is_some_and(|n| n.is_zero());
        let e2 = self.set2.set_len().is_some_and(|n| n.is_zero());
        if e1 || e2 {
            return true;
        }
        // Otherwise, we can only know by enumerating
        self.to_ord_set().is_some_and(|s| s.is_empty())
    }

    /// Convert to an eager OrdSet
    /// Iterates the enumerable set and filters by membership in the other
    pub fn to_ord_set(&self) -> Option<OrdSet<Value>> {
        // Prefer to iterate the smaller set if both are enumerable
        let iter1 = self.set1.iter_set();
        let iter2 = self.set2.iter_set();

        match (iter1, iter2) {
            (Some(it1), Some(it2)) => {
                // Both enumerable - choose smaller one to iterate
                let len1 = self.set1.set_len();
                let len2 = self.set2.set_len();
                match (len1, len2) {
                    (Some(l1), Some(l2)) if l1 <= l2 => Some(
                        it1.filter(|v| self.set2.set_contains(v).unwrap_or(false))
                            .collect(),
                    ),
                    (Some(_), Some(_)) => Some(
                        it2.filter(|v| self.set1.set_contains(v).unwrap_or(false))
                            .collect(),
                    ),
                    // Unknown lengths - just use set1
                    _ => Some(
                        it1.filter(|v| self.set2.set_contains(v).unwrap_or(false))
                            .collect(),
                    ),
                }
            }
            (Some(it), None) => {
                // Only set1 enumerable - iterate it, filter by set2 membership
                Some(
                    it.filter(|v| self.set2.set_contains(v).unwrap_or(false))
                        .collect(),
                )
            }
            (None, Some(it)) => {
                // Only set2 enumerable - iterate it, filter by set1 membership
                Some(
                    it.filter(|v| self.set1.set_contains(v).unwrap_or(false))
                        .collect(),
                )
            }
            (None, None) => {
                // Neither enumerable - cannot compute
                None
            }
        }
    }

    /// Convert to an eager SortedSet
    pub fn to_sorted_set(&self) -> Option<SortedSet> {
        self.to_ord_set().map(|s| SortedSet::from_ord_set(&s))
    }

    /// Iterate over all elements (only if at least one operand is enumerable)
    pub fn iter(&self) -> Option<Box<dyn Iterator<Item = Value> + '_>> {
        let set = self.to_ord_set()?;
        Some(Box::new(set.into_iter()))
    }
}

impl fmt::Debug for SetCapValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SetCap({:?}, {:?})", self.set1, self.set2)
    }
}

impl Ord for SetCapValue {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.set1.cmp(&other.set1) {
            Ordering::Equal => self.set2.cmp(&other.set2),
            ord => ord,
        }
    }
}

impl PartialOrd for SetCapValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SetCapValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for SetCapValue {}

impl Hash for SetCapValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        "SetCap".hash(state);
        self.set1.hash(state);
        self.set2.hash(state);
    }
}

/// Lazy set difference (S1 \ S2)
///
/// Membership is computed lazily: v \in S1 \ S2 iff v \in S1 AND v \notin S2
/// Enumeration happens by iterating S1 (if enumerable) and filtering out S2 members.
#[derive(Clone)]
pub struct SetDiffValue {
    pub set1: Box<Value>,
    pub set2: Box<Value>,
}

impl SetDiffValue {
    /// Create a new lazy set difference
    pub fn new(set1: Value, set2: Value) -> Self {
        SetDiffValue {
            set1: Box::new(set1),
            set2: Box::new(set2),
        }
    }

    /// Check if a value is in this set difference
    /// v \in S1 \ S2 iff v \in S1 AND v \notin S2
    pub fn contains(&self, v: &Value) -> bool {
        let in1 = self.set1.set_contains(v).unwrap_or(false);
        if !in1 {
            return false;
        }
        !self.set2.set_contains(v).unwrap_or(false)
    }

    /// Check if the difference is enumerable (LHS must be enumerable)
    pub fn is_enumerable(&self) -> bool {
        self.set1.iter_set().is_some()
    }

    /// Get the cardinality (only if LHS is enumerable)
    pub fn len(&self) -> Option<BigInt> {
        let set = self.to_ord_set()?;
        Some(BigInt::from(set.len()))
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        // Empty if LHS is empty
        let e1 = self.set1.set_len().is_some_and(|n| n.is_zero());
        if e1 {
            return true;
        }
        // Otherwise, check by enumeration if possible
        self.to_ord_set().is_some_and(|s| s.is_empty())
    }

    /// Convert to an eager OrdSet (only if LHS is enumerable)
    pub fn to_ord_set(&self) -> Option<OrdSet<Value>> {
        let iter = self.set1.iter_set()?;
        Some(
            iter.filter(|v| !self.set2.set_contains(v).unwrap_or(false))
                .collect(),
        )
    }

    /// Convert to an eager SortedSet (only if LHS is enumerable)
    pub fn to_sorted_set(&self) -> Option<SortedSet> {
        self.to_ord_set().map(|s| SortedSet::from_ord_set(&s))
    }

    /// Iterate over all elements (only if LHS is enumerable)
    pub fn iter(&self) -> Option<Box<dyn Iterator<Item = Value> + '_>> {
        let set = self.to_ord_set()?;
        Some(Box::new(set.into_iter()))
    }
}

impl fmt::Debug for SetDiffValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SetDiff({:?}, {:?})", self.set1, self.set2)
    }
}

impl Ord for SetDiffValue {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.set1.cmp(&other.set1) {
            Ordering::Equal => self.set2.cmp(&other.set2),
            ord => ord,
        }
    }
}

impl PartialOrd for SetDiffValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SetDiffValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for SetDiffValue {}

impl Hash for SetDiffValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        "SetDiff".hash(state);
        self.set1.hash(state);
        self.set2.hash(state);
    }
}

/// Lazy sequence set value representing Seq(S) - the set of all finite sequences over S
///
/// This is infinite in general, but supports membership checking:
/// v \in Seq(S) iff v is a sequence AND all elements of v are in S
#[derive(Clone)]
pub struct SeqSetValue {
    /// The base set S
    pub base: Box<Value>,
}

impl SeqSetValue {
    /// Create a new sequence set
    pub fn new(base: Value) -> Self {
        SeqSetValue {
            base: Box::new(base),
        }
    }

    /// Check if a value is in this sequence set
    /// v \in Seq(S) iff v is a sequence AND all elements are in S
    ///
    /// In TLA+, a sequence is a function from 1..n to some set.
    /// This means:
    /// - Value::Seq and Value::Tuple are explicit sequences
    /// - Value::IntFunc with domain 1..n is also a valid sequence
    /// - Value::Func with domain {1, 2, ..., n} is also a valid sequence
    pub fn contains(&self, v: &Value) -> bool {
        match v {
            Value::Seq(seq) => {
                // All elements must be in the base set
                seq.iter()
                    .all(|elem| self.base.set_contains(elem).unwrap_or(false))
            }
            Value::Tuple(elems) => {
                // All elements must be in the base set
                elems
                    .iter()
                    .all(|elem| self.base.set_contains(elem).unwrap_or(false))
            }
            // IntFunc with domain 1..n is a valid sequence
            Value::IntFunc(f) => {
                // Domain must start at 1 (sequence indexing in TLA+)
                if f.min != 1 {
                    return false;
                }
                // All values must be in the base set
                f.values
                    .iter()
                    .all(|elem| self.base.set_contains(elem).unwrap_or(false))
            }
            // Func with domain {1, 2, ..., n} is a valid sequence
            Value::Func(f) => {
                // Check if domain is exactly 1..n
                let entries = f.entries();
                let n = entries.len();
                if n == 0 {
                    // Empty function is the empty sequence
                    return true;
                }
                // Domain must be exactly {1, 2, ..., n}
                for (i, (k, _)) in entries.iter().enumerate() {
                    let expected = Value::SmallInt((i + 1) as i64);
                    if k != &expected {
                        return false;
                    }
                }
                // All values must be in the base set
                entries
                    .iter()
                    .all(|(_, v)| self.base.set_contains(v).unwrap_or(false))
            }
            _ => false,
        }
    }
}

impl fmt::Debug for SeqSetValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SeqSet({:?})", self.base)
    }
}

impl Ord for SeqSetValue {
    fn cmp(&self, other: &Self) -> Ordering {
        self.base.cmp(&other.base)
    }
}

impl PartialOrd for SeqSetValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SeqSetValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for SeqSetValue {}

impl Hash for SeqSetValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        "SeqSet".hash(state);
        self.base.hash(state);
    }
}

/// Lazy k-subset value representing Ksubsets(S, k) without allocating all C(n,k) subsets
///
/// This is a performance optimization that allows membership checking without enumeration.
/// Membership check: x \in Ksubsets(S, k) iff x is a set with |x| = k and x \subseteq S
#[derive(Clone)]
pub struct KSubsetValue {
    /// The base set S
    pub base: Box<Value>,
    /// The subset size k
    pub k: usize,
}

impl KSubsetValue {
    /// Create a new k-subset set
    pub fn new(base: Value, k: usize) -> Self {
        KSubsetValue {
            base: Box::new(base),
            k,
        }
    }

    /// Check if a value is in this k-subset set
    /// v \in Ksubsets(S, k) iff v is a set with |v| = k and v \subseteq S
    pub fn contains(&self, v: &Value) -> bool {
        // Must be a set with exactly k elements
        if let Some(set) = v.to_ord_set() {
            if set.len() != self.k {
                return false;
            }
            // All elements must be in base
            for elem in set.iter() {
                if !self.base.set_contains(elem).unwrap_or(false) {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    /// Check if enumerable (base must be enumerable)
    pub fn is_enumerable(&self) -> bool {
        self.base.iter_set().is_some()
    }

    /// Get the cardinality: C(n, k) = n! / (k! * (n-k)!)
    pub fn len(&self) -> Option<BigInt> {
        let n = self.base.set_len()?.to_usize()?;
        if self.k > n {
            return Some(BigInt::zero());
        }
        Some(binomial(n, self.k))
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len().is_some_and(|n| n.is_zero())
    }

    /// Convert to an eager OrdSet
    pub fn to_ord_set(&self) -> Option<OrdSet<Value>> {
        let base_set = self.base.to_ord_set()?;
        let elements: Vec<_> = base_set.into_iter().collect();
        let n = elements.len();

        if self.k > n {
            return Some(OrdSet::new());
        }

        let mut result = OrdSet::new();

        // k=0 special case: return set containing empty set
        if self.k == 0 {
            result.insert(Value::set(vec![]));
            return Some(result);
        }

        // Generate all k-combinations using iterative algorithm
        let mut indices: Vec<usize> = (0..self.k).collect();
        loop {
            // Create subset from current indices
            let subset: Vec<Value> = indices.iter().map(|&i| elements[i].clone()).collect();
            result.insert(Value::set(subset));

            // Find rightmost index that can be incremented
            let mut i = self.k;
            while i > 0 {
                i -= 1;
                if indices[i] < n - self.k + i {
                    break;
                }
            }
            if i == 0 && indices[0] >= n - self.k {
                break;
            }

            // Increment and reset subsequent indices
            indices[i] += 1;
            for j in (i + 1)..self.k {
                indices[j] = indices[j - 1] + 1;
            }
        }

        Some(result)
    }

    /// Convert to an eager SortedSet
    pub fn to_sorted_set(&self) -> Option<SortedSet> {
        self.to_ord_set().map(|s| SortedSet::from_ord_set(&s))
    }

    /// Iterate over all k-subsets
    pub fn iter(&self) -> Option<impl Iterator<Item = Value> + '_> {
        let base_set = self.base.to_ord_set()?;
        Some(KSubsetIterator::new(base_set, self.k))
    }
}

/// Iterator over k-element subsets
pub struct KSubsetIterator {
    elements: Vec<Value>,
    indices: Vec<usize>,
    k: usize,
    n: usize,
    done: bool,
}

impl KSubsetIterator {
    /// Create a new KSubsetIterator over all k-element subsets of the given set.
    pub fn new(base: OrdSet<Value>, k: usize) -> Self {
        let elements: Vec<Value> = base.into_iter().collect();
        let n = elements.len();
        let done = k > n;
        KSubsetIterator {
            elements,
            indices: if k <= n { (0..k).collect() } else { vec![] },
            k,
            n,
            done,
        }
    }

    /// Create a new KSubsetIterator from a SortedSet.
    pub fn from_sorted(base: &SortedSet, k: usize) -> Self {
        let elements: Vec<Value> = base.iter().cloned().collect();
        let n = elements.len();
        let done = k > n;
        KSubsetIterator {
            elements,
            indices: if k <= n { (0..k).collect() } else { vec![] },
            k,
            n,
            done,
        }
    }
}

impl Iterator for KSubsetIterator {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Special case k=0: return empty set once
        if self.k == 0 {
            self.done = true;
            return Some(Value::set(vec![]));
        }

        // Create subset from current indices
        let subset: Vec<Value> = self
            .indices
            .iter()
            .map(|&i| self.elements[i].clone())
            .collect();
        let result = Value::set(subset);

        // Find rightmost index that can be incremented
        let mut i = self.k;
        while i > 0 {
            i -= 1;
            if self.indices[i] < self.n - self.k + i {
                break;
            }
        }
        if i == 0 && self.indices[0] >= self.n - self.k {
            self.done = true;
            return Some(result);
        }

        // Increment and reset subsequent indices
        self.indices[i] += 1;
        for j in (i + 1)..self.k {
            self.indices[j] = self.indices[j - 1] + 1;
        }

        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            return (0, Some(0));
        }
        // Approximate: we can compute exact but it's complex
        (0, None)
    }
}

impl fmt::Debug for KSubsetValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KSubset({:?}, {})", self.base, self.k)
    }
}

impl Ord for KSubsetValue {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.k.cmp(&other.k) {
            Ordering::Equal => self.base.cmp(&other.base),
            ord => ord,
        }
    }
}

impl PartialOrd for KSubsetValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for KSubsetValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for KSubsetValue {}

impl Hash for KSubsetValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        "KSubset".hash(state);
        self.k.hash(state);
        self.base.hash(state);
    }
}

/// Compute binomial coefficient C(n, k) = n! / (k! * (n-k)!)
fn binomial(n: usize, k: usize) -> BigInt {
    if k > n {
        return BigInt::zero();
    }
    if k == 0 || k == n {
        return BigInt::one();
    }
    // Use smaller k for efficiency: C(n, k) = C(n, n-k)
    let k = k.min(n - k);
    let mut result = BigInt::one();
    for i in 0..k {
        result = result * BigInt::from(n - i) / BigInt::from(i + 1);
    }
    result
}

/// Lazy big union value representing UNION S without allocating all elements
///
/// This is a performance optimization for membership checking.
/// Membership check: x \in UNION S iff exists s \in S : x \in s
#[derive(Clone)]
pub struct UnionValue {
    /// The set of sets S
    pub set: Box<Value>,
}

impl UnionValue {
    /// Create a new lazy big union
    pub fn new(set: Value) -> Self {
        UnionValue { set: Box::new(set) }
    }

    /// Check if a value is in this union
    /// v \in UNION S iff exists s \in S : v \in s
    pub fn contains(&self, v: &Value) -> Option<bool> {
        // Need to enumerate the outer set
        let iter = self.set.iter_set()?;
        for inner_set in iter {
            if inner_set.set_contains(v).unwrap_or(false) {
                return Some(true);
            }
        }
        Some(false)
    }

    /// Check if enumerable (outer set and all inner sets must be enumerable)
    pub fn is_enumerable(&self) -> bool {
        if let Some(iter) = self.set.iter_set() {
            for inner_set in iter {
                if inner_set.iter_set().is_none() {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    /// Get the cardinality (requires enumeration)
    pub fn len(&self) -> Option<BigInt> {
        let set = self.to_ord_set()?;
        Some(BigInt::from(set.len()))
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        // Empty if outer is empty, or all inner sets are empty
        if let Some(iter) = self.set.iter_set() {
            for inner_set in iter {
                if !inner_set.set_len().map_or(true, |n| n.is_zero()) {
                    return false;
                }
            }
            true
        } else {
            // Non-enumerable outer set - assume not empty
            false
        }
    }

    /// Convert to an eager OrdSet
    pub fn to_ord_set(&self) -> Option<OrdSet<Value>> {
        let iter = self.set.iter_set()?;
        let mut result = OrdSet::new();
        for inner_set in iter {
            let inner_elements = inner_set.to_ord_set()?;
            for elem in inner_elements.into_iter() {
                result.insert(elem);
            }
        }
        Some(result)
    }

    /// Convert to an eager SortedSet
    pub fn to_sorted_set(&self) -> Option<SortedSet> {
        self.to_ord_set().map(|s| SortedSet::from_ord_set(&s))
    }

    /// Iterate over all elements
    pub fn iter(&self) -> Option<impl Iterator<Item = Value> + '_> {
        let set = self.to_ord_set()?;
        Some(set.into_iter())
    }
}

impl fmt::Debug for UnionValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Union({:?})", self.set)
    }
}

impl Ord for UnionValue {
    fn cmp(&self, other: &Self) -> Ordering {
        self.set.cmp(&other.set)
    }
}

impl PartialOrd for UnionValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for UnionValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for UnionValue {}

impl Hash for UnionValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        "Union".hash(state);
        self.set.hash(state);
    }
}

/// Global counter for generating unique SetPred IDs
static SET_PRED_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Lazy set filter ({x \in S : P(x)})
///
/// This value represents a filtered set without eagerly evaluating the predicate
/// for all elements. This is essential for filters over non-enumerable sets like STRING.
///
/// Membership checking requires an evaluation context, so `set_contains` returns None.
/// The actual membership check must be done in eval.rs using `check_set_pred_membership`.
///
/// Enumeration is only possible when the source set is enumerable.
#[derive(Clone)]
pub struct SetPredValue {
    /// Unique identifier for this set (for hashing/comparison)
    pub id: u64,
    /// The source set S
    pub source: Box<Value>,
    /// Bound variable for the filter (boxed to reduce Value enum size)
    pub bound: Box<BoundVar>,
    /// The predicate expression P(x)
    pub pred: Arc<Spanned<Expr>>,
    /// Captured environment at creation time
    pub env: HashMap<Arc<str>, Value>,
}

impl SetPredValue {
    /// Create a new lazy set filter
    pub fn new(
        source: Value,
        bound: BoundVar,
        pred: Spanned<Expr>,
        env: HashMap<Arc<str>, Value>,
    ) -> Self {
        SetPredValue {
            id: SET_PRED_ID_COUNTER.fetch_add(1, AtomicOrdering::SeqCst),
            source: Box::new(source),
            bound: Box::new(bound),
            pred: Arc::new(pred),
            env,
        }
    }

    /// Check if the source set is enumerable
    pub fn is_enumerable(&self) -> bool {
        self.source.iter_set().is_some()
    }

    /// Check if the source is a non-enumerable infinite set (STRING, AnySet)
    pub fn is_infinite(&self) -> bool {
        matches!(self.source.as_ref(), Value::StringSet | Value::AnySet)
    }

    /// Check if the element is in the source set (prerequisite for predicate check)
    /// Returns None if source can't do membership check
    pub fn source_contains(&self, v: &Value) -> Option<bool> {
        self.source.set_contains(v)
    }
}

impl fmt::Debug for SetPredValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SetPred({:?}, {:?})", self.source, self.bound.name)
    }
}

impl Ord for SetPredValue {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare by unique ID (identity comparison)
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for SetPredValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SetPredValue {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for SetPredValue {}

impl Hash for SetPredValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        "SetPred".hash(state);
        self.id.hash(state);
    }
}

/// Component domain type for multi-argument lazy functions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComponentDomain {
    Nat,
    Int,
    Real,
    String,
    Finite(OrdSet<Value>),
}

impl ComponentDomain {
    /// Check if a value is in this component domain
    pub fn contains(&self, v: &Value) -> bool {
        match self {
            ComponentDomain::Nat => match v {
                Value::SmallInt(n) => *n >= 0,
                Value::Int(n) => n >= &BigInt::zero(),
                _ => false,
            },
            ComponentDomain::Int => matches!(v, Value::SmallInt(_) | Value::Int(_)),
            ComponentDomain::Real => matches!(v, Value::SmallInt(_) | Value::Int(_)), // Int ⊆ Real
            ComponentDomain::String => matches!(v, Value::String(_)),
            ComponentDomain::Finite(s) => s.contains(v),
        }
    }

    /// Check if this domain is infinite
    pub fn is_infinite(&self) -> bool {
        matches!(
            self,
            ComponentDomain::Nat
                | ComponentDomain::Int
                | ComponentDomain::Real
                | ComponentDomain::String
        )
    }
}

/// Domain marker for lazy functions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LazyDomain {
    Nat,
    Int,
    Real,                          // Int ⊆ Real, TLC doesn't support actual reals
    String,                        // The set of all strings
    Product(Vec<ComponentDomain>), // Cartesian product of component domains
    /// General domain represented as an arbitrary set-like Value.
    /// Used for domains like SUBSET Int, [S -> T], etc.
    /// Membership is checked via Value::contains.
    General(Box<Value>),
}

/// A lazily evaluated function value for non-enumerable domains (e.g. Nat, Int).
///
/// This supports `f[x]` by evaluating the function body on-demand.
/// For multi-argument functions, stores multiple bounds and uses tuple keys.
#[derive(Clone)]
pub struct LazyFuncValue {
    pub id: u64,
    /// Optional name for self-recursive bindings (e.g., `LET f[n \\in Nat] == ... f[n-1] ...`)
    pub name: Option<Arc<str>>,
    pub domain: LazyDomain,
    /// Bound variables - single element for unary functions, multiple for multi-arg
    pub bounds: Vec<BoundVar>,
    pub body: Arc<Spanned<Expr>>,
    pub env: HashMap<Arc<str>, Value>,
    pub memo: Arc<Mutex<StdHashMap<Value, Value>>>,
}

/// A TLA+ function value
///
/// Functions in TLA+ are total mappings from a domain set to values.
/// [x \in S |-> e] creates a function with domain S.
///
/// Internally stored as a sorted array of (key, value) pairs for efficiency.
/// This avoids the allocation overhead of B-tree based persistent data structures.
pub struct FuncValue {
    /// Key-value pairs sorted by key
    /// Invariant: keys are unique and sorted by Value's Ord implementation
    entries: Arc<[(Value, Value)]>,
    /// Cached fingerprint for efficient state hashing
    /// Computed lazily on first access. Since FuncValue is immutable,
    /// the fingerprint remains valid for the lifetime of the value.
    /// Uses OnceLock for thread-safety in parallel model checking.
    cached_fp: std::sync::OnceLock<u64>,
}

/// Builder for constructing FuncValue incrementally.
///
/// Collects key-value pairs and sorts them when building the final FuncValue.
/// For ordered insertion (keys already sorted), use `build_presorted` for O(n).
pub struct FuncBuilder {
    entries: Vec<(Value, Value)>,
}

/// Array-backed function for small integer interval domains.
///
/// This is a performance optimization for functions with domain `a..b` (integer interval).
/// Instead of using OrdMap, values are stored in a contiguous array indexed by `(key - min)`.
/// This makes lookup O(1) and EXCEPT operations O(n) array clone instead of O(log n) B-tree ops,
/// which is faster for small domains due to eliminated allocation overhead.
///
/// Example: `pc = [i \in 1..4 |-> "V0"]` with N=4 creates:
/// - min=1, max=4, values=["V0", "V0", "V0", "V0"]
/// - `pc[2]` = `values[2-1]` = `values[1]` = "V0"
/// - `[pc EXCEPT ![3] = "AC"]` clones array and sets `values[2]` = "AC"
///
/// IntIntervalFunc uses `Arc<Vec<Value>>` to enable copy-on-write (COW) optimization.
/// When except() is called and we're the only owner, we can mutate in place
/// via Arc::make_mut() without cloning the entire array.
#[derive(Clone)]
pub struct IntIntervalFunc {
    /// Minimum domain element (inclusive)
    pub min: i64,
    /// Maximum domain element (inclusive)
    pub max: i64,
    /// Values array: `values[(key - min) as usize] = f[key]`
    /// Uses `Arc<Vec>` instead of `Arc<[T]>` to enable `Arc::make_mut` COW optimization.
    pub values: Arc<Vec<Value>>,
    /// Cached fingerprint for efficient state hashing
    cached_fp: std::sync::OnceLock<u64>,
}

/// Global counter for generating unique closure IDs
static CLOSURE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);
static LAZY_FUNC_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A closure value for higher-order operator arguments
///
/// Closures are created from LAMBDA expressions passed to operators
/// that take operator parameters (e.g., `ChooseOne(S, P(_))`).
#[derive(Clone)]
pub struct ClosureValue {
    /// Unique identifier for this closure (for hashing/comparison)
    pub id: u64,
    /// Parameter names from the lambda
    pub params: Vec<String>,
    /// Lambda body expression
    pub body: Arc<Spanned<Expr>>,
    /// Captured environment at closure creation time
    pub env: HashMap<Arc<str>, Value>,
}

impl ClosureValue {
    /// Create a new closure with a unique ID
    pub fn new(params: Vec<String>, body: Spanned<Expr>, env: HashMap<Arc<str>, Value>) -> Self {
        ClosureValue {
            id: CLOSURE_ID_COUNTER.fetch_add(1, AtomicOrdering::SeqCst),
            params,
            body: Arc::new(body),
            env,
        }
    }
}

/// Array-backed record for TLA+ records with string field names.
///
/// This is a performance optimization over `OrdMap<Arc<str>, Value>`.
/// Field name-value pairs are stored in a contiguous sorted array, providing:
/// - O(log n) field access via binary search
/// - O(n) iteration with excellent cache locality (no B-tree traversal)
/// - O(n) EXCEPT operations (clone array, modify in place)
/// - Cached fingerprint for efficient state hashing
///
/// For typical small records (3-10 fields), this is faster than OrdMap due to:
/// - No node allocations during iteration
/// - Better cache locality for all operations
/// - Reduced allocation overhead
///
/// Uses `Arc<Vec>` to enable copy-on-write (COW) optimization via `Arc::make_mut`.
#[derive(Clone)]
pub struct RecordValue {
    /// Field name-value pairs sorted by field name
    /// Invariant: field names are unique and sorted alphabetically
    entries: Arc<Vec<(Arc<str>, Value)>>,
    /// Cached fingerprint for efficient state hashing
    /// Computed lazily on first access. Since RecordValue is immutable,
    /// the fingerprint remains valid for the lifetime of the value.
    cached_fp: std::sync::OnceLock<u64>,
}

/// Builder for constructing RecordValue incrementally.
///
/// Collects field-value pairs and sorts them when building the final RecordValue.
/// For ordered insertion (fields already sorted), use `build_presorted` for O(n).
pub struct RecordBuilder {
    entries: Vec<(Arc<str>, Value)>,
}

impl RecordValue {
    /// Create an empty record
    pub fn new() -> Self {
        RecordValue {
            entries: Arc::new(Vec::new()),
            cached_fp: std::sync::OnceLock::new(),
        }
    }

    /// Create a record from pre-sorted field-value pairs
    /// Caller must ensure entries are sorted by field name
    pub fn from_sorted_entries(entries: Vec<(Arc<str>, Value)>) -> Self {
        RecordValue {
            entries: Arc::new(entries),
            cached_fp: std::sync::OnceLock::new(),
        }
    }

    /// Get the number of fields
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the record is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get a field value by name (O(log n) via binary search)
    #[inline]
    pub fn get(&self, field: &str) -> Option<&Value> {
        self.entries
            .binary_search_by(|(k, _)| k.as_ref().cmp(field))
            .ok()
            .map(|idx| &self.entries[idx].1)
    }

    /// Check if a field exists
    #[inline]
    pub fn contains_key(&self, field: &str) -> bool {
        self.entries
            .binary_search_by(|(k, _)| k.as_ref().cmp(field))
            .is_ok()
    }

    /// Iterate over field names
    pub fn keys(&self) -> impl Iterator<Item = &Arc<str>> {
        self.entries.iter().map(|(k, _)| k)
    }

    /// Iterate over field values
    pub fn values(&self) -> impl Iterator<Item = &Value> {
        self.entries.iter().map(|(_, v)| v)
    }

    /// Iterate over (field_name, value) pairs
    pub fn iter(&self) -> impl Iterator<Item = (&Arc<str>, &Value)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    /// Create a new record with one field updated (EXCEPT operation)
    /// Returns None if the field doesn't exist (matches TLC behavior)
    pub fn with_field(&self, field: Arc<str>, value: Value) -> Option<Self> {
        let idx = self
            .entries
            .binary_search_by(|(k, _)| k.as_ref().cmp(field.as_ref()))
            .ok()?;

        let mut new_entries = Arc::clone(&self.entries);
        Arc::make_mut(&mut new_entries)[idx].1 = value;
        Some(RecordValue {
            entries: new_entries,
            cached_fp: std::sync::OnceLock::new(),
        })
    }

    /// Create a new record with one field inserted or updated
    /// Unlike with_field, this will insert the field if it doesn't exist
    pub fn insert(&self, field: Arc<str>, value: Value) -> Self {
        let mut new_entries = Arc::clone(&self.entries);
        let entries = Arc::make_mut(&mut new_entries);

        match entries.binary_search_by(|(k, _)| k.as_ref().cmp(field.as_ref())) {
            Ok(idx) => entries[idx].1 = value,
            Err(idx) => entries.insert(idx, (field, value)),
        }

        RecordValue {
            entries: new_entries,
            cached_fp: std::sync::OnceLock::new(),
        }
    }

    /// Alias for insert (OrdMap compatibility)
    #[inline]
    pub fn update(&self, field: Arc<str>, value: Value) -> Self {
        self.insert(field, value)
    }

    /// Get cached fingerprint if available
    pub fn get_cached_fingerprint(&self) -> Option<u64> {
        self.cached_fp.get().copied()
    }

    /// Cache a fingerprint value. Returns the cached value (possibly different
    /// if another thread raced to set it first).
    pub fn cache_fingerprint(&self, fp: u64) -> u64 {
        *self.cached_fp.get_or_init(|| fp)
    }

    /// Convert to OrdMap (for compatibility with code expecting OrdMap)
    pub fn to_ord_map(&self) -> OrdMap<Arc<str>, Value> {
        self.entries
            .iter()
            .map(|(k, v)| (Arc::clone(k), v.clone()))
            .collect()
    }

    /// Create from OrdMap (for migration)
    pub fn from_ord_map(map: &OrdMap<Arc<str>, Value>) -> Self {
        let entries: Vec<_> = map
            .iter()
            .map(|(k, v)| (Arc::clone(k), v.clone()))
            .collect();
        RecordValue {
            entries: Arc::new(entries),
            cached_fp: std::sync::OnceLock::new(),
        }
    }

    /// Check if two RecordValues share the same underlying storage (pointer equality)
    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.entries, &other.entries)
    }
}

impl From<Vec<(String, Value)>> for RecordValue {
    fn from(entries: Vec<(String, Value)>) -> Self {
        entries
            .into_iter()
            .map(|(k, v)| (Arc::from(k), v))
            .collect()
    }
}

impl Default for RecordValue {
    fn default() -> Self {
        Self::new()
    }
}

impl RecordBuilder {
    /// Create a new empty builder
    pub fn new() -> Self {
        RecordBuilder {
            entries: Vec::new(),
        }
    }

    /// Create a new builder with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        RecordBuilder {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Add a field-value pair
    pub fn insert(&mut self, field: Arc<str>, value: Value) {
        self.entries.push((field, value));
    }

    /// Build the RecordValue, sorting entries by field name
    pub fn build(mut self) -> RecordValue {
        self.entries.sort_by(|(a, _), (b, _)| a.cmp(b));
        RecordValue::from_sorted_entries(self.entries)
    }

    /// Build the RecordValue without sorting (entries must already be sorted)
    pub fn build_presorted(self) -> RecordValue {
        RecordValue::from_sorted_entries(self.entries)
    }
}

impl Default for RecordBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for RecordValue {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: pointer equality
        if Arc::ptr_eq(&self.entries, &other.entries) {
            return true;
        }
        self.entries == other.entries
    }
}

impl Eq for RecordValue {}

impl PartialOrd for RecordValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RecordValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Fast path: pointer equality
        if Arc::ptr_eq(&self.entries, &other.entries) {
            return std::cmp::Ordering::Equal;
        }
        self.entries.cmp(&other.entries)
    }
}

impl std::hash::Hash for RecordValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.entries.hash(state)
    }
}

impl std::fmt::Debug for RecordValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.entries.iter().map(|(k, v)| (k.as_ref(), v)))
            .finish()
    }
}

impl<S: Into<Arc<str>>> FromIterator<(S, Value)> for RecordValue {
    fn from_iter<I: IntoIterator<Item = (S, Value)>>(iter: I) -> Self {
        let mut entries: Vec<_> = iter.into_iter().map(|(k, v)| (k.into(), v)).collect();
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));
        RecordValue::from_sorted_entries(entries)
    }
}

impl<'a> IntoIterator for &'a RecordValue {
    type Item = (&'a Arc<str>, &'a Value);
    type IntoIter = std::iter::Map<
        std::slice::Iter<'a, (Arc<str>, Value)>,
        fn(&'a (Arc<str>, Value)) -> (&'a Arc<str>, &'a Value),
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter().map(|(k, v)| (k, v))
    }
}

impl LazyFuncValue {
    /// Create a new lazy function with a single bound variable
    pub fn new(
        name: Option<Arc<str>>,
        domain: LazyDomain,
        bound: BoundVar,
        body: Spanned<Expr>,
        env: HashMap<Arc<str>, Value>,
    ) -> Self {
        #[cfg(feature = "memory-stats")]
        memory_stats::inc_lazy_func();

        LazyFuncValue {
            id: LAZY_FUNC_ID_COUNTER.fetch_add(1, AtomicOrdering::SeqCst),
            name,
            domain,
            bounds: vec![bound],
            body: Arc::new(body),
            env,
            memo: Arc::new(Mutex::new(StdHashMap::new())),
        }
    }

    /// Create a new lazy function with multiple bound variables (for multi-arg functions)
    pub fn new_multi(
        name: Option<Arc<str>>,
        domain: LazyDomain,
        bounds: Vec<BoundVar>,
        body: Spanned<Expr>,
        env: HashMap<Arc<str>, Value>,
    ) -> Self {
        #[cfg(feature = "memory-stats")]
        memory_stats::inc_lazy_func();

        LazyFuncValue {
            id: LAZY_FUNC_ID_COUNTER.fetch_add(1, AtomicOrdering::SeqCst),
            name,
            domain,
            bounds,
            body: Arc::new(body),
            env,
            memo: Arc::new(Mutex::new(StdHashMap::new())),
        }
    }

    /// Create a new LazyFunc with an exception value pre-loaded into the memo.
    /// Used for EXCEPT on lazy functions: [f EXCEPT ![k] = v]
    ///
    /// Note: clippy warns about mutable_key_type because Value contains LazyFuncValue
    /// which has interior mutability (memo cache). This is safe because:
    /// 1. Value's Hash/Eq use the stable `id` field, not the mutable memo
    /// 2. The memo is purely a cache that doesn't affect semantic equality
    #[allow(clippy::mutable_key_type)]
    pub fn with_exception(&self, key: Value, value: Value) -> Self {
        // Clone the existing memo and add the exception
        let new_memo = {
            let memo = self.memo.lock().unwrap();
            let mut new_map = memo.clone();
            new_map.insert(key, value);
            Arc::new(Mutex::new(new_map))
        };
        LazyFuncValue {
            id: LAZY_FUNC_ID_COUNTER.fetch_add(1, AtomicOrdering::SeqCst),
            name: self.name.clone(),
            domain: self.domain.clone(),
            bounds: self.bounds.clone(),
            body: self.body.clone(),
            env: self.env.clone(),
            memo: new_memo,
        }
    }

    /// Check if an argument is in the domain
    pub fn in_domain(&self, arg: &Value) -> bool {
        match &self.domain {
            LazyDomain::Nat => match arg {
                Value::SmallInt(n) => *n >= 0,
                Value::Int(n) => n >= &BigInt::zero(),
                _ => false,
            },
            LazyDomain::Int => matches!(arg, Value::SmallInt(_) | Value::Int(_)),
            LazyDomain::Real => matches!(arg, Value::SmallInt(_) | Value::Int(_)),
            LazyDomain::String => matches!(arg, Value::String(_)),
            LazyDomain::Product(components) => {
                // For product domains, arg should be a tuple with matching components
                match arg {
                    Value::Tuple(elems) if elems.len() == components.len() => elems
                        .iter()
                        .zip(components.iter())
                        .all(|(v, d)| d.contains(v)),
                    _ => false,
                }
            }
            LazyDomain::General(domain_val) => {
                // For general domains, use Value::set_contains for membership check
                domain_val.set_contains(arg).unwrap_or(false)
            }
        }
    }
}

impl Value {
    // === Constructors ===

    /// Create a boolean value
    pub fn bool(b: bool) -> Self {
        Value::Bool(b)
    }

    /// Create an integer value from i64 (uses SmallInt fast path)
    pub fn int(n: i64) -> Self {
        Value::SmallInt(n)
    }

    /// Create an integer value from BigInt
    /// Normalizes to SmallInt if the value fits in i64
    pub fn big_int(n: BigInt) -> Self {
        if let Some(small) = n.to_i64() {
            Value::SmallInt(small)
        } else {
            Value::Int(n)
        }
    }

    /// Create a string value
    pub fn string(s: impl Into<Arc<str>>) -> Self {
        Value::String(s.into())
    }

    /// Create an empty set
    pub fn empty_set() -> Self {
        Value::Set(SortedSet::new())
    }

    /// Create a set from an iterator
    pub fn set(values: impl IntoIterator<Item = Value>) -> Self {
        Value::Set(SortedSet::from_iter(values))
    }

    /// Create a sequence from values (0-indexed internally, 1-indexed in TLA+)
    pub fn seq(values: impl IntoIterator<Item = Value>) -> Self {
        Value::Seq(values.into_iter().collect())
    }

    /// Create a tuple from values
    pub fn tuple(values: impl IntoIterator<Item = Value>) -> Self {
        Value::Tuple(values.into_iter().collect())
    }

    /// Create a record from field-value pairs
    pub fn record(fields: impl IntoIterator<Item = (impl Into<Arc<str>>, Value)>) -> Self {
        Value::Record(fields.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    /// Create a function from a domain and mapping
    pub fn func(domain: OrdSet<Value>, mapping: OrdMap<Value, Value>) -> Self {
        Value::Func(FuncValue::new(domain, mapping))
    }

    /// Create an interned model value.
    ///
    /// Uses the string intern table to ensure pointer equality for repeated model values.
    /// This enables O(1) comparisons via Arc pointer equality.
    pub fn model_value(name: &str) -> Self {
        Value::ModelValue(intern_string(name))
    }

    // === Type predicates ===

    pub fn is_bool(&self) -> bool {
        matches!(self, Value::Bool(_))
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Value::String(_))
    }

    pub fn is_set(&self) -> bool {
        match self {
            Value::Set(_)
            | Value::Interval(_)
            | Value::Subset(_)
            | Value::FuncSet(_)
            | Value::RecordSet(_)
            | Value::TupleSet(_)
            | Value::SetCup(_)
            | Value::SetCap(_)
            | Value::SetDiff(_)
            | Value::SetPred(_)
            | Value::KSubset(_)
            | Value::BigUnion(_)
            | Value::StringSet
            | Value::AnySet
            | Value::SeqSet(_) => true,
            // ModelValue for infinite sets (Nat, Int, Real)
            Value::ModelValue(name) => matches!(name.as_ref(), "Nat" | "Int" | "Real"),
            _ => false,
        }
    }

    /// Check if this value is a lazy interval
    pub fn is_interval(&self) -> bool {
        matches!(self, Value::Interval(_))
    }

    /// Extract interval value
    pub fn as_interval(&self) -> Option<&IntervalValue> {
        match self {
            Value::Interval(i) => Some(i),
            _ => None,
        }
    }

    pub fn is_func(&self) -> bool {
        matches!(self, Value::Func(_) | Value::LazyFunc(_))
    }

    pub fn is_seq(&self) -> bool {
        matches!(self, Value::Seq(_))
    }

    pub fn is_record(&self) -> bool {
        matches!(self, Value::Record(_))
    }

    pub fn is_tuple(&self) -> bool {
        matches!(self, Value::Tuple(_))
    }

    /// Returns true if this value is a sequence or tuple (both are indexed collections)
    pub fn is_seq_or_tuple(&self) -> bool {
        matches!(self, Value::Seq(_) | Value::Tuple(_))
    }

    /// Extract elements from a Seq or Tuple.
    /// Both Seq and Tuple are indexed collections in TLA+.
    /// Returns Cow::Borrowed for Tuple (which uses Arc<[Value]>)
    /// and Cow::Owned for Seq (which uses im::Vector).
    #[inline]
    pub fn as_seq_or_tuple_elements(&self) -> Option<Cow<'_, [Value]>> {
        match self {
            Value::Seq(s) => Some(Cow::Owned(s.to_vec())),
            Value::Tuple(t) => Some(Cow::Borrowed(t.as_ref())),
            _ => None,
        }
    }

    /// Extract the SeqValue from a Seq
    #[inline]
    pub fn as_seq_value(&self) -> Option<&SeqValue> {
        match self {
            Value::Seq(s) => Some(s),
            _ => None,
        }
    }

    /// Extract elements from a Tuple
    #[inline]
    pub fn as_tuple_elements(&self) -> Option<&[Value]> {
        match self {
            Value::Tuple(t) => Some(t.as_ref()),
            _ => None,
        }
    }

    // === Extractors ===

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns a reference to the BigInt if this is a BigInt variant.
    /// For SmallInt, returns None (use to_bigint() or as_i64() instead).
    /// Returns the integer value as BigInt.
    /// For SmallInt, creates a new BigInt; for Int, clones the existing one.
    /// Returns None if not an integer type.
    pub fn as_int(&self) -> Option<BigInt> {
        match self {
            Value::SmallInt(n) => Some(BigInt::from(*n)),
            Value::Int(n) => Some(n.clone()),
            _ => None,
        }
    }

    /// Returns the integer value as i64 if it fits.
    /// Fast path for SmallInt, conversion for BigInt.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::SmallInt(n) => Some(*n),
            Value::Int(n) => n.to_i64(),
            _ => None,
        }
    }

    /// Returns true if this value is an integer (SmallInt or BigInt).
    pub fn is_int(&self) -> bool {
        matches!(self, Value::SmallInt(_) | Value::Int(_))
    }

    /// Converts the integer to BigInt (owned).
    /// Returns None if not an integer type.
    pub fn to_bigint(&self) -> Option<BigInt> {
        match self {
            Value::SmallInt(n) => Some(BigInt::from(*n)),
            Value::Int(n) => Some(n.clone()),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_set(&self) -> Option<&SortedSet> {
        match self {
            Value::Set(s) => Some(s),
            _ => None,
        }
    }

    /// Check if this value (as a set) contains another value
    /// Works for Set, Interval, Subset, FuncSet, RecordSet, TupleSet, SetCup, SetCap, SetDiff, KSubset, BigUnion, SeqSet types
    pub fn set_contains(&self, v: &Value) -> Option<bool> {
        match self {
            Value::Set(s) => Some(s.contains(v)),
            Value::Interval(iv) => Some(iv.contains(v)),
            Value::Subset(sv) => Some(sv.contains(v)),
            Value::FuncSet(fsv) => Some(fsv.contains(v)),
            Value::RecordSet(rsv) => Some(rsv.contains(v)),
            Value::TupleSet(tsv) => Some(tsv.contains(v)),
            Value::SetCup(scv) => Some(scv.contains(v)),
            Value::SetCap(scv) => Some(scv.contains(v)),
            Value::SetDiff(sdv) => Some(sdv.contains(v)),
            Value::KSubset(ksv) => Some(ksv.contains(v)),
            Value::BigUnion(uv) => uv.contains(v),
            Value::SeqSet(ssv) => Some(ssv.contains(v)),
            // StringSet contains all strings
            Value::StringSet => Some(matches!(v, Value::String(_))),
            // AnySet contains all values
            Value::AnySet => Some(true),
            // ModelValue for infinite sets (Nat, Int, Real)
            Value::ModelValue(name) => match name.as_ref() {
                "Nat" => Some(match v {
                    Value::SmallInt(n) => *n >= 0,
                    Value::Int(n) => n >= &BigInt::zero(),
                    _ => false,
                }),
                "Int" => Some(matches!(v, Value::SmallInt(_) | Value::Int(_))),
                "Real" => Some(matches!(v, Value::SmallInt(_) | Value::Int(_))), // Int ⊆ Real
                _ => None, // Other model values are not sets
            },
            _ => None,
        }
    }

    /// Convert this set-like value to a SortedSet
    /// Works for Set, Interval, Subset, FuncSet, RecordSet, TupleSet, SetCup, SetCap, SetDiff, KSubset, BigUnion types
    pub fn to_sorted_set(&self) -> Option<SortedSet> {
        match self {
            Value::Set(s) => Some(s.clone()),
            Value::Interval(iv) => Some(iv.to_sorted_set()),
            Value::Subset(sv) => sv.to_sorted_set(),
            Value::FuncSet(fsv) => fsv.to_sorted_set(),
            Value::RecordSet(rsv) => rsv.to_sorted_set(),
            Value::TupleSet(tsv) => tsv.to_sorted_set(),
            Value::SetCup(scv) => scv.to_sorted_set(),
            Value::SetCap(scv) => scv.to_sorted_set(),
            Value::SetDiff(sdv) => sdv.to_sorted_set(),
            Value::KSubset(ksv) => ksv.to_sorted_set(),
            Value::BigUnion(uv) => uv.to_sorted_set(),
            _ => None,
        }
    }

    /// Convert this set-like value to an OrdSet (for compatibility)
    /// Works for Set, Interval, Subset, FuncSet, RecordSet, TupleSet, SetCup, SetCap, SetDiff, KSubset, BigUnion types
    pub fn to_ord_set(&self) -> Option<OrdSet<Value>> {
        self.to_sorted_set().map(|s| s.to_ord_set())
    }

    /// Get the number of elements in this set-like value
    pub fn set_len(&self) -> Option<BigInt> {
        match self {
            Value::Set(s) => Some(BigInt::from(s.len())),
            Value::Interval(iv) => Some(iv.len()),
            Value::Subset(sv) => sv.len(),
            Value::FuncSet(fsv) => fsv.len(),
            Value::RecordSet(rsv) => rsv.len(),
            Value::TupleSet(tsv) => tsv.len(),
            Value::SetCup(scv) => scv.len(),
            Value::SetCap(scv) => scv.len(),
            Value::SetDiff(sdv) => sdv.len(),
            Value::KSubset(ksv) => ksv.len(),
            Value::BigUnion(uv) => uv.len(),
            _ => None,
        }
    }

    /// Iterate over this set-like value
    /// Returns boxed iterator for Set, Interval, Subset, FuncSet, RecordSet, TupleSet,
    /// SetCup, SetCap, SetDiff, KSubset, and BigUnion types (when enumerable)
    pub fn iter_set(&self) -> Option<Box<dyn Iterator<Item = Value> + '_>> {
        match self {
            Value::Set(s) => Some(Box::new(s.iter().cloned())),
            Value::Interval(iv) => Some(Box::new(iv.iter_values())),
            Value::Subset(sv) => {
                // For Subset, we need to eagerly convert and iterate
                let iter = sv.iter()?;
                Some(Box::new(iter))
            }
            Value::FuncSet(fsv) => {
                // For FuncSet, we need to eagerly convert and iterate
                let iter = fsv.iter()?;
                Some(Box::new(iter))
            }
            Value::RecordSet(rsv) => {
                let iter = rsv.iter()?;
                Some(Box::new(iter))
            }
            Value::TupleSet(tsv) => {
                let iter = tsv.iter()?;
                Some(Box::new(iter))
            }
            Value::SetCup(scv) => {
                // SetCup needs both operands to be enumerable
                let set = scv.to_ord_set()?;
                Some(Box::new(set.into_iter()))
            }
            Value::SetCap(scv) => {
                // SetCap can iterate if at least one operand is enumerable
                scv.iter()
            }
            Value::SetDiff(sdv) => {
                // SetDiff can iterate if LHS is enumerable
                sdv.iter()
            }
            Value::KSubset(ksv) => {
                // KSubset can iterate if base is enumerable
                let iter = ksv.iter()?;
                Some(Box::new(iter))
            }
            Value::BigUnion(uv) => {
                // BigUnion can iterate if outer and inner sets are enumerable
                let iter = uv.iter()?;
                Some(Box::new(iter))
            }
            _ => None,
        }
    }

    pub fn as_func(&self) -> Option<&FuncValue> {
        match self {
            Value::Func(f) => Some(f),
            _ => None,
        }
    }

    /// Coerce function-like values (Func, Tuple, Seq, IntFunc, Record) to FuncValue.
    /// This is used for operations that accept any function-like type (e.g., Bags).
    /// Returns None for non-function-like types.
    pub fn to_func_coerced(&self) -> Option<FuncValue> {
        match self {
            Value::Func(f) => Some(f.clone()),
            Value::IntFunc(f) => Some(f.to_func_value()),
            Value::Tuple(elems) => {
                // Convert to function with domain 1..n
                let entries: Vec<(Value, Value)> = elems
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (Value::SmallInt((i + 1) as i64), v.clone()))
                    .collect();
                Some(FuncValue::from_sorted_entries(entries))
            }
            Value::Seq(seq) => {
                // Convert to function with domain 1..n
                let entries: Vec<(Value, Value)> = seq
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (Value::SmallInt((i + 1) as i64), v.clone()))
                    .collect();
                Some(FuncValue::from_sorted_entries(entries))
            }
            Value::Record(rec) => {
                // Convert to function with string domain
                let mut entries: Vec<(Value, Value)> = rec
                    .iter()
                    .map(|(k, v)| (Value::String(k.clone()), v.clone()))
                    .collect();
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                Some(FuncValue::from_sorted_entries(entries))
            }
            _ => None,
        }
    }

    pub fn as_seq(&self) -> Option<Cow<'_, [Value]>> {
        // In TLA+, sequences and tuples share the same <<...>> syntax
        // and sequence operations work on both
        self.as_seq_or_tuple_elements()
    }

    pub fn as_record(&self) -> Option<&RecordValue> {
        match self {
            Value::Record(r) => Some(r),
            _ => None,
        }
    }

    pub fn as_tuple(&self) -> Option<&Arc<[Value]>> {
        match self {
            Value::Tuple(t) => Some(t),
            _ => None,
        }
    }

    // === Symmetry: Value permutation ===

    /// Apply a permutation function to all model values in this value.
    /// The permutation should be a function from model values to model values.
    /// Used for symmetry reduction in model checking.
    pub fn permute(&self, perm: &FuncValue) -> Value {
        match self {
            // Primitive values unchanged
            Value::Bool(_) | Value::SmallInt(_) | Value::Int(_) | Value::String(_) => self.clone(),

            // Model values: apply the permutation
            Value::ModelValue(name) => {
                let mv = Value::ModelValue(name.clone());
                // Look up in permutation function
                if let Some(permuted) = perm.mapping_get(&mv) {
                    permuted.clone()
                } else {
                    // Not in permutation domain - keep unchanged
                    mv
                }
            }

            // Sets: permute all elements
            Value::Set(s) => {
                let permuted: OrdSet<Value> = s.iter().map(|v| v.permute(perm)).collect();
                Value::Set(SortedSet::from_ord_set(&permuted))
            }

            // Sequences: permute all elements
            Value::Seq(s) => {
                let permuted: Vec<Value> = s.iter().map(|v| v.permute(perm)).collect();
                Value::Seq(permuted.into())
            }

            // Tuples: permute all elements
            Value::Tuple(t) => {
                let permuted: Vec<Value> = t.iter().map(|v| v.permute(perm)).collect();
                Value::Tuple(permuted.into())
            }

            // Records: permute all field values (keys are strings, unchanged)
            Value::Record(r) => {
                let permuted: RecordValue = r
                    .iter()
                    .map(|(k, v)| (Arc::clone(k), v.permute(perm)))
                    .collect();
                Value::Record(permuted)
            }

            // Functions: permute both domain and range values
            Value::Func(f) => {
                let permuted_domain: OrdSet<Value> =
                    f.domain_iter().map(|v| v.permute(perm)).collect();
                let permuted_mapping: OrdMap<Value, Value> = f
                    .mapping_iter()
                    .map(|(k, v)| (k.permute(perm), v.permute(perm)))
                    .collect();
                Value::Func(FuncValue::new(permuted_domain, permuted_mapping))
            }

            // IntFunc: domain is integers (unchanged), permute only values
            Value::IntFunc(f) => {
                let permuted_values: Vec<Value> =
                    f.values.iter().map(|v| v.permute(perm)).collect();
                Value::IntFunc(IntIntervalFunc::new(f.min, f.max, permuted_values))
            }

            // Lazy values: eagerly evaluate then permute
            // (For symmetry reduction, we need concrete values)
            Value::Interval(iv) => {
                // Intervals are integers, no model values
                Value::Interval(iv.clone())
            }

            // Other lazy types: convert to concrete and permute
            // Note: For efficiency, symmetry should be applied to states which
            // typically have concrete values, not lazy sets
            Value::Subset(sv) => {
                if let Some(set) = sv.to_ord_set() {
                    let permuted: OrdSet<Value> = set.iter().map(|v| v.permute(perm)).collect();
                    Value::Set(SortedSet::from_ord_set(&permuted))
                } else {
                    self.clone()
                }
            }

            // These lazy types typically shouldn't appear in states
            Value::FuncSet(_)
            | Value::RecordSet(_)
            | Value::TupleSet(_)
            | Value::SetCup(_)
            | Value::SetCap(_)
            | Value::SetDiff(_)
            | Value::SetPred(_)
            | Value::KSubset(_)
            | Value::BigUnion(_)
            | Value::StringSet
            | Value::AnySet
            | Value::SeqSet(_)
            | Value::LazyFunc(_)
            | Value::Closure(_) => self.clone(),
        }
    }

    // === Type name for error messages ===

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Bool(_) => "BOOLEAN",
            Value::SmallInt(_) | Value::Int(_) => "Int",
            Value::String(_) => "STRING",
            Value::Set(_)
            | Value::Interval(_)
            | Value::Subset(_)
            | Value::FuncSet(_)
            | Value::RecordSet(_)
            | Value::TupleSet(_)
            | Value::SetCup(_)
            | Value::SetCap(_)
            | Value::SetDiff(_)
            | Value::SetPred(_)
            | Value::KSubset(_)
            | Value::BigUnion(_)
            | Value::StringSet
            | Value::AnySet
            | Value::SeqSet(_) => "Set",
            Value::Func(_) | Value::IntFunc(_) | Value::LazyFunc(_) => "Function",
            Value::Seq(_) => "Seq",
            Value::Record(_) => "Record",
            Value::Tuple(_) => "Tuple",
            Value::ModelValue(_) => "ModelValue",
            Value::Closure(_) => "Closure",
        }
    }

    /// Check if this value is a closure
    pub fn is_closure(&self) -> bool {
        matches!(self, Value::Closure(_))
    }

    /// Extract closure value
    pub fn as_closure(&self) -> Option<&ClosureValue> {
        match self {
            Value::Closure(c) => Some(c),
            _ => None,
        }
    }

    /// Extend a fingerprint with this value using TLC-compatible FP64 algorithm.
    ///
    /// This method implements TLC's incremental fingerprinting approach where
    /// fingerprints are extended one component at a time. This is critical for
    /// performance when modifying large composite values like functions.
    ///
    /// Each value type follows TLC's fingerprinting pattern:
    /// 1. Extend with type tag (from ValueConstants)
    /// 2. Extend with length/size if applicable
    /// 3. Extend with component values recursively
    ///
    /// # Arguments
    /// * `fp` - The current fingerprint to extend
    ///
    /// # Returns
    /// The extended fingerprint including this value
    pub fn fingerprint_extend(&self, fp: u64) -> u64 {
        use crate::fingerprint::{
            fp64_extend_bigint, fp64_extend_i32, fp64_extend_i64, fp64_extend_str, value_tags::*,
        };

        match self {
            Value::Bool(b) => {
                // TLC: fp = FP64.Extend(fp, BOOLVALUE); fp = FP64.Extend(fp, (val) ? 't' : 'f');
                let fp = fp64_extend_i64(fp, BOOLVALUE);
                // TLC uses 't' or 'f' as char, which is extended as a single byte
                let c = if *b { b't' } else { b'f' };
                crate::fingerprint::fp64_extend_byte(fp, c)
            }
            Value::SmallInt(n) => {
                // TLC: fp = FP64.Extend(FP64.Extend(fp, INTVALUE), this.val);
                let fp = fp64_extend_i64(fp, INTVALUE);
                // TLC's IntValue uses int (32-bit), but we may have larger values
                if *n >= i32::MIN as i64 && *n <= i32::MAX as i64 {
                    fp64_extend_i32(fp, *n as i32)
                } else {
                    fp64_extend_i64(fp, *n)
                }
            }
            Value::Int(n) => {
                let fp = fp64_extend_i64(fp, INTVALUE);
                fp64_extend_bigint(fp, n)
            }
            Value::String(s) => {
                // TLC: fp = FP64.Extend(fp, STRINGVALUE); fp = FP64.Extend(fp, len); fp = FP64.Extend(fp, str);
                let fp = fp64_extend_i64(fp, STRINGVALUE);
                let fp = fp64_extend_i32(fp, s.len() as i32);
                fp64_extend_str(fp, s)
            }
            Value::Set(set) => {
                // TLC SetEnumValue: fp = FP64.Extend(fp, SETENUMVALUE); fp = FP64.Extend(fp, sz);
                // for each elem: fp = elem.fingerPrint(fp);
                let mut fp = fp64_extend_i64(fp, SETENUMVALUE);
                fp = fp64_extend_i32(fp, set.len() as i32);
                for elem in set.iter() {
                    fp = elem.fingerprint_extend(fp);
                }
                fp
            }
            Value::Interval(intv) => {
                // TLC IntervalValue fingerprints as SETENUMVALUE (a set)
                // Each element is fingerprinted as INTVALUE
                let mut fp = fp64_extend_i64(fp, SETENUMVALUE);
                let len = &intv.high - &intv.low + num_bigint::BigInt::from(1);
                let len_i32 = len.to_i32().unwrap_or(i32::MAX);
                fp = fp64_extend_i32(fp, len_i32);
                for val in intv.iter_values() {
                    fp = val.fingerprint_extend(fp);
                }
                fp
            }
            Value::Func(func) => {
                // TLC FcnRcdValue: fp = FP64.Extend(fp, FCNRCDVALUE); fp = FP64.Extend(fp, flen);
                // for each (domain, value): fp = domain.fingerPrint(fp); fp = values.fingerPrint(fp);
                let mut fp = fp64_extend_i64(fp, FCNRCDVALUE);
                let entries = func.entries();
                fp = fp64_extend_i32(fp, entries.len() as i32);
                for (key, val) in entries.iter() {
                    fp = key.fingerprint_extend(fp);
                    fp = val.fingerprint_extend(fp);
                }
                fp
            }
            Value::IntFunc(func) => {
                // IntFunc is semantically a function, fingerprint as FCNRCDVALUE
                // Domain is min..max (integers)
                let mut fp = fp64_extend_i64(fp, FCNRCDVALUE);
                let len = func.values.len();
                fp = fp64_extend_i32(fp, len as i32);
                // For IntIntervalFunc, domain is min..max
                for (i, val) in func.values.iter().enumerate() {
                    // Domain key: min + i
                    let key = func.min + i as i64;
                    let fp_tmp = fp64_extend_i64(fp, INTVALUE);
                    let fp_tmp = if key >= i32::MIN as i64 && key <= i32::MAX as i64 {
                        fp64_extend_i32(fp_tmp, key as i32)
                    } else {
                        fp64_extend_i64(fp_tmp, key)
                    };
                    fp = val.fingerprint_extend(fp_tmp);
                }
                fp
            }
            Value::LazyFunc(lazy) => {
                // Lazy functions cannot be enumerated without evaluation context.
                // They are identified by their unique ID for fingerprinting purposes.
                // This is a rare case - most functions become Func or IntFunc during evaluation.
                let fp = fp64_extend_i64(fp, FCNLAMBDAVALUE);
                fp64_extend_i64(fp, lazy.id as i64)
            }
            Value::Tuple(elems) => {
                // TLC TupleValue: fingerprinted as FCNRCDVALUE with integer domain 1..n
                // fp = FP64.Extend(fp, FCNRCDVALUE); fp = FP64.Extend(fp, len);
                // for i in 1..len: fp = FP64.Extend(fp, INTVALUE); fp = FP64.Extend(fp, i); fp = elems[i].fingerPrint(fp);
                let mut fp = fp64_extend_i64(fp, FCNRCDVALUE);
                fp = fp64_extend_i32(fp, elems.len() as i32);
                for (i, elem) in elems.iter().enumerate() {
                    // Domain key: 1-indexed
                    fp = fp64_extend_i64(fp, INTVALUE);
                    fp = fp64_extend_i32(fp, (i + 1) as i32);
                    fp = elem.fingerprint_extend(fp);
                }
                fp
            }
            Value::Seq(seq) => {
                // Sequences are like tuples: functions from 1..n
                let mut fp = fp64_extend_i64(fp, FCNRCDVALUE);
                fp = fp64_extend_i32(fp, seq.len() as i32);
                for (i, elem) in seq.iter().enumerate() {
                    fp = fp64_extend_i64(fp, INTVALUE);
                    fp = fp64_extend_i32(fp, (i + 1) as i32);
                    fp = elem.fingerprint_extend(fp);
                }
                fp
            }
            Value::Record(rec) => {
                // TLC RecordValue: fingerprinted as FCNRCDVALUE
                // Keys are strings, sorted
                let mut fp = fp64_extend_i64(fp, FCNRCDVALUE);
                fp = fp64_extend_i32(fp, rec.len() as i32);
                for (key, val) in rec.iter() {
                    // String key
                    let fp_tmp = fp64_extend_i64(fp, STRINGVALUE);
                    let fp_tmp = fp64_extend_i32(fp_tmp, key.len() as i32);
                    let fp_tmp = fp64_extend_str(fp_tmp, key);
                    fp = val.fingerprint_extend(fp_tmp);
                }
                fp
            }
            Value::ModelValue(name) => {
                // TLC ModelValue: MODELVALUE tag + name string
                let fp = fp64_extend_i64(fp, MODELVALUE);
                fp64_extend_str(fp, name)
            }
            // Lazy set types - convert to concrete representation for fingerprinting
            Value::Subset(subset) => {
                if let Some(set) = subset.to_sorted_set() {
                    Value::Set(set).fingerprint_extend(fp)
                } else {
                    // If we can't enumerate, use a fallback
                    let fp = fp64_extend_i64(fp, SUBSETVALUE);
                    subset.base.fingerprint_extend(fp)
                }
            }
            Value::FuncSet(funcset) => {
                let fp = fp64_extend_i64(fp, SETOFFCNSVALUE);
                let fp = funcset.domain.fingerprint_extend(fp);
                funcset.codomain.fingerprint_extend(fp)
            }
            Value::RecordSet(recset) => {
                let mut fp = fp64_extend_i64(fp, SETOFRCDSVALUE);
                fp = fp64_extend_i32(fp, recset.fields.len() as i32);
                for (name, vals) in &recset.fields {
                    let fp_tmp = fp64_extend_str(fp, name);
                    fp = vals.fingerprint_extend(fp_tmp);
                }
                fp
            }
            Value::TupleSet(tupset) => {
                let mut fp = fp64_extend_i64(fp, SETOFTUPLESVALUE);
                fp = fp64_extend_i32(fp, tupset.components.len() as i32);
                for comp in &tupset.components {
                    fp = comp.fingerprint_extend(fp);
                }
                fp
            }
            Value::SetCup(cup) => {
                // Union of two sets - try to enumerate, otherwise use structural fp
                if let Some(set) = cup.to_sorted_set() {
                    Value::Set(set).fingerprint_extend(fp)
                } else {
                    let fp = fp64_extend_i64(fp, SETCUPVALUE);
                    let fp = cup.set1.fingerprint_extend(fp);
                    cup.set2.fingerprint_extend(fp)
                }
            }
            Value::SetCap(cap) => {
                if let Some(set) = cap.to_sorted_set() {
                    Value::Set(set).fingerprint_extend(fp)
                } else {
                    let fp = fp64_extend_i64(fp, SETCAPVALUE);
                    let fp = cap.set1.fingerprint_extend(fp);
                    cap.set2.fingerprint_extend(fp)
                }
            }
            Value::SetDiff(diff) => {
                if let Some(set) = diff.to_sorted_set() {
                    Value::Set(set).fingerprint_extend(fp)
                } else {
                    let fp = fp64_extend_i64(fp, SETDIFFVALUE);
                    let fp = diff.set1.fingerprint_extend(fp);
                    diff.set2.fingerprint_extend(fp)
                }
            }
            Value::SetPred(pred) => {
                // SetPred can't be enumerated without context, use structural fp
                let fp = fp64_extend_i64(fp, SETPREDVALUE);
                pred.source.fingerprint_extend(fp)
            }
            Value::KSubset(ksubset) => {
                if let Some(set) = ksubset.to_sorted_set() {
                    Value::Set(set).fingerprint_extend(fp)
                } else {
                    // Fallback: structural fingerprint
                    let fp = fp64_extend_i64(fp, SUBSETVALUE);
                    let fp = ksubset.base.fingerprint_extend(fp);
                    fp64_extend_i32(fp, ksubset.k as i32)
                }
            }
            Value::BigUnion(union) => {
                if let Some(set) = union.to_sorted_set() {
                    Value::Set(set).fingerprint_extend(fp)
                } else {
                    let fp = fp64_extend_i64(fp, UNIONVALUE);
                    union.set.fingerprint_extend(fp)
                }
            }
            Value::StringSet => {
                // Infinite set - use a unique tag
                fp64_extend_i64(fp, STRINGVALUE + 100) // Arbitrary unique value
            }
            Value::AnySet => {
                // Infinite set
                fp64_extend_i64(fp, USERVALUE + 100)
            }
            Value::SeqSet(seqset) => {
                // Seq(S) - fingerprint based on base set
                let fp = fp64_extend_i64(fp, SETENUMVALUE + 100);
                seqset.base.fingerprint_extend(fp)
            }
            Value::Closure(_) => {
                // Closures shouldn't typically be fingerprinted for state storage
                // Use a fallback
                fp64_extend_i64(fp, OPLAMBDAVALUE)
            }
        }
    }
}

// === Ordering for OrdSet/OrdMap ===
//
// TLA+ values need total ordering for deterministic set iteration.
// We use a type-based ordering first, then value-based within types.

impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        // Helper to compare Tuple/Seq with FuncValue without allocation.
        // In TLA+, tuples/sequences are functions with domain 1..n.
        // Returns None if FuncValue domain doesn't match tuple domain structure.
        fn cmp_tuple_with_func(tuple: &[Value], func: &FuncValue) -> Ordering {
            let n = tuple.len();
            let entries = func.entries();

            // Check lengths match
            if n != entries.len() {
                return n.cmp(&entries.len());
            }

            // Compare element by element
            // Tuple domain is {1, 2, ..., n}, FuncValue entries are sorted by key
            for (i, elem) in tuple.iter().enumerate() {
                let expected_key = Value::SmallInt((i + 1) as i64);
                let (key, val) = &entries[i];

                // Compare domain element
                match expected_key.cmp(key) {
                    Ordering::Equal => {}
                    ord => return ord,
                }

                // Compare value
                match elem.cmp(val) {
                    Ordering::Equal => {}
                    ord => return ord,
                }
            }
            Ordering::Equal
        }

        // Helper to compare Record with FuncValue without allocation.
        // In TLA+, records are functions with string domains.
        fn cmp_record_with_func(record: &RecordValue, func: &FuncValue) -> Ordering {
            let entries = func.entries();

            // Check lengths match
            if record.len() != entries.len() {
                return record.len().cmp(&entries.len());
            }

            // Compare element by element
            // Record keys are sorted in RecordValue, FuncValue entries are sorted by key
            let mut record_iter = record.iter();
            for (func_key, func_val) in entries.iter() {
                if let Some((rec_key, rec_val)) = record_iter.next() {
                    // Compare domain element (convert record key to Value::String)
                    let rec_key_val = Value::String(Arc::clone(rec_key));
                    match rec_key_val.cmp(func_key) {
                        Ordering::Equal => {}
                        ord => return ord,
                    }

                    // Compare value
                    match rec_val.cmp(func_val) {
                        Ordering::Equal => {}
                        ord => return ord,
                    }
                } else {
                    // Record exhausted before func - shouldn't happen if lengths match
                    return Ordering::Less;
                }
            }
            Ordering::Equal
        }

        // Helper to compare FuncValue with IntIntervalFunc without allocation.
        fn cmp_func_with_intfunc(func: &FuncValue, intfunc: &IntIntervalFunc) -> Ordering {
            let func_entries = func.entries();
            let intfunc_len = intfunc.values.len();

            // Check lengths match
            if func_entries.len() != intfunc_len {
                return func_entries.len().cmp(&intfunc_len);
            }

            // Compare element by element
            for (i, (func_key, func_val)) in func_entries.iter().enumerate() {
                let int_key = Value::SmallInt(intfunc.min + i as i64);

                // Compare domain element
                match func_key.cmp(&int_key) {
                    Ordering::Equal => {}
                    ord => return ord,
                }

                // Compare value
                match func_val.cmp(&intfunc.values[i]) {
                    Ordering::Equal => {}
                    ord => return ord,
                }
            }
            Ordering::Equal
        }

        // Special case: Tuple/Seq vs Func comparison
        // In TLA+, <<a,b,c>> = [i \in 1..3 |-> ...] if mappings are equal
        match (self, other) {
            (Value::Tuple(a), Value::Func(b)) => {
                return cmp_tuple_with_func(a.as_ref(), b);
            }
            (Value::Seq(a), Value::Func(b)) => {
                return cmp_tuple_with_func(&a.to_vec(), b);
            }
            (Value::Func(a), Value::Tuple(b)) => {
                return cmp_tuple_with_func(b.as_ref(), a).reverse();
            }
            (Value::Func(a), Value::Seq(b)) => {
                return cmp_tuple_with_func(&b.to_vec(), a).reverse();
            }
            // Tuple/Seq vs IntFunc: IntFunc with domain 1..n should equal the tuple
            (Value::Tuple(a), Value::IntFunc(b)) => {
                // Compare lengths first
                if a.len() != b.values.len() {
                    return a.len().cmp(&b.values.len());
                }
                // Domain must be 1..n for equality
                let n = a.len() as i64;
                if b.min != 1 || b.max != n {
                    // Domain mismatch - compare first domain element
                    // Tuple domain starts at 1, IntFunc starts at b.min
                    return 1i64.cmp(&b.min);
                }
                // Same domain - compare values directly
                return a.iter().cmp(b.values.iter());
            }
            (Value::Seq(a), Value::IntFunc(b)) => {
                // Compare lengths first
                if a.len() != b.values.len() {
                    return a.len().cmp(&b.values.len());
                }
                // Domain must be 1..n for equality
                let n = a.len() as i64;
                if b.min != 1 || b.max != n {
                    // Domain mismatch - compare first domain element
                    // Tuple domain starts at 1, IntFunc starts at b.min
                    return 1i64.cmp(&b.min);
                }
                // Same domain - compare values directly
                return a.iter().cmp(b.values.iter());
            }
            (Value::IntFunc(a), Value::Tuple(b)) => {
                // Compare lengths first
                if a.values.len() != b.len() {
                    return a.values.len().cmp(&b.len());
                }
                let n = b.len() as i64;
                if a.min != 1 || a.max != n {
                    // Domain mismatch
                    return a.min.cmp(&1i64);
                }
                return a.values.iter().cmp(b.iter());
            }
            (Value::IntFunc(a), Value::Seq(b)) => {
                // Compare lengths first
                if a.values.len() != b.len() {
                    return a.values.len().cmp(&b.len());
                }
                let n = b.len() as i64;
                if a.min != 1 || a.max != n {
                    // Domain mismatch
                    return a.min.cmp(&1i64);
                }
                return a.values.iter().cmp(b.iter());
            }
            // Also handle Tuple vs Seq comparison (they're equivalent)
            (Value::Tuple(a), Value::Seq(b)) => {
                // Compare element by element
                for (av, bv) in a.iter().zip(b.iter()) {
                    match av.cmp(bv) {
                        Ordering::Equal => continue,
                        ord => return ord,
                    }
                }
                return a.len().cmp(&b.len());
            }
            (Value::Seq(a), Value::Tuple(b)) => {
                // Compare element by element
                for (av, bv) in a.iter().zip(b.iter()) {
                    match av.cmp(bv) {
                        Ordering::Equal => continue,
                        ord => return ord,
                    }
                }
                return a.len().cmp(&b.len());
            }
            // Record vs Func comparison: records are functions with string domains
            (Value::Record(r), Value::Func(f)) => {
                return cmp_record_with_func(r, f);
            }
            (Value::Func(f), Value::Record(r)) => {
                return cmp_record_with_func(r, f).reverse();
            }
            // Func vs IntFunc: compare directly without allocation
            (Value::Func(f), Value::IntFunc(i)) => {
                return cmp_func_with_intfunc(f, i);
            }
            (Value::IntFunc(i), Value::Func(f)) => {
                return cmp_func_with_intfunc(f, i).reverse();
            }
            _ => {}
        }

        fn type_order(v: &Value) -> u8 {
            match v {
                Value::Bool(_) => 0,
                Value::SmallInt(_) | Value::Int(_) => 1,
                Value::String(_) => 2,
                Value::ModelValue(_) => 3,
                Value::Tuple(_) => 4,
                Value::Seq(_) => 5,
                Value::Record(_) => 6,
                Value::Set(_)
                | Value::Interval(_)
                | Value::Subset(_)
                | Value::FuncSet(_)
                | Value::RecordSet(_)
                | Value::TupleSet(_)
                | Value::SetCup(_)
                | Value::SetCap(_)
                | Value::SetDiff(_)
                | Value::SetPred(_)
                | Value::KSubset(_)
                | Value::BigUnion(_)
                | Value::StringSet
                | Value::AnySet
                | Value::SeqSet(_) => 7,
                Value::Func(_) | Value::IntFunc(_) => 8,
                Value::LazyFunc(_) => 9,
                Value::Closure(_) => 10,
            }
        }

        let ord = type_order(self).cmp(&type_order(other));
        if ord != Ordering::Equal {
            return ord;
        }

        match (self, other) {
            (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
            // Integer comparisons: handle all SmallInt/BigInt combinations
            (Value::SmallInt(a), Value::SmallInt(b)) => a.cmp(b),
            (Value::SmallInt(a), Value::Int(b)) => {
                // Fast path: try converting BigInt to i64 to avoid allocation
                if let Some(b_i64) = b.to_i64() {
                    a.cmp(&b_i64)
                } else {
                    // BigInt is out of i64 range, so SmallInt is always smaller in magnitude
                    // But we need proper ordering: negative BigInt < any SmallInt < positive BigInt
                    if b.sign() == num_bigint::Sign::Minus {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                }
            }
            (Value::Int(a), Value::SmallInt(b)) => {
                // Fast path: try converting BigInt to i64 to avoid allocation
                if let Some(a_i64) = a.to_i64() {
                    a_i64.cmp(b)
                } else {
                    // BigInt is out of i64 range
                    if a.sign() == num_bigint::Sign::Minus {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                }
            }
            (Value::Int(a), Value::Int(b)) => a.cmp(b),
            (Value::String(a), Value::String(b)) => {
                // Fast path: pointer equality for Arc<str>
                if Arc::ptr_eq(a, b) {
                    Ordering::Equal
                } else {
                    a.cmp(b)
                }
            }
            (Value::ModelValue(a), Value::ModelValue(b)) => {
                // Fast path: pointer equality for Arc<str>
                if Arc::ptr_eq(a, b) {
                    Ordering::Equal
                } else {
                    a.cmp(b)
                }
            }
            (Value::Tuple(a), Value::Tuple(b)) => {
                // Fast path: pointer equality for Arc<[Value]>
                if Arc::ptr_eq(a, b) {
                    Ordering::Equal
                } else {
                    a.cmp(b)
                }
            }
            (Value::Seq(a), Value::Seq(b)) => {
                // Fast path: pointer equality for SeqValue
                if a.ptr_eq(b) {
                    Ordering::Equal
                } else {
                    a.cmp(b)
                }
            }
            (Value::Record(a), Value::Record(b)) => {
                // Fast path: pointer equality for OrdMap
                if a.ptr_eq(b) {
                    Ordering::Equal
                } else {
                    a.cmp(b)
                }
            }
            // Non-enumerable set singletons must still have total ordering
            (Value::StringSet, Value::StringSet) => Ordering::Equal,
            (Value::AnySet, Value::AnySet) => Ordering::Equal,
            (Value::StringSet, Value::AnySet) => Ordering::Less,
            (Value::AnySet, Value::StringSet) => Ordering::Greater,
            (Value::StringSet, _) => Ordering::Greater,
            (_, Value::StringSet) => Ordering::Less,
            (Value::AnySet, _) => Ordering::Greater,
            (_, Value::AnySet) => Ordering::Less,
            // Two intervals: compare by low, then high
            (Value::Interval(a), Value::Interval(b)) => a.cmp(b),
            // Interval vs Set: compare extensionally (convert interval to set)
            (Value::Interval(iv), Value::Set(s)) => {
                let mut iv_iter = iv.iter_values();
                let mut s_iter = s.iter();
                loop {
                    match (iv_iter.next(), s_iter.next()) {
                        (Some(a), Some(b)) => {
                            let cmp = a.cmp(b);
                            if cmp != Ordering::Equal {
                                return cmp;
                            }
                        }
                        (Some(_), None) => return Ordering::Greater,
                        (None, Some(_)) => return Ordering::Less,
                        (None, None) => return Ordering::Equal,
                    }
                }
            }
            (Value::Set(s), Value::Interval(iv)) => {
                // Reverse comparison
                match Value::Interval(iv.clone()).cmp(&Value::Set(s.clone())) {
                    Ordering::Less => Ordering::Greater,
                    Ordering::Greater => Ordering::Less,
                    Ordering::Equal => Ordering::Equal,
                }
            }
            (Value::Set(a), Value::Set(b)) => {
                // Fast path: pointer equality for OrdSet (structural sharing)
                if a.ptr_eq(b) {
                    return Ordering::Equal;
                }
                // Compare sets by iterating elements
                let mut ai = a.iter();
                let mut bi = b.iter();
                loop {
                    match (ai.next(), bi.next()) {
                        (Some(av), Some(bv)) => {
                            let cmp = av.cmp(bv);
                            if cmp != Ordering::Equal {
                                return cmp;
                            }
                        }
                        (Some(_), None) => return Ordering::Greater,
                        (None, Some(_)) => return Ordering::Less,
                        (None, None) => return Ordering::Equal,
                    }
                }
            }
            // Two Subsets: compare by base set
            (Value::Subset(a), Value::Subset(b)) => a.base.cmp(&b.base),
            // Subset vs Set/Interval: compare extensionally (eagerly)
            (Value::Subset(sv), Value::Set(_)) | (Value::Subset(sv), Value::Interval(_)) => {
                let sv_set = sv.to_sorted_set().unwrap_or_else(SortedSet::new);
                let other_set = match other {
                    Value::Set(s) => s.clone(),
                    Value::Interval(iv) => iv.to_sorted_set(),
                    _ => unreachable!(),
                };
                let mut sv_iter = sv_set.iter();
                let mut other_iter = other_set.iter();
                loop {
                    match (sv_iter.next(), other_iter.next()) {
                        (Some(av), Some(bv)) => {
                            let cmp = av.cmp(bv);
                            if cmp != Ordering::Equal {
                                return cmp;
                            }
                        }
                        (Some(_), None) => return Ordering::Greater,
                        (None, Some(_)) => return Ordering::Less,
                        (None, None) => return Ordering::Equal,
                    }
                }
            }
            (Value::Set(_), Value::Subset(sv)) | (Value::Interval(_), Value::Subset(sv)) => {
                // Reverse comparison
                match Value::Subset(sv.clone()).cmp(self) {
                    Ordering::Less => Ordering::Greater,
                    Ordering::Greater => Ordering::Less,
                    Ordering::Equal => Ordering::Equal,
                }
            }
            // Two FuncSets: compare by domain, then codomain
            (Value::FuncSet(a), Value::FuncSet(b)) => a.cmp(b),
            // FuncSet vs other set types: compare extensionally (eagerly)
            (Value::FuncSet(fsv), _) | (_, Value::FuncSet(fsv)) => {
                let fsv_set = fsv.to_ord_set().unwrap_or_default();
                let other_set = match (self, other) {
                    (Value::FuncSet(_), o) => o.to_ord_set().unwrap_or_default(),
                    (s, Value::FuncSet(_)) => s.to_ord_set().unwrap_or_default(),
                    _ => unreachable!(),
                };
                let is_self_funcset = matches!(self, Value::FuncSet(_));
                let (self_set, other_set) = if is_self_funcset {
                    (fsv_set, other_set)
                } else {
                    (other_set, fsv_set)
                };
                let mut self_iter = self_set.iter();
                let mut other_iter = other_set.iter();
                loop {
                    match (self_iter.next(), other_iter.next()) {
                        (Some(av), Some(bv)) => {
                            let cmp = av.cmp(bv);
                            if cmp != Ordering::Equal {
                                return cmp;
                            }
                        }
                        (Some(_), None) => return Ordering::Greater,
                        (None, Some(_)) => return Ordering::Less,
                        (None, None) => return Ordering::Equal,
                    }
                }
            }
            (Value::RecordSet(a), Value::RecordSet(b)) => a.cmp(b),
            // RecordSet vs other set types: compare extensionally (eagerly)
            (Value::RecordSet(rsv), _) | (_, Value::RecordSet(rsv)) => {
                let rsv_set = rsv.to_ord_set().unwrap_or_default();
                let other_set = match (self, other) {
                    (Value::RecordSet(_), o) => o.to_ord_set().unwrap_or_default(),
                    (s, Value::RecordSet(_)) => s.to_ord_set().unwrap_or_default(),
                    _ => unreachable!(),
                };
                let is_self_recordset = matches!(self, Value::RecordSet(_));
                let (self_set, other_set) = if is_self_recordset {
                    (rsv_set, other_set)
                } else {
                    (other_set, rsv_set)
                };
                let mut self_iter = self_set.iter();
                let mut other_iter = other_set.iter();
                loop {
                    match (self_iter.next(), other_iter.next()) {
                        (Some(av), Some(bv)) => {
                            let cmp = av.cmp(bv);
                            if cmp != Ordering::Equal {
                                return cmp;
                            }
                        }
                        (Some(_), None) => return Ordering::Greater,
                        (None, Some(_)) => return Ordering::Less,
                        (None, None) => return Ordering::Equal,
                    }
                }
            }
            // Two TupleSets: compare by components
            (Value::TupleSet(a), Value::TupleSet(b)) => a.cmp(b),
            // TupleSet vs other set types: compare extensionally (eagerly)
            (Value::TupleSet(tsv), _) | (_, Value::TupleSet(tsv)) => {
                let tsv_set = tsv.to_ord_set().unwrap_or_default();
                let other_set = match (self, other) {
                    (Value::TupleSet(_), o) => o.to_ord_set().unwrap_or_default(),
                    (s, Value::TupleSet(_)) => s.to_ord_set().unwrap_or_default(),
                    _ => unreachable!(),
                };
                let is_self_tupleset = matches!(self, Value::TupleSet(_));
                let (self_set, other_set) = if is_self_tupleset {
                    (tsv_set, other_set)
                } else {
                    (other_set, tsv_set)
                };
                let mut self_iter = self_set.iter();
                let mut other_iter = other_set.iter();
                loop {
                    match (self_iter.next(), other_iter.next()) {
                        (Some(av), Some(bv)) => {
                            let cmp = av.cmp(bv);
                            if cmp != Ordering::Equal {
                                return cmp;
                            }
                        }
                        (Some(_), None) => return Ordering::Greater,
                        (None, Some(_)) => return Ordering::Less,
                        (None, None) => return Ordering::Equal,
                    }
                }
            }
            // Two SetCups: compare by operands
            (Value::SetCup(a), Value::SetCup(b)) => a.cmp(b),
            // Two SetCaps: compare by operands
            (Value::SetCap(a), Value::SetCap(b)) => a.cmp(b),
            // Two SetDiffs: compare by operands
            (Value::SetDiff(a), Value::SetDiff(b)) => a.cmp(b),
            // Lazy set ops vs other set types: compare extensionally if possible
            (Value::SetCup(scv), _) | (_, Value::SetCup(scv)) => {
                let scv_set = scv.to_ord_set().unwrap_or_default();
                let other_set = match (self, other) {
                    (Value::SetCup(_), o) => o.to_ord_set().unwrap_or_default(),
                    (s, Value::SetCup(_)) => s.to_ord_set().unwrap_or_default(),
                    _ => unreachable!(),
                };
                let is_self_setcup = matches!(self, Value::SetCup(_));
                let (self_set, other_set) = if is_self_setcup {
                    (scv_set, other_set)
                } else {
                    (other_set, scv_set)
                };
                let mut self_iter = self_set.iter();
                let mut other_iter = other_set.iter();
                loop {
                    match (self_iter.next(), other_iter.next()) {
                        (Some(av), Some(bv)) => {
                            let cmp = av.cmp(bv);
                            if cmp != Ordering::Equal {
                                return cmp;
                            }
                        }
                        (Some(_), None) => return Ordering::Greater,
                        (None, Some(_)) => return Ordering::Less,
                        (None, None) => return Ordering::Equal,
                    }
                }
            }
            (Value::SetCap(scv), _) | (_, Value::SetCap(scv)) => {
                let scv_set = scv.to_ord_set().unwrap_or_default();
                let other_set = match (self, other) {
                    (Value::SetCap(_), o) => o.to_ord_set().unwrap_or_default(),
                    (s, Value::SetCap(_)) => s.to_ord_set().unwrap_or_default(),
                    _ => unreachable!(),
                };
                let is_self_setcap = matches!(self, Value::SetCap(_));
                let (self_set, other_set) = if is_self_setcap {
                    (scv_set, other_set)
                } else {
                    (other_set, scv_set)
                };
                let mut self_iter = self_set.iter();
                let mut other_iter = other_set.iter();
                loop {
                    match (self_iter.next(), other_iter.next()) {
                        (Some(av), Some(bv)) => {
                            let cmp = av.cmp(bv);
                            if cmp != Ordering::Equal {
                                return cmp;
                            }
                        }
                        (Some(_), None) => return Ordering::Greater,
                        (None, Some(_)) => return Ordering::Less,
                        (None, None) => return Ordering::Equal,
                    }
                }
            }
            (Value::SetDiff(sdv), _) | (_, Value::SetDiff(sdv)) => {
                let sdv_set = sdv.to_ord_set().unwrap_or_default();
                let other_set = match (self, other) {
                    (Value::SetDiff(_), o) => o.to_ord_set().unwrap_or_default(),
                    (s, Value::SetDiff(_)) => s.to_ord_set().unwrap_or_default(),
                    _ => unreachable!(),
                };
                let is_self_setdiff = matches!(self, Value::SetDiff(_));
                let (self_set, other_set) = if is_self_setdiff {
                    (sdv_set, other_set)
                } else {
                    (other_set, sdv_set)
                };
                let mut self_iter = self_set.iter();
                let mut other_iter = other_set.iter();
                loop {
                    match (self_iter.next(), other_iter.next()) {
                        (Some(av), Some(bv)) => {
                            let cmp = av.cmp(bv);
                            if cmp != Ordering::Equal {
                                return cmp;
                            }
                        }
                        (Some(_), None) => return Ordering::Greater,
                        (None, Some(_)) => return Ordering::Less,
                        (None, None) => return Ordering::Equal,
                    }
                }
            }
            (Value::Func(a), Value::Func(b)) => {
                // Fast path: pointer equality for entries array
                if a.ptr_eq(b) {
                    return Ordering::Equal;
                }
                // Compare entries directly (sorted by key)
                a.cmp(b)
            }
            (Value::IntFunc(a), Value::IntFunc(b)) => {
                // Fast path: pointer equality for values array
                if Arc::ptr_eq(&a.values, &b.values) && a.min == b.min && a.max == b.max {
                    return Ordering::Equal;
                }
                // Compare by domain (min/max) first, then values
                match a.min.cmp(&b.min) {
                    Ordering::Equal => {}
                    ord => return ord,
                }
                match a.max.cmp(&b.max) {
                    Ordering::Equal => {}
                    ord => return ord,
                }
                a.values.iter().cmp(b.values.iter())
            }
            // Lazy functions are compared by their unique ID (not extensionally)
            (Value::LazyFunc(a), Value::LazyFunc(b)) => a.id.cmp(&b.id),
            // Closures are compared by their unique ID
            (Value::Closure(a), Value::Closure(b)) => a.id.cmp(&b.id),
            // SetPred values are compared by their unique ID (can't evaluate predicates without ctx)
            (Value::SetPred(a), Value::SetPred(b)) => a.id.cmp(&b.id),
            // SetPred vs other set types: compare by ID only (can't enumerate)
            (Value::SetPred(_), _) | (_, Value::SetPred(_)) => {
                // SetPred can't be extensionally compared without evaluation context
                // Use a consistent ordering based on type + id
                let is_self_setpred = matches!(self, Value::SetPred(_));
                if is_self_setpred {
                    // SetPred always greater than other set types for consistent ordering
                    Ordering::Greater
                } else {
                    // Other set types less than SetPred
                    Ordering::Less
                }
            }
            // KSubset values: compare by k first, then base
            (Value::KSubset(a), Value::KSubset(b)) => a.cmp(b),
            // KSubset vs other set types: compare extensionally (eagerly)
            (Value::KSubset(ksv), other_val) => {
                let ksv_set = ksv.to_ord_set().unwrap_or_default();
                let other_set = other_val.to_ord_set().unwrap_or_default();
                ksv_set.cmp(&other_set)
            }
            (other_val, Value::KSubset(ksv)) => {
                let other_set = other_val.to_ord_set().unwrap_or_default();
                let ksv_set = ksv.to_ord_set().unwrap_or_default();
                other_set.cmp(&ksv_set)
            }
            // BigUnion values: compare by underlying set
            (Value::BigUnion(a), Value::BigUnion(b)) => a.cmp(b),
            // BigUnion vs other set types: compare extensionally (eagerly)
            (Value::BigUnion(uv), other_val) => {
                let uv_set = uv.to_ord_set().unwrap_or_default();
                let other_set = other_val.to_ord_set().unwrap_or_default();
                uv_set.cmp(&other_set)
            }
            (other_val, Value::BigUnion(uv)) => {
                let other_set = other_val.to_ord_set().unwrap_or_default();
                let uv_set = uv.to_ord_set().unwrap_or_default();
                other_set.cmp(&uv_set)
            }
            _ => unreachable!("type_order ensures same types"),
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Value {}

/// Hash an i64 value in a way that produces the same result as BigInt::from(n).to_bytes_le()
/// This avoids allocating a BigInt for small integers (the common case).
#[inline]
fn hash_i64_as_bigint<H: Hasher>(n: i64, state: &mut H) {
    use num_bigint::Sign;

    if n == 0 {
        Sign::NoSign.hash(state);
        // Empty slice for zero - same as BigInt::from(0).to_bytes_le().1
        let empty: [u8; 0] = [];
        empty.hash(state);
    } else if n > 0 {
        Sign::Plus.hash(state);
        // Compute minimal little-endian bytes (same as BigInt)
        let unsigned = n as u64;
        let byte_len = (64 - unsigned.leading_zeros()).div_ceil(8) as usize;
        unsigned.to_le_bytes()[..byte_len].hash(state);
    } else {
        Sign::Minus.hash(state);
        // Handle negative: compute absolute value
        // Special case: i64::MIN cannot be negated directly
        let abs_val = if n == i64::MIN {
            (i64::MAX as u64) + 1
        } else {
            (-n) as u64
        };
        let byte_len = (64 - abs_val.leading_zeros()).div_ceil(8) as usize;
        abs_val.to_le_bytes()[..byte_len].hash(state);
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Use a common discriminant for all set types so they hash the same
        // when they represent the same set extensionally.
        // Similarly, Tuple/Seq/Record/Func all use the same discriminant (6)
        // since they are all functions in TLA+ (tuples/seqs have domain 1..n,
        // records have string domain).
        match self {
            Value::Bool(_) => 0u8.hash(state),
            Value::SmallInt(_) | Value::Int(_) => 1u8.hash(state),
            Value::String(_) => 2u8.hash(state),
            Value::ModelValue(_) => 3u8.hash(state),
            // All function-like types (Tuple, Seq, Record, Func, IntFunc) use same discriminant
            // since they are semantically functions and should hash equally when equal
            Value::Tuple(_)
            | Value::Seq(_)
            | Value::Record(_)
            | Value::Func(_)
            | Value::IntFunc(_) => 6u8.hash(state),
            Value::Set(_)
            | Value::Interval(_)
            | Value::Subset(_)
            | Value::FuncSet(_)
            | Value::RecordSet(_)
            | Value::TupleSet(_)
            | Value::SetCup(_)
            | Value::SetCap(_)
            | Value::SetDiff(_)
            | Value::SetPred(_)
            | Value::KSubset(_)
            | Value::BigUnion(_)
            | Value::StringSet
            | Value::AnySet
            | Value::SeqSet(_) => 7u8.hash(state),
            Value::LazyFunc(_) => 9u8.hash(state),
            Value::Closure(_) => 10u8.hash(state),
        }
        match self {
            Value::Bool(b) => b.hash(state),
            // SmallInt and Int must hash the same for equal values
            Value::SmallInt(n) => {
                // Hash i64 directly, producing same result as BigInt::from(n).to_bytes_le()
                hash_i64_as_bigint(*n, state);
            }
            Value::Int(n) => {
                // Fast path: if fits in i64, use direct hashing
                if let Some(small) = n.to_i64() {
                    hash_i64_as_bigint(small, state);
                } else {
                    // Slow path: large integers need byte conversion
                    let (sign, bytes) = n.to_bytes_le();
                    sign.hash(state);
                    bytes.hash(state);
                }
            }
            Value::String(s) => s.hash(state),
            Value::ModelValue(s) => s.hash(state),
            // Tuple/Seq: function with domain 1..n
            // Hash domain (integers 1..n) then mapping (index -> value)
            Value::Tuple(t) => {
                // Hash domain: 1, 2, ..., n (using direct i64 hashing to avoid BigInt allocation)
                for i in 1..=t.len() {
                    1u8.hash(state); // Discriminant for integers
                    hash_i64_as_bigint(i as i64, state);
                }
                // Hash mapping: (1, v1), (2, v2), ..., (n, vn)
                for (i, v) in t.iter().enumerate() {
                    1u8.hash(state); // Discriminant for integers
                    hash_i64_as_bigint((i + 1) as i64, state);
                    v.hash(state);
                }
            }
            Value::Seq(s) => {
                // Hash domain: 1, 2, ..., n (using direct i64 hashing to avoid BigInt allocation)
                for i in 1..=s.len() {
                    1u8.hash(state); // Discriminant for integers
                    hash_i64_as_bigint(i as i64, state);
                }
                // Hash mapping: (1, v1), (2, v2), ..., (n, vn)
                for (i, v) in s.iter().enumerate() {
                    1u8.hash(state); // Discriminant for integers
                    hash_i64_as_bigint((i + 1) as i64, state);
                    v.hash(state);
                }
            }
            // Record: function with string domain
            // Hash domain (strings) then mapping (string -> value)
            // Optimized: avoid Vec allocation and Arc cloning by hashing directly
            Value::Record(r) => {
                // Hash domain (keys in sorted order)
                // Must match Value::String hash: discriminant 2, then string bytes
                for k in r.keys() {
                    2u8.hash(state); // String discriminant
                    k.hash(state);
                }
                // Hash mapping (key-value pairs in sorted order)
                for (k, v) in r.iter() {
                    2u8.hash(state); // String discriminant
                    k.hash(state);
                    v.hash(state);
                }
            }
            Value::Set(s) => {
                for v in s.iter() {
                    v.hash(state);
                }
            }
            // Interval is hashed by iterating elements (extensional)
            Value::Interval(iv) => {
                for v in iv.iter_values() {
                    v.hash(state);
                }
            }
            // Subset is hashed by iterating elements (extensional)
            Value::Subset(sv) => {
                if let Some(iter) = sv.iter() {
                    for v in iter {
                        v.hash(state);
                    }
                }
            }
            // FuncSet is hashed by iterating elements in sorted order (extensional)
            // Use to_ord_set() to ensure consistent ordering across different set representations
            Value::FuncSet(fsv) => {
                if let Some(set) = fsv.to_ord_set() {
                    for v in set.iter() {
                        v.hash(state);
                    }
                }
            }
            // RecordSet is hashed by iterating elements in sorted order (extensional)
            // Use to_ord_set() to ensure consistent ordering across different set representations
            Value::RecordSet(rsv) => {
                if let Some(set) = rsv.to_ord_set() {
                    for v in set.iter() {
                        v.hash(state);
                    }
                }
            }
            // TupleSet is hashed by iterating elements in sorted order (extensional)
            // Use to_ord_set() to ensure consistent ordering across different set representations
            Value::TupleSet(tsv) => {
                if let Some(set) = tsv.to_ord_set() {
                    for v in set.iter() {
                        v.hash(state);
                    }
                }
            }
            // SetCup is hashed by iterating elements (extensional) if enumerable
            Value::SetCup(scv) => {
                if let Some(set) = scv.to_ord_set() {
                    for v in set.iter() {
                        v.hash(state);
                    }
                } else {
                    // Non-enumerable - hash by structure
                    scv.hash(state);
                }
            }
            // SetCap is hashed by iterating elements (extensional) if enumerable
            Value::SetCap(scv) => {
                if let Some(set) = scv.to_ord_set() {
                    for v in set.iter() {
                        v.hash(state);
                    }
                } else {
                    // Non-enumerable - hash by structure
                    scv.hash(state);
                }
            }
            // SetDiff is hashed by iterating elements (extensional) if enumerable
            Value::SetDiff(sdv) => {
                if let Some(set) = sdv.to_ord_set() {
                    for v in set.iter() {
                        v.hash(state);
                    }
                } else {
                    // Non-enumerable - hash by structure
                    sdv.hash(state);
                }
            }
            // SetPred is hashed by its unique ID (can't evaluate predicates without ctx)
            Value::SetPred(spv) => spv.hash(state),
            // KSubset is hashed by iterating elements (extensional) if enumerable
            Value::KSubset(ksv) => {
                if let Some(iter) = ksv.iter() {
                    for v in iter {
                        v.hash(state);
                    }
                } else {
                    // Non-enumerable - hash by structure
                    ksv.hash(state);
                }
            }
            // BigUnion is hashed by iterating elements (extensional) if enumerable
            Value::BigUnion(uv) => {
                if let Some(set) = uv.to_ord_set() {
                    for v in set.iter() {
                        v.hash(state);
                    }
                } else {
                    // Non-enumerable - hash by structure
                    uv.hash(state);
                }
            }
            Value::Func(f) => {
                // Hash domain first (sorted iteration order matches Record's OrdMap keys)
                for k in f.domain_iter() {
                    k.hash(state);
                }
                // Hash all (key, value) pairs - entries are sorted so order is deterministic
                for (k, v) in f.mapping_iter() {
                    k.hash(state);
                    v.hash(state);
                }
            }
            Value::IntFunc(f) => {
                // Hash domain (min..max as integers)
                for i in f.min..=f.max {
                    1u8.hash(state); // Integer discriminant
                    hash_i64_as_bigint(i, state);
                }
                // Hash mapping (key, value pairs)
                for (i, v) in f.values.iter().enumerate() {
                    1u8.hash(state); // Integer discriminant
                    hash_i64_as_bigint(f.min + i as i64, state);
                    v.hash(state);
                }
            }
            // Lazy functions are hashed by their unique ID (not extensionally)
            Value::LazyFunc(f) => f.id.hash(state),
            // Closures are hashed by their unique ID
            Value::Closure(c) => c.id.hash(state),
            // StringSet is a singleton, hash it as a constant
            Value::StringSet => "STRING".hash(state),
            // AnySet is a singleton, hash it as a constant
            Value::AnySet => "ANY".hash(state),
            // SeqSet is hashed by its base set
            Value::SeqSet(ssv) => {
                "SEQSET".hash(state);
                ssv.base.hash(state);
            }
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Bool(b) => write!(f, "{}", if *b { "TRUE" } else { "FALSE" }),
            Value::SmallInt(n) => write!(f, "{}", n),
            Value::Int(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "{:?}", s),
            Value::ModelValue(s) => write!(f, "@{}", s),
            Value::Set(s) => {
                write!(f, "{{")?;
                for (i, v) in s.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", v)?;
                }
                write!(f, "}}")
            }
            Value::Interval(iv) => {
                // Display as interval notation for debugging
                write!(f, "{}..{}", iv.low, iv.high)
            }
            Value::Subset(sv) => {
                // Display as SUBSET base notation
                write!(f, "SUBSET {:?}", sv.base)
            }
            Value::FuncSet(fsv) => {
                // Display as function set notation
                write!(f, "[{:?} -> {:?}]", fsv.domain, fsv.codomain)
            }
            Value::RecordSet(rsv) => {
                // Display as record set notation
                write!(f, "[")?;
                for (i, (k, v)) in rsv.fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {:?}", k, v)?;
                }
                write!(f, "]")
            }
            Value::TupleSet(tsv) => {
                // Display as tuple set notation (cartesian product)
                if tsv.components.is_empty() {
                    write!(f, "{{<<>>}}")
                } else {
                    for (i, c) in tsv.components.iter().enumerate() {
                        if i > 0 {
                            write!(f, " \\X ")?;
                        }
                        write!(f, "{:?}", c)?;
                    }
                    Ok(())
                }
            }
            Value::SetCup(scv) => {
                // Display as union notation
                write!(f, "({:?} \\cup {:?})", scv.set1, scv.set2)
            }
            Value::SetCap(scv) => {
                // Display as intersection notation
                write!(f, "({:?} \\cap {:?})", scv.set1, scv.set2)
            }
            Value::SetDiff(sdv) => {
                // Display as set difference notation
                write!(f, "({:?} \\ {:?})", sdv.set1, sdv.set2)
            }
            Value::SetPred(spv) => {
                // Display as set filter notation
                write!(f, "{{x \\in {:?} : <pred#{}>}}", spv.source, spv.id)
            }
            Value::KSubset(ksv) => {
                // Display as Ksubsets notation
                write!(f, "Ksubsets({:?}, {})", ksv.base, ksv.k)
            }
            Value::BigUnion(uv) => {
                // Display as UNION notation
                write!(f, "UNION {:?}", uv.set)
            }
            Value::Seq(s) => {
                write!(f, "<<")?;
                for (i, v) in s.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", v)?;
                }
                write!(f, ">>")
            }
            Value::Tuple(t) => {
                write!(f, "<<")?;
                for (i, v) in t.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", v)?;
                }
                write!(f, ">>")
            }
            Value::Record(r) => {
                write!(f, "[")?;
                for (i, (k, v)) in r.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{} |-> {:?}", k, v)?;
                }
                write!(f, "]")
            }
            Value::Func(func) => {
                write!(f, "[")?;
                for (i, (k, v)) in func.mapping_iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?} |-> {:?}", k, v)?;
                }
                write!(f, "]")
            }
            Value::IntFunc(func) => {
                write!(f, "[")?;
                for (i, v) in func.values.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{} |-> {:?}", func.min + i as i64, v)?;
                }
                write!(f, "]")
            }
            Value::LazyFunc(func) => match &func.name {
                Some(name) => write!(f, "<lazy-func:{}#{}>", name, func.id),
                None => write!(f, "<lazy-func#{}>", func.id),
            },
            Value::Closure(c) => {
                write!(f, "LAMBDA ")?;
                for (i, p) in c.params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, " : <body#{}>", c.id)
            }
            Value::StringSet => write!(f, "STRING"),
            Value::AnySet => write!(f, "ANY"),
            Value::SeqSet(ssv) => write!(f, "Seq({:?})", ssv.base),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// === FuncBuilder implementations ===

impl FuncBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        FuncBuilder {
            entries: Vec::new(),
        }
    }

    /// Create a builder with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        FuncBuilder {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Insert a key-value pair. Duplicate keys will be deduplicated during build.
    #[inline]
    pub fn insert(&mut self, key: Value, value: Value) {
        self.entries.push((key, value));
    }

    /// Build the FuncValue, sorting and deduplicating entries.
    pub fn build(mut self) -> FuncValue {
        self.entries.sort_by(|a, b| a.0.cmp(&b.0));
        self.entries.dedup_by(|a, b| a.0 == b.0);
        #[cfg(feature = "memory-stats")]
        memory_stats::inc_func_entries(self.entries.len());

        FuncValue {
            entries: self.entries.into(),
            cached_fp: std::sync::OnceLock::new(),
        }
    }

    /// Build the FuncValue assuming entries are already sorted and unique.
    /// This is O(n) instead of O(n log n).
    pub fn build_presorted(self) -> FuncValue {
        #[cfg(feature = "memory-stats")]
        memory_stats::inc_func_entries(self.entries.len());

        FuncValue {
            entries: self.entries.into(),
            cached_fp: std::sync::OnceLock::new(),
        }
    }
}

impl Default for FuncBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// === FuncValue implementations ===

impl Clone for FuncValue {
    fn clone(&self) -> Self {
        // Clone the cached fingerprint - since FuncValue is immutable,
        // a clone has the same fingerprint as the original.
        // OnceLock doesn't impl Clone, so we manually copy if set.
        let cached = self.cached_fp.get().copied();
        let new_cache = std::sync::OnceLock::new();
        if let Some(fp) = cached {
            let _ = new_cache.set(fp);
        }
        FuncValue {
            entries: self.entries.clone(),
            cached_fp: new_cache,
        }
    }
}

impl FuncValue {
    /// Singleton empty function.
    fn empty() -> &'static FuncValue {
        static EMPTY: std::sync::OnceLock<FuncValue> = std::sync::OnceLock::new();
        EMPTY.get_or_init(|| FuncValue {
            entries: Arc::new([]),
            cached_fp: std::sync::OnceLock::new(),
        })
    }

    /// Create a new function from OrdSet domain and OrdMap mapping.
    /// This is the migration path from the old API.
    pub fn new(domain: OrdSet<Value>, mapping: OrdMap<Value, Value>) -> Self {
        if domain.is_empty() {
            return FuncValue::empty().clone();
        }
        // Build entries from mapping, sorted by key (domain iteration order)
        let mut entries: Vec<(Value, Value)> = Vec::with_capacity(domain.len());
        for key in domain.iter() {
            if let Some(value) = mapping.get(key) {
                entries.push((key.clone(), value.clone()));
            }
        }
        #[cfg(feature = "memory-stats")]
        memory_stats::inc_func_entries(entries.len());

        FuncValue {
            entries: entries.into(),
            cached_fp: std::sync::OnceLock::new(),
        }
    }

    /// Create a function from pre-sorted (key, value) pairs.
    /// Entries must be sorted by key and unique.
    pub fn from_sorted_entries(entries: Vec<(Value, Value)>) -> Self {
        if entries.is_empty() {
            return FuncValue::empty().clone();
        }
        #[cfg(feature = "memory-stats")]
        memory_stats::inc_func_entries(entries.len());

        FuncValue {
            entries: entries.into(),
            cached_fp: std::sync::OnceLock::new(),
        }
    }

    /// Apply the function to an argument (lookup by key).
    pub fn apply(&self, arg: &Value) -> Option<&Value> {
        self.entries
            .binary_search_by(|(k, _)| k.cmp(arg))
            .ok()
            .map(|idx| &self.entries[idx].1)
    }

    /// Update the function at a single point: [f EXCEPT ![x] = y]
    ///
    /// Optimization: If the new value equals the old value at that point,
    /// returns a clone of self (avoiding allocation).
    pub fn except(&self, arg: Value, value: Value) -> Self {
        match self.entries.binary_search_by(|(k, _)| k.cmp(&arg)) {
            Ok(idx) => {
                // Key exists - check if value changed
                if self.entries[idx].1 == value {
                    return self.clone();
                }
                // Clone array and update the value
                let mut new_entries: Vec<(Value, Value)> = self.entries.to_vec();
                new_entries[idx].1 = value;
                FuncValue {
                    entries: new_entries.into(),
                    cached_fp: std::sync::OnceLock::new(),
                }
            }
            Err(_) => {
                // Key not in domain - return unchanged (TLA+ function semantics)
                self.clone()
            }
        }
    }

    /// Get the cached fingerprint if already computed.
    #[inline]
    pub fn get_cached_fingerprint(&self) -> Option<u64> {
        self.cached_fp.get().copied()
    }

    /// Set the cached fingerprint (internal use by state.rs fingerprinting).
    ///
    /// Returns the fingerprint value. If already cached, returns the cached value.
    #[inline]
    pub fn cache_fingerprint(&self, fp: u64) -> u64 {
        *self.cached_fp.get_or_init(|| fp)
    }

    // === Domain accessors (migration from OrdSet<Value>) ===

    /// Get the number of elements in the domain.
    #[inline]
    pub fn domain_len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the domain is empty.
    #[inline]
    pub fn domain_is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Check if a value is in the domain.
    #[inline]
    pub fn domain_contains(&self, key: &Value) -> bool {
        self.entries.binary_search_by(|(k, _)| k.cmp(key)).is_ok()
    }

    /// Iterator over domain elements (keys).
    pub fn domain_iter(&self) -> impl Iterator<Item = &Value> + '_ {
        self.entries.iter().map(|(k, _)| k)
    }

    /// Convert domain to SortedSet.
    pub fn domain_as_sorted_set(&self) -> SortedSet {
        let keys: Vec<Value> = self.entries.iter().map(|(k, _)| k.clone()).collect();
        SortedSet::from_sorted_vec(keys)
    }

    /// Convert domain to OrdSet (for backward compatibility).
    pub fn domain_as_ord_set(&self) -> OrdSet<Value> {
        self.entries.iter().map(|(k, _)| k.clone()).collect()
    }

    // === Mapping accessors (migration from OrdMap<Value, Value>) ===

    /// Get a value by key (same as apply).
    #[inline]
    pub fn mapping_get(&self, key: &Value) -> Option<&Value> {
        self.apply(key)
    }

    /// Iterator over values.
    pub fn mapping_values(&self) -> impl Iterator<Item = &Value> + '_ {
        self.entries.iter().map(|(_, v)| v)
    }

    /// Iterator over (key, value) pairs.
    pub fn mapping_iter(&self) -> impl Iterator<Item = (&Value, &Value)> + '_ {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    /// Get the raw entries slice.
    #[inline]
    pub fn entries(&self) -> &[(Value, Value)] {
        &self.entries
    }

    /// Check if two FuncValues share the same entries Arc (pointer equality).
    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.entries, &other.entries)
    }

    /// Compose two permutations: (self ∘ other)(x) = self(other(x))
    ///
    /// For symmetry reduction, we need to compose permutations to compute
    /// the group closure. If permutations operate on disjoint domains
    /// (e.g., Permutations(A) ∪ Permutations(B)), the composition combines
    /// both mappings.
    ///
    /// Returns a new permutation that is the composition of self and other.
    pub fn compose_perm(&self, other: &FuncValue) -> FuncValue {
        use std::collections::BTreeMap;

        // Collect all keys from both domains
        // Allow: Value contains OnceLock for fingerprint caching, but Ord is stable
        #[allow(clippy::mutable_key_type)]
        let mut combined: BTreeMap<Value, Value> = BTreeMap::new();

        // For each key in other's domain: compose mappings
        for (key, val) in other.entries.iter() {
            // Apply self to the result of other
            let final_val = self.apply(val).cloned().unwrap_or_else(|| val.clone());
            combined.insert(key.clone(), final_val);
        }

        // For keys in self's domain but not in other's: use self's mapping directly
        // (This handles disjoint domain composition)
        for (key, val) in self.entries.iter() {
            combined.entry(key.clone()).or_insert_with(|| val.clone());
        }

        // Convert to sorted entries
        let entries: Vec<(Value, Value)> = combined.into_iter().collect();
        FuncValue::from_sorted_entries(entries)
    }
}

impl Ord for FuncValue {
    fn cmp(&self, other: &Self) -> Ordering {
        self.entries.cmp(&other.entries)
    }
}

impl PartialOrd for FuncValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for FuncValue {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries
    }
}

impl Eq for FuncValue {}

impl IntIntervalFunc {
    /// Create a new int-interval function with given bounds and values.
    ///
    /// # Arguments
    /// * `min` - Minimum domain element (inclusive)
    /// * `max` - Maximum domain element (inclusive)
    /// * `values` - Values array where `values[i] = f[min + i]`
    pub fn new(min: i64, max: i64, values: Vec<Value>) -> Self {
        debug_assert_eq!(values.len(), (max - min + 1) as usize);
        #[cfg(feature = "memory-stats")]
        memory_stats::inc_int_func(values.len());

        IntIntervalFunc {
            min,
            max,
            values: Arc::new(values),
            cached_fp: std::sync::OnceLock::new(),
        }
    }

    /// Create from an iterator over domain-value pairs.
    /// Domain must be a contiguous integer interval starting at `min`.
    pub fn from_iter(min: i64, iter: impl IntoIterator<Item = Value>) -> Self {
        let values: Vec<Value> = iter.into_iter().collect();
        let max = min + values.len() as i64 - 1;
        #[cfg(feature = "memory-stats")]
        memory_stats::inc_int_func(values.len());

        IntIntervalFunc {
            min,
            max,
            values: Arc::new(values),
            cached_fp: std::sync::OnceLock::new(),
        }
    }

    /// Apply the function to an argument. Returns None if out of bounds.
    #[inline]
    pub fn apply(&self, arg: &Value) -> Option<&Value> {
        let idx = match arg {
            Value::SmallInt(n) => *n,
            Value::Int(n) => n.to_i64()?,
            _ => return None,
        };
        if idx >= self.min && idx <= self.max {
            Some(&self.values[(idx - self.min) as usize])
        } else {
            None
        }
    }

    /// Update the function at a single point: [f EXCEPT ![x] = y]
    ///
    /// Returns an IntIntervalFunc with the updated value. Uses interning for small
    /// functions to deduplicate memory when shared.
    #[inline]
    pub fn except(self, arg: &Value, value: Value) -> Self {
        #[cfg(feature = "memory-stats")]
        memory_stats::inc_int_func_except();

        let idx = match arg {
            Value::SmallInt(n) => *n,
            Value::Int(n) => match n.to_i64() {
                Some(i) => i,
                None => return self,
            },
            _ => return self,
        };
        if idx < self.min || idx > self.max {
            return self;
        }
        let arr_idx = (idx - self.min) as usize;
        // Short-circuit: if value unchanged, return self (preserves cached fingerprint)
        if self.values[arr_idx] == value {
            return self;
        }

        // For small functions, try to find an already-interned version first
        // This avoids allocating a new Vec when we would just return an existing one
        if self.values.len() <= MAX_INTERN_INT_FUNC_SIZE {
            // Fast path: check if the modified version is already interned
            if let Some(interned) =
                try_get_interned_modified(self.min, self.max, &self.values, arr_idx, &value)
            {
                return IntIntervalFunc {
                    min: self.min,
                    max: self.max,
                    values: interned,
                    cached_fp: std::sync::OnceLock::new(),
                };
            }

            // Medium path: if we're the sole owner, modify in place (COW)
            // If refcount == 1, the value is NOT in the intern table (which would add +1)
            // so we can safely modify in place, then intern the result
            if Arc::strong_count(&self.values) == 1 {
                let mut new_self = self;
                Arc::make_mut(&mut new_self.values)[arr_idx] = value;
                // Extract Vec and re-intern with new fingerprint
                let vec = Arc::try_unwrap(new_self.values).unwrap_or_else(|arc| (*arc).clone());
                let interned = intern_int_func_array(new_self.min, new_self.max, vec);
                return IntIntervalFunc {
                    min: new_self.min,
                    max: new_self.max,
                    values: interned,
                    cached_fp: std::sync::OnceLock::new(),
                };
            }

            // Slow path: clone the values array and intern it
            let mut new_values: Vec<Value> = self.values.to_vec();
            new_values[arr_idx] = value;
            let interned = intern_int_func_array(self.min, self.max, new_values);
            return IntIntervalFunc {
                min: self.min,
                max: self.max,
                values: interned,
                cached_fp: std::sync::OnceLock::new(),
            };
        }

        // For larger functions, use COW
        #[cfg(feature = "memory-stats")]
        if Arc::strong_count(&self.values) > 1 {
            memory_stats::inc_int_func_except_clone();
        }
        let mut new_self = self;
        Arc::make_mut(&mut new_self.values)[arr_idx] = value;
        new_self.cached_fp = std::sync::OnceLock::new();
        new_self
    }

    /// Get the domain as an OrdSet (for compatibility)
    pub fn domain_set(&self) -> OrdSet<Value> {
        (self.min..=self.max).map(Value::SmallInt).collect()
    }

    /// Get the cached fingerprint if already computed.
    #[inline]
    pub fn get_cached_fingerprint(&self) -> Option<u64> {
        self.cached_fp.get().copied()
    }

    /// Set the cached fingerprint (internal use by state.rs fingerprinting).
    #[inline]
    pub fn cache_fingerprint(&self, fp: u64) -> u64 {
        *self.cached_fp.get_or_init(|| fp)
    }

    /// Convert to FuncValue (for compatibility with code expecting FuncValue)
    pub fn to_func_value(&self) -> FuncValue {
        let domain: OrdSet<Value> = (self.min..=self.max).map(Value::SmallInt).collect();
        let mapping: OrdMap<Value, Value> = (self.min..=self.max)
            .zip(self.values.iter().cloned())
            .map(|(k, v)| (Value::SmallInt(k), v))
            .collect();
        FuncValue::new(domain, mapping)
    }

    /// Number of elements in the domain
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if domain is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl Ord for IntIntervalFunc {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare by domain bounds first
        match self.min.cmp(&other.min) {
            Ordering::Equal => {}
            ord => return ord,
        }
        match self.max.cmp(&other.max) {
            Ordering::Equal => {}
            ord => return ord,
        }
        // Then compare values element-wise
        self.values.iter().cmp(other.values.iter())
    }
}

impl PartialOrd for IntIntervalFunc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for IntIntervalFunc {
    fn eq(&self, other: &Self) -> bool {
        self.min == other.min && self.max == other.max && self.values == other.values
    }
}

impl Eq for IntIntervalFunc {}

// === Utility functions ===

/// Create a range set {a..b} (inclusive) as a lazy interval
pub fn range_set(a: &BigInt, b: &BigInt) -> Value {
    if a > b {
        return Value::empty_set();
    }
    Value::Interval(IntervalValue::new(a.clone(), b.clone()))
}

/// Create the BOOLEAN set {TRUE, FALSE}
pub fn boolean_set() -> Value {
    Value::set([Value::Bool(false), Value::Bool(true)])
}

/// Compute SUBSET S (powerset)
///
/// Returns `EvalError::SetTooLarge` if the set has more than 20 elements,
/// since 2^20 = 1M subsets would be too large to enumerate.
pub fn powerset(set: &OrdSet<Value>) -> EvalResult<Value> {
    let elements: Vec<_> = set.iter().cloned().collect();
    let n = elements.len();

    if n > 20 {
        // Safety limit - 2^20 = 1M elements
        return Err(EvalError::SetTooLarge { span: None });
    }

    let mut result = OrdSet::new();
    for mask in 0..(1u64 << n) {
        let mut subset = OrdSet::new();
        for (i, elem) in elements.iter().enumerate() {
            if mask & (1 << i) != 0 {
                subset.insert(elem.clone());
            }
        }
        result.insert(Value::Set(SortedSet::from_ord_set(&subset)));
    }
    Ok(Value::Set(SortedSet::from_ord_set(&result)))
}

/// Compute UNION S (big union - union of all sets in S)
pub fn big_union(set: &OrdSet<Value>) -> Option<Value> {
    let mut result = OrdSet::new();
    for elem in set.iter() {
        // Use to_ord_set() to handle lazy value types (Interval, FuncSet, Subset)
        let inner = elem.to_ord_set()?;
        for v in inner.iter() {
            result.insert(v.clone());
        }
    }
    Some(Value::Set(SortedSet::from_ord_set(&result)))
}

/// Compute Cartesian product S1 \X S2 \X ...
pub fn cartesian_product(sets: &[&OrdSet<Value>]) -> Value {
    if sets.is_empty() {
        return Value::set([Value::Tuple(Vec::new().into())]);
    }

    // Build up tuples incrementally
    let mut result: Vec<Vec<Value>> = vec![vec![]];

    for set in sets {
        let mut new_result = Vec::new();
        for tuple in &result {
            for elem in set.iter() {
                let mut new_tuple = tuple.clone();
                new_tuple.push(elem.clone());
                new_result.push(new_tuple);
            }
        }
        result = new_result;
    }

    Value::set(result.into_iter().map(|v| Value::Tuple(v.into())))
}

fn checked_pow_with_limit(base: usize, exp: usize, limit: usize) -> Option<usize> {
    if exp == 0 {
        return Some(1);
    }
    if base == 0 {
        return Some(0);
    }
    if base == 1 {
        return Some(1);
    }

    let mut result = 1usize;
    for _ in 0..exp {
        result = result.checked_mul(base)?;
        if result > limit {
            return None;
        }
    }
    Some(result)
}

/// Compute the function set [S -> T]
///
/// Returns `EvalError::SetTooLarge` if |T|^|S| exceeds 100,000 functions.
pub fn func_set(domain: &OrdSet<Value>, range: &OrdSet<Value>) -> EvalResult<Value> {
    const MAX_FUNC_SET_SIZE: usize = 100_000;

    let domain_vec: Vec<_> = domain.iter().cloned().collect();
    let range_vec: Vec<_> = range.iter().cloned().collect();

    if domain_vec.is_empty() {
        // Empty domain: single function with empty mapping
        return Ok(Value::set([Value::Func(FuncValue::new(
            OrdSet::new(),
            OrdMap::new(),
        ))]));
    }

    let n = domain_vec.len();
    let m = range_vec.len();

    let total = checked_pow_with_limit(m, n, MAX_FUNC_SET_SIZE)
        .ok_or(EvalError::SetTooLarge { span: None })?;

    // Generate all functions by iterating through all possible mappings
    let mut result = OrdSet::new();

    for i in 0..total {
        let mut mapping = OrdMap::new();
        let mut idx = i;
        for d in &domain_vec {
            let r_idx = idx % m;
            mapping.insert(d.clone(), range_vec[r_idx].clone());
            idx /= m;
        }
        result.insert(Value::Func(FuncValue::new(domain.clone(), mapping)));
    }

    Ok(Value::Set(SortedSet::from_ord_set(&result)))
}

/// Compute record set [field1: S1, field2: S2, ...]
pub fn record_set(fields: &[(Arc<str>, OrdSet<Value>)]) -> Value {
    if fields.is_empty() {
        return Value::set([Value::Record(RecordValue::new())]);
    }

    // Build up records incrementally
    let mut result: Vec<RecordValue> = vec![RecordValue::new()];

    for (name, set) in fields {
        let mut new_result = Vec::new();
        for rec in &result {
            for elem in set.iter() {
                let new_rec = rec.insert(Arc::clone(name), elem.clone());
                new_result.push(new_rec);
            }
        }
        result = new_result;
    }

    Value::set(result.into_iter().map(Value::Record))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_values() {
        assert_eq!(Value::bool(true), Value::Bool(true));
        assert_eq!(Value::bool(false).as_bool(), Some(false));
    }

    #[test]
    fn test_int_values() {
        // Value::int() now creates SmallInt for i64 values
        assert_eq!(Value::int(42), Value::SmallInt(42));
        assert_eq!(Value::int(-5).as_i64(), Some(-5));
        // SmallInt and Int should compare equal for same value
        assert_eq!(Value::SmallInt(42), Value::big_int(BigInt::from(42)));
        // big_int normalizes to SmallInt when value fits
        assert_eq!(Value::big_int(BigInt::from(100)), Value::SmallInt(100));
    }

    #[test]
    fn test_string_values() {
        let s = Value::string("hello");
        assert_eq!(s.as_string(), Some("hello"));
    }

    #[test]
    fn test_set_values() {
        let s = Value::set([Value::int(1), Value::int(2), Value::int(3)]);
        let set = s.as_set().unwrap();
        assert_eq!(set.len(), 3);
        assert!(set.contains(&Value::int(2)));
    }

    #[test]
    fn test_set_deduplication() {
        let s = Value::set([Value::int(1), Value::int(1), Value::int(2)]);
        let set = s.as_set().unwrap();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_seq_values() {
        let s = Value::seq([Value::int(1), Value::int(2), Value::int(3)]);
        let seq = s.as_seq().unwrap();
        assert_eq!(seq.len(), 3);
        assert_eq!(seq[0], Value::int(1));
    }

    #[test]
    fn test_tuple_values() {
        let t = Value::tuple([Value::int(1), Value::string("a"), Value::bool(true)]);
        let tuple = t.as_tuple().unwrap();
        assert_eq!(tuple.len(), 3);
    }

    #[test]
    fn test_record_values() {
        let r = Value::record([("x", Value::int(1)), ("y", Value::int(2))]);
        let rec = r.as_record().unwrap();
        assert_eq!(rec.len(), 2);
        assert_eq!(rec.get(&Arc::from("x")), Some(&Value::int(1)));
    }

    #[test]
    fn test_func_values() {
        let domain: OrdSet<Value> = [Value::int(1), Value::int(2)].into_iter().collect();
        let mapping: OrdMap<Value, Value> = [
            (Value::int(1), Value::string("a")),
            (Value::int(2), Value::string("b")),
        ]
        .into_iter()
        .collect();
        let f = Value::func(domain, mapping);
        let func = f.as_func().unwrap();
        assert_eq!(func.apply(&Value::int(1)), Some(&Value::string("a")));
        assert_eq!(func.apply(&Value::int(3)), None);
    }

    #[test]
    fn test_value_ordering() {
        // Different types ordered by type
        assert!(Value::bool(true) < Value::int(0));
        assert!(Value::int(0) < Value::string(""));

        // Same type ordered by value
        assert!(Value::int(1) < Value::int(2));
        assert!(Value::string("a") < Value::string("b"));
    }

    #[test]
    fn test_range_set() {
        let r = range_set(&BigInt::from(1), &BigInt::from(3));
        // range_set now returns Value::Interval (lazy), so use to_ord_set()
        let set = r.to_ord_set().unwrap();
        assert_eq!(set.len(), 3);
        assert!(set.contains(&Value::int(1)));
        assert!(set.contains(&Value::int(2)));
        assert!(set.contains(&Value::int(3)));
    }

    #[test]
    fn test_empty_range() {
        let r = range_set(&BigInt::from(5), &BigInt::from(3));
        let set = r.as_set().unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn test_powerset() {
        let s: OrdSet<Value> = [Value::int(1), Value::int(2)].into_iter().collect();
        let ps = powerset(&s).unwrap();
        let ps_set = ps.as_set().unwrap();
        assert_eq!(ps_set.len(), 4); // 2^2 = 4

        // Contains empty set
        assert!(ps_set.contains(&Value::empty_set()));
        // Contains full set
        assert!(ps_set.contains(&Value::set([Value::int(1), Value::int(2)])));
    }

    #[test]
    fn test_powerset_too_large() {
        // 21 elements would produce 2^21 > 2M subsets
        let s: OrdSet<Value> = (0..21).map(Value::int).collect();
        let result = powerset(&s);
        assert!(result.is_err());
    }

    #[test]
    fn test_big_union() {
        let s: OrdSet<Value> = [
            Value::set([Value::int(1), Value::int(2)]),
            Value::set([Value::int(2), Value::int(3)]),
        ]
        .into_iter()
        .collect();

        let u = big_union(&s).unwrap();
        let u_set = u.as_set().unwrap();
        assert_eq!(u_set.len(), 3);
        assert!(u_set.contains(&Value::int(1)));
        assert!(u_set.contains(&Value::int(2)));
        assert!(u_set.contains(&Value::int(3)));
    }

    #[test]
    fn test_cartesian_product() {
        let s1: OrdSet<Value> = [Value::int(1), Value::int(2)].into_iter().collect();
        let s2: OrdSet<Value> = [Value::string("a"), Value::string("b")]
            .into_iter()
            .collect();

        let cp = cartesian_product(&[&s1, &s2]);
        let cp_set = cp.as_set().unwrap();
        assert_eq!(cp_set.len(), 4); // 2 * 2

        assert!(cp_set.contains(&Value::tuple([Value::int(1), Value::string("a")])));
        assert!(cp_set.contains(&Value::tuple([Value::int(2), Value::string("b")])));
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Value::bool(true)), "TRUE");
        assert_eq!(format!("{}", Value::int(42)), "42");
        assert_eq!(format!("{}", Value::string("hi")), "\"hi\"");
        assert_eq!(
            format!("{}", Value::set([Value::int(1), Value::int(2)])),
            "{1, 2}"
        );
        assert_eq!(
            format!("{}", Value::seq([Value::int(1), Value::int(2)])),
            "<<1, 2>>"
        );
    }

    // === Lazy Set Operation Tests ===

    #[test]
    fn test_setcup_enumerable() {
        // Two enumerable sets -> can convert to OrdSet
        let s1 = Value::set([Value::int(1), Value::int(2)]);
        let s2 = Value::set([Value::int(2), Value::int(3)]);
        let cup = SetCupValue::new(s1, s2);

        // Membership tests
        assert!(cup.contains(&Value::int(1)));
        assert!(cup.contains(&Value::int(2)));
        assert!(cup.contains(&Value::int(3)));
        assert!(!cup.contains(&Value::int(4)));

        // Should be enumerable
        assert!(cup.is_enumerable());
        let set = cup.to_ord_set().unwrap();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_setcup_with_stringset() {
        // STRING \cup {1, 2} -> non-enumerable union (STRING is infinite)
        let s1 = Value::StringSet;
        let s2 = Value::set([Value::int(1), Value::int(2)]);
        let cup = SetCupValue::new(s1, s2);

        // Membership: strings are in STRING, ints are in the finite set
        assert!(cup.contains(&Value::string("hello")));
        assert!(cup.contains(&Value::int(1)));
        assert!(cup.contains(&Value::int(2)));
        assert!(!cup.contains(&Value::int(3)));

        // Not enumerable (STRING is infinite)
        assert!(!cup.is_enumerable());
        assert!(cup.to_ord_set().is_none());
    }

    #[test]
    fn test_setcup_with_anyset() {
        // ANY \cup {1, 2} -> non-enumerable union (ANY is universal)
        let s1 = Value::AnySet;
        let s2 = Value::set([Value::int(1), Value::int(2)]);
        let cup = SetCupValue::new(s1, s2);

        // Any value should be in the union (ANY contains everything)
        assert!(cup.contains(&Value::string("hello")));
        assert!(cup.contains(&Value::int(42)));
        assert!(cup.contains(&Value::bool(true)));

        // Not enumerable (ANY is infinite/universal)
        assert!(!cup.is_enumerable());
    }

    #[test]
    fn test_setcap_enumerable() {
        // Two enumerable sets -> intersection
        let s1 = Value::set([Value::int(1), Value::int(2), Value::int(3)]);
        let s2 = Value::set([Value::int(2), Value::int(3), Value::int(4)]);
        let cap = SetCapValue::new(s1, s2);

        // Membership tests
        assert!(!cap.contains(&Value::int(1)));
        assert!(cap.contains(&Value::int(2)));
        assert!(cap.contains(&Value::int(3)));
        assert!(!cap.contains(&Value::int(4)));

        // Should be enumerable
        assert!(cap.is_enumerable());
        let set = cap.to_ord_set().unwrap();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_setcap_with_stringset() {
        // STRING \cap {"a", "b", 1, 2} -> only strings pass both membership checks
        let s1 = Value::StringSet;
        let s2 = Value::set([
            Value::string("a"),
            Value::string("b"),
            Value::int(1),
            Value::int(2),
        ]);
        let cap = SetCapValue::new(s1, s2);

        // Membership: only strings in s2 are in the intersection
        assert!(cap.contains(&Value::string("a")));
        assert!(cap.contains(&Value::string("b")));
        assert!(!cap.contains(&Value::int(1))); // INT not in STRING
        assert!(!cap.contains(&Value::int(2)));
        assert!(!cap.contains(&Value::string("c"))); // not in s2

        // Enumerable because s2 is finite
        assert!(cap.is_enumerable());
        let set = cap.to_ord_set().unwrap();
        assert_eq!(set.len(), 2); // "a" and "b"
    }

    #[test]
    fn test_setdiff_enumerable() {
        // {1, 2, 3} \ {2, 3, 4} -> {1}
        let s1 = Value::set([Value::int(1), Value::int(2), Value::int(3)]);
        let s2 = Value::set([Value::int(2), Value::int(3), Value::int(4)]);
        let diff = SetDiffValue::new(s1, s2);

        // Membership tests
        assert!(diff.contains(&Value::int(1)));
        assert!(!diff.contains(&Value::int(2)));
        assert!(!diff.contains(&Value::int(3)));
        assert!(!diff.contains(&Value::int(4)));

        // Should be enumerable
        assert!(diff.is_enumerable());
        let set = diff.to_ord_set().unwrap();
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_setdiff_with_stringset_rhs() {
        // {1, "a", 2, "b"} \ STRING -> {1, 2} (strings removed)
        let s1 = Value::set([
            Value::int(1),
            Value::string("a"),
            Value::int(2),
            Value::string("b"),
        ]);
        let s2 = Value::StringSet;
        let diff = SetDiffValue::new(s1, s2);

        // Membership: only non-strings remain
        assert!(diff.contains(&Value::int(1)));
        assert!(diff.contains(&Value::int(2)));
        assert!(!diff.contains(&Value::string("a")));
        assert!(!diff.contains(&Value::string("b")));

        // Enumerable because LHS is finite
        assert!(diff.is_enumerable());
        let set = diff.to_ord_set().unwrap();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_setdiff_with_anyset_rhs() {
        // {1, 2, 3} \ ANY -> {} (everything removed)
        let s1 = Value::set([Value::int(1), Value::int(2), Value::int(3)]);
        let s2 = Value::AnySet;
        let diff = SetDiffValue::new(s1, s2);

        // Membership: nothing remains (ANY contains everything)
        assert!(!diff.contains(&Value::int(1)));
        assert!(!diff.contains(&Value::int(2)));
        assert!(!diff.contains(&Value::int(3)));

        // Enumerable because LHS is finite
        assert!(diff.is_enumerable());
        let set = diff.to_ord_set().unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn test_lazy_set_ops_value_enum() {
        // Test that Value enum variants work correctly
        let s1 = Value::set([Value::int(1), Value::int(2)]);
        let s2 = Value::set([Value::int(2), Value::int(3)]);

        // Create lazy set operation values
        let cup = Value::SetCup(SetCupValue::new(s1.clone(), s2.clone()));
        let cap = Value::SetCap(SetCapValue::new(s1.clone(), s2.clone()));
        let diff = Value::SetDiff(SetDiffValue::new(s1.clone(), s2.clone()));

        // All should be considered sets
        assert!(cup.is_set());
        assert!(cap.is_set());
        assert!(diff.is_set());

        // All should support set_contains
        assert!(cup.set_contains(&Value::int(1)).unwrap());
        assert!(cap.set_contains(&Value::int(2)).unwrap());
        assert!(diff.set_contains(&Value::int(1)).unwrap());

        // All should support to_ord_set
        assert!(cup.to_ord_set().is_some());
        assert!(cap.to_ord_set().is_some());
        assert!(diff.to_ord_set().is_some());
    }

    #[test]
    fn test_setpred_basic() {
        use tla_core::ast::BoundVar;
        use tla_core::Spanned;

        // Create a basic SetPred value with STRING source
        let source = Value::StringSet;
        let bound = BoundVar {
            name: Spanned::new("x".to_string(), Default::default()),
            domain: None,
            pattern: None,
        };
        let pred = Spanned::new(Expr::Bool(true), Default::default());
        let env = HashMap::new();

        let spv = SetPredValue::new(source, bound, pred, env);

        // Basic properties
        assert!(spv.is_infinite()); // STRING is infinite
        assert!(!spv.is_enumerable()); // Can't enumerate STRING

        // Source contains check
        assert_eq!(spv.source_contains(&Value::string("hello")), Some(true));
        assert_eq!(spv.source_contains(&Value::int(42)), Some(false));

        // Unique IDs
        let source2 = Value::StringSet;
        let bound2 = BoundVar {
            name: Spanned::new("y".to_string(), Default::default()),
            domain: None,
            pattern: None,
        };
        let pred2 = Spanned::new(Expr::Bool(false), Default::default());
        let spv2 = SetPredValue::new(source2, bound2, pred2, HashMap::new());

        assert_ne!(spv.id, spv2.id);
    }

    #[test]
    fn test_setpred_value_variant() {
        use tla_core::ast::BoundVar;
        use tla_core::Spanned;

        // Create a SetPred Value
        let source = Value::AnySet;
        let bound = BoundVar {
            name: Spanned::new("x".to_string(), Default::default()),
            domain: None,
            pattern: None,
        };
        let pred = Spanned::new(Expr::Bool(true), Default::default());
        let spv = SetPredValue::new(source, bound, pred, HashMap::new());
        let val = Value::SetPred(spv);

        // Should be recognized as a set
        assert!(val.is_set());

        // set_contains returns None (requires evaluation context)
        assert!(val.set_contains(&Value::int(42)).is_none());

        // iter_set returns None (non-enumerable source)
        assert!(val.iter_set().is_none());

        // to_ord_set returns None (non-enumerable source)
        assert!(val.to_ord_set().is_none());
    }

    #[test]
    fn test_setpred_ordering() {
        use tla_core::ast::BoundVar;
        use tla_core::Spanned;

        // Create two SetPred values
        let bound = BoundVar {
            name: Spanned::new("x".to_string(), Default::default()),
            domain: None,
            pattern: None,
        };
        let pred = Spanned::new(Expr::Bool(true), Default::default());

        let spv1 = SetPredValue::new(
            Value::StringSet,
            bound.clone(),
            pred.clone(),
            HashMap::new(),
        );
        let spv2 = SetPredValue::new(
            Value::StringSet,
            bound.clone(),
            pred.clone(),
            HashMap::new(),
        );

        // Different IDs should have different ordering
        assert!(spv1.id < spv2.id);
        assert!(spv1 < spv2);

        // Same SetPred should be equal to itself
        assert_eq!(spv1.cmp(&spv1), std::cmp::Ordering::Equal);
    }

    // === KSubsetValue Tests ===

    #[test]
    fn test_ksubset_basic() {
        // Ksubsets({1, 2, 3}, 2) should have C(3,2) = 3 elements
        let base = Value::set([Value::int(1), Value::int(2), Value::int(3)]);
        let ksv = KSubsetValue::new(base, 2);

        // Cardinality
        assert_eq!(ksv.len().unwrap(), BigInt::from(3));

        // Enumerable
        assert!(ksv.is_enumerable());

        // Convert to set
        let set = ksv.to_ord_set().unwrap();
        assert_eq!(set.len(), 3);

        // Each element should be a 2-element set
        for elem in set.iter() {
            let inner = elem.to_ord_set().unwrap();
            assert_eq!(inner.len(), 2);
        }
    }

    #[test]
    fn test_ksubset_membership() {
        let base = Value::set([Value::int(1), Value::int(2), Value::int(3)]);
        let ksv = KSubsetValue::new(base, 2);

        // Valid 2-element subset
        assert!(ksv.contains(&Value::set([Value::int(1), Value::int(2)])));
        assert!(ksv.contains(&Value::set([Value::int(1), Value::int(3)])));
        assert!(ksv.contains(&Value::set([Value::int(2), Value::int(3)])));

        // Invalid: wrong size
        assert!(!ksv.contains(&Value::set([Value::int(1)])));
        assert!(!ksv.contains(&Value::set([Value::int(1), Value::int(2), Value::int(3)])));

        // Invalid: not a subset of base
        assert!(!ksv.contains(&Value::set([Value::int(1), Value::int(4)])));
    }

    #[test]
    fn test_ksubset_edge_cases() {
        let base = Value::set([Value::int(1), Value::int(2), Value::int(3)]);

        // k = 0: returns set containing empty set
        let ksv0 = KSubsetValue::new(base.clone(), 0);
        assert_eq!(ksv0.len().unwrap(), BigInt::from(1));
        let set0 = ksv0.to_ord_set().unwrap();
        assert_eq!(set0.len(), 1);
        assert!(ksv0.contains(&Value::set(vec![])));

        // k > n: returns empty set
        let ksv5 = KSubsetValue::new(base.clone(), 5);
        assert_eq!(ksv5.len().unwrap(), BigInt::from(0));
        assert!(ksv5.is_empty());
        let set5 = ksv5.to_ord_set().unwrap();
        assert!(set5.is_empty());

        // k = n: returns set containing the full base set
        let ksv3 = KSubsetValue::new(base.clone(), 3);
        assert_eq!(ksv3.len().unwrap(), BigInt::from(1));
        let set3 = ksv3.to_ord_set().unwrap();
        assert_eq!(set3.len(), 1);
    }

    #[test]
    fn test_ksubset_value_variant() {
        let base = Value::set([Value::int(1), Value::int(2)]);
        let val = Value::KSubset(KSubsetValue::new(base, 1));

        // Should be recognized as a set
        assert!(val.is_set());

        // set_contains should work
        assert!(val.set_contains(&Value::set([Value::int(1)])).unwrap());
        assert!(val.set_contains(&Value::set([Value::int(2)])).unwrap());
        assert!(!val.set_contains(&Value::set([Value::int(3)])).unwrap());

        // iter_set should work
        let iter = val.iter_set().unwrap();
        assert_eq!(iter.count(), 2);

        // to_ord_set should work
        let set = val.to_ord_set().unwrap();
        assert_eq!(set.len(), 2);
    }

    // === UnionValue Tests ===

    #[test]
    fn test_union_basic() {
        // UNION {{1, 2}, {2, 3}} = {1, 2, 3}
        let set_of_sets = Value::set([
            Value::set([Value::int(1), Value::int(2)]),
            Value::set([Value::int(2), Value::int(3)]),
        ]);
        let uv = UnionValue::new(set_of_sets);

        // Enumerable
        assert!(uv.is_enumerable());

        // Convert to set
        let set = uv.to_ord_set().unwrap();
        assert_eq!(set.len(), 3);
        assert!(set.contains(&Value::int(1)));
        assert!(set.contains(&Value::int(2)));
        assert!(set.contains(&Value::int(3)));
    }

    #[test]
    fn test_union_membership() {
        let set_of_sets = Value::set([
            Value::set([Value::int(1), Value::int(2)]),
            Value::set([Value::int(3), Value::int(4)]),
        ]);
        let uv = UnionValue::new(set_of_sets);

        // Elements in any inner set
        assert!(uv.contains(&Value::int(1)).unwrap());
        assert!(uv.contains(&Value::int(2)).unwrap());
        assert!(uv.contains(&Value::int(3)).unwrap());
        assert!(uv.contains(&Value::int(4)).unwrap());

        // Elements not in any inner set
        assert!(!uv.contains(&Value::int(5)).unwrap());
        assert!(!uv.contains(&Value::int(0)).unwrap());
    }

    #[test]
    fn test_union_empty_cases() {
        // UNION {} = {}
        let empty = Value::set(vec![]);
        let uv_empty = UnionValue::new(empty);
        assert!(uv_empty.is_empty());
        let set = uv_empty.to_ord_set().unwrap();
        assert!(set.is_empty());

        // UNION {{}} = {}
        let set_of_empty = Value::set([Value::set(vec![])]);
        let uv_single_empty = UnionValue::new(set_of_empty);
        assert!(uv_single_empty.is_empty());
        let set2 = uv_single_empty.to_ord_set().unwrap();
        assert!(set2.is_empty());

        // UNION {{1}, {}} = {1}
        let mixed = Value::set([Value::set([Value::int(1)]), Value::set(vec![])]);
        let uv_mixed = UnionValue::new(mixed);
        assert!(!uv_mixed.is_empty());
        let set3 = uv_mixed.to_ord_set().unwrap();
        assert_eq!(set3.len(), 1);
    }

    #[test]
    fn test_union_value_variant() {
        let set_of_sets = Value::set([Value::set([Value::int(1), Value::int(2)])]);
        let val = Value::BigUnion(UnionValue::new(set_of_sets));

        // Should be recognized as a set
        assert!(val.is_set());

        // set_contains should work
        assert!(val.set_contains(&Value::int(1)).unwrap());
        assert!(val.set_contains(&Value::int(2)).unwrap());
        assert!(!val.set_contains(&Value::int(3)).unwrap());

        // iter_set should work
        let iter = val.iter_set().unwrap();
        assert_eq!(iter.count(), 2);

        // to_ord_set should work
        let set = val.to_ord_set().unwrap();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_binomial() {
        // Test binomial coefficient function
        assert_eq!(binomial(5, 0), BigInt::from(1));
        assert_eq!(binomial(5, 1), BigInt::from(5));
        assert_eq!(binomial(5, 2), BigInt::from(10));
        assert_eq!(binomial(5, 3), BigInt::from(10));
        assert_eq!(binomial(5, 4), BigInt::from(5));
        assert_eq!(binomial(5, 5), BigInt::from(1));
        assert_eq!(binomial(5, 6), BigInt::from(0)); // k > n

        // Larger values
        assert_eq!(binomial(10, 5), BigInt::from(252));
        assert_eq!(binomial(20, 10), BigInt::from(184756));
    }

    // === Value::permute tests for symmetry reduction ===

    #[test]
    fn test_permute_model_value() {
        // Create a permutation: @a |-> @b, @b |-> @a
        let a = Value::model_value("a");
        let b = Value::model_value("b");

        let domain: OrdSet<Value> = [a.clone(), b.clone()].into_iter().collect();
        let mapping: OrdMap<Value, Value> = [
            (a.clone(), b.clone()), // a |-> b
            (b.clone(), a.clone()), // b |-> a
        ]
        .into_iter()
        .collect();
        let perm = FuncValue::new(domain, mapping);

        // Permuting model values
        assert_eq!(a.permute(&perm), b);
        assert_eq!(b.permute(&perm), a);

        // Model value not in permutation domain - unchanged
        let c = Value::model_value("c");
        assert_eq!(c.permute(&perm), c);

        // Primitive values - unchanged
        assert_eq!(Value::int(42).permute(&perm), Value::int(42));
        assert_eq!(Value::bool(true).permute(&perm), Value::bool(true));
        assert_eq!(
            Value::string("hello").permute(&perm),
            Value::string("hello")
        );
    }

    #[test]
    fn test_permute_set() {
        let a = Value::model_value("a");
        let b = Value::model_value("b");

        let domain: OrdSet<Value> = [a.clone(), b.clone()].into_iter().collect();
        let mapping: OrdMap<Value, Value> = [(a.clone(), b.clone()), (b.clone(), a.clone())]
            .into_iter()
            .collect();
        let perm = FuncValue::new(domain, mapping);

        // Set {@a, @b, 1} permuted -> {@b, @a, 1}
        let set = Value::set([a.clone(), b.clone(), Value::int(1)]);
        let permuted = set.permute(&perm);

        // Should have same elements (different order won't matter for set equality)
        let permuted_set = permuted.as_set().unwrap();
        assert_eq!(permuted_set.len(), 3);
        assert!(permuted_set.contains(&a)); // @b permuted is @a
        assert!(permuted_set.contains(&b)); // @a permuted is @b
        assert!(permuted_set.contains(&Value::int(1)));
    }

    #[test]
    fn test_permute_seq() {
        let a = Value::model_value("a");
        let b = Value::model_value("b");

        let domain: OrdSet<Value> = [a.clone(), b.clone()].into_iter().collect();
        let mapping: OrdMap<Value, Value> = [(a.clone(), b.clone()), (b.clone(), a.clone())]
            .into_iter()
            .collect();
        let perm = FuncValue::new(domain, mapping);

        // Sequence <<@a, @b, 1>> permuted -> <<@b, @a, 1>>
        let seq = Value::seq([a.clone(), b.clone(), Value::int(1)]);
        let permuted = seq.permute(&perm);

        let permuted_seq = permuted.as_seq().unwrap();
        assert_eq!(permuted_seq.len(), 3);
        assert_eq!(permuted_seq[0], b); // @a -> @b
        assert_eq!(permuted_seq[1], a); // @b -> @a
        assert_eq!(permuted_seq[2], Value::int(1)); // unchanged
    }

    #[test]
    fn test_permute_record() {
        let a = Value::model_value("a");
        let b = Value::model_value("b");

        let domain: OrdSet<Value> = [a.clone(), b.clone()].into_iter().collect();
        let mapping: OrdMap<Value, Value> = [(a.clone(), b.clone()), (b.clone(), a.clone())]
            .into_iter()
            .collect();
        let perm = FuncValue::new(domain, mapping);

        // Record [x |-> @a, y |-> @b] permuted -> [x |-> @b, y |-> @a]
        let rec = Value::record([("x", a.clone()), ("y", b.clone())]);
        let permuted = rec.permute(&perm);

        let permuted_rec = permuted.as_record().unwrap();
        assert_eq!(permuted_rec.get(&Arc::from("x")), Some(&b));
        assert_eq!(permuted_rec.get(&Arc::from("y")), Some(&a));
    }

    #[test]
    fn test_permute_function() {
        let a = Value::model_value("a");
        let b = Value::model_value("b");

        let domain: OrdSet<Value> = [a.clone(), b.clone()].into_iter().collect();
        let mapping: OrdMap<Value, Value> = [(a.clone(), b.clone()), (b.clone(), a.clone())]
            .into_iter()
            .collect();
        let perm = FuncValue::new(domain, mapping);

        // Function [@a |-> 1, @b |-> 2] permuted -> [@b |-> 1, @a |-> 2]
        let func_domain: OrdSet<Value> = [a.clone(), b.clone()].into_iter().collect();
        let func_mapping: OrdMap<Value, Value> =
            [(a.clone(), Value::int(1)), (b.clone(), Value::int(2))]
                .into_iter()
                .collect();
        let func = Value::Func(FuncValue::new(func_domain, func_mapping));
        let permuted = func.permute(&perm);

        let permuted_func = permuted.as_func().unwrap();
        // Domain permuted: @a -> @b, @b -> @a
        assert!(permuted_func.domain_contains(&a));
        assert!(permuted_func.domain_contains(&b));
        // [@a |-> 1, @b |-> 2] with permutation (a<->b) gives [@b |-> 1, @a |-> 2]
        assert_eq!(permuted_func.mapping_get(&b), Some(&Value::int(1)));
        assert_eq!(permuted_func.mapping_get(&a), Some(&Value::int(2)));
    }

    #[test]
    fn test_record_func_hash_equivalence() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_value(v: &Value) -> u64 {
            let mut hasher = DefaultHasher::new();
            v.hash(&mut hasher);
            hasher.finish()
        }

        // Record [a |-> 1, b |-> 2, c |-> 3]
        let rec = Value::record([
            ("a", Value::int(1)),
            ("b", Value::int(2)),
            ("c", Value::int(3)),
        ]);

        // Equivalent Func with domain {"a", "b", "c"}
        let domain: OrdSet<Value> = [Value::string("a"), Value::string("b"), Value::string("c")]
            .into_iter()
            .collect();
        let mapping: OrdMap<Value, Value> = [
            (Value::string("a"), Value::int(1)),
            (Value::string("b"), Value::int(2)),
            (Value::string("c"), Value::int(3)),
        ]
        .into_iter()
        .collect();
        let func = Value::Func(FuncValue::new(domain, mapping));

        // They should be equal (same mathematical function)
        assert_eq!(rec, func, "Record and Func should be equal");

        // They should hash the same
        let rec_hash = hash_value(&rec);
        let func_hash = hash_value(&func);
        assert_eq!(rec_hash, func_hash, "Record and Func should hash the same");
    }

    #[test]
    fn test_recordset_funcset_hash_equivalence() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_value(v: &Value) -> u64 {
            let mut hasher = DefaultHasher::new();
            v.hash(&mut hasher);
            hasher.finish()
        }

        // RecordSet [a : {1,2}, b : {1,2}]
        let rsv = RecordSetValue::new([
            (Arc::from("a"), Value::set([Value::int(1), Value::int(2)])),
            (Arc::from("b"), Value::set([Value::int(1), Value::int(2)])),
        ]);
        let record_set = Value::RecordSet(rsv.clone());

        // Equivalent FuncSet [{"a","b"} -> {1,2}]
        let domain = Value::set([Value::string("a"), Value::string("b")]);
        let codomain = Value::set([Value::int(1), Value::int(2)]);
        let fsv = FuncSetValue::new(domain, codomain);
        let func_set = Value::FuncSet(fsv.clone());

        // They should be equal (same set of functions)
        assert_eq!(
            record_set, func_set,
            "RecordSet and FuncSet should be equal"
        );

        // They should hash the same
        let rsv_hash = hash_value(&record_set);
        let fsv_hash = hash_value(&func_set);
        assert_eq!(
            rsv_hash, fsv_hash,
            "RecordSet and FuncSet should hash the same"
        );
    }

    #[test]
    fn test_smallint_bigint_hash_equivalence() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_value(v: &Value) -> u64 {
            let mut hasher = DefaultHasher::new();
            v.hash(&mut hasher);
            hasher.finish()
        }

        // Test that SmallInt and equivalent BigInt hash the same
        let test_values: &[i64] = &[
            0,
            1,
            -1,
            5,
            -5,
            127,
            128,
            255,
            256,
            1000,
            -1000,
            i64::MAX,
            i64::MIN,
        ];

        for &n in test_values {
            let small = Value::SmallInt(n);
            // Create a BigInt directly (bypassing normalization)
            let big = Value::Int(BigInt::from(n));

            // They should be equal
            assert_eq!(
                small, big,
                "SmallInt({}) should equal Int(BigInt({}))",
                n, n
            );

            // They should hash the same
            let small_hash = hash_value(&small);
            let big_hash = hash_value(&big);
            assert_eq!(
                small_hash, big_hash,
                "SmallInt({}) and Int(BigInt({})) should hash the same",
                n, n
            );
        }
    }

    #[test]
    fn test_funcset_iterator_produces_seq_for_domain_1_n() {
        // FuncSetIterator should produce Seq values when domain is 1..n
        // (because in TLA+, functions with domain 1..n are semantically sequences)
        use crate::value::{FuncSetValue, IntervalValue, SortedSet};

        // Create [1..4 -> {"A", "B"}]
        let domain = Value::Interval(IntervalValue::new(BigInt::from(1), BigInt::from(4)));
        let codomain = Value::Set(SortedSet::from_iter(vec![
            Value::String("A".into()),
            Value::String("B".into()),
        ]));

        let func_set = FuncSetValue::new(domain, codomain);
        let iter = func_set.iter().expect("should be able to iterate");

        // Check that all produced values are Seq (domain 1..n creates sequences)
        let mut found_seq = false;
        for func in iter.take(5) {
            match func {
                Value::Seq(s) => {
                    found_seq = true;
                    // All sequence elements should be in codomain
                    assert_eq!(s.len(), 4, "Sequence should have length 4 (domain 1..4)");
                }
                _ => panic!(
                    "FuncSetIterator produced unexpected type for domain 1..n: {:?}",
                    func
                ),
            }
        }

        // We should find Seq (not IntFunc or Func) for domain 1..n
        assert!(
            found_seq,
            "FuncSetIterator should produce Seq for domain starting at 1"
        );
    }

    #[test]
    fn test_funcset_iterator_produces_intfunc_for_non_one_start() {
        // FuncSetIterator should produce IntFunc when domain is NOT 1..n (e.g., 2..5)
        use crate::value::{FuncSetValue, IntervalValue, SortedSet};

        // Create [2..4 -> {"A", "B"}] - domain starts at 2, not 1
        let domain = Value::Interval(IntervalValue::new(BigInt::from(2), BigInt::from(4)));
        let codomain = Value::Set(SortedSet::from_iter(vec![
            Value::String("A".into()),
            Value::String("B".into()),
        ]));

        let func_set = FuncSetValue::new(domain, codomain);
        let iter = func_set.iter().expect("should be able to iterate");

        // Check that all produced values are IntFunc (domain 2..n is NOT a sequence)
        let mut found_intfunc = false;
        for func in iter.take(5) {
            match func {
                Value::IntFunc(_) => found_intfunc = true,
                _ => panic!(
                    "FuncSetIterator produced unexpected type for domain 2..n: {:?}",
                    func
                ),
            }
        }

        assert!(
            found_intfunc,
            "FuncSetIterator should produce IntFunc for domain not starting at 1"
        );
    }
}
