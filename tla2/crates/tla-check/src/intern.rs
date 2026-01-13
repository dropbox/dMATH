//! Value Interning for TLA+ Model Checking
//!
//! This module provides a value interning system that eliminates Arc atomic
//! contention during parallel model checking. Values are stored once in a
//! global table and referenced by lightweight handles.
//!
//! # Design
//!
//! In standard TLA2, `ArrayState` stores `Box<[Value]>` where `Value` contains
//! Arc-based shared data (Sets, Funcs, Seqs, etc.). Cloning a state requires
//! atomic reference count increments for each Arc, causing cache line bouncing
//! and contention across threads.
//!
//! Value interning replaces this with:
//! - `ValueHandle(u64)`: A lightweight handle (the value's fingerprint)
//! - `ValueInterner`: A global table mapping handles to values
//! - `HandleState`: State as `Box<[ValueHandle]>` - pure memcpy clone
//!
//! # Performance
//!
//! - State cloning: O(n) memcpy instead of O(n) atomic operations
//! - Value deduplication: Automatic (same value = same fingerprint = same handle)
//! - Value lookup: O(1) DashMap access when needed
//!
//! # Usage
//!
//! ```ignore
//! let interner = ValueInterner::new();
//! let handle = interner.intern(Value::SmallInt(42));
//! let value = interner.get(handle);
//! ```

use crate::state::{value_fingerprint, DiffSuccessor, Fingerprint};
use crate::var_index::{VarIndex, VarRegistry};
use crate::Value;
use dashmap::DashMap;
use rustc_hash::FxHasher;
use std::hash::BuildHasherDefault;
use std::sync::OnceLock;

/// FxHasher-based BuildHasher for faster hashing of u64 handles.
type FxBuildHasher = BuildHasherDefault<FxHasher>;

/// FxHasher-based DashMap for concurrent handle -> Value storage.
type FxDashMap<K, V> = DashMap<K, V, FxBuildHasher>;

// ============================================================================
// ValueHandle - Lightweight reference to an interned value
// ============================================================================

/// A lightweight handle to an interned value.
///
/// This is just the value's 64-bit fingerprint, used as a key into the
/// global value table. Handles are `Copy` and can be cloned without
/// any atomic operations.
///
/// # Invariants
///
/// - A valid handle always corresponds to a value in the interner
/// - Two handles are equal iff they refer to the same value
/// - The handle value is the fingerprint of the interned value
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ValueHandle(pub u64);

impl ValueHandle {
    /// Create a handle from a raw fingerprint value.
    ///
    /// # Safety
    ///
    /// The caller must ensure the fingerprint corresponds to a value
    /// that has been interned in the global interner.
    #[inline]
    pub const fn from_raw(fp: u64) -> Self {
        ValueHandle(fp)
    }

    /// Get the raw fingerprint value.
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

// ============================================================================
// ValueInterner - Global thread-safe value table
// ============================================================================

/// Global thread-safe value interner.
///
/// Values are stored in a DashMap keyed by their fingerprint. The interner
/// provides O(1) insertion and lookup with minimal contention.
///
/// # Thread Safety
///
/// The interner uses DashMap which provides fine-grained locking at the
/// shard level, allowing multiple threads to insert/lookup concurrently
/// with minimal blocking.
pub struct ValueInterner {
    /// Map from fingerprint to value
    values: FxDashMap<u64, Value>,
}

impl ValueInterner {
    /// Create a new value interner.
    pub fn new() -> Self {
        ValueInterner {
            values: DashMap::with_hasher(FxBuildHasher::default()),
        }
    }

    /// Intern a value and return its handle.
    ///
    /// If the value (by fingerprint) was already interned, returns the
    /// existing handle. Otherwise, stores the value and returns a new handle.
    ///
    /// # Performance
    ///
    /// This computes the value's fingerprint (O(value size)) and does a
    /// DashMap insertion (O(1) amortized).
    #[inline]
    pub fn intern(&self, value: Value) -> ValueHandle {
        let fp = value_fingerprint(&value);
        // Use entry API to avoid race conditions
        self.values.entry(fp).or_insert(value);
        ValueHandle(fp)
    }

    /// Intern a value with a pre-computed fingerprint.
    ///
    /// Use this when you already have the fingerprint (e.g., from DiffSuccessor)
    /// to avoid recomputing it.
    #[inline]
    pub fn intern_with_fp(&self, value: Value, fp: u64) -> ValueHandle {
        self.values.entry(fp).or_insert(value);
        ValueHandle(fp)
    }

    /// Get the value for a handle.
    ///
    /// # Panics
    ///
    /// Panics if the handle is invalid (not interned).
    #[inline]
    pub fn get(&self, handle: ValueHandle) -> Value {
        self.values
            .get(&handle.0)
            .map(|v| v.clone())
            .expect("invalid handle: value not interned")
    }

    /// Get the value for a handle, returning None if not found.
    #[inline]
    pub fn try_get(&self, handle: ValueHandle) -> Option<Value> {
        self.values.get(&handle.0).map(|v| v.clone())
    }

    /// Check if a handle is valid (has been interned).
    #[inline]
    pub fn contains(&self, handle: ValueHandle) -> bool {
        self.values.contains_key(&handle.0)
    }

    /// Get the number of interned values.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the interner is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Clear all interned values.
    ///
    /// # Warning
    ///
    /// This invalidates all existing handles!
    pub fn clear(&self) {
        self.values.clear();
    }
}

impl Default for ValueInterner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Global Interner Instance
// ============================================================================

/// Global value interner singleton.
///
/// Use `get_interner()` to access this.
static GLOBAL_INTERNER: OnceLock<ValueInterner> = OnceLock::new();

/// Get the global value interner.
///
/// This initializes the interner on first call.
#[inline]
pub fn get_interner() -> &'static ValueInterner {
    GLOBAL_INTERNER.get_or_init(ValueInterner::new)
}

/// Clear the global interner.
///
/// # Warning
///
/// This invalidates all existing handles! Only call between model checking runs.
pub fn clear_global_interner() {
    if let Some(interner) = GLOBAL_INTERNER.get() {
        interner.clear();
    }
}

// ============================================================================
// HandleState - Lightweight state using handles
// ============================================================================

/// Array-based state using value handles instead of values.
///
/// This is a lightweight alternative to `ArrayState` that uses `ValueHandle`
/// instead of `Value`. Cloning this state is a pure memcpy with no atomic
/// operations, eliminating the Arc contention that limits parallel scaling.
///
/// # Usage
///
/// HandleState is used during parallel BFS exploration where state cloning
/// is a hot path. Values are looked up from the interner only when needed
/// (invariant checking, trace reconstruction).
#[derive(Clone)]
pub struct HandleState {
    /// Value handles indexed by VarIndex
    handles: Box<[ValueHandle]>,
    /// Pre-computed state fingerprint (from handle fingerprints)
    fingerprint: Fingerprint,
}

impl HandleState {
    /// Create a HandleState from an existing set of handles and fingerprint.
    #[inline]
    pub fn new(handles: Box<[ValueHandle]>, fingerprint: Fingerprint) -> Self {
        HandleState {
            handles,
            fingerprint,
        }
    }

    /// Create a HandleState by interning values from a regular ArrayState.
    pub fn from_values(values: &[Value], registry: &VarRegistry, interner: &ValueInterner) -> Self {
        let handles: Vec<ValueHandle> = values.iter().map(|v| interner.intern(v.clone())).collect();

        // Compute fingerprint from handle fingerprints
        let fingerprint = compute_handle_fingerprint(&handles, registry);

        HandleState {
            handles: handles.into_boxed_slice(),
            fingerprint,
        }
    }

    /// Get the handle at an index.
    #[inline]
    pub fn get(&self, idx: VarIndex) -> ValueHandle {
        self.handles[idx.as_usize()]
    }

    /// Get all handles.
    #[inline]
    pub fn handles(&self) -> &[ValueHandle] {
        &self.handles
    }

    /// Get the pre-computed fingerprint.
    #[inline]
    pub fn fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }

    /// Get the number of variables.
    #[inline]
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    /// Materialize values from handles using the interner.
    ///
    /// This is used when actual values are needed (invariant checking, etc.)
    pub fn materialize(&self, interner: &ValueInterner) -> Vec<Value> {
        self.handles.iter().map(|h| interner.get(*h)).collect()
    }

    /// Get values as a slice for array-based evaluation.
    ///
    /// This materializes all values into a Vec and returns it.
    /// Use this for compiled guard evaluation.
    #[inline]
    pub fn values(&self, interner: &ValueInterner) -> Vec<Value> {
        self.materialize(interner)
    }
}

impl DiffSuccessor {
    /// Convert a DiffSuccessor to HandleState by interning values.
    ///
    /// This is the key optimization: instead of cloning the base ArrayState
    /// (which clones all Arc values), we clone the base HandleState (memcpy)
    /// and intern only the new values.
    ///
    /// # Arguments
    ///
    /// * `base` - The parent HandleState
    /// * `interner` - The value interner to store new values
    ///
    /// # Returns
    ///
    /// A new HandleState with the changes applied
    pub fn into_handle_state(self, base: &HandleState, interner: &ValueInterner) -> HandleState {
        // Clone handles (memcpy - no Arc operations!)
        let mut handles: Vec<ValueHandle> = base.handles.to_vec();

        // Intern new values and update handles
        for (idx, value) in self.changes {
            let fp = value_fingerprint(&value);
            let handle = interner.intern_with_fp(value, fp);
            handles[idx.as_usize()] = handle;
        }

        HandleState {
            handles: handles.into_boxed_slice(),
            fingerprint: self.fingerprint,
        }
    }
}

// ============================================================================
// DiffHandleSuccessor - Lightweight successor for parallel exploration
// ============================================================================

/// A successor state represented as a diff from the parent, using handles.
///
/// This is the handle-based equivalent of `DiffSuccessor`. It stores only
/// the changed variable handles and the pre-computed fingerprint.
#[derive(Clone)]
pub struct DiffHandleSuccessor {
    /// Pre-computed fingerprint of the successor state
    pub fingerprint: Fingerprint,
    /// Changed variables: (index, new handle)
    pub changes: Vec<(VarIndex, ValueHandle)>,
}

impl DiffHandleSuccessor {
    /// Create a new DiffHandleSuccessor.
    #[inline]
    pub fn new(fingerprint: Fingerprint, changes: Vec<(VarIndex, ValueHandle)>) -> Self {
        DiffHandleSuccessor {
            fingerprint,
            changes,
        }
    }

    /// Materialize into a HandleState by applying changes to the base.
    #[inline]
    pub fn into_handle_state(self, base: &HandleState) -> HandleState {
        let mut handles: Vec<ValueHandle> = base.handles.to_vec();

        for (idx, handle) in self.changes {
            handles[idx.as_usize()] = handle;
        }

        HandleState {
            handles: handles.into_boxed_slice(),
            fingerprint: self.fingerprint,
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute a state fingerprint from handle fingerprints.
///
/// Uses the same algorithm as ArrayState fingerprinting for consistency.
fn compute_handle_fingerprint(handles: &[ValueHandle], registry: &VarRegistry) -> Fingerprint {
    const FNV_PRIME: u64 = 0x100000001b3;

    // XOR together salted handle values
    let mut combined_xor: u64 = 0;
    for (i, handle) in handles.iter().enumerate() {
        let idx = VarIndex(i as u16);
        let salt = registry.fp_salt(idx);
        // Add 1 to handle value to avoid zero-contribution for zero fingerprints
        let contribution = salt.wrapping_mul(handle.0.wrapping_add(1));
        combined_xor ^= contribution;
    }

    // Final mixing
    let mut mixed = combined_xor;
    mixed ^= mixed >> 33;
    mixed = mixed.wrapping_mul(FNV_PRIME);
    mixed ^= mixed >> 33;

    Fingerprint(mixed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_handle_copy() {
        let handle = ValueHandle(42);
        let copy = handle; // Copy, not move
        assert_eq!(handle, copy);
    }

    #[test]
    fn test_interner_basic() {
        let interner = ValueInterner::new();

        let v1 = Value::SmallInt(42);
        let v2 = Value::SmallInt(42);
        let v3 = Value::SmallInt(100);

        let h1 = interner.intern(v1);
        let h2 = interner.intern(v2);
        let h3 = interner.intern(v3);

        // Same value = same handle
        assert_eq!(h1, h2);
        // Different value = different handle
        assert_ne!(h1, h3);

        // Can retrieve values
        assert_eq!(interner.get(h1), Value::SmallInt(42));
        assert_eq!(interner.get(h3), Value::SmallInt(100));
    }

    #[test]
    fn test_handle_state_clone_is_fast() {
        // Create a HandleState
        let handles = vec![ValueHandle(1), ValueHandle(2), ValueHandle(3)];
        let state = HandleState {
            handles: handles.into_boxed_slice(),
            fingerprint: Fingerprint(12345),
        };

        // Clone should be fast (memcpy)
        let cloned = state.clone();
        assert_eq!(state.fingerprint(), cloned.fingerprint());
        assert_eq!(state.handles(), cloned.handles());
    }
}
