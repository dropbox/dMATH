//! Memory arena for efficient state storage
//!
//! This module provides arena-based allocation to reduce heap fragmentation
//! during model checking. Instead of individual heap allocations for each
//! state's backing storage, we allocate from pre-sized memory regions.
//!
//! # Benefits
//! - O(1) allocation (bump pointer increment)
//! - No per-object malloc overhead
//! - Better cache locality (contiguous memory)
//! - Bulk deallocation (drop entire arena)
//!
//! # Usage
//! ```ignore
//! let arena = StateArena::with_capacity(1_000_000); // 1M states
//! let slice = arena.alloc_slice(&values);
//! // slice is valid for arena's lifetime
//! ```

use bumpalo::Bump;
use std::cell::RefCell;

use crate::Value;

/// Pre-allocated memory arena for state storage.
///
/// The arena is optimized for the model checking use case:
/// - States are allocated during BFS exploration
/// - All states are dropped together at the end
/// - No individual deallocation needed
pub struct StateArena {
    /// The underlying bump allocator
    bump: Bump,
    /// Statistics: total bytes allocated
    allocated_bytes: RefCell<usize>,
    /// Statistics: number of allocations
    allocation_count: RefCell<usize>,
}

impl StateArena {
    /// Create an arena pre-sized for the expected number of states.
    ///
    /// Default estimate: 500 bytes per state. For MCBakery with 655K states,
    /// this would pre-allocate ~328 MB.
    ///
    /// # Arguments
    /// * `estimated_states` - Expected number of states to explore
    pub fn with_capacity(estimated_states: usize) -> Self {
        // Estimate 500 bytes per state (target from roadmap)
        let bytes = estimated_states.saturating_mul(500);
        StateArena {
            bump: Bump::with_capacity(bytes),
            allocated_bytes: RefCell::new(0),
            allocation_count: RefCell::new(0),
        }
    }

    /// Create an arena with specific byte capacity.
    pub fn with_byte_capacity(bytes: usize) -> Self {
        StateArena {
            bump: Bump::with_capacity(bytes),
            allocated_bytes: RefCell::new(0),
            allocation_count: RefCell::new(0),
        }
    }

    /// Create a default arena (1M state capacity).
    pub fn new() -> Self {
        Self::with_capacity(1_000_000)
    }

    /// Allocate a slice of Values by cloning from a source slice.
    ///
    /// The returned slice is valid for the lifetime of the arena.
    pub fn alloc_slice(&self, values: &[Value]) -> &[Value] {
        if values.is_empty() {
            return &[];
        }

        // Track statistics
        let size = std::mem::size_of_val(values);
        *self.allocated_bytes.borrow_mut() += size;
        *self.allocation_count.borrow_mut() += 1;

        // Allocate and initialize with clones
        self.bump.alloc_slice_fill_iter(values.iter().cloned())
    }

    /// Allocate a Vec's contents into the arena, returning a slice.
    ///
    /// This is useful when you've built a Vec and want to store it
    /// in arena memory.
    pub fn alloc_vec(&self, vec: Vec<Value>) -> &[Value] {
        if vec.is_empty() {
            return &[];
        }

        let size = std::mem::size_of::<Value>() * vec.len();
        *self.allocated_bytes.borrow_mut() += size;
        *self.allocation_count.borrow_mut() += 1;

        self.bump.alloc_slice_fill_iter(vec)
    }

    /// Allocate an uninitialized slice of Values.
    ///
    /// # Safety
    /// Caller must initialize all elements before reading them.
    pub fn alloc_uninit(&self, len: usize) -> &mut [std::mem::MaybeUninit<Value>] {
        if len == 0 {
            return &mut [];
        }

        let size = std::mem::size_of::<Value>() * len;
        *self.allocated_bytes.borrow_mut() += size;
        *self.allocation_count.borrow_mut() += 1;

        self.bump
            .alloc_slice_fill_with(len, |_| std::mem::MaybeUninit::uninit())
    }

    /// Allocate a single Value.
    pub fn alloc_value(&self, value: Value) -> &Value {
        let size = std::mem::size_of::<Value>();
        *self.allocated_bytes.borrow_mut() += size;
        *self.allocation_count.borrow_mut() += 1;

        self.bump.alloc(value)
    }

    /// Get the total bytes allocated from this arena.
    pub fn allocated_bytes(&self) -> usize {
        *self.allocated_bytes.borrow()
    }

    /// Get the number of allocations made from this arena.
    pub fn allocation_count(&self) -> usize {
        *self.allocation_count.borrow()
    }

    /// Get the underlying bump allocator's current chunk capacity.
    pub fn chunk_capacity(&self) -> usize {
        self.bump.chunk_capacity()
    }

    /// Get the total memory allocated by the bump allocator.
    pub fn allocated_bytes_including_metadata(&self) -> usize {
        self.bump.allocated_bytes()
    }

    /// Reset the arena, deallocating all memory and starting fresh.
    ///
    /// This is useful for reusing an arena across multiple model checking runs.
    pub fn reset(&mut self) {
        self.bump.reset();
        *self.allocated_bytes.borrow_mut() = 0;
        *self.allocation_count.borrow_mut() = 0;
    }
}

impl Default for StateArena {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for StateArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateArena")
            .field("allocated_bytes", &self.allocated_bytes())
            .field("allocation_count", &self.allocation_count())
            .field("chunk_capacity", &self.chunk_capacity())
            .finish()
    }
}

/// Thread-local arena for parallel model checking.
///
/// Each worker thread gets its own arena to avoid contention.
/// This is used when the ModelChecker is run in parallel mode.
#[derive(Default)]
pub struct ThreadLocalArena {
    arena: StateArena,
}

impl ThreadLocalArena {
    /// Create a new thread-local arena with default capacity.
    pub fn new() -> Self {
        ThreadLocalArena {
            arena: StateArena::new(),
        }
    }

    /// Create with specific state capacity.
    pub fn with_capacity(estimated_states: usize) -> Self {
        ThreadLocalArena {
            arena: StateArena::with_capacity(estimated_states),
        }
    }

    /// Get a reference to the arena.
    pub fn get(&self) -> &StateArena {
        &self.arena
    }

    /// Reset the arena for reuse.
    pub fn reset(&mut self) {
        self.arena.reset();
    }
}

// ============================================================================
// BulkStateStorage - Contiguous storage for multiple states
// ============================================================================

/// Contiguous storage for multiple states, reducing heap fragmentation.
///
/// Instead of 655K individual `Box<[Value]>` allocations (one per state),
/// this stores all values in a single large buffer. States are accessed
/// by index into the buffer.
///
/// # Memory Layout
///
/// ```text
/// [state_0_var_0, state_0_var_1, ..., state_1_var_0, state_1_var_1, ...]
/// |<------- vars_per_state ------->|<------- vars_per_state ------->|
/// ```
///
/// # Usage
///
/// ```ignore
/// // Pre-allocate for 1M states with 5 variables each
/// let mut storage = BulkStateStorage::new(5, 1_000_000);
///
/// // Add a state
/// let idx = storage.push_state(&values);
///
/// // Access state values
/// let values = storage.get_state(idx);
/// ```
pub struct BulkStateStorage {
    /// Contiguous storage for all state values
    values: Vec<Value>,
    /// Number of variables per state
    vars_per_state: usize,
    /// Number of states currently stored
    num_states: usize,
    /// Cached fingerprints (fingerprint, combined_xor) per state
    fingerprints: Vec<Option<(Fingerprint, u64)>>,
}

use crate::state::Fingerprint;

impl BulkStateStorage {
    /// Create bulk storage with capacity for the given number of states.
    ///
    /// # Arguments
    /// * `vars_per_state` - Number of variables per state (from VarRegistry)
    /// * `capacity` - Initial capacity in number of states
    pub fn new(vars_per_state: usize, capacity: usize) -> Self {
        let total_values = capacity.saturating_mul(vars_per_state);
        BulkStateStorage {
            values: Vec::with_capacity(total_values),
            vars_per_state,
            num_states: 0,
            fingerprints: Vec::with_capacity(capacity),
        }
    }

    /// Create empty storage (no pre-allocation)
    pub fn empty(vars_per_state: usize) -> Self {
        BulkStateStorage {
            values: Vec::new(),
            vars_per_state,
            num_states: 0,
            fingerprints: Vec::new(),
        }
    }

    /// Add a state and return its index.
    ///
    /// # Arguments
    /// * `values` - Values for the state (must have length == vars_per_state)
    ///
    /// # Panics
    /// Panics if `values.len() != vars_per_state`
    pub fn push_state(&mut self, values: &[Value]) -> u32 {
        assert_eq!(
            values.len(),
            self.vars_per_state,
            "State values length mismatch"
        );

        let idx = self.num_states;
        self.values.extend(values.iter().cloned());
        self.fingerprints.push(None);
        self.num_states += 1;
        idx as u32
    }

    /// Add a state from an iterator of values.
    pub fn push_state_iter(&mut self, values: impl IntoIterator<Item = Value>) -> u32 {
        let idx = self.num_states;
        let start_len = self.values.len();

        self.values.extend(values);

        // Verify we got the expected number of values
        let added = self.values.len() - start_len;
        assert_eq!(
            added, self.vars_per_state,
            "State values count mismatch: expected {}, got {}",
            self.vars_per_state, added
        );

        self.fingerprints.push(None);
        self.num_states += 1;
        idx as u32
    }

    /// Get values for a state by index.
    #[inline]
    pub fn get_state(&self, idx: u32) -> &[Value] {
        let start = (idx as usize) * self.vars_per_state;
        &self.values[start..start + self.vars_per_state]
    }

    /// Get mutable values for a state by index.
    #[inline]
    pub fn get_state_mut(&mut self, idx: u32) -> &mut [Value] {
        let start = (idx as usize) * self.vars_per_state;
        // Invalidate fingerprint cache
        self.fingerprints[idx as usize] = None;
        &mut self.values[start..start + self.vars_per_state]
    }

    /// Get a specific value from a state.
    #[inline]
    pub fn get_value(&self, state_idx: u32, var_idx: usize) -> &Value {
        let offset = (state_idx as usize) * self.vars_per_state + var_idx;
        &self.values[offset]
    }

    /// Set a specific value in a state.
    #[inline]
    pub fn set_value(&mut self, state_idx: u32, var_idx: usize, value: Value) {
        let offset = (state_idx as usize) * self.vars_per_state + var_idx;
        self.values[offset] = value;
        // Invalidate fingerprint cache
        self.fingerprints[state_idx as usize] = None;
    }

    /// Get or compute fingerprint for a state.
    pub fn fingerprint(
        &mut self,
        idx: u32,
        registry: &crate::var_index::VarRegistry,
    ) -> Fingerprint {
        if let Some((fp, _)) = self.fingerprints[idx as usize] {
            return fp;
        }

        // Compute fingerprint using same algorithm as ArrayState
        let values = self.get_state(idx);
        let fp = crate::state::compute_fingerprint_from_array(values, registry);

        // Cache the fingerprint (combined_xor would need separate computation, store 0 for now)
        self.fingerprints[idx as usize] = Some((fp, 0));
        fp
    }

    /// Get cached fingerprint if available.
    #[inline]
    pub fn cached_fingerprint(&self, idx: u32) -> Option<Fingerprint> {
        self.fingerprints[idx as usize].map(|(fp, _)| fp)
    }

    /// Number of states stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.num_states
    }

    /// Whether storage is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_states == 0
    }

    /// Number of variables per state.
    #[inline]
    pub fn vars_per_state(&self) -> usize {
        self.vars_per_state
    }

    /// Total memory used in bytes (approximate).
    pub fn memory_usage(&self) -> usize {
        // Vec<Value> capacity
        let values_mem = self.values.capacity() * std::mem::size_of::<Value>();
        // Vec fingerprints
        let fp_mem =
            self.fingerprints.capacity() * std::mem::size_of::<Option<(Fingerprint, u64)>>();
        values_mem + fp_mem
    }

    /// Reserve capacity for additional states.
    pub fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional * self.vars_per_state);
        self.fingerprints.reserve(additional);
    }
}

impl std::fmt::Debug for BulkStateStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BulkStateStorage")
            .field("num_states", &self.num_states)
            .field("vars_per_state", &self.vars_per_state)
            .field("memory_usage", &self.memory_usage())
            .finish()
    }
}

/// A handle to a state stored in BulkStateStorage.
///
/// This is a lightweight reference (8 bytes) that can be used instead of
/// cloning an entire ArrayState. The handle is valid as long as the
/// BulkStateStorage is not dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BulkStateHandle {
    /// Index into BulkStateStorage
    pub index: u32,
    /// Cached fingerprint (if computed)
    pub fingerprint: Option<Fingerprint>,
}

impl BulkStateHandle {
    /// Create a new handle with the given index.
    #[inline]
    pub fn new(index: u32) -> Self {
        BulkStateHandle {
            index,
            fingerprint: None,
        }
    }

    /// Create a handle with a pre-computed fingerprint.
    #[inline]
    pub fn with_fingerprint(index: u32, fingerprint: Fingerprint) -> Self {
        BulkStateHandle {
            index,
            fingerprint: Some(fingerprint),
        }
    }
}

impl BulkStateStorage {
    /// Push a state from a value slice directly (no State/OrdMap creation).
    ///
    /// Values must be in VarRegistry index order. This is the most memory-efficient
    /// way to add states, avoiding OrdMap allocation entirely.
    ///
    /// # Arguments
    /// * `values` - Values in VarRegistry index order
    ///
    /// # Returns
    /// The index of the newly added state.
    ///
    /// # Panics
    /// Panics if `values.len() != vars_per_state`.
    #[inline]
    pub fn push_from_values(&mut self, values: &[Value]) -> u32 {
        assert_eq!(
            values.len(),
            self.vars_per_state,
            "Value slice size mismatch: expected {}, got {}",
            self.vars_per_state,
            values.len()
        );

        let idx = self.num_states;

        // Push all values - they're already in registry order
        self.values.extend_from_slice(values);
        self.fingerprints.push(None);
        self.num_states += 1;
        idx as u32
    }

    /// Convert a State to a bulk state entry, returning the index.
    ///
    /// This is useful for batch-converting initial states from State to
    /// bulk storage format.
    pub fn push_from_state(
        &mut self,
        state: &crate::state::State,
        registry: &crate::var_index::VarRegistry,
    ) -> u32 {
        assert_eq!(
            registry.len(),
            self.vars_per_state,
            "Registry size mismatch"
        );

        let idx = self.num_states;

        // Add values in registry index order
        for (var_idx, _name) in registry.iter() {
            let name = registry.name(var_idx);
            let value = state.get(name).cloned().unwrap_or(Value::Bool(false));
            self.values.push(value);
        }

        self.fingerprints.push(None);
        self.num_states += 1;
        idx as u32
    }

    /// Batch convert multiple States to bulk storage.
    ///
    /// This is more efficient than individual push_from_state calls because
    /// it pre-allocates the entire buffer.
    pub fn push_states_from_iter<'a>(
        &mut self,
        states: impl Iterator<Item = &'a crate::state::State>,
        registry: &crate::var_index::VarRegistry,
    ) -> Vec<u32> {
        states
            .map(|state| self.push_from_state(state, registry))
            .collect()
    }

    /// Create a BulkStateStorage from a slice of States.
    ///
    /// This is the most efficient way to bulk-convert initial states.
    pub fn from_states(
        states: &[crate::state::State],
        registry: &crate::var_index::VarRegistry,
    ) -> Self {
        let num_states = states.len();
        let vars_per_state = registry.len();
        let mut storage = BulkStateStorage::new(vars_per_state, num_states);

        for state in states {
            storage.push_from_state(state, registry);
        }

        storage
    }

    /// Convert a bulk state to an ArrayState.
    ///
    /// This creates an independent ArrayState that owns its values.
    /// Use sparingly - the point of bulk storage is to avoid this.
    pub fn to_array_state(&self, idx: u32) -> crate::state::ArrayState {
        let values = self.get_state(idx);
        crate::state::ArrayState::from_values(values.to_vec())
    }

    /// Get an iterator over all state handles.
    pub fn handles(&self) -> impl Iterator<Item = BulkStateHandle> {
        (0..self.num_states as u32).map(BulkStateHandle::new)
    }

    /// Get an iterator over all state handles with fingerprints.
    ///
    /// Computes fingerprints for all states that don't have them cached.
    pub fn handles_with_fingerprints(
        &mut self,
        registry: &crate::var_index::VarRegistry,
    ) -> Vec<BulkStateHandle> {
        (0..self.num_states as u32)
            .map(|idx| {
                let fp = self.fingerprint(idx, registry);
                BulkStateHandle::with_fingerprint(idx, fp)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic() {
        let arena = StateArena::new();

        // Allocate some values
        let values = vec![Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)];
        let slice = arena.alloc_slice(&values);

        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0], Value::SmallInt(1));
        assert_eq!(slice[1], Value::SmallInt(2));
        assert_eq!(slice[2], Value::SmallInt(3));

        // Statistics should be updated
        assert!(arena.allocated_bytes() > 0);
        assert_eq!(arena.allocation_count(), 1);
    }

    #[test]
    fn test_arena_empty_slice() {
        let arena = StateArena::new();
        let slice = arena.alloc_slice(&[]);
        assert!(slice.is_empty());
        assert_eq!(arena.allocation_count(), 0);
    }

    #[test]
    fn test_arena_vec() {
        let arena = StateArena::new();
        let vec = vec![Value::Bool(true), Value::Bool(false)];
        let slice = arena.alloc_vec(vec);

        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0], Value::Bool(true));
        assert_eq!(slice[1], Value::Bool(false));
    }

    #[test]
    fn test_arena_capacity() {
        // Pre-allocate for 1000 states
        let arena = StateArena::with_capacity(1000);
        // Should have at least 500KB capacity (1000 * 500 bytes)
        assert!(arena.chunk_capacity() >= 500_000);
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = StateArena::new();

        // Allocate some data
        let _ = arena.alloc_slice(&[Value::SmallInt(1)]);
        assert!(arena.allocated_bytes() > 0);

        // Reset
        arena.reset();
        assert_eq!(arena.allocated_bytes(), 0);
        assert_eq!(arena.allocation_count(), 0);
    }

    #[test]
    fn test_thread_local_arena() {
        let tl_arena = ThreadLocalArena::new();
        let arena = tl_arena.get();

        let slice = arena.alloc_slice(&[Value::SmallInt(42)]);
        assert_eq!(slice.len(), 1);
        assert_eq!(slice[0], Value::SmallInt(42));
    }

    // BulkStateStorage tests

    #[test]
    fn test_bulk_storage_basic() {
        let mut storage = BulkStateStorage::new(3, 100);

        // Add a state
        let values = vec![Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)];
        let idx = storage.push_state(&values);

        assert_eq!(idx, 0);
        assert_eq!(storage.len(), 1);
        assert_eq!(storage.vars_per_state(), 3);

        // Retrieve state
        let retrieved = storage.get_state(0);
        assert_eq!(retrieved.len(), 3);
        assert_eq!(retrieved[0], Value::SmallInt(1));
        assert_eq!(retrieved[1], Value::SmallInt(2));
        assert_eq!(retrieved[2], Value::SmallInt(3));
    }

    #[test]
    fn test_bulk_storage_multiple_states() {
        let mut storage = BulkStateStorage::new(2, 100);

        // Add multiple states
        let idx0 = storage.push_state(&[Value::Bool(true), Value::SmallInt(10)]);
        let idx1 = storage.push_state(&[Value::Bool(false), Value::SmallInt(20)]);
        let idx2 = storage.push_state(&[Value::Bool(true), Value::SmallInt(30)]);

        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
        assert_eq!(storage.len(), 3);

        // Verify each state
        assert_eq!(storage.get_state(0)[0], Value::Bool(true));
        assert_eq!(storage.get_state(0)[1], Value::SmallInt(10));
        assert_eq!(storage.get_state(1)[0], Value::Bool(false));
        assert_eq!(storage.get_state(1)[1], Value::SmallInt(20));
        assert_eq!(storage.get_state(2)[0], Value::Bool(true));
        assert_eq!(storage.get_state(2)[1], Value::SmallInt(30));
    }

    #[test]
    fn test_bulk_storage_get_set_value() {
        let mut storage = BulkStateStorage::new(3, 10);
        storage.push_state(&[Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)]);

        // Get individual value
        assert_eq!(storage.get_value(0, 1), &Value::SmallInt(2));

        // Set individual value
        storage.set_value(0, 1, Value::SmallInt(42));
        assert_eq!(storage.get_value(0, 1), &Value::SmallInt(42));

        // Other values unchanged
        assert_eq!(storage.get_value(0, 0), &Value::SmallInt(1));
        assert_eq!(storage.get_value(0, 2), &Value::SmallInt(3));
    }

    #[test]
    fn test_bulk_storage_push_state_iter() {
        let mut storage = BulkStateStorage::new(3, 10);
        let values = vec![Value::Bool(true), Value::SmallInt(42), Value::Bool(false)];
        let idx = storage.push_state_iter(values);

        assert_eq!(idx, 0);
        assert_eq!(storage.get_state(0)[0], Value::Bool(true));
        assert_eq!(storage.get_state(0)[1], Value::SmallInt(42));
        assert_eq!(storage.get_state(0)[2], Value::Bool(false));
    }

    #[test]
    fn test_bulk_storage_memory_efficiency() {
        // Compare memory: 100K states with 5 vars each
        let num_states = 100_000;
        let vars_per_state = 5;

        let mut storage = BulkStateStorage::new(vars_per_state, num_states);

        // Add states
        for i in 0..num_states {
            let values: Vec<Value> = (0..vars_per_state)
                .map(|v| Value::SmallInt((i * vars_per_state + v) as i64))
                .collect();
            storage.push_state(&values);
        }

        assert_eq!(storage.len(), num_states);

        // Memory should be roughly: num_states * vars_per_state * size_of::<Value>()
        // Plus fingerprint storage
        let expected_values_mem = num_states * vars_per_state * std::mem::size_of::<Value>();
        let actual_mem = storage.memory_usage();

        // Actual should be close to expected (within 2x for overhead)
        assert!(
            actual_mem <= expected_values_mem * 2,
            "Memory usage {} should be <= {} (2x expected)",
            actual_mem,
            expected_values_mem * 2
        );
    }

    #[test]
    fn test_bulk_storage_empty() {
        let storage = BulkStateStorage::empty(5);
        assert!(storage.is_empty());
        assert_eq!(storage.len(), 0);
        assert_eq!(storage.vars_per_state(), 5);
    }

    #[test]
    fn test_bulk_storage_from_states() {
        use crate::state::State;
        use crate::var_index::VarRegistry;

        let registry = VarRegistry::from_names(["x", "y"]);

        let states = vec![
            State::from_pairs([("x", Value::SmallInt(1)), ("y", Value::SmallInt(10))]),
            State::from_pairs([("x", Value::SmallInt(2)), ("y", Value::SmallInt(20))]),
            State::from_pairs([("x", Value::SmallInt(3)), ("y", Value::SmallInt(30))]),
        ];

        let storage = BulkStateStorage::from_states(&states, &registry);

        assert_eq!(storage.len(), 3);
        assert_eq!(storage.vars_per_state(), 2);

        // Check values are correct
        assert_eq!(storage.get_state(0)[0], Value::SmallInt(1));
        assert_eq!(storage.get_state(0)[1], Value::SmallInt(10));
        assert_eq!(storage.get_state(1)[0], Value::SmallInt(2));
        assert_eq!(storage.get_state(1)[1], Value::SmallInt(20));
        assert_eq!(storage.get_state(2)[0], Value::SmallInt(3));
        assert_eq!(storage.get_state(2)[1], Value::SmallInt(30));
    }

    #[test]
    fn test_bulk_storage_fingerprints() {
        use crate::state::{ArrayState, State};
        use crate::var_index::VarRegistry;

        let registry = VarRegistry::from_names(["a", "b"]);

        let state = State::from_pairs([("a", Value::SmallInt(42)), ("b", Value::Bool(true))]);

        // Create bulk storage entry
        let mut storage = BulkStateStorage::new(2, 10);
        let idx = storage.push_from_state(&state, &registry);

        // Compute fingerprint
        let bulk_fp = storage.fingerprint(idx, &registry);

        // Compare with ArrayState fingerprint
        let mut array_state = ArrayState::from_state(&state, &registry);
        let array_fp = array_state.fingerprint(&registry);

        assert_eq!(
            bulk_fp, array_fp,
            "BulkStateStorage fingerprint should match ArrayState fingerprint"
        );
    }

    #[test]
    fn test_bulk_storage_to_array_state() {
        use crate::state::State;
        use crate::var_index::VarRegistry;

        let registry = VarRegistry::from_names(["x", "y"]);

        let state = State::from_pairs([("x", Value::SmallInt(100)), ("y", Value::Bool(false))]);

        let mut storage = BulkStateStorage::new(2, 10);
        storage.push_from_state(&state, &registry);

        // Convert back to ArrayState
        let array_state = storage.to_array_state(0);

        assert_eq!(array_state.values()[0], Value::SmallInt(100));
        assert_eq!(array_state.values()[1], Value::Bool(false));
    }

    #[test]
    fn test_bulk_state_handle() {
        let handle1 = BulkStateHandle::new(0);
        assert_eq!(handle1.index, 0);
        assert_eq!(handle1.fingerprint, None);

        let fp = Fingerprint(0x1234567890abcdef);
        let handle2 = BulkStateHandle::with_fingerprint(42, fp);
        assert_eq!(handle2.index, 42);
        assert_eq!(handle2.fingerprint, Some(fp));
    }

    #[test]
    fn test_value_and_state_sizes() {
        use crate::state::ArrayState;
        use std::mem::size_of;

        let value_size = size_of::<Value>();
        let array_state_size = size_of::<ArrayState>();
        let box_slice_size = size_of::<Box<[Value]>>();

        // Print sizes for reference
        println!("Size of Value: {} bytes", value_size);
        println!("Size of ArrayState: {} bytes", array_state_size);
        println!("Size of Box<[Value]>: {} bytes", box_slice_size);

        // Value should be around 72 bytes (as mentioned in code comments)
        assert!(
            value_size <= 80,
            "Value size {} exceeds expected maximum",
            value_size
        );

        // For 5 variables per state at 72 bytes each = 360 bytes data
        // Plus ArrayState overhead (fp_cache, Box pointer)
        let vars_per_state = 5;
        let expected_data_per_state = vars_per_state * value_size;

        // With Box<[Value]>: 16 bytes pointer + data
        // Plus fp_cache Option: ~40 bytes
        let expected_total = expected_data_per_state + box_slice_size + 48; // approximate fp_cache

        println!(
            "For {} vars/state: ~{} bytes data, ~{} bytes total per ArrayState",
            vars_per_state, expected_data_per_state, expected_total
        );

        // BulkStateStorage should be more efficient
        let mut bulk = BulkStateStorage::new(vars_per_state, 1000);
        for i in 0..1000 {
            let values: Vec<Value> = (0..vars_per_state)
                .map(|v| Value::SmallInt((i * vars_per_state + v) as i64))
                .collect();
            bulk.push_state(&values);
        }

        let bulk_per_state = bulk.memory_usage() / 1000;
        println!(
            "BulkStateStorage: {} bytes/state (vs ~{} for ArrayState)",
            bulk_per_state, expected_total
        );

        // Bulk storage should use less memory per state than individual ArrayStates
        // (no Box overhead per state, no fp_cache per state)
        assert!(
            bulk_per_state <= expected_total,
            "Bulk storage {} should be <= ArrayState {}",
            bulk_per_state,
            expected_total
        );
    }
}
