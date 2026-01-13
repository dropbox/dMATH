//! TLA+ Runtime Library
//!
//! This crate provides runtime support for Rust code generated from TLA+ specifications.
//! It includes:
//! - The `StateMachine` trait that generated specs implement
//! - Runtime types that mirror TLA+ values (sets, sequences, functions)
//! - Helper functions for common TLA+ operations
//!
//! # Example
//!
//! ```ignore
//! // Generated code from a TLA+ spec
//! use tla_runtime::prelude::*;
//!
//! #[derive(Clone, Debug, PartialEq, Eq, Hash)]
//! struct CounterState {
//!     count: i64,
//! }
//!
//! struct Counter;
//!
//! impl StateMachine for Counter {
//!     type State = CounterState;
//!
//!     fn init(&self) -> Vec<Self::State> {
//!         vec![CounterState { count: 0 }]
//!     }
//!
//!     fn next(&self, state: &Self::State) -> Vec<Self::State> {
//!         vec![CounterState { count: state.count + 1 }]
//!     }
//! }
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::hash::Hash;

/// The core trait for TLA+ state machines.
///
/// Generated Rust code from TLA+ specifications implements this trait.
/// The trait supports:
/// - Multiple initial states (non-determinism in `Init`)
/// - Multiple successor states (non-determinism in `Next`)
/// - Invariant checking via the `check_invariant` method
pub trait StateMachine {
    /// The state type, generated as a struct from TLA+ variables
    type State: Clone + Eq + Hash + fmt::Debug;

    /// Generate all possible initial states.
    ///
    /// Corresponds to the `Init` predicate in TLA+.
    /// Returns an empty vector if no initial state satisfies Init.
    fn init(&self) -> Vec<Self::State>;

    /// Generate all successor states from the given state.
    ///
    /// Corresponds to the `Next` action in TLA+.
    /// Returns an empty vector if the state is deadlocked.
    fn next(&self, state: &Self::State) -> Vec<Self::State>;

    /// Check if an invariant holds on the given state.
    ///
    /// Returns `None` if no invariants are defined.
    /// Returns `Some(true)` if all invariants hold.
    /// Returns `Some(false)` if any invariant is violated.
    fn check_invariant(&self, _state: &Self::State) -> Option<bool> {
        None
    }

    /// Get the names of defined invariants.
    fn invariant_names(&self) -> Vec<&'static str> {
        vec![]
    }

    /// Check a specific invariant by name.
    fn check_named_invariant(&self, _name: &str, _state: &Self::State) -> Option<bool> {
        None
    }
}

/// A TLA+ set implemented as a sorted set for determinism.
///
/// This type mirrors TLA+ set semantics:
/// - Elements are unique
/// - Order doesn't matter for equality
/// - Iteration order is deterministic (sorted)
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TlaSet<T: Ord + Clone>(BTreeSet<T>);

impl<T: Ord + Clone> TlaSet<T> {
    /// Create an empty set
    pub fn new() -> Self {
        TlaSet(BTreeSet::new())
    }

    /// Check if element is in set
    pub fn contains(&self, elem: &T) -> bool {
        self.0.contains(elem)
    }

    /// Insert an element
    pub fn insert(&mut self, elem: T) {
        self.0.insert(elem);
    }

    /// Number of elements
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterate over elements in sorted order
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    /// Union of two sets
    pub fn union(&self, other: &TlaSet<T>) -> TlaSet<T> {
        TlaSet(self.0.union(&other.0).cloned().collect())
    }

    /// Intersection of two sets
    pub fn intersect(&self, other: &TlaSet<T>) -> TlaSet<T> {
        TlaSet(self.0.intersection(&other.0).cloned().collect())
    }

    /// Set difference
    pub fn difference(&self, other: &TlaSet<T>) -> TlaSet<T> {
        TlaSet(self.0.difference(&other.0).cloned().collect())
    }

    /// Check if this is a subset of other
    pub fn is_subset(&self, other: &TlaSet<T>) -> bool {
        self.0.is_subset(&other.0)
    }
}

impl<T: Ord + Clone> Default for TlaSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord + Clone + fmt::Debug> fmt::Debug for TlaSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        let mut first = true;
        for elem in &self.0 {
            if !first {
                write!(f, ", ")?;
            }
            first = false;
            write!(f, "{:?}", elem)?;
        }
        write!(f, "}}")
    }
}

impl<T: Ord + Clone> IntoIterator for TlaSet<T> {
    type Item = T;
    type IntoIter = std::collections::btree_set::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T: Ord + Clone> IntoIterator for &'a TlaSet<T> {
    type Item = &'a T;
    type IntoIter = std::collections::btree_set::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<T: Ord + Clone> FromIterator<T> for TlaSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        TlaSet(iter.into_iter().collect())
    }
}

/// Create a TlaSet from a list of elements
#[macro_export]
macro_rules! tla_set {
    () => { $crate::TlaSet::new() };
    ($($elem:expr),+ $(,)?) => {
        [$($elem),+].into_iter().collect::<$crate::TlaSet<_>>()
    };
}

/// A TLA+ function implemented as a sorted map for determinism.
///
/// TLA+ functions are total mappings from a domain to values.
/// This is different from Rust functions - it's more like a HashMap.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TlaFunc<K: Ord + Clone, V: Ord + Clone>(BTreeMap<K, V>);

impl<K: Ord + Clone, V: Ord + Clone> TlaFunc<K, V> {
    /// Create an empty function
    pub fn new() -> Self {
        TlaFunc(BTreeMap::new())
    }

    /// Apply the function to an argument
    pub fn apply(&self, key: &K) -> Option<&V> {
        self.0.get(key)
    }

    /// Get the domain of the function
    pub fn domain(&self) -> TlaSet<K> {
        TlaSet::from_iter(self.0.keys().cloned())
    }

    /// Update the function at a single point (EXCEPT) - returns new function
    pub fn except(&self, key: K, value: V) -> Self {
        let mut new_map = self.0.clone();
        new_map.insert(key, value);
        TlaFunc(new_map)
    }

    /// Update the function at a single point in place (for EXCEPT code generation)
    pub fn update(&mut self, key: K, value: V) {
        self.0.insert(key, value);
    }

    /// Iterate over (key, value) pairs
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.0.iter()
    }
}

impl<K: Ord + Clone, V: Ord + Clone> Default for TlaFunc<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord + Clone + fmt::Debug, V: Ord + Clone + fmt::Debug> fmt::Debug for TlaFunc<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        let mut first = true;
        for (k, v) in &self.0 {
            if !first {
                write!(f, ", ")?;
            }
            first = false;
            write!(f, "{:?} |-> {:?}", k, v)?;
        }
        write!(f, "]")
    }
}

impl<K: Ord + Clone, V: Ord + Clone> FromIterator<(K, V)> for TlaFunc<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        TlaFunc(iter.into_iter().collect())
    }
}

/// A TLA+ record implemented as a sorted map from field names to values.
///
/// TLA+ records are like structs: [a |-> 1, b |-> 2]
/// Field access: r.a
/// Field update: [r EXCEPT !.a = 3]
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TlaRecord<V: Ord + Clone>(BTreeMap<String, V>);

impl<V: Ord + Clone> TlaRecord<V> {
    /// Create an empty record
    pub fn new() -> Self {
        TlaRecord(BTreeMap::new())
    }

    /// Create a record from field-value pairs
    pub fn from_fields(fields: impl IntoIterator<Item = (impl Into<String>, V)>) -> Self {
        TlaRecord(fields.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    /// Get a field value
    pub fn get(&self, field: &str) -> Option<&V> {
        self.0.get(field)
    }

    /// Set a field value (in place)
    pub fn set(&mut self, field: impl Into<String>, value: V) {
        self.0.insert(field.into(), value);
    }

    /// Get the set of field names
    pub fn fields(&self) -> TlaSet<String> {
        TlaSet::from_iter(self.0.keys().cloned())
    }

    /// Iterate over (field, value) pairs
    pub fn iter(&self) -> impl Iterator<Item = (&String, &V)> {
        self.0.iter()
    }
}

impl<V: Ord + Clone> Default for TlaRecord<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Ord + Clone + fmt::Debug> fmt::Debug for TlaRecord<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        let mut first = true;
        for (k, v) in &self.0 {
            if !first {
                write!(f, ", ")?;
            }
            first = false;
            write!(f, "{} |-> {:?}", k, v)?;
        }
        write!(f, "]")
    }
}

impl<V: Ord + Clone> FromIterator<(String, V)> for TlaRecord<V> {
    fn from_iter<I: IntoIterator<Item = (String, V)>>(iter: I) -> Self {
        TlaRecord(iter.into_iter().collect())
    }
}

/// Create a TlaRecord from field-value pairs
#[macro_export]
macro_rules! tla_record {
    () => { $crate::TlaRecord::new() };
    ($($field:ident => $value:expr),+ $(,)?) => {
        $crate::TlaRecord::from_fields([$(
            (stringify!($field), $value)
        ),+])
    };
}

/// Create a range set {a..b} inclusive
pub fn range_set(a: i64, b: i64) -> TlaSet<i64> {
    if a > b {
        TlaSet::new()
    } else {
        (a..=b).collect()
    }
}

/// The BOOLEAN set {FALSE, TRUE}
pub fn boolean_set() -> TlaSet<bool> {
    tla_set![false, true]
}

/// Compute the powerset (SUBSET S)
pub fn powerset<T: Ord + Clone>(set: &TlaSet<T>) -> TlaSet<TlaSet<T>> {
    let elements: Vec<_> = set.iter().cloned().collect();
    let n = elements.len();

    if n > 20 {
        panic!("SUBSET of set with {} elements would be too large", n);
    }

    let mut result = TlaSet::new();
    for mask in 0..(1u64 << n) {
        let mut subset = TlaSet::new();
        for (i, elem) in elements.iter().enumerate() {
            if mask & (1 << i) != 0 {
                subset.insert(elem.clone());
            }
        }
        result.insert(subset);
    }
    result
}

/// Compute the Cartesian product of two sets
pub fn cartesian_product<T: Ord + Clone, U: Ord + Clone>(
    a: &TlaSet<T>,
    b: &TlaSet<U>,
) -> TlaSet<(T, U)> {
    let mut result = TlaSet::new();
    for x in a {
        for y in b {
            result.insert((x.clone(), y.clone()));
        }
    }
    result
}

// ============================================================================
// Model Checker Helper
// ============================================================================

/// Result of a model checking run
#[derive(Debug)]
pub struct ModelCheckResult<S> {
    /// Number of states explored
    pub states_explored: usize,
    /// Number of distinct states found
    pub distinct_states: usize,
    /// If an invariant was violated, the state that violated it
    pub violation: Option<InvariantViolation<S>>,
    /// If a deadlock was found (state with no successors), the deadlocked state
    pub deadlock: Option<S>,
}

/// Information about an invariant violation
#[derive(Debug)]
pub struct InvariantViolation<S> {
    /// The state that violated the invariant
    pub state: S,
    /// Name of the violated invariant (if known)
    pub invariant_name: Option<String>,
}

impl<S> ModelCheckResult<S> {
    /// Check if the model check succeeded (no violations or deadlocks)
    pub fn is_ok(&self) -> bool {
        self.violation.is_none() && self.deadlock.is_none()
    }
}

/// A simple BFS model checker that uses the StateMachine trait
///
/// This can be used to verify that generated code behaves correctly:
/// - Explores all reachable states from Init via Next
/// - Checks invariants on each state
/// - Detects deadlocks (states with no successors)
///
/// # Example
/// ```ignore
/// use tla_runtime::prelude::*;
/// use tla_runtime::model_check;
///
/// let spec = MyGeneratedSpec;
/// let result = model_check(&spec, 1000);
/// assert!(result.is_ok());
/// ```
pub fn model_check<M: StateMachine>(machine: &M, max_states: usize) -> ModelCheckResult<M::State> {
    use std::collections::{HashSet, VecDeque};

    let mut seen: HashSet<M::State> = HashSet::new();
    let mut queue: VecDeque<M::State> = VecDeque::new();
    let mut states_explored = 0;

    // Initialize with all initial states
    for state in machine.init() {
        if seen.insert(state.clone()) {
            queue.push_back(state);
        }
    }

    // BFS exploration
    while let Some(state) = queue.pop_front() {
        states_explored += 1;

        if states_explored > max_states {
            break;
        }

        // Check invariants
        if let Some(false) = machine.check_invariant(&state) {
            return ModelCheckResult {
                states_explored,
                distinct_states: seen.len(),
                violation: Some(InvariantViolation {
                    state,
                    invariant_name: None,
                }),
                deadlock: None,
            };
        }

        // Generate successors
        let successors = machine.next(&state);

        // Check for deadlock
        if successors.is_empty() {
            return ModelCheckResult {
                states_explored,
                distinct_states: seen.len(),
                violation: None,
                deadlock: Some(state),
            };
        }

        // Add new states to queue
        for succ in successors {
            if seen.insert(succ.clone()) {
                queue.push_back(succ);
            }
        }
    }

    ModelCheckResult {
        states_explored,
        distinct_states: seen.len(),
        violation: None,
        deadlock: None,
    }
}

/// Model check with a custom invariant function
///
/// This allows checking invariants that aren't part of the spec's check_invariant
pub fn model_check_with_invariant<M, F>(
    machine: &M,
    max_states: usize,
    invariant: F,
) -> ModelCheckResult<M::State>
where
    M: StateMachine,
    F: Fn(&M::State) -> bool,
{
    use std::collections::{HashSet, VecDeque};

    let mut seen: HashSet<M::State> = HashSet::new();
    let mut queue: VecDeque<M::State> = VecDeque::new();
    let mut states_explored = 0;

    // Initialize with all initial states
    for state in machine.init() {
        if seen.insert(state.clone()) {
            queue.push_back(state);
        }
    }

    // BFS exploration
    while let Some(state) = queue.pop_front() {
        states_explored += 1;

        if states_explored > max_states {
            break;
        }

        // Check custom invariant
        if !invariant(&state) {
            return ModelCheckResult {
                states_explored,
                distinct_states: seen.len(),
                violation: Some(InvariantViolation {
                    state,
                    invariant_name: Some("custom".to_string()),
                }),
                deadlock: None,
            };
        }

        // Generate successors
        let successors = machine.next(&state);

        // Check for deadlock (don't report if state is same as before)
        if successors.is_empty() {
            return ModelCheckResult {
                states_explored,
                distinct_states: seen.len(),
                violation: None,
                deadlock: Some(state),
            };
        }

        // Add new states to queue
        for succ in successors {
            if seen.insert(succ.clone()) {
                queue.push_back(succ);
            }
        }
    }

    ModelCheckResult {
        states_explored,
        distinct_states: seen.len(),
        violation: None,
        deadlock: None,
    }
}

/// Collect all reachable states (up to a limit) without checking invariants
///
/// Useful for testing that a spec generates the expected state space
pub fn collect_states<M: StateMachine>(machine: &M, max_states: usize) -> Vec<M::State> {
    use std::collections::{HashSet, VecDeque};

    let mut seen: HashSet<M::State> = HashSet::new();
    let mut queue: VecDeque<M::State> = VecDeque::new();
    let mut result: Vec<M::State> = Vec::new();

    // Initialize with all initial states
    for state in machine.init() {
        if seen.insert(state.clone()) {
            queue.push_back(state);
        }
    }

    // BFS exploration
    while let Some(state) = queue.pop_front() {
        if result.len() >= max_states {
            break;
        }

        result.push(state.clone());

        // Generate successors
        for succ in machine.next(&state) {
            if seen.insert(succ.clone()) {
                queue.push_back(succ);
            }
        }
    }

    result
}

/// Prelude module - import everything commonly needed
pub mod prelude {
    pub use super::{
        boolean_set, cartesian_product, collect_states, model_check, model_check_with_invariant,
        powerset, range_set, ModelCheckResult, StateMachine, TlaFunc, TlaRecord, TlaSet,
    };
    pub use crate::{tla_record, tla_set};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tla_set_basic() {
        let s = tla_set![1, 2, 3];
        assert_eq!(s.len(), 3);
        assert!(s.contains(&2));
        assert!(!s.contains(&4));
    }

    #[test]
    fn test_tla_set_union() {
        let a = tla_set![1, 2];
        let b = tla_set![2, 3];
        let c = a.union(&b);
        assert_eq!(c.len(), 3);
        assert!(c.contains(&1));
        assert!(c.contains(&2));
        assert!(c.contains(&3));
    }

    #[test]
    fn test_tla_set_intersect() {
        let a = tla_set![1, 2, 3];
        let b = tla_set![2, 3, 4];
        let c = a.intersect(&b);
        assert_eq!(c.len(), 2);
        assert!(c.contains(&2));
        assert!(c.contains(&3));
    }

    #[test]
    fn test_tla_func_basic() {
        let f: TlaFunc<i32, &str> = [(1, "a"), (2, "b")].into_iter().collect();
        assert_eq!(f.apply(&1), Some(&"a"));
        assert_eq!(f.apply(&2), Some(&"b"));
        assert_eq!(f.apply(&3), None);
    }

    #[test]
    fn test_tla_func_except() {
        let f: TlaFunc<i32, &str> = [(1, "a"), (2, "b")].into_iter().collect();
        let g = f.except(2, "c");
        assert_eq!(f.apply(&2), Some(&"b")); // Original unchanged
        assert_eq!(g.apply(&2), Some(&"c")); // New has update
    }

    #[test]
    fn test_range_set() {
        let r = range_set(1, 3);
        assert_eq!(r.len(), 3);
        assert!(r.contains(&1));
        assert!(r.contains(&2));
        assert!(r.contains(&3));
    }

    #[test]
    fn test_range_set_empty() {
        let r = range_set(5, 3);
        assert!(r.is_empty());
    }

    #[test]
    fn test_powerset() {
        let s = tla_set![1, 2];
        let ps = powerset(&s);
        assert_eq!(ps.len(), 4); // 2^2 = 4

        assert!(ps.contains(&tla_set![]));
        assert!(ps.contains(&tla_set![1]));
        assert!(ps.contains(&tla_set![2]));
        assert!(ps.contains(&tla_set![1, 2]));
    }

    #[test]
    fn test_cartesian_product() {
        let a = tla_set![1, 2];
        let b = tla_set!["x", "y"];
        let cp = cartesian_product(&a, &b);
        assert_eq!(cp.len(), 4);
        assert!(cp.contains(&(1, "x")));
        assert!(cp.contains(&(2, "y")));
    }

    // Test the StateMachine trait with a simple counter
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct CounterState {
        count: i64,
    }

    struct Counter {
        max: i64,
    }

    impl StateMachine for Counter {
        type State = CounterState;

        fn init(&self) -> Vec<Self::State> {
            vec![CounterState { count: 0 }]
        }

        fn next(&self, state: &Self::State) -> Vec<Self::State> {
            if state.count < self.max {
                vec![CounterState {
                    count: state.count + 1,
                }]
            } else {
                vec![] // Deadlock at max
            }
        }

        fn check_invariant(&self, state: &Self::State) -> Option<bool> {
            Some(state.count <= self.max)
        }
    }

    #[test]
    fn test_state_machine_counter() {
        let machine = Counter { max: 3 };

        let init_states = machine.init();
        assert_eq!(init_states.len(), 1);
        assert_eq!(init_states[0].count, 0);

        let next_states = machine.next(&init_states[0]);
        assert_eq!(next_states.len(), 1);
        assert_eq!(next_states[0].count, 1);

        // Check invariant
        assert_eq!(
            machine.check_invariant(&CounterState { count: 2 }),
            Some(true)
        );
        assert_eq!(
            machine.check_invariant(&CounterState { count: 5 }),
            Some(false)
        );
    }

    #[test]
    fn test_tla_func_update() {
        let mut f: TlaFunc<i32, &str> = [(1, "a"), (2, "b")].into_iter().collect();
        f.update(2, "c");
        assert_eq!(f.apply(&2), Some(&"c"));
        f.update(3, "d");
        assert_eq!(f.apply(&3), Some(&"d"));
    }

    #[test]
    fn test_tla_record_basic() {
        let r = TlaRecord::from_fields([("a", 1), ("b", 2)]);
        assert_eq!(r.get("a"), Some(&1));
        assert_eq!(r.get("b"), Some(&2));
        assert_eq!(r.get("c"), None);
    }

    #[test]
    fn test_tla_record_set() {
        let mut r = TlaRecord::from_fields([("x", 10), ("y", 20)]);
        r.set("x", 30);
        assert_eq!(r.get("x"), Some(&30));
        r.set("z", 40);
        assert_eq!(r.get("z"), Some(&40));
    }

    #[test]
    fn test_tla_record_macro() {
        let r = tla_record![x => 1, y => 2];
        assert_eq!(r.get("x"), Some(&1));
        assert_eq!(r.get("y"), Some(&2));
    }

    #[test]
    fn test_tla_record_fields() {
        let r = TlaRecord::from_fields([("a", 1), ("b", 2)]);
        let fields = r.fields();
        assert_eq!(fields.len(), 2);
        assert!(fields.contains(&"a".to_string()));
        assert!(fields.contains(&"b".to_string()));
    }

    // Tests for model checker helper
    #[test]
    fn test_model_check_basic() {
        let machine = Counter { max: 5 };
        let result = model_check(&machine, 100);

        // Counter deadlocks at max
        assert!(result.violation.is_none());
        assert!(result.deadlock.is_some());
        assert_eq!(result.deadlock.as_ref().unwrap().count, 5);
        assert_eq!(result.distinct_states, 6); // 0, 1, 2, 3, 4, 5
    }

    #[test]
    fn test_model_check_invariant_violation() {
        // A machine that violates its invariant
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        struct BadState {
            val: i64,
        }

        struct BadMachine;
        impl StateMachine for BadMachine {
            type State = BadState;
            fn init(&self) -> Vec<Self::State> {
                vec![BadState { val: 0 }]
            }
            fn next(&self, state: &Self::State) -> Vec<Self::State> {
                vec![BadState { val: state.val + 1 }]
            }
            fn check_invariant(&self, state: &Self::State) -> Option<bool> {
                Some(state.val < 3) // Violated when val >= 3
            }
        }

        let result = model_check(&BadMachine, 100);
        assert!(result.violation.is_some());
        assert_eq!(result.violation.as_ref().unwrap().state.val, 3);
    }

    #[test]
    fn test_model_check_with_custom_invariant() {
        let machine = Counter { max: 10 };
        let result = model_check_with_invariant(&machine, 100, |s| s.count <= 5);

        // Custom invariant should be violated at count = 6
        assert!(result.violation.is_some());
        assert_eq!(result.violation.as_ref().unwrap().state.count, 6);
    }

    #[test]
    fn test_collect_states() {
        let machine = Counter { max: 5 };
        let states = collect_states(&machine, 100);

        // Should collect all 6 states
        assert_eq!(states.len(), 6);

        // Verify states are 0 through 5
        let counts: Vec<_> = states.iter().map(|s| s.count).collect();
        for i in 0..=5 {
            assert!(counts.contains(&i), "Missing state with count={}", i);
        }
    }

    #[test]
    fn test_collect_states_limited() {
        let machine = Counter { max: 100 };
        let states = collect_states(&machine, 10);

        // Should be limited to 10 states
        assert_eq!(states.len(), 10);
    }

    #[test]
    fn test_monitored_state_machine() {
        let machine = Counter { max: 5 };
        let mut monitored = MonitoredStateMachine::new(machine);

        // Initial state should be 0
        assert_eq!(monitored.state().count, 0);

        // Apply valid transitions
        assert!(monitored.step().is_ok());
        assert_eq!(monitored.state().count, 1);

        assert!(monitored.step().is_ok());
        assert_eq!(monitored.state().count, 2);
    }

    #[test]
    fn test_monitored_detects_deadlock() {
        let machine = Counter { max: 2 };
        let mut monitored = MonitoredStateMachine::new(machine);

        assert!(monitored.step().is_ok()); // 0 -> 1
        assert!(monitored.step().is_ok()); // 1 -> 2
        assert!(matches!(
            monitored.step(),
            Err(SpecViolation::Deadlock { .. })
        ));
    }

    #[test]
    fn test_monitored_detects_invariant_violation() {
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        struct BadState {
            val: i64,
        }

        struct BadMachine;
        impl StateMachine for BadMachine {
            type State = BadState;
            fn init(&self) -> Vec<Self::State> {
                vec![BadState { val: 0 }]
            }
            fn next(&self, s: &Self::State) -> Vec<Self::State> {
                vec![BadState { val: s.val + 1 }]
            }
            fn check_invariant(&self, s: &Self::State) -> Option<bool> {
                Some(s.val < 3)
            }
        }

        let mut monitored = MonitoredStateMachine::new(BadMachine);
        assert!(monitored.step().is_ok()); // 0 -> 1
        assert!(monitored.step().is_ok()); // 1 -> 2
        assert!(matches!(
            monitored.step(),
            Err(SpecViolation::InvariantViolated { .. })
        )); // 2 -> 3
    }
}

// =============================================================================
// RUNTIME MONITORING (Layer 4 of Verification Stack)
// =============================================================================

/// A wrapper that monitors a state machine at runtime, checking invariants
/// on every transition. Use this in production to catch specification violations.
///
/// # Example
///
/// ```ignore
/// let machine = MyGeneratedSpec::new();
/// let mut monitored = MonitoredStateMachine::new(machine);
///
/// // In your application loop
/// loop {
///     match monitored.apply(&action) {
///         Ok(()) => { /* action succeeded */ }
///         Err(SpecViolation::InvariantViolated { .. }) => {
///             // Log and alert! Specification was violated
///             panic!("Invariant violated in production!");
///         }
///         Err(SpecViolation::ActionNotEnabled { .. }) => {
///             // Action wasn't enabled - may be expected
///         }
///         Err(SpecViolation::Deadlock { .. }) => {
///             // No actions enabled - system is stuck
///         }
///     }
/// }
/// ```
pub struct MonitoredStateMachine<M: StateMachine> {
    machine: M,
    state: M::State,
    config: MonitorConfig,
    stats: MonitorStats,
}

/// Configuration for runtime monitoring
#[derive(Clone, Debug)]
pub struct MonitorConfig {
    /// Whether to check invariants on every transition (default: true)
    pub check_invariants: bool,
    /// Whether to log state transitions (default: false)
    pub log_transitions: bool,
    /// Maximum number of violations before panicking (default: None = unlimited)
    pub max_violations: Option<usize>,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            check_invariants: true,
            log_transitions: false,
            max_violations: None,
        }
    }
}

/// Statistics collected during monitored execution
#[derive(Clone, Debug, Default)]
pub struct MonitorStats {
    /// Total number of transitions
    pub transitions: u64,
    /// Number of invariant violations detected
    pub violations: u64,
    /// Number of deadlocks detected
    pub deadlocks: u64,
    /// Number of disabled actions attempted
    pub disabled_actions: u64,
}

/// Errors that can occur during monitored execution
#[derive(Debug, Clone)]
pub enum SpecViolation<S> {
    /// An invariant was violated after a transition
    InvariantViolated {
        /// The state that violated the invariant
        state: S,
        /// Index of the action that caused the violation
        action_index: usize,
        /// Names of violated invariants (if available)
        violated_invariants: Vec<String>,
    },
    /// No actions are enabled (deadlock)
    Deadlock {
        /// The deadlocked state
        state: S,
    },
    /// The requested action was not enabled
    ActionNotEnabled {
        /// Index of the action that was not enabled
        action_index: usize,
    },
}

impl<M: StateMachine> MonitoredStateMachine<M> {
    /// Create a new monitored state machine, starting from an initial state
    pub fn new(machine: M) -> Self {
        let init_states = machine.init();
        let state = init_states
            .into_iter()
            .next()
            .expect("StateMachine must have at least one initial state");
        Self {
            machine,
            state,
            config: MonitorConfig::default(),
            stats: MonitorStats::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(machine: M, config: MonitorConfig) -> Self {
        let init_states = machine.init();
        let state = init_states
            .into_iter()
            .next()
            .expect("StateMachine must have at least one initial state");
        Self {
            machine,
            state,
            config,
            stats: MonitorStats::default(),
        }
    }

    /// Get the current state
    pub fn state(&self) -> &M::State {
        &self.state
    }

    /// Get statistics
    pub fn stats(&self) -> &MonitorStats {
        &self.stats
    }

    /// Take a step (non-deterministically choosing the first enabled action)
    pub fn step(&mut self) -> Result<(), SpecViolation<M::State>> {
        let successors = self.machine.next(&self.state);

        if successors.is_empty() {
            self.stats.deadlocks += 1;
            return Err(SpecViolation::Deadlock {
                state: self.state.clone(),
            });
        }

        // Take the first successor
        let next_state = successors.into_iter().next().unwrap();

        // Check invariant if enabled
        if self.config.check_invariants {
            if let Some(false) = self.machine.check_invariant(&next_state) {
                self.stats.violations += 1;
                return Err(SpecViolation::InvariantViolated {
                    state: next_state,
                    action_index: 0,
                    violated_invariants: vec![],
                });
            }
        }

        self.state = next_state;
        self.stats.transitions += 1;
        Ok(())
    }

    /// Apply a specific action by index
    pub fn apply(&mut self, action_index: usize) -> Result<(), SpecViolation<M::State>> {
        let successors = self.machine.next(&self.state);

        if successors.is_empty() {
            self.stats.deadlocks += 1;
            return Err(SpecViolation::Deadlock {
                state: self.state.clone(),
            });
        }

        if action_index >= successors.len() {
            self.stats.disabled_actions += 1;
            return Err(SpecViolation::ActionNotEnabled { action_index });
        }

        let next_state = successors.into_iter().nth(action_index).unwrap();

        // Check invariant if enabled
        if self.config.check_invariants {
            if let Some(false) = self.machine.check_invariant(&next_state) {
                self.stats.violations += 1;
                return Err(SpecViolation::InvariantViolated {
                    state: next_state,
                    action_index,
                    violated_invariants: vec![],
                });
            }
        }

        self.state = next_state;
        self.stats.transitions += 1;
        Ok(())
    }

    /// Reset to an initial state
    pub fn reset(&mut self) {
        let init_states = self.machine.init();
        self.state = init_states
            .into_iter()
            .next()
            .expect("StateMachine must have at least one initial state");
    }

    /// Get the number of enabled actions in the current state
    pub fn enabled_actions(&self) -> usize {
        self.machine.next(&self.state).len()
    }
}
