//! TLA+ State representation
//!
//! A state in TLA+ is a mapping from variable names to values. States are:
//! - Immutable: Transitions create new states rather than mutating existing ones
//! - Hashable: For efficient state-space exploration (detecting duplicates)
//! - Comparable: For deterministic ordering
//!
//! # Fingerprinting
//!
//! States are identified by a 64-bit fingerprint computed via FNV-1a hash.
//! This matches TLC's approach (though TLC uses 64-bit fingerprints too).
//! Collision probability is acceptable for model checking purposes.

use crate::value::FuncValue;
use crate::var_index::{VarIndex, VarRegistry};
use crate::Value;
use im::OrdMap;
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};

// Profiling counters for symmetry fingerprinting
static SYMMETRY_FP_CALLS: AtomicU64 = AtomicU64::new(0);
static SYMMETRY_FP_US: AtomicU64 = AtomicU64::new(0);

/// Print and reset symmetry fingerprinting statistics
pub fn print_symmetry_stats() {
    let calls = SYMMETRY_FP_CALLS.swap(0, AtomicOrdering::Relaxed);
    let us = SYMMETRY_FP_US.swap(0, AtomicOrdering::Relaxed);
    if calls > 0 {
        eprintln!(
            "=== Symmetry Fingerprint Profile ===\n  Calls: {}\n  Time: {:.3}s\n  Avg: {:.1}µs/call",
            calls,
            us as f64 / 1_000_000.0,
            us as f64 / calls as f64
        );
    }
}

/// A 64-bit state fingerprint for fast state comparison
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fingerprint(pub u64);

impl fmt::Debug for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FP({:016x})", self.0)
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// A TLA+ state: a mapping from variable names to values
///
/// States are the nodes in the state graph explored by the model checker.
/// Each state represents a possible configuration of the system.
pub struct State {
    /// Variable bindings
    vars: OrdMap<Arc<str>, Value>,
    /// Cached fingerprint (computed at construction time)
    fingerprint: Fingerprint,
    /// Cached canonical (symmetry-reduced) fingerprint, populated lazily
    /// Using OnceLock for thread-safe lazy initialization
    canonical_fingerprint: OnceLock<Fingerprint>,
}

impl Clone for State {
    fn clone(&self) -> Self {
        State {
            vars: self.vars.clone(),
            fingerprint: self.fingerprint,
            // Clone the cached canonical fingerprint if available
            canonical_fingerprint: self
                .canonical_fingerprint
                .get()
                .map(|fp| {
                    let lock = OnceLock::new();
                    let _ = lock.set(*fp);
                    lock
                })
                .unwrap_or_default(),
        }
    }
}

impl State {
    /// Create an empty state
    pub fn new() -> Self {
        let vars = OrdMap::new();
        let fingerprint = compute_fingerprint(&vars);
        State {
            vars,
            fingerprint,
            canonical_fingerprint: OnceLock::new(),
        }
    }

    /// Create a state from a map of variables
    pub fn from_vars(vars: OrdMap<Arc<str>, Value>) -> Self {
        let fingerprint = compute_fingerprint(&vars);
        State {
            vars,
            fingerprint,
            canonical_fingerprint: OnceLock::new(),
        }
    }

    /// Create a state from an iterator of (name, value) pairs
    pub fn from_pairs(iter: impl IntoIterator<Item = (impl Into<Arc<str>>, Value)>) -> Self {
        let vars: OrdMap<Arc<str>, Value> = iter.into_iter().map(|(k, v)| (k.into(), v)).collect();
        State::from_vars(vars)
    }

    /// Create a state from an array of values indexed by variable index
    ///
    /// This is faster than `from_pairs` when you have pre-sorted values
    /// matching the VarRegistry order. Uses ArrayState internally for
    /// efficient fingerprint computation.
    ///
    /// # Arguments
    /// * `values` - Values in VarRegistry index order
    /// * `registry` - The variable registry mapping indices to names
    pub fn from_indexed(values: &[Value], registry: &VarRegistry) -> Self {
        // Build OrdMap from values in index order
        let vars: OrdMap<Arc<str>, Value> = registry
            .iter()
            .map(|(idx, name)| (Arc::clone(name), values[idx.as_usize()].clone()))
            .collect();

        // Compute fingerprint using the fast array-based method
        let fingerprint = compute_fingerprint_from_array(values, registry);

        State {
            vars,
            fingerprint,
            canonical_fingerprint: OnceLock::new(),
        }
    }

    /// Create a state from an ArrayState
    pub fn from_array_state(array_state: &mut ArrayState, registry: &VarRegistry) -> Self {
        let vars: OrdMap<Arc<str>, Value> = registry
            .iter()
            .map(|(idx, name)| {
                (
                    Arc::clone(name),
                    array_state.values()[idx.as_usize()].clone(),
                )
            })
            .collect();

        // Get or compute fingerprint
        let fingerprint = array_state.fingerprint(registry);

        State {
            vars,
            fingerprint,
            canonical_fingerprint: OnceLock::new(),
        }
    }

    /// Get a variable's value
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.vars.get(name)
    }

    /// Set a variable's value, returning a new state
    pub fn with_var(&self, name: impl Into<Arc<str>>, value: Value) -> State {
        let mut new_vars = self.vars.clone();
        new_vars.insert(name.into(), value);
        State::from_vars(new_vars)
    }

    /// Update multiple variables at once
    pub fn with_vars(
        &self,
        updates: impl IntoIterator<Item = (impl Into<Arc<str>>, Value)>,
    ) -> State {
        let mut new_vars = self.vars.clone();
        for (name, value) in updates {
            new_vars.insert(name.into(), value);
        }
        State::from_vars(new_vars)
    }

    /// Get all variable names
    pub fn var_names(&self) -> impl Iterator<Item = &Arc<str>> {
        self.vars.keys()
    }

    /// Get all variables as (name, value) pairs
    pub fn vars(&self) -> impl Iterator<Item = (&Arc<str>, &Value)> {
        self.vars.iter()
    }

    /// Number of variables
    pub fn len(&self) -> usize {
        self.vars.len()
    }

    /// Check if state has no variables
    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }

    /// Convert state variables to an array indexed by VarRegistry indices.
    ///
    /// This enables efficient array-based state variable binding via `EvalCtx::bind_state_array`,
    /// avoiding HashMap lookups during evaluation. Used by liveness checking.
    ///
    /// Returns a boxed slice of values in VarRegistry index order.
    pub fn to_values(&self, registry: &VarRegistry) -> Box<[Value]> {
        let mut values = Vec::with_capacity(registry.len());
        for (_idx, name) in registry.iter() {
            values.push(
                self.vars
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| Value::Bool(false)),
            );
        }
        values.into_boxed_slice()
    }

    /// Compute the fingerprint of this state
    ///
    /// Uses FNV-1a hash for fast fingerprinting. The fingerprint is
    /// deterministic across runs (depends only on var names and values).
    pub fn fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }

    /// Compute the canonical fingerprint under symmetry permutations
    ///
    /// For symmetry reduction, we need to identify symmetric states as equivalent.
    /// This is done by finding the lexicographically smallest permuted state
    /// and returning its fingerprint (TLC-compatible algorithm).
    ///
    /// IMPORTANT: We find lexmin(S, P1(S), P2(S), ...) then fingerprint it.
    /// NOT min(fp(S), fp(P1(S)), ...) - fingerprint order != lexicographic order!
    ///
    /// If `perms` is empty, returns the regular fingerprint.
    /// Results are cached for efficiency when called multiple times on the same state.
    pub fn fingerprint_with_symmetry(&self, perms: &[FuncValue]) -> Fingerprint {
        if perms.is_empty() {
            return self.fingerprint();
        }

        // Return cached value if available
        if let Some(&cached) = self.canonical_fingerprint.get() {
            return cached;
        }

        let start = std::time::Instant::now();
        SYMMETRY_FP_CALLS.fetch_add(1, AtomicOrdering::Relaxed);

        // TLC algorithm: find lexicographically minimum STATE, then fingerprint it
        let mut min_vars = &self.vars;
        #[allow(unused_assignments)] // Initial None may be overwritten; warning is spurious
        let mut permuted_storage: Option<OrdMap<Arc<str>, Value>> = None;

        for perm in perms {
            let permuted: OrdMap<Arc<str>, Value> = self
                .vars
                .iter()
                .map(|(name, value)| (name.clone(), value.permute(perm)))
                .collect();

            // Compare lexicographically: variable by variable in sorted order
            if compare_vars_lexicographic(&permuted, min_vars) == Ordering::Less {
                permuted_storage = Some(permuted);
                min_vars = permuted_storage.as_ref().unwrap();
            }
        }

        // Fingerprint the lexicographically minimum state
        let canonical_fp = if std::ptr::eq(min_vars, &self.vars) {
            self.fingerprint()
        } else {
            compute_fingerprint(min_vars)
        };

        // Cache the result (ignore if another thread beat us to it)
        let _ = self.canonical_fingerprint.set(canonical_fp);

        SYMMETRY_FP_US.fetch_add(start.elapsed().as_micros() as u64, AtomicOrdering::Relaxed);
        canonical_fp
    }

    /// Apply a permutation to all values in this state
    ///
    /// Returns a new state with all model values permuted according to the given
    /// permutation function. Used for symmetry reduction.
    pub fn permute(&self, perm: &FuncValue) -> State {
        let permuted_vars: OrdMap<Arc<str>, Value> = self
            .vars
            .iter()
            .map(|(name, value)| (name.clone(), value.permute(perm)))
            .collect();
        State::from_vars(permuted_vars)
    }
}

/// Compute the fingerprint of a single value
///
/// This is useful for TLCExt!TLCFP which returns the fingerprint of a value.
/// Uses TLC-compatible FP64 polynomial rolling hash for deterministic fingerprinting.
///
/// For FuncValue, the fingerprint is cached to avoid re-computation
/// when the same function is fingerprinted multiple times.
pub fn value_fingerprint(value: &Value) -> u64 {
    use crate::fingerprint::FP64_INIT;

    // Special case for Set: use cached fingerprint if available.
    if let Value::Set(set) = value {
        if let Some(cached) = set.get_cached_value_fingerprint() {
            return cached;
        }
        let fp = value.fingerprint_extend(FP64_INIT);
        set.cache_value_fingerprint(fp);
        return fp;
    }

    // Special case for Func: use cached fingerprint if available
    if let Value::Func(func) = value {
        if let Some(cached) = func.get_cached_fingerprint() {
            return cached;
        }
        let fp = value.fingerprint_extend(FP64_INIT);
        func.cache_fingerprint(fp);
        return fp;
    }

    // Special case for IntFunc: use cached fingerprint if available
    if let Value::IntFunc(func) = value {
        if let Some(cached) = func.get_cached_fingerprint() {
            return cached;
        }
        let fp = value.fingerprint_extend(FP64_INIT);
        func.cache_fingerprint(fp);
        return fp;
    }

    // Special case for Record: use cached fingerprint if available
    if let Value::Record(rec) = value {
        if let Some(cached) = rec.get_cached_fingerprint() {
            return cached;
        }
        let fp = value.fingerprint_extend(FP64_INIT);
        rec.cache_fingerprint(fp);
        return fp;
    }

    // Special case for Seq: use cached fingerprint if available
    if let Value::Seq(seq) = value {
        if let Some(cached) = seq.get_cached_fingerprint() {
            return cached;
        }
        let fp = value.fingerprint_extend(FP64_INIT);
        seq.cache_fingerprint(fp);
        return fp;
    }

    value.fingerprint_extend(FP64_INIT)
}

/// Hash a small integer directly into the running FNV-1a hash
/// This is an optimization to avoid creating temporary Value::SmallInt wrappers
#[inline(always)]
fn hash_smallint(mut hash: u64, n: i64) -> u64 {
    const FNV_PRIME: u64 = 0x100000001b3;

    // Type tag for integers (same as SmallInt/Int)
    hash ^= 1u64;
    hash = hash.wrapping_mul(FNV_PRIME);

    // Hash using the same two's-complement byte representation as BigInt
    let bytes = n.to_le_bytes();
    let mut len = bytes.len();
    if n >= 0 {
        while len > 1 && bytes[len - 1] == 0x00 && (bytes[len - 2] & 0x80) == 0 {
            len -= 1;
        }
    } else {
        while len > 1 && bytes[len - 1] == 0xFF && (bytes[len - 2] & 0x80) != 0 {
            len -= 1;
        }
    }
    for &byte in &bytes[..len] {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Hash a string directly into the running FNV-1a hash
/// This is an optimization to avoid creating temporary Value::String wrappers
#[inline(always)]
fn hash_string(mut hash: u64, s: &str) -> u64 {
    const FNV_PRIME: u64 = 0x100000001b3;

    // Type tag for strings (tag 2)
    hash ^= 2u64;
    hash = hash.wrapping_mul(FNV_PRIME);

    // Hash the string bytes
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Hash a value into the running FNV-1a hash
fn hash_value(mut hash: u64, value: &Value) -> u64 {
    const FNV_PRIME: u64 = 0x100000001b3;

    // Type tag (Interval, Subset, FuncSet, RecordSet, TupleSet use same tag as Set for consistency)
    // Tuple, Seq, Record, Func all use the same tag (4) since they are all functions in TLA+
    // (tuples/seqs have domain 1..n, records have string domain)
    let tag = match value {
        Value::Bool(_) => 0u8,
        Value::SmallInt(_) | Value::Int(_) => 1,
        Value::String(_) => 2,
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
        | Value::SeqSet(_) => 3,
        // All function-like types use same tag for equivalence
        Value::Func(_) | Value::IntFunc(_) | Value::Seq(_) | Value::Record(_) | Value::Tuple(_) => {
            4
        }
        Value::ModelValue(_) => 8,
        Value::Closure(_) => 9,
        Value::LazyFunc(_) => 10,
    };
    hash ^= tag as u64;
    hash = hash.wrapping_mul(FNV_PRIME);

    match value {
        Value::Bool(b) => {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        Value::SmallInt(n) => {
            // Hash SmallInt using the same two's-complement byte representation
            // as BigInt::to_signed_bytes_le(), but without allocating a BigInt/Vec.
            let bytes = n.to_le_bytes();
            let mut len = bytes.len();
            if *n >= 0 {
                while len > 1 && bytes[len - 1] == 0x00 && (bytes[len - 2] & 0x80) == 0 {
                    len -= 1;
                }
            } else {
                while len > 1 && bytes[len - 1] == 0xFF && (bytes[len - 2] & 0x80) != 0 {
                    len -= 1;
                }
            }
            for &byte in &bytes[..len] {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
        Value::Int(n) => {
            // Hash the bytes of the integer
            for byte in n.to_signed_bytes_le() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
        Value::String(s) | Value::ModelValue(s) => {
            for byte in s.bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
        Value::Set(set) => {
            // Hash set cardinality first (helps with collision resistance)
            hash ^= set.len() as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
            // Hash elements in sorted order (OrdSet guarantees this)
            for elem in set {
                hash = hash_value(hash, elem);
            }
        }
        Value::Interval(iv) => {
            // Hash interval elements (same as Set for correctness)
            // For efficiency, we iterate through the interval values
            // Fast path: avoid BigInt allocation when bounds fit in i64
            let len = match (iv.low.to_i64(), iv.high.to_i64()) {
                (Some(low), Some(high)) => (high - low + 1) as u64,
                _ => {
                    let len = &iv.high - &iv.low + BigInt::one();
                    len.to_u64().unwrap_or_else(|| {
                        iv.high
                            .to_u64()
                            .unwrap_or(0)
                            .wrapping_sub(iv.low.to_u64().unwrap_or(0))
                    })
                }
            };
            hash ^= len;
            hash = hash.wrapping_mul(FNV_PRIME);
            for v in iv.iter_values() {
                hash = hash_value(hash, &v);
            }
        }
        Value::Subset(sv) => {
            // Hash subset elements (same as Set for correctness)
            // This eagerly iterates - could be expensive for large powersets
            if let Some(iter) = sv.iter() {
                let elements: Vec<_> = iter.collect();
                hash ^= elements.len() as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
                for elem in elements {
                    hash = hash_value(hash, &elem);
                }
            }
        }
        Value::FuncSet(fsv) => {
            // Hash function set elements in sorted order (same as Set for correctness)
            // Use to_ord_set() to ensure consistent ordering across different set representations
            if let Some(set) = fsv.to_ord_set() {
                hash ^= set.len() as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
                for elem in set.iter() {
                    hash = hash_value(hash, elem);
                }
            }
        }
        Value::RecordSet(rsv) => {
            // Hash record set elements in sorted order (same as Set for correctness)
            // Use to_ord_set() to ensure consistent ordering across different set representations
            if let Some(set) = rsv.to_ord_set() {
                hash ^= set.len() as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
                for elem in set.iter() {
                    hash = hash_value(hash, elem);
                }
            }
        }
        Value::TupleSet(tsv) => {
            // Hash tuple set elements in sorted order (same as Set for correctness)
            // Use to_ord_set() to ensure consistent ordering across different set representations
            if let Some(set) = tsv.to_ord_set() {
                hash ^= set.len() as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
                for elem in set.iter() {
                    hash = hash_value(hash, elem);
                }
            }
        }
        Value::SetCup(scv) => {
            // Hash set union elements if enumerable
            if let Some(set) = scv.to_ord_set() {
                hash ^= set.len() as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
                for elem in set.iter() {
                    hash = hash_value(hash, elem);
                }
            } else {
                // Non-enumerable - hash by structure
                hash = hash_value(hash, &scv.set1);
                hash = hash_value(hash, &scv.set2);
            }
        }
        Value::SetCap(scv) => {
            // Hash set intersection elements if enumerable
            if let Some(set) = scv.to_ord_set() {
                hash ^= set.len() as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
                for elem in set.iter() {
                    hash = hash_value(hash, elem);
                }
            } else {
                // Non-enumerable - hash by structure
                hash = hash_value(hash, &scv.set1);
                hash = hash_value(hash, &scv.set2);
            }
        }
        Value::SetDiff(sdv) => {
            // Hash set difference elements if enumerable
            if let Some(set) = sdv.to_ord_set() {
                hash ^= set.len() as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
                for elem in set.iter() {
                    hash = hash_value(hash, elem);
                }
            } else {
                // Non-enumerable - hash by structure
                hash = hash_value(hash, &sdv.set1);
                hash = hash_value(hash, &sdv.set2);
            }
        }
        Value::Func(func) => {
            // Hash domain size then key-value pairs
            // Note: Domain elements are NOT hashed separately because they're
            // identical to the mapping keys (domain = set of keys). Hashing
            // both would be redundant. The mapping iteration produces the same
            // semantic fingerprint since it covers all keys.
            hash ^= func.domain_len() as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
            // Hash mapping (key-value pairs) - keys are implicitly the domain
            for (k, v) in func.mapping_iter() {
                hash = hash_value(hash, k);
                hash = hash_value(hash, v);
            }
        }
        Value::IntFunc(func) => {
            // Array-backed function for integer interval domains
            // Hash domain size (length), then key-value pairs (key = index)
            hash ^= func.len() as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
            // Hash mapping: (min, v0), (min+1, v1), ..., (max, vn)
            for (i, v) in func.values.iter().enumerate() {
                hash = hash_smallint(hash, func.min + i as i64);
                hash = hash_value(hash, v);
            }
        }
        // Lazy functions are hashed by their unique ID (not extensionally)
        Value::LazyFunc(func) => {
            hash ^= func.id;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        Value::Seq(seq) => {
            // Seq is a function with domain 1..n
            // Hash like Func: domain size, then mapping (no separate domain)
            hash ^= seq.len() as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
            // Hash mapping: (1, v1), (2, v2), ..., (n, vn)
            for (i, v) in seq.iter().enumerate() {
                hash = hash_smallint(hash, (i + 1) as i64);
                hash = hash_value(hash, v);
            }
        }
        Value::Tuple(tup) => {
            // Tuple is a function with domain 1..n
            // Hash like Func: domain size, then mapping (no separate domain)
            hash ^= tup.len() as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
            // Hash mapping: (1, v1), (2, v2), ..., (n, vn)
            for (i, v) in tup.iter().enumerate() {
                hash = hash_smallint(hash, (i + 1) as i64);
                hash = hash_value(hash, v);
            }
        }
        Value::Record(rec) => {
            // Record is a function with string domain
            // Hash like Func: domain size, then mapping (no separate domain)
            hash ^= rec.len() as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
            // Hash mapping (key-value pairs)
            // Domain elements (field names) are NOT hashed separately because
            // they're identical to the mapping keys. This matches the Func
            // and Tuple/Seq optimizations above.
            // Uses hash_string to avoid creating temporary Value::String wrappers.
            for (name, val) in rec {
                hash = hash_string(hash, name);
                hash = hash_value(hash, val);
            }
        }
        // Closures are hashed by their unique ID
        Value::Closure(c) => {
            hash ^= c.id;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        // SetPred is hashed by its unique ID (can't evaluate predicates without ctx)
        Value::SetPred(spv) => {
            hash ^= spv.id;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        // KSubset is hashed by iterating elements if enumerable
        Value::KSubset(ksv) => {
            if let Some(iter) = ksv.iter() {
                let elements: Vec<_> = iter.collect();
                hash ^= elements.len() as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
                for elem in elements {
                    hash = hash_value(hash, &elem);
                }
            } else {
                // Non-enumerable - hash by structure
                hash ^= ksv.k as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
                hash = hash_value(hash, &ksv.base);
            }
        }
        // BigUnion is hashed by iterating elements if enumerable
        Value::BigUnion(uv) => {
            if let Some(set) = uv.to_ord_set() {
                hash ^= set.len() as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
                for elem in set.iter() {
                    hash = hash_value(hash, elem);
                }
            } else {
                // Non-enumerable - hash by structure
                hash = hash_value(hash, &uv.set);
            }
        }
        // StringSet is infinite - hash it as a special constant
        Value::StringSet => {
            // Use a constant marker for STRING (infinite set)
            hash ^= 0x5354_5249_4E47_5345; // "STRINGSE" in hex
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        // AnySet is infinite - hash it as a special constant
        Value::AnySet => {
            hash ^= 0x414E_5953_4554_5F5F; // "ANYSET__" in hex
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        // SeqSet is infinite - hash it by the base set
        Value::SeqSet(ssv) => {
            hash ^= 0x5345_5153_4554_5F5F; // "SEQSET__" in hex
            hash = hash.wrapping_mul(FNV_PRIME);
            hash = hash_value(hash, &ssv.base);
        }
    }

    hash
}

// ============================================================================
// XXH3-based fingerprinting (for benchmarking/comparison)
// ============================================================================

/// Compute fingerprint using xxh3 (faster than FNV-1a for larger values)
///
/// This is an alternative fingerprint implementation for benchmarking.
/// Note: This produces DIFFERENT fingerprints than `value_fingerprint`.
pub fn value_fingerprint_xxh3(value: &Value) -> u64 {
    use xxhash_rust::xxh3::xxh3_64;

    // Pre-allocate buffer for serialization
    let mut buffer = Vec::with_capacity(256);
    serialize_value_for_hash(&mut buffer, value);
    xxh3_64(&buffer)
}

/// Serialize a value into bytes for hashing
/// This produces a deterministic byte representation
fn serialize_value_for_hash(buffer: &mut Vec<u8>, value: &Value) {
    match value {
        Value::Bool(b) => {
            buffer.push(0); // type tag
            buffer.push(*b as u8);
        }
        Value::SmallInt(n) => {
            buffer.push(1); // type tag
            buffer.extend_from_slice(&n.to_le_bytes());
        }
        Value::Int(n) => {
            buffer.push(1); // type tag (same as SmallInt)
            buffer.extend_from_slice(&n.to_signed_bytes_le());
        }
        Value::String(s) | Value::ModelValue(s) => {
            buffer.push(if matches!(value, Value::String(_)) {
                2
            } else {
                8
            }); // type tag
            buffer.extend_from_slice(&(s.len() as u32).to_le_bytes());
            buffer.extend_from_slice(s.as_bytes());
        }
        Value::Set(set) => {
            buffer.push(3); // type tag
            buffer.extend_from_slice(&(set.len() as u32).to_le_bytes());
            for elem in set {
                serialize_value_for_hash(buffer, elem);
            }
        }
        Value::Func(func) => {
            buffer.push(4); // type tag
            buffer.extend_from_slice(&(func.domain_len() as u32).to_le_bytes());
            for (k, v) in func.mapping_iter() {
                serialize_value_for_hash(buffer, k);
                serialize_value_for_hash(buffer, v);
            }
        }
        Value::Tuple(t) => {
            buffer.push(4); // type tag (same as Func)
            buffer.extend_from_slice(&(t.len() as u32).to_le_bytes());
            for (i, v) in t.iter().enumerate() {
                // Key is 1-based index
                buffer.push(1); // SmallInt tag
                buffer.extend_from_slice(&((i + 1) as i64).to_le_bytes());
                serialize_value_for_hash(buffer, v);
            }
        }
        Value::Seq(s) => {
            buffer.push(4); // type tag (same as Func)
            buffer.extend_from_slice(&(s.len() as u32).to_le_bytes());
            for (i, v) in s.iter().enumerate() {
                buffer.push(1); // SmallInt tag
                buffer.extend_from_slice(&((i + 1) as i64).to_le_bytes());
                serialize_value_for_hash(buffer, v);
            }
        }
        Value::Record(r) => {
            buffer.push(4); // type tag (same as Func)
            buffer.extend_from_slice(&(r.len() as u32).to_le_bytes());
            for (k, v) in r.iter() {
                buffer.push(2); // String tag
                buffer.extend_from_slice(&(k.len() as u32).to_le_bytes());
                buffer.extend_from_slice(k.as_bytes());
                serialize_value_for_hash(buffer, v);
            }
        }
        Value::IntFunc(func) => {
            buffer.push(4); // type tag (same as Func)
            buffer.extend_from_slice(&(func.values.len() as u32).to_le_bytes());
            for (i, v) in func.values.iter().enumerate() {
                let key = func.min + i as i64;
                buffer.push(1); // SmallInt tag
                buffer.extend_from_slice(&key.to_le_bytes());
                serialize_value_for_hash(buffer, v);
            }
        }
        // For complex lazy types, fall back to FNV fingerprint
        Value::Interval(_)
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
        | Value::SeqSet(_)
        | Value::Closure(_)
        | Value::LazyFunc(_) => {
            // Fall back to FNV for complex types
            let fp = hash_value(0xcbf29ce484222325, value);
            buffer.push(255); // special tag for FNV fallback
            buffer.extend_from_slice(&fp.to_le_bytes());
        }
    }
}

// ============================================================================
// ahash-based fingerprinting (streaming, no buffer allocation)
// ============================================================================

/// Compute fingerprint using ahash streaming API (AES-NI accelerated)
///
/// This avoids buffer allocation by using ahash's streaming interface.
/// Expected to be faster than XXH3 (which requires serialization) and FNV-1a
/// (which is inherently slow) for most value sizes.
///
/// Note: This produces DIFFERENT fingerprints than `value_fingerprint` or `value_fingerprint_xxh3`.
pub fn value_fingerprint_ahash(value: &Value) -> u64 {
    use ahash::AHasher;
    use std::hash::Hasher;

    let mut hasher = AHasher::default();
    hash_value_ahash(&mut hasher, value);
    hasher.finish()
}

/// Hash a value using ahash streaming API
fn hash_value_ahash(hasher: &mut ahash::AHasher, value: &Value) {
    use std::hash::Hasher;

    // Type tag (same scheme as FNV-1a for consistency)
    let tag: u8 = match value {
        Value::Bool(_) => 0,
        Value::SmallInt(_) | Value::Int(_) => 1,
        Value::String(_) => 2,
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
        | Value::SeqSet(_) => 3,
        Value::Func(_) | Value::IntFunc(_) | Value::Seq(_) | Value::Record(_) | Value::Tuple(_) => {
            4
        }
        Value::ModelValue(_) => 8,
        Value::Closure(_) => 9,
        Value::LazyFunc(_) => 10,
    };
    hasher.write_u8(tag);

    match value {
        Value::Bool(b) => {
            hasher.write_u8(*b as u8);
        }
        Value::SmallInt(n) => {
            hasher.write_i64(*n);
        }
        Value::Int(n) => {
            // Hash as signed bytes (same canonical form as FNV)
            let bytes = n.to_signed_bytes_le();
            hasher.write(&bytes);
        }
        Value::String(s) | Value::ModelValue(s) => {
            hasher.write(s.as_bytes());
        }
        Value::Set(set) => {
            hasher.write_usize(set.len());
            for elem in set {
                hash_value_ahash(hasher, elem);
            }
        }
        Value::Func(func) => {
            hasher.write_usize(func.domain_len());
            for (k, v) in func.mapping_iter() {
                hash_value_ahash(hasher, k);
                hash_value_ahash(hasher, v);
            }
        }
        Value::IntFunc(func) => {
            hasher.write_usize(func.values.len());
            for (i, v) in func.values.iter().enumerate() {
                let key = func.min + i as i64;
                hasher.write_i64(key);
                hash_value_ahash(hasher, v);
            }
        }
        Value::Tuple(t) => {
            hasher.write_usize(t.len());
            for (i, v) in t.iter().enumerate() {
                hasher.write_i64((i + 1) as i64); // 1-based index
                hash_value_ahash(hasher, v);
            }
        }
        Value::Seq(s) => {
            hasher.write_usize(s.len());
            for (i, v) in s.iter().enumerate() {
                hasher.write_i64((i + 1) as i64);
                hash_value_ahash(hasher, v);
            }
        }
        Value::Record(r) => {
            hasher.write_usize(r.len());
            for (k, v) in r.iter() {
                hasher.write(k.as_bytes());
                hash_value_ahash(hasher, v);
            }
        }
        Value::Interval(iv) => {
            // Hash interval as its bounds
            if let (Some(low), Some(high)) = (iv.low.to_i64(), iv.high.to_i64()) {
                hasher.write_i64(low);
                hasher.write_i64(high);
            } else {
                hasher.write(&iv.low.to_signed_bytes_le());
                hasher.write(&iv.high.to_signed_bytes_le());
            }
        }
        // For lazy types, hash by structure (not evaluated)
        Value::Subset(s) => {
            hasher.write_usize(42); // marker for subset
            hash_value_ahash(hasher, &s.base);
        }
        Value::FuncSet(fs) => {
            hasher.write_usize(43);
            hash_value_ahash(hasher, &fs.domain);
            hash_value_ahash(hasher, &fs.codomain);
        }
        Value::RecordSet(rs) => {
            hasher.write_usize(44);
            hasher.write_usize(rs.fields.len());
            for (name, domain) in rs.fields.iter() {
                hasher.write(name.as_bytes());
                hash_value_ahash(hasher, domain);
            }
        }
        Value::TupleSet(ts) => {
            hasher.write_usize(45);
            hasher.write_usize(ts.components.len());
            for domain in &ts.components {
                hash_value_ahash(hasher, domain);
            }
        }
        Value::SetCup(sc) => {
            hasher.write_usize(46);
            hash_value_ahash(hasher, &sc.set1);
            hash_value_ahash(hasher, &sc.set2);
        }
        Value::SetCap(sc) => {
            hasher.write_usize(47);
            hash_value_ahash(hasher, &sc.set1);
            hash_value_ahash(hasher, &sc.set2);
        }
        Value::SetDiff(sd) => {
            hasher.write_usize(48);
            hash_value_ahash(hasher, &sd.set1);
            hash_value_ahash(hasher, &sd.set2);
        }
        Value::SetPred(_) => {
            // SetPred contains closures which are complex; use fallback
            hasher.write_u64(hash_value(0xcbf29ce484222325, value));
        }
        Value::KSubset(ks) => {
            hasher.write_usize(50);
            hasher.write_usize(ks.k);
            hash_value_ahash(hasher, &ks.base);
        }
        Value::BigUnion(bu) => {
            hasher.write_usize(51);
            hash_value_ahash(hasher, &bu.set);
        }
        Value::StringSet => {
            hasher.write_usize(52);
        }
        Value::AnySet => {
            hasher.write_usize(53);
        }
        Value::SeqSet(ss) => {
            hasher.write_usize(54);
            hash_value_ahash(hasher, &ss.base);
        }
        Value::Closure(_) | Value::LazyFunc(_) => {
            // Fall back to FNV for closures/lazy functions
            hasher.write_u64(hash_value(0xcbf29ce484222325, value));
        }
    }
}

/// Compute salt for a variable at given position (same algorithm as VarRegistry)
fn compute_var_salt_inline(idx: usize, name: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    for byte in (idx as u64).to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    for byte in name.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash ^= 0xFF;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash
}

/// Compare two variable maps lexicographically (TLC-compatible).
///
/// Variables are in sorted order (OrdMap keys), so we compare value by value
/// in alphabetical variable name order.
fn compare_vars_lexicographic(a: &OrdMap<Arc<str>, Value>, b: &OrdMap<Arc<str>, Value>) -> Ordering {
    for ((_, va), (_, vb)) in a.iter().zip(b.iter()) {
        match va.cmp(vb) {
            Ordering::Equal => continue,
            other => return other,
        }
    }
    Ordering::Equal
}

fn compute_fingerprint(vars: &OrdMap<Arc<str>, Value>) -> Fingerprint {
    const FNV_PRIME: u64 = 0x100000001b3;

    // Use XOR-based combination (same algorithm as compute_fingerprint_from_array)
    // Variables in OrdMap are in sorted order, matching registry order
    let mut combined = 0u64;
    for (i, (name, value)) in vars.iter().enumerate() {
        let salt = compute_var_salt_inline(i, name);
        let value_fp = value_fingerprint(value);
        let contribution = salt.wrapping_mul(value_fp.wrapping_add(1));
        combined ^= contribution;
    }

    // Final mixing to improve distribution
    combined = combined.wrapping_mul(FNV_PRIME);
    combined ^= combined >> 33;
    combined = combined.wrapping_mul(FNV_PRIME);

    Fingerprint(combined)
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "State{{")?;
        let mut first = true;
        for (name, value) in &self.vars {
            if !first {
                write!(f, ", ")?;
            }
            first = false;
            write!(f, "{}: {:?}", name, value)?;
        }
        write!(f, "}}")
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "/\\ state")?;
        for (name, value) in &self.vars {
            writeln!(f, "   /\\ {} = {}", name, value)?;
        }
        Ok(())
    }
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.vars == other.vars
    }
}

impl Eq for State {}

impl Hash for State {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fingerprint().0.hash(state);
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare by fingerprint for speed; fall back to full comparison
        let fp_cmp = self.fingerprint().cmp(&other.fingerprint());
        if fp_cmp != Ordering::Equal {
            return fp_cmp;
        }
        // Same fingerprint - compare content (rare, only on collision)
        self.vars.cmp(&other.vars)
    }
}

/// Compute fingerprint from an array of values and variable registry
///
/// Uses pre-computed salts from the registry to avoid hashing variable names
/// in the hot path. Each variable contributes independently via XOR combination
/// of its salt with its value's fingerprint.
///
/// NOTE: This produces different fingerprints than `compute_fingerprint` (State-based).
/// ArrayState and State should not be mixed in the same seen set.
pub fn compute_fingerprint_from_array(values: &[Value], registry: &VarRegistry) -> Fingerprint {
    const FNV_PRIME: u64 = 0x100000001b3;

    // Combine per-variable contributions via XOR
    // Each contribution is: salt[i] mixed with value_fingerprint(values[i])
    let mut combined = 0u64;
    for (i, value) in values.iter().enumerate() {
        let salt = registry.fp_salt(VarIndex(i as u16));
        let value_fp = value_fingerprint(value);
        // Mix salt and value fingerprint together, then XOR into result
        // Use wrapping_mul for mixing to get better avalanche effect
        let contribution = salt.wrapping_mul(value_fp.wrapping_add(1));
        combined ^= contribution;
    }

    // Final mixing to improve distribution
    combined = finalize_fingerprint_xor(combined, FNV_PRIME);

    Fingerprint(combined)
}

#[inline]
fn finalize_fingerprint_xor(mut combined: u64, fnv_prime: u64) -> u64 {
    combined = combined.wrapping_mul(fnv_prime);
    combined ^= combined >> 33;
    combined = combined.wrapping_mul(fnv_prime);
    combined
}

/// Entry for undo stack during bind/unbind enumeration
///
/// Used by `ArrayState::bind()` and `ArrayState::unbind_to()` to implement
/// TLC-style mutable state exploration with backtracking.
///
/// See designs/2026-01-13-bind-unbind-architecture.md for design rationale.
#[derive(Clone, Debug)]
pub struct UndoEntry {
    /// The variable index that was bound
    pub idx: VarIndex,
    /// The previous value before binding (to restore on unbind)
    pub old_value: Value,
}

struct ArrayStateFpCache {
    /// XOR of salted contributions (pre-final-mix)
    combined_xor: u64,
    /// Final mixed fingerprint
    fingerprint: Fingerprint,
    /// Optional per-variable cached value fingerprints.
    ///
    /// When present, incremental fingerprint updates can avoid recomputing the
    /// old value's fingerprint on each update.
    ///
    /// This cache is intentionally dropped on `Clone` to avoid per-successor
    /// allocations in the BFS hot path. It can be rehydrated on demand via
    /// `ArrayState::ensure_fp_cache_with_value_fps`.
    value_fps: Option<Box<[u64]>>,
}

impl Clone for ArrayStateFpCache {
    fn clone(&self) -> Self {
        Self {
            combined_xor: self.combined_xor,
            fingerprint: self.fingerprint,
            value_fps: None,
        }
    }
}

/// Array-based state for O(1) variable access
///
/// Unlike `State` which uses OrdMap, this stores values in a fixed-size array
/// indexed by VarIndex. This provides O(1) get/set operations and single-allocation
/// state creation, which is critical for high-performance model checking.
///
/// # Usage
///
/// ```ignore
/// let mut state = ArrayState::new(registry.len());
/// state.set(idx_x, Value::int(1));
/// state.set(idx_y, Value::int(2));
/// let fp = state.fingerprint(registry);
/// ```
#[derive(Clone)]
pub struct ArrayState {
    /// Values indexed by VarIndex
    values: Box<[Value]>,
    /// Cached fingerprint + per-variable fingerprint cache (None until computed)
    fp_cache: Option<ArrayStateFpCache>,
}

impl ArrayState {
    /// Create a new ArrayState with the given number of variables
    /// All values are initialized to Bool(false) as a placeholder
    #[inline]
    pub fn new(num_vars: usize) -> Self {
        #[cfg(feature = "memory-stats")]
        {
            crate::value::memory_stats::inc_array_state();
            crate::value::memory_stats::inc_array_state_bytes(num_vars);
        }

        ArrayState {
            values: vec![Value::Bool(false); num_vars].into_boxed_slice(),
            fp_cache: None,
        }
    }

    /// Create an ArrayState directly from a Vec of values
    /// Useful for testing and cases where you already have the values
    #[inline]
    pub fn from_values(values: Vec<Value>) -> Self {
        #[cfg(feature = "memory-stats")]
        {
            crate::value::memory_stats::inc_array_state();
            crate::value::memory_stats::inc_array_state_bytes(values.len());
        }

        ArrayState {
            values: values.into_boxed_slice(),
            fp_cache: None,
        }
    }

    /// Create an ArrayState from a State using the given registry
    ///
    /// NOTE: Does NOT copy the State's fingerprint because ArrayState uses
    /// a different fingerprinting algorithm optimized for array-based states.
    /// The fingerprint will be computed on demand using `fingerprint(&registry)`.
    pub fn from_state(state: &State, registry: &VarRegistry) -> Self {
        #[cfg(feature = "memory-stats")]
        {
            crate::value::memory_stats::inc_array_state();
            crate::value::memory_stats::inc_array_state_bytes(registry.len());
        }

        let mut values: Vec<Value> = vec![Value::Bool(false); registry.len()];
        for (name, value) in state.vars() {
            if let Some(idx) = registry.get(name) {
                values[idx.as_usize()] = value.clone();
            }
        }
        ArrayState {
            values: values.into_boxed_slice(),
            fp_cache: None, // Will be computed on demand
        }
    }

    /// Overwrite all values from a slice.
    ///
    /// This is useful for reusing a single `ArrayState` allocation as a scratch buffer
    /// (e.g., when reading from bulk state storage).
    #[inline]
    pub fn overwrite_from_slice(&mut self, values: &[Value]) {
        // `clone_from_slice` panics on length mismatch.
        self.values.clone_from_slice(values);
        self.fp_cache = None; // Invalidate cached fingerprint + per-var cache
    }

    /// Get a value by index
    #[inline(always)]
    pub fn get(&self, idx: VarIndex) -> &Value {
        &self.values[idx.as_usize()]
    }

    /// Set a value by index
    #[inline(always)]
    pub fn set(&mut self, idx: VarIndex, value: Value) {
        self.values[idx.as_usize()] = value;
        self.fp_cache = None; // Invalidate cached fingerprint + per-var cache
    }

    /// Set a value by index, updating the cached fingerprint incrementally when available.
    ///
    /// If this ArrayState has a cached fingerprint (via a prior call to `fingerprint()`),
    /// we can update it in O(1) using XOR-delta updates instead of recomputing over all vars.
    ///
    /// If no fingerprint cache is present, this behaves like `set()` and leaves the cache empty.
    #[inline]
    pub fn set_with_registry(&mut self, idx: VarIndex, value: Value, registry: &VarRegistry) {
        let Some(cache) = self.fp_cache.as_mut() else {
            self.values[idx.as_usize()] = value;
            return;
        };

        debug_assert_eq!(registry.len(), self.values.len());

        let idx_usize = idx.as_usize();
        let new_fp = value_fingerprint(&value);
        let old_fp = cache
            .value_fps
            .as_ref()
            .map(|fps| fps[idx_usize])
            .unwrap_or_else(|| value_fingerprint(&self.values[idx_usize]));
        if old_fp != new_fp {
            let salt = registry.fp_salt(idx);
            let old_contrib = salt.wrapping_mul(old_fp.wrapping_add(1));
            let new_contrib = salt.wrapping_mul(new_fp.wrapping_add(1));
            cache.combined_xor ^= old_contrib ^ new_contrib;

            const FNV_PRIME: u64 = 0x100000001b3;
            let mixed = finalize_fingerprint_xor(cache.combined_xor, FNV_PRIME);
            cache.fingerprint = Fingerprint(mixed);
            if let Some(fps) = cache.value_fps.as_mut() {
                fps[idx_usize] = new_fp;
            }
        }

        self.values[idx_usize] = value;
    }

    /// Set a value by index with a precomputed value fingerprint.
    ///
    /// This is a specialization for hot-path enumeration that avoids recomputing
    /// `value_fingerprint()` when the caller already has it.
    #[inline]
    pub fn set_with_fp(
        &mut self,
        idx: VarIndex,
        value: Value,
        value_fp: u64,
        registry: &VarRegistry,
    ) {
        let Some(cache) = self.fp_cache.as_mut() else {
            self.values[idx.as_usize()] = value;
            return;
        };

        debug_assert_eq!(registry.len(), self.values.len());

        let idx_usize = idx.as_usize();
        let old_fp = cache
            .value_fps
            .as_ref()
            .map(|fps| fps[idx_usize])
            .unwrap_or_else(|| value_fingerprint(&self.values[idx_usize]));
        if old_fp != value_fp {
            let salt = registry.fp_salt(idx);
            let old_contrib = salt.wrapping_mul(old_fp.wrapping_add(1));
            let new_contrib = salt.wrapping_mul(value_fp.wrapping_add(1));
            cache.combined_xor ^= old_contrib ^ new_contrib;

            const FNV_PRIME: u64 = 0x100000001b3;
            let mixed = finalize_fingerprint_xor(cache.combined_xor, FNV_PRIME);
            cache.fingerprint = Fingerprint(mixed);
            if let Some(fps) = cache.value_fps.as_mut() {
                fps[idx_usize] = value_fp;
            }
        }

        self.values[idx_usize] = value;
    }

    /// Get the values array
    #[inline]
    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// Compute and cache the fingerprint
    pub fn fingerprint(&mut self, registry: &VarRegistry) -> Fingerprint {
        if let Some(cache) = &self.fp_cache {
            return cache.fingerprint;
        }

        const FNV_PRIME: u64 = 0x100000001b3;

        let mut combined_xor = 0u64;

        for (i, value) in self.values.iter().enumerate() {
            let value_fp = value_fingerprint(value);
            let salt = registry.fp_salt(VarIndex(i as u16));
            let contribution = salt.wrapping_mul(value_fp.wrapping_add(1));
            combined_xor ^= contribution;
        }

        let mixed = finalize_fingerprint_xor(combined_xor, FNV_PRIME);
        let fp = Fingerprint(mixed);

        self.fp_cache = Some(ArrayStateFpCache {
            combined_xor,
            fingerprint: fp,
            value_fps: None,
        });

        fp
    }

    /// Ensure the fingerprint cache includes per-variable value fingerprints.
    ///
    /// This supports fast incremental updates via `set_with_registry`/`set_with_fp` without
    /// recomputing old-value fingerprints on every update.
    pub fn ensure_fp_cache_with_value_fps(&mut self, registry: &VarRegistry) {
        if self.fp_cache.is_none() {
            // Compute fingerprint + value_fps together in one pass.
            const FNV_PRIME: u64 = 0x100000001b3;

            let mut combined_xor = 0u64;
            let mut value_fps: Vec<u64> = Vec::with_capacity(self.values.len());

            for (i, value) in self.values.iter().enumerate() {
                let value_fp = value_fingerprint(value);
                value_fps.push(value_fp);
                let salt = registry.fp_salt(VarIndex(i as u16));
                let contribution = salt.wrapping_mul(value_fp.wrapping_add(1));
                combined_xor ^= contribution;
            }

            let mixed = finalize_fingerprint_xor(combined_xor, FNV_PRIME);
            let fp = Fingerprint(mixed);

            self.fp_cache = Some(ArrayStateFpCache {
                combined_xor,
                fingerprint: fp,
                value_fps: Some(value_fps.into_boxed_slice()),
            });
            return;
        }

        let Some(cache) = self.fp_cache.as_mut() else {
            return;
        };
        if cache.value_fps.is_some() {
            return;
        }

        let mut value_fps: Vec<u64> = Vec::with_capacity(self.values.len());
        for value in self.values.iter() {
            value_fps.push(value_fingerprint(value));
        }
        cache.value_fps = Some(value_fps.into_boxed_slice());
    }

    /// Get cached fingerprint if available
    #[inline]
    pub fn cached_fingerprint(&self) -> Option<Fingerprint> {
        self.fp_cache.as_ref().map(|c| c.fingerprint)
    }

    /// Convert to State using the given registry
    ///
    /// The State will have its fingerprint computed using State's algorithm,
    /// which is compatible with ArrayState's algorithm (both use XOR-based combination).
    pub fn to_state(&self, registry: &VarRegistry) -> State {
        let vars: OrdMap<Arc<str>, Value> = registry
            .iter()
            .map(|(idx, name)| (Arc::clone(name), self.values[idx.as_usize()].clone()))
            .collect();

        // Compute fingerprint using State's algorithm (which is compatible with ArrayState)
        let fingerprint = compute_fingerprint(&vars);

        State {
            vars,
            fingerprint,
            canonical_fingerprint: OnceLock::new(),
        }
    }

    /// Number of variables
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    // ========================================================================
    // Bind/Unbind Methods for TLC-Style Enumeration
    //
    // These methods implement TLC's mutable state exploration pattern:
    // - bind() saves old value, sets new value
    // - unbind() restores previous value from undo stack
    // - unbind_to() batch restores to a save point (for disjunction branches)
    // - snapshot() creates an immutable State copy for storing successors
    //
    // This avoids cloning the entire state per branch, matching TLC's O(1)
    // bind/unbind performance. See designs/2026-01-13-bind-unbind-architecture.md
    // ========================================================================

    /// Bind a variable to a new value, recording the old value for later unbind.
    ///
    /// This is an O(1) operation that:
    /// 1. Pushes the old value onto the undo stack
    /// 2. Sets the new value in the values array
    /// 3. Invalidates the fingerprint cache
    ///
    /// Use with `unbind()` or `unbind_to()` to restore the previous value.
    ///
    /// # Pattern
    /// ```ignore
    /// let save_point = undo.len();
    /// state.bind(idx, new_value, &mut undo);
    /// // ... explore with new binding ...
    /// state.unbind_to(&mut undo, save_point);  // restore
    /// ```
    #[inline]
    pub fn bind(&mut self, idx: VarIndex, value: Value, undo: &mut Vec<UndoEntry>) {
        let old = std::mem::replace(&mut self.values[idx.as_usize()], value);
        undo.push(UndoEntry { idx, old_value: old });
        self.fp_cache = None; // Invalidate fingerprint
    }

    /// Unbind the most recent binding (restore previous value).
    ///
    /// This is an O(1) operation that pops the last entry from the undo stack
    /// and restores the old value.
    ///
    /// # Panics
    /// Panics if the undo stack is empty.
    ///
    /// # Note
    /// For unbinding multiple bindings, prefer `unbind_to()` which only
    /// invalidates the fingerprint cache once.
    #[inline]
    pub fn unbind(&mut self, undo: &mut Vec<UndoEntry>) {
        let entry = undo.pop().expect("unbind called with empty undo stack");
        self.values[entry.idx.as_usize()] = entry.old_value;
        self.fp_cache = None; // Invalidate fingerprint
    }

    /// Unbind multiple bindings at once, restoring to a save point.
    ///
    /// This is optimized for disjunction branches where we need to restore
    /// multiple bindings after exploring one branch. It only invalidates the
    /// fingerprint cache once at the end (z4 solver pattern).
    ///
    /// # Arguments
    /// * `undo` - The undo stack
    /// * `target_len` - The save point to restore to (typically `undo.len()` before binding)
    ///
    /// # Pattern
    /// ```ignore
    /// // Disjunction: try a, restore, try b, restore
    /// let save_point = undo.len();
    /// enumerate_branch_a(state, undo, ...)?;
    /// state.unbind_to(undo, save_point);  // restore for b
    /// enumerate_branch_b(state, undo, ...)?;
    /// state.unbind_to(undo, save_point);  // restore for caller
    /// ```
    #[inline]
    pub fn unbind_to(&mut self, undo: &mut Vec<UndoEntry>, target_len: usize) {
        if undo.len() <= target_len {
            return; // Nothing to unbind
        }
        // Inline loop - avoid repeated function call overhead (z4 solver pattern)
        while undo.len() > target_len {
            let entry = undo.pop().expect("checked above");
            self.values[entry.idx.as_usize()] = entry.old_value;
        }
        // Invalidate fingerprint once at end, not per unbind
        self.fp_cache = None;
    }

    /// Create a snapshot of the current state for storing as a successor.
    ///
    /// This is the ONLY place we clone values during bind/unbind enumeration.
    /// All other operations (bind, unbind) are O(1) mutations.
    ///
    /// Returns a full `State` suitable for storing in the visited set or
    /// successor list.
    #[inline]
    pub fn snapshot(&self, registry: &VarRegistry) -> State {
        State::from_indexed(&self.values, registry)
    }
}

impl fmt::Debug for ArrayState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ArrayState({} vars", self.values.len())?;
        if let Some(cache) = &self.fp_cache {
            write!(f, ", fp={:016x}", cache.fingerprint.0)?;
        }
        write!(f, ")")
    }
}

impl Hash for ArrayState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // If fingerprint is cached, use it; otherwise hash the values array
        if let Some(cache) = &self.fp_cache {
            cache.fingerprint.0.hash(state);
        } else {
            // Fall back to hashing all values
            for v in self.values.iter() {
                // Just hash the discriminant and first few bytes for speed
                std::mem::discriminant(v).hash(state);
            }
        }
    }
}

// ============================================================================
// DiffSuccessor - Compact diff-based successor representation
// ============================================================================

/// Inline capacity for DiffSuccessor changes.
/// Most TLA+ actions change 1-4 variables. This avoids heap allocation
/// for the common case, reducing allocator pressure in the hot path.
pub const DIFF_INLINE_CAPACITY: usize = 4;

/// Type alias for the changes storage in DiffSuccessor.
/// Uses SmallVec to store up to 4 changes inline (no heap allocation).
pub type DiffChanges = SmallVec<[(VarIndex, Value); DIFF_INLINE_CAPACITY]>;

/// A compact representation of a successor state as a diff from a base state.
///
/// This avoids cloning the entire ArrayState during enumeration. Only the
/// changed values and the result fingerprint are stored. The full ArrayState
/// can be materialized later (only for unique states, ~5% of successors).
///
/// # Performance
///
/// For specs with high duplicate rates (95%+), this can dramatically reduce
/// the number of Value clones:
/// - Before: Clone N values per successor (for dedup check AND queueing)
/// - After: Clone N values only for unique successors (~5%)
///
/// Uses SmallVec to store changes inline (up to 4 changes) to avoid heap
/// allocation for most successors, reducing allocator pressure.
#[derive(Clone)]
pub struct DiffSuccessor {
    /// Fingerprint of the result state
    pub fingerprint: Fingerprint,
    /// Changes from base state: (VarIndex, new_value)
    /// Uses SmallVec to avoid heap allocation for small change sets.
    pub changes: DiffChanges,
}

impl DiffSuccessor {
    /// Create a new DiffSuccessor with the given fingerprint and changes.
    #[inline]
    pub fn new(fingerprint: Fingerprint, changes: Vec<(VarIndex, Value)>) -> Self {
        Self {
            fingerprint,
            changes: SmallVec::from_vec(changes),
        }
    }

    /// Create a new DiffSuccessor with the given fingerprint and changes (SmallVec).
    /// This is the preferred constructor when building changes incrementally.
    #[inline]
    pub fn from_smallvec(fingerprint: Fingerprint, changes: DiffChanges) -> Self {
        Self {
            fingerprint,
            changes,
        }
    }

    /// Materialize this diff into a full ArrayState by applying changes to the base.
    ///
    /// This clones the base ArrayState and clones the changed values.
    /// Prefer `into_array_state` when you can consume the DiffSuccessor.
    #[inline]
    pub fn materialize(&self, base: &ArrayState, registry: &VarRegistry) -> ArrayState {
        let mut result = base.clone();

        let (mut combined_xor, base_value_fps): (u64, Option<&[u64]>) = match &base.fp_cache {
            Some(cache) => (cache.combined_xor, cache.value_fps.as_deref()),
            None => (combined_xor_from_array(&base.values, registry), None),
        };

        for (idx, value) in &self.changes {
            let idx_usize = idx.as_usize();
            let old_fp = base_value_fps
                .map(|fps| fps[idx_usize])
                .unwrap_or_else(|| value_fingerprint(&base.values[idx_usize]));
            let new_fp = value_fingerprint(value);
            if old_fp != new_fp {
                let salt = registry.fp_salt(*idx);
                let old_contrib = salt.wrapping_mul(old_fp.wrapping_add(1));
                let new_contrib = salt.wrapping_mul(new_fp.wrapping_add(1));
                combined_xor ^= old_contrib ^ new_contrib;
            }

            result.values[idx_usize] = value.clone();
        }

        // The fingerprint was pre-computed during enumeration - store it in cache
        result.fp_cache = Some(ArrayStateFpCache {
            combined_xor,
            fingerprint: self.fingerprint,
            value_fps: None,
        });

        debug_assert_eq!(
            self.fingerprint,
            {
                const FNV_PRIME: u64 = 0x100000001b3;
                Fingerprint(finalize_fingerprint_xor(combined_xor, FNV_PRIME))
            },
            "DiffSuccessor fingerprint does not match computed combined_xor"
        );

        result
    }

    /// Materialize this diff into a full ArrayState by applying changes to the base.
    ///
    /// Consumes the DiffSuccessor, moving changed Values into the result to avoid extra clones.
    #[inline]
    pub fn into_array_state(self, base: &ArrayState, registry: &VarRegistry) -> ArrayState {
        let mut result = base.clone();

        let (mut combined_xor, base_value_fps): (u64, Option<&[u64]>) = match &base.fp_cache {
            Some(cache) => (cache.combined_xor, cache.value_fps.as_deref()),
            None => (combined_xor_from_array(&base.values, registry), None),
        };

        for (idx, value) in self.changes {
            let idx_usize = idx.as_usize();
            let old_fp = base_value_fps
                .map(|fps| fps[idx_usize])
                .unwrap_or_else(|| value_fingerprint(&base.values[idx_usize]));
            let new_fp = value_fingerprint(&value);
            if old_fp != new_fp {
                let salt = registry.fp_salt(idx);
                let old_contrib = salt.wrapping_mul(old_fp.wrapping_add(1));
                let new_contrib = salt.wrapping_mul(new_fp.wrapping_add(1));
                combined_xor ^= old_contrib ^ new_contrib;
            }

            result.values[idx_usize] = value;
        }

        result.fp_cache = Some(ArrayStateFpCache {
            combined_xor,
            fingerprint: self.fingerprint,
            value_fps: None,
        });

        debug_assert_eq!(
            self.fingerprint,
            {
                const FNV_PRIME: u64 = 0x100000001b3;
                Fingerprint(finalize_fingerprint_xor(combined_xor, FNV_PRIME))
            },
            "DiffSuccessor fingerprint does not match computed combined_xor"
        );

        result
    }

    /// Get the number of changed variables
    #[inline]
    pub fn num_changes(&self) -> usize {
        self.changes.len()
    }
}

impl std::fmt::Debug for DiffSuccessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DiffSuccessor(fp={:016x}, {} changes)",
            self.fingerprint.0,
            self.changes.len()
        )
    }
}

/// Compute a successor fingerprint incrementally from a base state and changes.
///
/// This avoids constructing the full state just to compute the fingerprint.
/// Uses the XOR-based fingerprint algorithm which supports incremental updates.
///
/// # Arguments
/// * `base` - Base ArrayState (must have fingerprint cache with value_fps)
/// * `changes` - List of (VarIndex, new_value) pairs
/// * `registry` - Variable registry for salt values
pub fn compute_diff_fingerprint(
    base: &ArrayState,
    changes: &[(VarIndex, Value)],
    registry: &VarRegistry,
) -> Fingerprint {
    // Use cached combined_xor when available, otherwise compute it from the base state.
    let (mut combined_xor, base_value_fps): (u64, Option<&[u64]>) = match &base.fp_cache {
        Some(cache) => (cache.combined_xor, cache.value_fps.as_deref()),
        None => (combined_xor_from_array(&base.values, registry), None),
    };

    // Apply changes: XOR out old contribution, XOR in new contribution.
    //
    // We do NOT require base.value_fps to be present: computing old value fingerprints on-demand
    // for just the changed variables is much cheaper than cloning/materializing full states.
    for (idx, new_value) in changes {
        let idx_usize = idx.as_usize();
        let old_fp = base_value_fps
            .map(|fps| fps[idx_usize])
            .unwrap_or_else(|| value_fingerprint(&base.values[idx_usize]));
        let new_fp = value_fingerprint(new_value);

        if old_fp != new_fp {
            let salt = registry.fp_salt(*idx);
            let old_contrib = salt.wrapping_mul(old_fp.wrapping_add(1));
            let new_contrib = salt.wrapping_mul(new_fp.wrapping_add(1));
            combined_xor ^= old_contrib ^ new_contrib;
        }
    }

    const FNV_PRIME: u64 = 0x100000001b3;
    Fingerprint(finalize_fingerprint_xor(combined_xor, FNV_PRIME))
}

#[inline]
fn combined_xor_from_array(values: &[Value], registry: &VarRegistry) -> u64 {
    let mut combined_xor = 0u64;
    for (i, value) in values.iter().enumerate() {
        let value_fp = value_fingerprint(value);
        let salt = registry.fp_salt(VarIndex(i as u16));
        let contribution = salt.wrapping_mul(value_fp.wrapping_add(1));
        combined_xor ^= contribution;
    }
    combined_xor
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::ToBigInt;

    #[test]
    fn test_state_new() {
        let state = State::new();
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
    }

    #[test]
    fn test_state_from_iter() {
        let state = State::from_pairs([("x", Value::int(1)), ("y", Value::int(2))]);
        assert_eq!(state.len(), 2);
        assert_eq!(state.get("x"), Some(&Value::int(1)));
        assert_eq!(state.get("y"), Some(&Value::int(2)));
    }

    #[test]
    fn test_state_with_var() {
        let s1 = State::new();
        let s2 = s1.with_var("x", Value::int(1));
        let s3 = s2.with_var("y", Value::int(2));

        assert!(s1.is_empty());
        assert_eq!(s2.len(), 1);
        assert_eq!(s3.len(), 2);
    }

    #[test]
    fn test_fingerprint_deterministic() {
        let s1 = State::from_pairs([("x", Value::int(1)), ("y", Value::int(2))]);
        let s2 = State::from_pairs([
            ("y", Value::int(2)), // Different order
            ("x", Value::int(1)),
        ]);

        // Same content should have same fingerprint
        assert_eq!(s1.fingerprint(), s2.fingerprint());
    }

    #[test]
    fn test_smallint_signed_bytes_matches_bigint() {
        fn smallint_signed_bytes_le(n: i64) -> Vec<u8> {
            let bytes = n.to_le_bytes();
            let mut len = bytes.len();
            if n >= 0 {
                while len > 1 && bytes[len - 1] == 0x00 && (bytes[len - 2] & 0x80) == 0 {
                    len -= 1;
                }
            } else {
                while len > 1 && bytes[len - 1] == 0xFF && (bytes[len - 2] & 0x80) != 0 {
                    len -= 1;
                }
            }
            bytes[..len].to_vec()
        }

        for n in [
            0i64,
            1,
            -1,
            2,
            -2,
            127,
            128,
            255,
            256,
            -127,
            -128,
            -129,
            i64::MAX,
            i64::MIN,
        ] {
            let big = n.to_bigint().unwrap();
            assert_eq!(smallint_signed_bytes_le(n), big.to_signed_bytes_le());
        }
    }

    #[test]
    fn test_fingerprint_different_values() {
        let s1 = State::from_pairs([("x", Value::int(1))]);
        let s2 = State::from_pairs([("x", Value::int(2))]);

        assert_ne!(s1.fingerprint(), s2.fingerprint());
    }

    #[test]
    fn test_fingerprint_different_vars() {
        let s1 = State::from_pairs([("x", Value::int(1))]);
        let s2 = State::from_pairs([("y", Value::int(1))]);

        assert_ne!(s1.fingerprint(), s2.fingerprint());
    }

    #[test]
    fn test_state_equality() {
        let s1 = State::from_pairs([("x", Value::int(1))]);
        let s2 = State::from_pairs([("x", Value::int(1))]);
        let s3 = State::from_pairs([("x", Value::int(2))]);

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_fingerprint_complex_values() {
        let s1 = State::from_pairs([
            ("set", Value::set([Value::int(1), Value::int(2)])),
            ("seq", Value::seq([Value::string("a"), Value::string("b")])),
        ]);
        let s2 = State::from_pairs([
            ("seq", Value::seq([Value::string("a"), Value::string("b")])),
            ("set", Value::set([Value::int(2), Value::int(1)])), // Set order doesn't matter
        ]);

        assert_eq!(s1.fingerprint(), s2.fingerprint());
    }

    #[test]
    fn test_array_state_basic() {
        // Create a registry with sorted variable names
        let registry = VarRegistry::from_names(["x", "y", "z"]);

        let mut array_state = ArrayState::new(3);
        array_state.set(VarIndex(0), Value::int(1)); // x
        array_state.set(VarIndex(1), Value::int(2)); // y
        array_state.set(VarIndex(2), Value::int(3)); // z

        assert_eq!(array_state.get(VarIndex(0)), &Value::int(1));
        assert_eq!(array_state.get(VarIndex(1)), &Value::int(2));
        assert_eq!(array_state.get(VarIndex(2)), &Value::int(3));
        assert_eq!(array_state.len(), 3);

        // Convert to State and verify
        let state = array_state.to_state(&registry);
        assert_eq!(state.get("x"), Some(&Value::int(1)));
        assert_eq!(state.get("y"), Some(&Value::int(2)));
        assert_eq!(state.get("z"), Some(&Value::int(3)));
    }

    #[test]
    fn test_array_state_fingerprint_matches_state() {
        // Create a registry with sorted variable names
        // Variables in registry must be in sorted order to match OrdMap iteration
        let registry = VarRegistry::from_names(["x", "y"]);

        // Create a State using the normal API
        let state = State::from_pairs([("x", Value::int(1)), ("y", Value::int(2))]);

        // Create an equivalent ArrayState
        let mut array_state = ArrayState::new(2);
        array_state.set(VarIndex(0), Value::int(1)); // x
        array_state.set(VarIndex(1), Value::int(2)); // y

        // Fingerprints should match
        let state_fp = state.fingerprint();
        let array_fp = array_state.fingerprint(&registry);
        assert_eq!(
            state_fp, array_fp,
            "State fp {:016x} != ArrayState fp {:016x}",
            state_fp.0, array_fp.0
        );
    }

    #[test]
    fn test_array_state_incremental_fingerprint_updates() {
        let registry = VarRegistry::from_names(["x", "y", "z"]);

        let mut a1 = ArrayState::new(3);
        a1.set(VarIndex(0), Value::int(1));
        let y0 = Value::set([Value::int(1), Value::int(2)]);
        a1.set(VarIndex(1), y0.clone());
        a1.set(VarIndex(2), Value::int(3));

        // Prime fingerprint cache.
        let fp0 = a1.fingerprint(&registry);

        // Update using incremental fingerprint path with caller-provided fp.
        let y1 = Value::set([Value::int(1), Value::int(2), Value::int(4)]);
        let y1_fp = value_fingerprint(&y1);
        a1.set_with_fp(VarIndex(1), y1.clone(), y1_fp, &registry);
        let fp1_inc = a1.fingerprint(&registry);

        // Recompute from scratch and ensure the incremental update matches.
        let mut a2 = ArrayState::new(3);
        a2.set(VarIndex(0), Value::int(1));
        a2.set(VarIndex(1), y1);
        a2.set(VarIndex(2), Value::int(3));
        let fp1_full = a2.fingerprint(&registry);

        assert_eq!(fp1_inc, fp1_full);

        // Round-trip back to the original value and ensure we get back the original fp.
        a1.set_with_registry(VarIndex(1), y0, &registry);
        let fp0_again = a1.fingerprint(&registry);
        assert_eq!(fp0_again, fp0);
    }

    #[test]
    fn test_array_state_from_state() {
        let registry = VarRegistry::from_names(["a", "b"]);
        let state = State::from_pairs([("a", Value::int(10)), ("b", Value::int(20))]);

        let array_state = ArrayState::from_state(&state, &registry);
        assert_eq!(array_state.get(VarIndex(0)), &Value::int(10)); // a
        assert_eq!(array_state.get(VarIndex(1)), &Value::int(20)); // b

        // Fingerprint is NOT copied (computed on demand using ArrayState's algorithm)
        // This ensures consistency - all ArrayState fingerprints use the same algorithm
        assert_eq!(array_state.cached_fingerprint(), None);

        // When fingerprint is computed, it should match State's fingerprint
        // (both use the same XOR-based algorithm now)
        let mut array_state_mut = array_state;
        let array_fp = array_state_mut.fingerprint(&registry);
        assert_eq!(array_fp, state.fingerprint());
    }

    #[test]
    fn test_state_from_indexed() {
        // Variables in sorted order
        let registry = VarRegistry::from_names(["x", "y", "z"]);
        let values = [Value::int(1), Value::int(2), Value::int(3)];

        // Create state using indexed method
        let state_indexed = State::from_indexed(&values, &registry);

        // Create state using traditional method
        let state_pairs = State::from_pairs([
            ("x", Value::int(1)),
            ("y", Value::int(2)),
            ("z", Value::int(3)),
        ]);

        // Should have same fingerprint
        assert_eq!(
            state_indexed.fingerprint(),
            state_pairs.fingerprint(),
            "from_indexed and from_pairs should produce same fingerprint"
        );

        // And same content
        assert_eq!(state_indexed.get("x"), Some(&Value::int(1)));
        assert_eq!(state_indexed.get("y"), Some(&Value::int(2)));
        assert_eq!(state_indexed.get("z"), Some(&Value::int(3)));
    }

    #[test]
    fn test_state_from_array_state() {
        let registry = VarRegistry::from_names(["a", "b"]);

        let mut array_state = ArrayState::new(2);
        array_state.set(VarIndex(0), Value::int(100)); // a
        array_state.set(VarIndex(1), Value::int(200)); // b

        let state = State::from_array_state(&mut array_state, &registry);

        assert_eq!(state.get("a"), Some(&Value::int(100)));
        assert_eq!(state.get("b"), Some(&Value::int(200)));

        // Fingerprint should match
        let state_pairs = State::from_pairs([("a", Value::int(100)), ("b", Value::int(200))]);
        assert_eq!(state.fingerprint(), state_pairs.fingerprint());
    }

    // ============================================================================
    // Bind/Unbind Tests (Issue #101 - TLC-Style Enumeration)
    // ============================================================================

    #[test]
    fn test_array_state_bind_unbind_basic() {
        // Test basic bind/unbind operations
        let _registry = VarRegistry::from_names(["x", "y", "z"]);
        let mut state = ArrayState::new(3);
        let mut undo: Vec<UndoEntry> = Vec::new();

        // Initial state: all false (default)
        assert_eq!(state.get(VarIndex(0)), &Value::Bool(false));

        // Bind x = 1
        state.bind(VarIndex(0), Value::int(1), &mut undo);
        assert_eq!(state.get(VarIndex(0)), &Value::int(1));
        assert_eq!(undo.len(), 1);

        // Bind y = 2
        state.bind(VarIndex(1), Value::int(2), &mut undo);
        assert_eq!(state.get(VarIndex(1)), &Value::int(2));
        assert_eq!(undo.len(), 2);

        // Unbind y (restore to false)
        state.unbind(&mut undo);
        assert_eq!(state.get(VarIndex(1)), &Value::Bool(false));
        assert_eq!(undo.len(), 1);
        // x should still be bound
        assert_eq!(state.get(VarIndex(0)), &Value::int(1));

        // Unbind x (restore to false)
        state.unbind(&mut undo);
        assert_eq!(state.get(VarIndex(0)), &Value::Bool(false));
        assert_eq!(undo.len(), 0);
    }

    #[test]
    fn test_array_state_unbind_to() {
        // Test unbind_to for batch restore (disjunction pattern)
        let mut state = ArrayState::new(3);
        let mut undo: Vec<UndoEntry> = Vec::new();

        // Bind x = 1, y = 2
        state.bind(VarIndex(0), Value::int(1), &mut undo);
        state.bind(VarIndex(1), Value::int(2), &mut undo);
        let save_point = undo.len(); // save_point = 2

        // Branch A: bind z = 100
        state.bind(VarIndex(2), Value::int(100), &mut undo);
        assert_eq!(state.get(VarIndex(2)), &Value::int(100));
        assert_eq!(undo.len(), 3);

        // Restore for branch B
        state.unbind_to(&mut undo, save_point);
        assert_eq!(state.get(VarIndex(2)), &Value::Bool(false)); // restored
        assert_eq!(undo.len(), 2);
        // x, y still bound
        assert_eq!(state.get(VarIndex(0)), &Value::int(1));
        assert_eq!(state.get(VarIndex(1)), &Value::int(2));

        // Branch B: bind z = 200
        state.bind(VarIndex(2), Value::int(200), &mut undo);
        assert_eq!(state.get(VarIndex(2)), &Value::int(200));

        // Restore to start
        state.unbind_to(&mut undo, 0);
        assert_eq!(state.get(VarIndex(0)), &Value::Bool(false));
        assert_eq!(state.get(VarIndex(1)), &Value::Bool(false));
        assert_eq!(state.get(VarIndex(2)), &Value::Bool(false));
        assert_eq!(undo.len(), 0);
    }

    #[test]
    fn test_array_state_unbind_to_empty_is_noop() {
        // unbind_to with target >= len should be a no-op
        let mut state = ArrayState::new(2);
        let mut undo: Vec<UndoEntry> = Vec::new();

        state.bind(VarIndex(0), Value::int(1), &mut undo);
        assert_eq!(undo.len(), 1);

        // target_len >= undo.len() - no change
        state.unbind_to(&mut undo, 1);
        assert_eq!(undo.len(), 1);
        assert_eq!(state.get(VarIndex(0)), &Value::int(1));

        state.unbind_to(&mut undo, 100);
        assert_eq!(undo.len(), 1);
        assert_eq!(state.get(VarIndex(0)), &Value::int(1));
    }

    #[test]
    fn test_array_state_snapshot_creates_correct_state() {
        let registry = VarRegistry::from_names(["a", "b"]);
        let mut state = ArrayState::new(2);
        let mut undo: Vec<UndoEntry> = Vec::new();

        // Bind values
        state.bind(VarIndex(0), Value::int(10), &mut undo);
        state.bind(VarIndex(1), Value::int(20), &mut undo);

        // Take snapshot
        let snapshot = state.snapshot(&registry);

        // Verify snapshot has correct values
        assert_eq!(snapshot.get("a"), Some(&Value::int(10)));
        assert_eq!(snapshot.get("b"), Some(&Value::int(20)));

        // Modify original state - snapshot should be unchanged
        state.unbind_to(&mut undo, 0);
        assert_eq!(state.get(VarIndex(0)), &Value::Bool(false));
        assert_eq!(snapshot.get("a"), Some(&Value::int(10))); // unchanged
    }

    #[test]
    fn test_array_state_bind_rebind_same_var() {
        // Test binding the same variable multiple times
        let mut state = ArrayState::new(2);
        let mut undo: Vec<UndoEntry> = Vec::new();

        // Bind x = 1
        state.bind(VarIndex(0), Value::int(1), &mut undo);
        assert_eq!(state.get(VarIndex(0)), &Value::int(1));

        // Rebind x = 2 (overwrite)
        state.bind(VarIndex(0), Value::int(2), &mut undo);
        assert_eq!(state.get(VarIndex(0)), &Value::int(2));
        assert_eq!(undo.len(), 2);

        // Unbind back to x = 1
        state.unbind(&mut undo);
        assert_eq!(state.get(VarIndex(0)), &Value::int(1));

        // Unbind back to default
        state.unbind(&mut undo);
        assert_eq!(state.get(VarIndex(0)), &Value::Bool(false));
    }

    #[test]
    fn test_array_state_bind_invalidates_fingerprint() {
        let registry = VarRegistry::from_names(["x"]);
        let mut state = ArrayState::new(1);
        let mut undo: Vec<UndoEntry> = Vec::new();

        // Compute fingerprint
        let fp1 = state.fingerprint(&registry);
        assert!(state.cached_fingerprint().is_some());

        // Bind should invalidate cache
        state.bind(VarIndex(0), Value::int(1), &mut undo);
        assert!(state.cached_fingerprint().is_none());

        // Recompute - should be different
        let fp2 = state.fingerprint(&registry);
        assert_ne!(fp1, fp2);

        // Unbind should also invalidate cache
        state.unbind(&mut undo);
        assert!(state.cached_fingerprint().is_none());

        // Recompute - should match original
        let fp3 = state.fingerprint(&registry);
        assert_eq!(fp1, fp3);
    }

    // ============================================================================
    // Symmetry Canonicalization Tests (Issue #86 Verification)
    // ============================================================================

    /// Helper to create a swap permutation for two model values
    fn create_swap_permutation(a: &str, b: &str) -> FuncValue {
        // Permutation: a -> b, b -> a
        let mut entries = vec![
            (Value::model_value(a), Value::model_value(b)),
            (Value::model_value(b), Value::model_value(a)),
        ];
        entries.sort_by(|(ka, _), (kb, _)| ka.cmp(kb));
        FuncValue::from_sorted_entries(entries)
    }

    /// Helper to create a cyclic permutation for three model values: a -> b -> c -> a
    fn create_cycle_permutation(a: &str, b: &str, c: &str) -> FuncValue {
        let mut entries = vec![
            (Value::model_value(a), Value::model_value(b)),
            (Value::model_value(b), Value::model_value(c)),
            (Value::model_value(c), Value::model_value(a)),
        ];
        entries.sort_by(|(ka, _), (kb, _)| ka.cmp(kb));
        FuncValue::from_sorted_entries(entries)
    }

    #[test]
    fn test_symmetry_canonical_fingerprint_invariant() {
        // Issue #86: Symmetric states MUST get the same canonical fingerprint.
        //
        // Given permutation P that swaps a1 <-> a2:
        //   S1 = {x: a1, y: a2}
        //   S2 = {x: a2, y: a1} = permute(S1, P)
        //
        // Required invariant: S1.fingerprint_with_symmetry([P]) == S2.fingerprint_with_symmetry([P])
        //
        // BUG (pre-fix): TLA2 picked min(fp(S), fp(P(S))) instead of fp(lexmin(S, P(S)))
        // This violated the invariant because fingerprint order != lexicographic order.

        let perm = create_swap_permutation("a1", "a2");

        // Create symmetric states
        let s1 = State::from_pairs([
            ("x", Value::model_value("a1")),
            ("y", Value::model_value("a2")),
        ]);
        let s2 = State::from_pairs([
            ("x", Value::model_value("a2")),
            ("y", Value::model_value("a1")),
        ]);

        // Verify they are symmetric (one is the permutation of the other)
        let s1_permuted = s1.permute(&perm);
        assert_eq!(s1_permuted.vars, s2.vars, "S2 should equal permute(S1, P)");

        let s2_permuted = s2.permute(&perm);
        assert_eq!(s2_permuted.vars, s1.vars, "S1 should equal permute(S2, P)");

        // KEY INVARIANT: Symmetric states must have the same canonical fingerprint
        let perms = vec![perm];
        let fp1 = s1.fingerprint_with_symmetry(&perms);
        let fp2 = s2.fingerprint_with_symmetry(&perms);

        assert_eq!(
            fp1, fp2,
            "Symmetric states must have same canonical fingerprint!\n\
             S1 = {{x: a1, y: a2}}, fp = {:?}\n\
             S2 = {{x: a2, y: a1}}, fp = {:?}\n\
             These states are in the same orbit under the swap permutation.",
            fp1, fp2
        );
    }

    #[test]
    fn test_symmetry_three_element_orbit() {
        // Test with 3-element symmetry group (S3 style).
        // This matches PaxosCommit's 3-acceptor SYMMETRY which triggers the bug.
        //
        // Given cyclic permutation P: a1 -> a2 -> a3 -> a1
        // States S1, S2, S3 are all in the same orbit.

        let perm_cycle = create_cycle_permutation("a1", "a2", "a3");
        let perm_cycle2 = create_cycle_permutation("a1", "a3", "a2"); // inverse

        // Create three states in the same orbit
        let s1 = State::from_pairs([
            ("x", Value::model_value("a1")),
            ("y", Value::model_value("a2")),
            ("z", Value::model_value("a3")),
        ]);
        let s2 = State::from_pairs([
            ("x", Value::model_value("a2")),
            ("y", Value::model_value("a3")),
            ("z", Value::model_value("a1")),
        ]);
        let s3 = State::from_pairs([
            ("x", Value::model_value("a3")),
            ("y", Value::model_value("a1")),
            ("z", Value::model_value("a2")),
        ]);

        // Verify orbit relationship: S2 = P(S1), S3 = P(S2), S1 = P(S3)
        assert_eq!(s1.permute(&perm_cycle).vars, s2.vars);
        assert_eq!(s2.permute(&perm_cycle).vars, s3.vars);
        assert_eq!(s3.permute(&perm_cycle).vars, s1.vars);

        // Verify inverse relationship for perm_cycle2: S3 = P⁻¹(S1), S1 = P⁻¹(S2), S2 = P⁻¹(S3)
        assert_eq!(s1.permute(&perm_cycle2).vars, s3.vars);
        assert_eq!(s2.permute(&perm_cycle2).vars, s1.vars);
        assert_eq!(s3.permute(&perm_cycle2).vars, s2.vars);

        // All permutations in the group (6 elements for S3)
        let perms = vec![
            perm_cycle.clone(),
            perm_cycle2.clone(),
            // Full group would include: identity, (a1 a2), (a1 a3), (a2 a3)
            // For this test, using just the two cycles is sufficient
        ];

        let fp1 = s1.fingerprint_with_symmetry(&perms);
        let fp2 = s2.fingerprint_with_symmetry(&perms);
        let fp3 = s3.fingerprint_with_symmetry(&perms);

        assert_eq!(
            fp1, fp2,
            "States in same orbit must have same canonical fingerprint (S1 vs S2)"
        );
        assert_eq!(
            fp2, fp3,
            "States in same orbit must have same canonical fingerprint (S2 vs S3)"
        );
    }

    #[test]
    fn test_symmetry_different_orbits_different_fingerprints() {
        // States in DIFFERENT orbits should have DIFFERENT canonical fingerprints
        // (unless there's an accidental collision, which is extremely rare).

        let perm = create_swap_permutation("a1", "a2");

        // S1 and S2 are in the same orbit (a1, a2 swappable)
        let s1 = State::from_pairs([
            ("x", Value::model_value("a1")),
            ("y", Value::model_value("a2")),
        ]);
        let s2 = State::from_pairs([
            ("x", Value::model_value("a2")),
            ("y", Value::model_value("a1")),
        ]);

        // S3 is in a different orbit (both variables have same value)
        let s3 = State::from_pairs([
            ("x", Value::model_value("a1")),
            ("y", Value::model_value("a1")),
        ]);

        let perms = vec![perm];

        let fp1 = s1.fingerprint_with_symmetry(&perms);
        let fp2 = s2.fingerprint_with_symmetry(&perms);
        let fp3 = s3.fingerprint_with_symmetry(&perms);

        // Same orbit -> same fingerprint
        assert_eq!(fp1, fp2, "Same orbit must have same fingerprint");

        // Different orbit -> different fingerprint (with overwhelming probability)
        assert_ne!(
            fp1, fp3,
            "Different orbits should have different fingerprints"
        );
    }

    #[test]
    fn test_symmetry_empty_permutations() {
        // With no permutations, fingerprint_with_symmetry should return regular fingerprint
        let s = State::from_pairs([("x", Value::int(42))]);

        let fp_regular = s.fingerprint();
        let fp_symmetry = s.fingerprint_with_symmetry(&[]);

        assert_eq!(fp_regular, fp_symmetry);
    }

    #[test]
    fn test_symmetry_nested_structures() {
        // Test with nested values (sets containing model values)
        // This is closer to real PaxosCommit states with msets of acceptors.

        let perm = create_swap_permutation("a1", "a2");

        // Create sets containing model values
        let set1 = Value::set(vec![Value::model_value("a1")]);
        let set2 = Value::set(vec![Value::model_value("a2")]);

        let s1 = State::from_pairs([("votes", set1.clone()), ("decided", set2.clone())]);
        let s2 = State::from_pairs([("votes", set2), ("decided", set1)]);

        let perms = vec![perm];

        // Verify symmetry
        let s1_permuted = s1.permute(&perms[0]);
        assert_eq!(s1_permuted.vars, s2.vars, "S2 = permute(S1)");

        // Canonical fingerprints must match
        let fp1 = s1.fingerprint_with_symmetry(&perms);
        let fp2 = s2.fingerprint_with_symmetry(&perms);

        assert_eq!(
            fp1, fp2,
            "Symmetric states with nested structures must have same canonical fingerprint"
        );
    }
}
