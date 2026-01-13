//! Name representation
//!
//! Hierarchical names like `Nat.add` or `List.map`.
//!
//! This module provides optional name interning for performance-critical paths
//! like .olean loading where the same names are constructed many times.
//!
//! # Hash Caching
//!
//! Name hashing is optimized by computing and caching the hash value when
//! the Name is created. This avoids repeated recursive traversal during
//! HashMap lookups, which is a significant performance win for large
//! environments with many constants.

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::str::FromStr;
use std::sync::{Arc, OnceLock, RwLock};
use std::thread_local;

/// Global name interner for deduplicating Name allocations.
/// Uses a read-write lock for thread-safe caching.
static NAME_INTERNER: OnceLock<NameInterner> = OnceLock::new();

thread_local! {
    // Thread-local cache to avoid global lock contention during parallel imports.
    static TLS_NAME_CACHE: RefCell<HashMap<String, Arc<Name>>> =
        RefCell::new(HashMap::new());
}

/// Thread-safe name interner that caches Name instances by their string representation.
/// This significantly reduces allocations when parsing .olean files where the same
/// names (like "Nat", "Nat.add", etc.) appear thousands of times.
pub struct NameInterner {
    cache: RwLock<HashMap<String, Arc<Name>>>,
}

impl NameInterner {
    /// Create a new empty interner
    fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Get the global interner instance
    pub fn global() -> &'static Self {
        NAME_INTERNER.get_or_init(NameInterner::new)
    }

    /// Intern a name from a dotted string like "Nat.add".
    /// Returns an Arc to a cached Name if one exists, otherwise creates and caches a new one.
    pub fn intern(&self, s: &str) -> Arc<Name> {
        if let Some(name) = TLS_NAME_CACHE.with(|cache| cache.borrow().get(s).cloned()) {
            return name;
        }

        // Fast path: check read lock first
        if let Some(name) = self
            .cache
            .read()
            .expect("name interner lock poisoned")
            .get(s)
            .cloned()
        {
            TLS_NAME_CACHE.with(|cache| {
                cache.borrow_mut().insert(s.to_string(), name.clone());
            });
            return name;
        }

        // Slow path: acquire write lock and insert
        let owned = s.to_string();
        let new_name = Arc::new(Name::from_string_uncached(&owned));
        let interned = {
            let mut cache = self.cache.write().expect("name interner lock poisoned");
            cache
                .entry(owned.clone())
                .or_insert_with(|| Arc::clone(&new_name))
                .clone()
        };
        TLS_NAME_CACHE.with(|cache| {
            cache.borrow_mut().insert(owned, interned.clone());
        });
        interned
    }

    /// Intern a name, returning the Name directly (not Arc).
    /// Clones from the cache, which is cheap for Arc-based Names.
    pub fn intern_name(&self, s: &str) -> Name {
        (*self.intern(s)).clone()
    }

    /// Clear the interner cache (global plus the current thread's cache).
    pub fn clear(&self) {
        let mut cache = self.cache.write().expect("name interner lock poisoned");
        cache.clear();
        TLS_NAME_CACHE.with(|cache| cache.borrow_mut().clear());
    }

    /// Get the number of cached names
    pub fn len(&self) -> usize {
        self.cache
            .read()
            .expect("name interner lock poisoned")
            .len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Inner representation of a hierarchical name.
/// This is the recursive structure that forms the name tree.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NameInner {
    /// Anonymous name
    Anon,
    /// String component
    Str(Arc<Name>, Arc<str>),
    /// Numeric component (for auto-generated names)
    Num(Arc<Name>, u64),
}

/// Hierarchical name with cached hash.
///
/// Names like `Nat.add` or `List.map` are represented as a tree of
/// string and numeric components. The hash is computed once at creation
/// time and cached for O(1) HashMap operations.
#[derive(Clone, Debug)]
pub struct Name {
    inner: NameInner,
    /// Cached hash value, computed at creation time
    cached_hash: u64,
}

// Custom Serialize: only serialize the inner value
impl Serialize for Name {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.inner.serialize(serializer)
    }
}

// Custom Deserialize: deserialize inner and recompute hash
impl<'de> Deserialize<'de> for Name {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let inner = NameInner::deserialize(deserializer)?;
        Ok(Self::from_inner(inner))
    }
}

impl Name {
    /// Compute the hash for a NameInner recursively (used only at construction time)
    fn compute_hash(inner: &NameInner) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        inner.hash(&mut hasher);
        hasher.finish()
    }

    /// Create a Name from a NameInner, computing the hash
    fn from_inner(inner: NameInner) -> Self {
        let cached_hash = Self::compute_hash(&inner);
        Name { inner, cached_hash }
    }
}

impl PartialEq for Name {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: if hashes differ, names differ
        if self.cached_hash != other.cached_hash {
            return false;
        }
        // Hashes match, need full comparison
        self.inner == other.inner
    }
}

impl Eq for Name {}

impl Hash for Name {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // O(1) hash using cached value
        self.cached_hash.hash(state);
    }
}

impl Name {
    /// Create anonymous name
    pub fn anon() -> Self {
        Self::from_inner(NameInner::Anon)
    }

    /// Append a string component
    #[must_use]
    pub fn str(self, s: impl AsRef<str>) -> Self {
        Self::from_inner(NameInner::Str(Arc::new(self), Arc::from(s.as_ref())))
    }

    /// Append a numeric component
    #[must_use]
    pub fn num(self, n: u64) -> Self {
        Self::from_inner(NameInner::Num(Arc::new(self), n))
    }

    /// Check if this is the anonymous name
    pub fn is_anon(&self) -> bool {
        matches!(self.inner, NameInner::Anon)
    }

    /// Get the inner representation (for pattern matching)
    #[inline]
    pub fn inner(&self) -> &NameInner {
        &self.inner
    }

    /// Create from a dotted string like "Nat.add" (uncached - always allocates)
    /// For high-throughput parsing, prefer `Name::interned()` which uses caching.
    fn from_string_uncached(s: &str) -> Self {
        s.split('.').fold(Name::anon(), |acc, part| {
            if let Ok(n) = part.parse::<u64>() {
                acc.num(n)
            } else {
                acc.str(part)
            }
        })
    }

    /// Create from a dotted string like "Nat.add"
    /// Convenience wrapper around FromStr implementation
    #[inline]
    pub fn from_string(s: &str) -> Self {
        s.parse().expect("Name::from_str is infallible")
    }

    /// Create from a dotted string using the global interner.
    /// This is more efficient when the same names are created many times,
    /// as it returns a clone of a cached Name.
    #[inline]
    pub fn interned(s: &str) -> Self {
        NameInterner::global().intern_name(s)
    }

    /// Create from a dotted string using the global interner, returning `Arc<Name>`.
    /// Most efficient for repeated use since it avoids even the clone.
    #[inline]
    pub fn interned_arc(s: &str) -> Arc<Name> {
        NameInterner::global().intern(s)
    }
}

impl FromStr for Name {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.split('.').fold(Name::anon(), |acc, part| {
            if let Ok(n) = part.parse::<u64>() {
                acc.num(n)
            } else {
                acc.str(part)
            }
        }))
    }
}

impl std::fmt::Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            NameInner::Anon => write!(f, "[anonymous]"),
            NameInner::Str(prefix, s) => {
                if prefix.is_anon() {
                    write!(f, "{s}")
                } else {
                    write!(f, "{prefix}.{s}")
                }
            }
            NameInner::Num(prefix, n) => {
                if prefix.is_anon() {
                    write!(f, "{n}")
                } else {
                    write!(f, "{prefix}.{n}")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_from_str() {
        let name: Name = "Nat.add".parse().unwrap();
        assert_eq!(name.to_string(), "Nat.add");
    }

    #[test]
    fn test_name_interned() {
        let name1 = Name::interned("Nat.add");
        let name2 = Name::interned("Nat.add");
        assert_eq!(name1, name2);
        assert_eq!(name1.to_string(), "Nat.add");
    }

    #[test]
    fn test_name_interner_arc_reuse() {
        // Get two Arc<Name> for the same string
        let arc1 = Name::interned_arc("List.map");
        let arc2 = Name::interned_arc("List.map");
        // They should point to the same allocation
        assert!(Arc::ptr_eq(&arc1, &arc2));
    }

    #[test]
    fn test_interner_caches_names() {
        // Use a UUID-like unique name to avoid collision with parallel tests
        let unique_name = format!(
            "test.unique.name.{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        // First intern should add to cache
        let arc1 = Name::interned_arc(&unique_name);

        // Second intern of same name should return same Arc
        let arc2 = Name::interned_arc(&unique_name);

        // Verify they point to the same allocation (proves caching works)
        assert!(Arc::ptr_eq(&arc1, &arc2));
    }

    #[test]
    fn test_interner_numeric_components() {
        let name = Name::interned("Nat._root_.1.2.3");
        assert_eq!(name.to_string(), "Nat._root_.1.2.3");
    }

    #[test]
    fn test_thread_local_cache_shares_entries_across_threads() {
        // Intern in a background thread
        let handle = std::thread::spawn(|| Name::interned_arc("ThreadLocal.test.name"));
        let from_thread = handle.join().unwrap();

        // Main thread should reuse the same allocation
        let from_main = Name::interned_arc("ThreadLocal.test.name");
        assert!(Arc::ptr_eq(&from_thread, &from_main));
    }
}
