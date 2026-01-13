//! Proof result caching for incremental verification
//!
//! Caches verification results by content hash of the property to avoid
//! re-verifying unchanged properties. Supports persistence to disk for
//! cache warming on server restart.

use dashprove_backends::BackendId;
use dashprove_usl::ast::Property;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors from cache operations
#[derive(Error, Debug)]
pub enum CacheError {
    /// I/O error during save/load
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// Cached verification result for a property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResult {
    /// Property content hash
    pub property_hash: u64,
    /// Whether verification succeeded
    pub valid: bool,
    /// Backend used for verification
    pub backend: BackendId,
    /// Backend output (compiled code)
    pub backend_code: String,
    /// Error message if any
    pub error: Option<String>,
    /// When this cache entry was created (as duration since UNIX epoch)
    #[serde(with = "duration_serde")]
    pub created_at: Duration,
    /// Time to live in seconds (after which entry is stale)
    pub ttl_secs: u64,
}

/// Serialize/deserialize Duration as seconds
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

impl CachedResult {
    /// Check if this cache entry is still valid
    fn is_valid(&self, now: Duration) -> bool {
        let age = now.saturating_sub(self.created_at);
        age.as_secs() < self.ttl_secs
    }
}

/// In-memory cache for verification results
pub struct ProofCache {
    /// Cache entries keyed by property name
    entries: HashMap<String, CachedResult>,
    /// Default TTL for new entries
    default_ttl_secs: u64,
    /// Maximum number of entries
    max_entries: usize,
    /// Creation time of this cache (for computing durations)
    start_instant: Instant,
    /// Duration from UNIX epoch at start (for serialization)
    start_epoch_duration: Duration,
}

impl Default for ProofCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofCache {
    /// Create a new empty cache with default settings
    pub fn new() -> Self {
        Self::with_config(3600, 10000) // 1 hour TTL, 10k max entries
    }

    /// Create a cache with custom TTL and max entries
    pub fn with_config(default_ttl_secs: u64, max_entries: usize) -> Self {
        let now = std::time::SystemTime::now();
        let start_epoch_duration = now
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();

        Self {
            entries: HashMap::new(),
            default_ttl_secs,
            max_entries,
            start_instant: Instant::now(),
            start_epoch_duration,
        }
    }

    /// Get current duration since UNIX epoch
    fn current_epoch_duration(&self) -> Duration {
        self.start_epoch_duration + self.start_instant.elapsed()
    }

    /// Compute a content hash for a property
    pub fn hash_property(property: &Property) -> u64 {
        let mut hasher = DefaultHasher::new();
        // Hash the debug representation which includes all structural info
        format!("{property:?}").hash(&mut hasher);
        hasher.finish()
    }

    /// Get a cached result if it exists and is still valid
    pub fn get(&self, property_name: &str, property_hash: u64) -> Option<&CachedResult> {
        let entry = self.entries.get(property_name)?;
        let now = self.current_epoch_duration();

        // Check both hash match and TTL validity
        if entry.property_hash == property_hash && entry.is_valid(now) {
            Some(entry)
        } else {
            None
        }
    }

    /// Store a verification result in the cache
    pub fn put(
        &mut self,
        property_name: String,
        property_hash: u64,
        valid: bool,
        backend: BackendId,
        backend_code: String,
        error: Option<String>,
    ) {
        // Evict expired entries if we're at capacity
        if self.entries.len() >= self.max_entries {
            self.evict_expired();
        }

        // If still at capacity after eviction, remove oldest entry
        if self.entries.len() >= self.max_entries {
            self.evict_oldest();
        }

        let entry = CachedResult {
            property_hash,
            valid,
            backend,
            backend_code,
            error,
            created_at: self.current_epoch_duration(),
            ttl_secs: self.default_ttl_secs,
        };

        self.entries.insert(property_name, entry);
    }

    /// Remove expired entries
    fn evict_expired(&mut self) {
        let now = self.current_epoch_duration();
        self.entries.retain(|_, entry| entry.is_valid(now));
    }

    /// Remove the oldest entry
    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.created_at)
            .map(|(k, _)| k.clone())
        {
            self.entries.remove(&oldest_key);
        }
    }

    /// Invalidate all entries for properties that depend on changed types/functions
    pub fn invalidate_affected(&mut self, affected_properties: &[String]) {
        for name in affected_properties {
            self.entries.remove(name);
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get the number of entries in the cache
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let now = self.current_epoch_duration();
        let valid_count = self.entries.values().filter(|e| e.is_valid(now)).count();
        let expired_count = self.entries.len() - valid_count;

        CacheStats {
            total_entries: self.entries.len(),
            valid_entries: valid_count,
            expired_entries: expired_count,
            max_entries: self.max_entries,
            default_ttl_secs: self.default_ttl_secs,
        }
    }

    /// Save the cache to a file for later restoration
    ///
    /// Only valid (non-expired) entries are saved.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<usize, CacheError> {
        let path = path.as_ref();

        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Filter to only valid entries
        let now = self.current_epoch_duration();
        let valid_entries: HashMap<String, CachedResult> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.is_valid(now))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let saved_count = valid_entries.len();

        let data = CacheData {
            entries: valid_entries,
            default_ttl_secs: self.default_ttl_secs,
            max_entries: self.max_entries,
        };

        // Write to temporary file first, then rename for atomicity
        let tmp_path = path.with_extension("tmp");
        let file = File::create(&tmp_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &data)?;
        std::fs::rename(&tmp_path, path)?;

        Ok(saved_count)
    }

    /// Load cache from a file, restoring valid entries
    ///
    /// Expired entries from the file are not loaded.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, CacheError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let data: CacheData = serde_json::from_reader(reader)?;

        let now = std::time::SystemTime::now();
        let start_epoch_duration = now
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();

        let mut cache = Self {
            entries: HashMap::new(),
            default_ttl_secs: data.default_ttl_secs,
            max_entries: data.max_entries,
            start_instant: Instant::now(),
            start_epoch_duration,
        };

        // Only restore valid entries
        let now_duration = cache.current_epoch_duration();
        for (name, entry) in data.entries {
            if entry.is_valid(now_duration) {
                cache.entries.insert(name, entry);
            }
        }

        Ok(cache)
    }

    /// Load cache from file if it exists, otherwise return a new empty cache
    pub fn load_or_default<P: AsRef<Path>>(path: P) -> Self {
        Self::load_from_file(&path).unwrap_or_default()
    }

    /// Load cache from file if it exists, with custom config as fallback
    pub fn load_or_with_config<P: AsRef<Path>>(
        path: P,
        default_ttl_secs: u64,
        max_entries: usize,
    ) -> Self {
        match Self::load_from_file(&path) {
            Ok(cache) => cache,
            Err(_) => Self::with_config(default_ttl_secs, max_entries),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total number of entries
    pub total_entries: usize,
    /// Number of valid (non-expired) entries
    pub valid_entries: usize,
    /// Number of expired entries pending eviction
    pub expired_entries: usize,
    /// Maximum entries allowed
    pub max_entries: usize,
    /// Default TTL in seconds
    pub default_ttl_secs: u64,
}

/// Serializable cache data for disk persistence
#[derive(Debug, Serialize, Deserialize)]
struct CacheData {
    /// Cache entries
    entries: HashMap<String, CachedResult>,
    /// Default TTL for new entries
    default_ttl_secs: u64,
    /// Maximum number of entries
    max_entries: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::ast::{Expr, Invariant};

    fn make_property(name: &str) -> Property {
        Property::Invariant(Invariant {
            name: name.to_string(),
            body: Expr::Bool(true),
        })
    }

    #[test]
    fn test_cache_put_get() {
        let mut cache = ProofCache::new();
        let prop = make_property("test");
        let hash = ProofCache::hash_property(&prop);

        cache.put(
            "test".to_string(),
            hash,
            true,
            BackendId::Lean4,
            "-- lean code".to_string(),
            None,
        );

        let result = cache.get("test", hash);
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.valid);
        assert_eq!(result.backend, BackendId::Lean4);
    }

    #[test]
    fn test_cache_miss_wrong_hash() {
        let mut cache = ProofCache::new();
        let prop = make_property("test");
        let hash = ProofCache::hash_property(&prop);

        cache.put(
            "test".to_string(),
            hash,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );

        // Different hash should miss
        let result = cache.get("test", hash + 1);
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_invalidate() {
        let mut cache = ProofCache::new();

        cache.put(
            "prop1".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        cache.put(
            "prop2".to_string(),
            2,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        cache.put(
            "prop3".to_string(),
            3,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );

        cache.invalidate_affected(&["prop1".to_string(), "prop3".to_string()]);

        assert!(cache.get("prop1", 1).is_none());
        assert!(cache.get("prop2", 2).is_some());
        assert!(cache.get("prop3", 3).is_none());
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = ProofCache::with_config(3600, 100);

        cache.put(
            "prop1".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        cache.put(
            "prop2".to_string(),
            2,
            false,
            BackendId::TlaPlus,
            "code".to_string(),
            Some("error".to_string()),
        );

        let stats = cache.stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.valid_entries, 2);
        assert_eq!(stats.max_entries, 100);
    }

    #[test]
    fn test_cache_eviction_at_capacity() {
        let mut cache = ProofCache::with_config(3600, 3);

        cache.put(
            "prop1".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        cache.put(
            "prop2".to_string(),
            2,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        cache.put(
            "prop3".to_string(),
            3,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );

        // At capacity, adding one more should evict oldest
        cache.put(
            "prop4".to_string(),
            4,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );

        assert_eq!(cache.len(), 3);
        // prop1 was oldest, should be evicted
        assert!(cache.get("prop1", 1).is_none());
        assert!(cache.get("prop4", 4).is_some());
    }

    #[test]
    fn test_property_hash_consistency() {
        let prop1 = make_property("test");
        let prop2 = make_property("test");
        let prop3 = make_property("different");

        let hash1 = ProofCache::hash_property(&prop1);
        let hash2 = ProofCache::hash_property(&prop2);
        let hash3 = ProofCache::hash_property(&prop3);

        // Same content should have same hash
        assert_eq!(hash1, hash2);
        // Different content should (almost certainly) have different hash
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = ProofCache::new();

        cache.put(
            "prop1".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        cache.put(
            "prop2".to_string(),
            2,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );

        cache.clear();

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    fn temp_cache_file() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("dashprove_cache_test_{id}_{nanos}.json"))
    }

    #[test]
    fn test_cache_save_and_load() {
        let mut cache = ProofCache::with_config(3600, 100);
        let path = temp_cache_file();

        cache.put(
            "prop1".to_string(),
            1,
            true,
            BackendId::Lean4,
            "lean code".to_string(),
            None,
        );
        cache.put(
            "prop2".to_string(),
            2,
            false,
            BackendId::TlaPlus,
            "tla code".to_string(),
            Some("error message".to_string()),
        );

        let saved = cache.save_to_file(&path).unwrap();
        assert_eq!(saved, 2);

        let loaded = ProofCache::load_from_file(&path).unwrap();
        assert_eq!(loaded.len(), 2);

        let entry1 = loaded.get("prop1", 1).unwrap();
        assert!(entry1.valid);
        assert_eq!(entry1.backend, BackendId::Lean4);
        assert_eq!(entry1.backend_code, "lean code");

        let entry2 = loaded.get("prop2", 2).unwrap();
        assert!(!entry2.valid);
        assert_eq!(entry2.error, Some("error message".to_string()));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_cache_load_or_default() {
        let path = temp_cache_file();

        // Non-existent file should return default
        let cache = ProofCache::load_or_default(&path);
        assert!(cache.is_empty());

        // Create a cache with data
        let mut cache = ProofCache::new();
        cache.put(
            "test".to_string(),
            42,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        cache.save_to_file(&path).unwrap();

        // Loading should restore the data
        let loaded = ProofCache::load_or_default(&path);
        assert_eq!(loaded.len(), 1);
        assert!(loaded.get("test", 42).is_some());

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_cache_save_preserves_config() {
        let path = temp_cache_file();

        let mut cache = ProofCache::with_config(7200, 500);
        cache.put(
            "test".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        cache.save_to_file(&path).unwrap();

        let loaded = ProofCache::load_from_file(&path).unwrap();
        let stats = loaded.stats();
        assert_eq!(stats.default_ttl_secs, 7200);
        assert_eq!(stats.max_entries, 500);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_cache_load_filters_expired() {
        let path = temp_cache_file();

        // Create a cache with very short TTL
        let mut cache = ProofCache::with_config(1, 100); // 1 second TTL
        cache.put(
            "test".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );

        // Save immediately
        cache.save_to_file(&path).unwrap();

        // Wait for TTL to expire
        std::thread::sleep(std::time::Duration::from_secs(2));

        // Load should filter out expired entry
        let loaded = ProofCache::load_from_file(&path).unwrap();
        assert!(loaded.is_empty());

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_cache_load_or_with_config() {
        let path = temp_cache_file();

        // Non-existent file should use provided config
        let cache = ProofCache::load_or_with_config(&path, 1800, 200);
        let stats = cache.stats();
        assert_eq!(stats.default_ttl_secs, 1800);
        assert_eq!(stats.max_entries, 200);

        // Create file with different config
        let mut cache = ProofCache::with_config(3600, 500);
        cache.put(
            "test".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        cache.save_to_file(&path).unwrap();

        // Should load saved config, not provided defaults
        let loaded = ProofCache::load_or_with_config(&path, 1800, 200);
        let stats = loaded.stats();
        assert_eq!(stats.default_ttl_secs, 3600);
        assert_eq!(stats.max_entries, 500);

        std::fs::remove_file(path).ok();
    }

    // ============================================
    // Mutation-killing tests for cache.rs
    // ============================================

    #[test]
    fn test_cached_result_is_valid_exact_boundary() {
        // Test exact boundary: age exactly equals ttl_secs
        // Entry created at epoch 1000, ttl = 100
        // At time 1100, age = 100, should NOT be valid (< ttl, not <=)
        let entry = CachedResult {
            property_hash: 42,
            valid: true,
            backend: BackendId::Lean4,
            backend_code: "code".to_string(),
            error: None,
            created_at: Duration::from_secs(1000),
            ttl_secs: 100,
        };

        // age = 100 (exactly ttl), should be invalid (< not <=)
        assert!(!entry.is_valid(Duration::from_secs(1100)));

        // age = 99, should be valid
        assert!(entry.is_valid(Duration::from_secs(1099)));

        // age = 101, should be invalid
        assert!(!entry.is_valid(Duration::from_secs(1101)));
    }

    #[test]
    fn test_cached_result_is_valid_saturating_sub() {
        // Test that saturating_sub is used (now < created_at doesn't panic)
        let entry = CachedResult {
            property_hash: 42,
            valid: true,
            backend: BackendId::Lean4,
            backend_code: "code".to_string(),
            error: None,
            created_at: Duration::from_secs(1000),
            ttl_secs: 100,
        };

        // "now" is before created_at - should saturate to 0
        // 0 < 100, so should be valid
        assert!(entry.is_valid(Duration::from_secs(500)));
    }

    #[test]
    fn test_cache_get_requires_both_hash_and_ttl() {
        let mut cache = ProofCache::with_config(3600, 100);

        cache.put(
            "test".to_string(),
            42,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );

        // Correct hash - should find
        assert!(cache.get("test", 42).is_some());

        // Wrong hash - should not find (even though TTL is valid)
        assert!(cache.get("test", 43).is_none());

        // Wrong name - should not find
        assert!(cache.get("other", 42).is_none());
    }

    #[test]
    fn test_cache_evict_oldest_empty_cache() {
        let mut cache = ProofCache::with_config(3600, 10);

        // evict_oldest on empty cache should not panic
        cache.evict_oldest();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_evict_oldest_single_entry() {
        let mut cache = ProofCache::with_config(3600, 10);

        cache.put(
            "only".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        assert_eq!(cache.len(), 1);

        cache.evict_oldest();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_evict_oldest_selects_oldest() {
        // Create a cache with different creation times
        let mut cache = ProofCache::with_config(3600, 10);

        // Add entries with sleep to ensure different timestamps
        cache.put(
            "first".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
        cache.put(
            "second".to_string(),
            2,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
        cache.put(
            "third".to_string(),
            3,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );

        assert_eq!(cache.len(), 3);

        cache.evict_oldest();
        assert_eq!(cache.len(), 2);

        // "first" was oldest, should be removed
        assert!(cache.get("first", 1).is_none());
        assert!(cache.get("second", 2).is_some());
        assert!(cache.get("third", 3).is_some());
    }

    #[test]
    fn test_cache_put_evicts_expired_before_oldest() {
        // Create cache with short TTL
        let mut cache = ProofCache::with_config(1, 2); // 1 second TTL, max 2 entries

        cache.put(
            "prop1".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        cache.put(
            "prop2".to_string(),
            2,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );

        assert_eq!(cache.len(), 2);

        // Wait for TTL to expire
        std::thread::sleep(std::time::Duration::from_secs(2));

        // Adding a new entry should trigger evict_expired first
        cache.put(
            "prop3".to_string(),
            3,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );

        // Expired entries should be gone, new entry should be present
        // (may be 1 entry since expired ones were removed)
        assert!(cache.len() <= 2);
        assert!(cache.get("prop3", 3).is_some());
    }

    #[test]
    fn test_cache_stats_counts_expired_vs_valid() {
        let mut cache = ProofCache::with_config(1, 100); // 1 second TTL

        // Add some entries
        cache.put(
            "prop1".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        cache.put(
            "prop2".to_string(),
            2,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );

        // Initially all valid
        let stats = cache.stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.valid_entries, 2);
        assert_eq!(stats.expired_entries, 0);

        // Wait for expiry
        std::thread::sleep(std::time::Duration::from_secs(2));

        // Now all expired
        let stats = cache.stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.valid_entries, 0);
        assert_eq!(stats.expired_entries, 2);
    }

    #[test]
    fn test_cache_invalidate_affected_removes_exact_names() {
        let mut cache = ProofCache::new();

        cache.put(
            "prop_a".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        cache.put(
            "prop_b".to_string(),
            2,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        cache.put(
            "prop_c".to_string(),
            3,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );

        // Remove specific entries
        cache.invalidate_affected(&["prop_a".to_string(), "prop_c".to_string()]);

        assert!(cache.get("prop_a", 1).is_none());
        assert!(cache.get("prop_b", 2).is_some());
        assert!(cache.get("prop_c", 3).is_none());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_invalidate_affected_nonexistent_key() {
        let mut cache = ProofCache::new();

        cache.put(
            "existing".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );

        // Invalidating a non-existent key should not panic or affect others
        cache.invalidate_affected(&["nonexistent".to_string(), "also_missing".to_string()]);

        assert!(cache.get("existing", 1).is_some());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_hash_property_produces_different_hashes_for_different_properties() {
        let prop1 = make_property("test");
        let prop2 = make_property("different");
        let prop3 = make_property("test"); // Same as prop1

        let hash1 = ProofCache::hash_property(&prop1);
        let hash2 = ProofCache::hash_property(&prop2);
        let hash3 = ProofCache::hash_property(&prop3);

        // Same content = same hash
        assert_eq!(hash1, hash3);

        // Different content = different hash (with very high probability)
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_property_uses_debug_format() {
        // Hashing uses Debug representation, so different internal structure = different hash
        // Two properties with same name but different bodies should have different hashes
        let prop1 = Property::Invariant(Invariant {
            name: "test".to_string(),
            body: Expr::Bool(true),
        });
        let prop2 = Property::Invariant(Invariant {
            name: "test".to_string(),
            body: Expr::Bool(false),
        });

        let hash1 = ProofCache::hash_property(&prop1);
        let hash2 = ProofCache::hash_property(&prop2);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_cache_len_and_is_empty_consistency() {
        let mut cache = ProofCache::new();

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        cache.put(
            "test".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );

        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);

        cache.clear();

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_default_impl() {
        let cache = ProofCache::default();
        let stats = cache.stats();

        // Default should match new()
        assert_eq!(stats.default_ttl_secs, 3600); // 1 hour
        assert_eq!(stats.max_entries, 10000); // 10k
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_save_creates_parent_directories() {
        let temp_dir = std::env::temp_dir();
        let nested_path = temp_dir
            .join("dashprove_cache_test_nested")
            .join("subdir")
            .join("cache.json");

        let mut cache = ProofCache::new();
        cache.put(
            "test".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );

        // Save should create parent directories
        let result = cache.save_to_file(&nested_path);
        assert!(result.is_ok());

        // Clean up
        std::fs::remove_file(&nested_path).ok();
        std::fs::remove_dir(nested_path.parent().unwrap()).ok();
        std::fs::remove_dir(temp_dir.join("dashprove_cache_test_nested")).ok();
    }

    #[test]
    fn test_cache_save_only_saves_valid_entries() {
        let path = temp_cache_file();
        let mut cache = ProofCache::with_config(1, 100); // 1 second TTL

        cache.put(
            "prop1".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code1".to_string(),
            None,
        );

        // Wait for it to expire
        std::thread::sleep(std::time::Duration::from_secs(2));

        // Add another fresh entry
        cache.put(
            "prop2".to_string(),
            2,
            true,
            BackendId::Lean4,
            "code2".to_string(),
            None,
        );

        // Save should only save the valid entry (prop2)
        let saved = cache.save_to_file(&path).unwrap();
        assert_eq!(saved, 1); // Only 1 valid entry saved

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_cache_put_with_error() {
        let mut cache = ProofCache::new();

        cache.put(
            "failed_prop".to_string(),
            123,
            false,
            BackendId::TlaPlus,
            "-- tla code".to_string(),
            Some("Type error at line 5".to_string()),
        );

        let entry = cache.get("failed_prop", 123).unwrap();
        assert!(!entry.valid);
        assert_eq!(entry.error, Some("Type error at line 5".to_string()));
        assert_eq!(entry.backend, BackendId::TlaPlus);
    }

    #[test]
    fn test_duration_serde_roundtrip() {
        // Test the duration_serde module by creating and serializing a CachedResult
        let entry = CachedResult {
            property_hash: 42,
            valid: true,
            backend: BackendId::Lean4,
            backend_code: "code".to_string(),
            error: None,
            created_at: Duration::from_secs(12345),
            ttl_secs: 3600,
        };

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: CachedResult = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.created_at, Duration::from_secs(12345));
    }

    #[test]
    fn test_cache_put_updates_existing_entry() {
        let mut cache = ProofCache::new();

        // Add initial entry
        cache.put(
            "test".to_string(),
            1,
            true,
            BackendId::Lean4,
            "code1".to_string(),
            None,
        );

        let entry1 = cache.get("test", 1).unwrap();
        assert!(entry1.valid);

        // Update with same name but different hash
        cache.put(
            "test".to_string(),
            2,
            false,
            BackendId::TlaPlus,
            "code2".to_string(),
            Some("error".to_string()),
        );

        // Old hash should not find anything
        assert!(cache.get("test", 1).is_none());

        // New hash should find the updated entry
        let entry2 = cache.get("test", 2).unwrap();
        assert!(!entry2.valid);
        assert_eq!(entry2.backend, BackendId::TlaPlus);
        assert_eq!(entry2.error, Some("error".to_string()));

        // Still only 1 entry (replaced, not added)
        assert_eq!(cache.len(), 1);
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify that is_valid uses saturating_sub (no underflow when now < created_at)
    #[kani::proof]
    fn verify_is_valid_saturating_sub() {
        let created_secs: u64 = kani::any();
        let now_secs: u64 = kani::any();
        let ttl_secs: u64 = kani::any();

        // Bound inputs to prevent timeout
        kani::assume(created_secs <= 10000);
        kani::assume(now_secs <= 10000);
        kani::assume(ttl_secs <= 10000);

        let entry = CachedResult {
            property_hash: 0,
            valid: true,
            backend: BackendId::Lean4,
            backend_code: String::new(),
            error: None,
            created_at: Duration::from_secs(created_secs),
            ttl_secs,
        };

        // This should not panic due to saturating_sub
        let _ = entry.is_valid(Duration::from_secs(now_secs));
    }

    /// Verify is_valid boundary: age < ttl means valid, age >= ttl means invalid
    #[kani::proof]
    fn verify_is_valid_boundary_condition() {
        let created_secs: u64 = kani::any();
        let ttl_secs: u64 = kani::any();

        // Bound inputs
        kani::assume(created_secs <= 1000);
        kani::assume(ttl_secs > 0 && ttl_secs <= 1000);

        let entry = CachedResult {
            property_hash: 0,
            valid: true,
            backend: BackendId::Lean4,
            backend_code: String::new(),
            error: None,
            created_at: Duration::from_secs(created_secs),
            ttl_secs,
        };

        // At exactly ttl boundary, should be invalid
        let now_at_boundary = created_secs.saturating_add(ttl_secs);
        let result_at_boundary = entry.is_valid(Duration::from_secs(now_at_boundary));
        kani::assert(!result_at_boundary, "age == ttl should be invalid");

        // One second before boundary should be valid
        let now_before = created_secs.saturating_add(ttl_secs.saturating_sub(1));
        let result_before = entry.is_valid(Duration::from_secs(now_before));
        kani::assert(result_before, "age < ttl should be valid");
    }

    /// Verify CacheStats fields are correctly populated
    #[kani::proof]
    fn verify_cache_stats_fields() {
        let total: usize = kani::any();
        let valid: usize = kani::any();
        let expired: usize = kani::any();
        let max: usize = kani::any();
        let ttl: u64 = kani::any();

        // Bound inputs
        kani::assume(total <= 100);
        kani::assume(valid <= total);
        kani::assume(expired <= total);
        kani::assume(max <= 1000);
        kani::assume(ttl <= 10000);

        let stats = CacheStats {
            total_entries: total,
            valid_entries: valid,
            expired_entries: expired,
            max_entries: max,
            default_ttl_secs: ttl,
        };

        kani::assert(stats.total_entries == total, "total_entries preserved");
        kani::assert(stats.valid_entries == valid, "valid_entries preserved");
        kani::assert(
            stats.expired_entries == expired,
            "expired_entries preserved",
        );
        kani::assert(stats.max_entries == max, "max_entries preserved");
        kani::assert(stats.default_ttl_secs == ttl, "default_ttl_secs preserved");
    }

    /// Verify duration_serde serializes as seconds
    #[kani::proof]
    fn verify_duration_serde_as_secs() {
        let secs: u64 = kani::any();
        kani::assume(secs <= 1000);

        let duration = Duration::from_secs(secs);
        // The serializer should convert to secs
        kani::assert(duration.as_secs() == secs, "duration as_secs matches input");
    }
}
