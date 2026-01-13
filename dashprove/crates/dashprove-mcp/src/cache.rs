//! Verification result caching for DashProve MCP
//!
//! Provides a thread-safe cache for verification results to avoid re-computing
//! identical verifications. Supports TTL, max size limits, LRU eviction, and
//! persistence (save/load to disk).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use tokio::sync::{oneshot, RwLock};
use tokio::task::JoinHandle;
use tracing::warn;

/// Default cache time-to-live (5 minutes)
pub const DEFAULT_TTL_SECS: u64 = 300;

/// Default maximum cache entries
pub const DEFAULT_MAX_ENTRIES: usize = 1000;

/// Cache key for verification results
///
/// The key is computed from the normalized spec content, backend list, and strategy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Hash of the normalized spec content
    spec_hash: u64,
    /// Sorted list of backend names
    backends: Vec<String>,
    /// Verification strategy
    strategy: String,
    /// Whether typecheck only
    typecheck_only: bool,
}

impl CacheKey {
    /// Create a new cache key from verification parameters
    pub fn new(spec: &str, backends: &[String], strategy: &str, typecheck_only: bool) -> Self {
        // Normalize and hash the spec content
        let normalized = spec.trim();
        let spec_hash = Self::hash_string(normalized);

        // Sort backends for consistent key generation
        let mut backends: Vec<String> = backends.iter().map(|s| s.to_lowercase()).collect();
        backends.sort();

        Self {
            spec_hash,
            backends,
            strategy: strategy.to_lowercase(),
            typecheck_only,
        }
    }

    /// Hash a string using a fast non-cryptographic hash
    fn hash_string(s: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }
}

/// Cached verification entry
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    /// The cached result
    pub result: T,
    /// When this entry was created
    pub created_at: Instant,
    /// Last access time (for LRU eviction)
    pub last_accessed: Instant,
    /// Number of times this entry was hit
    pub hit_count: u64,
}

impl<T: Clone> CacheEntry<T> {
    /// Create a new cache entry
    pub fn new(result: T) -> Self {
        let now = Instant::now();
        Self {
            result,
            created_at: now,
            last_accessed: now,
            hit_count: 0,
        }
    }

    /// Check if this entry has expired
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }

    /// Record a cache hit
    pub fn record_hit(&mut self) {
        self.last_accessed = Instant::now();
        self.hit_count += 1;
    }

    /// Get age in seconds
    pub fn age_secs(&self) -> u64 {
        self.created_at.elapsed().as_secs()
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Current number of entries
    pub entries: usize,
    /// Number of entries evicted
    pub evictions: u64,
    /// Number of expired entries removed
    pub expirations: u64,
}

impl CacheStats {
    /// Calculate hit rate as a percentage
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Time-to-live for cache entries in seconds
    #[serde(with = "duration_secs")]
    pub ttl: Duration,
    /// Maximum number of entries
    pub max_entries: usize,
    /// Whether caching is enabled
    pub enabled: bool,
}

/// Serde helper for Duration as seconds
mod duration_secs {
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

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(DEFAULT_TTL_SECS),
            max_entries: DEFAULT_MAX_ENTRIES,
            enabled: true,
        }
    }
}

// ============================================================================
// Persistence support
// ============================================================================

/// Error type for cache persistence operations
#[derive(Debug)]
pub enum CachePersistenceError {
    /// IO error during file operations
    Io(std::io::Error),
    /// Serialization/deserialization error
    Serde(serde_json::Error),
}

impl std::fmt::Display for CachePersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Serde(e) => write!(f, "Serialization error: {}", e),
        }
    }
}

impl std::error::Error for CachePersistenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Serde(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for CachePersistenceError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for CachePersistenceError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serde(e)
    }
}

/// Serializable cache key for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableCacheKey {
    /// Hash of the normalized spec content
    pub spec_hash: u64,
    /// Sorted list of backend names
    pub backends: Vec<String>,
    /// Verification strategy
    pub strategy: String,
    /// Whether typecheck only
    pub typecheck_only: bool,
}

impl From<&CacheKey> for SerializableCacheKey {
    fn from(key: &CacheKey) -> Self {
        Self {
            spec_hash: key.spec_hash,
            backends: key.backends.clone(),
            strategy: key.strategy.clone(),
            typecheck_only: key.typecheck_only,
        }
    }
}

impl From<SerializableCacheKey> for CacheKey {
    fn from(key: SerializableCacheKey) -> Self {
        Self {
            spec_hash: key.spec_hash,
            backends: key.backends,
            strategy: key.strategy,
            typecheck_only: key.typecheck_only,
        }
    }
}

/// Serializable cache entry for persistence
///
/// Uses SystemTime instead of Instant since Instant is not serializable.
/// On load, times are recalculated relative to the current instant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableCacheEntry<T> {
    /// The cached result
    pub result: T,
    /// Age in seconds at time of save (used to calculate remaining TTL on load)
    pub age_secs: u64,
    /// Number of times this entry was hit
    pub hit_count: u64,
}

/// Serializable cache snapshot for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSnapshot<T> {
    /// Version for forward compatibility
    pub version: u32,
    /// Unix timestamp when the snapshot was created
    pub saved_at: u64,
    /// Cache configuration
    pub config: CacheConfig,
    /// Cache statistics
    pub stats: CacheStats,
    /// Cache entries
    pub entries: Vec<(SerializableCacheKey, SerializableCacheEntry<T>)>,
}

impl<T> CacheSnapshot<T> {
    /// Current snapshot format version
    pub const CURRENT_VERSION: u32 = 1;
}

/// Result of a save operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveResult {
    /// Number of entries saved
    pub entries_saved: usize,
    /// Path where the cache was saved
    pub path: String,
    /// Size of the saved file in bytes
    pub size_bytes: u64,
}

/// Result of a load operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadResult {
    /// Number of entries loaded
    pub entries_loaded: usize,
    /// Number of entries skipped (already expired)
    pub entries_expired: usize,
    /// Path from which the cache was loaded
    pub path: String,
    /// Age of the snapshot in seconds
    pub snapshot_age_secs: u64,
}

// ============================================================================
// Automatic persistence
// ============================================================================

/// Periodically saves a cache to disk on a fixed interval
pub struct CacheAutoSaver<T: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static>
{
    stop_tx: Option<oneshot::Sender<()>>,
    handle: Option<JoinHandle<()>>,
    interval: Duration,
    path: PathBuf,
    _marker: PhantomData<T>,
}

impl<T> CacheAutoSaver<T>
where
    T: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
{
    /// Start periodically saving the cache to `path` every `interval`
    pub fn start(cache: SharedVerificationCache<T>, path: PathBuf, interval: Duration) -> Self {
        assert!(
            !interval.is_zero(),
            "CacheAutoSaver requires a non-zero interval"
        );

        let (stop_tx, mut stop_rx) = oneshot::channel::<()>();
        let save_path = path.clone();

        let handle = tokio::spawn(async move {
            loop {
                let sleeper = tokio::time::sleep(interval);
                tokio::pin!(sleeper);
                tokio::select! {
                    _ = &mut stop_rx => {
                        break;
                    }
                    _ = &mut sleeper => {
                        if let Err(e) = cache.save_to_file(&save_path).await {
                            warn!("Failed to auto-save cache to {:?}: {}", save_path, e);
                        }
                    }
                }
            }
        });

        Self {
            stop_tx: Some(stop_tx),
            handle: Some(handle),
            interval,
            path,
            _marker: PhantomData,
        }
    }

    /// Stop the background task and wait for it to finish
    pub async fn stop(mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }

    /// Interval between auto-saves
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Path where snapshots are written
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl<T> Drop for CacheAutoSaver<T>
where
    T: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
{
    fn drop(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

/// Thread-safe verification result cache
///
/// Uses RwLock for concurrent read access with exclusive write access.
pub struct VerificationCache<T: Clone + Send + Sync> {
    /// Cache entries indexed by key
    entries: RwLock<HashMap<CacheKey, CacheEntry<T>>>,
    /// Cache configuration (mutable at runtime)
    config: RwLock<CacheConfig>,
    /// Cache statistics
    stats: RwLock<CacheStats>,
}

impl<T: Clone + Send + Sync> VerificationCache<T> {
    /// Create a new cache with default configuration
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create a new cache with custom configuration
    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            config: RwLock::new(config),
            stats: RwLock::new(CacheStats::default()),
        }
    }

    /// Create a disabled cache (all operations are no-ops)
    pub fn disabled() -> Self {
        Self::with_config(CacheConfig {
            enabled: false,
            ..Default::default()
        })
    }

    /// Check if caching is enabled
    pub async fn is_enabled(&self) -> bool {
        let config = self.config.read().await;
        config.enabled
    }

    /// Get the current cache configuration
    pub async fn config(&self) -> CacheConfig {
        let config = self.config.read().await;
        config.clone()
    }

    /// Update the cache configuration at runtime
    ///
    /// If `clear_on_change` is true and configuration changed significantly
    /// (TTL decreased or cache disabled), the cache will be cleared.
    pub async fn update_config(&self, new_config: CacheConfig, clear_on_change: bool) {
        let old_config = {
            let config = self.config.read().await;
            config.clone()
        };

        // Determine if we should clear cache
        let should_clear = clear_on_change
            && (new_config.ttl < old_config.ttl
                || (!new_config.enabled && old_config.enabled)
                || new_config.max_entries < old_config.max_entries);

        // Update config
        {
            let mut config = self.config.write().await;
            *config = new_config;
        }

        // Clear if needed
        if should_clear {
            self.clear().await;
        }
    }

    /// Set whether caching is enabled
    pub async fn set_enabled(&self, enabled: bool) {
        let mut config = self.config.write().await;
        config.enabled = enabled;
    }

    /// Set the TTL for cache entries
    pub async fn set_ttl(&self, ttl: Duration) {
        let mut config = self.config.write().await;
        config.ttl = ttl;
    }

    /// Set the maximum number of entries
    pub async fn set_max_entries(&self, max_entries: usize) {
        let mut config = self.config.write().await;
        config.max_entries = max_entries;
    }

    /// Get a cached result if available and not expired
    pub async fn get(&self, key: &CacheKey) -> Option<T> {
        let (enabled, ttl) = {
            let config = self.config.read().await;
            (config.enabled, config.ttl)
        };

        if !enabled {
            return None;
        }

        // First try read-only access
        {
            let entries = self.entries.read().await;
            if let Some(entry) = entries.get(key) {
                if entry.is_expired(ttl) {
                    // Entry expired, will be cleaned up later
                    drop(entries);
                    let mut stats = self.stats.write().await;
                    stats.misses += 1;
                    return None;
                }
                // Clone the result before dropping the lock
                let result = entry.result.clone();
                drop(entries);

                // Upgrade to write lock to update last_accessed
                let mut entries = self.entries.write().await;
                if let Some(entry) = entries.get_mut(key) {
                    entry.record_hit();
                }

                let mut stats = self.stats.write().await;
                stats.hits += 1;

                return Some(result);
            }
        }

        let mut stats = self.stats.write().await;
        stats.misses += 1;
        None
    }

    /// Insert a result into the cache
    pub async fn insert(&self, key: CacheKey, result: T) {
        let (enabled, max_entries) = {
            let config = self.config.read().await;
            (config.enabled, config.max_entries)
        };

        if !enabled {
            return;
        }

        let mut entries = self.entries.write().await;

        // Check if we need to evict entries
        if entries.len() >= max_entries {
            self.evict_lru(&mut entries).await;
        }

        entries.insert(key, CacheEntry::new(result));

        let mut stats = self.stats.write().await;
        stats.entries = entries.len();
    }

    /// Evict the least recently used entry
    async fn evict_lru(&self, entries: &mut HashMap<CacheKey, CacheEntry<T>>) {
        // Find the oldest entry by last_accessed
        let oldest_key = entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| key.clone());

        if let Some(key) = oldest_key {
            entries.remove(&key);
            let mut stats = self.stats.write().await;
            stats.evictions += 1;
        }
    }

    /// Remove expired entries
    pub async fn cleanup_expired(&self) {
        let (enabled, ttl) = {
            let config = self.config.read().await;
            (config.enabled, config.ttl)
        };

        if !enabled {
            return;
        }

        let mut entries = self.entries.write().await;

        let expired_keys: Vec<CacheKey> = entries
            .iter()
            .filter(|(_, entry)| entry.is_expired(ttl))
            .map(|(key, _)| key.clone())
            .collect();

        let expired_count = expired_keys.len();
        for key in expired_keys {
            entries.remove(&key);
        }

        if expired_count > 0 {
            let mut stats = self.stats.write().await;
            stats.expirations += expired_count as u64;
            stats.entries = entries.len();
        }
    }

    /// Clear all cached entries
    pub async fn clear(&self) {
        let mut entries = self.entries.write().await;
        entries.clear();

        let mut stats = self.stats.write().await;
        stats.entries = 0;
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Get current number of entries
    pub async fn len(&self) -> usize {
        let entries = self.entries.read().await;
        entries.len()
    }

    /// Check if cache is empty
    pub async fn is_empty(&self) -> bool {
        let entries = self.entries.read().await;
        entries.is_empty()
    }
}

// Persistence methods - require Serialize + DeserializeOwned
impl<T: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>> VerificationCache<T> {
    /// Save the cache to a file
    ///
    /// Creates a JSON snapshot of the cache including all entries, configuration,
    /// and statistics. Entries that have already expired are not saved.
    pub async fn save_to_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<SaveResult, CachePersistenceError> {
        let path = path.as_ref();
        let (config, stats, ttl) = {
            let config = self.config.read().await;
            let stats = self.stats.read().await;
            (config.clone(), stats.clone(), config.ttl)
        };

        // Collect non-expired entries
        let entries = self.entries.read().await;
        let serializable_entries: Vec<_> = entries
            .iter()
            .filter(|(_, entry)| !entry.is_expired(ttl))
            .map(|(key, entry)| {
                (
                    SerializableCacheKey::from(key),
                    SerializableCacheEntry {
                        result: entry.result.clone(),
                        age_secs: entry.age_secs(),
                        hit_count: entry.hit_count,
                    },
                )
            })
            .collect();
        drop(entries);

        let snapshot = CacheSnapshot {
            version: CacheSnapshot::<T>::CURRENT_VERSION,
            saved_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            config,
            stats,
            entries: serializable_entries,
        };

        // Write to file with buffered IO
        let file = std::fs::File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &snapshot)?;

        // Get file size
        let metadata = std::fs::metadata(path)?;
        let size_bytes = metadata.len();

        Ok(SaveResult {
            entries_saved: snapshot.entries.len(),
            path: path.display().to_string(),
            size_bytes,
        })
    }

    /// Load the cache from a file
    ///
    /// Replaces the current cache contents with entries from the file.
    /// Entries that would already be expired (based on their age at save time
    /// plus time since save) are skipped. Configuration is updated from the file.
    ///
    /// If `merge` is true, entries are added to the existing cache rather than
    /// replacing it. Duplicate keys are overwritten with the loaded values.
    pub async fn load_from_file<P: AsRef<Path>>(
        &self,
        path: P,
        merge: bool,
    ) -> Result<LoadResult, CachePersistenceError> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        let snapshot: CacheSnapshot<T> = serde_json::from_reader(reader)?;

        // Calculate how long ago the snapshot was saved
        let now_unix = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let snapshot_age_secs = now_unix.saturating_sub(snapshot.saved_at);

        // Update configuration
        {
            let mut config = self.config.write().await;
            *config = snapshot.config.clone();
        }

        // Get TTL for expiration check
        let ttl_secs = snapshot.config.ttl.as_secs();

        // Clear existing entries unless merging
        if !merge {
            let mut entries = self.entries.write().await;
            entries.clear();
        }

        let now = Instant::now();
        let mut entries_loaded = 0;
        let mut entries_expired = 0;

        // Load entries, filtering out expired ones
        {
            let mut entries = self.entries.write().await;
            for (ser_key, ser_entry) in snapshot.entries {
                // Calculate effective age: age at save + time since save
                let effective_age = ser_entry.age_secs + snapshot_age_secs;

                // Skip if already expired
                if effective_age >= ttl_secs {
                    entries_expired += 1;
                    continue;
                }

                // Reconstruct the cache entry with adjusted timestamps
                // We subtract the effective age from now to get the created_at time
                let created_at = now - Duration::from_secs(effective_age);
                let entry = CacheEntry {
                    result: ser_entry.result,
                    created_at,
                    last_accessed: now, // Reset last_accessed to now on load
                    hit_count: ser_entry.hit_count,
                };

                let key = CacheKey::from(ser_key);
                entries.insert(key, entry);
                entries_loaded += 1;
            }

            // Update stats
            let mut stats = self.stats.write().await;
            stats.entries = entries.len();
        }

        Ok(LoadResult {
            entries_loaded,
            entries_expired,
            path: path.display().to_string(),
            snapshot_age_secs,
        })
    }

    /// Create a cache snapshot without writing to disk
    ///
    /// Useful for custom persistence implementations or serializing to other formats.
    pub async fn snapshot(&self) -> CacheSnapshot<T> {
        let (config, stats, ttl) = {
            let config = self.config.read().await;
            let stats = self.stats.read().await;
            (config.clone(), stats.clone(), config.ttl)
        };

        let entries = self.entries.read().await;
        let serializable_entries: Vec<_> = entries
            .iter()
            .filter(|(_, entry)| !entry.is_expired(ttl))
            .map(|(key, entry)| {
                (
                    SerializableCacheKey::from(key),
                    SerializableCacheEntry {
                        result: entry.result.clone(),
                        age_secs: entry.age_secs(),
                        hit_count: entry.hit_count,
                    },
                )
            })
            .collect();

        CacheSnapshot {
            version: CacheSnapshot::<T>::CURRENT_VERSION,
            saved_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            config,
            stats,
            entries: serializable_entries,
        }
    }

    /// Load from a cache snapshot
    ///
    /// Similar to load_from_file but takes an in-memory snapshot.
    pub async fn load_snapshot(&self, snapshot: CacheSnapshot<T>, merge: bool) -> LoadResult {
        let now_unix = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let snapshot_age_secs = now_unix.saturating_sub(snapshot.saved_at);

        // Update configuration
        {
            let mut config = self.config.write().await;
            *config = snapshot.config.clone();
        }

        let ttl_secs = snapshot.config.ttl.as_secs();

        if !merge {
            let mut entries = self.entries.write().await;
            entries.clear();
        }

        let now = Instant::now();
        let mut entries_loaded = 0;
        let mut entries_expired = 0;

        {
            let mut entries = self.entries.write().await;
            for (ser_key, ser_entry) in snapshot.entries {
                let effective_age = ser_entry.age_secs + snapshot_age_secs;

                if effective_age >= ttl_secs {
                    entries_expired += 1;
                    continue;
                }

                let created_at = now - Duration::from_secs(effective_age);
                let entry = CacheEntry {
                    result: ser_entry.result,
                    created_at,
                    last_accessed: now,
                    hit_count: ser_entry.hit_count,
                };

                let key = CacheKey::from(ser_key);
                entries.insert(key, entry);
                entries_loaded += 1;
            }

            let mut stats = self.stats.write().await;
            stats.entries = entries.len();
        }

        LoadResult {
            entries_loaded,
            entries_expired,
            path: String::new(),
            snapshot_age_secs,
        }
    }
}

impl<T: Clone + Send + Sync> Default for VerificationCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Result type that includes cache metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResult<T> {
    /// The verification result
    pub result: T,
    /// Whether this was a cache hit
    pub cache_hit: bool,
    /// Age of the cached entry in seconds (if cache hit)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_age_secs: Option<u64>,
}

impl<T> CachedResult<T> {
    /// Create a new cache miss result
    pub fn miss(result: T) -> Self {
        Self {
            result,
            cache_hit: false,
            cache_age_secs: None,
        }
    }

    /// Create a cache hit result
    pub fn hit(result: T, age_secs: u64) -> Self {
        Self {
            result,
            cache_hit: true,
            cache_age_secs: Some(age_secs),
        }
    }
}

/// Shared verification cache instance type
pub type SharedVerificationCache<T> = Arc<VerificationCache<T>>;

/// Create a new shared verification cache
pub fn new_shared_cache<T: Clone + Send + Sync>() -> SharedVerificationCache<T> {
    Arc::new(VerificationCache::new())
}

/// Create a new shared cache with custom config
pub fn new_shared_cache_with_config<T: Clone + Send + Sync>(
    config: CacheConfig,
) -> SharedVerificationCache<T> {
    Arc::new(VerificationCache::with_config(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[test]
    fn test_cache_key_normalization() {
        // Keys with same content but different whitespace should be equal
        let key1 = CacheKey::new("property P1: x > 0", &["lean4".into()], "auto", false);
        let key2 = CacheKey::new("  property P1: x > 0  ", &["lean4".into()], "auto", false);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_backend_ordering() {
        // Keys with same backends in different order should be equal
        let key1 = CacheKey::new("spec", &["lean4".into(), "kani".into()], "auto", false);
        let key2 = CacheKey::new("spec", &["kani".into(), "lean4".into()], "auto", false);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_case_insensitive() {
        // Backend names should be case-insensitive
        let key1 = CacheKey::new("spec", &["LEAN4".into()], "AUTO", false);
        let key2 = CacheKey::new("spec", &["lean4".into()], "auto", false);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_different_typecheck() {
        // typecheck_only difference should create different keys
        let key1 = CacheKey::new("spec", &["lean4".into()], "auto", true);
        let key2 = CacheKey::new("spec", &["lean4".into()], "auto", false);
        assert_ne!(key1, key2);
    }

    #[tokio::test]
    async fn test_cache_insert_and_get() {
        let cache: VerificationCache<String> = VerificationCache::new();
        let key = CacheKey::new("test spec", &["lean4".into()], "auto", false);

        cache.insert(key.clone(), "result".to_string()).await;

        let result = cache.get(&key).await;
        assert_eq!(result, Some("result".to_string()));
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let cache: VerificationCache<String> = VerificationCache::new();
        let key = CacheKey::new("test spec", &["lean4".into()], "auto", false);

        let result = cache.get(&key).await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let config = CacheConfig {
            ttl: Duration::from_millis(50),
            max_entries: 100,
            enabled: true,
        };
        let cache: VerificationCache<String> = VerificationCache::with_config(config);
        let key = CacheKey::new("test spec", &["lean4".into()], "auto", false);

        cache.insert(key.clone(), "result".to_string()).await;

        // Should be present immediately
        let result = cache.get(&key).await;
        assert!(result.is_some());

        // Wait for expiration
        sleep(Duration::from_millis(100)).await;

        // Should be expired
        let result = cache.get(&key).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_cache_lru_eviction() {
        let config = CacheConfig {
            ttl: Duration::from_secs(300),
            max_entries: 2,
            enabled: true,
        };
        let cache: VerificationCache<String> = VerificationCache::with_config(config);

        let key1 = CacheKey::new("spec1", &[], "auto", false);
        let key2 = CacheKey::new("spec2", &[], "auto", false);
        let key3 = CacheKey::new("spec3", &[], "auto", false);

        cache.insert(key1.clone(), "result1".to_string()).await;
        cache.insert(key2.clone(), "result2".to_string()).await;

        // Access key1 to make it more recently used
        let _ = cache.get(&key1).await;

        // Insert key3, should evict key2 (least recently used)
        cache.insert(key3.clone(), "result3".to_string()).await;

        // key1 should still be present
        assert!(cache.get(&key1).await.is_some());
        // key2 should be evicted
        assert!(cache.get(&key2).await.is_none());
        // key3 should be present
        assert!(cache.get(&key3).await.is_some());
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let cache: VerificationCache<String> = VerificationCache::new();
        let key = CacheKey::new("test spec", &["lean4".into()], "auto", false);

        // Miss
        let _ = cache.get(&key).await;

        // Insert
        cache.insert(key.clone(), "result".to_string()).await;

        // Hit
        let _ = cache.get(&key).await;

        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.hit_rate(), 50.0);
    }

    #[tokio::test]
    async fn test_cache_disabled() {
        let cache: VerificationCache<String> = VerificationCache::disabled();
        let key = CacheKey::new("test spec", &["lean4".into()], "auto", false);

        cache.insert(key.clone(), "result".to_string()).await;

        // Should always miss when disabled
        let result = cache.get(&key).await;
        assert!(result.is_none());
        assert!(!cache.is_enabled().await);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let cache: VerificationCache<String> = VerificationCache::new();
        let key = CacheKey::new("test spec", &["lean4".into()], "auto", false);

        cache.insert(key.clone(), "result".to_string()).await;
        assert_eq!(cache.len().await, 1);

        cache.clear().await;
        assert!(cache.is_empty().await);
    }

    #[tokio::test]
    async fn test_cleanup_expired() {
        let config = CacheConfig {
            ttl: Duration::from_millis(50),
            max_entries: 100,
            enabled: true,
        };
        let cache: VerificationCache<String> = VerificationCache::with_config(config);

        let key1 = CacheKey::new("spec1", &[], "auto", false);
        let key2 = CacheKey::new("spec2", &[], "auto", false);

        cache.insert(key1.clone(), "result1".to_string()).await;

        // Wait for key1 to expire
        sleep(Duration::from_millis(60)).await;

        // Insert key2 (not expired yet)
        cache.insert(key2.clone(), "result2".to_string()).await;

        // Cleanup expired entries
        cache.cleanup_expired().await;

        // key1 should be removed
        assert!(cache.get(&key1).await.is_none());
        // key2 should still be present
        assert!(cache.get(&key2).await.is_some());

        let stats = cache.stats().await;
        assert!(stats.expirations >= 1);
    }

    // Persistence tests
    // =========================================================================

    #[tokio::test]
    async fn test_cache_save_and_load() {
        let cache: VerificationCache<String> = VerificationCache::new();
        let key1 = CacheKey::new("spec1", &["lean4".into()], "auto", false);
        let key2 = CacheKey::new("spec2", &["kani".into()], "first", false);

        cache.insert(key1.clone(), "result1".to_string()).await;
        cache.insert(key2.clone(), "result2".to_string()).await;

        // Save to temp file
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_cache_save_load.json");

        let save_result = cache.save_to_file(&path).await.expect("save failed");
        assert_eq!(save_result.entries_saved, 2);
        assert!(save_result.size_bytes > 0);

        // Create new cache and load
        let cache2: VerificationCache<String> = VerificationCache::new();
        let load_result = cache2
            .load_from_file(&path, false)
            .await
            .expect("load failed");
        assert_eq!(load_result.entries_loaded, 2);
        assert_eq!(load_result.entries_expired, 0);

        // Verify entries
        assert_eq!(cache2.get(&key1).await, Some("result1".to_string()));
        assert_eq!(cache2.get(&key2).await, Some("result2".to_string()));

        // Cleanup
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_cache_auto_saver_writes_periodically() {
        let cache: SharedVerificationCache<String> = new_shared_cache();
        let key = CacheKey::new("auto-save-spec", &["lean4".into()], "auto", false);
        cache.insert(key.clone(), "result".to_string()).await;

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_cache_auto_saver.json");

        let autosaver =
            CacheAutoSaver::start(cache.clone(), path.clone(), Duration::from_millis(50));

        // Wait for at least one auto-save cycle to complete
        sleep(Duration::from_millis(160)).await;
        autosaver.stop().await;

        // Verify snapshot exists and can be loaded
        assert!(std::path::Path::new(&path).exists());

        let loaded_cache: VerificationCache<String> = VerificationCache::new();
        let load_result = loaded_cache
            .load_from_file(&path, false)
            .await
            .expect("load failed");
        assert_eq!(load_result.entries_loaded, 1);

        // Cleanup
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_cache_save_filters_expired() {
        let config = CacheConfig {
            ttl: Duration::from_millis(50),
            max_entries: 100,
            enabled: true,
        };
        let cache: VerificationCache<String> = VerificationCache::with_config(config);

        let key1 = CacheKey::new("spec1", &[], "auto", false);
        cache.insert(key1.clone(), "result1".to_string()).await;

        // Wait for expiration
        sleep(Duration::from_millis(60)).await;

        // Add fresh entry
        let key2 = CacheKey::new("spec2", &[], "auto", false);
        cache.insert(key2.clone(), "result2".to_string()).await;

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_cache_expired.json");

        // Save should only include non-expired entry
        let save_result = cache.save_to_file(&path).await.expect("save failed");
        assert_eq!(save_result.entries_saved, 1);

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_cache_load_expired_entries_skipped() {
        let config = CacheConfig {
            ttl: Duration::from_secs(1),
            max_entries: 100,
            enabled: true,
        };
        let cache: VerificationCache<String> = VerificationCache::with_config(config);

        let key = CacheKey::new("spec", &[], "auto", false);
        cache.insert(key.clone(), "result".to_string()).await;

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_cache_load_expired.json");
        cache.save_to_file(&path).await.expect("save failed");

        // Wait for TTL to expire
        sleep(Duration::from_millis(1100)).await;

        // Load into new cache - entry should be skipped as expired
        let cache2: VerificationCache<String> = VerificationCache::new();
        let load_result = cache2
            .load_from_file(&path, false)
            .await
            .expect("load failed");
        assert_eq!(load_result.entries_loaded, 0);
        assert_eq!(load_result.entries_expired, 1);

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_cache_load_merge() {
        let cache1: VerificationCache<String> = VerificationCache::new();
        let key1 = CacheKey::new("spec1", &[], "auto", false);
        cache1.insert(key1.clone(), "result1".to_string()).await;

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_cache_merge.json");
        cache1.save_to_file(&path).await.expect("save failed");

        // Create cache with different entry
        let cache2: VerificationCache<String> = VerificationCache::new();
        let key2 = CacheKey::new("spec2", &[], "auto", false);
        cache2.insert(key2.clone(), "result2".to_string()).await;

        // Load with merge=true
        let load_result = cache2
            .load_from_file(&path, true)
            .await
            .expect("load failed");
        assert_eq!(load_result.entries_loaded, 1);

        // Both entries should be present
        assert_eq!(cache2.get(&key1).await, Some("result1".to_string()));
        assert_eq!(cache2.get(&key2).await, Some("result2".to_string()));
        assert_eq!(cache2.len().await, 2);

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_cache_load_replace() {
        let cache1: VerificationCache<String> = VerificationCache::new();
        let key1 = CacheKey::new("spec1", &[], "auto", false);
        cache1.insert(key1.clone(), "result1".to_string()).await;

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_cache_replace.json");
        cache1.save_to_file(&path).await.expect("save failed");

        // Create cache with different entry
        let cache2: VerificationCache<String> = VerificationCache::new();
        let key2 = CacheKey::new("spec2", &[], "auto", false);
        cache2.insert(key2.clone(), "result2".to_string()).await;

        // Load with merge=false (replace)
        cache2
            .load_from_file(&path, false)
            .await
            .expect("load failed");

        // Only loaded entry should be present
        assert_eq!(cache2.get(&key1).await, Some("result1".to_string()));
        assert_eq!(cache2.get(&key2).await, None);
        assert_eq!(cache2.len().await, 1);

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_cache_snapshot() {
        let cache: VerificationCache<String> = VerificationCache::new();
        let key = CacheKey::new("spec", &["lean4".into()], "auto", false);
        cache.insert(key.clone(), "result".to_string()).await;

        let snapshot = cache.snapshot().await;
        assert_eq!(snapshot.version, CacheSnapshot::<String>::CURRENT_VERSION);
        assert_eq!(snapshot.entries.len(), 1);
        assert!(snapshot.saved_at > 0);
    }

    #[tokio::test]
    async fn test_cache_load_snapshot() {
        let cache1: VerificationCache<String> = VerificationCache::new();
        let key = CacheKey::new("spec", &[], "auto", false);
        cache1.insert(key.clone(), "result".to_string()).await;

        let snapshot = cache1.snapshot().await;

        let cache2: VerificationCache<String> = VerificationCache::new();
        let load_result = cache2.load_snapshot(snapshot, false).await;
        assert_eq!(load_result.entries_loaded, 1);
        assert_eq!(cache2.get(&key).await, Some("result".to_string()));
    }

    #[tokio::test]
    async fn test_cache_persistence_preserves_hit_count() {
        let cache: VerificationCache<String> = VerificationCache::new();
        let key = CacheKey::new("spec", &[], "auto", false);
        cache.insert(key.clone(), "result".to_string()).await;

        // Generate some hits
        for _ in 0..5 {
            cache.get(&key).await;
        }

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_cache_hit_count.json");
        cache.save_to_file(&path).await.expect("save failed");

        // Verify hit count in snapshot
        let snapshot: CacheSnapshot<String> = {
            let file = std::fs::File::open(&path).unwrap();
            serde_json::from_reader(file).unwrap()
        };
        assert_eq!(snapshot.entries[0].1.hit_count, 5);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_cache_persistence_error_display() {
        let io_err = CachePersistenceError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        assert!(io_err.to_string().contains("IO error"));

        let serde_err =
            CachePersistenceError::Serde(serde_json::from_str::<String>("invalid").unwrap_err());
        assert!(serde_err.to_string().contains("Serialization error"));
    }
}
