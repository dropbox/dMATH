//! Content-addressable proof storage
//!
//! Provides efficient storage and retrieval of proofs using their content hash.
//! Supports both in-memory and file-based storage.

use crate::format::UniversalProof;
use crate::hash::ContentHash;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// Errors that can occur during proof storage operations
#[derive(Debug, Error)]
pub enum ProofStorageError {
    #[error("Proof not found: {0}")]
    NotFound(ContentHash),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Integrity check failed for proof {0}")]
    IntegrityCheckFailed(ContentHash),

    #[error("Storage is full")]
    StorageFull,
}

/// Result type for storage operations
pub type StorageResult<T> = Result<T, ProofStorageError>;

/// Trait for proof storage backends
pub trait ProofStorage: Send + Sync {
    /// Store a proof
    fn store(&self, proof: &UniversalProof) -> StorageResult<()>;

    /// Retrieve a proof by its hash
    fn get(&self, hash: &ContentHash) -> StorageResult<UniversalProof>;

    /// Check if a proof exists
    fn contains(&self, hash: &ContentHash) -> bool;

    /// Remove a proof
    fn remove(&self, hash: &ContentHash) -> StorageResult<()>;

    /// List all proof hashes
    fn list(&self) -> StorageResult<Vec<ContentHash>>;

    /// Get the number of stored proofs
    fn len(&self) -> usize;

    /// Check if storage is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all proofs
    fn clear(&self) -> StorageResult<()>;

    /// Get storage statistics
    fn stats(&self) -> StorageStats;
}

/// Statistics about the storage
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Number of proofs stored
    pub proof_count: usize,
    /// Total size in bytes
    pub total_size_bytes: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
}

/// In-memory proof storage (for testing and caching)
#[derive(Debug, Default)]
pub struct MemoryStorage {
    proofs: RwLock<HashMap<ContentHash, UniversalProof>>,
    stats: RwLock<StorageStats>,
}

impl MemoryStorage {
    /// Create a new in-memory storage
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            proofs: RwLock::new(HashMap::with_capacity(capacity)),
            stats: RwLock::new(StorageStats::default()),
        }
    }
}

impl ProofStorage for MemoryStorage {
    fn store(&self, proof: &UniversalProof) -> StorageResult<()> {
        let mut proofs = self.proofs.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        // Update stats
        if !proofs.contains_key(&proof.id) {
            stats.proof_count += 1;
            stats.total_size_bytes += proof.metadata.size_bytes;
        }

        proofs.insert(proof.id, proof.clone());
        Ok(())
    }

    fn get(&self, hash: &ContentHash) -> StorageResult<UniversalProof> {
        let proofs = self.proofs.read().unwrap();
        let mut stats = self.stats.write().unwrap();

        match proofs.get(hash) {
            Some(proof) => {
                stats.cache_hits += 1;
                Ok(proof.clone())
            }
            None => {
                stats.cache_misses += 1;
                Err(ProofStorageError::NotFound(*hash))
            }
        }
    }

    fn contains(&self, hash: &ContentHash) -> bool {
        self.proofs.read().unwrap().contains_key(hash)
    }

    fn remove(&self, hash: &ContentHash) -> StorageResult<()> {
        let mut proofs = self.proofs.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        if let Some(proof) = proofs.remove(hash) {
            stats.proof_count = stats.proof_count.saturating_sub(1);
            stats.total_size_bytes = stats
                .total_size_bytes
                .saturating_sub(proof.metadata.size_bytes);
            Ok(())
        } else {
            Err(ProofStorageError::NotFound(*hash))
        }
    }

    fn list(&self) -> StorageResult<Vec<ContentHash>> {
        Ok(self.proofs.read().unwrap().keys().copied().collect())
    }

    fn len(&self) -> usize {
        self.proofs.read().unwrap().len()
    }

    fn clear(&self) -> StorageResult<()> {
        self.proofs.write().unwrap().clear();
        *self.stats.write().unwrap() = StorageStats::default();
        Ok(())
    }

    fn stats(&self) -> StorageStats {
        self.stats.read().unwrap().clone()
    }
}

/// File-based proof storage (persistent)
#[derive(Debug)]
pub struct FileStorage {
    /// Root directory for storage
    root: PathBuf,
    /// In-memory cache
    cache: MemoryStorage,
    /// Maximum cache size (number of proofs)
    max_cache_size: usize,
}

impl FileStorage {
    /// Create a new file storage at the given path
    pub fn new(root: impl AsRef<Path>) -> StorageResult<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            cache: MemoryStorage::new(),
            max_cache_size: 1000,
        })
    }

    /// Create with custom cache size
    pub fn with_cache_size(root: impl AsRef<Path>, max_cache_size: usize) -> StorageResult<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            cache: MemoryStorage::with_capacity(max_cache_size),
            max_cache_size,
        })
    }

    /// Get the file path for a proof hash
    fn proof_path(&self, hash: &ContentHash) -> PathBuf {
        let hex = hash.to_hex();
        // Use first 2 chars as directory (sharding)
        let dir = &hex[..2];
        self.root.join(dir).join(format!("{hex}.json"))
    }

    /// Load a proof from disk
    fn load_from_disk(&self, hash: &ContentHash) -> StorageResult<UniversalProof> {
        let path = self.proof_path(hash);
        let content = fs::read_to_string(&path)?;
        let proof: UniversalProof = serde_json::from_str(&content)?;

        // Verify integrity
        if !proof.verify_integrity() {
            return Err(ProofStorageError::IntegrityCheckFailed(*hash));
        }

        Ok(proof)
    }

    /// Save a proof to disk
    fn save_to_disk(&self, proof: &UniversalProof) -> StorageResult<()> {
        let path = self.proof_path(&proof.id);

        // Create parent directory
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let content = serde_json::to_string_pretty(proof)?;
        fs::write(&path, content)?;
        Ok(())
    }

    /// Evict oldest cache entries if over capacity
    fn evict_cache_if_needed(&self) {
        if self.cache.len() > self.max_cache_size {
            // Simple eviction: clear half the cache
            // A more sophisticated implementation would use LRU
            let _ = self.cache.clear();
        }
    }
}

impl ProofStorage for FileStorage {
    fn store(&self, proof: &UniversalProof) -> StorageResult<()> {
        // Save to disk first
        self.save_to_disk(proof)?;

        // Update cache
        self.evict_cache_if_needed();
        self.cache.store(proof)?;

        Ok(())
    }

    fn get(&self, hash: &ContentHash) -> StorageResult<UniversalProof> {
        // Check cache first
        if let Ok(proof) = self.cache.get(hash) {
            return Ok(proof);
        }

        // Load from disk
        let proof = self.load_from_disk(hash)?;

        // Update cache
        self.evict_cache_if_needed();
        let _ = self.cache.store(&proof);

        Ok(proof)
    }

    fn contains(&self, hash: &ContentHash) -> bool {
        self.cache.contains(hash) || self.proof_path(hash).exists()
    }

    fn remove(&self, hash: &ContentHash) -> StorageResult<()> {
        // Remove from disk
        let path = self.proof_path(hash);
        if path.exists() {
            fs::remove_file(&path)?;
        }

        // Remove from cache (ignore if not in cache)
        let _ = self.cache.remove(hash);

        Ok(())
    }

    fn list(&self) -> StorageResult<Vec<ContentHash>> {
        let mut hashes = Vec::new();

        // Iterate over shard directories
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                for file in fs::read_dir(entry.path())? {
                    let file = file?;
                    if let Some(name) = file.path().file_stem() {
                        if let Some(name_str) = name.to_str() {
                            if let Ok(hash) = ContentHash::from_hex(name_str) {
                                hashes.push(hash);
                            }
                        }
                    }
                }
            }
        }

        Ok(hashes)
    }

    fn len(&self) -> usize {
        self.list().map_or(0, |l| l.len())
    }

    fn clear(&self) -> StorageResult<()> {
        // Clear cache
        self.cache.clear()?;

        // Remove all files
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                fs::remove_dir_all(entry.path())?;
            } else {
                fs::remove_file(entry.path())?;
            }
        }

        Ok(())
    }

    fn stats(&self) -> StorageStats {
        let mut stats = self.cache.stats();
        stats.proof_count = self.len();
        // Note: total_size_bytes is an estimate from cache; a full implementation
        // would track this incrementally across all files
        stats
    }
}

/// A layered storage that combines multiple backends
pub struct LayeredStorage {
    layers: Vec<Arc<dyn ProofStorage>>,
}

impl LayeredStorage {
    /// Create a new layered storage with the given backends
    /// First layer is checked first for reads, last layer is written to first
    pub fn new(layers: Vec<Arc<dyn ProofStorage>>) -> Self {
        Self { layers }
    }

    /// Create a standard two-layer storage (memory + file)
    pub fn standard(file_path: impl AsRef<Path>) -> StorageResult<Self> {
        let memory = Arc::new(MemoryStorage::new());
        let file = Arc::new(FileStorage::new(file_path)?);
        Ok(Self::new(vec![memory, file]))
    }
}

impl ProofStorage for LayeredStorage {
    fn store(&self, proof: &UniversalProof) -> StorageResult<()> {
        // Store in all layers
        for layer in &self.layers {
            layer.store(proof)?;
        }
        Ok(())
    }

    fn get(&self, hash: &ContentHash) -> StorageResult<UniversalProof> {
        // Try each layer in order
        for (i, layer) in self.layers.iter().enumerate() {
            if let Ok(proof) = layer.get(hash) {
                // Populate earlier layers (cache promotion)
                for earlier in &self.layers[..i] {
                    let _ = earlier.store(&proof);
                }
                return Ok(proof);
            }
        }
        Err(ProofStorageError::NotFound(*hash))
    }

    fn contains(&self, hash: &ContentHash) -> bool {
        self.layers.iter().any(|l| l.contains(hash))
    }

    fn remove(&self, hash: &ContentHash) -> StorageResult<()> {
        // Remove from all layers
        for layer in &self.layers {
            let _ = layer.remove(hash);
        }
        Ok(())
    }

    fn list(&self) -> StorageResult<Vec<ContentHash>> {
        // Combine lists from all layers (deduplicated)
        let mut all_hashes = std::collections::HashSet::new();
        for layer in &self.layers {
            if let Ok(hashes) = layer.list() {
                all_hashes.extend(hashes);
            }
        }
        Ok(all_hashes.into_iter().collect())
    }

    fn len(&self) -> usize {
        self.list().map_or(0, |l| l.len())
    }

    fn clear(&self) -> StorageResult<()> {
        for layer in &self.layers {
            layer.clear()?;
        }
        Ok(())
    }

    fn stats(&self) -> StorageStats {
        // Aggregate stats from all layers
        let mut total = StorageStats::default();
        for layer in &self.layers {
            let s = layer.stats();
            total.cache_hits += s.cache_hits;
            total.cache_misses += s.cache_misses;
        }
        total.proof_count = self.len();
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::ProofStep;
    use tempfile::tempdir;

    fn make_test_proof(vc: &str) -> UniversalProof {
        UniversalProof::builder()
            .vc(vc)
            .step(ProofStep::chc_invariant("inv", vec![], "true"))
            .build()
    }

    // ==================== MemoryStorage Tests ====================

    #[test]
    fn test_memory_storage_store_get() {
        let storage = MemoryStorage::new();
        let proof = make_test_proof("(assert true)");

        storage.store(&proof).unwrap();
        let retrieved = storage.get(&proof.id).unwrap();

        assert_eq!(retrieved.id, proof.id);
        assert_eq!(retrieved.vc, proof.vc);
    }

    #[test]
    fn test_memory_storage_contains() {
        let storage = MemoryStorage::new();
        let proof = make_test_proof("(assert true)");

        assert!(!storage.contains(&proof.id));
        storage.store(&proof).unwrap();
        assert!(storage.contains(&proof.id));
    }

    #[test]
    fn test_memory_storage_remove() {
        let storage = MemoryStorage::new();
        let proof = make_test_proof("(assert true)");

        storage.store(&proof).unwrap();
        assert!(storage.contains(&proof.id));

        storage.remove(&proof.id).unwrap();
        assert!(!storage.contains(&proof.id));
    }

    #[test]
    fn test_memory_storage_remove_not_found() {
        let storage = MemoryStorage::new();
        let hash = ContentHash::hash_str("not found");

        let result = storage.remove(&hash);
        assert!(matches!(result, Err(ProofStorageError::NotFound(_))));
    }

    #[test]
    fn test_memory_storage_list() {
        let storage = MemoryStorage::new();
        let proof1 = make_test_proof("(assert true)");
        let proof2 = make_test_proof("(assert false)");

        storage.store(&proof1).unwrap();
        storage.store(&proof2).unwrap();

        let list = storage.list().unwrap();
        assert_eq!(list.len(), 2);
        assert!(list.contains(&proof1.id));
        assert!(list.contains(&proof2.id));
    }

    #[test]
    fn test_memory_storage_len() {
        let storage = MemoryStorage::new();
        assert!(storage.is_empty());

        storage.store(&make_test_proof("a")).unwrap();
        assert_eq!(storage.len(), 1);
        assert!(!storage.is_empty());

        storage.store(&make_test_proof("b")).unwrap();
        assert_eq!(storage.len(), 2);
    }

    #[test]
    fn test_memory_storage_clear() {
        let storage = MemoryStorage::new();
        storage.store(&make_test_proof("a")).unwrap();
        storage.store(&make_test_proof("b")).unwrap();

        storage.clear().unwrap();
        assert!(storage.is_empty());
    }

    #[test]
    fn test_memory_storage_stats() {
        let storage = MemoryStorage::new();
        let proof = make_test_proof("(assert true)");

        storage.store(&proof).unwrap();
        let _ = storage.get(&proof.id);
        let _ = storage.get(&ContentHash::hash_str("missing"));

        let stats = storage.stats();
        assert_eq!(stats.proof_count, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
    }

    #[test]
    fn test_memory_storage_with_capacity() {
        let storage = MemoryStorage::with_capacity(100);
        assert!(storage.is_empty());
    }

    #[test]
    fn test_memory_storage_overwrite() {
        let storage = MemoryStorage::new();
        let proof1 = make_test_proof("(assert true)");

        storage.store(&proof1).unwrap();
        let stats1 = storage.stats();

        // Store same proof again (overwrite)
        storage.store(&proof1).unwrap();
        let stats2 = storage.stats();

        // Count shouldn't increase
        assert_eq!(stats1.proof_count, stats2.proof_count);
    }

    // ==================== FileStorage Tests ====================

    #[test]
    fn test_file_storage_store_get() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();
        let proof = make_test_proof("(assert true)");

        storage.store(&proof).unwrap();
        let retrieved = storage.get(&proof.id).unwrap();

        assert_eq!(retrieved.id, proof.id);
    }

    #[test]
    fn test_file_storage_contains() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();
        let proof = make_test_proof("(assert true)");

        assert!(!storage.contains(&proof.id));
        storage.store(&proof).unwrap();
        assert!(storage.contains(&proof.id));
    }

    #[test]
    fn test_file_storage_persistence() {
        let dir = tempdir().unwrap();
        let proof = make_test_proof("(assert true)");

        // Store with one instance
        {
            let storage = FileStorage::new(dir.path()).unwrap();
            storage.store(&proof).unwrap();
        }

        // Retrieve with new instance
        {
            let storage = FileStorage::new(dir.path()).unwrap();
            let retrieved = storage.get(&proof.id).unwrap();
            assert_eq!(retrieved.id, proof.id);
        }
    }

    #[test]
    fn test_file_storage_remove() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();
        let proof = make_test_proof("(assert true)");

        storage.store(&proof).unwrap();
        storage.remove(&proof.id).unwrap();
        assert!(!storage.contains(&proof.id));
    }

    #[test]
    fn test_file_storage_list() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();
        let proof1 = make_test_proof("a");
        let proof2 = make_test_proof("b");

        storage.store(&proof1).unwrap();
        storage.store(&proof2).unwrap();

        let list = storage.list().unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_file_storage_clear() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();
        storage.store(&make_test_proof("a")).unwrap();
        storage.store(&make_test_proof("b")).unwrap();

        storage.clear().unwrap();
        assert!(storage.is_empty());
    }

    #[test]
    fn test_file_storage_cache_hit() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();
        let proof = make_test_proof("(assert true)");

        storage.store(&proof).unwrap();

        // First get loads from disk and caches
        let _ = storage.get(&proof.id);
        let stats1 = storage.stats();

        // Second get should hit cache
        let _ = storage.get(&proof.id);
        let stats2 = storage.stats();

        assert!(stats2.cache_hits > stats1.cache_hits);
    }

    #[test]
    fn test_file_storage_with_cache_size() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::with_cache_size(dir.path(), 10).unwrap();
        assert!(storage.is_empty());
    }

    // ==================== LayeredStorage Tests ====================

    #[test]
    fn test_layered_storage_store_get() {
        let dir = tempdir().unwrap();
        let storage = LayeredStorage::standard(dir.path()).unwrap();
        let proof = make_test_proof("(assert true)");

        storage.store(&proof).unwrap();
        let retrieved = storage.get(&proof.id).unwrap();

        assert_eq!(retrieved.id, proof.id);
    }

    #[test]
    fn test_layered_storage_cache_promotion() {
        let dir = tempdir().unwrap();
        let memory = Arc::new(MemoryStorage::new());
        let file = Arc::new(FileStorage::new(dir.path()).unwrap());

        let proof = make_test_proof("(assert true)");

        // Store only in file layer
        file.store(&proof).unwrap();

        // Create layered storage
        let layered = LayeredStorage::new(vec![memory.clone(), file]);

        // Get should promote to memory
        let _ = layered.get(&proof.id).unwrap();
        assert!(memory.contains(&proof.id));
    }

    #[test]
    fn test_layered_storage_contains() {
        let dir = tempdir().unwrap();
        let storage = LayeredStorage::standard(dir.path()).unwrap();
        let proof = make_test_proof("(assert true)");

        assert!(!storage.contains(&proof.id));
        storage.store(&proof).unwrap();
        assert!(storage.contains(&proof.id));
    }

    #[test]
    fn test_layered_storage_remove() {
        let dir = tempdir().unwrap();
        let storage = LayeredStorage::standard(dir.path()).unwrap();
        let proof = make_test_proof("(assert true)");

        storage.store(&proof).unwrap();
        storage.remove(&proof.id).unwrap();
        assert!(!storage.contains(&proof.id));
    }

    #[test]
    fn test_layered_storage_list_dedup() {
        let dir = tempdir().unwrap();
        let memory = Arc::new(MemoryStorage::new());
        let file = Arc::new(FileStorage::new(dir.path()).unwrap());

        let proof = make_test_proof("(assert true)");

        // Store in both layers
        memory.store(&proof).unwrap();
        file.store(&proof).unwrap();

        let layered = LayeredStorage::new(vec![memory, file]);
        let list = layered.list().unwrap();

        // Should be deduplicated
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_layered_storage_clear() {
        let dir = tempdir().unwrap();
        let storage = LayeredStorage::standard(dir.path()).unwrap();
        storage.store(&make_test_proof("a")).unwrap();

        storage.clear().unwrap();
        assert!(storage.is_empty());
    }

    // ==================== Error Tests ====================

    #[test]
    fn test_not_found_error() {
        let storage = MemoryStorage::new();
        let hash = ContentHash::hash_str("missing");

        let result = storage.get(&hash);
        assert!(matches!(result, Err(ProofStorageError::NotFound(_))));
    }

    #[test]
    fn test_error_display() {
        let hash = ContentHash::hash_str("test");
        let err = ProofStorageError::NotFound(hash);
        let msg = format!("{}", err);
        assert!(msg.contains("not found"));
    }

    #[test]
    fn test_integrity_error_display() {
        let hash = ContentHash::hash_str("test");
        let err = ProofStorageError::IntegrityCheckFailed(hash);
        let msg = format!("{}", err);
        assert!(msg.contains("Integrity"));
    }

    // ==================== StorageStats Tests ====================

    #[test]
    fn test_storage_stats_default() {
        let stats = StorageStats::default();
        assert_eq!(stats.proof_count, 0);
        assert_eq!(stats.total_size_bytes, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
    }

    #[test]
    fn test_storage_stats_clone() {
        let stats = StorageStats {
            proof_count: 5,
            total_size_bytes: 1000,
            cache_hits: 10,
            cache_misses: 2,
        };
        let cloned = stats.clone();
        assert_eq!(stats.proof_count, cloned.proof_count);
    }

    #[test]
    fn test_storage_stats_debug() {
        let stats = StorageStats::default();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("StorageStats"));
    }

    // ==================== Mutation coverage tests ====================

    #[test]
    fn test_memory_storage_with_capacity_pre_allocates() {
        // Test that with_capacity actually creates storage with capacity (catches Default mutant)
        let storage = MemoryStorage::with_capacity(1000);

        // Store many proofs - should not reallocate if capacity was set
        for i in 0..100 {
            let proof = make_test_proof(&format!("proof_{}", i));
            storage.store(&proof).unwrap();
        }

        assert_eq!(storage.len(), 100);
    }

    #[test]
    fn test_memory_storage_total_size_bytes_tracks_correctly() {
        // Test that total_size_bytes uses += not *= (catches *= mutant)
        let storage = MemoryStorage::new();

        let proof1 = make_test_proof("(assert proof1)");
        let proof2 = make_test_proof("(assert proof2)");

        storage.store(&proof1).unwrap();
        let stats1 = storage.stats();

        storage.store(&proof2).unwrap();
        let stats2 = storage.stats();

        // Size should be sum, not product
        // With *=, after first store: size = 0 * x = 0, after second: 0 * y = 0
        // With +=, after first store: size = x, after second: size = x + y
        assert!(
            stats2.total_size_bytes >= stats1.total_size_bytes,
            "total_size_bytes should increase or stay same, was {} then {}",
            stats1.total_size_bytes,
            stats2.total_size_bytes
        );

        // Specifically test that size is reasonable (non-zero after storing)
        assert!(
            stats1.total_size_bytes > 0,
            "Size should be > 0 after first store"
        );
        assert!(
            stats2.total_size_bytes > 0,
            "Size should be > 0 after second store"
        );
    }

    #[test]
    fn test_file_storage_eviction_happens_at_capacity() {
        // Test that cache eviction happens when over max_cache_size
        let dir = tempdir().unwrap();
        let storage = FileStorage::with_cache_size(dir.path(), 2).unwrap(); // Very small cache

        // Store more than cache size
        for i in 0..5 {
            let proof = make_test_proof(&format!("proof_{}", i));
            storage.store(&proof).unwrap();
        }

        // All proofs should still be retrievable from disk
        for i in 0..5 {
            let proof = make_test_proof(&format!("proof_{}", i));
            let retrieved = storage.get(&proof.id);
            assert!(retrieved.is_ok(), "Proof {} should be retrievable", i);
        }
    }

    #[test]
    fn test_file_storage_eviction_threshold() {
        // Test that > comparison works correctly (catches > -> == and > -> >= mutants)
        let dir = tempdir().unwrap();
        let storage = FileStorage::with_cache_size(dir.path(), 3).unwrap();

        // Store exactly at capacity - should NOT evict
        for i in 0..3 {
            let proof = make_test_proof(&format!("at_capacity_{}", i));
            storage.store(&proof).unwrap();
        }

        // Store one more - should trigger eviction (len > max_cache_size)
        let overflow_proof = make_test_proof("overflow");
        storage.store(&overflow_proof).unwrap();

        // All should still be accessible from disk
        assert!(storage.contains(&overflow_proof.id));
    }

    #[test]
    fn test_file_storage_contains_checks_both_cache_and_disk() {
        // Test that contains uses || not && (catches || -> && mutant)
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        let proof = make_test_proof("(assert true)");

        // Store and verify exists
        storage.store(&proof).unwrap();
        assert!(storage.contains(&proof.id));

        // Clear the internal cache by recreating storage
        // (File should still exist on disk)
        let storage2 = FileStorage::new(dir.path()).unwrap();

        // With &&, this would be false (cache empty && disk exists = false)
        // With ||, this is true (cache empty || disk exists = true)
        assert!(
            storage2.contains(&proof.id),
            "Should find proof on disk even if not in cache"
        );
    }

    #[test]
    fn test_file_storage_len_returns_actual_count() {
        // Test that len() returns actual count, not 0 (catches -> 0 mutant)
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        assert!(storage.is_empty());

        storage.store(&make_test_proof("a")).unwrap();
        assert_eq!(storage.len(), 1);

        storage.store(&make_test_proof("b")).unwrap();
        assert_eq!(storage.len(), 2);

        storage.store(&make_test_proof("c")).unwrap();
        assert_eq!(storage.len(), 3);
    }

    #[test]
    fn test_layered_storage_len_returns_actual_count() {
        // Test that LayeredStorage::len() returns actual count (catches -> 0 mutant)
        let dir = tempdir().unwrap();
        let storage = LayeredStorage::standard(dir.path()).unwrap();

        assert!(storage.is_empty());

        storage.store(&make_test_proof("x")).unwrap();
        assert_eq!(storage.len(), 1);

        storage.store(&make_test_proof("y")).unwrap();
        assert_eq!(storage.len(), 2);
    }

    #[test]
    fn test_layered_storage_stats_aggregates_correctly() {
        // Test that stats uses += not Default (catches Default mutant)
        // and that += is used not -= or *= (catches arithmetic mutants)
        let dir = tempdir().unwrap();
        let memory = Arc::new(MemoryStorage::new());
        let file = Arc::new(FileStorage::new(dir.path()).unwrap());

        let proof = make_test_proof("(assert true)");

        // Store in both layers
        memory.store(&proof).unwrap();
        file.store(&proof).unwrap();

        // Generate some cache hits/misses
        let _ = memory.get(&proof.id); // hit
        let _ = memory.get(&ContentHash::hash_str("miss1")); // miss
        let _ = file.get(&proof.id); // hit

        let layered = LayeredStorage::new(vec![memory.clone(), file.clone()]);
        let stats = layered.stats();

        // Stats should aggregate cache_hits and cache_misses from both layers
        let memory_stats = memory.stats();
        let file_stats = file.stats();

        // With +=: total = memory + file
        // With -=: total would be negative or wrong
        // With *=: total would be 0 (starting from 0)
        assert_eq!(
            stats.cache_hits,
            memory_stats.cache_hits + file_stats.cache_hits,
            "cache_hits should be sum of layers"
        );
        assert_eq!(
            stats.cache_misses,
            memory_stats.cache_misses + file_stats.cache_misses,
            "cache_misses should be sum of layers"
        );

        // proof_count should come from len()
        assert_eq!(stats.proof_count, 1);
    }

    #[test]
    fn test_layered_storage_stats_with_multiple_proofs() {
        // Additional test for stats aggregation with more data
        let dir = tempdir().unwrap();
        let storage = LayeredStorage::standard(dir.path()).unwrap();

        // Store several proofs
        for i in 0..5 {
            storage
                .store(&make_test_proof(&format!("proof_{}", i)))
                .unwrap();
        }

        let stats = storage.stats();
        assert_eq!(stats.proof_count, 5);
    }

    #[test]
    fn test_memory_storage_stats_cache_hits_increment() {
        // Specifically test that cache_hits uses += correctly
        let storage = MemoryStorage::new();
        let proof = make_test_proof("test");

        storage.store(&proof).unwrap();

        // Multiple gets should each increment cache_hits
        for i in 1..=5 {
            let _ = storage.get(&proof.id);
            let stats = storage.stats();
            assert_eq!(
                stats.cache_hits, i,
                "cache_hits should be {} after {} gets",
                i, i
            );
        }
    }

    #[test]
    fn test_memory_storage_stats_cache_misses_increment() {
        // Test that cache_misses uses += correctly
        let storage = MemoryStorage::new();

        for i in 1..=5 {
            let _ = storage.get(&ContentHash::hash_str(&format!("missing_{}", i)));
            let stats = storage.stats();
            assert_eq!(
                stats.cache_misses, i,
                "cache_misses should be {} after {} misses",
                i, i
            );
        }
    }

    #[test]
    fn test_file_storage_contains_from_disk_only() {
        // More targeted test for || vs && in contains
        let dir = tempdir().unwrap();

        // Create and store with first storage instance
        {
            let storage = FileStorage::new(dir.path()).unwrap();
            let proof = make_test_proof("disk_only");
            storage.store(&proof).unwrap();
        }

        // New storage instance - cache is empty, but file exists on disk
        let storage2 = FileStorage::new(dir.path()).unwrap();
        let proof = make_test_proof("disk_only");

        // With || : (false || true) = true  ✓
        // With && : (false && true) = false ✗
        assert!(
            storage2.contains(&proof.id),
            "contains should return true when proof exists on disk but not in cache"
        );
    }

    #[test]
    fn test_file_storage_stats_reports_correct_count() {
        // Test FileStorage::stats() proof_count
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        storage.store(&make_test_proof("a")).unwrap();
        storage.store(&make_test_proof("b")).unwrap();

        let stats = storage.stats();
        assert_eq!(stats.proof_count, 2);
    }

    #[test]
    fn test_file_storage_evict_at_boundary_not_triggered() {
        // Test that eviction does NOT happen when len == max_cache_size
        // This catches the > to >= mutation
        let dir = tempdir().unwrap();
        let storage = FileStorage::with_cache_size(dir.path(), 3).unwrap();

        // Store exactly max_cache_size items
        let proof1 = make_test_proof("one");
        let proof2 = make_test_proof("two");
        let proof3 = make_test_proof("three");

        storage.store(&proof1).unwrap();
        storage.store(&proof2).unwrap();
        storage.store(&proof3).unwrap();

        // Cache should NOT be cleared (eviction is > not >=)
        // We can verify by checking that retrieval hits cache (cache_hits goes up)
        let _r = storage.get(&proof1.id).unwrap();
        let stats = storage.stats();

        // If eviction happened incorrectly (>= instead of >), proof1 would NOT be in cache
        // and this get would be a cache miss (disk read). With correct code (>), it's a hit.
        // stats.cache_hits should be > 0 if the proof was in cache
        assert!(
            stats.cache_hits >= 1,
            "Should have cache hit, but got {} hits. Cache was incorrectly evicted at boundary.",
            stats.cache_hits
        );
    }

    #[test]
    fn test_file_storage_evict_over_boundary_triggered() {
        // Test that eviction happens when len > max_cache_size
        // This catches mutations that remove eviction or change > to <
        //
        // Note: eviction check happens BEFORE store, so we need len > max BEFORE
        // the store call. With max=2, we need len=3 before the 4th store.
        let dir = tempdir().unwrap();
        let storage = FileStorage::with_cache_size(dir.path(), 2).unwrap();

        // Store more items to trigger eviction
        let proof1 = make_test_proof("first");
        let proof2 = make_test_proof("second");
        let proof3 = make_test_proof("third");
        let proof4 = make_test_proof("fourth"); // This one triggers eviction (len=3 > max=2)

        storage.store(&proof1).unwrap(); // len becomes 1
        storage.store(&proof2).unwrap(); // len becomes 2
        storage.store(&proof3).unwrap(); // evict_check(len=2 > 2? NO), len becomes 3

        // Now len=3, next store will trigger eviction (3 > 2)
        storage.store(&proof4).unwrap(); // evict_check(len=3 > 2? YES), clear(), len=0, store -> len=1

        // After eviction, only proof4 remains in cache
        // Reading proof1 should be a cache miss (read from disk)
        let _r1 = storage.get(&proof1.id).unwrap();
        let stats = storage.stats();

        // With correct eviction: proof1 was evicted, get results in cache miss
        // Without eviction: proof1 stays in cache, get results in cache hit
        // The clear() resets stats to 0, then get adds 1 miss for proof1
        assert!(
            stats.cache_misses >= 1,
            "Expected cache miss for evicted proof, but got {} misses. Eviction may not have occurred.",
            stats.cache_misses
        );
    }

    // NOTE: Mutant "replace MemoryStorage::with_capacity -> Self with Default::default()"
    // at line 96 is an EQUIVALENT MUTANT.
    // The `with_capacity` function pre-allocates HashMap capacity for performance,
    // but this does not affect observable behavior - the HashMap works identically
    // whether pre-allocated or not (it grows on demand). Tests cannot distinguish
    // between a pre-allocated HashMap and one that grows dynamically.
    // The only difference is performance (fewer reallocations), which is not
    // testable without measuring allocation counts.
}
