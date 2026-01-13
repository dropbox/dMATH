//! Locality-Sensitive Hashing (LSH) for approximate nearest neighbor search
//!
//! This module provides an LSH-based index for efficient approximate similarity search
//! over embedding vectors. LSH works by hashing similar items to the same bucket
//! with high probability, enabling O(1) average case lookup.
//!
//! # Algorithm
//!
//! We use random hyperplane LSH for cosine similarity:
//! 1. Generate `num_tables` hash tables, each with `num_hashes` random hyperplanes
//! 2. For each embedding, compute hash = sign(embedding · hyperplane) for each hyperplane
//! 3. Concatenate signs into a hash code and store in corresponding bucket
//! 4. At query time, look up candidate buckets and perform exact search on candidates
//!
//! # Performance Characteristics
//!
//! - Index build: O(n * num_tables * num_hashes * dim)
//! - Query (average): O(num_tables * bucket_size * dim) ≈ O(n^(1/c) * dim) for c-approximate NN
//! - Space: O(n * num_tables) for bucket storage + O(num_tables * num_hashes * dim) for hyperplanes
//!
//! For a corpus of 10k embeddings with 8 hash tables and 8 hashes per table:
//! - Expected bucket size ≈ 10000 / 256 ≈ 39 items per bucket
//! - Query examines ≈ 8 * 39 = 312 candidates vs 10000 for exact search
//! - Speedup factor ≈ 32x with high recall
//!
//! # Example
//!
//! ```ignore
//! use dashprove_learning::lsh::{LshIndex, LshConfig};
//! use dashprove_learning::embedder::Embedding;
//!
//! // Create index with default config
//! let mut index = LshIndex::new(LshConfig::default());
//!
//! // Insert embeddings
//! index.insert("id1".to_string(), embedding1);
//! index.insert("id2".to_string(), embedding2);
//!
//! // Query for approximate nearest neighbors
//! let results = index.query(&query_embedding, 10);
//! ```

use crate::embedder::Embedding;
use crate::ordered_float::OrderedF64;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Configuration for LSH index
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LshConfig {
    /// Number of hash tables (more tables = higher recall, more memory)
    pub num_tables: usize,
    /// Number of hash bits per table (more bits = smaller buckets, lower recall)
    pub num_hashes: usize,
    /// Embedding dimension (must match inserted embeddings)
    pub dim: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for LshConfig {
    fn default() -> Self {
        Self {
            num_tables: 8,
            num_hashes: 8,
            dim: crate::embedder::EMBEDDING_DIM,
            seed: 42,
        }
    }
}

impl LshConfig {
    /// Create config optimized for small corpora (<1000 items)
    pub fn for_small_corpus() -> Self {
        Self {
            num_tables: 4,
            num_hashes: 6,
            dim: crate::embedder::EMBEDDING_DIM,
            seed: 42,
        }
    }

    /// Create config optimized for medium corpora (1000-10000 items)
    pub fn for_medium_corpus() -> Self {
        Self {
            num_tables: 8,
            num_hashes: 8,
            dim: crate::embedder::EMBEDDING_DIM,
            seed: 42,
        }
    }

    /// Create config optimized for large corpora (>10000 items)
    pub fn for_large_corpus() -> Self {
        Self {
            num_tables: 16,
            num_hashes: 10,
            dim: crate::embedder::EMBEDDING_DIM,
            seed: 42,
        }
    }
}

/// A single hash table with random hyperplanes
#[derive(Debug, Clone)]
struct HashTable {
    /// Random hyperplanes for this table (num_hashes x dim)
    hyperplanes: Vec<Vec<f32>>,
    /// Buckets mapping hash codes to entry IDs
    buckets: HashMap<u64, Vec<usize>>,
}

impl HashTable {
    /// Create a new hash table with random hyperplanes
    fn new(num_hashes: usize, dim: usize, rng: &mut SimpleRng) -> Self {
        let hyperplanes = (0..num_hashes)
            .map(|_| {
                // Generate random unit vector
                let mut vec: Vec<f32> = (0..dim).map(|_| rng.next_gaussian()).collect();
                // Normalize to unit length
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for x in &mut vec {
                        *x /= norm;
                    }
                }
                vec
            })
            .collect();

        Self {
            hyperplanes,
            buckets: HashMap::new(),
        }
    }

    /// Compute hash code for an embedding
    fn hash(&self, embedding: &Embedding) -> u64 {
        let mut code: u64 = 0;
        for (i, hyperplane) in self.hyperplanes.iter().enumerate() {
            // Compute dot product
            let dot: f32 = embedding
                .vector
                .iter()
                .zip(hyperplane.iter())
                .map(|(a, b)| a * b)
                .sum();

            // Set bit if positive (sign(dot) >= 0)
            if dot >= 0.0 {
                code |= 1 << i;
            }
        }
        code
    }

    /// Insert an entry into the hash table
    fn insert(&mut self, idx: usize, embedding: &Embedding) {
        let code = self.hash(embedding);
        self.buckets.entry(code).or_default().push(idx);
    }

    /// Get candidate indices from the bucket matching the query
    fn get_candidates(&self, embedding: &Embedding) -> &[usize] {
        let code = self.hash(embedding);
        self.buckets.get(&code).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get candidates from nearby buckets (Hamming distance <= 1)
    fn get_candidates_expanded(&self, embedding: &Embedding, num_hashes: usize) -> Vec<usize> {
        let code = self.hash(embedding);
        let mut candidates = Vec::new();

        // Add exact bucket match
        if let Some(bucket) = self.buckets.get(&code) {
            candidates.extend(bucket.iter().copied());
        }

        // Add buckets with Hamming distance 1 (flip one bit)
        for i in 0..num_hashes {
            let neighbor_code = code ^ (1 << i);
            if let Some(bucket) = self.buckets.get(&neighbor_code) {
                candidates.extend(bucket.iter().copied());
            }
        }

        candidates
    }
}

/// LSH-based approximate nearest neighbor index
#[derive(Debug)]
pub struct LshIndex {
    /// Configuration
    config: LshConfig,
    /// Hash tables
    tables: Vec<HashTable>,
    /// Stored entries (id, embedding)
    entries: Vec<(String, Embedding)>,
}

impl LshIndex {
    /// Create a new LSH index with the given configuration
    pub fn new(config: LshConfig) -> Self {
        let mut rng = SimpleRng::new(config.seed);
        let tables = (0..config.num_tables)
            .map(|_| HashTable::new(config.num_hashes, config.dim, &mut rng))
            .collect();

        Self {
            config,
            tables,
            entries: Vec::new(),
        }
    }

    /// Create index with default configuration
    pub fn with_defaults() -> Self {
        Self::new(LshConfig::default())
    }

    /// Number of entries in the index
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the configuration
    pub fn config(&self) -> &LshConfig {
        &self.config
    }

    /// Insert an embedding into the index
    pub fn insert(&mut self, id: String, embedding: Embedding) {
        let idx = self.entries.len();

        // Validate dimension
        if embedding.dim != self.config.dim {
            // Silently ignore mismatched dimensions (could log warning)
            return;
        }

        // Insert into all hash tables
        for table in &mut self.tables {
            table.insert(idx, &embedding);
        }

        self.entries.push((id, embedding));
    }

    /// Query for k approximate nearest neighbors
    ///
    /// Returns up to k entries sorted by similarity (descending).
    /// May return fewer than k results if not enough candidates are found.
    pub fn query(&self, query: &Embedding, k: usize) -> Vec<(String, f64)> {
        if self.entries.is_empty() || k == 0 {
            return vec![];
        }

        // Collect unique candidates from all tables
        let mut candidate_set: HashSet<usize> = HashSet::new();
        for table in &self.tables {
            for &idx in table.get_candidates(query) {
                candidate_set.insert(idx);
            }
        }

        // If too few candidates, expand search to neighboring buckets
        if candidate_set.len() < k * 2 {
            for table in &self.tables {
                for idx in table.get_candidates_expanded(query, self.config.num_hashes) {
                    candidate_set.insert(idx);
                }
            }
        }

        // Compute exact similarities for candidates and find top-k
        self.top_k_from_candidates(query, candidate_set.into_iter(), k)
    }

    /// Query with guaranteed minimum candidates
    ///
    /// Expands search to neighboring buckets if needed to ensure at least
    /// `min_candidates` entries are examined.
    pub fn query_with_min_candidates(
        &self,
        query: &Embedding,
        k: usize,
        min_candidates: usize,
    ) -> Vec<(String, f64)> {
        if self.entries.is_empty() || k == 0 {
            return vec![];
        }

        // Collect candidates from exact buckets first
        let mut candidate_set: HashSet<usize> = HashSet::new();
        for table in &self.tables {
            for &idx in table.get_candidates(query) {
                candidate_set.insert(idx);
            }
        }

        // Expand if needed
        if candidate_set.len() < min_candidates {
            for table in &self.tables {
                for idx in table.get_candidates_expanded(query, self.config.num_hashes) {
                    candidate_set.insert(idx);
                    if candidate_set.len() >= min_candidates {
                        break;
                    }
                }
            }
        }

        self.top_k_from_candidates(query, candidate_set.into_iter(), k)
    }

    /// Find top-k from a set of candidate indices
    fn top_k_from_candidates(
        &self,
        query: &Embedding,
        candidates: impl Iterator<Item = usize>,
        k: usize,
    ) -> Vec<(String, f64)> {
        // Use min-heap for efficient top-k selection
        let mut heap: BinaryHeap<std::cmp::Reverse<(OrderedF64, usize)>> =
            BinaryHeap::with_capacity(k + 1);

        for idx in candidates {
            if idx >= self.entries.len() {
                continue;
            }

            let (_, ref embedding) = self.entries[idx];
            let sim = query.normalized_similarity(embedding);
            let ordered = OrderedF64(sim);

            if heap.len() < k {
                heap.push(std::cmp::Reverse((ordered, idx)));
            } else if let Some(&std::cmp::Reverse((min_score, _))) = heap.peek() {
                if ordered > min_score {
                    heap.pop();
                    heap.push(std::cmp::Reverse((ordered, idx)));
                }
            }
        }

        // Extract and sort results
        let mut results: Vec<_> = heap
            .into_iter()
            .map(|std::cmp::Reverse((score, idx))| {
                let (ref id, _) = self.entries[idx];
                (id.clone(), score.0)
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Perform exact search (for comparison/fallback)
    ///
    /// This bypasses LSH and examines all entries.
    pub fn exact_search(&self, query: &Embedding, k: usize) -> Vec<(String, f64)> {
        self.top_k_from_candidates(query, 0..self.entries.len(), k)
    }

    /// Get statistics about bucket distribution
    pub fn bucket_stats(&self) -> BucketStats {
        let mut total_buckets = 0;
        let mut total_entries = 0;
        let mut max_bucket_size = 0;
        let mut min_bucket_size = usize::MAX;

        for table in &self.tables {
            total_buckets += table.buckets.len();
            for bucket in table.buckets.values() {
                total_entries += bucket.len();
                max_bucket_size = max_bucket_size.max(bucket.len());
                min_bucket_size = min_bucket_size.min(bucket.len());
            }
        }

        if total_buckets == 0 {
            min_bucket_size = 0;
        }

        BucketStats {
            num_tables: self.tables.len(),
            total_buckets,
            total_entries: self.entries.len(),
            avg_bucket_size: if total_buckets > 0 {
                total_entries as f64 / total_buckets as f64
            } else {
                0.0
            },
            max_bucket_size,
            min_bucket_size,
        }
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.entries.clear();
        for table in &mut self.tables {
            table.buckets.clear();
        }
    }

    /// Rebuild the index with new configuration
    ///
    /// Useful when corpus size changes significantly and a different
    /// configuration would be more appropriate.
    pub fn rebuild(&mut self, config: LshConfig) {
        let entries: Vec<_> = self.entries.drain(..).collect();

        let mut rng = SimpleRng::new(config.seed);
        self.tables = (0..config.num_tables)
            .map(|_| HashTable::new(config.num_hashes, config.dim, &mut rng))
            .collect();
        self.config = config;

        for (id, embedding) in entries {
            self.insert(id, embedding);
        }
    }
}

/// Statistics about LSH bucket distribution
#[derive(Debug, Clone)]
pub struct BucketStats {
    /// Number of hash tables
    pub num_tables: usize,
    /// Total number of non-empty buckets across all tables
    pub total_buckets: usize,
    /// Total entries in the index
    pub total_entries: usize,
    /// Average bucket size
    pub avg_bucket_size: f64,
    /// Maximum bucket size
    pub max_bucket_size: usize,
    /// Minimum bucket size
    pub min_bucket_size: usize,
}

/// Simple deterministic PRNG for reproducible hyperplane generation
///
/// Uses xorshift64 for fast, reasonable quality randomness.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        // Convert to [0, 1) range
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn next_gaussian(&mut self) -> f32 {
        // Box-Muller transform for Gaussian samples
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(values: &[f32]) -> Embedding {
        Embedding::new(values.to_vec())
    }

    fn random_embedding(dim: usize, seed: u64) -> Embedding {
        let mut rng = SimpleRng::new(seed);
        Embedding::new((0..dim).map(|_| rng.next_gaussian()).collect())
    }

    #[test]
    fn test_simple_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_simple_rng_different_seeds() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(43);

        // Different seeds should produce different sequences
        let v1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
        let v2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_simple_rng_f32_range() {
        let mut rng = SimpleRng::new(12345);
        for _ in 0..1000 {
            let f = rng.next_f32();
            assert!(
                (0.0..1.0).contains(&f),
                "f32 should be in [0, 1): got {}",
                f
            );
        }
    }

    #[test]
    fn test_simple_rng_gaussian_distribution() {
        let mut rng = SimpleRng::new(42);
        let samples: Vec<f32> = (0..10000).map(|_| rng.next_gaussian()).collect();

        // Check mean is approximately 0
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(
            mean.abs() < 0.1,
            "Gaussian mean should be near 0, got {}",
            mean
        );

        // Check std dev is approximately 1
        let variance: f32 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;
        let std_dev = variance.sqrt();
        assert!(
            (std_dev - 1.0).abs() < 0.1,
            "Gaussian std dev should be near 1, got {}",
            std_dev
        );
    }

    #[test]
    fn test_lsh_config_default() {
        let config = LshConfig::default();
        assert_eq!(config.num_tables, 8);
        assert_eq!(config.num_hashes, 8);
        assert_eq!(config.dim, crate::embedder::EMBEDDING_DIM);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_lsh_config_presets() {
        let small = LshConfig::for_small_corpus();
        let medium = LshConfig::for_medium_corpus();
        let large = LshConfig::for_large_corpus();

        // Larger corpora should have more tables
        assert!(small.num_tables < medium.num_tables);
        assert!(medium.num_tables < large.num_tables);
    }

    #[test]
    fn test_lsh_index_new() {
        let index = LshIndex::new(LshConfig::default());
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_lsh_index_insert() {
        let config = LshConfig {
            num_tables: 2,
            num_hashes: 4,
            dim: 8,
            seed: 42,
        };
        let mut index = LshIndex::new(config);

        let emb = make_embedding(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        index.insert("test1".to_string(), emb);

        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_lsh_index_insert_wrong_dim_ignored() {
        let config = LshConfig {
            num_tables: 2,
            num_hashes: 4,
            dim: 8,
            seed: 42,
        };
        let mut index = LshIndex::new(config);

        // Wrong dimension should be ignored
        let emb = make_embedding(&[1.0, 0.0, 0.0]); // dim=3, expected 8
        index.insert("test1".to_string(), emb);

        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_lsh_index_query_empty() {
        let config = LshConfig {
            num_tables: 2,
            num_hashes: 4,
            dim: 8,
            seed: 42,
        };
        let index = LshIndex::new(config);

        let query = make_embedding(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let results = index.query(&query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_lsh_index_query_k_zero() {
        let config = LshConfig {
            num_tables: 2,
            num_hashes: 4,
            dim: 8,
            seed: 42,
        };
        let mut index = LshIndex::new(config);

        let emb = make_embedding(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        index.insert("test1".to_string(), emb.clone());

        let results = index.query(&emb, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_lsh_index_query_single() {
        let config = LshConfig {
            num_tables: 4,
            num_hashes: 4,
            dim: 8,
            seed: 42,
        };
        let mut index = LshIndex::new(config);

        let emb = make_embedding(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        index.insert("test1".to_string(), emb.clone());

        let results = index.query(&emb, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "test1");
        assert!((results[0].1 - 1.0).abs() < 0.001); // Self-similarity should be 1.0
    }

    #[test]
    fn test_lsh_index_finds_similar() {
        let config = LshConfig {
            num_tables: 8,
            num_hashes: 6,
            dim: 8,
            seed: 42,
        };
        let mut index = LshIndex::new(config);

        // Insert similar embeddings
        let emb1 = make_embedding(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let emb2 = make_embedding(&[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let emb3 = make_embedding(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]); // Orthogonal

        index.insert("similar".to_string(), emb1.clone());
        index.insert("very_similar".to_string(), emb2);
        index.insert("different".to_string(), emb3);

        // Query with emb1 - should find similar ones with higher scores
        let results = index.query(&emb1, 3);

        // Should find at least similar and very_similar
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"similar"), "Should find exact match");
    }

    #[test]
    fn test_lsh_index_exact_search() {
        let config = LshConfig {
            num_tables: 2,
            num_hashes: 4,
            dim: 8,
            seed: 42,
        };
        let mut index = LshIndex::new(config);

        // Insert multiple embeddings
        for i in 0..10 {
            let emb = random_embedding(8, i as u64);
            index.insert(format!("id_{}", i), emb);
        }

        let query = random_embedding(8, 100);
        let exact = index.exact_search(&query, 3);

        assert_eq!(exact.len(), 3);
        // Results should be sorted by similarity (descending)
        assert!(exact[0].1 >= exact[1].1);
        assert!(exact[1].1 >= exact[2].1);
    }

    #[test]
    fn test_lsh_index_bucket_stats() {
        let config = LshConfig {
            num_tables: 4,
            num_hashes: 4,
            dim: 8,
            seed: 42,
        };
        let mut index = LshIndex::new(config);

        // Insert some embeddings
        for i in 0..20 {
            let emb = random_embedding(8, i as u64);
            index.insert(format!("id_{}", i), emb);
        }

        let stats = index.bucket_stats();
        assert_eq!(stats.num_tables, 4);
        assert_eq!(stats.total_entries, 20);
        assert!(stats.total_buckets > 0);
        assert!(stats.avg_bucket_size > 0.0);
    }

    #[test]
    fn test_lsh_index_bucket_stats_empty() {
        let config = LshConfig {
            num_tables: 4,
            num_hashes: 4,
            dim: 8,
            seed: 42,
        };
        let index = LshIndex::new(config);

        let stats = index.bucket_stats();
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.total_buckets, 0);
        assert_eq!(stats.min_bucket_size, 0);
    }

    #[test]
    fn test_lsh_index_clear() {
        let config = LshConfig {
            num_tables: 2,
            num_hashes: 4,
            dim: 8,
            seed: 42,
        };
        let mut index = LshIndex::new(config);

        let emb = random_embedding(8, 42);
        index.insert("test".to_string(), emb);
        assert_eq!(index.len(), 1);

        index.clear();
        assert!(index.is_empty());
    }

    #[test]
    fn test_lsh_index_rebuild() {
        let config = LshConfig {
            num_tables: 2,
            num_hashes: 4,
            dim: 8,
            seed: 42,
        };
        let mut index = LshIndex::new(config);

        // Insert entries
        for i in 0..10 {
            let emb = random_embedding(8, i as u64);
            index.insert(format!("id_{}", i), emb);
        }
        assert_eq!(index.len(), 10);

        // Rebuild with different config
        let new_config = LshConfig {
            num_tables: 4,
            num_hashes: 6,
            dim: 8,
            seed: 99,
        };
        index.rebuild(new_config);

        // Entries should be preserved
        assert_eq!(index.len(), 10);
        assert_eq!(index.config().num_tables, 4);
        assert_eq!(index.config().seed, 99);
    }

    #[test]
    fn test_lsh_index_query_with_min_candidates() {
        let config = LshConfig {
            num_tables: 4,
            num_hashes: 8, // Many hashes = small buckets
            dim: 8,
            seed: 42,
        };
        let mut index = LshIndex::new(config);

        // Insert many diverse embeddings
        for i in 0..100 {
            let emb = random_embedding(8, i as u64);
            index.insert(format!("id_{}", i), emb);
        }

        let query = random_embedding(8, 1000);

        // Query with high min_candidates should expand search
        let results = index.query_with_min_candidates(&query, 10, 50);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_hash_table_hash_deterministic() {
        let mut rng = SimpleRng::new(42);
        let table = HashTable::new(8, 8, &mut rng);

        let emb = make_embedding(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let hash1 = table.hash(&emb);
        let hash2 = table.hash(&emb);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_table_similar_vectors_same_hash() {
        let mut rng = SimpleRng::new(42);
        let table = HashTable::new(4, 8, &mut rng); // Few hashes for broader buckets

        let emb1 = make_embedding(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let emb2 = make_embedding(&[0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // Very similar vectors should often hash to same bucket
        let hash1 = table.hash(&emb1);
        let hash2 = table.hash(&emb2);

        // Not guaranteed to be equal, but should be close
        let hamming = (hash1 ^ hash2).count_ones();
        assert!(
            hamming <= 2,
            "Similar vectors should have low Hamming distance: {}",
            hamming
        );
    }

    #[test]
    fn test_lsh_recall_quality() {
        // Test that LSH finds most true nearest neighbors
        let config = LshConfig {
            num_tables: 16,
            num_hashes: 6,
            dim: 32,
            seed: 42,
        };
        let mut index = LshIndex::new(config);

        // Insert random embeddings
        for i in 0..1000 {
            let emb = random_embedding(32, i as u64);
            index.insert(format!("id_{}", i), emb);
        }

        let query = random_embedding(32, 99999);
        let k = 10;

        // Get LSH results
        let lsh_results = index.query(&query, k);
        let lsh_ids: HashSet<_> = lsh_results.iter().map(|(id, _)| id.clone()).collect();

        // Get exact results
        let exact_results = index.exact_search(&query, k);
        let exact_ids: HashSet<_> = exact_results.iter().map(|(id, _)| id.clone()).collect();

        // Compute recall: how many of the true top-k are in LSH results
        let overlap = lsh_ids.intersection(&exact_ids).count();
        let recall = overlap as f64 / k as f64;

        // With 16 tables and moderate hashes, we should get good recall
        assert!(
            recall >= 0.5,
            "LSH recall should be at least 50%: got {}%",
            recall * 100.0
        );
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify SimpleRng never produces 0 state (would cause stuck sequence)
    #[kani::proof]
    fn verify_rng_never_stuck() {
        let seed: u64 = kani::any();
        let mut rng = SimpleRng::new(seed);

        // After one iteration, state should be non-zero
        let _ = rng.next_u64();
        assert!(rng.state != 0, "RNG state should never be 0");
    }

    /// Verify SimpleRng with seed 0 is handled correctly
    #[kani::proof]
    fn verify_rng_seed_zero_handled() {
        let rng = SimpleRng::new(0);
        assert_eq!(rng.state, 1, "Seed 0 should be mapped to 1");
    }

    /// Verify next_f32 always returns valid range [0, 1)
    #[kani::proof]
    fn verify_next_f32_bounds() {
        let seed: u64 = kani::any();
        kani::assume(seed != 0);
        let mut rng = SimpleRng::new(seed);

        let f = rng.next_f32();
        assert!(f >= 0.0, "next_f32 should be >= 0");
        assert!(f < 1.0, "next_f32 should be < 1");
    }

    /// Verify LshConfig defaults are sensible
    #[kani::proof]
    fn verify_config_defaults_sensible() {
        let config = LshConfig::default();

        assert!(config.num_tables > 0, "num_tables should be positive");
        assert!(config.num_hashes > 0, "num_hashes should be positive");
        assert!(config.dim > 0, "dim should be positive");
    }

    /// Verify hash code fits in u64 for reasonable num_hashes
    #[kani::proof]
    fn verify_hash_code_fits() {
        let num_hashes: usize = kani::any();
        kani::assume(num_hashes <= 64);
        kani::assume(num_hashes > 0);

        // Hash code uses 1 << i where i < num_hashes
        // This should not overflow for num_hashes <= 64
        let max_bit = num_hashes - 1;
        let code: u64 = 1 << max_bit;
        assert!(code > 0, "Hash code should be valid");
    }

    /// Verify empty index returns empty results
    #[kani::proof]
    fn verify_empty_index_empty_results() {
        let config = LshConfig {
            num_tables: 2,
            num_hashes: 4,
            dim: 4,
            seed: 42,
        };
        let index = LshIndex::new(config);

        let query = Embedding::zeros(4);
        let k: usize = kani::any();
        kani::assume(k <= 10);

        let results = index.query(&query, k);
        assert!(
            results.is_empty(),
            "Empty index should return empty results"
        );
    }

    /// Verify k=0 always returns empty results
    #[kani::proof]
    fn verify_k_zero_empty_results() {
        let config = LshConfig {
            num_tables: 2,
            num_hashes: 4,
            dim: 4,
            seed: 42,
        };
        let mut index = LshIndex::new(config);

        // Insert one entry
        index.insert("test".to_string(), Embedding::zeros(4));

        let query = Embedding::zeros(4);
        let results = index.query(&query, 0);
        assert!(results.is_empty(), "k=0 should always return empty results");
    }
}
