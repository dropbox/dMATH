//! Hybrid PQ-LSH index for memory-efficient approximate search
//!
//! This module combines Product Quantization (PQ) with Locality-Sensitive Hashing (LSH)
//! to provide memory-efficient approximate nearest neighbor search:
//!
//! - **LSH for candidate filtering**: O(1) lookup to find candidate buckets
//! - **PQ for compact storage**: 48x compression of embedding storage
//! - **PQ distance for reranking**: Fast approximate distance using precomputed tables
//!
//! # Architecture
//!
//! ```text
//! Query embedding
//!       │
//!       ▼
//! ┌─────────────┐
//! │    LSH      │ → Candidate bucket IDs (O(1) lookup)
//! └─────────────┘
//!       │
//!       ▼
//! ┌─────────────┐
//! │  PQ Rerank  │ → Score candidates using PQ distance table (O(M) per candidate)
//! └─────────────┘
//!       │
//!       ▼
//!    Top-k results
//! ```
//!
//! # Memory Comparison for 100K embeddings (96 dimensions)
//!
//! | Method | Memory | Speed |
//! |--------|--------|-------|
//! | Exact search | 38.4 MB | O(N*D) per query |
//! | LSH only | 38.4 MB + hash tables | O(bucket_size * D) |
//! | PQ only | 0.8 MB + codebooks | O(N*M) per query |
//! | **PQ-LSH** | **0.8 MB + codebooks + hash** | **O(bucket_size * M)** |
//!
//! PQ-LSH achieves both memory efficiency (from PQ) and query speed (from LSH).

use super::types::ProofId;
use crate::embedder::Embedding;
use crate::lsh::{LshConfig, LshIndex};
use crate::pq::{PqConfig, PqError, ProductQuantizer};
use crate::similarity::SimilarProof;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for hybrid PQ-LSH index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqLshConfig {
    /// LSH configuration for candidate filtering
    pub lsh: LshConfig,
    /// PQ configuration for compact storage
    pub pq: PqConfig,
}

impl Default for PqLshConfig {
    fn default() -> Self {
        Self {
            lsh: LshConfig::default(),
            pq: PqConfig::fast(),
        }
    }
}

impl PqLshConfig {
    /// Config optimized for small corpora (<1000 entries)
    pub fn for_small_corpus() -> Self {
        Self {
            lsh: LshConfig::for_small_corpus(),
            pq: PqConfig::fast(),
        }
    }

    /// Config optimized for medium corpora (1000-10000 entries)
    pub fn for_medium_corpus() -> Self {
        Self {
            lsh: LshConfig::for_medium_corpus(),
            pq: PqConfig::fast(),
        }
    }

    /// Config optimized for large corpora (>10000 entries)
    pub fn for_large_corpus() -> Self {
        Self {
            lsh: LshConfig::for_large_corpus(),
            pq: PqConfig::default(), // Higher accuracy for large corpus
        }
    }

    /// Config optimized for very large corpora (>100000 entries)
    pub fn for_huge_corpus() -> Self {
        Self {
            lsh: LshConfig {
                num_tables: 32,
                num_hashes: 12,
                ..LshConfig::default()
            },
            pq: PqConfig::accurate(),
        }
    }
}

/// Hybrid PQ-LSH index for ProofCorpus
///
/// Uses LSH for fast candidate filtering and PQ for memory-efficient
/// storage and distance computation.
#[derive(Debug)]
pub struct ProofCorpusPqLsh {
    /// LSH index for candidate generation
    lsh: LshIndex,
    /// Product quantizer for encoding embeddings
    quantizer: ProductQuantizer,
    /// PQ codes for each entry: entries[i] = (proof_id, pq_codes)
    entries: Vec<(ProofId, Vec<u8>)>,
    /// Mapping from LSH IDs to entry indices
    id_to_idx: HashMap<String, usize>,
    /// Configuration used
    config: PqLshConfig,
}

impl ProofCorpusPqLsh {
    /// Build a PQ-LSH index from a ProofCorpus
    ///
    /// Requires at least `pq.num_centroids` embeddings for training.
    /// Returns None if insufficient embeddings are available.
    pub fn build(corpus: &super::storage::ProofCorpus) -> Result<Option<Self>, PqError> {
        Self::build_with_config(corpus, PqLshConfig::default())
    }

    /// Build with custom configuration
    pub fn build_with_config(
        corpus: &super::storage::ProofCorpus,
        config: PqLshConfig,
    ) -> Result<Option<Self>, PqError> {
        // Collect embeddings for training
        let training_data: Vec<Embedding> = corpus
            .entries()
            .filter_map(|e| e.embedding.clone())
            .collect();

        if training_data.is_empty() {
            return Ok(None);
        }

        if training_data.len() < config.pq.num_centroids {
            return Err(PqError::InsufficientData(format!(
                "Need at least {} embeddings for {} centroids, got {}",
                config.pq.num_centroids,
                config.pq.num_centroids,
                training_data.len()
            )));
        }

        // Train the PQ quantizer
        let quantizer = ProductQuantizer::train(&training_data, config.pq.clone())?;

        // Build LSH index and encode entries
        let mut lsh = LshIndex::new(config.lsh.clone());
        let mut entries = Vec::new();
        let mut id_to_idx = HashMap::new();

        for entry in corpus.entries() {
            if let Some(ref embedding) = entry.embedding {
                let idx = entries.len();
                let lsh_id = entry.id.0.clone();

                // Add to LSH for candidate filtering
                lsh.insert(lsh_id.clone(), embedding.clone());

                // Store PQ codes
                let pq_codes = quantizer.encode(embedding);
                entries.push((entry.id.clone(), pq_codes));
                id_to_idx.insert(lsh_id, idx);
            }
        }

        if entries.is_empty() {
            return Ok(None);
        }

        Ok(Some(Self {
            lsh,
            quantizer,
            entries,
            id_to_idx,
            config,
        }))
    }

    /// Build with configuration auto-selected based on corpus size
    pub fn build_auto_config(
        corpus: &super::storage::ProofCorpus,
    ) -> Result<Option<Self>, PqError> {
        let count = corpus.embedding_count();
        let config = if count < 1000 {
            PqLshConfig::for_small_corpus()
        } else if count < 10000 {
            PqLshConfig::for_medium_corpus()
        } else if count < 100000 {
            PqLshConfig::for_large_corpus()
        } else {
            PqLshConfig::for_huge_corpus()
        };

        Self::build_with_config(corpus, config)
    }

    /// Number of indexed entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> PqLshMemoryStats {
        let num_entries = self.entries.len();
        let num_subspaces = self.config.pq.num_subspaces;

        // PQ codes memory: entries * subspaces * 1 byte
        let pq_codes_bytes = num_entries * num_subspaces;

        // Codebook memory
        let codebook_bytes = self.quantizer.codebook_size_bytes();

        // LSH hash tables (rough estimate)
        let lsh_stats = self.lsh.bucket_stats();
        let lsh_bytes = lsh_stats.total_buckets * 8 + // bucket overhead
                        num_entries * 8 * self.config.lsh.num_tables; // entry storage

        // Compare to raw embedding storage
        let raw_bytes = num_entries * self.quantizer.dim() * 4;

        PqLshMemoryStats {
            num_entries,
            pq_codes_bytes,
            codebook_bytes,
            lsh_bytes,
            total_bytes: pq_codes_bytes + codebook_bytes + lsh_bytes,
            raw_equivalent_bytes: raw_bytes,
            compression_ratio: if pq_codes_bytes + codebook_bytes > 0 {
                raw_bytes as f64 / (pq_codes_bytes + codebook_bytes) as f64
            } else {
                0.0
            },
        }
    }

    /// Find approximate k nearest neighbors
    ///
    /// Uses LSH to find candidate buckets, then ranks candidates using
    /// PQ asymmetric distance computation.
    pub fn find_similar_approximate(
        &self,
        corpus: &super::storage::ProofCorpus,
        query: &Embedding,
        k: usize,
    ) -> Vec<SimilarProof> {
        if self.entries.is_empty() || k == 0 {
            return vec![];
        }

        // Get LSH candidates
        let lsh_results = self.lsh.query(query, k * 4); // Over-retrieve for better recall

        // Build PQ distance table for fast reranking
        let distance_table = self.quantizer.build_distance_table(query);

        // Rerank candidates using PQ distance
        let mut scored: Vec<(usize, f64)> = lsh_results
            .iter()
            .filter_map(|(lsh_id, _)| {
                let idx = *self.id_to_idx.get(lsh_id)?;
                let (_, ref codes) = self.entries[idx];
                let dist = self.quantizer.distance_with_table(&distance_table, codes);
                let sim = ProductQuantizer::distance_to_similarity(dist);
                Some((idx, sim))
            })
            .collect();

        // Sort by similarity (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        // Convert to SimilarProof
        scored
            .into_iter()
            .filter_map(|(idx, sim)| {
                let (proof_id, _) = &self.entries[idx];
                let entry = corpus.get(proof_id)?;
                Some(SimilarProof {
                    id: entry.id.clone(),
                    property: entry.property.clone(),
                    backend: entry.backend,
                    tactics: entry.tactics.clone(),
                    similarity: sim,
                })
            })
            .collect()
    }

    /// Find similar with expanded candidate search
    ///
    /// Ensures at least `min_candidates` are examined for better recall.
    pub fn find_similar_with_min_candidates(
        &self,
        corpus: &super::storage::ProofCorpus,
        query: &Embedding,
        k: usize,
        min_candidates: usize,
    ) -> Vec<SimilarProof> {
        if self.entries.is_empty() || k == 0 {
            return vec![];
        }

        // Get expanded LSH candidates
        let lsh_results = self
            .lsh
            .query_with_min_candidates(query, min_candidates, min_candidates);

        // Build PQ distance table
        let distance_table = self.quantizer.build_distance_table(query);

        // Score all candidates
        let mut scored: Vec<(usize, f64)> = lsh_results
            .iter()
            .filter_map(|(lsh_id, _)| {
                let idx = *self.id_to_idx.get(lsh_id)?;
                let (_, ref codes) = self.entries[idx];
                let dist = self.quantizer.distance_with_table(&distance_table, codes);
                let sim = ProductQuantizer::distance_to_similarity(dist);
                Some((idx, sim))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        scored
            .into_iter()
            .filter_map(|(idx, sim)| {
                let (proof_id, _) = &self.entries[idx];
                let entry = corpus.get(proof_id)?;
                Some(SimilarProof {
                    id: entry.id.clone(),
                    property: entry.property.clone(),
                    backend: entry.backend,
                    tactics: entry.tactics.clone(),
                    similarity: sim,
                })
            })
            .collect()
    }

    /// Exact search using PQ (bypasses LSH, examines all entries)
    ///
    /// Useful for comparing recall quality.
    pub fn find_similar_exact_pq(
        &self,
        corpus: &super::storage::ProofCorpus,
        query: &Embedding,
        k: usize,
    ) -> Vec<SimilarProof> {
        if self.entries.is_empty() || k == 0 {
            return vec![];
        }

        let distance_table = self.quantizer.build_distance_table(query);

        // Score all entries
        let mut scored: Vec<(usize, f64)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(idx, (_, codes))| {
                let dist = self.quantizer.distance_with_table(&distance_table, codes);
                let sim = ProductQuantizer::distance_to_similarity(dist);
                (idx, sim)
            })
            .collect();

        // Partial sort for efficiency
        let k = k.min(scored.len());
        if k > 0 {
            scored.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        scored.truncate(k);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .filter_map(|(idx, sim)| {
                let (proof_id, _) = &self.entries[idx];
                let entry = corpus.get(proof_id)?;
                Some(SimilarProof {
                    id: entry.id.clone(),
                    property: entry.property.clone(),
                    backend: entry.backend,
                    tactics: entry.tactics.clone(),
                    similarity: sim,
                })
            })
            .collect()
    }

    /// Measure recall of PQ-LSH vs exact PQ search
    ///
    /// Returns fraction of exact top-k results found by approximate search.
    pub fn measure_recall(
        &self,
        corpus: &super::storage::ProofCorpus,
        k: usize,
        num_samples: usize,
    ) -> f64 {
        if self.is_empty() || k == 0 || num_samples == 0 {
            return 0.0;
        }

        let step = self.len().max(1) / num_samples.max(1);
        let step = step.max(1);

        let mut total_recall = 0.0;
        let mut samples_tested = 0;

        let entries: Vec<_> = corpus.entries().collect();
        for i in (0..entries.len()).step_by(step) {
            if samples_tested >= num_samples {
                break;
            }

            let entry = entries[i];
            if let Some(ref query) = entry.embedding {
                let approx = self.find_similar_approximate(corpus, query, k);
                let exact = self.find_similar_exact_pq(corpus, query, k);

                let approx_ids: std::collections::HashSet<_> =
                    approx.iter().map(|r| r.id.clone()).collect();
                let exact_ids: std::collections::HashSet<_> =
                    exact.iter().map(|r| r.id.clone()).collect();

                if !exact_ids.is_empty() {
                    let overlap = approx_ids.intersection(&exact_ids).count();
                    total_recall += overlap as f64 / exact_ids.len() as f64;
                    samples_tested += 1;
                }
            }
        }

        if samples_tested > 0 {
            total_recall / samples_tested as f64
        } else {
            0.0
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &PqLshConfig {
        &self.config
    }

    /// Get the underlying quantizer
    ///
    /// Useful for persisting the trained quantizer separately.
    pub fn quantizer(&self) -> &ProductQuantizer {
        &self.quantizer
    }

    /// Save the trained PQ quantizer to a file
    ///
    /// The quantizer contains the trained codebooks which are expensive to compute.
    /// Saving allows reusing the same quantizer without retraining.
    pub fn save_quantizer<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), crate::LearningError> {
        self.quantizer.save_to_file(path)
    }

    /// Build a PQ-LSH index using a pre-trained quantizer
    ///
    /// This skips the expensive k-means training step by reusing an existing quantizer.
    /// The quantizer must have been trained on similar data for good results.
    pub fn build_with_quantizer(
        corpus: &super::storage::ProofCorpus,
        quantizer: ProductQuantizer,
        config: PqLshConfig,
    ) -> Option<Self> {
        // Build LSH index and encode entries using pre-trained quantizer
        let mut lsh = LshIndex::new(config.lsh.clone());
        let mut entries = Vec::new();
        let mut id_to_idx = HashMap::new();

        for entry in corpus.entries() {
            if let Some(ref embedding) = entry.embedding {
                let idx = entries.len();
                let lsh_id = entry.id.0.clone();

                // Add to LSH for candidate filtering
                lsh.insert(lsh_id.clone(), embedding.clone());

                // Store PQ codes using pre-trained quantizer
                let pq_codes = quantizer.encode(embedding);
                entries.push((entry.id.clone(), pq_codes));
                id_to_idx.insert(lsh_id, idx);
            }
        }

        if entries.is_empty() {
            return None;
        }

        Some(Self {
            lsh,
            quantizer,
            entries,
            id_to_idx,
            config,
        })
    }

    /// Build a PQ-LSH index, loading quantizer from file if available
    ///
    /// If a quantizer file exists at the given path, it will be loaded and used.
    /// Otherwise, a new quantizer is trained and saved to the path.
    ///
    /// This is useful for incremental updates where you want to reuse the same
    /// quantizer across runs without retraining.
    pub fn build_with_cached_quantizer<P: AsRef<std::path::Path>>(
        corpus: &super::storage::ProofCorpus,
        quantizer_path: P,
        config: PqLshConfig,
    ) -> Result<Option<Self>, crate::LearningError> {
        let path = quantizer_path.as_ref();

        // Try to load existing quantizer
        if let Some(quantizer) = ProductQuantizer::load_if_exists(path)? {
            // Verify dimension matches
            if quantizer.dim() == crate::embedder::EMBEDDING_DIM {
                return Ok(Self::build_with_quantizer(corpus, quantizer, config));
            }
            // Dimension mismatch - fall through to retrain
        }

        // No cached quantizer or dimension mismatch - train new one
        match Self::build_with_config(corpus, config)? {
            Some(index) => {
                // Cache the trained quantizer
                index.save_quantizer(path)?;
                Ok(Some(index))
            }
            None => Ok(None),
        }
    }
}

/// Memory usage statistics for PQ-LSH index
#[derive(Debug, Clone)]
pub struct PqLshMemoryStats {
    /// Number of indexed entries
    pub num_entries: usize,
    /// Bytes used for PQ codes
    pub pq_codes_bytes: usize,
    /// Bytes used for PQ codebooks
    pub codebook_bytes: usize,
    /// Bytes used for LSH tables (estimate)
    pub lsh_bytes: usize,
    /// Total memory usage
    pub total_bytes: usize,
    /// Equivalent raw storage size
    pub raw_equivalent_bytes: usize,
    /// Compression ratio (raw / compressed)
    pub compression_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::EMBEDDING_DIM;
    use crate::LearnableResult;
    use dashprove_backends::traits::{BackendId, VerificationStatus};
    use dashprove_usl::ast::{Expr, Invariant, Property};
    use std::time::Duration;

    fn make_property(name: &str) -> Property {
        Property::Invariant(Invariant {
            name: name.to_string(),
            body: Expr::Bool(true),
        })
    }

    fn make_result(name: &str) -> LearnableResult {
        LearnableResult {
            property: make_property(name),
            backend: BackendId::Lean4,
            status: VerificationStatus::Proven,
            tactics: vec!["simp".to_string()],
            time_taken: Duration::from_millis(100),
            proof_output: None,
        }
    }

    fn random_embedding(seed: u64) -> Embedding {
        let mut state = seed;
        let mut vector = Vec::with_capacity(EMBEDDING_DIM);
        for _ in 0..EMBEDDING_DIM {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let f = (state as f32) / (u64::MAX as f32);
            vector.push(f * 2.0 - 1.0);
        }
        // Normalize
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut vector {
                *x /= norm;
            }
        }
        Embedding::new(vector)
    }

    #[test]
    fn test_pq_lsh_config_default() {
        let config = PqLshConfig::default();
        assert_eq!(config.lsh.num_tables, 8);
        assert_eq!(config.pq.num_centroids, 64); // fast config
    }

    #[test]
    fn test_pq_lsh_config_presets() {
        let small = PqLshConfig::for_small_corpus();
        let large = PqLshConfig::for_large_corpus();
        let huge = PqLshConfig::for_huge_corpus();

        assert!(small.lsh.num_tables < large.lsh.num_tables);
        assert!(large.lsh.num_tables < huge.lsh.num_tables);
    }

    #[test]
    fn test_build_empty_corpus() {
        let corpus = super::super::storage::ProofCorpus::new();
        let result = ProofCorpusPqLsh::build(&corpus);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_build_insufficient_embeddings() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Insert fewer embeddings than required centroids
        for i in 0..10 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        // PqConfig::fast requires 64 centroids, so 10 embeddings should fail
        let result = ProofCorpusPqLsh::build(&corpus);
        assert!(matches!(result, Err(PqError::InsufficientData(_))));
    }

    #[test]
    fn test_build_with_sufficient_embeddings() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Insert enough embeddings (need at least 64 for fast config)
        for i in 0..100 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let index = ProofCorpusPqLsh::build(&corpus).unwrap();
        assert!(index.is_some());
        let index = index.unwrap();
        assert_eq!(index.len(), 100);
    }

    #[test]
    fn test_find_similar_approximate() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..200 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let index = ProofCorpusPqLsh::build(&corpus).unwrap().unwrap();

        let query = random_embedding(42);
        let results = index.find_similar_approximate(&corpus, &query, 10);

        assert!(!results.is_empty());
        assert!(results.len() <= 10);

        // Results should be sorted by similarity
        for i in 1..results.len() {
            assert!(results[i].similarity <= results[i - 1].similarity);
        }
    }

    #[test]
    fn test_find_similar_exact_pq() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..100 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let index = ProofCorpusPqLsh::build(&corpus).unwrap().unwrap();

        let query = random_embedding(50);
        let results = index.find_similar_exact_pq(&corpus, &query, 5);

        assert_eq!(results.len(), 5);

        // Results should be sorted
        for i in 1..results.len() {
            assert!(results[i].similarity <= results[i - 1].similarity);
        }
    }

    #[test]
    fn test_memory_stats() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..200 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let index = ProofCorpusPqLsh::build(&corpus).unwrap().unwrap();
        let stats = index.memory_stats();

        assert_eq!(stats.num_entries, 200);
        assert!(stats.compression_ratio > 1.0); // Should compress
        assert!(stats.total_bytes < stats.raw_equivalent_bytes);
    }

    #[test]
    fn test_measure_recall() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..200 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let index = ProofCorpusPqLsh::build(&corpus).unwrap().unwrap();
        let recall = index.measure_recall(&corpus, 10, 20);

        // Recall should be positive
        assert!(recall > 0.0);
        assert!(recall <= 1.0);
    }

    #[test]
    fn test_find_similar_with_min_candidates() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..200 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let index = ProofCorpusPqLsh::build(&corpus).unwrap().unwrap();

        let query = random_embedding(100);
        let results = index.find_similar_with_min_candidates(&corpus, &query, 10, 50);

        assert!(!results.is_empty());
        assert!(results.len() <= 10);
    }

    #[test]
    fn test_auto_config() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Small corpus
        for i in 0..100 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let index = ProofCorpusPqLsh::build_auto_config(&corpus).unwrap();
        assert!(index.is_some());
        let index = index.unwrap();

        // Should use small corpus config
        assert!(index.config().lsh.num_tables <= 8);
    }

    #[test]
    fn test_save_load_quantizer() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..100 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let index = ProofCorpusPqLsh::build(&corpus).unwrap().unwrap();

        // Save quantizer
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("quantizer.json");
        index.save_quantizer(&path).unwrap();
        assert!(path.exists());

        // Load and rebuild with saved quantizer
        let loaded_quantizer = crate::pq::ProductQuantizer::load_from_file(&path).unwrap();
        let config = PqLshConfig::default();
        let rebuilt = ProofCorpusPqLsh::build_with_quantizer(&corpus, loaded_quantizer, config);
        assert!(rebuilt.is_some());
        let rebuilt = rebuilt.unwrap();
        assert_eq!(rebuilt.len(), 100);

        // Verify search works on rebuilt index
        let query = random_embedding(42);
        let results = rebuilt.find_similar_approximate(&corpus, &query, 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_build_with_cached_quantizer() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..100 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cached_quantizer.json");

        // First build - should train and save quantizer
        assert!(!path.exists());
        let config = PqLshConfig::default();
        let index1 = ProofCorpusPqLsh::build_with_cached_quantizer(&corpus, &path, config.clone())
            .unwrap()
            .unwrap();
        assert!(path.exists()); // Quantizer was saved

        // Second build - should load existing quantizer
        let index2 = ProofCorpusPqLsh::build_with_cached_quantizer(&corpus, &path, config)
            .unwrap()
            .unwrap();
        assert_eq!(index1.len(), index2.len());

        // Both indexes should produce same results for same query
        let query = random_embedding(50);
        let results1 = index1.find_similar_approximate(&corpus, &query, 5);
        let results2 = index2.find_similar_approximate(&corpus, &query, 5);
        assert_eq!(results1.len(), results2.len());
    }

    #[test]
    fn test_get_quantizer() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..100 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let index = ProofCorpusPqLsh::build(&corpus).unwrap().unwrap();
        let quantizer = index.quantizer();

        // Quantizer should have correct dimension
        assert_eq!(quantizer.dim(), EMBEDDING_DIM);

        // Should be able to encode embeddings
        let test_emb = random_embedding(999);
        let codes = quantizer.encode(&test_emb);
        assert_eq!(codes.len(), quantizer.config().num_subspaces);
    }

    #[test]
    fn test_pq_lsh_config_serialize() {
        let config = PqLshConfig::for_large_corpus();

        // Serialize to JSON
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("num_tables"));
        assert!(json.contains("num_centroids"));

        // Deserialize back
        let loaded: PqLshConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.lsh.num_tables, config.lsh.num_tables);
        assert_eq!(loaded.pq.num_centroids, config.pq.num_centroids);
    }
}
