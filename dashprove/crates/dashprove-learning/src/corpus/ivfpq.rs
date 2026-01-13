//! IVFPQ (Inverted File with Product Quantization) for billion-scale search
//!
//! IVFPQ combines coarse quantization (IVF) with fine quantization (PQ) to enable
//! efficient approximate nearest neighbor search at scale:
//!
//! - **Coarse Quantization (IVF)**: Partitions vectors into `nlist` Voronoi cells
//! - **Fine Quantization (PQ)**: Compresses residuals within each cell using PQ
//! - **Inverted Index**: Stores vectors grouped by their assigned cell
//!
//! # Architecture
//!
//! ```text
//! Training:
//!   Embeddings → K-means (nlist centroids) → Assign to cells → Train PQ on residuals
//!
//! Indexing:
//!   embedding → find nearest cell → compute residual → PQ encode → store in cell
//!
//! Search:
//!   query → find nprobe nearest cells → search within cells using PQ distance
//! ```
//!
//! # Memory Usage
//!
//! For N vectors, D dimensions, nlist cells, M PQ subspaces:
//! - Coarse centroids: nlist * D * 4 bytes
//! - PQ codebooks: M * K * (D/M) * 4 bytes
//! - Encoded vectors: N * M bytes
//!
//! # When to Use IVFPQ
//!
//! IVFPQ is ideal when:
//! - Corpus size > 100,000 entries
//! - Memory is constrained
//! - Sub-linear search time is needed
//! - Some recall loss is acceptable
//!
//! For smaller corpora (<10,000), PQ-LSH or OPQ-LSH may be more appropriate.
//!
//! # Example
//!
//! ```ignore
//! use dashprove_learning::corpus::{ProofCorpus, ProofCorpusIvfPq, IvfPqConfig};
//!
//! // Build IVFPQ index
//! let config = IvfPqConfig::for_large_corpus();
//! let index = ProofCorpusIvfPq::build_with_config(&corpus, config)?;
//!
//! // Search with nprobe=10 (search 10 nearest cells)
//! let results = index.find_similar_approximate(&corpus, &query, 10, 10);
//! ```

use super::types::ProofId;
use crate::distance::euclidean_distance_sq;
use crate::embedder::Embedding;
use crate::pq::{kmeans_parallel, PqConfig, PqError, ProductQuantizer, SimpleRng};
use crate::similarity::SimilarProof;
use serde::{Deserialize, Serialize};

/// Configuration for IVFPQ index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfPqConfig {
    /// Number of Voronoi cells (coarse quantizer centroids)
    /// Typical values: sqrt(N) to 4*sqrt(N) where N is corpus size
    pub nlist: usize,
    /// Number of cells to probe during search
    /// Higher nprobe = better recall but slower search
    pub nprobe: usize,
    /// Product quantizer configuration
    pub pq: PqConfig,
    /// Number of k-means iterations for coarse quantizer training
    pub coarse_kmeans_iterations: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self {
            nlist: 256,
            nprobe: 8,
            pq: PqConfig::default(),
            coarse_kmeans_iterations: 25,
            seed: 42,
        }
    }
}

impl IvfPqConfig {
    /// Config for small corpora (1K-10K entries)
    pub fn for_small_corpus() -> Self {
        Self {
            nlist: 64,
            nprobe: 8,
            pq: PqConfig::fast(),
            coarse_kmeans_iterations: 20,
            seed: 42,
        }
    }

    /// Config for medium corpora (10K-100K entries)
    pub fn for_medium_corpus() -> Self {
        Self {
            nlist: 256,
            nprobe: 16,
            pq: PqConfig::default(),
            coarse_kmeans_iterations: 25,
            seed: 42,
        }
    }

    /// Config for large corpora (100K-1M entries)
    pub fn for_large_corpus() -> Self {
        Self {
            nlist: 1024,
            nprobe: 32,
            pq: PqConfig::default(),
            coarse_kmeans_iterations: 30,
            seed: 42,
        }
    }

    /// Config for very large corpora (>1M entries)
    pub fn for_huge_corpus() -> Self {
        Self {
            nlist: 4096,
            nprobe: 64,
            pq: PqConfig::accurate(),
            coarse_kmeans_iterations: 40,
            seed: 42,
        }
    }

    /// Auto-select config based on corpus size
    pub fn auto(corpus_size: usize) -> Self {
        if corpus_size < 10_000 {
            Self::for_small_corpus()
        } else if corpus_size < 100_000 {
            Self::for_medium_corpus()
        } else if corpus_size < 1_000_000 {
            Self::for_large_corpus()
        } else {
            Self::for_huge_corpus()
        }
    }

    /// Validate configuration
    pub fn validate(&self, dim: usize) -> Result<(), PqError> {
        if self.nlist == 0 {
            return Err(PqError::InvalidConfig("nlist must be > 0".into()));
        }
        if self.nprobe == 0 {
            return Err(PqError::InvalidConfig("nprobe must be > 0".into()));
        }
        if self.nprobe > self.nlist {
            return Err(PqError::InvalidConfig("nprobe cannot exceed nlist".into()));
        }
        self.pq.validate(dim)?;
        Ok(())
    }
}

/// Entry stored in an inverted list
#[derive(Debug, Clone)]
struct IvfEntry {
    /// Index of this entry in the original order
    idx: usize,
    /// Proof ID
    proof_id: ProofId,
    /// PQ-encoded residual vector
    codes: Vec<u8>,
}

/// IVFPQ index for ProofCorpus
///
/// Combines coarse quantization (inverted file) with fine quantization (PQ)
/// for efficient billion-scale approximate nearest neighbor search.
#[derive(Debug)]
pub struct ProofCorpusIvfPq {
    /// Coarse quantizer centroids [nlist][dim]
    coarse_centroids: Vec<Vec<f32>>,
    /// Inverted lists: one list per coarse cell
    inverted_lists: Vec<Vec<IvfEntry>>,
    /// Product quantizer for residual encoding
    pq: ProductQuantizer,
    /// Total number of indexed entries
    num_entries: usize,
    /// Configuration
    config: IvfPqConfig,
    /// Original embedding dimension
    dim: usize,
}

impl ProofCorpusIvfPq {
    /// Build an IVFPQ index from a ProofCorpus
    pub fn build(corpus: &super::storage::ProofCorpus) -> Result<Option<Self>, PqError> {
        let count = corpus.embedding_count();
        let config = IvfPqConfig::auto(count);
        Self::build_with_config(corpus, config)
    }

    /// Build with custom configuration
    pub fn build_with_config(
        corpus: &super::storage::ProofCorpus,
        config: IvfPqConfig,
    ) -> Result<Option<Self>, PqError> {
        // Collect embeddings
        let entries_with_embeddings: Vec<_> = corpus
            .entries()
            .filter_map(|e| e.embedding.as_ref().map(|emb| (e.id.clone(), emb.clone())))
            .collect();

        if entries_with_embeddings.is_empty() {
            return Ok(None);
        }

        let dim = entries_with_embeddings[0].1.dim;
        config.validate(dim)?;

        let n = entries_with_embeddings.len();

        // Need enough embeddings to train PQ on residuals
        if n < config.pq.num_centroids {
            return Err(PqError::InsufficientData(format!(
                "Need at least {} embeddings for {} PQ centroids, got {}",
                config.pq.num_centroids, config.pq.num_centroids, n
            )));
        }

        // Adjust nlist if we have fewer embeddings than requested cells
        let actual_nlist = config.nlist.min(n);

        // Extract embedding vectors for training
        let training_vecs: Vec<Vec<f32>> = entries_with_embeddings
            .iter()
            .map(|(_, e)| e.vector.clone())
            .collect();

        // Train coarse quantizer with parallelism for large datasets
        let mut rng = SimpleRng::new(config.seed);
        let num_threads = std::thread::available_parallelism().map_or(4, |p| p.get());
        let coarse_centroids = kmeans_parallel(
            &training_vecs,
            actual_nlist,
            config.coarse_kmeans_iterations,
            &mut rng,
            num_threads,
        );

        // Assign each embedding to its nearest coarse centroid
        let assignments: Vec<usize> = training_vecs
            .iter()
            .map(|v| find_nearest(v, &coarse_centroids))
            .collect();

        // Compute residuals (embedding - assigned centroid)
        let residuals: Vec<Embedding> = training_vecs
            .iter()
            .zip(assignments.iter())
            .map(|(v, &cell)| {
                let centroid = &coarse_centroids[cell];
                let residual: Vec<f32> =
                    v.iter().zip(centroid.iter()).map(|(a, b)| a - b).collect();
                Embedding::new(residual)
            })
            .collect();

        // Train PQ on residuals
        let pq = ProductQuantizer::train(&residuals, config.pq.clone())?;

        // Build inverted lists
        let mut inverted_lists: Vec<Vec<IvfEntry>> = vec![Vec::new(); actual_nlist];

        for (idx, ((proof_id, _), &cell)) in entries_with_embeddings
            .iter()
            .zip(assignments.iter())
            .enumerate()
        {
            let codes = pq.encode(&residuals[idx]);
            inverted_lists[cell].push(IvfEntry {
                idx,
                proof_id: proof_id.clone(),
                codes,
            });
        }

        Ok(Some(Self {
            coarse_centroids,
            inverted_lists,
            pq,
            num_entries: n,
            config: IvfPqConfig {
                nlist: actual_nlist,
                ..config
            },
            dim,
        }))
    }

    /// Number of indexed entries
    pub fn len(&self) -> usize {
        self.num_entries
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.num_entries == 0
    }

    /// Get the configuration
    pub fn config(&self) -> &IvfPqConfig {
        &self.config
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> IvfPqMemoryStats {
        let nlist = self.coarse_centroids.len();
        let dim = self.dim;
        let num_subspaces = self.config.pq.num_subspaces;

        // Coarse centroids memory
        let coarse_bytes = nlist * dim * 4;

        // PQ codebooks memory
        let codebook_bytes = self.pq.codebook_size_bytes();

        // Encoded vectors memory
        let codes_bytes = self.num_entries * num_subspaces;

        // Inverted list overhead (rough estimate)
        let list_overhead = nlist * 24 + self.num_entries * 32; // Vec overhead + entry overhead

        let total_bytes = coarse_bytes + codebook_bytes + codes_bytes + list_overhead;
        let raw_bytes = self.num_entries * dim * 4;

        IvfPqMemoryStats {
            num_entries: self.num_entries,
            nlist,
            coarse_bytes,
            codebook_bytes,
            codes_bytes,
            list_overhead,
            total_bytes,
            raw_equivalent_bytes: raw_bytes,
            compression_ratio: if total_bytes > 0 {
                raw_bytes as f64 / total_bytes as f64
            } else {
                0.0
            },
        }
    }

    /// Get inverted list statistics
    pub fn list_stats(&self) -> IvfListStats {
        let sizes: Vec<usize> = self.inverted_lists.iter().map(|l| l.len()).collect();
        let total: usize = sizes.iter().sum();
        let non_empty = sizes.iter().filter(|&&s| s > 0).count();
        let min = sizes.iter().copied().min().unwrap_or(0);
        let max = sizes.iter().copied().max().unwrap_or(0);
        let mean = if !sizes.is_empty() {
            total as f64 / sizes.len() as f64
        } else {
            0.0
        };

        IvfListStats {
            nlist: self.coarse_centroids.len(),
            non_empty_lists: non_empty,
            min_list_size: min,
            max_list_size: max,
            mean_list_size: mean,
        }
    }

    /// Find approximate k nearest neighbors
    ///
    /// Searches `nprobe` nearest coarse cells and returns top-k results.
    pub fn find_similar_approximate(
        &self,
        corpus: &super::storage::ProofCorpus,
        query: &Embedding,
        k: usize,
        nprobe: usize,
    ) -> Vec<SimilarProof> {
        if self.is_empty() || k == 0 || query.dim != self.dim {
            return vec![];
        }

        let nprobe = nprobe.min(self.coarse_centroids.len());

        // Find nprobe nearest coarse centroids
        let cells = self.find_nearest_cells(&query.vector, nprobe);

        // Build PQ distance tables for each cell's residual computation
        // For IVFPQ, we compute: dist(q, x) ≈ dist(q - c, r) where c is centroid, r is residual
        let mut scored: Vec<(usize, &ProofId, f64)> = Vec::new();

        for cell_idx in cells {
            let centroid = &self.coarse_centroids[cell_idx];

            // Compute query residual relative to this centroid
            let query_residual: Vec<f32> = query
                .vector
                .iter()
                .zip(centroid.iter())
                .map(|(q, c)| q - c)
                .collect();
            let query_residual_emb = Embedding::new(query_residual);

            // Build distance table for this cell
            let distance_table = self.pq.build_distance_table(&query_residual_emb);

            // Score all entries in this cell
            for entry in &self.inverted_lists[cell_idx] {
                let dist = self.pq.distance_with_table(&distance_table, &entry.codes);
                let sim = ProductQuantizer::distance_to_similarity(dist);
                scored.push((entry.idx, &entry.proof_id, sim));
            }
        }

        // Sort by similarity descending and take top-k
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        // Convert to SimilarProof
        scored
            .into_iter()
            .filter_map(|(_, proof_id, sim)| {
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

    /// Find similar with default nprobe from config
    pub fn find_similar(
        &self,
        corpus: &super::storage::ProofCorpus,
        query: &Embedding,
        k: usize,
    ) -> Vec<SimilarProof> {
        self.find_similar_approximate(corpus, query, k, self.config.nprobe)
    }

    /// Exact search (searches all cells)
    ///
    /// Useful for measuring recall of approximate search.
    pub fn find_similar_exact(
        &self,
        corpus: &super::storage::ProofCorpus,
        query: &Embedding,
        k: usize,
    ) -> Vec<SimilarProof> {
        self.find_similar_approximate(corpus, query, k, self.coarse_centroids.len())
    }

    /// Measure recall of approximate search vs exact search
    ///
    /// Uses sample queries from the indexed embeddings.
    pub fn measure_recall(
        &self,
        corpus: &super::storage::ProofCorpus,
        k: usize,
        nprobe: usize,
        num_samples: usize,
    ) -> f64 {
        if self.is_empty() || k == 0 || num_samples == 0 {
            return 0.0;
        }

        let step = self.num_entries.max(1) / num_samples.max(1);
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
                let approx = self.find_similar_approximate(corpus, query, k, nprobe);
                let exact = self.find_similar_exact(corpus, query, k);

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

    /// Measure quantization error of the PQ encoding
    ///
    /// Compares original residuals with reconstructed residuals.
    /// Lower error indicates better quantization.
    pub fn measure_quantization_error(
        &self,
        corpus: &super::storage::ProofCorpus,
        num_samples: usize,
    ) -> f64 {
        if self.is_empty() || num_samples == 0 {
            return 0.0;
        }

        let step = self.num_entries.max(1) / num_samples.max(1);
        let step = step.max(1);

        let mut total_error = 0.0;
        let mut samples_tested = 0;

        let entries_vec: Vec<_> = corpus.entries().collect();

        for i in (0..entries_vec.len()).step_by(step) {
            if samples_tested >= num_samples {
                break;
            }

            let entry = entries_vec[i];
            if let Some(ref embedding) = entry.embedding {
                // Find which cell this embedding is in
                let cell = find_nearest(&embedding.vector, &self.coarse_centroids);
                let centroid = &self.coarse_centroids[cell];

                // Compute residual
                let residual: Vec<f32> = embedding
                    .vector
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let residual_emb = Embedding::new(residual.clone());

                // Encode and decode
                let codes = self.pq.encode(&residual_emb);
                let reconstructed = self.pq.decode(&codes);

                // Compute squared error using SIMD-accelerated distance
                let error: f32 = euclidean_distance_sq(&residual, &reconstructed.vector);

                total_error += error as f64;
                samples_tested += 1;
            }
        }

        if samples_tested > 0 {
            total_error / samples_tested as f64
        } else {
            0.0
        }
    }

    /// Find nprobe nearest coarse centroids to query
    fn find_nearest_cells(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut distances: Vec<(usize, f32)> = self
            .coarse_centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let dist = squared_distance(query, c);
                (i, dist)
            })
            .collect();

        // Partial sort to find nprobe nearest
        let nprobe = nprobe.min(distances.len());
        if nprobe > 0 && nprobe < distances.len() {
            distances.select_nth_unstable_by(nprobe - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        distances.truncate(nprobe);

        distances.into_iter().map(|(i, _)| i).collect()
    }

    /// Save the index to files
    ///
    /// Saves configuration, coarse centroids, and PQ quantizer.
    /// Note: Inverted lists must be rebuilt from corpus.
    pub fn save_config<P: AsRef<std::path::Path>>(
        &self,
        config_path: P,
        quantizer_path: P,
    ) -> Result<(), crate::LearningError> {
        // Save config + coarse centroids
        let saveable = IvfPqSaveable {
            config: self.config.clone(),
            coarse_centroids: self.coarse_centroids.clone(),
            dim: self.dim,
        };
        crate::io::write_json_atomic(&config_path, &saveable)?;

        // Save PQ quantizer
        self.pq.save_to_file(quantizer_path)?;

        Ok(())
    }

    /// Load and rebuild index from saved config
    pub fn load_and_rebuild<P: AsRef<std::path::Path>>(
        corpus: &super::storage::ProofCorpus,
        config_path: P,
        quantizer_path: P,
    ) -> Result<Option<Self>, crate::LearningError> {
        // Load config + coarse centroids
        let saveable: IvfPqSaveable = crate::io::read_json(&config_path)?;

        // Load PQ quantizer
        let pq = ProductQuantizer::load_from_file(quantizer_path)?;

        // Rebuild inverted lists
        let mut inverted_lists: Vec<Vec<IvfEntry>> =
            vec![Vec::new(); saveable.coarse_centroids.len()];
        let mut num_entries = 0;

        for entry in corpus.entries() {
            if let Some(ref embedding) = entry.embedding {
                // Find nearest coarse centroid
                let cell = find_nearest(&embedding.vector, &saveable.coarse_centroids);

                // Compute residual
                let centroid = &saveable.coarse_centroids[cell];
                let residual: Vec<f32> = embedding
                    .vector
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let residual_emb = Embedding::new(residual);

                // Encode residual
                let codes = pq.encode(&residual_emb);

                inverted_lists[cell].push(IvfEntry {
                    idx: num_entries,
                    proof_id: entry.id.clone(),
                    codes,
                });
                num_entries += 1;
            }
        }

        if num_entries == 0 {
            return Ok(None);
        }

        Ok(Some(Self {
            coarse_centroids: saveable.coarse_centroids,
            inverted_lists,
            pq,
            num_entries,
            config: saveable.config,
            dim: saveable.dim,
        }))
    }
}

/// Saveable portion of IVFPQ index (excludes inverted lists which can be rebuilt)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IvfPqSaveable {
    config: IvfPqConfig,
    coarse_centroids: Vec<Vec<f32>>,
    dim: usize,
}

/// Memory usage statistics for IVFPQ index
#[derive(Debug, Clone)]
pub struct IvfPqMemoryStats {
    /// Number of indexed entries
    pub num_entries: usize,
    /// Number of coarse quantizer cells
    pub nlist: usize,
    /// Bytes used for coarse centroids
    pub coarse_bytes: usize,
    /// Bytes used for PQ codebooks
    pub codebook_bytes: usize,
    /// Bytes used for encoded vectors
    pub codes_bytes: usize,
    /// Overhead for inverted list structures
    pub list_overhead: usize,
    /// Total memory usage
    pub total_bytes: usize,
    /// Equivalent raw storage size
    pub raw_equivalent_bytes: usize,
    /// Compression ratio (raw / compressed)
    pub compression_ratio: f64,
}

/// Statistics about inverted list distribution
#[derive(Debug, Clone)]
pub struct IvfListStats {
    /// Total number of lists
    pub nlist: usize,
    /// Number of non-empty lists
    pub non_empty_lists: usize,
    /// Smallest list size
    pub min_list_size: usize,
    /// Largest list size
    pub max_list_size: usize,
    /// Average list size
    pub mean_list_size: f64,
}

// ============== Helper functions ==============

fn find_nearest(point: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0;
    let mut best_dist = f32::MAX;

    for (i, centroid) in centroids.iter().enumerate() {
        let dist = euclidean_distance_sq(point, centroid);
        if dist < best_dist {
            best_dist = dist;
            best = i;
        }
    }

    best
}

fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    euclidean_distance_sq(a, b)
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
        let mut rng = SimpleRng::new(seed);
        let mut vector = Vec::with_capacity(EMBEDDING_DIM);
        for _ in 0..EMBEDDING_DIM {
            vector.push(rng.next_f32() * 2.0 - 1.0);
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
    fn test_ivfpq_config_default() {
        let config = IvfPqConfig::default();
        assert_eq!(config.nlist, 256);
        assert_eq!(config.nprobe, 8);
        assert!(config.validate(EMBEDDING_DIM).is_ok());
    }

    #[test]
    fn test_ivfpq_config_presets() {
        let small = IvfPqConfig::for_small_corpus();
        let medium = IvfPqConfig::for_medium_corpus();
        let large = IvfPqConfig::for_large_corpus();
        let huge = IvfPqConfig::for_huge_corpus();

        assert!(small.nlist < medium.nlist);
        assert!(medium.nlist < large.nlist);
        assert!(large.nlist < huge.nlist);
    }

    #[test]
    fn test_ivfpq_config_auto() {
        let config_1k = IvfPqConfig::auto(1_000);
        let config_50k = IvfPqConfig::auto(50_000);
        let config_500k = IvfPqConfig::auto(500_000);
        let config_5m = IvfPqConfig::auto(5_000_000);

        assert!(config_1k.nlist < config_50k.nlist);
        assert!(config_50k.nlist < config_500k.nlist);
        assert!(config_500k.nlist < config_5m.nlist);
    }

    #[test]
    fn test_ivfpq_config_validation() {
        // nlist = 0 should fail
        let config_nlist_zero = IvfPqConfig {
            nlist: 0,
            ..Default::default()
        };
        assert!(config_nlist_zero.validate(EMBEDDING_DIM).is_err());

        // nprobe = 0 should fail
        let config_nprobe_zero = IvfPqConfig {
            nprobe: 0,
            ..Default::default()
        };
        assert!(config_nprobe_zero.validate(EMBEDDING_DIM).is_err());

        // nprobe > nlist should fail
        let config_nprobe_exceeds = IvfPqConfig {
            nprobe: 300,
            nlist: 256,
            ..Default::default()
        };
        assert!(config_nprobe_exceeds.validate(EMBEDDING_DIM).is_err());
    }

    #[test]
    fn test_build_empty_corpus() {
        let corpus = super::super::storage::ProofCorpus::new();
        let result = ProofCorpusIvfPq::build(&corpus);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_build_insufficient_embeddings() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Insert fewer embeddings than needed for PQ centroids
        for i in 0..10 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        // Should fail due to insufficient data
        let result = ProofCorpusIvfPq::build(&corpus);
        assert!(matches!(result, Err(PqError::InsufficientData(_))));
    }

    #[test]
    fn test_build_with_sufficient_embeddings() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Insert enough embeddings
        for i in 0..500 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = IvfPqConfig {
            nlist: 32,
            nprobe: 4,
            pq: PqConfig::fast(),
            coarse_kmeans_iterations: 10,
            seed: 42,
        };

        let index = ProofCorpusIvfPq::build_with_config(&corpus, config);
        assert!(index.is_ok());
        let index = index.unwrap();
        assert!(index.is_some());
        let index = index.unwrap();
        assert_eq!(index.len(), 500);
    }

    #[test]
    fn test_find_similar_approximate() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..500 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = IvfPqConfig {
            nlist: 32,
            nprobe: 8,
            pq: PqConfig::fast(),
            coarse_kmeans_iterations: 10,
            seed: 42,
        };

        let index = ProofCorpusIvfPq::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        let query = random_embedding(42);
        let results = index.find_similar_approximate(&corpus, &query, 10, 8);

        assert!(!results.is_empty());
        assert!(results.len() <= 10);

        // Results should be sorted by similarity
        for i in 1..results.len() {
            assert!(
                results[i].similarity <= results[i - 1].similarity,
                "Results should be sorted by similarity"
            );
        }
    }

    #[test]
    fn test_find_similar_exact() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..300 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = IvfPqConfig {
            nlist: 16,
            nprobe: 16, // All cells
            pq: PqConfig::fast(),
            coarse_kmeans_iterations: 10,
            seed: 42,
        };

        let index = ProofCorpusIvfPq::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        let query = random_embedding(50);
        let exact = index.find_similar_exact(&corpus, &query, 10);

        assert_eq!(exact.len(), 10);

        // With nprobe = nlist, should search all cells
        for i in 1..exact.len() {
            assert!(exact[i].similarity <= exact[i - 1].similarity);
        }
    }

    #[test]
    fn test_memory_stats() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..500 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = IvfPqConfig {
            nlist: 32,
            nprobe: 8,
            pq: PqConfig::fast(),
            coarse_kmeans_iterations: 10,
            seed: 42,
        };

        let index = ProofCorpusIvfPq::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        let stats = index.memory_stats();

        assert_eq!(stats.num_entries, 500);
        assert_eq!(stats.nlist, 32);
        assert!(stats.compression_ratio > 1.0); // Should compress
        assert!(stats.total_bytes < stats.raw_equivalent_bytes);
    }

    #[test]
    fn test_list_stats() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..500 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = IvfPqConfig {
            nlist: 32,
            nprobe: 8,
            pq: PqConfig::fast(),
            coarse_kmeans_iterations: 10,
            seed: 42,
        };

        let index = ProofCorpusIvfPq::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        let stats = index.list_stats();

        assert_eq!(stats.nlist, 32);
        assert!(stats.non_empty_lists > 0);
        assert!(stats.max_list_size >= stats.min_list_size);
        // Mean should be close to 500/32 ≈ 15.6
        assert!(stats.mean_list_size > 10.0);
        assert!(stats.mean_list_size < 25.0);
    }

    #[test]
    fn test_measure_recall() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..300 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = IvfPqConfig {
            nlist: 16,
            nprobe: 8,
            pq: PqConfig::fast(),
            coarse_kmeans_iterations: 10,
            seed: 42,
        };

        let index = ProofCorpusIvfPq::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        let recall = index.measure_recall(&corpus, 10, 8, 20);

        // With nprobe=8 out of 16 cells, recall should be reasonable
        assert!(recall > 0.0);
        assert!(recall <= 1.0);
    }

    #[test]
    fn test_recall_improves_with_nprobe() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..500 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = IvfPqConfig {
            nlist: 32,
            nprobe: 8, // Will test different values
            pq: PqConfig::fast(),
            coarse_kmeans_iterations: 10,
            seed: 42,
        };

        let index = ProofCorpusIvfPq::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        // Recall should generally improve with more probes
        let recall_4 = index.measure_recall(&corpus, 10, 4, 20);
        let recall_16 = index.measure_recall(&corpus, 10, 16, 20);
        let recall_32 = index.measure_recall(&corpus, 10, 32, 20);

        // With more probes, recall should not decrease significantly
        // (may not strictly increase due to sampling variance)
        assert!(
            recall_32 >= recall_4 * 0.8,
            "Recall with nprobe=32 ({}) should be >= 80% of recall with nprobe=4 ({})",
            recall_32,
            recall_4
        );
        assert!(
            recall_16 >= recall_4 * 0.8,
            "Recall with nprobe=16 ({}) should be >= 80% of recall with nprobe=4 ({})",
            recall_16,
            recall_4
        );
    }

    #[test]
    fn test_save_and_load() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..300 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = IvfPqConfig {
            nlist: 16,
            nprobe: 8,
            pq: PqConfig::fast(),
            coarse_kmeans_iterations: 10,
            seed: 42,
        };

        let index = ProofCorpusIvfPq::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        // Save
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("ivfpq_config.json");
        let quantizer_path = dir.path().join("ivfpq_pq.json");

        index.save_config(&config_path, &quantizer_path).unwrap();

        assert!(config_path.exists());
        assert!(quantizer_path.exists());

        // Load and rebuild
        let loaded = ProofCorpusIvfPq::load_and_rebuild(&corpus, &config_path, &quantizer_path)
            .unwrap()
            .unwrap();

        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.config().nlist, index.config().nlist);

        // Verify search works on loaded index
        let query = random_embedding(50);
        let results1 = index.find_similar(&corpus, &query, 5);
        let results2 = loaded.find_similar(&corpus, &query, 5);

        // Results should be similar (not identical due to reconstruction)
        assert_eq!(results1.len(), results2.len());
    }

    #[test]
    fn test_nprobe_clamping() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..200 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = IvfPqConfig {
            nlist: 16,
            nprobe: 8,
            pq: PqConfig::fast(),
            coarse_kmeans_iterations: 10,
            seed: 42,
        };

        let index = ProofCorpusIvfPq::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        // Search with nprobe > nlist should work (clamped)
        let query = random_embedding(42);
        let results = index.find_similar_approximate(&corpus, &query, 10, 100);

        assert!(!results.is_empty());
    }

    #[test]
    fn test_kmeans_convergence() {
        // Test k-means on simple synthetic data
        let data: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![10.0, 10.1],
        ];

        let mut rng = SimpleRng::new(42);
        // Use single thread for small dataset (kmeans_parallel falls back to sequential)
        let centroids = kmeans_parallel(&data, 2, 20, &mut rng, 1);

        assert_eq!(centroids.len(), 2);

        // Centroids should be near (0,0) and (10,10)
        let mut near_zero = false;
        let mut near_ten = false;

        for c in &centroids {
            if squared_distance(c, &[0.0, 0.0]) < 1.0 {
                near_zero = true;
            }
            if squared_distance(c, &[10.0, 10.0]) < 1.0 {
                near_ten = true;
            }
        }

        assert!(near_zero && near_ten);
    }

    #[test]
    fn test_quantization_error() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..300 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = IvfPqConfig {
            nlist: 16,
            nprobe: 8,
            pq: PqConfig::fast(),
            coarse_kmeans_iterations: 10,
            seed: 42,
        };

        let index = ProofCorpusIvfPq::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        let error = index.measure_quantization_error(&corpus, 50);

        // Quantization error should be positive (some loss)
        assert!(error >= 0.0);
        // But not too large for reasonable quantization
        assert!(error < 10.0);
    }
}
