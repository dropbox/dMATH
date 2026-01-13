//! Streaming and batched index builders for very large corpora
//!
//! When dealing with corpora too large to fit in memory, these builders provide
//! ways to construct indices incrementally:
//!
//! - **StreamingIvfBuilder**: Train quantizers on a sample, then stream through
//!   data to build inverted lists without loading everything at once
//! - **BatchedHnswBuilder**: Process embeddings in configurable batches with
//!   progress callbacks, suitable for memory-constrained environments
//!
//! # Example: Streaming IVF Build
//!
//! ```ignore
//! use dashprove_learning::corpus::{StreamingIvfBuilder, IvfPqConfig};
//!
//! // Create builder with config
//! let config = IvfPqConfig::for_large_corpus();
//! let mut builder = StreamingIvfBuilder::new(config, 96)?; // 96-dim embeddings
//!
//! // Phase 1: Train on a sample (e.g., first 10K embeddings)
//! for (id, embedding) in sample_iter {
//!     builder.add_training_sample(id, embedding);
//! }
//! builder.train()?;
//!
//! // Phase 2: Stream through all data to build index
//! for (id, embedding) in full_data_iter {
//!     builder.add_entry(id, embedding)?;
//! }
//!
//! // Finalize and get the index
//! let index = builder.build()?;
//! ```
//!
//! # Example: Batched HNSW Build
//!
//! ```ignore
//! use dashprove_learning::corpus::{BatchedHnswBuilder, HnswConfig};
//!
//! // Create builder with progress callback
//! let config = HnswConfig::default();
//! let mut builder = BatchedHnswBuilder::new(config)?;
//! builder.set_batch_size(1000);
//! builder.set_progress_callback(|progress| {
//!     println!("Progress: {:.1}%", progress.percentage());
//! });
//!
//! // Add entries (automatically processed in batches)
//! for (id, embedding) in data_iter {
//!     builder.add_entry(id, embedding)?;
//! }
//!
//! // Finalize
//! let index = builder.build()?;
//! ```

use super::types::ProofId;
use crate::distance::euclidean_distance_sq;
use crate::embedder::Embedding;
use crate::pq::{
    kmeans_parallel, OpqConfig, OptimizedProductQuantizer, PqConfig, PqError, ProductQuantizer,
    SimpleRng,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Progress information for streaming/batched builds
#[derive(Debug, Clone)]
pub struct BuildProgress {
    /// Current phase of the build
    pub phase: BuildPhase,
    /// Number of items processed in current phase
    pub items_processed: usize,
    /// Total items expected (if known)
    pub total_items: Option<usize>,
    /// Elapsed time in seconds
    pub elapsed_secs: f64,
}

impl BuildProgress {
    /// Get percentage completion (0-100) if total is known
    pub fn percentage(&self) -> Option<f64> {
        self.total_items
            .map(|total| (self.items_processed as f64 / total as f64) * 100.0)
    }

    /// Estimate remaining time based on current rate
    pub fn estimated_remaining_secs(&self) -> Option<f64> {
        self.total_items.and_then(|total| {
            if self.items_processed > 0 && self.elapsed_secs > 0.0 {
                let rate = self.items_processed as f64 / self.elapsed_secs;
                let remaining = total.saturating_sub(self.items_processed);
                Some(remaining as f64 / rate)
            } else {
                None
            }
        })
    }
}

/// Build phases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildPhase {
    /// Collecting training samples
    CollectingSamples,
    /// Training quantizers (k-means, OPQ rotation)
    Training,
    /// Building index structure
    Indexing,
    /// Finalizing index
    Finalizing,
}

/// Error type for streaming builders
#[derive(Debug, Clone)]
pub enum StreamingBuildError {
    /// Not enough training samples
    InsufficientTrainingSamples { need: usize, have: usize },
    /// Training not complete before indexing
    NotTrained,
    /// Dimension mismatch
    DimensionMismatch { expected: usize, got: usize },
    /// Duplicate proof ID
    DuplicateId(ProofId),
    /// PQ training error
    PqError(PqError),
    /// Invalid configuration
    InvalidConfig(String),
    /// Index is empty
    EmptyIndex,
}

impl std::fmt::Display for StreamingBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamingBuildError::InsufficientTrainingSamples { need, have } => {
                write!(
                    f,
                    "Insufficient training samples: need {}, have {}",
                    need, have
                )
            }
            StreamingBuildError::NotTrained => write!(f, "Must call train() before adding entries"),
            StreamingBuildError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            StreamingBuildError::DuplicateId(id) => write!(f, "Duplicate proof ID: {}", id),
            StreamingBuildError::PqError(e) => write!(f, "PQ error: {}", e),
            StreamingBuildError::InvalidConfig(s) => write!(f, "Invalid config: {}", s),
            StreamingBuildError::EmptyIndex => write!(f, "Index is empty"),
        }
    }
}

impl std::error::Error for StreamingBuildError {}

impl From<PqError> for StreamingBuildError {
    fn from(e: PqError) -> Self {
        StreamingBuildError::PqError(e)
    }
}

// ============================================================================
// Streaming IVF Builder
// ============================================================================

/// Configuration for streaming IVF builds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingIvfConfig {
    /// Number of Voronoi cells (coarse quantizer centroids)
    pub nlist: usize,
    /// Number of cells to probe during search (default)
    pub nprobe: usize,
    /// Product quantizer configuration
    pub pq: PqConfig,
    /// Use OPQ rotation for better accuracy
    pub use_opq: bool,
    /// OPQ configuration (used if use_opq is true)
    pub opq: Option<OpqConfig>,
    /// Number of k-means iterations for coarse quantizer
    pub coarse_kmeans_iterations: usize,
    /// Minimum training samples required
    pub min_training_samples: usize,
    /// Random seed
    pub seed: u64,
}

impl Default for StreamingIvfConfig {
    fn default() -> Self {
        Self {
            nlist: 256,
            nprobe: 8,
            pq: PqConfig::default(),
            use_opq: false,
            opq: None,
            coarse_kmeans_iterations: 25,
            min_training_samples: 1000,
            seed: 42,
        }
    }
}

impl StreamingIvfConfig {
    /// Config for large corpus with PQ
    pub fn for_large_corpus() -> Self {
        Self {
            nlist: 1024,
            nprobe: 32,
            pq: PqConfig::default(),
            use_opq: false,
            opq: None,
            coarse_kmeans_iterations: 30,
            min_training_samples: 10000,
            seed: 42,
        }
    }

    /// Config for large corpus with OPQ (better recall)
    pub fn for_large_corpus_opq() -> Self {
        Self {
            nlist: 1024,
            nprobe: 32,
            pq: PqConfig::default(),
            use_opq: true,
            opq: Some(OpqConfig::default()),
            coarse_kmeans_iterations: 30,
            min_training_samples: 10000,
            seed: 42,
        }
    }

    /// Config for billion-scale corpus
    pub fn for_billion_scale() -> Self {
        Self {
            nlist: 4096,
            nprobe: 64,
            pq: PqConfig::accurate(),
            use_opq: true,
            opq: Some(OpqConfig::default()),
            coarse_kmeans_iterations: 40,
            min_training_samples: 100000,
            seed: 42,
        }
    }

    /// Validate configuration
    pub fn validate(&self, dim: usize) -> Result<(), StreamingBuildError> {
        if self.nlist == 0 {
            return Err(StreamingBuildError::InvalidConfig(
                "nlist must be > 0".into(),
            ));
        }
        if self.nprobe == 0 || self.nprobe > self.nlist {
            return Err(StreamingBuildError::InvalidConfig(
                "nprobe must be > 0 and <= nlist".into(),
            ));
        }
        self.pq.validate(dim).map_err(StreamingBuildError::from)?;
        Ok(())
    }
}

/// Entry stored in an inverted list (streaming build)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StreamingIvfEntry {
    proof_id: ProofId,
    codes: Vec<u8>,
}

/// Trained quantizers for IVF
#[derive(Debug, Clone, Serialize, Deserialize)]
enum TrainedQuantizer {
    Pq(ProductQuantizer),
    Opq(OptimizedProductQuantizer),
}

/// Streaming builder for IVF indices
///
/// Allows building IVFPQ/IVFOPQ indices without loading all data into memory:
/// 1. Collect training samples (reservoir sampling or explicit)
/// 2. Train coarse quantizer (k-means) and fine quantizer (PQ/OPQ)
/// 3. Stream through full data to build inverted lists
pub struct StreamingIvfBuilder {
    /// Configuration
    config: StreamingIvfConfig,
    /// Embedding dimension
    dim: usize,
    /// Training samples
    training_samples: Vec<(ProofId, Embedding)>,
    /// Trained coarse centroids
    coarse_centroids: Option<Vec<Vec<f32>>>,
    /// Trained quantizer
    quantizer: Option<TrainedQuantizer>,
    /// Inverted lists (populated during streaming)
    inverted_lists: Vec<Vec<StreamingIvfEntry>>,
    /// Track inserted IDs
    inserted_ids: HashMap<ProofId, usize>,
    /// Total entries added
    num_entries: usize,
    /// Build start time
    start_time: Option<std::time::Instant>,
    /// Progress callback
    progress_callback: Option<Box<dyn Fn(BuildProgress) + Send + Sync>>,
}

impl StreamingIvfBuilder {
    /// Create a new streaming IVF builder
    pub fn new(config: StreamingIvfConfig, dim: usize) -> Result<Self, StreamingBuildError> {
        config.validate(dim)?;
        Ok(Self {
            config,
            dim,
            training_samples: Vec::new(),
            coarse_centroids: None,
            quantizer: None,
            inverted_lists: Vec::new(),
            inserted_ids: HashMap::new(),
            num_entries: 0,
            start_time: None,
            progress_callback: None,
        })
    }

    /// Set progress callback
    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: Fn(BuildProgress) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
    }

    /// Add a training sample
    ///
    /// Call this for at least `min_training_samples` embeddings before calling `train()`.
    /// For large datasets, you can use reservoir sampling to get a representative sample.
    pub fn add_training_sample(&mut self, id: ProofId, embedding: Embedding) {
        if self.start_time.is_none() {
            self.start_time = Some(std::time::Instant::now());
        }
        self.training_samples.push((id, embedding));

        // Report progress
        if let Some(ref callback) = self.progress_callback {
            let elapsed = self
                .start_time
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            callback(BuildProgress {
                phase: BuildPhase::CollectingSamples,
                items_processed: self.training_samples.len(),
                total_items: Some(self.config.min_training_samples),
                elapsed_secs: elapsed,
            });
        }
    }

    /// Number of training samples collected
    pub fn training_sample_count(&self) -> usize {
        self.training_samples.len()
    }

    /// Train quantizers on collected samples
    ///
    /// Must be called after adding at least `min_training_samples` samples.
    /// After training, call `add_entry()` to stream through all data.
    pub fn train(&mut self) -> Result<(), StreamingBuildError> {
        let n = self.training_samples.len();
        if n < self.config.min_training_samples {
            return Err(StreamingBuildError::InsufficientTrainingSamples {
                need: self.config.min_training_samples,
                have: n,
            });
        }

        // Report training phase
        if let Some(ref callback) = self.progress_callback {
            let elapsed = self
                .start_time
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            callback(BuildProgress {
                phase: BuildPhase::Training,
                items_processed: 0,
                total_items: None,
                elapsed_secs: elapsed,
            });
        }

        // Extract vectors for training
        let training_vecs: Vec<Vec<f32>> = self
            .training_samples
            .iter()
            .map(|(_, e)| e.vector.clone())
            .collect();

        // Train coarse quantizer with parallelism for large datasets
        let mut rng = SimpleRng::new(self.config.seed);
        let actual_nlist = self.config.nlist.min(n);
        let num_threads = std::thread::available_parallelism().map_or(4, |p| p.get());
        let coarse_centroids = kmeans_parallel(
            &training_vecs,
            actual_nlist,
            self.config.coarse_kmeans_iterations,
            &mut rng,
            num_threads,
        );

        // Assign training samples to cells and compute residuals
        let assignments: Vec<usize> = training_vecs
            .iter()
            .map(|v| find_nearest(v, &coarse_centroids))
            .collect();

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

        // Train PQ or OPQ on residuals
        let quantizer = if self.config.use_opq {
            let opq_config = self.config.opq.clone().unwrap_or_default();
            let opq = OptimizedProductQuantizer::train(&residuals, opq_config)?;
            TrainedQuantizer::Opq(opq)
        } else {
            let pq = ProductQuantizer::train(&residuals, self.config.pq.clone())?;
            TrainedQuantizer::Pq(pq)
        };

        // Initialize inverted lists
        self.inverted_lists = vec![Vec::new(); actual_nlist];
        self.coarse_centroids = Some(coarse_centroids);
        self.quantizer = Some(quantizer);

        Ok(())
    }

    /// Check if training is complete
    pub fn is_trained(&self) -> bool {
        self.coarse_centroids.is_some() && self.quantizer.is_some()
    }

    /// Add an entry to the index (streaming phase)
    ///
    /// Must call `train()` first. This method can be called millions of times
    /// to stream through data without loading everything into memory.
    pub fn add_entry(
        &mut self,
        id: ProofId,
        embedding: Embedding,
    ) -> Result<(), StreamingBuildError> {
        if !self.is_trained() {
            return Err(StreamingBuildError::NotTrained);
        }

        if embedding.dim != self.dim {
            return Err(StreamingBuildError::DimensionMismatch {
                expected: self.dim,
                got: embedding.dim,
            });
        }

        if self.inserted_ids.contains_key(&id) {
            return Err(StreamingBuildError::DuplicateId(id));
        }

        let coarse_centroids = self.coarse_centroids.as_ref().unwrap();
        let cell = find_nearest(&embedding.vector, coarse_centroids);

        // Compute residual
        let centroid = &coarse_centroids[cell];
        let residual_vec: Vec<f32> = embedding
            .vector
            .iter()
            .zip(centroid.iter())
            .map(|(a, b)| a - b)
            .collect();
        let residual = Embedding::new(residual_vec);

        // Encode residual
        let codes = match self.quantizer.as_ref().unwrap() {
            TrainedQuantizer::Pq(pq) => pq.encode(&residual),
            TrainedQuantizer::Opq(opq) => opq.encode(&residual),
        };

        // Add to inverted list
        let idx = self.num_entries;
        self.inverted_lists[cell].push(StreamingIvfEntry {
            proof_id: id.clone(),
            codes,
        });
        self.inserted_ids.insert(id, idx);
        self.num_entries += 1;

        // Report progress periodically
        if self.num_entries.is_multiple_of(10000) {
            if let Some(ref callback) = self.progress_callback {
                let elapsed = self
                    .start_time
                    .map(|t| t.elapsed().as_secs_f64())
                    .unwrap_or(0.0);
                callback(BuildProgress {
                    phase: BuildPhase::Indexing,
                    items_processed: self.num_entries,
                    total_items: None,
                    elapsed_secs: elapsed,
                });
            }
        }

        Ok(())
    }

    /// Number of entries added
    pub fn entry_count(&self) -> usize {
        self.num_entries
    }

    /// Build the final index
    ///
    /// Returns a `StreamingIvfIndex` that can be used for approximate nearest neighbor search.
    pub fn build(mut self) -> Result<StreamingIvfIndex, StreamingBuildError> {
        if self.num_entries == 0 {
            return Err(StreamingBuildError::EmptyIndex);
        }

        // Report finalization
        if let Some(ref callback) = self.progress_callback {
            let elapsed = self
                .start_time
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            callback(BuildProgress {
                phase: BuildPhase::Finalizing,
                items_processed: self.num_entries,
                total_items: Some(self.num_entries),
                elapsed_secs: elapsed,
            });
        }

        Ok(StreamingIvfIndex {
            coarse_centroids: self.coarse_centroids.take().unwrap(),
            inverted_lists: std::mem::take(&mut self.inverted_lists),
            quantizer: self.quantizer.take().unwrap(),
            num_entries: self.num_entries,
            config: self.config,
            dim: self.dim,
        })
    }
}

/// IVF index built via streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingIvfIndex {
    coarse_centroids: Vec<Vec<f32>>,
    inverted_lists: Vec<Vec<StreamingIvfEntry>>,
    quantizer: TrainedQuantizer,
    num_entries: usize,
    config: StreamingIvfConfig,
    dim: usize,
}

impl StreamingIvfIndex {
    /// Number of indexed entries
    pub fn len(&self) -> usize {
        self.num_entries
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.num_entries == 0
    }

    /// Get configuration
    pub fn config(&self) -> &StreamingIvfConfig {
        &self.config
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> StreamingIvfMemoryStats {
        let nlist = self.coarse_centroids.len();
        let dim = self.dim;

        // Coarse centroids
        let coarse_bytes = nlist * dim * 4;

        // Codebook size
        let codebook_bytes = match &self.quantizer {
            TrainedQuantizer::Pq(pq) => pq.codebook_size_bytes(),
            TrainedQuantizer::Opq(opq) => opq.codebook_size_bytes(),
        };

        // Encoded vectors
        let num_subspaces = match &self.quantizer {
            TrainedQuantizer::Pq(pq) => pq.config().num_subspaces,
            TrainedQuantizer::Opq(opq) => opq.pq_config().num_subspaces,
        };
        let codes_bytes = self.num_entries * num_subspaces;

        // Rotation matrix (OPQ only)
        let rotation_bytes = match &self.quantizer {
            TrainedQuantizer::Pq(_) => 0,
            TrainedQuantizer::Opq(opq) => opq.total_size_bytes() - opq.codebook_size_bytes(),
        };

        // Overhead
        let list_overhead = nlist * 24 + self.num_entries * 32;

        let total_bytes =
            coarse_bytes + codebook_bytes + codes_bytes + rotation_bytes + list_overhead;
        let raw_bytes = self.num_entries * dim * 4;

        StreamingIvfMemoryStats {
            num_entries: self.num_entries,
            nlist,
            coarse_bytes,
            codebook_bytes,
            codes_bytes,
            rotation_bytes,
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
    pub fn list_stats(&self) -> StreamingIvfListStats {
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

        StreamingIvfListStats {
            nlist: self.coarse_centroids.len(),
            non_empty_lists: non_empty,
            min_list_size: min,
            max_list_size: max,
            mean_list_size: mean,
        }
    }

    /// Find approximate k nearest neighbors
    pub fn find_similar(&self, query: &Embedding, k: usize, nprobe: usize) -> Vec<(ProofId, f64)> {
        if self.is_empty() || k == 0 || query.dim != self.dim {
            return vec![];
        }

        let nprobe = nprobe.min(self.coarse_centroids.len());

        // Find nearest coarse cells
        let cells = self.find_nearest_cells(&query.vector, nprobe);

        // Search within cells
        let mut candidates: Vec<(ProofId, f64)> = Vec::new();

        for cell in cells {
            let centroid = &self.coarse_centroids[cell];
            let query_residual: Vec<f32> = query
                .vector
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| a - b)
                .collect();

            // Build distance table
            let query_residual_emb = Embedding::new(query_residual);
            let distances = match &self.quantizer {
                TrainedQuantizer::Pq(pq) => pq.build_distance_table(&query_residual_emb),
                TrainedQuantizer::Opq(opq) => opq.build_distance_table(&query_residual_emb),
            };

            // Score entries in this cell
            for entry in &self.inverted_lists[cell] {
                let dist = match &self.quantizer {
                    TrainedQuantizer::Pq(pq) => pq.distance_with_table(&distances, &entry.codes),
                    TrainedQuantizer::Opq(opq) => opq.distance_with_table(&distances, &entry.codes),
                };
                candidates.push((entry.proof_id.clone(), dist as f64));
            }
        }

        // Sort by distance and return top k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        candidates
    }

    fn find_nearest_cells(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut cell_dists: Vec<(usize, f32)> = self
            .coarse_centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, euclidean_distance_sq(query, c)))
            .collect();
        cell_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        cell_dists
            .into_iter()
            .take(nprobe)
            .map(|(i, _)| i)
            .collect()
    }

    // ========== Persistence methods ==========

    /// Save the streaming IVF index to a JSON file
    ///
    /// This persists the coarse centroids, inverted lists, and quantizer so the
    /// index can be loaded without rebuilding.
    pub fn save_to_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), crate::LearningError> {
        crate::io::write_json_atomic(path, self)
    }

    /// Load a streaming IVF index from a JSON file
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, crate::LearningError> {
        crate::io::read_json(path)
    }

    /// Load a streaming IVF index if the file exists, otherwise return None
    pub fn load_if_exists<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Option<Self>, crate::LearningError> {
        match std::fs::File::open(path.as_ref()) {
            Ok(_) => Ok(Some(Self::load_from_file(path)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    // ========== Parallel batch query methods ==========

    /// Process multiple queries in parallel using std::thread
    ///
    /// This is useful when you have a batch of queries to process and want to
    /// utilize multiple CPU cores. The parallelism is controlled by `num_threads`.
    ///
    /// # Arguments
    /// * `queries` - Slice of query embeddings
    /// * `k` - Number of nearest neighbors to return per query
    /// * `nprobe` - Number of cells to probe per query
    /// * `num_threads` - Number of threads to use (clamped to queries.len())
    ///
    /// # Returns
    /// Vector of results, one per query, in the same order as the input queries
    pub fn find_similar_batch(
        &self,
        queries: &[Embedding],
        k: usize,
        nprobe: usize,
        num_threads: usize,
    ) -> Vec<Vec<(ProofId, f64)>> {
        if queries.is_empty() {
            return vec![];
        }

        let num_threads = num_threads.min(queries.len()).max(1);

        if num_threads == 1 {
            // Single-threaded fallback
            return queries
                .iter()
                .map(|q| self.find_similar(q, k, nprobe))
                .collect();
        }

        // Use scoped threads for parallel processing
        let chunk_size = queries.len().div_ceil(num_threads);
        let mut results: Vec<Vec<(ProofId, f64)>> = vec![vec![]; queries.len()];

        std::thread::scope(|s| {
            let mut handles = Vec::new();

            for (chunk_idx, chunk) in queries.chunks(chunk_size).enumerate() {
                let start_idx = chunk_idx * chunk_size;
                let handle = s.spawn(move || {
                    chunk
                        .iter()
                        .enumerate()
                        .map(|(i, q)| (start_idx + i, self.find_similar(q, k, nprobe)))
                        .collect::<Vec<_>>()
                });
                handles.push(handle);
            }

            for handle in handles {
                let chunk_results = handle.join().expect("Thread panicked");
                for (idx, result) in chunk_results {
                    results[idx] = result;
                }
            }
        });

        results
    }
}

/// Memory statistics for streaming IVF index
#[derive(Debug, Clone)]
pub struct StreamingIvfMemoryStats {
    pub num_entries: usize,
    pub nlist: usize,
    pub coarse_bytes: usize,
    pub codebook_bytes: usize,
    pub codes_bytes: usize,
    pub rotation_bytes: usize,
    pub list_overhead: usize,
    pub total_bytes: usize,
    pub raw_equivalent_bytes: usize,
    pub compression_ratio: f64,
}

/// List statistics for streaming IVF index
#[derive(Debug, Clone)]
pub struct StreamingIvfListStats {
    pub nlist: usize,
    pub non_empty_lists: usize,
    pub min_list_size: usize,
    pub max_list_size: usize,
    pub mean_list_size: f64,
}

// ============================================================================
// Batched HNSW Builder
// ============================================================================

/// Configuration for batched HNSW builds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedHnswConfig {
    /// Maximum connections per node (M)
    pub m: usize,
    /// Maximum connections at layer 0 (typically 2*M)
    pub m0: usize,
    /// Search width during construction
    pub ef_construction: usize,
    /// Default search width during queries
    pub ef_search: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Random seed
    pub seed: u64,
}

impl Default for BatchedHnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 50,
            batch_size: 10000,
            seed: 42,
        }
    }
}

impl BatchedHnswConfig {
    /// Fast build configuration
    pub fn fast() -> Self {
        Self {
            m: 12,
            m0: 24,
            ef_construction: 100,
            ef_search: 30,
            batch_size: 10000,
            seed: 42,
        }
    }

    /// High recall configuration
    pub fn high_recall() -> Self {
        Self {
            m: 32,
            m0: 64,
            ef_construction: 400,
            ef_search: 100,
            batch_size: 5000,
            seed: 42,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), StreamingBuildError> {
        if self.m == 0 {
            return Err(StreamingBuildError::InvalidConfig("m must be > 0".into()));
        }
        if self.m0 < self.m {
            return Err(StreamingBuildError::InvalidConfig("m0 must be >= m".into()));
        }
        if self.ef_construction < self.m {
            return Err(StreamingBuildError::InvalidConfig(
                "ef_construction must be >= m".into(),
            ));
        }
        if self.batch_size == 0 {
            return Err(StreamingBuildError::InvalidConfig(
                "batch_size must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// Node in batched HNSW graph
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BatchedHnswNode {
    #[allow(dead_code)]
    idx: usize,
    proof_id: ProofId,
    embedding: Embedding,
    neighbors: Vec<Vec<usize>>,
    max_layer: usize,
}

/// Batched builder for HNSW indices
///
/// Processes insertions in configurable batches with progress callbacks,
/// suitable for memory-constrained environments or progress monitoring.
pub struct BatchedHnswBuilder {
    config: BatchedHnswConfig,
    nodes: Vec<BatchedHnswNode>,
    id_to_idx: HashMap<ProofId, usize>,
    entry_point: Option<usize>,
    max_layer: usize,
    dim: Option<usize>,
    ml: f64,
    rng_state: u64,
    pending_batch: Vec<(ProofId, Embedding)>,
    start_time: Option<std::time::Instant>,
    progress_callback: Option<Box<dyn Fn(BuildProgress) + Send + Sync>>,
}

impl BatchedHnswBuilder {
    /// Create a new batched HNSW builder
    pub fn new(config: BatchedHnswConfig) -> Result<Self, StreamingBuildError> {
        config.validate()?;
        let ml = 1.0 / (config.m as f64).ln();
        let rng_state = config.seed;

        Ok(Self {
            config,
            nodes: Vec::new(),
            id_to_idx: HashMap::new(),
            entry_point: None,
            max_layer: 0,
            dim: None,
            ml,
            rng_state,
            pending_batch: Vec::new(),
            start_time: None,
            progress_callback: None,
        })
    }

    /// Set progress callback
    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: Fn(BuildProgress) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
    }

    /// Set batch size
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.config.batch_size = batch_size;
    }

    /// Add an entry to the pending batch
    ///
    /// When batch size is reached, automatically processes the batch.
    pub fn add_entry(
        &mut self,
        id: ProofId,
        embedding: Embedding,
    ) -> Result<(), StreamingBuildError> {
        if self.start_time.is_none() {
            self.start_time = Some(std::time::Instant::now());
        }

        // Check dimension
        if let Some(dim) = self.dim {
            if embedding.dim != dim {
                return Err(StreamingBuildError::DimensionMismatch {
                    expected: dim,
                    got: embedding.dim,
                });
            }
        } else {
            self.dim = Some(embedding.dim);
        }

        // Check for duplicates (both in already-processed nodes and pending batch)
        if self.id_to_idx.contains_key(&id) {
            return Err(StreamingBuildError::DuplicateId(id));
        }
        if self.pending_batch.iter().any(|(pid, _)| pid == &id) {
            return Err(StreamingBuildError::DuplicateId(id));
        }

        self.pending_batch.push((id, embedding));

        // Process batch if full
        if self.pending_batch.len() >= self.config.batch_size {
            self.process_batch()?;
        }

        Ok(())
    }

    /// Process pending batch
    fn process_batch(&mut self) -> Result<(), StreamingBuildError> {
        if self.pending_batch.is_empty() {
            return Ok(());
        }

        let batch = std::mem::take(&mut self.pending_batch);

        for (id, embedding) in batch {
            self.insert_single(id, embedding)?;
        }

        // Report progress
        if let Some(ref callback) = self.progress_callback {
            let elapsed = self
                .start_time
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            callback(BuildProgress {
                phase: BuildPhase::Indexing,
                items_processed: self.nodes.len(),
                total_items: None,
                elapsed_secs: elapsed,
            });
        }

        Ok(())
    }

    /// Insert a single entry into the graph
    fn insert_single(
        &mut self,
        id: ProofId,
        embedding: Embedding,
    ) -> Result<(), StreamingBuildError> {
        let idx = self.nodes.len();
        let level = self.random_level();

        let node = BatchedHnswNode {
            idx,
            proof_id: id.clone(),
            embedding,
            neighbors: vec![Vec::new(); level + 1],
            max_layer: level,
        };

        self.nodes.push(node);
        self.id_to_idx.insert(id, idx);

        // First node becomes entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(idx);
            self.max_layer = level;
            return Ok(());
        }

        let ep = self.entry_point.unwrap();

        // Search from top layer to level+1
        let mut current_ep = ep;
        for layer in (level + 1..=self.max_layer).rev() {
            current_ep = self.search_layer_single(idx, current_ep, layer);
        }

        // Search and connect at each layer
        for layer in (0..=level.min(self.max_layer)).rev() {
            let m_max = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };

            let candidates =
                self.search_layer(idx, vec![current_ep], self.config.ef_construction, layer);
            let neighbors = self.select_neighbors(idx, &candidates, m_max);

            self.nodes[idx].neighbors[layer] = neighbors.clone();

            // Connect neighbors back
            for &neighbor_idx in &neighbors {
                if self.nodes[neighbor_idx].max_layer >= layer {
                    self.nodes[neighbor_idx].neighbors[layer].push(idx);

                    // Prune if needed
                    if self.nodes[neighbor_idx].neighbors[layer].len() > m_max {
                        let pruned = self.prune_neighbors(neighbor_idx, layer, m_max);
                        self.nodes[neighbor_idx].neighbors[layer] = pruned;
                    }
                }
            }

            if !candidates.is_empty() {
                current_ep = candidates[0].0;
            }
        }

        // Update entry point if higher level
        if level > self.max_layer {
            self.entry_point = Some(idx);
            self.max_layer = level;
        }

        Ok(())
    }

    fn random_level(&mut self) -> usize {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let uniform = (self.rng_state >> 33) as f64 / (1u64 << 31) as f64;
        (-uniform.ln() * self.ml).floor() as usize
    }

    fn search_layer_single(&self, query_idx: usize, ep: usize, layer: usize) -> usize {
        let mut best = ep;
        let mut best_dist = self.distance(query_idx, ep);
        let mut changed = true;

        while changed {
            changed = false;
            if self.nodes[best].max_layer >= layer {
                for &neighbor in &self.nodes[best].neighbors[layer] {
                    let dist = self.distance(query_idx, neighbor);
                    if dist < best_dist {
                        best = neighbor;
                        best_dist = dist;
                        changed = true;
                    }
                }
            }
        }
        best
    }

    fn search_layer(
        &self,
        query_idx: usize,
        entry_points: Vec<usize>,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        use std::collections::{BinaryHeap, HashSet};

        let mut visited: HashSet<usize> = entry_points.iter().copied().collect();
        let mut candidates: BinaryHeap<
            std::cmp::Reverse<(crate::ordered_float::OrderedF32, usize)>,
        > = BinaryHeap::new();
        let mut results: BinaryHeap<(crate::ordered_float::OrderedF32, usize)> = BinaryHeap::new();

        for &ep in &entry_points {
            let dist = self.distance(query_idx, ep);
            candidates.push(std::cmp::Reverse((
                crate::ordered_float::OrderedF32(dist),
                ep,
            )));
            results.push((crate::ordered_float::OrderedF32(dist), ep));
        }

        while let Some(std::cmp::Reverse((crate::ordered_float::OrderedF32(c_dist), c_idx))) =
            candidates.pop()
        {
            let f_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);
            if c_dist > f_dist && results.len() >= ef {
                break;
            }

            if self.nodes[c_idx].max_layer >= layer {
                for &neighbor in &self.nodes[c_idx].neighbors[layer] {
                    if visited.insert(neighbor) {
                        let dist = self.distance(query_idx, neighbor);
                        let f_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);

                        if dist < f_dist || results.len() < ef {
                            candidates.push(std::cmp::Reverse((
                                crate::ordered_float::OrderedF32(dist),
                                neighbor,
                            )));
                            results.push((crate::ordered_float::OrderedF32(dist), neighbor));
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        results
            .into_sorted_vec()
            .into_iter()
            .map(|(d, idx)| (idx, d.0))
            .collect()
    }

    fn select_neighbors(
        &self,
        _query_idx: usize,
        candidates: &[(usize, f32)],
        m_max: usize,
    ) -> Vec<usize> {
        candidates.iter().take(m_max).map(|(idx, _)| *idx).collect()
    }

    fn prune_neighbors(&self, node_idx: usize, layer: usize, m_max: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = self.nodes[node_idx].neighbors[layer]
            .iter()
            .map(|&n| (n, self.distance(node_idx, n)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(m_max).map(|(idx, _)| idx).collect()
    }

    fn distance(&self, a: usize, b: usize) -> f32 {
        let va = &self.nodes[a].embedding.vector;
        let vb = &self.nodes[b].embedding.vector;
        euclidean_distance_sq(va, vb)
    }

    /// Number of nodes added
    pub fn node_count(&self) -> usize {
        self.nodes.len() + self.pending_batch.len()
    }

    /// Build the final index
    pub fn build(mut self) -> Result<BatchedHnswIndex, StreamingBuildError> {
        // Process any remaining batch
        self.process_batch()?;

        if self.nodes.is_empty() {
            return Err(StreamingBuildError::EmptyIndex);
        }

        // Report finalization
        if let Some(ref callback) = self.progress_callback {
            let elapsed = self
                .start_time
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            callback(BuildProgress {
                phase: BuildPhase::Finalizing,
                items_processed: self.nodes.len(),
                total_items: Some(self.nodes.len()),
                elapsed_secs: elapsed,
            });
        }

        Ok(BatchedHnswIndex {
            nodes: self.nodes,
            id_to_idx: self.id_to_idx,
            entry_point: self.entry_point,
            max_layer: self.max_layer,
            dim: self.dim.unwrap(),
            config: self.config,
        })
    }
}

/// HNSW index built via batched builder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedHnswIndex {
    nodes: Vec<BatchedHnswNode>,
    #[allow(dead_code)]
    id_to_idx: HashMap<ProofId, usize>,
    entry_point: Option<usize>,
    max_layer: usize,
    dim: usize,
    config: BatchedHnswConfig,
}

impl BatchedHnswIndex {
    /// Number of indexed entries
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get configuration
    pub fn config(&self) -> &BatchedHnswConfig {
        &self.config
    }

    /// Dimension of stored embeddings
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> BatchedHnswMemoryStats {
        let n = self.nodes.len();
        let embedding_bytes = n * self.dim * 4;

        // Graph structure
        let graph_edges: usize = self
            .nodes
            .iter()
            .map(|n| n.neighbors.iter().map(|l| l.len()).sum::<usize>())
            .sum();
        let graph_bytes = graph_edges * 8; // usize per edge

        // Node overhead
        let node_overhead = n * 64; // approximate

        let total_bytes = embedding_bytes + graph_bytes + node_overhead;

        BatchedHnswMemoryStats {
            num_entries: n,
            embedding_bytes,
            graph_bytes,
            node_overhead,
            total_bytes,
            graph_edges,
            max_layer: self.max_layer,
        }
    }

    /// Find k nearest neighbors
    pub fn find_similar(&self, query: &Embedding, k: usize) -> Vec<(ProofId, f64)> {
        self.find_similar_with_ef(query, k, self.config.ef_search)
    }

    /// Find k nearest neighbors with custom ef
    pub fn find_similar_with_ef(
        &self,
        query: &Embedding,
        k: usize,
        ef: usize,
    ) -> Vec<(ProofId, f64)> {
        if self.nodes.is_empty() || k == 0 || query.dim != self.dim {
            return vec![];
        }

        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return vec![],
        };

        // Search from top layer
        let mut current_ep = ep;
        for layer in (1..=self.max_layer).rev() {
            current_ep = self.search_layer_single_query(query, current_ep, layer);
        }

        // Search layer 0 with ef
        let candidates = self.search_layer_query(query, vec![current_ep], ef.max(k), 0);

        candidates
            .into_iter()
            .take(k)
            .map(|(idx, dist)| (self.nodes[idx].proof_id.clone(), dist as f64))
            .collect()
    }

    fn search_layer_single_query(&self, query: &Embedding, ep: usize, layer: usize) -> usize {
        let mut best = ep;
        let mut best_dist = self.distance_to_query(query, ep);
        let mut changed = true;

        while changed {
            changed = false;
            if self.nodes[best].max_layer >= layer {
                for &neighbor in &self.nodes[best].neighbors[layer] {
                    let dist = self.distance_to_query(query, neighbor);
                    if dist < best_dist {
                        best = neighbor;
                        best_dist = dist;
                        changed = true;
                    }
                }
            }
        }
        best
    }

    fn search_layer_query(
        &self,
        query: &Embedding,
        entry_points: Vec<usize>,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        use std::collections::{BinaryHeap, HashSet};

        let mut visited: HashSet<usize> = entry_points.iter().copied().collect();
        let mut candidates: BinaryHeap<
            std::cmp::Reverse<(crate::ordered_float::OrderedF32, usize)>,
        > = BinaryHeap::new();
        let mut results: BinaryHeap<(crate::ordered_float::OrderedF32, usize)> = BinaryHeap::new();

        for &ep in &entry_points {
            let dist = self.distance_to_query(query, ep);
            candidates.push(std::cmp::Reverse((
                crate::ordered_float::OrderedF32(dist),
                ep,
            )));
            results.push((crate::ordered_float::OrderedF32(dist), ep));
        }

        while let Some(std::cmp::Reverse((crate::ordered_float::OrderedF32(c_dist), c_idx))) =
            candidates.pop()
        {
            let f_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);
            if c_dist > f_dist && results.len() >= ef {
                break;
            }

            if self.nodes[c_idx].max_layer >= layer {
                for &neighbor in &self.nodes[c_idx].neighbors[layer] {
                    if visited.insert(neighbor) {
                        let dist = self.distance_to_query(query, neighbor);
                        let f_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);

                        if dist < f_dist || results.len() < ef {
                            candidates.push(std::cmp::Reverse((
                                crate::ordered_float::OrderedF32(dist),
                                neighbor,
                            )));
                            results.push((crate::ordered_float::OrderedF32(dist), neighbor));
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        results
            .into_sorted_vec()
            .into_iter()
            .map(|(d, idx)| (idx, d.0))
            .collect()
    }

    fn distance_to_query(&self, query: &Embedding, idx: usize) -> f32 {
        euclidean_distance_sq(&query.vector, &self.nodes[idx].embedding.vector)
    }

    // ========== Persistence methods ==========

    /// Save the batched HNSW index to a JSON file
    ///
    /// This persists the graph structure, embeddings, and configuration so the
    /// index can be loaded without rebuilding.
    pub fn save_to_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), crate::LearningError> {
        crate::io::write_json_atomic(path, self)
    }

    /// Load a batched HNSW index from a JSON file
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, crate::LearningError> {
        crate::io::read_json(path)
    }

    /// Load a batched HNSW index if the file exists, otherwise return None
    pub fn load_if_exists<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Option<Self>, crate::LearningError> {
        match std::fs::File::open(path.as_ref()) {
            Ok(_) => Ok(Some(Self::load_from_file(path)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    // ========== Parallel batch query methods ==========

    /// Process multiple queries in parallel using std::thread
    ///
    /// This is useful when you have a batch of queries to process and want to
    /// utilize multiple CPU cores. The parallelism is controlled by `num_threads`.
    ///
    /// # Arguments
    /// * `queries` - Slice of query embeddings
    /// * `k` - Number of nearest neighbors to return per query
    /// * `num_threads` - Number of threads to use (clamped to queries.len())
    ///
    /// # Returns
    /// Vector of results, one per query, in the same order as the input queries
    pub fn find_similar_batch(
        &self,
        queries: &[Embedding],
        k: usize,
        num_threads: usize,
    ) -> Vec<Vec<(ProofId, f64)>> {
        self.find_similar_batch_with_ef(queries, k, self.config.ef_search, num_threads)
    }

    /// Process multiple queries in parallel with custom ef parameter
    ///
    /// # Arguments
    /// * `queries` - Slice of query embeddings
    /// * `k` - Number of nearest neighbors to return per query
    /// * `ef` - Search expansion factor (higher = better recall, slower)
    /// * `num_threads` - Number of threads to use
    pub fn find_similar_batch_with_ef(
        &self,
        queries: &[Embedding],
        k: usize,
        ef: usize,
        num_threads: usize,
    ) -> Vec<Vec<(ProofId, f64)>> {
        if queries.is_empty() {
            return vec![];
        }

        let num_threads = num_threads.min(queries.len()).max(1);

        if num_threads == 1 {
            // Single-threaded fallback
            return queries
                .iter()
                .map(|q| self.find_similar_with_ef(q, k, ef))
                .collect();
        }

        // Use scoped threads for parallel processing
        let chunk_size = queries.len().div_ceil(num_threads);
        let mut results: Vec<Vec<(ProofId, f64)>> = vec![vec![]; queries.len()];

        std::thread::scope(|s| {
            let mut handles = Vec::new();

            for (chunk_idx, chunk) in queries.chunks(chunk_size).enumerate() {
                let start_idx = chunk_idx * chunk_size;
                let handle = s.spawn(move || {
                    chunk
                        .iter()
                        .enumerate()
                        .map(|(i, q)| (start_idx + i, self.find_similar_with_ef(q, k, ef)))
                        .collect::<Vec<_>>()
                });
                handles.push(handle);
            }

            for handle in handles {
                let chunk_results = handle.join().expect("Thread panicked");
                for (idx, result) in chunk_results {
                    results[idx] = result;
                }
            }
        });

        results
    }
}

/// Memory statistics for batched HNSW index
#[derive(Debug, Clone)]
pub struct BatchedHnswMemoryStats {
    pub num_entries: usize,
    pub embedding_bytes: usize,
    pub graph_bytes: usize,
    pub node_overhead: usize,
    pub total_bytes: usize,
    pub graph_edges: usize,
    pub max_layer: usize,
}

// ============================================================================
// Helper functions
// ============================================================================

/// Find nearest centroid
fn find_nearest(point: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, euclidean_distance_sq(point, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    fn make_embedding(dim: usize, seed: u64) -> Embedding {
        let mut rng = SimpleRng::new(seed);
        let vector: Vec<f32> = (0..dim)
            .map(|_| (rng.next_u64() % 1000) as f32 / 1000.0)
            .collect();
        Embedding::new(vector)
    }

    fn make_proof_id(idx: usize) -> ProofId {
        ProofId(format!("proof_{}", idx))
    }

    // StreamingIvfBuilder tests

    #[test]
    fn test_streaming_ivf_builder_new() {
        let config = StreamingIvfConfig::default();
        let builder = StreamingIvfBuilder::new(config, 96);
        assert!(builder.is_ok());
    }

    #[test]
    fn test_streaming_ivf_builder_insufficient_samples() {
        let mut config = StreamingIvfConfig::default();
        config.min_training_samples = 100;
        let mut builder = StreamingIvfBuilder::new(config, 96).unwrap();

        // Add fewer samples than required
        for i in 0..50 {
            builder.add_training_sample(make_proof_id(i), make_embedding(96, i as u64));
        }

        let result = builder.train();
        assert!(matches!(
            result,
            Err(StreamingBuildError::InsufficientTrainingSamples { .. })
        ));
    }

    #[test]
    fn test_streaming_ivf_builder_train_and_add() {
        let mut config = StreamingIvfConfig::default();
        config.min_training_samples = 100;
        config.nlist = 16;
        config.pq.num_subspaces = 8;
        config.pq.num_centroids = 32;

        let mut builder = StreamingIvfBuilder::new(config, 96).unwrap();

        // Add training samples
        for i in 0..150 {
            builder.add_training_sample(make_proof_id(i), make_embedding(96, i as u64));
        }

        assert!(builder.train().is_ok());
        assert!(builder.is_trained());

        // Add more entries via streaming
        for i in 150..300 {
            assert!(builder
                .add_entry(make_proof_id(i), make_embedding(96, i as u64))
                .is_ok());
        }

        let index = builder.build().unwrap();
        assert_eq!(index.len(), 150); // Training samples become entries
    }

    #[test]
    fn test_streaming_ivf_builder_duplicate_id() {
        let mut config = StreamingIvfConfig::default();
        config.min_training_samples = 100;
        config.nlist = 16;
        config.pq.num_subspaces = 8;
        config.pq.num_centroids = 32;

        let mut builder = StreamingIvfBuilder::new(config, 96).unwrap();

        for i in 0..100 {
            builder.add_training_sample(make_proof_id(i), make_embedding(96, i as u64));
        }
        builder.train().unwrap();

        // Add same ID twice
        builder
            .add_entry(make_proof_id(1000), make_embedding(96, 1000))
            .unwrap();
        let result = builder.add_entry(make_proof_id(1000), make_embedding(96, 1001));
        assert!(matches!(result, Err(StreamingBuildError::DuplicateId(_))));
    }

    #[test]
    fn test_streaming_ivf_index_search() {
        let mut config = StreamingIvfConfig::default();
        config.min_training_samples = 100;
        config.nlist = 16;
        config.nprobe = 4;
        config.pq.num_subspaces = 8;
        config.pq.num_centroids = 32;

        let mut builder = StreamingIvfBuilder::new(config, 96).unwrap();

        // Add training samples
        for i in 0..150 {
            builder.add_training_sample(make_proof_id(i), make_embedding(96, i as u64));
        }
        builder.train().unwrap();

        // Add more entries via streaming (these become searchable)
        for i in 0..200 {
            builder
                .add_entry(make_proof_id(i + 1000), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();
        let query = make_embedding(96, 0);
        let results = index.find_similar(&query, 10, 4);

        assert!(!results.is_empty());
        assert!(results.len() <= 10);
        // Query embedding with seed 0 should be similar to proof_1000 (same seed as query)
        assert!(results.iter().any(|(id, _)| id.0 == "proof_1000"));
    }

    #[test]
    fn test_streaming_ivf_opq() {
        let mut config = StreamingIvfConfig::for_large_corpus_opq();
        config.min_training_samples = 100;
        config.nlist = 16;
        config.nprobe = 4; // Must be <= nlist
        if let Some(ref mut opq) = config.opq {
            opq.pq.num_subspaces = 8;
            opq.pq.num_centroids = 32;
        }

        let mut builder = StreamingIvfBuilder::new(config, 96).unwrap();

        // Add training samples
        for i in 0..150 {
            builder.add_training_sample(make_proof_id(i), make_embedding(96, i as u64));
        }
        builder.train().unwrap();

        // Add entries via streaming
        for i in 0..100 {
            builder
                .add_entry(make_proof_id(i + 1000), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();
        let stats = index.memory_stats();

        // OPQ should have rotation bytes
        assert!(stats.rotation_bytes > 0);
    }

    // BatchedHnswBuilder tests

    #[test]
    fn test_batched_hnsw_builder_new() {
        let config = BatchedHnswConfig::default();
        let builder = BatchedHnswBuilder::new(config);
        assert!(builder.is_ok());
    }

    #[test]
    fn test_batched_hnsw_builder_add_entries() {
        let mut config = BatchedHnswConfig::default();
        config.batch_size = 50;

        let mut builder = BatchedHnswBuilder::new(config).unwrap();

        for i in 0..100 {
            assert!(builder
                .add_entry(make_proof_id(i), make_embedding(96, i as u64))
                .is_ok());
        }

        let index = builder.build().unwrap();
        assert_eq!(index.len(), 100);
    }

    #[test]
    fn test_batched_hnsw_builder_duplicate_id() {
        let config = BatchedHnswConfig::default();
        let mut builder = BatchedHnswBuilder::new(config).unwrap();

        builder
            .add_entry(make_proof_id(0), make_embedding(96, 0))
            .unwrap();
        let result = builder.add_entry(make_proof_id(0), make_embedding(96, 1));

        assert!(matches!(result, Err(StreamingBuildError::DuplicateId(_))));
    }

    #[test]
    fn test_batched_hnsw_builder_dimension_mismatch() {
        let config = BatchedHnswConfig::default();
        let mut builder = BatchedHnswBuilder::new(config).unwrap();

        builder
            .add_entry(make_proof_id(0), make_embedding(96, 0))
            .unwrap();
        let result = builder.add_entry(make_proof_id(1), make_embedding(64, 1));

        assert!(matches!(
            result,
            Err(StreamingBuildError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_batched_hnsw_index_search() {
        let mut config = BatchedHnswConfig::default();
        config.batch_size = 50;
        config.m = 8;
        config.m0 = 16;
        config.ef_construction = 50;

        let mut builder = BatchedHnswBuilder::new(config).unwrap();

        for i in 0..200 {
            builder
                .add_entry(make_proof_id(i), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();
        let query = make_embedding(96, 0);
        let results = index.find_similar(&query, 10);

        assert!(!results.is_empty());
        assert!(results.len() <= 10);
        // Query should match itself (seed 0)
        assert!(results.iter().any(|(id, _)| id.0 == "proof_0"));
    }

    #[test]
    fn test_batched_hnsw_index_search_with_ef() {
        let config = BatchedHnswConfig::default();
        let mut builder = BatchedHnswBuilder::new(config).unwrap();

        for i in 0..100 {
            builder
                .add_entry(make_proof_id(i), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();
        let query = make_embedding(96, 50);

        let results_low_ef = index.find_similar_with_ef(&query, 10, 20);
        let results_high_ef = index.find_similar_with_ef(&query, 10, 100);

        // Both should return results
        assert!(!results_low_ef.is_empty());
        assert!(!results_high_ef.is_empty());
    }

    #[test]
    fn test_batched_hnsw_memory_stats() {
        let config = BatchedHnswConfig::default();
        let mut builder = BatchedHnswBuilder::new(config).unwrap();

        for i in 0..100 {
            builder
                .add_entry(make_proof_id(i), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();
        let stats = index.memory_stats();

        assert_eq!(stats.num_entries, 100);
        assert!(stats.embedding_bytes > 0);
        assert!(stats.graph_edges > 0);
    }

    #[test]
    fn test_build_progress() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let mut config = BatchedHnswConfig::default();
        config.batch_size = 25;

        let mut builder = BatchedHnswBuilder::new(config).unwrap();

        let callback_count = Arc::new(AtomicUsize::new(0));
        let callback_count_clone = callback_count.clone();

        builder.set_progress_callback(move |_progress| {
            callback_count_clone.fetch_add(1, Ordering::SeqCst);
        });

        for i in 0..100 {
            builder
                .add_entry(make_proof_id(i), make_embedding(96, i as u64))
                .unwrap();
        }

        let _index = builder.build().unwrap();

        // Should have received multiple progress callbacks
        assert!(callback_count.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn test_streaming_ivf_list_stats() {
        let mut config = StreamingIvfConfig::default();
        config.min_training_samples = 100;
        config.nlist = 8;
        config.nprobe = 4;
        config.pq.num_subspaces = 8;
        config.pq.num_centroids = 32;

        let mut builder = StreamingIvfBuilder::new(config, 96).unwrap();

        // Add training samples
        for i in 0..150 {
            builder.add_training_sample(make_proof_id(i), make_embedding(96, i as u64));
        }
        builder.train().unwrap();

        // Add entries via streaming
        for i in 0..200 {
            builder
                .add_entry(make_proof_id(i + 1000), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();
        let stats = index.list_stats();

        assert_eq!(stats.nlist, 8);
        assert!(stats.non_empty_lists > 0);
        assert!(stats.mean_list_size > 0.0);
    }

    // Edge case tests

    #[test]
    fn test_streaming_ivf_empty_index() {
        let mut config = StreamingIvfConfig::default();
        config.min_training_samples = 100;
        config.nlist = 16;
        config.nprobe = 4;
        config.pq.num_subspaces = 8;
        config.pq.num_centroids = 32;

        let mut builder = StreamingIvfBuilder::new(config, 96).unwrap();

        for i in 0..100 {
            builder.add_training_sample(make_proof_id(i), make_embedding(96, i as u64));
        }
        builder.train().unwrap();

        // Don't add any entries after training - building empty index should error
        let result = builder.build();
        assert!(matches!(result, Err(StreamingBuildError::EmptyIndex)));
    }

    #[test]
    fn test_batched_hnsw_empty_index() {
        let config = BatchedHnswConfig::default();
        let builder = BatchedHnswBuilder::new(config).unwrap();

        let result = builder.build();
        assert!(matches!(result, Err(StreamingBuildError::EmptyIndex)));
    }

    // Persistence tests

    #[test]
    fn test_streaming_ivf_index_save_load() {
        let mut config = StreamingIvfConfig::default();
        config.min_training_samples = 100;
        config.nlist = 16;
        config.nprobe = 4;
        config.pq.num_subspaces = 8;
        config.pq.num_centroids = 32;

        let mut builder = StreamingIvfBuilder::new(config, 96).unwrap();

        // Add training samples
        for i in 0..150 {
            builder.add_training_sample(make_proof_id(i), make_embedding(96, i as u64));
        }
        builder.train().unwrap();

        // Add entries via streaming
        for i in 0..100 {
            builder
                .add_entry(make_proof_id(i + 1000), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();
        let original_len = index.len();
        let original_stats = index.memory_stats();

        // Save and load
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_streaming_ivf_index.json");
        index.save_to_file(&path).unwrap();

        let loaded = StreamingIvfIndex::load_from_file(&path).unwrap();

        assert_eq!(loaded.len(), original_len);
        assert_eq!(
            loaded.memory_stats().num_entries,
            original_stats.num_entries
        );

        // Test search on loaded index
        let query = make_embedding(96, 0);
        let results = loaded.find_similar(&query, 10, 4);
        assert!(!results.is_empty());
        assert!(results.iter().any(|(id, _)| id.0 == "proof_1000"));

        // Clean up
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_streaming_ivf_index_load_if_exists() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_streaming_ivf_nonexistent.json");

        // Should return None for non-existent file
        let result = StreamingIvfIndex::load_if_exists(&path).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_streaming_ivf_opq_save_load() {
        let mut config = StreamingIvfConfig::for_large_corpus_opq();
        config.min_training_samples = 100;
        config.nlist = 16;
        config.nprobe = 4;
        if let Some(ref mut opq) = config.opq {
            opq.pq.num_subspaces = 8;
            opq.pq.num_centroids = 32;
        }

        let mut builder = StreamingIvfBuilder::new(config, 96).unwrap();

        // Add training samples
        for i in 0..150 {
            builder.add_training_sample(make_proof_id(i), make_embedding(96, i as u64));
        }
        builder.train().unwrap();

        // Add entries via streaming
        for i in 0..100 {
            builder
                .add_entry(make_proof_id(i + 1000), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();
        let original_stats = index.memory_stats();

        // Save and load
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_streaming_ivf_opq_index.json");
        index.save_to_file(&path).unwrap();

        let loaded = StreamingIvfIndex::load_from_file(&path).unwrap();

        // OPQ should preserve rotation bytes
        assert!(loaded.memory_stats().rotation_bytes > 0);
        assert_eq!(
            loaded.memory_stats().rotation_bytes,
            original_stats.rotation_bytes
        );

        // Clean up
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_batched_hnsw_index_save_load() {
        let mut config = BatchedHnswConfig::default();
        config.batch_size = 50;
        config.m = 8;
        config.m0 = 16;
        config.ef_construction = 50;

        let mut builder = BatchedHnswBuilder::new(config).unwrap();

        for i in 0..100 {
            builder
                .add_entry(make_proof_id(i), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();
        let original_len = index.len();
        let original_stats = index.memory_stats();

        // Save and load
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_batched_hnsw_index.json");
        index.save_to_file(&path).unwrap();

        let loaded = BatchedHnswIndex::load_from_file(&path).unwrap();

        assert_eq!(loaded.len(), original_len);
        assert_eq!(
            loaded.memory_stats().graph_edges,
            original_stats.graph_edges
        );
        assert_eq!(loaded.memory_stats().max_layer, original_stats.max_layer);

        // Test search on loaded index
        let query = make_embedding(96, 0);
        let results = loaded.find_similar(&query, 10);
        assert!(!results.is_empty());
        assert!(results.iter().any(|(id, _)| id.0 == "proof_0"));

        // Clean up
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_batched_hnsw_index_load_if_exists() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_batched_hnsw_nonexistent.json");

        // Should return None for non-existent file
        let result = BatchedHnswIndex::load_if_exists(&path).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_batched_hnsw_index_search_after_load() {
        let config = BatchedHnswConfig::default();
        let mut builder = BatchedHnswBuilder::new(config).unwrap();

        for i in 0..200 {
            builder
                .add_entry(make_proof_id(i), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();

        // Save and load
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_batched_hnsw_search.json");
        index.save_to_file(&path).unwrap();

        let loaded = BatchedHnswIndex::load_from_file(&path).unwrap();

        // Multiple search queries should work correctly
        for seed in [0u64, 50, 100, 150] {
            let query = make_embedding(96, seed);
            let results = loaded.find_similar(&query, 5);
            assert!(!results.is_empty());
            // The embedding with the same seed should be in top results
            let expected_id = format!("proof_{}", seed);
            assert!(results.iter().any(|(id, _)| id.0 == expected_id));
        }

        // Clean up
        std::fs::remove_file(&path).ok();
    }

    // Parallel batch query tests

    #[test]
    fn test_streaming_ivf_batch_query_single_thread() {
        let mut config = StreamingIvfConfig::default();
        config.min_training_samples = 100;
        config.nlist = 16;
        config.nprobe = 4;
        config.pq.num_subspaces = 8;
        config.pq.num_centroids = 32;

        let mut builder = StreamingIvfBuilder::new(config, 96).unwrap();

        // Add training samples
        for i in 0..150 {
            builder.add_training_sample(make_proof_id(i), make_embedding(96, i as u64));
        }
        builder.train().unwrap();

        // Add entries via streaming
        for i in 0..200 {
            builder
                .add_entry(make_proof_id(i + 1000), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();

        // Create batch of queries
        let queries: Vec<Embedding> = (0..10).map(|i| make_embedding(96, i as u64)).collect();

        // Single-threaded batch query
        let results = index.find_similar_batch(&queries, 5, 4, 1);

        assert_eq!(results.len(), 10);
        for (i, result) in results.iter().enumerate() {
            assert!(!result.is_empty());
            // Should find embedding with same seed
            let expected_id = format!("proof_{}", 1000 + i);
            assert!(result.iter().any(|(id, _)| id.0 == expected_id));
        }
    }

    #[test]
    fn test_streaming_ivf_batch_query_multi_thread() {
        let mut config = StreamingIvfConfig::default();
        config.min_training_samples = 100;
        config.nlist = 16;
        config.nprobe = 4;
        config.pq.num_subspaces = 8;
        config.pq.num_centroids = 32;

        let mut builder = StreamingIvfBuilder::new(config, 96).unwrap();

        // Add training samples
        for i in 0..150 {
            builder.add_training_sample(make_proof_id(i), make_embedding(96, i as u64));
        }
        builder.train().unwrap();

        // Add entries via streaming
        for i in 0..200 {
            builder
                .add_entry(make_proof_id(i + 1000), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();

        // Create batch of queries
        let queries: Vec<Embedding> = (0..20).map(|i| make_embedding(96, i as u64)).collect();

        // Multi-threaded batch query (4 threads)
        let results = index.find_similar_batch(&queries, 5, 4, 4);

        assert_eq!(results.len(), 20);
        // Results should be in correct order
        for (i, result) in results.iter().enumerate() {
            assert!(!result.is_empty());
            let expected_id = format!("proof_{}", 1000 + i);
            assert!(result.iter().any(|(id, _)| id.0 == expected_id));
        }
    }

    #[test]
    fn test_streaming_ivf_batch_query_empty() {
        let mut config = StreamingIvfConfig::default();
        config.min_training_samples = 100;
        config.nlist = 16;
        config.nprobe = 4;
        config.pq.num_subspaces = 8;
        config.pq.num_centroids = 32;

        let mut builder = StreamingIvfBuilder::new(config, 96).unwrap();

        for i in 0..150 {
            builder.add_training_sample(make_proof_id(i), make_embedding(96, i as u64));
        }
        builder.train().unwrap();

        for i in 0..100 {
            builder
                .add_entry(make_proof_id(i + 1000), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();

        // Empty query batch
        let results = index.find_similar_batch(&[], 5, 4, 4);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batched_hnsw_batch_query_single_thread() {
        let mut config = BatchedHnswConfig::default();
        config.batch_size = 50;
        config.m = 8;
        config.m0 = 16;
        config.ef_construction = 50;

        let mut builder = BatchedHnswBuilder::new(config).unwrap();

        for i in 0..200 {
            builder
                .add_entry(make_proof_id(i), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();

        // Create batch of queries
        let queries: Vec<Embedding> = (0..10).map(|i| make_embedding(96, i as u64)).collect();

        // Single-threaded batch query
        let results = index.find_similar_batch(&queries, 5, 1);

        assert_eq!(results.len(), 10);
        for (i, result) in results.iter().enumerate() {
            assert!(!result.is_empty());
            // Should find embedding with same seed
            let expected_id = format!("proof_{}", i);
            assert!(result.iter().any(|(id, _)| id.0 == expected_id));
        }
    }

    #[test]
    fn test_batched_hnsw_batch_query_multi_thread() {
        let mut config = BatchedHnswConfig::default();
        config.batch_size = 50;
        config.m = 8;
        config.m0 = 16;
        config.ef_construction = 50;

        let mut builder = BatchedHnswBuilder::new(config).unwrap();

        for i in 0..200 {
            builder
                .add_entry(make_proof_id(i), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();

        // Create batch of queries
        let queries: Vec<Embedding> = (0..20).map(|i| make_embedding(96, i as u64)).collect();

        // Multi-threaded batch query (4 threads)
        let results = index.find_similar_batch(&queries, 5, 4);

        assert_eq!(results.len(), 20);
        // Results should be in correct order
        for (i, result) in results.iter().enumerate() {
            assert!(!result.is_empty());
            let expected_id = format!("proof_{}", i);
            assert!(result.iter().any(|(id, _)| id.0 == expected_id));
        }
    }

    #[test]
    fn test_batched_hnsw_batch_query_with_ef() {
        let config = BatchedHnswConfig::default();
        let mut builder = BatchedHnswBuilder::new(config).unwrap();

        for i in 0..100 {
            builder
                .add_entry(make_proof_id(i), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();

        let queries: Vec<Embedding> = (0..5).map(|i| make_embedding(96, i as u64)).collect();

        // Test with custom ef
        let results_low_ef = index.find_similar_batch_with_ef(&queries, 5, 20, 2);
        let results_high_ef = index.find_similar_batch_with_ef(&queries, 5, 100, 2);

        assert_eq!(results_low_ef.len(), 5);
        assert_eq!(results_high_ef.len(), 5);

        // Both should return results
        for result in &results_low_ef {
            assert!(!result.is_empty());
        }
        for result in &results_high_ef {
            assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_batched_hnsw_batch_query_empty() {
        let config = BatchedHnswConfig::default();
        let mut builder = BatchedHnswBuilder::new(config).unwrap();

        for i in 0..100 {
            builder
                .add_entry(make_proof_id(i), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();

        // Empty query batch
        let results = index.find_similar_batch(&[], 5, 4);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batched_hnsw_batch_query_thread_count_clamping() {
        let config = BatchedHnswConfig::default();
        let mut builder = BatchedHnswBuilder::new(config).unwrap();

        for i in 0..50 {
            builder
                .add_entry(make_proof_id(i), make_embedding(96, i as u64))
                .unwrap();
        }

        let index = builder.build().unwrap();

        // More threads than queries - should clamp to query count
        let queries: Vec<Embedding> = (0..3).map(|i| make_embedding(96, i as u64)).collect();
        let results = index.find_similar_batch(&queries, 5, 100); // 100 threads for 3 queries

        assert_eq!(results.len(), 3);
        for (i, result) in results.iter().enumerate() {
            assert!(!result.is_empty());
            let expected_id = format!("proof_{}", i);
            assert!(result.iter().any(|(id, _)| id.0 == expected_id));
        }
    }
}
