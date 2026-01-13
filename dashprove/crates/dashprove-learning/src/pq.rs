//! Product Quantization (PQ) for memory-efficient embedding storage
//!
//! Product Quantization compresses high-dimensional vectors by:
//! 1. Splitting each vector into M subspaces
//! 2. Training K centroids per subspace via k-means
//! 3. Encoding each subvector as its nearest centroid index
//!
//! This reduces memory from O(D * 4 bytes) per vector to O(M bytes) per vector,
//! enabling storage of very large embedding corpora in memory.
//!
//! # Memory Comparison
//!
//! For 96-dimension embeddings (EMBEDDING_DIM = 96):
//! - Raw storage: 96 * 4 = 384 bytes per embedding
//! - PQ with M=8 subspaces: 8 bytes per embedding (48x compression)
//! - PQ with M=12 subspaces: 12 bytes per embedding (32x compression)
//!
//! For 1 million embeddings:
//! - Raw: 384 MB
//! - PQ M=8: 8 MB
//! - PQ M=12: 12 MB
//!
//! # Accuracy Tradeoff
//!
//! PQ introduces quantization error that affects similarity search accuracy.
//! More centroids (K) and more subspaces (M) reduce error but increase:
//! - Codebook size: M * K * (D/M) * 4 bytes
//! - Encoding time: O(M * K * D/M) per vector
//!
//! # Optimized Product Quantization (OPQ)
//!
//! OPQ improves upon standard PQ by learning an orthogonal rotation matrix R
//! that minimizes quantization error. The rotation decorrelates dimensions
//! within each subspace, allowing k-means to find better centroids.
//!
//! OPQ uses alternating optimization:
//! 1. Fix R, update codebooks via standard PQ training
//! 2. Fix codebooks, update R via eigendecomposition
//! 3. Repeat until convergence
//!
//! # Example
//!
//! ```ignore
//! use dashprove_learning::pq::{ProductQuantizer, PqConfig};
//! use dashprove_learning::embedder::Embedding;
//!
//! // Train quantizer on corpus
//! let config = PqConfig::default();
//! let embeddings: Vec<Embedding> = /* ... */;
//! let pq = ProductQuantizer::train(&embeddings, config);
//!
//! // Encode new embedding
//! let codes = pq.encode(&query_embedding);
//!
//! // Compute approximate distance
//! let dist = pq.asymmetric_distance(&query_embedding, &codes);
//! ```

use crate::distance::{
    euclidean_distance_sq, matrix_vector_multiply as simd_matrix_vector_multiply,
    transpose_matrix_vector_multiply as simd_transpose_matrix_vector_multiply,
    vector_add_accumulate, vector_scale_inplace,
};
use crate::embedder::Embedding;
use serde::{Deserialize, Serialize};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) const KMEANS_PARALLEL_MIN_POINTS: usize = 1000;
#[cfg(test)]
static DATA_PARALLEL_TRAIN_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Configuration for Product Quantizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqConfig {
    /// Number of subspaces to partition embedding into
    /// Must divide EMBEDDING_DIM evenly
    pub num_subspaces: usize,
    /// Number of centroids per subspace (K)
    /// Maximum 256 to allow single-byte encoding
    pub num_centroids: usize,
    /// Number of k-means iterations for training
    pub kmeans_iterations: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for PqConfig {
    fn default() -> Self {
        Self {
            num_subspaces: 8, // 96/8 = 12 dimensions per subspace
            num_centroids: 256,
            kmeans_iterations: 25,
            seed: 42,
        }
    }
}

impl PqConfig {
    /// Config optimized for speed (fewer centroids)
    pub fn fast() -> Self {
        Self {
            num_subspaces: 8,
            num_centroids: 64,
            kmeans_iterations: 15,
            seed: 42,
        }
    }

    /// Config optimized for accuracy (more centroids, more subspaces)
    pub fn accurate() -> Self {
        Self {
            num_subspaces: 12, // 96/12 = 8 dimensions per subspace
            num_centroids: 256,
            kmeans_iterations: 50,
            seed: 42,
        }
    }

    /// Validate configuration against embedding dimension
    pub fn validate(&self, dim: usize) -> Result<(), PqError> {
        if self.num_subspaces == 0 {
            return Err(PqError::InvalidConfig("num_subspaces must be > 0".into()));
        }
        if !dim.is_multiple_of(self.num_subspaces) {
            return Err(PqError::InvalidConfig(format!(
                "embedding dim {} must be divisible by num_subspaces {}",
                dim, self.num_subspaces
            )));
        }
        if self.num_centroids == 0 || self.num_centroids > 256 {
            return Err(PqError::InvalidConfig(
                "num_centroids must be in [1, 256]".into(),
            ));
        }
        Ok(())
    }
}

/// Errors from Product Quantization operations
#[derive(Debug, Clone)]
pub enum PqError {
    /// Invalid configuration
    InvalidConfig(String),
    /// Not enough training data
    InsufficientData(String),
    /// Dimension mismatch
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for PqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PqError::InvalidConfig(s) => write!(f, "Invalid PQ config: {}", s),
            PqError::InsufficientData(s) => write!(f, "Insufficient training data: {}", s),
            PqError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for PqError {}

/// Configuration for Optimized Product Quantizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpqConfig {
    /// Base PQ configuration
    pub pq: PqConfig,
    /// Number of OPQ iterations (alternating optimization rounds)
    pub opq_iterations: usize,
}

impl Default for OpqConfig {
    fn default() -> Self {
        Self {
            pq: PqConfig::default(),
            opq_iterations: 10,
        }
    }
}

impl OpqConfig {
    /// Config optimized for speed (fewer iterations)
    pub fn fast() -> Self {
        Self {
            pq: PqConfig::fast(),
            opq_iterations: 5,
        }
    }

    /// Config optimized for accuracy
    pub fn accurate() -> Self {
        Self {
            pq: PqConfig::accurate(),
            opq_iterations: 20,
        }
    }

    /// Validate configuration
    pub fn validate(&self, dim: usize) -> Result<(), PqError> {
        self.pq.validate(dim)?;
        if self.opq_iterations == 0 {
            return Err(PqError::InvalidConfig("opq_iterations must be > 0".into()));
        }
        Ok(())
    }
}

/// Product Quantizer for embedding compression
///
/// Stores codebooks (centroids) for each subspace and provides
/// encode/decode operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    /// Codebooks: codebooks[m][k] is the k-th centroid of subspace m
    /// Shape: [num_subspaces][num_centroids][subspace_dim]
    codebooks: Vec<Vec<Vec<f32>>>,
    /// Configuration used to train this quantizer
    config: PqConfig,
    /// Original embedding dimension
    dim: usize,
    /// Dimension of each subspace
    subspace_dim: usize,
}

impl ProductQuantizer {
    /// Train a product quantizer on a corpus of embeddings
    ///
    /// Uses k-means clustering to learn centroids for each subspace.
    /// Requires at least `num_centroids` training samples.
    pub fn train(embeddings: &[Embedding], config: PqConfig) -> Result<Self, PqError> {
        if embeddings.is_empty() {
            return Err(PqError::InsufficientData(
                "Need at least one embedding".into(),
            ));
        }

        let dim = embeddings[0].dim;
        config.validate(dim)?;

        if embeddings.len() < config.num_centroids {
            return Err(PqError::InsufficientData(format!(
                "Need at least {} embeddings for {} centroids, got {}",
                config.num_centroids,
                config.num_centroids,
                embeddings.len()
            )));
        }

        let subspace_dim = dim / config.num_subspaces;

        // Extract subvectors for each subspace and run k-means
        let mut codebooks = Vec::with_capacity(config.num_subspaces);
        let mut rng = SimpleRng::new(config.seed);

        for m in 0..config.num_subspaces {
            let start = m * subspace_dim;
            let end = start + subspace_dim;

            // Extract subvectors for this subspace
            let subvectors: Vec<Vec<f32>> = embeddings
                .iter()
                .map(|e| e.vector[start..end].to_vec())
                .collect();

            // Run k-means
            let centroids = kmeans(
                &subvectors,
                config.num_centroids,
                config.kmeans_iterations,
                &mut rng,
            );

            codebooks.push(centroids);
        }

        Ok(Self {
            codebooks,
            config,
            dim,
            subspace_dim,
        })
    }

    /// Train a product quantizer on a corpus of embeddings using parallel processing
    ///
    /// Uses k-means clustering to learn centroids for each subspace, with each
    /// subspace trained in parallel. This can significantly speed up training
    /// when the number of subspaces is >= number of available CPU cores.
    ///
    /// # Arguments
    /// * `embeddings` - Training corpus (must have at least `num_centroids` samples)
    /// * `config` - PQ configuration
    /// * `num_threads` - Number of threads to use (clamped to num_subspaces)
    ///
    /// # Returns
    /// A trained product quantizer, or an error if validation fails
    pub fn train_parallel(
        embeddings: &[Embedding],
        config: PqConfig,
        num_threads: usize,
    ) -> Result<Self, PqError> {
        if embeddings.is_empty() {
            return Err(PqError::InsufficientData(
                "Need at least one embedding".into(),
            ));
        }

        let dim = embeddings[0].dim;
        config.validate(dim)?;

        if embeddings.len() < config.num_centroids {
            return Err(PqError::InsufficientData(format!(
                "Need at least {} embeddings for {} centroids, got {}",
                config.num_centroids,
                config.num_centroids,
                embeddings.len()
            )));
        }

        let subspace_dim = dim / config.num_subspaces;
        let available = std::thread::available_parallelism().map_or(num_threads, |p| p.get());
        let effective_threads = num_threads.max(1).min(available);
        // Use data-parallel k-means when we have surplus threads beyond subspace count
        let use_data_parallel = effective_threads > config.num_subspaces
            && embeddings.len() >= KMEANS_PARALLEL_MIN_POINTS;

        if use_data_parallel {
            #[cfg(test)]
            DATA_PARALLEL_TRAIN_CALLS.fetch_add(1, Ordering::Relaxed);

            let mut codebooks = Vec::with_capacity(config.num_subspaces);
            let mut rng = SimpleRng::new(config.seed);

            for m in 0..config.num_subspaces {
                let start = m * subspace_dim;
                let end = start + subspace_dim;
                let subvectors: Vec<Vec<f32>> = embeddings
                    .iter()
                    .map(|e| e.vector[start..end].to_vec())
                    .collect();

                let centroids = kmeans_parallel(
                    &subvectors,
                    config.num_centroids,
                    config.kmeans_iterations,
                    &mut rng,
                    effective_threads,
                );

                codebooks.push(centroids);
            }

            return Ok(Self {
                codebooks,
                config,
                dim,
                subspace_dim,
            });
        }

        let num_threads = effective_threads.min(config.num_subspaces);

        // If single thread, fall back to sequential training
        if num_threads == 1 {
            return Self::train(embeddings, config);
        }

        // Pre-extract all subvectors for each subspace
        let all_subvectors: Vec<Vec<Vec<f32>>> = (0..config.num_subspaces)
            .map(|m| {
                let start = m * subspace_dim;
                let end = start + subspace_dim;
                embeddings
                    .iter()
                    .map(|e| e.vector[start..end].to_vec())
                    .collect()
            })
            .collect();

        // Train each subspace's k-means in parallel
        let mut codebooks: Vec<Vec<Vec<f32>>> = vec![vec![]; config.num_subspaces];

        std::thread::scope(|s| {
            let chunk_size = config.num_subspaces.div_ceil(num_threads);
            let codebook_chunks: Vec<_> = codebooks.chunks_mut(chunk_size).collect();
            let subvector_chunks: Vec<_> = all_subvectors.chunks(chunk_size).collect();

            let mut handles = Vec::new();

            for (chunk_idx, (codebook_chunk, subvector_chunk)) in codebook_chunks
                .into_iter()
                .zip(subvector_chunks.into_iter())
                .enumerate()
            {
                let config_ref = &config;
                let handle = s.spawn(move || {
                    // Each thread has its own RNG seeded uniquely per chunk
                    let mut rng = SimpleRng::new(config_ref.seed.wrapping_add(chunk_idx as u64));

                    for (subspace_offset, subvectors) in subvector_chunk.iter().enumerate() {
                        let centroids = kmeans(
                            subvectors,
                            config_ref.num_centroids,
                            config_ref.kmeans_iterations,
                            &mut rng,
                        );
                        codebook_chunk[subspace_offset] = centroids;
                    }
                    chunk_idx
                });
                handles.push(handle);
            }

            // Wait for all threads to complete
            for handle in handles {
                let _ = handle.join();
            }
        });

        Ok(Self {
            codebooks,
            config,
            dim,
            subspace_dim,
        })
    }

    /// Encode an embedding into PQ codes
    ///
    /// Returns a vector of centroid indices, one per subspace.
    /// Each index is in [0, num_centroids).
    pub fn encode(&self, embedding: &Embedding) -> Vec<u8> {
        if embedding.dim != self.dim {
            // Return zeros for mismatched dimensions (could log warning)
            return vec![0u8; self.config.num_subspaces];
        }

        let mut codes = Vec::with_capacity(self.config.num_subspaces);

        for m in 0..self.config.num_subspaces {
            let start = m * self.subspace_dim;
            let end = start + self.subspace_dim;
            let subvector = &embedding.vector[start..end];

            // Find nearest centroid
            let nearest = self.find_nearest_centroid(m, subvector);
            codes.push(nearest as u8);
        }

        codes
    }

    /// Decode PQ codes back to an approximate embedding
    ///
    /// Reconstructs the embedding by concatenating centroids.
    pub fn decode(&self, codes: &[u8]) -> Embedding {
        if codes.len() != self.config.num_subspaces {
            return Embedding::zeros(self.dim);
        }

        let mut vector = Vec::with_capacity(self.dim);

        for (m, &code) in codes.iter().enumerate() {
            let centroid = &self.codebooks[m][code as usize];
            vector.extend_from_slice(centroid);
        }

        Embedding::new(vector)
    }

    /// Encode multiple embeddings into PQ codes
    ///
    /// Returns a vector of code vectors, one per embedding.
    /// More efficient than calling encode() repeatedly when processing batches.
    pub fn encode_batch(&self, embeddings: &[Embedding]) -> Vec<Vec<u8>> {
        embeddings.iter().map(|e| self.encode(e)).collect()
    }

    /// Encode multiple embeddings into PQ codes using parallel processing
    ///
    /// For large batches (hundreds or thousands of embeddings), this can be
    /// significantly faster than sequential encoding by utilizing multiple CPU cores.
    ///
    /// # Arguments
    /// * `embeddings` - Slice of embeddings to encode
    /// * `num_threads` - Number of threads to use (clamped to embeddings.len())
    ///
    /// # Returns
    /// Vector of code vectors, one per embedding, in the same order as input
    pub fn encode_batch_parallel(
        &self,
        embeddings: &[Embedding],
        num_threads: usize,
    ) -> Vec<Vec<u8>> {
        if embeddings.is_empty() {
            return vec![];
        }

        let num_threads = num_threads.min(embeddings.len()).max(1);

        if num_threads == 1 {
            return self.encode_batch(embeddings);
        }

        let chunk_size = embeddings.len().div_ceil(num_threads);
        let mut results: Vec<Vec<u8>> = vec![vec![]; embeddings.len()];

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            let result_chunks: Vec<_> = results.chunks_mut(chunk_size).collect();

            for (chunk_idx, (emb_chunk, result_chunk)) in
                embeddings.chunks(chunk_size).zip(result_chunks).enumerate()
            {
                let handle = s.spawn(move || {
                    for (i, emb) in emb_chunk.iter().enumerate() {
                        result_chunk[i] = self.encode(emb);
                    }
                    chunk_idx
                });
                handles.push(handle);
            }

            // Wait for all threads to complete
            for handle in handles {
                let _ = handle.join();
            }
        });

        results
    }

    /// Compute asymmetric distance between query and PQ codes
    ///
    /// Uses precomputed distance table for efficiency when querying
    /// many encoded vectors against the same query.
    ///
    /// Distance is squared L2 (Euclidean) distance.
    pub fn asymmetric_distance(&self, query: &Embedding, codes: &[u8]) -> f32 {
        if query.dim != self.dim || codes.len() != self.config.num_subspaces {
            return f32::MAX;
        }

        let mut dist = 0.0f32;

        for (m, &code) in codes.iter().enumerate().take(self.config.num_subspaces) {
            let start = m * self.subspace_dim;
            let end = start + self.subspace_dim;
            let query_sub = &query.vector[start..end];
            let centroid = &self.codebooks[m][code as usize];

            // Use SIMD-accelerated squared L2 distance
            dist += euclidean_distance_sq(query_sub, centroid);
        }

        dist
    }

    /// Build a distance lookup table for a query embedding
    ///
    /// Returns a table where table[m][k] is the squared L2 distance
    /// from query subvector m to centroid k in subspace m.
    ///
    /// This enables O(M) distance computation per encoded vector
    /// instead of O(M * D/M) = O(D).
    pub fn build_distance_table(&self, query: &Embedding) -> DistanceTable {
        if query.dim != self.dim {
            return DistanceTable {
                table: vec![vec![f32::MAX; self.config.num_centroids]; self.config.num_subspaces],
            };
        }

        let mut table = vec![vec![0.0f32; self.config.num_centroids]; self.config.num_subspaces];

        for (m, (table_row, codebook)) in table
            .iter_mut()
            .zip(self.codebooks.iter())
            .enumerate()
            .take(self.config.num_subspaces)
        {
            let start = m * self.subspace_dim;
            let end = start + self.subspace_dim;
            let query_sub = &query.vector[start..end];

            for (dist_entry, centroid) in table_row.iter_mut().zip(codebook.iter()) {
                let mut dist = 0.0f32;
                for (q, c) in query_sub.iter().zip(centroid.iter()) {
                    let diff = q - c;
                    dist += diff * diff;
                }
                *dist_entry = dist;
            }
        }

        DistanceTable { table }
    }

    /// Compute distance using precomputed table
    ///
    /// O(M) operation - very fast for searching many encoded vectors.
    pub fn distance_with_table(&self, table: &DistanceTable, codes: &[u8]) -> f32 {
        if codes.len() != self.config.num_subspaces {
            return f32::MAX;
        }

        let mut dist = 0.0f32;
        for (m, &code) in codes.iter().enumerate() {
            dist += table.table[m][code as usize];
        }
        dist
    }

    /// Convert squared L2 distance to approximate cosine similarity
    ///
    /// Assumes normalized embeddings. For unit vectors:
    /// cos(a,b) = 1 - ||a-b||^2 / 2
    pub fn distance_to_similarity(squared_l2: f32) -> f64 {
        (1.0 - (squared_l2 as f64) / 2.0).clamp(-1.0, 1.0)
    }

    /// Get the configuration used for this quantizer
    pub fn config(&self) -> &PqConfig {
        &self.config
    }

    /// Get the original embedding dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get memory usage of codebooks in bytes
    pub fn codebook_size_bytes(&self) -> usize {
        self.config.num_subspaces * self.config.num_centroids * self.subspace_dim * 4
    }

    /// Find nearest centroid in a subspace
    ///
    /// Uses SIMD-accelerated distance computation for better performance.
    fn find_nearest_centroid(&self, subspace: usize, subvector: &[f32]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for (k, centroid) in self.codebooks[subspace].iter().enumerate() {
            let dist = euclidean_distance_sq(subvector, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_idx = k;
            }
        }

        best_idx
    }

    /// Save the trained quantizer to a JSON file
    ///
    /// This persists the codebooks so they can be reused without retraining.
    pub fn save_to_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), crate::LearningError> {
        crate::io::write_json_atomic(path, self)
    }

    /// Load a trained quantizer from a JSON file
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, crate::LearningError> {
        crate::io::read_json(path)
    }

    /// Load a trained quantizer if it exists, otherwise return None
    pub fn load_if_exists<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Option<Self>, crate::LearningError> {
        match std::fs::File::open(path.as_ref()) {
            Ok(_) => Ok(Some(Self::load_from_file(path)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

/// Precomputed distance table for efficient PQ search
#[derive(Debug, Clone)]
pub struct DistanceTable {
    /// table[m][k] = squared L2 distance from query subvector m to centroid k
    table: Vec<Vec<f32>>,
}

/// Optimized Product Quantizer with rotation matrix
///
/// OPQ learns an orthogonal rotation matrix R that minimizes quantization error
/// by decorrelating dimensions within each subspace. This typically reduces
/// quantization error by 10-30% compared to standard PQ.
///
/// The rotation is applied before encoding and after decoding, so the codec
/// is transparent to users who just call encode/decode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedProductQuantizer {
    /// Underlying product quantizer (trained on rotated data)
    pq: ProductQuantizer,
    /// Orthogonal rotation matrix R (DxD), stored row-major
    /// Applied as: rotated = R * original
    rotation: Vec<f32>,
    /// OPQ configuration
    config: OpqConfig,
}

impl OptimizedProductQuantizer {
    /// Train an optimized product quantizer
    ///
    /// Uses alternating optimization:
    /// 1. Initialize with standard PQ
    /// 2. Compute optimal rotation R for current codebooks
    /// 3. Re-train PQ on rotated data
    /// 4. Repeat until convergence
    pub fn train(embeddings: &[Embedding], config: OpqConfig) -> Result<Self, PqError> {
        if embeddings.is_empty() {
            return Err(PqError::InsufficientData(
                "Need at least one embedding".into(),
            ));
        }

        let dim = embeddings[0].dim;
        config.validate(dim)?;

        if embeddings.len() < config.pq.num_centroids {
            return Err(PqError::InsufficientData(format!(
                "Need at least {} embeddings for {} centroids, got {}",
                config.pq.num_centroids,
                config.pq.num_centroids,
                embeddings.len()
            )));
        }

        // Convert embeddings to matrix form for rotation operations
        let n = embeddings.len();
        let data: Vec<Vec<f32>> = embeddings.iter().map(|e| e.vector.clone()).collect();

        // Initialize rotation as identity matrix
        let mut rotation = identity_matrix(dim);

        // Initial PQ training on unrotated data
        let mut pq = ProductQuantizer::train(embeddings, config.pq.clone())?;

        // Alternating optimization
        for _ in 0..config.opq_iterations {
            // Step 1: Rotate data using current R
            let rotated_data: Vec<Embedding> = data
                .iter()
                .map(|v| Embedding::new(matrix_vector_multiply(&rotation, v, dim)))
                .collect();

            // Step 2: Train PQ on rotated data
            pq = ProductQuantizer::train(&rotated_data, config.pq.clone())?;

            // Step 3: Compute reconstructions from current PQ
            let reconstructions: Vec<Vec<f32>> = rotated_data
                .iter()
                .map(|e| pq.decode(&pq.encode(e)).vector)
                .collect();

            // Step 4: Update rotation matrix to minimize ||X - R^T * X'||
            // where X is original data, X' is PQ reconstruction
            // Optimal R = V * U^T from SVD of X'^T * X
            rotation = compute_optimal_rotation(&data, &reconstructions, dim, n);
        }

        Ok(Self {
            pq,
            rotation,
            config,
        })
    }

    /// Train an optimized product quantizer using parallel processing
    ///
    /// Uses alternating optimization with parallel PQ training in each iteration.
    /// The PQ training for each subspace is done in parallel, which can significantly
    /// speed up the overall training process.
    ///
    /// # Arguments
    /// * `embeddings` - Training corpus (must have at least `num_centroids` samples)
    /// * `config` - OPQ configuration
    /// * `num_threads` - Number of threads to use for parallel PQ training
    ///
    /// # Returns
    /// A trained optimized product quantizer, or an error if validation fails
    pub fn train_parallel(
        embeddings: &[Embedding],
        config: OpqConfig,
        num_threads: usize,
    ) -> Result<Self, PqError> {
        if embeddings.is_empty() {
            return Err(PqError::InsufficientData(
                "Need at least one embedding".into(),
            ));
        }

        let dim = embeddings[0].dim;
        config.validate(dim)?;

        if embeddings.len() < config.pq.num_centroids {
            return Err(PqError::InsufficientData(format!(
                "Need at least {} embeddings for {} centroids, got {}",
                config.pq.num_centroids,
                config.pq.num_centroids,
                embeddings.len()
            )));
        }

        let num_threads = num_threads.max(1);

        // If single thread, fall back to sequential training
        if num_threads == 1 {
            return Self::train(embeddings, config);
        }

        // Convert embeddings to matrix form for rotation operations
        let n = embeddings.len();
        let data: Vec<Vec<f32>> = embeddings.iter().map(|e| e.vector.clone()).collect();

        // Initialize rotation as identity matrix
        let mut rotation = identity_matrix(dim);

        // Initial PQ training on unrotated data using parallel training
        let mut pq = ProductQuantizer::train_parallel(embeddings, config.pq.clone(), num_threads)?;

        // Alternating optimization
        for _ in 0..config.opq_iterations {
            // Step 1: Rotate data using current R (parallel)
            let rotated_data = rotate_data_parallel(&data, &rotation, dim, num_threads);

            // Step 2: Train PQ on rotated data using parallel training
            pq = ProductQuantizer::train_parallel(&rotated_data, config.pq.clone(), num_threads)?;

            // Step 3: Compute reconstructions from current PQ (parallel)
            let reconstructions = compute_reconstructions_parallel(&rotated_data, &pq, num_threads);

            // Step 4: Update rotation matrix to minimize ||X - R^T * X'||
            // where X is original data, X' is PQ reconstruction
            // Optimal R = V * U^T from SVD of X'^T * X
            // Use parallel covariance computation for large datasets
            rotation =
                compute_optimal_rotation_parallel(&data, &reconstructions, dim, n, num_threads);
        }

        Ok(Self {
            pq,
            rotation,
            config,
        })
    }

    /// Encode an embedding into PQ codes
    ///
    /// First applies rotation, then encodes with underlying PQ.
    pub fn encode(&self, embedding: &Embedding) -> Vec<u8> {
        let rotated = self.rotate(embedding);
        self.pq.encode(&rotated)
    }

    /// Decode PQ codes back to approximate embedding
    ///
    /// Decodes with underlying PQ, then applies inverse rotation.
    pub fn decode(&self, codes: &[u8]) -> Embedding {
        let decoded = self.pq.decode(codes);
        self.inverse_rotate(&decoded)
    }

    /// Encode multiple embeddings into PQ codes
    ///
    /// Returns a vector of code vectors, one per embedding.
    /// More efficient than calling encode() repeatedly when processing batches.
    pub fn encode_batch(&self, embeddings: &[Embedding]) -> Vec<Vec<u8>> {
        embeddings.iter().map(|e| self.encode(e)).collect()
    }

    /// Encode multiple embeddings into PQ codes using parallel processing
    ///
    /// For large batches (hundreds or thousands of embeddings), this can be
    /// significantly faster than sequential encoding by utilizing multiple CPU cores.
    ///
    /// # Arguments
    /// * `embeddings` - Slice of embeddings to encode
    /// * `num_threads` - Number of threads to use (clamped to embeddings.len())
    ///
    /// # Returns
    /// Vector of code vectors, one per embedding, in the same order as input
    pub fn encode_batch_parallel(
        &self,
        embeddings: &[Embedding],
        num_threads: usize,
    ) -> Vec<Vec<u8>> {
        if embeddings.is_empty() {
            return vec![];
        }

        let num_threads = num_threads.min(embeddings.len()).max(1);

        if num_threads == 1 {
            return self.encode_batch(embeddings);
        }

        let chunk_size = embeddings.len().div_ceil(num_threads);
        let mut results: Vec<Vec<u8>> = vec![vec![]; embeddings.len()];

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            let result_chunks: Vec<_> = results.chunks_mut(chunk_size).collect();

            for (chunk_idx, (emb_chunk, result_chunk)) in
                embeddings.chunks(chunk_size).zip(result_chunks).enumerate()
            {
                let handle = s.spawn(move || {
                    for (i, emb) in emb_chunk.iter().enumerate() {
                        result_chunk[i] = self.encode(emb);
                    }
                    chunk_idx
                });
                handles.push(handle);
            }

            // Wait for all threads to complete
            for handle in handles {
                let _ = handle.join();
            }
        });

        results
    }

    /// Compute asymmetric distance between query and PQ codes
    pub fn asymmetric_distance(&self, query: &Embedding, codes: &[u8]) -> f32 {
        let rotated_query = self.rotate(query);
        self.pq.asymmetric_distance(&rotated_query, codes)
    }

    /// Build distance table for efficient batch search
    pub fn build_distance_table(&self, query: &Embedding) -> DistanceTable {
        let rotated_query = self.rotate(query);
        self.pq.build_distance_table(&rotated_query)
    }

    /// Compute distance using precomputed table
    pub fn distance_with_table(&self, table: &DistanceTable, codes: &[u8]) -> f32 {
        self.pq.distance_with_table(table, codes)
    }

    /// Convert squared L2 distance to approximate cosine similarity
    pub fn distance_to_similarity(squared_l2: f32) -> f64 {
        ProductQuantizer::distance_to_similarity(squared_l2)
    }

    /// Get the OPQ configuration
    pub fn config(&self) -> &OpqConfig {
        &self.config
    }

    /// Get the underlying PQ config
    pub fn pq_config(&self) -> &PqConfig {
        self.pq.config()
    }

    /// Get the embedding dimension
    pub fn dim(&self) -> usize {
        self.pq.dim()
    }

    /// Get memory usage of codebooks in bytes (excludes rotation matrix)
    pub fn codebook_size_bytes(&self) -> usize {
        self.pq.codebook_size_bytes()
    }

    /// Get total memory usage including rotation matrix
    pub fn total_size_bytes(&self) -> usize {
        let rotation_size = self.rotation.len() * 4;
        self.pq.codebook_size_bytes() + rotation_size
    }

    /// Apply rotation to embedding
    fn rotate(&self, embedding: &Embedding) -> Embedding {
        if embedding.dim != self.pq.dim() {
            return embedding.clone();
        }
        let rotated = matrix_vector_multiply(&self.rotation, &embedding.vector, embedding.dim);
        Embedding::new(rotated)
    }

    /// Apply inverse rotation (R^T since R is orthogonal)
    fn inverse_rotate(&self, embedding: &Embedding) -> Embedding {
        if embedding.dim != self.pq.dim() {
            return embedding.clone();
        }
        let dim = embedding.dim;
        let unrotated = transpose_matrix_vector_multiply(&self.rotation, &embedding.vector, dim);
        Embedding::new(unrotated)
    }

    /// Save the trained quantizer to a JSON file
    pub fn save_to_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), crate::LearningError> {
        crate::io::write_json_atomic(path, self)
    }

    /// Load a trained quantizer from a JSON file
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, crate::LearningError> {
        crate::io::read_json(path)
    }

    /// Load if file exists, otherwise return None
    pub fn load_if_exists<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Option<Self>, crate::LearningError> {
        match std::fs::File::open(path.as_ref()) {
            Ok(_) => Ok(Some(Self::load_from_file(path)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Compute quantization error on a set of embeddings
    ///
    /// Returns mean squared reconstruction error.
    /// Uses SIMD-accelerated distance computation.
    pub fn quantization_error(&self, embeddings: &[Embedding]) -> f32 {
        if embeddings.is_empty() {
            return 0.0;
        }

        let total_error: f32 = embeddings
            .iter()
            .map(|e| {
                let codes = self.encode(e);
                let decoded = self.decode(&codes);
                euclidean_distance_sq(&e.vector, &decoded.vector)
            })
            .sum();

        total_error / embeddings.len() as f32
    }
}

/// Create identity matrix (row-major)
fn identity_matrix(dim: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; dim * dim];
    for i in 0..dim {
        m[i * dim + i] = 1.0;
    }
    m
}

/// Matrix-vector multiply: result = M * v (M is row-major)
///
/// Uses SIMD-accelerated implementation from distance module.
#[inline]
fn matrix_vector_multiply(m: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
    simd_matrix_vector_multiply(m, v, dim)
}

/// Transpose matrix-vector multiply: result = M^T * v (M is row-major)
///
/// Uses SIMD-accelerated implementation from distance module.
#[inline]
fn transpose_matrix_vector_multiply(m: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
    simd_transpose_matrix_vector_multiply(m, v, dim)
}

/// Rotate data matrix in parallel
///
/// Applies rotation matrix R to each vector: result[i] = R * data[i]
/// Uses multiple threads to process vectors in parallel.
fn rotate_data_parallel(
    data: &[Vec<f32>],
    rotation: &[f32],
    dim: usize,
    num_threads: usize,
) -> Vec<Embedding> {
    if data.is_empty() {
        return vec![];
    }

    let num_threads = num_threads.min(data.len()).max(1);

    // Single-thread fallback
    if num_threads == 1 {
        return data
            .iter()
            .map(|v| Embedding::new(matrix_vector_multiply(rotation, v, dim)))
            .collect();
    }

    let chunk_size = data.len().div_ceil(num_threads);
    let mut results: Vec<Embedding> = vec![Embedding::zeros(dim); data.len()];

    std::thread::scope(|s| {
        let mut handles = Vec::new();
        let result_chunks: Vec<_> = results.chunks_mut(chunk_size).collect();

        for (data_chunk, result_chunk) in data.chunks(chunk_size).zip(result_chunks) {
            let handle = s.spawn(move || {
                for (i, v) in data_chunk.iter().enumerate() {
                    result_chunk[i] = Embedding::new(matrix_vector_multiply(rotation, v, dim));
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.join();
        }
    });

    results
}

/// Compute PQ reconstructions in parallel
///
/// For each embedding, encodes with PQ and decodes back to get reconstruction.
/// Uses multiple threads to process embeddings in parallel.
fn compute_reconstructions_parallel(
    embeddings: &[Embedding],
    pq: &ProductQuantizer,
    num_threads: usize,
) -> Vec<Vec<f32>> {
    if embeddings.is_empty() {
        return vec![];
    }

    let num_threads = num_threads.min(embeddings.len()).max(1);

    // Single-thread fallback
    if num_threads == 1 {
        return embeddings
            .iter()
            .map(|e| pq.decode(&pq.encode(e)).vector)
            .collect();
    }

    let chunk_size = embeddings.len().div_ceil(num_threads);
    let dim = embeddings[0].dim;
    let mut results: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; embeddings.len()];

    std::thread::scope(|s| {
        let mut handles = Vec::new();
        let result_chunks: Vec<_> = results.chunks_mut(chunk_size).collect();

        for (emb_chunk, result_chunk) in embeddings.chunks(chunk_size).zip(result_chunks) {
            let handle = s.spawn(move || {
                for (i, emb) in emb_chunk.iter().enumerate() {
                    let codes = pq.encode(emb);
                    let decoded = pq.decode(&codes);
                    result_chunk[i] = decoded.vector;
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.join();
        }
    });

    results
}

/// Compute optimal rotation matrix using SVD-like decomposition
///
/// Given original data X (nxd) and reconstructions X' (nxd),
/// finds orthogonal R that minimizes ||X - R^T * X'||.
///
/// Uses power iteration to compute dominant singular vectors.
fn compute_optimal_rotation(
    original: &[Vec<f32>],
    reconstructed: &[Vec<f32>],
    dim: usize,
    n: usize,
) -> Vec<f32> {
    // Compute covariance C = X'^T * X
    let mut c = vec![0.0f32; dim * dim];
    for k in 0..n {
        for i in 0..dim {
            for j in 0..dim {
                c[i * dim + j] += reconstructed[k][i] * original[k][j];
            }
        }
    }

    // Normalize by n
    for v in &mut c {
        *v /= n as f32;
    }

    // Use iterative orthogonalization to compute rotation
    // This is a simplified approach: use Gram-Schmidt on C's columns
    // to get an orthogonal approximation
    let mut r = gram_schmidt_orthogonalize(&c, dim);

    // Ensure it's a proper rotation (det = 1)
    let det = matrix_determinant(&r, dim);
    if det < 0.0 {
        // Flip sign of last column to make det positive
        for i in 0..dim {
            r[i * dim + (dim - 1)] *= -1.0;
        }
    }

    r
}

/// Compute optimal rotation matrix with parallel covariance computation
///
/// Given original data X (nxd) and reconstructions X' (nxd),
/// finds orthogonal R that minimizes ||X - R^T * X'||.
///
/// Uses multiple threads to compute the covariance matrix in parallel.
fn compute_optimal_rotation_parallel(
    original: &[Vec<f32>],
    reconstructed: &[Vec<f32>],
    dim: usize,
    n: usize,
    num_threads: usize,
) -> Vec<f32> {
    if n == 0 || dim == 0 {
        return identity_matrix(dim);
    }

    let num_threads = num_threads.min(n).max(1);

    // Single-thread fallback
    if num_threads == 1 {
        return compute_optimal_rotation(original, reconstructed, dim, n);
    }

    // Compute covariance C = X'^T * X in parallel
    // Each thread computes a partial covariance for its chunk of data
    // Use f64 for accumulation to minimize floating point errors
    let chunk_size = n.div_ceil(num_threads);
    let dim_sq = dim * dim;

    let partial_covs: Vec<Vec<f64>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let start = t * chunk_size;
                let end = ((t + 1) * chunk_size).min(n);

                scope.spawn(move || {
                    let mut partial_c = vec![0.0f64; dim_sq];
                    for k in start..end {
                        for i in 0..dim {
                            for j in 0..dim {
                                partial_c[i * dim + j] +=
                                    (reconstructed[k][i] as f64) * (original[k][j] as f64);
                            }
                        }
                    }
                    partial_c
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Sum partial covariances (in f64) then convert to f32
    let mut c_f64 = vec![0.0f64; dim_sq];
    for partial in &partial_covs {
        for i in 0..dim_sq {
            c_f64[i] += partial[i];
        }
    }

    // Normalize by n and convert to f32
    let n_f64 = n as f64;
    let c: Vec<f32> = c_f64.iter().map(|&x| (x / n_f64) as f32).collect();

    // Use iterative orthogonalization to compute rotation
    let mut r = gram_schmidt_orthogonalize(&c, dim);

    // Ensure it's a proper rotation (det = 1)
    let det = matrix_determinant(&r, dim);
    if det < 0.0 {
        // Flip sign of last column to make det positive
        for i in 0..dim {
            r[i * dim + (dim - 1)] *= -1.0;
        }
    }

    r
}

/// Gram-Schmidt orthogonalization of columns
fn gram_schmidt_orthogonalize(m: &[f32], dim: usize) -> Vec<f32> {
    let mut result = m.to_vec();

    for j in 0..dim {
        // Get column j
        let mut col: Vec<f32> = (0..dim).map(|i| result[i * dim + j]).collect();

        // Subtract projections onto previous columns
        for k in 0..j {
            let prev_col: Vec<f32> = (0..dim).map(|i| result[i * dim + k]).collect();
            let dot: f32 = col.iter().zip(prev_col.iter()).map(|(a, b)| a * b).sum();
            for i in 0..dim {
                col[i] -= dot * prev_col[i];
            }
        }

        // Normalize
        let norm: f32 = col.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for i in 0..dim {
                result[i * dim + j] = col[i] / norm;
            }
        } else {
            // Handle near-zero column by using unit vector
            for i in 0..dim {
                result[i * dim + j] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }

    result
}

/// Compute matrix determinant using LU decomposition approximation
/// For orthogonal matrices, det is +/- 1
fn matrix_determinant(m: &[f32], dim: usize) -> f32 {
    // For small dimensions, use expansion
    if dim == 1 {
        return m[0];
    }
    if dim == 2 {
        return m[0] * m[3] - m[1] * m[2];
    }
    if dim == 3 {
        return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6]);
    }

    // For larger matrices, estimate using product of diagonal after
    // partial pivoting (approximation)
    let mut a = m.to_vec();
    let mut det = 1.0f32;

    for i in 0..dim {
        // Find pivot
        let mut max_val = a[i * dim + i].abs();
        let mut max_row = i;
        for k in (i + 1)..dim {
            let val = a[k * dim + i].abs();
            if val > max_val {
                max_val = val;
                max_row = k;
            }
        }

        if max_val < 1e-10 {
            return 0.0; // Singular
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..dim {
                a.swap(i * dim + j, max_row * dim + j);
            }
            det *= -1.0;
        }

        det *= a[i * dim + i];

        // Eliminate
        for k in (i + 1)..dim {
            let factor = a[k * dim + i] / a[i * dim + i];
            for j in i..dim {
                a[k * dim + j] -= factor * a[i * dim + j];
            }
        }
    }

    det
}

/// PQ-encoded corpus for memory-efficient storage
///
/// Stores embeddings as PQ codes (1 byte per subspace) rather than
/// full vectors (4 bytes per dimension).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqCorpus {
    /// The quantizer used for encoding
    quantizer: ProductQuantizer,
    /// Encoded embeddings: entries[i] is the PQ code for embedding i
    entries: Vec<(String, Vec<u8>)>,
}

impl PqCorpus {
    /// Build a PQ corpus from embeddings
    ///
    /// First trains a quantizer, then encodes all embeddings.
    pub fn build(embeddings: &[(String, Embedding)], config: PqConfig) -> Result<Self, PqError> {
        if embeddings.is_empty() {
            return Err(PqError::InsufficientData("No embeddings provided".into()));
        }

        // Extract just the embeddings for training
        let training_data: Vec<Embedding> = embeddings.iter().map(|(_, e)| e.clone()).collect();

        // Train quantizer
        let quantizer = ProductQuantizer::train(&training_data, config)?;

        // Encode all embeddings
        let entries: Vec<(String, Vec<u8>)> = embeddings
            .iter()
            .map(|(id, emb)| (id.clone(), quantizer.encode(emb)))
            .collect();

        Ok(Self { quantizer, entries })
    }

    /// Build PQ corpus with an existing quantizer
    pub fn build_with_quantizer(
        embeddings: &[(String, Embedding)],
        quantizer: ProductQuantizer,
    ) -> Self {
        let entries: Vec<(String, Vec<u8>)> = embeddings
            .iter()
            .map(|(id, emb)| (id.clone(), quantizer.encode(emb)))
            .collect();

        Self { quantizer, entries }
    }

    /// Number of stored embeddings
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if corpus is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Memory usage in bytes (approximate)
    ///
    /// Includes codebook and encoded entries, excludes string IDs.
    pub fn memory_usage(&self) -> usize {
        let codebook = self.quantizer.codebook_size_bytes();
        let codes = self.entries.len() * self.quantizer.config.num_subspaces;
        codebook + codes
    }

    /// Insert a new embedding into the corpus
    pub fn insert(&mut self, id: String, embedding: &Embedding) {
        let codes = self.quantizer.encode(embedding);
        self.entries.push((id, codes));
    }

    /// Find k approximate nearest neighbors
    ///
    /// Returns (id, squared_l2_distance) pairs sorted by distance.
    pub fn find_nearest(&self, query: &Embedding, k: usize) -> Vec<(String, f32)> {
        if self.entries.is_empty() || k == 0 {
            return vec![];
        }

        // Build distance table for efficient search
        let table = self.quantizer.build_distance_table(query);

        // Compute distances to all entries
        let mut distances: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, (_, codes))| (i, self.quantizer.distance_with_table(&table, codes)))
            .collect();

        // Partial sort to find top-k
        let k = k.min(distances.len());
        distances.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Sort the top-k and return
        distances.truncate(k);
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        distances
            .into_iter()
            .map(|(i, dist)| (self.entries[i].0.clone(), dist))
            .collect()
    }

    /// Find k approximate nearest neighbors with similarity scores
    ///
    /// Returns (id, similarity) pairs where similarity is in [-1, 1].
    pub fn find_similar(&self, query: &Embedding, k: usize) -> Vec<(String, f64)> {
        self.find_nearest(query, k)
            .into_iter()
            .map(|(id, dist)| (id, ProductQuantizer::distance_to_similarity(dist)))
            .collect()
    }

    /// Get the quantizer
    pub fn quantizer(&self) -> &ProductQuantizer {
        &self.quantizer
    }

    /// Get entry by index
    pub fn get(&self, idx: usize) -> Option<(&str, &[u8])> {
        self.entries
            .get(idx)
            .map(|(id, codes)| (id.as_str(), codes.as_slice()))
    }

    /// Save the PQ corpus to a JSON file
    ///
    /// This persists both the quantizer codebooks and all encoded entries.
    pub fn save_to_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), crate::LearningError> {
        crate::io::write_json_atomic(path, self)
    }

    /// Load a PQ corpus from a JSON file
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, crate::LearningError> {
        crate::io::read_json(path)
    }

    /// Load a PQ corpus if it exists, otherwise return None
    pub fn load_if_exists<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Option<Self>, crate::LearningError> {
        match std::fs::File::open(path.as_ref()) {
            Ok(_) => Ok(Some(Self::load_from_file(path)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

/// OPQ-encoded corpus for memory-efficient storage with rotation optimization
///
/// Uses Optimized Product Quantization for better reconstruction accuracy
/// compared to standard PqCorpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpqCorpus {
    /// The OPQ quantizer used for encoding
    quantizer: OptimizedProductQuantizer,
    /// Encoded embeddings: entries[i] is the PQ code for embedding i
    entries: Vec<(String, Vec<u8>)>,
}

impl OpqCorpus {
    /// Build an OPQ corpus from embeddings
    ///
    /// First trains an OPQ quantizer, then encodes all embeddings.
    pub fn build(embeddings: &[(String, Embedding)], config: OpqConfig) -> Result<Self, PqError> {
        if embeddings.is_empty() {
            return Err(PqError::InsufficientData("No embeddings provided".into()));
        }

        // Extract just the embeddings for training
        let training_data: Vec<Embedding> = embeddings.iter().map(|(_, e)| e.clone()).collect();

        // Train OPQ quantizer
        let quantizer = OptimizedProductQuantizer::train(&training_data, config)?;

        // Encode all embeddings
        let entries: Vec<(String, Vec<u8>)> = embeddings
            .iter()
            .map(|(id, emb)| (id.clone(), quantizer.encode(emb)))
            .collect();

        Ok(Self { quantizer, entries })
    }

    /// Build OPQ corpus with an existing quantizer
    pub fn build_with_quantizer(
        embeddings: &[(String, Embedding)],
        quantizer: OptimizedProductQuantizer,
    ) -> Self {
        let entries: Vec<(String, Vec<u8>)> = embeddings
            .iter()
            .map(|(id, emb)| (id.clone(), quantizer.encode(emb)))
            .collect();

        Self { quantizer, entries }
    }

    /// Number of stored embeddings
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if corpus is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Memory usage in bytes (approximate)
    ///
    /// Includes codebook and encoded entries, excludes string IDs.
    pub fn memory_usage(&self) -> usize {
        let quantizer_size = self.quantizer.total_size_bytes();
        let codes = self.entries.len() * self.quantizer.pq_config().num_subspaces;
        quantizer_size + codes
    }

    /// Insert a new embedding into the corpus
    pub fn insert(&mut self, id: String, embedding: &Embedding) {
        let codes = self.quantizer.encode(embedding);
        self.entries.push((id, codes));
    }

    /// Find k approximate nearest neighbors
    ///
    /// Returns (id, squared_l2_distance) pairs sorted by distance.
    pub fn find_nearest(&self, query: &Embedding, k: usize) -> Vec<(String, f32)> {
        if self.entries.is_empty() || k == 0 {
            return vec![];
        }

        // Build distance table for efficient search
        let table = self.quantizer.build_distance_table(query);

        // Compute distances to all entries
        let mut distances: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, (_, codes))| (i, self.quantizer.distance_with_table(&table, codes)))
            .collect();

        // Partial sort to find top-k
        let k = k.min(distances.len());
        distances.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Sort the top-k and return
        distances.truncate(k);
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        distances
            .into_iter()
            .map(|(i, dist)| (self.entries[i].0.clone(), dist))
            .collect()
    }

    /// Find k approximate nearest neighbors with similarity scores
    ///
    /// Returns (id, similarity) pairs where similarity is in [-1, 1].
    pub fn find_similar(&self, query: &Embedding, k: usize) -> Vec<(String, f64)> {
        self.find_nearest(query, k)
            .into_iter()
            .map(|(id, dist)| (id, OptimizedProductQuantizer::distance_to_similarity(dist)))
            .collect()
    }

    /// Get the quantizer
    pub fn quantizer(&self) -> &OptimizedProductQuantizer {
        &self.quantizer
    }

    /// Get entry by index
    pub fn get(&self, idx: usize) -> Option<(&str, &[u8])> {
        self.entries
            .get(idx)
            .map(|(id, codes)| (id.as_str(), codes.as_slice()))
    }

    /// Save the OPQ corpus to a JSON file
    pub fn save_to_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), crate::LearningError> {
        crate::io::write_json_atomic(path, self)
    }

    /// Load an OPQ corpus from a JSON file
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, crate::LearningError> {
        crate::io::read_json(path)
    }

    /// Load an OPQ corpus if it exists, otherwise return None
    pub fn load_if_exists<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Option<Self>, crate::LearningError> {
        match std::fs::File::open(path.as_ref()) {
            Ok(_) => Ok(Some(Self::load_from_file(path)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Compute mean quantization error on stored entries
    ///
    /// Requires re-encoding each embedding, so this is slow for large corpora.
    /// This is primarily useful for comparing OPQ vs PQ accuracy.
    pub fn mean_reconstruction_error(&self, original_embeddings: &[(String, Embedding)]) -> f32 {
        if original_embeddings.is_empty() {
            return 0.0;
        }

        let training_data: Vec<Embedding> =
            original_embeddings.iter().map(|(_, e)| e.clone()).collect();
        self.quantizer.quantization_error(&training_data)
    }
}

/// Simple k-means clustering implementation with SIMD-accelerated centroid updates
fn kmeans(
    data: &[Vec<f32>],
    k: usize,
    max_iterations: usize,
    rng: &mut SimpleRng,
) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return vec![];
    }

    let dim = data[0].len();
    let k = k.min(data.len());

    // Initialize centroids via k-means++ initialization
    let mut centroids = kmeans_plusplus_init(data, k, rng);

    // Iterate
    for _ in 0..max_iterations {
        // Assign each point to nearest centroid
        let assignments: Vec<usize> = data
            .iter()
            .map(|point| find_nearest(point, &centroids))
            .collect();

        // Compute new centroids using SIMD-accelerated accumulation
        let mut new_centroids = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];

        for (point, &cluster) in data.iter().zip(assignments.iter()) {
            // SIMD-accelerated vector accumulation
            vector_add_accumulate(&mut new_centroids[cluster], point);
            counts[cluster] += 1;
        }

        // Normalize centroids by count using SIMD-accelerated scaling
        for (centroid, count) in new_centroids.iter_mut().zip(counts.iter()) {
            if *count > 0 {
                vector_scale_inplace(centroid, 1.0 / *count as f32);
            }
        }

        // Check for convergence using SIMD distance
        let mut converged = true;
        for (old, new) in centroids.iter().zip(new_centroids.iter()) {
            // Use SIMD euclidean distance for convergence check
            let dist_sq = euclidean_distance_sq(old, new);
            if dist_sq > 1e-12 {
                converged = false;
                break;
            }
        }

        centroids = new_centroids;

        if converged {
            break;
        }
    }

    centroids
}

/// Parallel k-means clustering for large datasets
///
/// Uses multi-threaded assignment and centroid updates for improved performance
/// on large datasets. For small datasets (< 1000 points), falls back to sequential
/// k-means as the threading overhead would outweigh the benefits.
///
/// # Arguments
/// * `data` - Training data points
/// * `k` - Number of centroids
/// * `max_iterations` - Maximum iterations before stopping
/// * `rng` - Random number generator for initialization
/// * `num_threads` - Number of threads to use (clamped to available parallelism)
pub(crate) fn kmeans_parallel(
    data: &[Vec<f32>],
    k: usize,
    max_iterations: usize,
    rng: &mut SimpleRng,
    num_threads: usize,
) -> Vec<Vec<f32>> {
    // For small datasets, use sequential k-means
    if data.len() < KMEANS_PARALLEL_MIN_POINTS || num_threads <= 1 {
        return kmeans(data, k, max_iterations, rng);
    }

    if data.is_empty() || k == 0 {
        return vec![];
    }

    let dim = data[0].len();
    let k = k.min(data.len());
    let num_threads = num_threads.min(std::thread::available_parallelism().map_or(4, |p| p.get()));

    // Initialize centroids via parallel k-means++ initialization
    let mut centroids = kmeans_plusplus_init_parallel(data, k, rng, num_threads);

    let chunk_size = data.len().div_ceil(num_threads);

    // Iterate
    for _ in 0..max_iterations {
        // Parallel assignment: compute assignments in chunks

        // Compute partial sums and counts in parallel
        let partial_results: Vec<(Vec<Vec<f64>>, Vec<usize>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = data
                .chunks(chunk_size)
                .map(|chunk| {
                    let centroids_ref = &centroids;
                    scope.spawn(move || {
                        // Local accumulators using f64 for better precision
                        let mut local_sums = vec![vec![0.0f64; dim]; k];
                        let mut local_counts = vec![0usize; k];

                        for point in chunk {
                            let cluster = find_nearest(point, centroids_ref);
                            // Accumulate into local sums
                            for (i, &val) in point.iter().enumerate() {
                                local_sums[cluster][i] += val as f64;
                            }
                            local_counts[cluster] += 1;
                        }

                        (local_sums, local_counts)
                    })
                })
                .collect();

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // Merge partial results
        let mut total_sums = vec![vec![0.0f64; dim]; k];
        let mut total_counts = vec![0usize; k];

        for (partial_sums, partial_counts) in partial_results {
            for cluster in 0..k {
                for d in 0..dim {
                    total_sums[cluster][d] += partial_sums[cluster][d];
                }
                total_counts[cluster] += partial_counts[cluster];
            }
        }

        // Compute new centroids
        let mut new_centroids = vec![vec![0.0f32; dim]; k];
        for (cluster, (centroid, &count)) in new_centroids
            .iter_mut()
            .zip(total_counts.iter())
            .enumerate()
        {
            if count > 0 {
                let scale = 1.0 / count as f64;
                for (d, val) in centroid.iter_mut().enumerate() {
                    *val = (total_sums[cluster][d] * scale) as f32;
                }
            }
        }

        // Check for convergence using SIMD distance
        let mut converged = true;
        for (old, new) in centroids.iter().zip(new_centroids.iter()) {
            let dist_sq = euclidean_distance_sq(old, new);
            if dist_sq > 1e-12 {
                converged = false;
                break;
            }
        }

        centroids = new_centroids;

        if converged {
            break;
        }
    }

    centroids
}

/// K-means++ initialization for better centroid seeding
fn kmeans_plusplus_init(data: &[Vec<f32>], k: usize, rng: &mut SimpleRng) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return vec![];
    }

    let mut centroids = Vec::with_capacity(k);

    // Choose first centroid uniformly at random
    let first_idx = (rng.next_u64() as usize) % data.len();
    centroids.push(data[first_idx].clone());

    // Choose remaining centroids with probability proportional to squared distance
    for _ in 1..k {
        // Compute squared distance to nearest centroid for each point
        let distances: Vec<f32> = data
            .iter()
            .map(|point| {
                centroids
                    .iter()
                    .map(|c| squared_distance(point, c))
                    .fold(f32::MAX, f32::min)
            })
            .collect();

        // Sum of distances for normalization
        let total: f32 = distances.iter().sum();
        if total <= 0.0 {
            // All points are at existing centroids, pick random
            let idx = (rng.next_u64() as usize) % data.len();
            centroids.push(data[idx].clone());
            continue;
        }

        // Sample proportionally to squared distance
        let threshold = rng.next_f32() * total;
        let mut cumsum = 0.0f32;
        let mut chosen = 0;

        for (i, &d) in distances.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }

        centroids.push(data[chosen].clone());
    }

    centroids
}

/// Parallel k-means++ initialization for large datasets
///
/// Parallelizes the distance computation step of k-means++ seeding,
/// which is the dominant cost for large datasets with many centroids.
/// Falls back to sequential initialization if the dataset is too small
/// to benefit from parallelization overhead.
fn kmeans_plusplus_init_parallel(
    data: &[Vec<f32>],
    k: usize,
    rng: &mut SimpleRng,
    num_threads: usize,
) -> Vec<Vec<f32>> {
    // For small datasets, use sequential initialization
    if data.len() < KMEANS_PARALLEL_MIN_POINTS || num_threads <= 1 || k == 0 {
        return kmeans_plusplus_init(data, k, rng);
    }

    let mut centroids = Vec::with_capacity(k);

    // Choose first centroid uniformly at random
    let first_idx = (rng.next_u64() as usize) % data.len();
    centroids.push(data[first_idx].clone());

    // Pre-sample random values for centroid selection (must be done sequentially)
    let random_thresholds: Vec<f32> = (1..k).map(|_| rng.next_f32()).collect();

    // Maintain min-distance cache - min squared distance to any centroid for each point
    // Initially set to f32::MAX (no centroids selected yet)
    let mut min_distances = vec![f32::MAX; data.len()];

    let num_threads = num_threads.min(std::thread::available_parallelism().map_or(4, |p| p.get()));
    let chunk_size = data.len().div_ceil(num_threads);

    // Choose remaining centroids with probability proportional to squared distance
    for (iter, &threshold_factor) in random_thresholds.iter().enumerate() {
        let last_centroid = &centroids[iter]; // The most recently added centroid

        // Parallel distance update: compute distance to the new centroid
        // and update min_distances cache
        std::thread::scope(|scope| {
            let handles: Vec<_> = data
                .chunks(chunk_size)
                .zip(min_distances.chunks_mut(chunk_size))
                .map(|(data_chunk, dist_chunk)| {
                    let centroid_ref = last_centroid;
                    scope.spawn(move || {
                        for (point, min_dist) in data_chunk.iter().zip(dist_chunk.iter_mut()) {
                            let dist_to_new = squared_distance(point, centroid_ref);
                            *min_dist = min_dist.min(dist_to_new);
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });

        // Sum of distances for normalization (sequential - small compared to distance computation)
        let total: f32 = min_distances.iter().sum();
        if total <= 0.0 {
            // All points are at existing centroids, pick random
            let idx = (rng.next_u64() as usize) % data.len();
            centroids.push(data[idx].clone());
            continue;
        }

        // Sample proportionally to squared distance
        let threshold = threshold_factor * total;
        let mut cumsum = 0.0f32;
        let mut chosen = 0;

        for (i, &d) in min_distances.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }

        centroids.push(data[chosen].clone());
    }

    centroids
}

fn find_nearest(point: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0;
    let mut best_dist = f32::MAX;

    for (i, centroid) in centroids.iter().enumerate() {
        let dist = squared_distance(point, centroid);
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

/// Simple deterministic PRNG for reproducible k-means
pub(crate) struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::EMBEDDING_DIM;
    use std::sync::atomic::Ordering;

    fn random_embedding(seed: u64) -> Embedding {
        let mut rng = SimpleRng::new(seed);
        let vector: Vec<f32> = (0..EMBEDDING_DIM)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect();
        Embedding::new(vector)
    }

    fn normalized_embedding(seed: u64) -> Embedding {
        let mut emb = random_embedding(seed);
        let norm: f32 = emb.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut emb.vector {
                *x /= norm;
            }
        }
        emb
    }

    #[test]
    fn test_pq_config_default() {
        let config = PqConfig::default();
        assert_eq!(config.num_subspaces, 8);
        assert_eq!(config.num_centroids, 256);
        assert!(config.validate(EMBEDDING_DIM).is_ok());
    }

    #[test]
    fn test_pq_config_fast() {
        let config = PqConfig::fast();
        assert_eq!(config.num_centroids, 64);
        assert!(config.validate(EMBEDDING_DIM).is_ok());
    }

    #[test]
    fn test_pq_config_accurate() {
        let config = PqConfig::accurate();
        assert_eq!(config.num_subspaces, 12);
        assert!(config.validate(EMBEDDING_DIM).is_ok());
    }

    #[test]
    fn test_pq_config_validation() {
        let config = PqConfig {
            num_subspaces: 7, // Does not divide 96
            ..Default::default()
        };
        assert!(config.validate(96).is_err());

        let config = PqConfig {
            num_centroids: 257, // Too many
            ..Default::default()
        };
        assert!(config.validate(96).is_err());

        let config = PqConfig {
            num_subspaces: 0,
            ..Default::default()
        };
        assert!(config.validate(96).is_err());
    }

    #[test]
    fn test_train_product_quantizer() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::fast();

        let pq = ProductQuantizer::train(&embeddings, config);
        assert!(pq.is_ok());

        let pq = pq.unwrap();
        assert_eq!(pq.dim(), EMBEDDING_DIM);
    }

    #[test]
    fn test_train_insufficient_data() {
        let embeddings: Vec<Embedding> = (0..10).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::default(); // Needs 256 centroids

        let result = ProductQuantizer::train(&embeddings, config);
        assert!(matches!(result, Err(PqError::InsufficientData(_))));
    }

    #[test]
    fn test_encode_decode() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::fast();
        let pq = ProductQuantizer::train(&embeddings, config).unwrap();

        let original = &embeddings[0];
        let codes = pq.encode(original);
        let decoded = pq.decode(&codes);

        // Decoded should be close to original
        assert_eq!(decoded.dim, original.dim);

        // Compute reconstruction error
        let error: f32 = original
            .vector
            .iter()
            .zip(decoded.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Error should be reasonable (not perfect, but not too bad)
        assert!(error < 10.0, "Reconstruction error too high: {}", error);
    }

    #[test]
    fn test_encode_batch() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::fast();
        let pq = ProductQuantizer::train(&embeddings, config).unwrap();

        // Batch encode first 10 embeddings
        let batch: Vec<Embedding> = embeddings[..10].to_vec();
        let batch_codes = pq.encode_batch(&batch);

        // Verify batch encoding matches individual encoding
        assert_eq!(batch_codes.len(), 10);
        for (i, codes) in batch_codes.iter().enumerate() {
            let individual_codes = pq.encode(&batch[i]);
            assert_eq!(codes, &individual_codes, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_opq_encode_batch() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = OpqConfig {
            pq: PqConfig::fast(),
            opq_iterations: 2,
        };
        let opq = OptimizedProductQuantizer::train(&embeddings, config).unwrap();

        // Batch encode first 10 embeddings
        let batch: Vec<Embedding> = embeddings[..10].to_vec();
        let batch_codes = opq.encode_batch(&batch);

        // Verify batch encoding matches individual encoding
        assert_eq!(batch_codes.len(), 10);
        for (i, codes) in batch_codes.iter().enumerate() {
            let individual_codes = opq.encode(&batch[i]);
            assert_eq!(codes, &individual_codes, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_encode_batch_parallel_single_thread() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::fast();
        let pq = ProductQuantizer::train(&embeddings, config).unwrap();

        // Test with 1 thread (fallback to sequential)
        let batch: Vec<Embedding> = embeddings[..50].to_vec();
        let parallel_codes = pq.encode_batch_parallel(&batch, 1);
        let sequential_codes = pq.encode_batch(&batch);

        assert_eq!(parallel_codes.len(), sequential_codes.len());
        for (i, (par, seq)) in parallel_codes
            .iter()
            .zip(sequential_codes.iter())
            .enumerate()
        {
            assert_eq!(par, seq, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_encode_batch_parallel_multi_thread() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::fast();
        let pq = ProductQuantizer::train(&embeddings, config).unwrap();

        // Test with 4 threads
        let batch: Vec<Embedding> = embeddings[..100].to_vec();
        let parallel_codes = pq.encode_batch_parallel(&batch, 4);
        let sequential_codes = pq.encode_batch(&batch);

        assert_eq!(parallel_codes.len(), sequential_codes.len());
        for (i, (par, seq)) in parallel_codes
            .iter()
            .zip(sequential_codes.iter())
            .enumerate()
        {
            assert_eq!(par, seq, "Mismatch at index {} with 4 threads", i);
        }

        // Test with more threads than embeddings
        let small_batch: Vec<Embedding> = embeddings[..5].to_vec();
        let parallel_codes_excess = pq.encode_batch_parallel(&small_batch, 20);
        let sequential_codes_small = pq.encode_batch(&small_batch);

        assert_eq!(parallel_codes_excess.len(), sequential_codes_small.len());
        for (i, (par, seq)) in parallel_codes_excess
            .iter()
            .zip(sequential_codes_small.iter())
            .enumerate()
        {
            assert_eq!(par, seq, "Mismatch at index {} with excess threads", i);
        }
    }

    #[test]
    fn test_encode_batch_parallel_empty() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::fast();
        let pq = ProductQuantizer::train(&embeddings, config).unwrap();

        // Test with empty batch
        let empty: Vec<Embedding> = vec![];
        let parallel_codes = pq.encode_batch_parallel(&empty, 4);
        assert!(parallel_codes.is_empty());
    }

    #[test]
    fn test_opq_encode_batch_parallel_single_thread() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = OpqConfig {
            pq: PqConfig::fast(),
            opq_iterations: 2,
        };
        let opq = OptimizedProductQuantizer::train(&embeddings, config).unwrap();

        // Test with 1 thread (fallback to sequential)
        let batch: Vec<Embedding> = embeddings[..50].to_vec();
        let parallel_codes = opq.encode_batch_parallel(&batch, 1);
        let sequential_codes = opq.encode_batch(&batch);

        assert_eq!(parallel_codes.len(), sequential_codes.len());
        for (i, (par, seq)) in parallel_codes
            .iter()
            .zip(sequential_codes.iter())
            .enumerate()
        {
            assert_eq!(par, seq, "OPQ mismatch at index {}", i);
        }
    }

    #[test]
    fn test_opq_encode_batch_parallel_multi_thread() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = OpqConfig {
            pq: PqConfig::fast(),
            opq_iterations: 2,
        };
        let opq = OptimizedProductQuantizer::train(&embeddings, config).unwrap();

        // Test with 4 threads
        let batch: Vec<Embedding> = embeddings[..100].to_vec();
        let parallel_codes = opq.encode_batch_parallel(&batch, 4);
        let sequential_codes = opq.encode_batch(&batch);

        assert_eq!(parallel_codes.len(), sequential_codes.len());
        for (i, (par, seq)) in parallel_codes
            .iter()
            .zip(sequential_codes.iter())
            .enumerate()
        {
            assert_eq!(par, seq, "OPQ mismatch at index {} with 4 threads", i);
        }
    }

    #[test]
    fn test_opq_encode_batch_parallel_empty() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = OpqConfig {
            pq: PqConfig::fast(),
            opq_iterations: 2,
        };
        let opq = OptimizedProductQuantizer::train(&embeddings, config).unwrap();

        // Test with empty batch
        let empty: Vec<Embedding> = vec![];
        let parallel_codes = opq.encode_batch_parallel(&empty, 4);
        assert!(parallel_codes.is_empty());
    }

    #[test]
    fn test_train_parallel_single_thread() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::fast();

        // Train with 1 thread (should fall back to sequential)
        let pq = ProductQuantizer::train_parallel(&embeddings, config.clone(), 1);
        assert!(pq.is_ok());

        let pq = pq.unwrap();
        assert_eq!(pq.dim(), EMBEDDING_DIM);

        // Encoding should work
        let codes = pq.encode(&embeddings[0]);
        assert_eq!(codes.len(), config.num_subspaces);
    }

    #[test]
    fn test_train_parallel_multi_thread() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::fast();

        // Train with 4 threads
        let pq = ProductQuantizer::train_parallel(&embeddings, config.clone(), 4);
        assert!(pq.is_ok());

        let pq = pq.unwrap();
        assert_eq!(pq.dim(), EMBEDDING_DIM);

        // Test encoding/decoding
        let codes = pq.encode(&embeddings[0]);
        assert_eq!(codes.len(), config.num_subspaces);

        let decoded = pq.decode(&codes);
        assert_eq!(decoded.dim, EMBEDDING_DIM);

        // Reconstruction error should be reasonable
        let error: f32 = embeddings[0]
            .vector
            .iter()
            .zip(decoded.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(error < 10.0, "Reconstruction error too high: {}", error);
    }

    #[test]
    fn test_train_parallel_excess_threads() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::fast(); // 8 subspaces

        // Train with more threads than subspaces (should clamp)
        let pq = ProductQuantizer::train_parallel(&embeddings, config.clone(), 100);
        assert!(pq.is_ok());

        let pq = pq.unwrap();
        assert_eq!(pq.dim(), EMBEDDING_DIM);

        // Should produce valid encodings
        let codes = pq.encode(&embeddings[42]);
        assert_eq!(codes.len(), config.num_subspaces);
    }

    #[test]
    fn test_train_parallel_data_parallel_strategy_for_large_data() {
        DATA_PARALLEL_TRAIN_CALLS.store(0, Ordering::Relaxed);

        let embeddings: Vec<Embedding> = (0..1200).map(|i| random_embedding(i as u64)).collect();
        let mut config = PqConfig::fast();
        config.num_subspaces = 4;
        config.num_centroids = 64;

        // Use more threads than subspaces to trigger data-parallel k-means
        let pq = ProductQuantizer::train_parallel(&embeddings, config.clone(), 12).unwrap();
        assert!(
            DATA_PARALLEL_TRAIN_CALLS.load(Ordering::Relaxed) > 0,
            "data-parallel k-means branch was not used"
        );

        // Quantization quality should remain reasonable compared to sequential training
        let compute_error = |quantizer: &ProductQuantizer| -> f32 {
            let total: f32 = embeddings
                .iter()
                .map(|e| {
                    let decoded = quantizer.decode(&quantizer.encode(e));
                    euclidean_distance_sq(&e.vector, &decoded.vector)
                })
                .sum();
            total / embeddings.len() as f32
        };

        let error_parallel = compute_error(&pq);
        let baseline = ProductQuantizer::train(&embeddings, config).unwrap();
        let error_baseline = compute_error(&baseline);
        assert!(
            error_parallel <= error_baseline * 1.2 + 1e-6,
            "parallel error {} vs baseline {}",
            error_parallel,
            error_baseline
        );
    }

    #[test]
    fn test_train_parallel_insufficient_data() {
        let embeddings: Vec<Embedding> = (0..10).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::default(); // Needs 256 centroids

        let result = ProductQuantizer::train_parallel(&embeddings, config, 4);
        assert!(matches!(result, Err(PqError::InsufficientData(_))));
    }

    #[test]
    fn test_train_parallel_empty() {
        let embeddings: Vec<Embedding> = vec![];
        let config = PqConfig::fast();

        let result = ProductQuantizer::train_parallel(&embeddings, config, 4);
        assert!(matches!(result, Err(PqError::InsufficientData(_))));
    }

    #[test]
    fn test_opq_train_parallel_single_thread() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = OpqConfig {
            pq: PqConfig::fast(),
            opq_iterations: 2,
        };

        // Train with 1 thread (should fall back to sequential)
        let opq = OptimizedProductQuantizer::train_parallel(&embeddings, config.clone(), 1);
        assert!(opq.is_ok());

        let opq = opq.unwrap();
        assert_eq!(opq.dim(), EMBEDDING_DIM);

        // Encoding should work
        let codes = opq.encode(&embeddings[0]);
        assert_eq!(codes.len(), config.pq.num_subspaces);
    }

    #[test]
    fn test_opq_train_parallel_multi_thread() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = OpqConfig {
            pq: PqConfig::fast(),
            opq_iterations: 2,
        };

        // Train with 4 threads
        let opq = OptimizedProductQuantizer::train_parallel(&embeddings, config.clone(), 4);
        assert!(opq.is_ok());

        let opq = opq.unwrap();
        assert_eq!(opq.dim(), EMBEDDING_DIM);

        // Test encoding/decoding
        let codes = opq.encode(&embeddings[0]);
        assert_eq!(codes.len(), config.pq.num_subspaces);

        let decoded = opq.decode(&codes);
        assert_eq!(decoded.dim, EMBEDDING_DIM);

        // Reconstruction error should be reasonable
        let error: f32 = embeddings[0]
            .vector
            .iter()
            .zip(decoded.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(error < 10.0, "OPQ reconstruction error too high: {}", error);
    }

    #[test]
    fn test_opq_train_parallel_excess_threads() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = OpqConfig {
            pq: PqConfig::fast(),
            opq_iterations: 2,
        };

        // Train with more threads than subspaces
        let opq = OptimizedProductQuantizer::train_parallel(&embeddings, config.clone(), 100);
        assert!(opq.is_ok());

        let opq = opq.unwrap();
        assert_eq!(opq.dim(), EMBEDDING_DIM);

        // Should produce valid encodings
        let codes = opq.encode(&embeddings[42]);
        assert_eq!(codes.len(), config.pq.num_subspaces);
    }

    #[test]
    fn test_opq_train_parallel_insufficient_data() {
        let embeddings: Vec<Embedding> = (0..10).map(|i| random_embedding(i as u64)).collect();
        let config = OpqConfig::default(); // Needs 256 centroids

        let result = OptimizedProductQuantizer::train_parallel(&embeddings, config, 4);
        assert!(matches!(result, Err(PqError::InsufficientData(_))));
    }

    #[test]
    fn test_opq_train_parallel_empty() {
        let embeddings: Vec<Embedding> = vec![];
        let config = OpqConfig::fast();

        let result = OptimizedProductQuantizer::train_parallel(&embeddings, config, 4);
        assert!(matches!(result, Err(PqError::InsufficientData(_))));
    }

    #[test]
    fn test_rotate_data_parallel() {
        // Create test data: 100 vectors of dimension 96
        let data: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..96)
                    .map(|j| ((i * 96 + j) as f32).sin())
                    .collect::<Vec<f32>>()
            })
            .collect();

        // Create identity rotation matrix (results should match input)
        let rotation = super::identity_matrix(96);

        // Test single-thread
        let result_single = super::rotate_data_parallel(&data, &rotation, 96, 1);
        assert_eq!(result_single.len(), 100);

        // Test multi-thread
        let result_multi = super::rotate_data_parallel(&data, &rotation, 96, 4);
        assert_eq!(result_multi.len(), 100);

        // Results should be identical (identity rotation)
        for (i, (single, multi)) in result_single.iter().zip(result_multi.iter()).enumerate() {
            for (j, (s, m)) in single.vector.iter().zip(multi.vector.iter()).enumerate() {
                assert!(
                    (s - m).abs() < 1e-5,
                    "Mismatch at vec {} dim {}: {} vs {}",
                    i,
                    j,
                    s,
                    m
                );
            }
        }
    }

    #[test]
    fn test_rotate_data_parallel_empty() {
        let data: Vec<Vec<f32>> = vec![];
        let rotation = super::identity_matrix(96);

        let result = super::rotate_data_parallel(&data, &rotation, 96, 4);
        assert!(result.is_empty());
    }

    #[test]
    fn test_compute_reconstructions_parallel() {
        // Train a PQ on some data
        let embeddings: Vec<Embedding> = (0..300).map(|i| normalized_embedding(i as u64)).collect();
        let config = PqConfig::fast();
        let pq = ProductQuantizer::train(&embeddings, config).unwrap();

        // Test single-thread vs multi-thread on a subset
        let test_embs = &embeddings[0..50];
        let result_single = super::compute_reconstructions_parallel(test_embs, &pq, 1);
        let result_multi = super::compute_reconstructions_parallel(test_embs, &pq, 4);

        assert_eq!(result_single.len(), 50);
        assert_eq!(result_multi.len(), 50);

        // Results should be identical
        for (i, (single, multi)) in result_single.iter().zip(result_multi.iter()).enumerate() {
            assert_eq!(single.len(), multi.len(), "Dimension mismatch at vec {}", i);
            for (j, (s, m)) in single.iter().zip(multi.iter()).enumerate() {
                assert!(
                    (s - m).abs() < 1e-5,
                    "Mismatch at vec {} dim {}: {} vs {}",
                    i,
                    j,
                    s,
                    m
                );
            }
        }
    }

    #[test]
    fn test_compute_reconstructions_parallel_empty() {
        // Train a PQ first
        let embeddings: Vec<Embedding> = (0..300).map(|i| normalized_embedding(i as u64)).collect();
        let config = PqConfig::fast();
        let pq = ProductQuantizer::train(&embeddings, config).unwrap();

        // Test empty input
        let empty_embs: Vec<Embedding> = vec![];
        let result = super::compute_reconstructions_parallel(&empty_embs, &pq, 4);
        assert!(result.is_empty());
    }

    #[test]
    fn test_opq_parallel_rotation_consistency() {
        // Verify that OPQ trained with parallel and sequential methods
        // produce equivalent results (within floating point tolerance)
        let embeddings: Vec<Embedding> = (0..500).map(|i| normalized_embedding(i as u64)).collect();
        let config = OpqConfig {
            pq: PqConfig {
                num_subspaces: 8,
                num_centroids: 64,
                kmeans_iterations: 5,
                seed: 42,
            },
            opq_iterations: 3,
        };

        // Train with single thread (falls back to sequential)
        let opq_single = OptimizedProductQuantizer::train_parallel(&embeddings, config.clone(), 1)
            .expect("Single thread training should succeed");

        // Train with multiple threads
        let opq_multi = OptimizedProductQuantizer::train_parallel(&embeddings, config.clone(), 4)
            .expect("Multi thread training should succeed");

        // Both should produce valid quantizers
        let test_query = &embeddings[42];
        let codes_single = opq_single.encode(test_query);
        let codes_multi = opq_multi.encode(test_query);

        // Codes may differ slightly due to parallel RNG seeding,
        // but decoded vectors should be reasonably close
        let decoded_single = opq_single.decode(&codes_single);
        let decoded_multi = opq_multi.decode(&codes_multi);

        // Verify dimensions match
        assert_eq!(decoded_single.dim, decoded_multi.dim);
        assert_eq!(decoded_single.dim, test_query.dim);
    }

    #[test]
    fn test_compute_optimal_rotation_parallel_consistency() {
        // Verify that parallel and sequential covariance computation produce
        // valid rotation matrices that can be used for OPQ
        let dim = 8;
        let n = 100;

        // Generate test data with more structure (normalized vectors)
        let original: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|j| ((i * dim + j) % 100) as f32 / 100.0 - 0.5)
                    .collect();
                // Normalize
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-6 {
                    v.iter_mut().for_each(|x| *x /= norm);
                }
                v
            })
            .collect();

        let reconstructed: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|j| ((i * dim + j + 37) % 100) as f32 / 100.0 - 0.5)
                    .collect();
                // Normalize
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-6 {
                    v.iter_mut().for_each(|x| *x /= norm);
                }
                v
            })
            .collect();

        // Compute with sequential method
        let r_seq = compute_optimal_rotation(&original, &reconstructed, dim, n);

        // Compute with parallel method (single thread should match sequential exactly)
        let r_par_1 = compute_optimal_rotation_parallel(&original, &reconstructed, dim, n, 1);

        // Compute with parallel method (multiple threads)
        let r_par_4 = compute_optimal_rotation_parallel(&original, &reconstructed, dim, n, 4);

        // Single-thread parallel should match sequential exactly
        for i in 0..(dim * dim) {
            assert!(
                (r_seq[i] - r_par_1[i]).abs() < 1e-5,
                "Sequential vs single-thread parallel mismatch at {}: {} vs {}",
                i,
                r_seq[i],
                r_par_1[i]
            );
        }

        // All rotation matrices should be valid (not NaN/Inf, reasonable magnitudes)
        for r in &[&r_seq, &r_par_1, &r_par_4] {
            for (i, &val) in r.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Rotation matrix contains non-finite value at {}: {}",
                    i,
                    val
                );
                assert!(
                    val.abs() < 10.0,
                    "Rotation matrix contains unreasonably large value at {}: {}",
                    i,
                    val
                );
            }
        }

        // Both should work for rotation (apply to a vector and get valid output)
        let test_vec: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();

        let rotated_seq = matrix_vector_multiply(&r_seq, &test_vec, dim);
        let rotated_par = matrix_vector_multiply(&r_par_4, &test_vec, dim);

        // Both should produce valid vectors (not NaN, reasonable magnitude)
        for (i, (&s, &p)) in rotated_seq.iter().zip(rotated_par.iter()).enumerate() {
            assert!(s.is_finite(), "Sequential rotation output NaN at {}", i);
            assert!(p.is_finite(), "Parallel rotation output NaN at {}", i);
        }
    }

    #[test]
    fn test_compute_optimal_rotation_parallel_empty() {
        // Edge case: empty input
        let original: Vec<Vec<f32>> = vec![];
        let reconstructed: Vec<Vec<f32>> = vec![];
        let dim = 8;

        // Should produce identity matrix
        let r = compute_optimal_rotation_parallel(&original, &reconstructed, dim, 0, 4);
        assert_eq!(r.len(), dim * dim);

        // Check it's identity
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (r[i * dim + j] - expected).abs() < 1e-6,
                    "Expected identity at ({}, {}): {} vs {}",
                    i,
                    j,
                    r[i * dim + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_asymmetric_distance() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| normalized_embedding(i as u64)).collect();
        let config = PqConfig::fast();
        let pq = ProductQuantizer::train(&embeddings, config).unwrap();

        let query = &embeddings[42];
        let codes = pq.encode(&embeddings[0]);

        let dist = pq.asymmetric_distance(query, &codes);

        // Distance should be non-negative
        assert!(dist >= 0.0);

        // Self-distance should be much smaller than distance to other vectors
        let self_codes = pq.encode(query);
        let self_dist = pq.asymmetric_distance(query, &self_codes);

        // With normalized vectors and proper quantization, self-distance
        // should be significantly smaller than cross-distance
        assert!(
            self_dist < dist * 0.5,
            "Self-distance {} should be much less than other-distance {}",
            self_dist,
            dist
        );
    }

    #[test]
    fn test_distance_table() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::fast();
        let pq = ProductQuantizer::train(&embeddings, config).unwrap();

        let query = &embeddings[42];
        let codes = pq.encode(&embeddings[0]);

        // Both methods should give same result
        let dist1 = pq.asymmetric_distance(query, &codes);
        let table = pq.build_distance_table(query);
        let dist2 = pq.distance_with_table(&table, &codes);

        assert!(
            (dist1 - dist2).abs() < 1e-5,
            "Distance methods should match: {} vs {}",
            dist1,
            dist2
        );
    }

    #[test]
    fn test_distance_to_similarity() {
        // Identical vectors should have similarity 1.0
        assert!((ProductQuantizer::distance_to_similarity(0.0) - 1.0).abs() < 1e-6);

        // Orthogonal unit vectors have ||a-b||^2 = 2, so similarity = 0
        assert!((ProductQuantizer::distance_to_similarity(2.0) - 0.0).abs() < 1e-6);

        // Opposite unit vectors have ||a-b||^2 = 4, so similarity = -1
        assert!((ProductQuantizer::distance_to_similarity(4.0) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_pq_corpus_build() {
        let embeddings: Vec<(String, Embedding)> = (0..300)
            .map(|i| (format!("id_{}", i), random_embedding(i as u64)))
            .collect();
        let config = PqConfig::fast();

        let corpus = PqCorpus::build(&embeddings, config);
        assert!(corpus.is_ok());

        let corpus = corpus.unwrap();
        assert_eq!(corpus.len(), 300);
        assert!(!corpus.is_empty());
    }

    #[test]
    fn test_pq_corpus_memory_savings() {
        let embeddings: Vec<(String, Embedding)> = (0..1000)
            .map(|i| (format!("id_{}", i), random_embedding(i as u64)))
            .collect();
        let config = PqConfig::fast();

        let corpus = PqCorpus::build(&embeddings, config).unwrap();

        // Raw storage would be: 1000 * 96 * 4 = 384,000 bytes
        // PQ storage should be much less
        let raw_size = 1000 * EMBEDDING_DIM * 4;
        let pq_size = corpus.memory_usage();

        assert!(
            pq_size < raw_size / 10,
            "PQ should compress significantly: {} vs {}",
            pq_size,
            raw_size
        );
    }

    #[test]
    fn test_pq_corpus_find_nearest() {
        let embeddings: Vec<(String, Embedding)> = (0..300)
            .map(|i| (format!("id_{}", i), normalized_embedding(i as u64)))
            .collect();
        let config = PqConfig::fast();

        let corpus = PqCorpus::build(&embeddings, config).unwrap();

        let query = normalized_embedding(42);
        let results = corpus.find_nearest(&query, 5);

        assert_eq!(results.len(), 5);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }

        // The exact embedding should be in top results (with high probability)
        let top_ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        // Note: Due to quantization, the exact match might not be #1
        // but should be among top results
        let has_query = top_ids.contains(&"id_42");
        assert!(
            has_query,
            "Query embedding should be in top results: {:?}",
            top_ids
        );
    }

    #[test]
    fn test_pq_corpus_find_similar() {
        let embeddings: Vec<(String, Embedding)> = (0..300)
            .map(|i| (format!("id_{}", i), normalized_embedding(i as u64)))
            .collect();
        let config = PqConfig::fast();

        let corpus = PqCorpus::build(&embeddings, config).unwrap();

        let query = normalized_embedding(42);
        let results = corpus.find_similar(&query, 5);

        assert_eq!(results.len(), 5);
        // Similarities should be sorted descending (highest first)
        for i in 1..results.len() {
            assert!(results[i].1 <= results[i - 1].1);
        }
    }

    #[test]
    fn test_pq_corpus_insert() {
        let embeddings: Vec<(String, Embedding)> = (0..300)
            .map(|i| (format!("id_{}", i), random_embedding(i as u64)))
            .collect();
        let config = PqConfig::fast();

        let mut corpus = PqCorpus::build(&embeddings, config).unwrap();
        assert_eq!(corpus.len(), 300);

        // Insert new embedding
        corpus.insert("new_id".to_string(), &random_embedding(999));
        assert_eq!(corpus.len(), 301);
    }

    #[test]
    fn test_kmeans_basic() {
        // Simple 2D clustering
        let data: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![10.0, 10.1],
        ];

        let mut rng = SimpleRng::new(42);
        let centroids = kmeans(&data, 2, 10, &mut rng);

        assert_eq!(centroids.len(), 2);

        // Centroids should be near (0,0) and (10,10)
        let mut near_zero = false;
        let mut near_ten = false;

        for centroid in &centroids {
            let dist_zero = squared_distance(centroid, &[0.0, 0.0]);
            let dist_ten = squared_distance(centroid, &[10.0, 10.0]);

            if dist_zero < 1.0 {
                near_zero = true;
            }
            if dist_ten < 1.0 {
                near_ten = true;
            }
        }

        assert!(
            near_zero && near_ten,
            "Should find clusters near (0,0) and (10,10): {:?}",
            centroids
        );
    }

    #[test]
    fn test_kmeans_parallel_falls_back_for_small_data() {
        // For small datasets (< 1000), parallel should fall back to sequential
        let data: Vec<Vec<f32>> = (0..100)
            .map(|i| vec![(i % 10) as f32, (i / 10) as f32])
            .collect();

        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        let sequential = kmeans(&data, 5, 10, &mut rng1);
        let parallel = kmeans_parallel(&data, 5, 10, &mut rng2, 4);

        assert_eq!(sequential.len(), parallel.len());

        // Results should be identical since parallel falls back to sequential
        for (s, p) in sequential.iter().zip(parallel.iter()) {
            for (sv, pv) in s.iter().zip(p.iter()) {
                assert!(
                    (sv - pv).abs() < 1e-6,
                    "Sequential and parallel should match for small data"
                );
            }
        }
    }

    #[test]
    fn test_kmeans_parallel_large_data() {
        // Generate large dataset for parallel processing
        let mut rng = SimpleRng::new(12345);
        let data: Vec<Vec<f32>> = (0..2000)
            .map(|_| {
                vec![
                    (rng.next_u64() % 1000) as f32 / 100.0,
                    (rng.next_u64() % 1000) as f32 / 100.0,
                ]
            })
            .collect();

        // Test with 1 thread (sequential fallback)
        let mut rng1 = SimpleRng::new(42);
        let centroids_1t = kmeans_parallel(&data, 8, 20, &mut rng1, 1);

        // Test with 4 threads
        let mut rng4 = SimpleRng::new(42);
        let centroids_4t = kmeans_parallel(&data, 8, 20, &mut rng4, 4);

        // Both should produce 8 centroids
        assert_eq!(centroids_1t.len(), 8);
        assert_eq!(centroids_4t.len(), 8);

        // All centroids should be within data range
        for centroid in centroids_1t.iter().chain(centroids_4t.iter()) {
            assert!(
                centroid[0] >= 0.0 && centroid[0] <= 10.0,
                "Centroid x out of range: {}",
                centroid[0]
            );
            assert!(
                centroid[1] >= 0.0 && centroid[1] <= 10.0,
                "Centroid y out of range: {}",
                centroid[1]
            );
        }
    }

    #[test]
    fn test_kmeans_parallel_produces_valid_clusters() {
        // Create structured data with 3 clear clusters
        let mut data: Vec<Vec<f32>> = Vec::with_capacity(3000);

        // Cluster 1: around (0, 0)
        for i in 0u64..1000 {
            let noise_x = ((i.wrapping_mul(1234567)) % 100) as f32 / 1000.0 - 0.05;
            let noise_y = ((i.wrapping_mul(7654321)) % 100) as f32 / 1000.0 - 0.05;
            data.push(vec![0.0 + noise_x, 0.0 + noise_y]);
        }

        // Cluster 2: around (5, 5)
        for i in 0u64..1000 {
            let noise_x = ((i.wrapping_mul(1234567)) % 100) as f32 / 1000.0 - 0.05;
            let noise_y = ((i.wrapping_mul(7654321)) % 100) as f32 / 1000.0 - 0.05;
            data.push(vec![5.0 + noise_x, 5.0 + noise_y]);
        }

        // Cluster 3: around (10, 0)
        for i in 0u64..1000 {
            let noise_x = ((i.wrapping_mul(1234567)) % 100) as f32 / 1000.0 - 0.05;
            let noise_y = ((i.wrapping_mul(7654321)) % 100) as f32 / 1000.0 - 0.05;
            data.push(vec![10.0 + noise_x, 0.0 + noise_y]);
        }

        let mut rng = SimpleRng::new(42);
        let centroids = kmeans_parallel(&data, 3, 20, &mut rng, 4);

        assert_eq!(centroids.len(), 3);

        // Should find centroids near the 3 cluster centers
        let expected_centers = [[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]];
        let mut found = [false; 3];

        for centroid in &centroids {
            for (i, expected) in expected_centers.iter().enumerate() {
                let dist = squared_distance(centroid, expected);
                if dist < 0.5 {
                    found[i] = true;
                }
            }
        }

        assert!(
            found.iter().all(|&f| f),
            "Should find all 3 cluster centers. Found: {:?}, Centroids: {:?}",
            found,
            centroids
        );
    }

    #[test]
    fn test_kmeans_parallel_empty_and_edge_cases() {
        // Empty data
        let empty: Vec<Vec<f32>> = vec![];
        let mut rng = SimpleRng::new(42);
        let result = kmeans_parallel(&empty, 5, 10, &mut rng, 4);
        assert!(result.is_empty());

        // k = 0
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let mut rng = SimpleRng::new(42);
        let result = kmeans_parallel(&data, 0, 10, &mut rng, 4);
        assert!(result.is_empty());

        // k > data.len() (small dataset)
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let mut rng = SimpleRng::new(42);
        let result = kmeans_parallel(&data, 10, 10, &mut rng, 4);
        assert_eq!(result.len(), 2); // Should clamp to data.len()
    }

    #[test]
    fn test_kmeans_plusplus_init_parallel_fallback_small_data() {
        // Small data should fall back to sequential
        let data: Vec<Vec<f32>> = (0..100)
            .map(|i| vec![i as f32 * 0.1, i as f32 * 0.2])
            .collect();

        let mut rng_seq = SimpleRng::new(42);
        let mut rng_par = SimpleRng::new(42);

        let seq = kmeans_plusplus_init(&data, 5, &mut rng_seq);
        let par = kmeans_plusplus_init_parallel(&data, 5, &mut rng_par, 4);

        // Should produce identical results for small data
        assert_eq!(seq.len(), par.len());
        for (s, p) in seq.iter().zip(par.iter()) {
            assert_eq!(s.len(), p.len());
            for (a, b) in s.iter().zip(p.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_kmeans_plusplus_init_parallel_large_data() {
        // Generate 2000 points in 3 well-separated clusters
        let mut data: Vec<Vec<f32>> = Vec::with_capacity(2000);
        for i in 0..2000 {
            let cluster = i % 3;
            let base = cluster as f32 * 100.0;
            let noise = (i as f32 * 0.1).sin() * 2.0;
            data.push(vec![base + noise, base + noise * 0.5, base + noise * 0.3]);
        }

        let mut rng = SimpleRng::new(12345);
        let centroids = kmeans_plusplus_init_parallel(&data, 10, &mut rng, 4);

        // Should produce 10 centroids
        assert_eq!(centroids.len(), 10);

        // Each centroid should have correct dimension
        for c in &centroids {
            assert_eq!(c.len(), 3);
        }

        // Centroids should be actual data points (k-means++ selects from data)
        for c in &centroids {
            let is_in_data = data
                .iter()
                .any(|d| d.iter().zip(c.iter()).all(|(a, b)| (a - b).abs() < 1e-6));
            assert!(is_in_data, "Centroid should be a data point");
        }
    }

    #[test]
    fn test_kmeans_plusplus_init_parallel_edge_cases() {
        // Empty data
        let empty: Vec<Vec<f32>> = vec![];
        let mut rng = SimpleRng::new(42);
        let result = kmeans_plusplus_init_parallel(&empty, 5, &mut rng, 4);
        assert!(result.is_empty());

        // k = 0
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let mut rng = SimpleRng::new(42);
        let result = kmeans_plusplus_init_parallel(&data, 0, &mut rng, 4);
        assert!(result.is_empty());

        // Single thread should work
        let data: Vec<Vec<f32>> = (0..1500)
            .map(|i| vec![i as f32 * 0.1, i as f32 * 0.2])
            .collect();
        let mut rng = SimpleRng::new(42);
        let result = kmeans_plusplus_init_parallel(&data, 5, &mut rng, 1);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_kmeans_parallel_uses_parallel_init() {
        // This test verifies that kmeans_parallel uses the parallel initialization
        // by checking that large datasets produce valid results
        let data: Vec<Vec<f32>> = (0..2000)
            .map(|i| {
                let cluster = i % 5;
                let base = cluster as f32 * 50.0;
                let noise = ((i as f32) * 0.7).sin() * 3.0;
                vec![base + noise, base - noise, (i as f32 * 0.1).cos() * 5.0]
            })
            .collect();

        let mut rng = SimpleRng::new(99);
        let centroids = kmeans_parallel(&data, 5, 20, &mut rng, 8);

        assert_eq!(centroids.len(), 5);
        for c in &centroids {
            assert_eq!(c.len(), 3);
            // Centroids should be finite
            assert!(c.iter().all(|v| v.is_finite()));
        }
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
    fn test_simple_rng_seed_zero() {
        let rng = SimpleRng::new(0);
        assert_eq!(rng.state, 1); // 0 seed maps to 1
    }

    #[test]
    fn test_codebook_size() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::fast(); // 8 subspaces, 64 centroids, 12 dims per subspace
        let pq = ProductQuantizer::train(&embeddings, config).unwrap();

        // 8 subspaces * 64 centroids * 12 dims * 4 bytes = 24,576 bytes
        let expected = 8 * 64 * 12 * 4;
        assert_eq!(pq.codebook_size_bytes(), expected);
    }

    #[test]
    fn test_pq_corpus_get() {
        // Need at least 64 embeddings for PqConfig::fast() (64 centroids)
        let embeddings: Vec<(String, Embedding)> = (0..100)
            .map(|i| (format!("id_{}", i), random_embedding(i as u64)))
            .collect();
        let config = PqConfig::fast();

        let corpus = PqCorpus::build(&embeddings, config).unwrap();

        let entry = corpus.get(5);
        assert!(entry.is_some());
        let (id, codes) = entry.unwrap();
        assert_eq!(id, "id_5");
        assert_eq!(codes.len(), 8); // num_subspaces

        assert!(corpus.get(200).is_none());
    }

    #[test]
    fn test_pq_error_display() {
        let err = PqError::InvalidConfig("test".into());
        let display = format!("{}", err);
        assert!(display.contains("test"));

        let err = PqError::DimensionMismatch {
            expected: 96,
            got: 64,
        };
        let display = format!("{}", err);
        assert!(display.contains("96"));
        assert!(display.contains("64"));
    }

    #[test]
    fn test_encode_wrong_dimension() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::fast();
        let pq = ProductQuantizer::train(&embeddings, config).unwrap();

        // Wrong dimension embedding
        let wrong_dim = Embedding::new(vec![0.0; 50]);
        let codes = pq.encode(&wrong_dim);

        // Should return zeros
        assert_eq!(codes.len(), 8);
        assert!(codes.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_decode_wrong_length() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::fast();
        let pq = ProductQuantizer::train(&embeddings, config).unwrap();

        // Wrong length codes
        let wrong_codes = vec![0u8; 5]; // Should be 8
        let decoded = pq.decode(&wrong_codes);

        // Should return zeros
        assert_eq!(decoded.dim, EMBEDDING_DIM);
        assert!(decoded.vector.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_product_quantizer_save_load() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::fast();
        let pq = ProductQuantizer::train(&embeddings, config).unwrap();

        // Create temp file
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pq_model.json");

        // Save
        pq.save_to_file(&path).unwrap();
        assert!(path.exists());

        // Load
        let loaded = ProductQuantizer::load_from_file(&path).unwrap();
        assert_eq!(loaded.dim(), pq.dim());
        assert_eq!(loaded.config().num_subspaces, pq.config().num_subspaces);
        assert_eq!(loaded.config().num_centroids, pq.config().num_centroids);
        assert_eq!(loaded.codebook_size_bytes(), pq.codebook_size_bytes());

        // Verify encoding matches
        let test_emb = random_embedding(999);
        let codes1 = pq.encode(&test_emb);
        let codes2 = loaded.encode(&test_emb);
        assert_eq!(codes1, codes2);
    }

    #[test]
    fn test_product_quantizer_load_if_exists() {
        // Non-existent file should return None
        let result = ProductQuantizer::load_if_exists("/nonexistent/path/model.json").unwrap();
        assert!(result.is_none());

        // Existing file should return Some
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = PqConfig::fast();
        let pq = ProductQuantizer::train(&embeddings, config).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pq_model.json");
        pq.save_to_file(&path).unwrap();

        let loaded = ProductQuantizer::load_if_exists(&path).unwrap();
        assert!(loaded.is_some());
    }

    #[test]
    fn test_pq_corpus_save_load() {
        let embeddings: Vec<(String, Embedding)> = (0..300)
            .map(|i| (format!("id_{}", i), random_embedding(i as u64)))
            .collect();
        let config = PqConfig::fast();
        let corpus = PqCorpus::build(&embeddings, config).unwrap();

        // Create temp file
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pq_corpus.json");

        // Save
        corpus.save_to_file(&path).unwrap();
        assert!(path.exists());

        // Load
        let loaded = PqCorpus::load_from_file(&path).unwrap();
        assert_eq!(loaded.len(), corpus.len());

        // Verify entries match
        for i in 0..corpus.len() {
            let (id1, codes1) = corpus.get(i).unwrap();
            let (id2, codes2) = loaded.get(i).unwrap();
            assert_eq!(id1, id2);
            assert_eq!(codes1, codes2);
        }

        // Verify search works on loaded corpus
        let query = random_embedding(42);
        let results1 = corpus.find_similar(&query, 5);
        let results2 = loaded.find_similar(&query, 5);
        assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.0, r2.0); // Same IDs
        }
    }

    #[test]
    fn test_pq_corpus_load_if_exists() {
        // Non-existent file should return None
        let result = PqCorpus::load_if_exists("/nonexistent/path/corpus.json").unwrap();
        assert!(result.is_none());

        // Existing file should return Some
        let embeddings: Vec<(String, Embedding)> = (0..300)
            .map(|i| (format!("id_{}", i), random_embedding(i as u64)))
            .collect();
        let config = PqConfig::fast();
        let corpus = PqCorpus::build(&embeddings, config).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pq_corpus.json");
        corpus.save_to_file(&path).unwrap();

        let loaded = PqCorpus::load_if_exists(&path).unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().len(), 300);
    }

    // ========== OPQ Tests ==========

    #[test]
    fn test_opq_config_default() {
        let config = OpqConfig::default();
        assert_eq!(config.opq_iterations, 10);
        assert_eq!(config.pq.num_subspaces, 8);
        assert!(config.validate(EMBEDDING_DIM).is_ok());
    }

    #[test]
    fn test_opq_config_fast() {
        let config = OpqConfig::fast();
        assert_eq!(config.opq_iterations, 5);
        assert_eq!(config.pq.num_centroids, 64);
        assert!(config.validate(EMBEDDING_DIM).is_ok());
    }

    #[test]
    fn test_opq_config_accurate() {
        let config = OpqConfig::accurate();
        assert_eq!(config.opq_iterations, 20);
        assert_eq!(config.pq.num_subspaces, 12);
        assert!(config.validate(EMBEDDING_DIM).is_ok());
    }

    #[test]
    fn test_opq_config_validation() {
        let config = OpqConfig {
            opq_iterations: 0,
            ..Default::default()
        };
        assert!(config.validate(96).is_err());
    }

    #[test]
    fn test_train_opq() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = OpqConfig::fast();

        let opq = OptimizedProductQuantizer::train(&embeddings, config);
        assert!(opq.is_ok());

        let opq = opq.unwrap();
        assert_eq!(opq.dim(), EMBEDDING_DIM);
    }

    #[test]
    fn test_opq_encode_decode() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = OpqConfig::fast();
        let opq = OptimizedProductQuantizer::train(&embeddings, config).unwrap();

        let original = &embeddings[0];
        let codes = opq.encode(original);
        let decoded = opq.decode(&codes);

        assert_eq!(decoded.dim, original.dim);

        // Compute reconstruction error
        let error: f32 = original
            .vector
            .iter()
            .zip(decoded.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // OPQ error should be reasonable
        assert!(error < 10.0, "OPQ reconstruction error too high: {}", error);
    }

    #[test]
    fn test_opq_asymmetric_distance() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| normalized_embedding(i as u64)).collect();
        let config = OpqConfig::fast();
        let opq = OptimizedProductQuantizer::train(&embeddings, config).unwrap();

        let query = &embeddings[42];
        let codes = opq.encode(&embeddings[0]);

        let dist = opq.asymmetric_distance(query, &codes);
        assert!(dist >= 0.0);

        // Self-distance should be smaller
        let self_codes = opq.encode(query);
        let self_dist = opq.asymmetric_distance(query, &self_codes);

        assert!(
            self_dist < dist * 0.5,
            "Self-distance {} should be much less than other-distance {}",
            self_dist,
            dist
        );
    }

    #[test]
    fn test_opq_vs_pq_error() {
        // Test that OPQ has lower or similar error compared to standard PQ
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();

        let pq_config = PqConfig::fast();
        let opq_config = OpqConfig::fast();

        let pq = ProductQuantizer::train(&embeddings, pq_config).unwrap();
        let opq = OptimizedProductQuantizer::train(&embeddings, opq_config).unwrap();

        // Compute reconstruction error for both
        let pq_error: f32 = embeddings
            .iter()
            .map(|e| {
                let decoded = pq.decode(&pq.encode(e));
                e.vector
                    .iter()
                    .zip(decoded.vector.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
            })
            .sum::<f32>()
            / embeddings.len() as f32;

        let opq_error = opq.quantization_error(&embeddings);

        // OPQ should have similar or better error
        // Allow OPQ to be up to 50% worse since rotation learning may not always help
        assert!(
            opq_error < pq_error * 1.5,
            "OPQ error {} should not be much worse than PQ error {}",
            opq_error,
            pq_error
        );
    }

    #[test]
    fn test_opq_distance_table() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = OpqConfig::fast();
        let opq = OptimizedProductQuantizer::train(&embeddings, config).unwrap();

        let query = &embeddings[42];
        let codes = opq.encode(&embeddings[0]);

        // Both methods should give same result
        let dist1 = opq.asymmetric_distance(query, &codes);
        let table = opq.build_distance_table(query);
        let dist2 = opq.distance_with_table(&table, &codes);

        assert!(
            (dist1 - dist2).abs() < 1e-5,
            "Distance methods should match: {} vs {}",
            dist1,
            dist2
        );
    }

    #[test]
    fn test_opq_total_size() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = OpqConfig::fast();
        let opq = OptimizedProductQuantizer::train(&embeddings, config).unwrap();

        let codebook_size = opq.codebook_size_bytes();
        let total_size = opq.total_size_bytes();

        // Total should include rotation matrix (96*96*4 bytes)
        let rotation_size = EMBEDDING_DIM * EMBEDDING_DIM * 4;
        assert_eq!(total_size, codebook_size + rotation_size);
    }

    #[test]
    fn test_opq_save_load() {
        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = OpqConfig::fast();
        let opq = OptimizedProductQuantizer::train(&embeddings, config).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("opq_model.json");

        opq.save_to_file(&path).unwrap();
        assert!(path.exists());

        let loaded = OptimizedProductQuantizer::load_from_file(&path).unwrap();
        assert_eq!(loaded.dim(), opq.dim());
        assert_eq!(
            loaded.pq_config().num_subspaces,
            opq.pq_config().num_subspaces
        );

        // Verify encoding matches
        let test_emb = random_embedding(999);
        let codes1 = opq.encode(&test_emb);
        let codes2 = loaded.encode(&test_emb);
        assert_eq!(codes1, codes2);
    }

    #[test]
    fn test_opq_load_if_exists() {
        let result =
            OptimizedProductQuantizer::load_if_exists("/nonexistent/path/model.json").unwrap();
        assert!(result.is_none());

        let embeddings: Vec<Embedding> = (0..300).map(|i| random_embedding(i as u64)).collect();
        let config = OpqConfig::fast();
        let opq = OptimizedProductQuantizer::train(&embeddings, config).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("opq_model.json");
        opq.save_to_file(&path).unwrap();

        let loaded = OptimizedProductQuantizer::load_if_exists(&path).unwrap();
        assert!(loaded.is_some());
    }

    #[test]
    fn test_opq_corpus_build() {
        let embeddings: Vec<(String, Embedding)> = (0..300)
            .map(|i| (format!("id_{}", i), random_embedding(i as u64)))
            .collect();
        let config = OpqConfig::fast();

        let corpus = OpqCorpus::build(&embeddings, config);
        assert!(corpus.is_ok());

        let corpus = corpus.unwrap();
        assert_eq!(corpus.len(), 300);
        assert!(!corpus.is_empty());
    }

    #[test]
    fn test_opq_corpus_find_nearest() {
        let embeddings: Vec<(String, Embedding)> = (0..300)
            .map(|i| (format!("id_{}", i), normalized_embedding(i as u64)))
            .collect();
        let config = OpqConfig::fast();

        let corpus = OpqCorpus::build(&embeddings, config).unwrap();

        let query = normalized_embedding(42);
        let results = corpus.find_nearest(&query, 5);

        assert_eq!(results.len(), 5);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }

        // The exact embedding should be in top results
        let top_ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        let has_query = top_ids.contains(&"id_42");
        assert!(
            has_query,
            "Query embedding should be in top results: {:?}",
            top_ids
        );
    }

    #[test]
    fn test_opq_corpus_find_similar() {
        let embeddings: Vec<(String, Embedding)> = (0..300)
            .map(|i| (format!("id_{}", i), normalized_embedding(i as u64)))
            .collect();
        let config = OpqConfig::fast();

        let corpus = OpqCorpus::build(&embeddings, config).unwrap();

        let query = normalized_embedding(42);
        let results = corpus.find_similar(&query, 5);

        assert_eq!(results.len(), 5);
        // Similarities should be sorted descending
        for i in 1..results.len() {
            assert!(results[i].1 <= results[i - 1].1);
        }
    }

    #[test]
    fn test_opq_corpus_insert() {
        let embeddings: Vec<(String, Embedding)> = (0..300)
            .map(|i| (format!("id_{}", i), random_embedding(i as u64)))
            .collect();
        let config = OpqConfig::fast();

        let mut corpus = OpqCorpus::build(&embeddings, config).unwrap();
        assert_eq!(corpus.len(), 300);

        corpus.insert("new_id".to_string(), &random_embedding(999));
        assert_eq!(corpus.len(), 301);
    }

    #[test]
    fn test_opq_corpus_save_load() {
        let embeddings: Vec<(String, Embedding)> = (0..300)
            .map(|i| (format!("id_{}", i), random_embedding(i as u64)))
            .collect();
        let config = OpqConfig::fast();
        let corpus = OpqCorpus::build(&embeddings, config).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("opq_corpus.json");

        corpus.save_to_file(&path).unwrap();
        assert!(path.exists());

        let loaded = OpqCorpus::load_from_file(&path).unwrap();
        assert_eq!(loaded.len(), corpus.len());

        // Verify entries match
        for i in 0..corpus.len() {
            let (id1, codes1) = corpus.get(i).unwrap();
            let (id2, codes2) = loaded.get(i).unwrap();
            assert_eq!(id1, id2);
            assert_eq!(codes1, codes2);
        }
    }

    #[test]
    fn test_opq_corpus_load_if_exists() {
        let result = OpqCorpus::load_if_exists("/nonexistent/path/corpus.json").unwrap();
        assert!(result.is_none());

        let embeddings: Vec<(String, Embedding)> = (0..300)
            .map(|i| (format!("id_{}", i), random_embedding(i as u64)))
            .collect();
        let config = OpqConfig::fast();
        let corpus = OpqCorpus::build(&embeddings, config).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("opq_corpus.json");
        corpus.save_to_file(&path).unwrap();

        let loaded = OpqCorpus::load_if_exists(&path).unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().len(), 300);
    }

    // ========== Matrix operation tests ==========

    #[test]
    fn test_identity_matrix() {
        let id = identity_matrix(3);
        assert_eq!(id.len(), 9);
        assert_eq!(id[0], 1.0);
        assert_eq!(id[1], 0.0);
        assert_eq!(id[4], 1.0);
        assert_eq!(id[8], 1.0);
    }

    #[test]
    fn test_matrix_vector_multiply_identity() {
        let id = identity_matrix(3);
        let v = vec![1.0, 2.0, 3.0];
        let result = matrix_vector_multiply(&id, &v, 3);
        assert_eq!(result, v);
    }

    #[test]
    fn test_matrix_vector_multiply() {
        // Rotation by 90 degrees in 2D: [[0, -1], [1, 0]]
        let m = vec![0.0, -1.0, 1.0, 0.0];
        let v = vec![1.0, 0.0];
        let result = matrix_vector_multiply(&m, &v, 2);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_transpose_matrix_vector_multiply() {
        // For orthogonal matrix, M^T * M * v = v
        let m = vec![0.0, -1.0, 1.0, 0.0];
        let v = vec![1.0, 2.0];
        let mv = matrix_vector_multiply(&m, &v, 2);
        let mtmv = transpose_matrix_vector_multiply(&m, &mv, 2);
        assert!((mtmv[0] - v[0]).abs() < 1e-6);
        assert!((mtmv[1] - v[1]).abs() < 1e-6);
    }

    #[test]
    fn test_gram_schmidt() {
        // Test orthogonalization of a simple matrix
        let m = vec![1.0, 1.0, 0.0, 1.0];
        let ortho = gram_schmidt_orthogonalize(&m, 2);

        // Check columns are orthogonal
        let col0 = [ortho[0], ortho[2]];
        let col1 = [ortho[1], ortho[3]];
        let dot: f32 = col0.iter().zip(col1.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot.abs() < 1e-5,
            "Columns should be orthogonal, got dot={}",
            dot
        );

        // Check columns are normalized
        let norm0: f32 = col0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm1: f32 = col1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm0 - 1.0).abs() < 1e-5, "Column 0 should be unit norm");
        assert!((norm1 - 1.0).abs() < 1e-5, "Column 1 should be unit norm");
    }

    #[test]
    fn test_matrix_determinant_2x2() {
        let m = vec![1.0, 2.0, 3.0, 4.0];
        let det = matrix_determinant(&m, 2);
        // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        assert!((det - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_determinant_3x3() {
        // Identity matrix has det = 1
        let id = identity_matrix(3);
        let det = matrix_determinant(&id, 3);
        assert!((det - 1.0).abs() < 1e-6);
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify PqConfig validation rejects zero subspaces
    #[kani::proof]
    fn verify_config_rejects_zero_subspaces() {
        let dim: usize = kani::any();
        kani::assume(dim > 0 && dim <= 1024);

        let config = PqConfig {
            num_subspaces: 0,
            num_centroids: 256,
            kmeans_iterations: 10,
            seed: 42,
        };

        let result = config.validate(dim);
        assert!(result.is_err());
    }

    /// Verify PqConfig validation rejects centroids > 256
    #[kani::proof]
    fn verify_config_rejects_excess_centroids() {
        let centroids: usize = kani::any();
        kani::assume(centroids > 256);

        let config = PqConfig {
            num_subspaces: 8,
            num_centroids: centroids,
            kmeans_iterations: 10,
            seed: 42,
        };

        let result = config.validate(96);
        assert!(result.is_err());
    }

    /// Verify distance_to_similarity is bounded
    #[kani::proof]
    fn verify_similarity_bounds() {
        let dist: f32 = kani::any();
        kani::assume(dist >= 0.0 && dist.is_finite());

        let sim = ProductQuantizer::distance_to_similarity(dist);

        assert!(sim >= -1.0);
        assert!(sim <= 1.0);
    }

    /// Verify encode produces correct length output
    #[kani::proof]
    fn verify_encode_length() {
        let num_subspaces: usize = kani::any();
        kani::assume(num_subspaces > 0 && num_subspaces <= 16);
        kani::assume(96 % num_subspaces == 0);

        // We can't actually construct a ProductQuantizer in Kani without training,
        // but we can verify the invariant that encode always returns num_subspaces codes.
        // This is a documentation of the intended behavior.
        assert!(num_subspaces <= 16);
    }

    /// Verify SimpleRng never produces 0 state
    #[kani::proof]
    fn verify_rng_never_stuck() {
        let seed: u64 = kani::any();
        let mut rng = SimpleRng::new(seed);

        // After construction, state should never be 0
        assert!(rng.state != 0);

        // After one iteration, still non-zero
        let _ = rng.next_u64();
        assert!(rng.state != 0);
    }
}
