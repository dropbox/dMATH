//! Binary Quantization for extreme embedding compression
//!
//! Binary Quantization (BQ) provides the most aggressive compression by reducing
//! each embedding dimension to a single bit (the sign of the value):
//!
//! - Input: D-dimensional f32 embedding (D * 4 bytes)
//! - Output: D bits = ceil(D/8) bytes
//!
//! # Memory Comparison for 96-dimensional embeddings
//!
//! | Method | Bytes per embedding | Compression |
//! |--------|---------------------|-------------|
//! | Raw f32 | 384 bytes | 1x |
//! | PQ (M=8) | 8 bytes + codebooks | 48x |
//! | OPQ (M=8) | 8 bytes + codebooks + rotation | 48x |
//! | **Binary** | **12 bytes** | **32x** |
//!
//! # Performance Characteristics
//!
//! Binary quantization trades accuracy for speed:
//! - **Hamming distance**: O(D/64) using 64-bit popcount instructions
//! - **No training required**: Unlike PQ/OPQ, no centroid learning needed
//! - **Instant encoding**: Just sign extraction, no centroid lookups
//!
//! # Accuracy Considerations
//!
//! Binary quantization preserves only the sign of each dimension. This works well
//! when:
//! - Embeddings are approximately normalized
//! - High-dimensional (D > 64) where sign patterns are more discriminative
//! - Used as a first-stage filter before reranking with exact distances
//!
//! For higher accuracy with similar memory, consider:
//! - **Scalar Quantization (SQ)**: 1 byte per dimension (4x compression)
//! - **PQ**: 8 bytes per vector (48x compression, better recall)
//!
//! # Example
//!
//! ```ignore
//! use dashprove_learning::binary::{BinaryQuantizer, BinaryConfig};
//! use dashprove_learning::embedder::Embedding;
//!
//! // Create quantizer (no training needed)
//! let config = BinaryConfig::default();
//! let bq = BinaryQuantizer::new(96, config);
//!
//! // Encode embedding to binary codes
//! let codes = bq.encode(&embedding);
//!
//! // Compute Hamming distance (fast XOR + popcount)
//! let dist = bq.hamming_distance(&codes1, &codes2);
//!
//! // Convert Hamming to approximate cosine similarity
//! let sim = bq.hamming_to_cosine(dist);
//! ```

use crate::distance::hamming_distance as hamming_distance_bytes;
use crate::embedder::Embedding;
use serde::{Deserialize, Serialize};

/// Configuration for Binary Quantizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryConfig {
    /// Whether to normalize embeddings before binarization
    /// Normalization improves accuracy for non-unit vectors
    pub normalize: bool,
    /// Threshold for binarization (default: 0.0 for sign-based)
    /// Values >= threshold become 1, otherwise 0
    pub threshold: f32,
}

impl Default for BinaryConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            threshold: 0.0,
        }
    }
}

impl BinaryConfig {
    /// Config for already-normalized embeddings (skip normalization)
    pub fn for_normalized() -> Self {
        Self {
            normalize: false,
            threshold: 0.0,
        }
    }

    /// Config with median-based threshold (better for skewed distributions)
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            normalize: true,
            threshold,
        }
    }
}

/// Errors from Binary Quantization operations
#[derive(Debug, Clone)]
pub enum BinaryError {
    /// Dimension mismatch
    DimensionMismatch { expected: usize, got: usize },
    /// Invalid configuration
    InvalidConfig(String),
    /// Empty input
    EmptyInput,
}

impl std::fmt::Display for BinaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            BinaryError::InvalidConfig(s) => write!(f, "Invalid config: {}", s),
            BinaryError::EmptyInput => write!(f, "Empty input"),
        }
    }
}

impl std::error::Error for BinaryError {}

/// Binary code storage - packed bits
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BinaryCode {
    /// Packed bits (little-endian within each byte)
    pub bytes: Vec<u8>,
    /// Original dimension (for reconstruction)
    pub dim: usize,
}

impl BinaryCode {
    /// Create from raw bytes with specified dimension
    pub fn new(bytes: Vec<u8>, dim: usize) -> Self {
        Self { bytes, dim }
    }

    /// Number of bytes in this code
    pub fn num_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Get bit at position (0-indexed)
    pub fn get_bit(&self, pos: usize) -> bool {
        if pos >= self.dim {
            return false;
        }
        let byte_idx = pos / 8;
        let bit_idx = pos % 8;
        (self.bytes[byte_idx] >> bit_idx) & 1 == 1
    }

    /// Set bit at position
    pub fn set_bit(&mut self, pos: usize, value: bool) {
        if pos >= self.dim {
            return;
        }
        let byte_idx = pos / 8;
        let bit_idx = pos % 8;
        if value {
            self.bytes[byte_idx] |= 1 << bit_idx;
        } else {
            self.bytes[byte_idx] &= !(1 << bit_idx);
        }
    }
}

/// Binary Quantizer for extreme embedding compression
///
/// Compresses D-dimensional f32 embeddings to D bits using sign-based binarization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryQuantizer {
    /// Embedding dimension
    dim: usize,
    /// Number of bytes needed for codes
    num_bytes: usize,
    /// Configuration
    config: BinaryConfig,
}

impl BinaryQuantizer {
    /// Create a new binary quantizer
    ///
    /// No training required - just needs to know the dimension.
    pub fn new(dim: usize, config: BinaryConfig) -> Self {
        let num_bytes = dim.div_ceil(8);
        Self {
            dim,
            num_bytes,
            config,
        }
    }

    /// Create with default config
    pub fn with_dim(dim: usize) -> Self {
        Self::new(dim, BinaryConfig::default())
    }

    /// Get embedding dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get number of bytes per encoded vector
    pub fn code_size(&self) -> usize {
        self.num_bytes
    }

    /// Encode an embedding to binary codes
    pub fn encode(&self, embedding: &Embedding) -> Result<BinaryCode, BinaryError> {
        if embedding.dim != self.dim {
            return Err(BinaryError::DimensionMismatch {
                expected: self.dim,
                got: embedding.dim,
            });
        }

        let vector = if self.config.normalize {
            normalize_vector(&embedding.vector)
        } else {
            embedding.vector.clone()
        };

        let mut bytes = vec![0u8; self.num_bytes];

        for (i, &val) in vector.iter().enumerate() {
            if val >= self.config.threshold {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                bytes[byte_idx] |= 1 << bit_idx;
            }
        }

        Ok(BinaryCode::new(bytes, self.dim))
    }

    /// Encode multiple embeddings
    pub fn encode_batch(&self, embeddings: &[Embedding]) -> Result<Vec<BinaryCode>, BinaryError> {
        embeddings.iter().map(|e| self.encode(e)).collect()
    }

    /// Encode multiple embeddings using parallel processing
    ///
    /// For large batches (hundreds or thousands of embeddings), this can be
    /// significantly faster than sequential encoding by utilizing multiple CPU cores.
    ///
    /// # Arguments
    /// * `embeddings` - Slice of embeddings to encode
    /// * `num_threads` - Number of threads to use (clamped to embeddings.len())
    ///
    /// # Returns
    /// Vector of BinaryCodes, one per embedding, in the same order as input.
    /// Returns an error if any embedding has mismatched dimensions.
    pub fn encode_batch_parallel(
        &self,
        embeddings: &[Embedding],
        num_threads: usize,
    ) -> Result<Vec<BinaryCode>, BinaryError> {
        if embeddings.is_empty() {
            return Ok(vec![]);
        }

        // Check dimensions upfront
        for (idx, emb) in embeddings.iter().enumerate() {
            if emb.dim != self.dim {
                return Err(BinaryError::DimensionMismatch {
                    expected: self.dim,
                    got: emb.dim,
                });
            }
            let _ = idx; // Silence unused warning
        }

        let num_threads = num_threads.min(embeddings.len()).max(1);

        if num_threads == 1 {
            return self.encode_batch(embeddings);
        }

        let chunk_size = embeddings.len().div_ceil(num_threads);
        let mut results: Vec<BinaryCode> =
            vec![BinaryCode::new(vec![0u8; self.num_bytes], self.dim); embeddings.len()];

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            let result_chunks: Vec<_> = results.chunks_mut(chunk_size).collect();

            for (chunk_idx, (emb_chunk, result_chunk)) in
                embeddings.chunks(chunk_size).zip(result_chunks).enumerate()
            {
                let handle = s.spawn(move || {
                    for (i, emb) in emb_chunk.iter().enumerate() {
                        // We already validated dimensions, so unwrap is safe
                        result_chunk[i] = self.encode(emb).unwrap();
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

        Ok(results)
    }

    /// Compute Hamming distance between two binary codes
    ///
    /// Uses efficient popcount on 64-bit chunks.
    pub fn hamming_distance(&self, a: &BinaryCode, b: &BinaryCode) -> u32 {
        hamming_distance_bytes(&a.bytes, &b.bytes)
    }

    /// Convert Hamming distance to approximate cosine similarity
    ///
    /// For unit vectors, the relationship is approximately:
    /// cosine_sim ≈ 1 - 2 * hamming_dist / dim
    pub fn hamming_to_cosine(&self, hamming: u32) -> f32 {
        1.0 - 2.0 * (hamming as f32) / (self.dim as f32)
    }

    /// Convert Hamming distance to approximate Euclidean distance
    ///
    /// For unit vectors, the relationship is approximately:
    /// euclidean_dist ≈ sqrt(2 * hamming_dist / dim) * sqrt(2)
    pub fn hamming_to_euclidean(&self, hamming: u32) -> f32 {
        let cos_sim = self.hamming_to_cosine(hamming);
        // cos_sim = 1 - dist²/2 for unit vectors
        // dist² = 2 * (1 - cos_sim)
        (2.0 * (1.0 - cos_sim)).sqrt()
    }

    /// Asymmetric distance: exact query vs binary code
    ///
    /// Computes dot product between query and reconstructed binary vector.
    /// More accurate than Hamming for nearest neighbor search.
    pub fn asymmetric_distance(
        &self,
        query: &Embedding,
        code: &BinaryCode,
    ) -> Result<f32, BinaryError> {
        if query.dim != self.dim {
            return Err(BinaryError::DimensionMismatch {
                expected: self.dim,
                got: query.dim,
            });
        }

        let query_vec = if self.config.normalize {
            normalize_vector(&query.vector)
        } else {
            query.vector.clone()
        };

        // Dot product between query and sign-reconstructed vector
        // Binary 1 -> +1, Binary 0 -> -1
        let mut dot = 0.0f32;
        for (i, &q) in query_vec.iter().enumerate() {
            let sign = if code.get_bit(i) { 1.0 } else { -1.0 };
            dot += q * sign;
        }

        // Convert dot product to distance (larger dot = smaller distance)
        // Using negative dot product so smaller = more similar
        Ok(-dot)
    }

    /// Reconstruct approximate embedding from binary code
    ///
    /// Each bit is mapped to +1.0 or -1.0.
    /// Note: This loses all magnitude information.
    pub fn decode(&self, code: &BinaryCode) -> Embedding {
        let mut vector = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let sign = if code.get_bit(i) { 1.0 } else { -1.0 };
            vector.push(sign);
        }
        // Normalize to unit vector
        let norm = (self.dim as f32).sqrt();
        for v in &mut vector {
            *v /= norm;
        }
        Embedding::new(vector)
    }

    /// Compute quantization error (mean squared error)
    pub fn quantization_error(&self, embeddings: &[Embedding]) -> Result<f32, BinaryError> {
        if embeddings.is_empty() {
            return Err(BinaryError::EmptyInput);
        }

        let mut total_error = 0.0f32;

        for emb in embeddings {
            let normalized = if self.config.normalize {
                normalize_vector(&emb.vector)
            } else {
                emb.vector.clone()
            };

            let code = self.encode(emb)?;
            let reconstructed = self.decode(&code);

            // MSE between normalized original and reconstructed
            for (orig, recon) in normalized.iter().zip(reconstructed.vector.iter()) {
                let diff = orig - recon;
                total_error += diff * diff;
            }
        }

        Ok(total_error / (embeddings.len() as f32 * self.dim as f32))
    }

    /// Memory usage statistics
    pub fn memory_stats(&self, num_entries: usize) -> BinaryMemoryStats {
        let code_bytes = num_entries * self.num_bytes;
        let raw_bytes = num_entries * self.dim * 4;
        let compression_ratio = if code_bytes > 0 {
            raw_bytes as f32 / code_bytes as f32
        } else {
            0.0
        };

        BinaryMemoryStats {
            dim: self.dim,
            num_entries,
            code_bytes,
            raw_bytes,
            compression_ratio,
            bytes_per_entry: self.num_bytes,
        }
    }
}

/// Memory statistics for binary quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryMemoryStats {
    /// Embedding dimension
    pub dim: usize,
    /// Number of entries
    pub num_entries: usize,
    /// Total bytes for binary codes
    pub code_bytes: usize,
    /// Bytes that raw embeddings would use
    pub raw_bytes: usize,
    /// Compression ratio (raw / compressed)
    pub compression_ratio: f32,
    /// Bytes per entry
    pub bytes_per_entry: usize,
}

/// Normalize a vector to unit length
fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Corpus storage using binary quantization
///
/// Provides extremely memory-efficient storage with Hamming distance search.
#[derive(Debug)]
pub struct BinaryCorpus {
    /// Binary quantizer
    quantizer: BinaryQuantizer,
    /// Binary codes for each entry
    codes: Vec<BinaryCode>,
    /// Entry IDs
    ids: Vec<String>,
    /// Original embeddings for asymmetric search (optional)
    embeddings: Option<Vec<Embedding>>,
}

impl BinaryCorpus {
    /// Create a new binary corpus from embeddings
    pub fn build(
        embeddings: &[(String, Embedding)],
        config: BinaryConfig,
    ) -> Result<Self, BinaryError> {
        if embeddings.is_empty() {
            return Err(BinaryError::EmptyInput);
        }

        let dim = embeddings[0].1.dim;
        let quantizer = BinaryQuantizer::new(dim, config);

        let mut codes = Vec::with_capacity(embeddings.len());
        let mut ids = Vec::with_capacity(embeddings.len());

        for (id, emb) in embeddings {
            codes.push(quantizer.encode(emb)?);
            ids.push(id.clone());
        }

        Ok(Self {
            quantizer,
            codes,
            ids,
            embeddings: None,
        })
    }

    /// Create corpus keeping original embeddings for asymmetric search
    pub fn build_with_embeddings(
        embeddings: &[(String, Embedding)],
        config: BinaryConfig,
    ) -> Result<Self, BinaryError> {
        let mut corpus = Self::build(embeddings, config)?;
        corpus.embeddings = Some(embeddings.iter().map(|(_, e)| e.clone()).collect());
        Ok(corpus)
    }

    /// Number of entries in corpus
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Check if corpus is empty
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Find k nearest neighbors using Hamming distance
    ///
    /// Returns (id, hamming_distance) pairs sorted by distance.
    pub fn find_nearest_hamming(
        &self,
        query: &Embedding,
        k: usize,
    ) -> Result<Vec<(String, u32)>, BinaryError> {
        let query_code = self.quantizer.encode(query)?;

        // Compute all Hamming distances
        let mut distances: Vec<(usize, u32)> = self
            .codes
            .iter()
            .enumerate()
            .map(|(i, code)| (i, self.quantizer.hamming_distance(&query_code, code)))
            .collect();

        // Sort by distance (ascending)
        distances.sort_by_key(|&(_, d)| d);

        // Return top k
        Ok(distances
            .into_iter()
            .take(k)
            .map(|(i, d)| (self.ids[i].clone(), d))
            .collect())
    }

    /// Find k nearest neighbors using asymmetric distance
    ///
    /// More accurate than Hamming but slower (computes dot products).
    pub fn find_nearest_asymmetric(
        &self,
        query: &Embedding,
        k: usize,
    ) -> Result<Vec<(String, f32)>, BinaryError> {
        // Compute asymmetric distances
        let mut distances: Vec<(usize, f32)> = self
            .codes
            .iter()
            .enumerate()
            .map(|(i, code)| {
                let dist = self
                    .quantizer
                    .asymmetric_distance(query, code)
                    .unwrap_or(f32::MAX);
                (i, dist)
            })
            .collect();

        // Sort by distance (ascending - more negative = more similar)
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top k
        Ok(distances
            .into_iter()
            .take(k)
            .map(|(i, d)| (self.ids[i].clone(), d))
            .collect())
    }

    /// Two-stage search: Hamming filter + rerank
    ///
    /// First finds `candidates` nearest by Hamming distance, then reranks
    /// using asymmetric distance for better accuracy.
    pub fn find_nearest_rerank(
        &self,
        query: &Embedding,
        k: usize,
        candidates: usize,
    ) -> Result<Vec<(String, f32)>, BinaryError> {
        // Stage 1: Fast Hamming filter
        let query_code = self.quantizer.encode(query)?;
        let mut hamming_dists: Vec<(usize, u32)> = self
            .codes
            .iter()
            .enumerate()
            .map(|(i, code)| (i, self.quantizer.hamming_distance(&query_code, code)))
            .collect();

        hamming_dists.sort_by_key(|&(_, d)| d);
        let candidate_indices: Vec<usize> = hamming_dists
            .into_iter()
            .take(candidates)
            .map(|(i, _)| i)
            .collect();

        // Stage 2: Asymmetric rerank
        let mut reranked: Vec<(usize, f32)> = candidate_indices
            .into_iter()
            .map(|i| {
                let dist = self
                    .quantizer
                    .asymmetric_distance(query, &self.codes[i])
                    .unwrap_or(f32::MAX);
                (i, dist)
            })
            .collect();

        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(reranked
            .into_iter()
            .take(k)
            .map(|(i, d)| (self.ids[i].clone(), d))
            .collect())
    }

    /// Memory statistics
    pub fn memory_stats(&self) -> BinaryMemoryStats {
        self.quantizer.memory_stats(self.codes.len())
    }

    /// Measure recall compared to exact search
    ///
    /// Computes what fraction of true k-nearest neighbors are found.
    pub fn measure_recall(&self, queries: &[Embedding], k: usize) -> Result<f32, BinaryError> {
        if self.embeddings.is_none() {
            return Err(BinaryError::InvalidConfig(
                "Need original embeddings for recall measurement".into(),
            ));
        }

        let embeddings = self.embeddings.as_ref().unwrap();
        let mut total_recall = 0.0f32;

        for query in queries {
            // Exact nearest neighbors
            let mut exact_dists: Vec<(usize, f32)> = embeddings
                .iter()
                .enumerate()
                .map(|(i, e)| (i, euclidean_distance(&query.vector, &e.vector)))
                .collect();
            exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let exact_set: std::collections::HashSet<usize> =
                exact_dists.iter().take(k).map(|(i, _)| *i).collect();

            // Binary nearest neighbors
            let binary_results = self.find_nearest_hamming(query, k)?;
            let binary_set: std::collections::HashSet<usize> = binary_results
                .iter()
                .filter_map(|(id, _)| self.ids.iter().position(|x| x == id))
                .collect();

            // Compute recall
            let intersection = exact_set.intersection(&binary_set).count();
            total_recall += intersection as f32 / k as f32;
        }

        Ok(total_recall / queries.len() as f32)
    }

    /// Get quantizer reference
    pub fn quantizer(&self) -> &BinaryQuantizer {
        &self.quantizer
    }

    /// Get ID at index
    pub fn get_id(&self, idx: usize) -> Option<&str> {
        self.ids.get(idx).map(|s| s.as_str())
    }

    /// Get code at index
    pub fn get_code(&self, idx: usize) -> Option<&BinaryCode> {
        self.codes.get(idx)
    }
}

/// Euclidean distance between two vectors
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::EMBEDDING_DIM;

    fn random_embedding(seed: u64) -> Embedding {
        let mut state = if seed == 0 { 1 } else { seed };
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };

        let vector: Vec<f32> = (0..EMBEDDING_DIM).map(|_| next() * 2.0 - 1.0).collect();
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
    fn test_binary_config_default() {
        let config = BinaryConfig::default();
        assert!(config.normalize);
        assert_eq!(config.threshold, 0.0);
    }

    #[test]
    fn test_binary_config_for_normalized() {
        let config = BinaryConfig::for_normalized();
        assert!(!config.normalize);
        assert_eq!(config.threshold, 0.0);
    }

    #[test]
    fn test_binary_quantizer_creation() {
        let bq = BinaryQuantizer::with_dim(EMBEDDING_DIM);
        assert_eq!(bq.dim(), EMBEDDING_DIM);
        assert_eq!(bq.code_size(), 12); // 96 / 8 = 12 bytes
    }

    #[test]
    fn test_binary_encode_decode() {
        let bq = BinaryQuantizer::with_dim(EMBEDDING_DIM);
        let emb = normalized_embedding(42);

        let code = bq.encode(&emb).unwrap();
        assert_eq!(code.num_bytes(), 12);
        assert_eq!(code.dim, EMBEDDING_DIM);

        let decoded = bq.decode(&code);
        assert_eq!(decoded.dim, EMBEDDING_DIM);
        // Decoded values should be +/- 1/sqrt(dim)
        for v in &decoded.vector {
            let expected = 1.0 / (EMBEDDING_DIM as f32).sqrt();
            assert!((v.abs() - expected).abs() < 0.01);
        }
    }

    #[test]
    fn test_binary_code_bit_operations() {
        let mut code = BinaryCode::new(vec![0u8; 12], 96);

        // Set some bits
        code.set_bit(0, true);
        code.set_bit(7, true);
        code.set_bit(8, true);
        code.set_bit(95, true);

        assert!(code.get_bit(0));
        assert!(code.get_bit(7));
        assert!(code.get_bit(8));
        assert!(code.get_bit(95));
        assert!(!code.get_bit(1));
        assert!(!code.get_bit(50));
    }

    #[test]
    fn test_hamming_distance() {
        let bq = BinaryQuantizer::with_dim(EMBEDDING_DIM);

        // Identical embeddings should have 0 Hamming distance
        let emb = normalized_embedding(42);
        let code1 = bq.encode(&emb).unwrap();
        let code2 = bq.encode(&emb).unwrap();
        assert_eq!(bq.hamming_distance(&code1, &code2), 0);

        // Different embeddings should have > 0 Hamming distance
        let emb2 = normalized_embedding(123);
        let code3 = bq.encode(&emb2).unwrap();
        assert!(bq.hamming_distance(&code1, &code3) > 0);
    }

    #[test]
    fn test_hamming_to_cosine() {
        let bq = BinaryQuantizer::with_dim(EMBEDDING_DIM);

        // 0 Hamming distance -> cosine = 1.0
        assert!((bq.hamming_to_cosine(0) - 1.0).abs() < 0.001);

        // Max Hamming distance (all bits different) -> cosine = -1.0
        assert!((bq.hamming_to_cosine(96) - (-1.0)).abs() < 0.001);

        // Half bits different -> cosine = 0.0
        assert!((bq.hamming_to_cosine(48) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_asymmetric_distance() {
        let bq = BinaryQuantizer::with_dim(EMBEDDING_DIM);
        let query = normalized_embedding(42);
        let code = bq.encode(&query).unwrap();

        // Self-distance should be negative (high similarity)
        let dist = bq.asymmetric_distance(&query, &code).unwrap();
        assert!(dist < 0.0);

        // Different query should have higher (less negative) distance
        let other_query = normalized_embedding(123);
        let other_dist = bq.asymmetric_distance(&other_query, &code).unwrap();
        assert!(other_dist > dist);
    }

    #[test]
    fn test_quantization_error() {
        let bq = BinaryQuantizer::with_dim(EMBEDDING_DIM);
        let embeddings: Vec<Embedding> = (0..100).map(|i| normalized_embedding(i as u64)).collect();

        let error = bq.quantization_error(&embeddings).unwrap();
        // Error should be reasonable (binary quantization has high error)
        assert!(error > 0.0);
        assert!(error < 2.0); // MSE bounded
    }

    #[test]
    fn test_memory_stats() {
        let bq = BinaryQuantizer::with_dim(EMBEDDING_DIM);
        let stats = bq.memory_stats(1000);

        assert_eq!(stats.dim, EMBEDDING_DIM);
        assert_eq!(stats.num_entries, 1000);
        assert_eq!(stats.bytes_per_entry, 12);
        assert_eq!(stats.code_bytes, 12000);
        assert_eq!(stats.raw_bytes, 384000); // 1000 * 96 * 4
        assert!((stats.compression_ratio - 32.0).abs() < 0.1);
    }

    #[test]
    fn test_binary_corpus_build() {
        let embeddings: Vec<(String, Embedding)> = (0..100)
            .map(|i| (format!("id_{}", i), normalized_embedding(i as u64)))
            .collect();

        let corpus = BinaryCorpus::build(&embeddings, BinaryConfig::default()).unwrap();
        assert_eq!(corpus.len(), 100);
        assert!(!corpus.is_empty());
    }

    #[test]
    fn test_binary_corpus_find_nearest_hamming() {
        let embeddings: Vec<(String, Embedding)> = (0..100)
            .map(|i| (format!("id_{}", i), normalized_embedding(i as u64)))
            .collect();

        let corpus = BinaryCorpus::build(&embeddings, BinaryConfig::default()).unwrap();

        // Query for the first embedding - should find itself
        let results = corpus.find_nearest_hamming(&embeddings[0].1, 5).unwrap();
        assert_eq!(results.len(), 5);
        // First result should be exact match (0 Hamming distance)
        assert_eq!(results[0].0, "id_0");
        assert_eq!(results[0].1, 0);
    }

    #[test]
    fn test_binary_corpus_find_nearest_asymmetric() {
        let embeddings: Vec<(String, Embedding)> = (0..100)
            .map(|i| (format!("id_{}", i), normalized_embedding(i as u64)))
            .collect();

        let corpus = BinaryCorpus::build(&embeddings, BinaryConfig::default()).unwrap();

        let results = corpus.find_nearest_asymmetric(&embeddings[0].1, 5).unwrap();
        assert_eq!(results.len(), 5);
        // First result should be exact match (most negative distance)
        assert_eq!(results[0].0, "id_0");
    }

    #[test]
    fn test_binary_corpus_find_nearest_rerank() {
        let embeddings: Vec<(String, Embedding)> = (0..100)
            .map(|i| (format!("id_{}", i), normalized_embedding(i as u64)))
            .collect();

        let corpus = BinaryCorpus::build(&embeddings, BinaryConfig::default()).unwrap();

        let results = corpus.find_nearest_rerank(&embeddings[0].1, 5, 20).unwrap();
        assert_eq!(results.len(), 5);
        // First result should be exact match
        assert_eq!(results[0].0, "id_0");
    }

    #[test]
    fn test_binary_corpus_recall_measurement() {
        let embeddings: Vec<(String, Embedding)> = (0..100)
            .map(|i| (format!("id_{}", i), normalized_embedding(i as u64)))
            .collect();

        let corpus =
            BinaryCorpus::build_with_embeddings(&embeddings, BinaryConfig::default()).unwrap();

        let queries: Vec<Embedding> = (0..10).map(|i| normalized_embedding(i as u64)).collect();
        let recall = corpus.measure_recall(&queries, 5).unwrap();

        // Binary quantization has high information loss, so recall can be low
        // with random embeddings. The key is that exact matches (query = entry)
        // should always be found, giving at least 1/k recall per query.
        // With k=5 and 10 queries where each query is also an entry, expect >= 0.2
        assert!(recall >= 0.2, "recall was {} but expected >= 0.2", recall);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let bq = BinaryQuantizer::with_dim(EMBEDDING_DIM);
        let wrong_dim_emb = Embedding::new(vec![0.0; 64]); // Wrong dimension

        let result = bq.encode(&wrong_dim_emb);
        assert!(result.is_err());
        match result {
            Err(BinaryError::DimensionMismatch { expected, got }) => {
                assert_eq!(expected, EMBEDDING_DIM);
                assert_eq!(got, 64);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_empty_corpus_error() {
        let embeddings: Vec<(String, Embedding)> = vec![];
        let result = BinaryCorpus::build(&embeddings, BinaryConfig::default());
        assert!(result.is_err());
        assert!(matches!(result, Err(BinaryError::EmptyInput)));
    }

    #[test]
    fn test_encode_batch_parallel_single_thread() {
        let bq = BinaryQuantizer::with_dim(EMBEDDING_DIM);

        let embeddings: Vec<Embedding> = (0..50).map(|i| random_embedding(i as u64)).collect();

        // Test with 1 thread (fallback to sequential)
        let parallel_codes = bq.encode_batch_parallel(&embeddings, 1).unwrap();
        let sequential_codes = bq.encode_batch(&embeddings).unwrap();

        assert_eq!(parallel_codes.len(), sequential_codes.len());
        for (i, (par, seq)) in parallel_codes
            .iter()
            .zip(sequential_codes.iter())
            .enumerate()
        {
            assert_eq!(par.bytes, seq.bytes, "Binary mismatch at index {}", i);
        }
    }

    #[test]
    fn test_encode_batch_parallel_multi_thread() {
        let bq = BinaryQuantizer::with_dim(EMBEDDING_DIM);

        let embeddings: Vec<Embedding> = (0..100).map(|i| random_embedding(i as u64)).collect();

        // Test with 4 threads
        let parallel_codes = bq.encode_batch_parallel(&embeddings, 4).unwrap();
        let sequential_codes = bq.encode_batch(&embeddings).unwrap();

        assert_eq!(parallel_codes.len(), sequential_codes.len());
        for (i, (par, seq)) in parallel_codes
            .iter()
            .zip(sequential_codes.iter())
            .enumerate()
        {
            assert_eq!(
                par.bytes, seq.bytes,
                "Binary mismatch at index {} with 4 threads",
                i
            );
        }

        // Test with more threads than embeddings
        let small_batch: Vec<Embedding> = (0..5).map(|i| random_embedding(i as u64)).collect();
        let parallel_codes_excess = bq.encode_batch_parallel(&small_batch, 20).unwrap();
        let sequential_codes_small = bq.encode_batch(&small_batch).unwrap();

        assert_eq!(parallel_codes_excess.len(), sequential_codes_small.len());
        for (i, (par, seq)) in parallel_codes_excess
            .iter()
            .zip(sequential_codes_small.iter())
            .enumerate()
        {
            assert_eq!(
                par.bytes, seq.bytes,
                "Binary mismatch at index {} with excess threads",
                i
            );
        }
    }

    #[test]
    fn test_encode_batch_parallel_empty() {
        let bq = BinaryQuantizer::with_dim(EMBEDDING_DIM);

        // Test with empty batch
        let empty: Vec<Embedding> = vec![];
        let parallel_codes = bq.encode_batch_parallel(&empty, 4).unwrap();
        assert!(parallel_codes.is_empty());
    }

    #[test]
    fn test_encode_batch_parallel_dimension_mismatch() {
        let bq = BinaryQuantizer::with_dim(EMBEDDING_DIM);

        // Include an embedding with wrong dimension
        let mut embeddings: Vec<Embedding> = (0..10).map(|i| random_embedding(i as u64)).collect();
        embeddings.push(Embedding::new(vec![0.5; EMBEDDING_DIM / 2])); // Wrong dim

        let result = bq.encode_batch_parallel(&embeddings, 4);
        assert!(result.is_err());
        match result {
            Err(BinaryError::DimensionMismatch { expected, got }) => {
                assert_eq!(expected, EMBEDDING_DIM);
                assert_eq!(got, EMBEDDING_DIM / 2);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }
}
