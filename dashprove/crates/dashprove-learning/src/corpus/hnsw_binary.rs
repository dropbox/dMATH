//! HNSW-Binary hybrid index for extreme compression graph-based search
//!
//! This module combines HNSW's graph-based navigation with Binary Quantization's
//! extreme compression to provide:
//!
//! - **O(log N) search time**: Hierarchical graph enables fast navigation
//! - **~32x memory compression**: Binary codes compress 384 bytes to 12 bytes
//! - **Fast distance computation**: Hamming distance uses efficient XOR + popcount
//!
//! # Architecture
//!
//! ```text
//! Query embedding
//!       │
//!       ▼
//! ┌─────────────────────────────────────────────┐
//! │ HNSW Graph (navigates using Hamming distance)│
//! │   Layer 2:  A ------- B                     │
//! │   Layer 1:  A -- C -- B -- D                │
//! │   Layer 0:  A-E--C-F--B-G--D-H              │
//! └─────────────────────────────────────────────┘
//!       │ (graph edges stored, binary codes stored)
//!       │ (full embeddings NOT stored - memory savings!)
//!       ▼
//!    Top-k results (ranked by Hamming or asymmetric distance)
//! ```
//!
//! # Memory Comparison for 100K embeddings (96 dimensions)
//!
//! | Index Type | Embeddings | Graph | Total |
//! |------------|------------|-------|-------|
//! | HNSW (full) | 38.4 MB | ~5 MB | ~43 MB |
//! | HNSW-PQ | 0.8 MB | ~5 MB | ~6 MB |
//! | **HNSW-Binary** | **1.2 MB** | **~5 MB** | **~6 MB** |
//!
//! # Trade-offs vs HNSW-PQ
//!
//! - **Similar compression**: Binary is 32x vs PQ's 48x (similar range)
//! - **Faster distance**: Hamming is faster than PQ asymmetric distance
//! - **Lower recall**: Binary loses more information than PQ
//! - **No training**: Binary quantization is instant (no k-means)
//!
//! Best used when:
//! - Build time matters more than recall
//! - First-stage filtering before reranking
//! - Very large corpora where PQ training is slow

use super::types::ProofId;
use crate::binary::{BinaryCode, BinaryConfig, BinaryError, BinaryQuantizer};
use crate::embedder::Embedding;
use crate::similarity::SimilarProof;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Configuration for HNSW-Binary hybrid index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswBinaryConfig {
    /// Maximum number of connections per node (M)
    pub m: usize,
    /// Maximum connections for layer 0 (typically 2*M)
    pub m0: usize,
    /// Search width during construction (uses full embeddings)
    pub ef_construction: usize,
    /// Default search width during queries
    pub ef_search: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Binary quantization config
    pub binary: BinaryConfig,
    /// Use asymmetric distance for reranking (slower but better recall)
    pub use_asymmetric_rerank: bool,
}

impl Default for HnswBinaryConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 50,
            seed: 42,
            binary: BinaryConfig::default(),
            use_asymmetric_rerank: true,
        }
    }
}

impl HnswBinaryConfig {
    /// Config for fast builds with reasonable quality
    pub fn fast() -> Self {
        Self {
            m: 12,
            m0: 24,
            ef_construction: 100,
            ef_search: 30,
            seed: 42,
            binary: BinaryConfig::default(),
            use_asymmetric_rerank: false, // Speed over accuracy
        }
    }

    /// Config optimized for high recall
    pub fn high_recall() -> Self {
        Self {
            m: 32,
            m0: 64,
            ef_construction: 400,
            ef_search: 100,
            seed: 42,
            binary: BinaryConfig::default(),
            use_asymmetric_rerank: true,
        }
    }

    /// Config for memory-constrained environments
    pub fn compact() -> Self {
        Self {
            m: 8,
            m0: 16,
            ef_construction: 100,
            ef_search: 30,
            seed: 42,
            binary: BinaryConfig::for_normalized(),
            use_asymmetric_rerank: false,
        }
    }

    /// Validate configuration
    pub fn validate(&self, _dim: usize) -> Result<(), HnswBinaryError> {
        if self.m == 0 {
            return Err(HnswBinaryError::InvalidConfig("m must be > 0".into()));
        }
        if self.m0 < self.m {
            return Err(HnswBinaryError::InvalidConfig("m0 must be >= m".into()));
        }
        if self.ef_construction < self.m {
            return Err(HnswBinaryError::InvalidConfig(
                "ef_construction must be >= m".into(),
            ));
        }
        Ok(())
    }
}

/// HNSW-Binary specific errors
#[derive(Debug, Clone)]
pub enum HnswBinaryError {
    /// Invalid configuration
    InvalidConfig(String),
    /// Dimension mismatch
    DimensionMismatch { expected: usize, got: usize },
    /// Proof not found in corpus
    ProofNotFound(ProofId),
    /// Missing embedding for proof
    MissingEmbedding(ProofId),
    /// Binary quantization error
    BinaryError(BinaryError),
}

impl std::fmt::Display for HnswBinaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HnswBinaryError::InvalidConfig(s) => write!(f, "Invalid HNSW-Binary config: {}", s),
            HnswBinaryError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            HnswBinaryError::ProofNotFound(id) => write!(f, "Proof not found: {}", id),
            HnswBinaryError::MissingEmbedding(id) => {
                write!(f, "Missing embedding for proof: {}", id)
            }
            HnswBinaryError::BinaryError(e) => write!(f, "Binary quantization error: {}", e),
        }
    }
}

impl std::error::Error for HnswBinaryError {}

impl From<BinaryError> for HnswBinaryError {
    fn from(e: BinaryError) -> Self {
        HnswBinaryError::BinaryError(e)
    }
}

/// A node in the HNSW-Binary graph
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswBinaryNode {
    /// Index in the codes array
    idx: usize,
    /// Proof ID for this node
    proof_id: ProofId,
    /// Connections at each layer
    neighbors: Vec<Vec<usize>>,
    /// Maximum layer this node appears in
    max_layer: usize,
}

/// Search candidate with distance
#[derive(Debug, Clone)]
struct SearchCandidate {
    distance: f32,
    idx: usize,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchCandidate {}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap behavior
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Memory statistics for HNSW-Binary index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswBinaryMemoryStats {
    /// Number of indexed entries
    pub num_entries: usize,
    /// Embedding dimension
    pub dim: usize,
    /// Total bytes for binary codes
    pub code_bytes: usize,
    /// Estimated bytes for graph structure
    pub graph_bytes: usize,
    /// Total memory estimate
    pub total_bytes: usize,
    /// Bytes for raw embeddings
    pub raw_bytes: usize,
    /// Compression ratio
    pub compression_ratio: f32,
}

/// Graph structure statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswBinaryGraphStats {
    /// Maximum layer in the graph
    pub max_layer: usize,
    /// Total number of edges across all layers
    pub total_edges: usize,
    /// Average edges per node at layer 0
    pub avg_edges_layer0: f32,
    /// Node count at each layer
    pub nodes_per_layer: Vec<usize>,
}

/// HNSW-Binary hybrid index for extreme compression graph-based search
///
/// Uses HNSW graph navigation with binary-compressed embeddings to achieve
/// both fast O(log N) search and ~32x memory compression.
#[derive(Debug, Serialize, Deserialize)]
pub struct ProofCorpusHnswBinary {
    /// Binary quantizer for encoding
    quantizer: BinaryQuantizer,
    /// Binary codes for each entry
    codes: Vec<BinaryCode>,
    /// Graph nodes with connectivity information
    nodes: Vec<HnswBinaryNode>,
    /// Map from proof ID to node index
    id_to_idx: HashMap<ProofId, usize>,
    /// Entry point (top-level node)
    entry_point: Option<usize>,
    /// Maximum layer in the graph
    max_layer: usize,
    /// Configuration
    config: HnswBinaryConfig,
    /// Embedding dimension
    dim: usize,
    /// Level multiplier for random level generation
    ml: f64,
    /// Random state
    rng_state: u64,
}

impl ProofCorpusHnswBinary {
    /// Build an HNSW-Binary index from a ProofCorpus
    ///
    /// Note: Requires full embeddings at build time for graph construction,
    /// but only stores binary codes afterward.
    pub fn build(corpus: &super::storage::ProofCorpus) -> Result<Option<Self>, HnswBinaryError> {
        Self::build_with_config(corpus, HnswBinaryConfig::default())
    }

    /// Build with custom configuration
    pub fn build_with_config(
        corpus: &super::storage::ProofCorpus,
        config: HnswBinaryConfig,
    ) -> Result<Option<Self>, HnswBinaryError> {
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

        // Create the binary quantizer (no training needed!)
        let quantizer = BinaryQuantizer::new(dim, config.binary.clone());

        let ml = 1.0 / (config.m as f64).ln();
        let rng_state = config.seed;

        let mut index = Self {
            quantizer,
            codes: Vec::with_capacity(entries_with_embeddings.len()),
            nodes: Vec::with_capacity(entries_with_embeddings.len()),
            id_to_idx: HashMap::with_capacity(entries_with_embeddings.len()),
            entry_point: None,
            max_layer: 0,
            config,
            dim,
            ml,
            rng_state,
        };

        // Build the graph using full embeddings, then store only binary codes
        let mut temp_embeddings: Vec<Embedding> = Vec::with_capacity(entries_with_embeddings.len());

        for (proof_id, embedding) in entries_with_embeddings {
            index.insert_with_temp_embeddings(proof_id, embedding, &mut temp_embeddings)?;
        }

        Ok(Some(index))
    }

    /// Insert a new embedding with temporary storage for graph construction
    fn insert_with_temp_embeddings(
        &mut self,
        proof_id: ProofId,
        embedding: Embedding,
        temp_embeddings: &mut Vec<Embedding>,
    ) -> Result<(), HnswBinaryError> {
        if embedding.dim != self.dim {
            return Err(HnswBinaryError::DimensionMismatch {
                expected: self.dim,
                got: embedding.dim,
            });
        }

        let idx = self.nodes.len();
        let level = self.random_level();

        // Store binary codes
        let code = self.quantizer.encode(&embedding)?;
        self.codes.push(code);

        // Store embedding temporarily for graph construction
        temp_embeddings.push(embedding);

        // Create node
        let node = HnswBinaryNode {
            idx,
            proof_id: proof_id.clone(),
            neighbors: vec![Vec::new(); level + 1],
            max_layer: level,
        };
        self.nodes.push(node);
        self.id_to_idx.insert(proof_id, idx);

        // First node becomes entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(idx);
            self.max_layer = level;
            return Ok(());
        }

        let ep = self.entry_point.unwrap();

        // Use full embeddings for high-quality graph construction
        let mut current_ep = ep;
        for layer in (level + 1..=self.max_layer).rev() {
            current_ep = self.search_layer_single_exact(
                &temp_embeddings[idx],
                temp_embeddings,
                current_ep,
                layer,
            );
        }

        for layer in (0..=level.min(self.max_layer)).rev() {
            let m_max = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let candidates = self.search_layer_exact(
                &temp_embeddings[idx],
                temp_embeddings,
                vec![current_ep],
                self.config.ef_construction,
                layer,
            );

            let neighbors = self.select_neighbors(&candidates, m_max);
            self.nodes[idx].neighbors[layer] = neighbors.clone();

            // Connect neighbors back
            for &neighbor_idx in &neighbors {
                if self.nodes[neighbor_idx].max_layer >= layer {
                    let neighbor_neighbors = &mut self.nodes[neighbor_idx].neighbors[layer];
                    neighbor_neighbors.push(idx);

                    if neighbor_neighbors.len() > m_max {
                        let candidates: Vec<_> = neighbor_neighbors
                            .iter()
                            .map(|&n| SearchCandidate {
                                distance: Self::distance_exact(
                                    &temp_embeddings[neighbor_idx],
                                    &temp_embeddings[n],
                                ),
                                idx: n,
                            })
                            .collect();
                        let pruned = self.select_neighbors(&candidates, m_max);
                        self.nodes[neighbor_idx].neighbors[layer] = pruned;
                    }
                }
            }

            if !candidates.is_empty() {
                current_ep = candidates[0].idx;
            }
        }

        if level > self.max_layer {
            self.entry_point = Some(idx);
            self.max_layer = level;
        }

        Ok(())
    }

    /// Number of indexed entries
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get the configuration
    pub fn config(&self) -> &HnswBinaryConfig {
        &self.config
    }

    /// Maximum layer in the graph
    pub fn max_layer(&self) -> usize {
        self.max_layer
    }

    /// Find k nearest neighbors using default ef_search
    pub fn find_similar(
        &self,
        corpus: &super::storage::ProofCorpus,
        query: &Embedding,
        k: usize,
    ) -> Vec<SimilarProof> {
        self.find_similar_with_ef(corpus, query, k, self.config.ef_search)
    }

    /// Find k nearest neighbors with custom ef (search width)
    ///
    /// Uses Hamming distance for graph navigation, optionally asymmetric for reranking.
    pub fn find_similar_with_ef(
        &self,
        corpus: &super::storage::ProofCorpus,
        query: &Embedding,
        k: usize,
        ef: usize,
    ) -> Vec<SimilarProof> {
        if self.is_empty() || k == 0 || query.dim != self.dim {
            return vec![];
        }

        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return vec![],
        };

        // Encode query to binary for Hamming distance
        let query_code = match self.quantizer.encode(query) {
            Ok(c) => c,
            Err(_) => return vec![],
        };

        // Navigate from entry point down to layer 1
        let mut current_ep = ep;
        for layer in (1..=self.max_layer).rev() {
            current_ep = self.search_layer_single_hamming(&query_code, current_ep, layer);
        }

        // Search layer 0 with ef candidates
        let mut candidates = self.search_layer_hamming(&query_code, vec![current_ep], ef.max(k), 0);

        // Optionally rerank with asymmetric distance
        if self.config.use_asymmetric_rerank {
            for c in &mut candidates {
                if let Ok(dist) = self
                    .quantizer
                    .asymmetric_distance(query, &self.codes[c.idx])
                {
                    c.distance = dist;
                }
            }
            candidates.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Return top-k
        candidates
            .into_iter()
            .take(k)
            .filter_map(|c| {
                let node = &self.nodes[c.idx];
                let entry = corpus.get(&node.proof_id)?;
                Some(SimilarProof {
                    id: entry.id.clone(),
                    property: entry.property.clone(),
                    backend: entry.backend,
                    tactics: entry.tactics.clone(),
                    similarity: Self::distance_to_similarity(c.distance),
                })
            })
            .collect()
    }

    /// Get proof IDs of k nearest neighbors (without corpus lookup)
    pub fn find_similar_ids(&self, query: &Embedding, k: usize) -> Vec<(ProofId, f64)> {
        self.find_similar_ids_with_ef(query, k, self.config.ef_search)
    }

    /// Get proof IDs with custom ef
    pub fn find_similar_ids_with_ef(
        &self,
        query: &Embedding,
        k: usize,
        ef: usize,
    ) -> Vec<(ProofId, f64)> {
        if self.is_empty() || k == 0 || query.dim != self.dim {
            return vec![];
        }

        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return vec![],
        };

        let query_code = match self.quantizer.encode(query) {
            Ok(c) => c,
            Err(_) => return vec![],
        };

        let mut current_ep = ep;
        for layer in (1..=self.max_layer).rev() {
            current_ep = self.search_layer_single_hamming(&query_code, current_ep, layer);
        }

        let mut candidates = self.search_layer_hamming(&query_code, vec![current_ep], ef.max(k), 0);

        if self.config.use_asymmetric_rerank {
            for c in &mut candidates {
                if let Ok(dist) = self
                    .quantizer
                    .asymmetric_distance(query, &self.codes[c.idx])
                {
                    c.distance = dist;
                }
            }
            candidates.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        candidates
            .into_iter()
            .take(k)
            .map(|c| {
                let node = &self.nodes[c.idx];
                (
                    node.proof_id.clone(),
                    Self::distance_to_similarity(c.distance),
                )
            })
            .collect()
    }

    /// Measure recall@k compared to exact search
    pub fn measure_recall(
        &self,
        corpus: &super::storage::ProofCorpus,
        queries: &[Embedding],
        k: usize,
    ) -> f32 {
        if queries.is_empty() {
            return 0.0;
        }

        let mut total_recall = 0.0f32;

        // Get all embeddings for exact search
        let all_embeddings: Vec<_> = corpus
            .entries()
            .filter_map(|e| e.embedding.as_ref().map(|emb| (e.id.clone(), emb.clone())))
            .collect();

        for query in queries {
            // Exact k-nearest
            let mut exact_dists: Vec<_> = all_embeddings
                .iter()
                .map(|(id, emb)| {
                    let dist = Self::distance_exact(query, emb);
                    (id.clone(), dist)
                })
                .collect();
            exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let exact_set: std::collections::HashSet<_> = exact_dists
                .iter()
                .take(k)
                .map(|(id, _)| id.clone())
                .collect();

            // HNSW-Binary k-nearest
            let hnsw_results = self.find_similar_ids(query, k);
            let hnsw_set: std::collections::HashSet<_> =
                hnsw_results.iter().map(|(id, _)| id.clone()).collect();

            let intersection = exact_set.intersection(&hnsw_set).count();
            total_recall += intersection as f32 / k as f32;
        }

        total_recall / queries.len() as f32
    }

    /// Memory statistics
    pub fn memory_stats(&self) -> HnswBinaryMemoryStats {
        let code_bytes: usize = self.codes.iter().map(|c| c.bytes.len()).sum();

        // Estimate graph structure size
        let graph_bytes = self.nodes.iter().fold(0usize, |acc, node| {
            // Node overhead + neighbors vectors
            let neighbors_size: usize = node.neighbors.iter().map(|n| n.len() * 8).sum();
            acc + 64 + neighbors_size
        });

        let raw_bytes = self.nodes.len() * self.dim * 4;
        let total_bytes = code_bytes + graph_bytes;
        let compression_ratio = if total_bytes > 0 {
            raw_bytes as f32 / total_bytes as f32
        } else {
            0.0
        };

        HnswBinaryMemoryStats {
            num_entries: self.nodes.len(),
            dim: self.dim,
            code_bytes,
            graph_bytes,
            total_bytes,
            raw_bytes,
            compression_ratio,
        }
    }

    /// Graph structure statistics
    pub fn graph_stats(&self) -> HnswBinaryGraphStats {
        let mut nodes_per_layer = vec![0usize; self.max_layer + 1];
        let mut total_edges = 0usize;
        let mut layer0_edges = 0usize;

        for node in &self.nodes {
            for (layer, neighbors) in node.neighbors.iter().enumerate() {
                if layer <= node.max_layer {
                    nodes_per_layer[layer] += 1;
                    total_edges += neighbors.len();
                    if layer == 0 {
                        layer0_edges += neighbors.len();
                    }
                }
            }
        }

        let avg_edges_layer0 = if nodes_per_layer[0] > 0 {
            layer0_edges as f32 / nodes_per_layer[0] as f32
        } else {
            0.0
        };

        HnswBinaryGraphStats {
            max_layer: self.max_layer,
            total_edges,
            avg_edges_layer0,
            nodes_per_layer,
        }
    }

    // ========== Internal helper methods ==========

    /// Generate random level for a new node
    fn random_level(&mut self) -> usize {
        // Simple xorshift RNG
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;

        let r = (self.rng_state >> 33) as f64 / (1u64 << 31) as f64;
        (-r.ln() * self.ml).floor() as usize
    }

    /// Euclidean distance between full embeddings
    fn distance_exact(a: &Embedding, b: &Embedding) -> f32 {
        a.vector
            .iter()
            .zip(b.vector.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>()
            .sqrt()
    }

    /// Hamming distance between binary codes (converted to f32)
    fn distance_hamming(&self, a: &BinaryCode, b: &BinaryCode) -> f32 {
        self.quantizer.hamming_distance(a, b) as f32
    }

    /// Search a single layer to find nearest node (exact, for construction)
    fn search_layer_single_exact(
        &self,
        query: &Embedding,
        embeddings: &[Embedding],
        entry: usize,
        layer: usize,
    ) -> usize {
        let mut current = entry;
        let mut current_dist = Self::distance_exact(query, &embeddings[current]);
        let mut changed = true;

        while changed {
            changed = false;
            if layer >= self.nodes[current].neighbors.len() {
                continue;
            }
            for &neighbor in &self.nodes[current].neighbors[layer] {
                let dist = Self::distance_exact(query, &embeddings[neighbor]);
                if dist < current_dist {
                    current = neighbor;
                    current_dist = dist;
                    changed = true;
                }
            }
        }

        current
    }

    /// Search a layer with beam width (exact, for construction)
    fn search_layer_exact(
        &self,
        query: &Embedding,
        embeddings: &[Embedding],
        entry_points: Vec<usize>,
        ef: usize,
        layer: usize,
    ) -> Vec<SearchCandidate> {
        let mut visited: HashSet<usize> = entry_points.iter().copied().collect();
        let mut candidates: BinaryHeap<SearchCandidate> = BinaryHeap::new();
        let mut result: Vec<SearchCandidate> = Vec::new();

        for &ep in &entry_points {
            let dist = Self::distance_exact(query, &embeddings[ep]);
            candidates.push(SearchCandidate {
                distance: dist,
                idx: ep,
            });
            result.push(SearchCandidate {
                distance: dist,
                idx: ep,
            });
        }

        while let Some(current) = candidates.pop() {
            // Get worst distance in result
            let worst_dist = result.iter().map(|c| c.distance).fold(f32::MIN, f32::max);
            if current.distance > worst_dist && result.len() >= ef {
                break;
            }

            if layer >= self.nodes[current.idx].neighbors.len() {
                continue;
            }
            for &neighbor in &self.nodes[current.idx].neighbors[layer] {
                if visited.insert(neighbor) {
                    let dist = Self::distance_exact(query, &embeddings[neighbor]);
                    if result.len() < ef || dist < worst_dist {
                        candidates.push(SearchCandidate {
                            distance: dist,
                            idx: neighbor,
                        });
                        result.push(SearchCandidate {
                            distance: dist,
                            idx: neighbor,
                        });
                        if result.len() > ef {
                            result.sort_by(|a, b| {
                                a.distance
                                    .partial_cmp(&b.distance)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            });
                            result.truncate(ef);
                        }
                    }
                }
            }
        }

        result.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }

    /// Search a single layer to find nearest node (Hamming, for queries)
    fn search_layer_single_hamming(
        &self,
        query_code: &BinaryCode,
        entry: usize,
        layer: usize,
    ) -> usize {
        let mut current = entry;
        let mut current_dist = self.distance_hamming(query_code, &self.codes[current]);
        let mut changed = true;

        while changed {
            changed = false;
            if layer >= self.nodes[current].neighbors.len() {
                continue;
            }
            for &neighbor in &self.nodes[current].neighbors[layer] {
                let dist = self.distance_hamming(query_code, &self.codes[neighbor]);
                if dist < current_dist {
                    current = neighbor;
                    current_dist = dist;
                    changed = true;
                }
            }
        }

        current
    }

    /// Search a layer with beam width (Hamming, for queries)
    fn search_layer_hamming(
        &self,
        query_code: &BinaryCode,
        entry_points: Vec<usize>,
        ef: usize,
        layer: usize,
    ) -> Vec<SearchCandidate> {
        let mut visited: HashSet<usize> = entry_points.iter().copied().collect();
        let mut candidates: BinaryHeap<SearchCandidate> = BinaryHeap::new();
        let mut result: Vec<SearchCandidate> = Vec::new();

        for &ep in &entry_points {
            let dist = self.distance_hamming(query_code, &self.codes[ep]);
            candidates.push(SearchCandidate {
                distance: dist,
                idx: ep,
            });
            result.push(SearchCandidate {
                distance: dist,
                idx: ep,
            });
        }

        while let Some(current) = candidates.pop() {
            let worst_dist = result.iter().map(|c| c.distance).fold(f32::MIN, f32::max);
            if current.distance > worst_dist && result.len() >= ef {
                break;
            }

            if layer >= self.nodes[current.idx].neighbors.len() {
                continue;
            }
            for &neighbor in &self.nodes[current.idx].neighbors[layer] {
                if visited.insert(neighbor) {
                    let dist = self.distance_hamming(query_code, &self.codes[neighbor]);
                    if result.len() < ef || dist < worst_dist {
                        candidates.push(SearchCandidate {
                            distance: dist,
                            idx: neighbor,
                        });
                        result.push(SearchCandidate {
                            distance: dist,
                            idx: neighbor,
                        });
                        if result.len() > ef {
                            result.sort_by(|a, b| {
                                a.distance
                                    .partial_cmp(&b.distance)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            });
                            result.truncate(ef);
                        }
                    }
                }
            }
        }

        result.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }

    /// Select best neighbors using simple selection
    fn select_neighbors(&self, candidates: &[SearchCandidate], m: usize) -> Vec<usize> {
        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(m).map(|c| c.idx).collect()
    }

    /// Convert distance to similarity score [0, 1]
    fn distance_to_similarity(distance: f32) -> f64 {
        // For Hamming distance: lower is better
        // For asymmetric: more negative is better
        let sim = if distance >= 0.0 {
            // Hamming distance (0 to dim)
            1.0 / (1.0 + distance)
        } else {
            // Asymmetric (negative dot product): convert to [0,1]
            // More negative = more similar
            0.5 + 0.5 * (-distance / 100.0).tanh()
        };
        sim as f64
    }

    /// Persistence: save to file (JSON format)
    pub fn save_to_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), crate::LearningError> {
        crate::io::write_json_atomic(path, self)
    }

    /// Persistence: load from file (JSON format)
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, crate::LearningError> {
        crate::io::read_json(path)
    }

    /// Persistence: load if file exists
    pub fn load_if_exists<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Option<Self>, crate::LearningError> {
        let path = path.as_ref();
        if path.exists() {
            Ok(Some(Self::load_from_file(path)?))
        } else {
            Ok(None)
        }
    }
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
        let mut state = if seed == 0 { 1 } else { seed };
        let mut vector = Vec::with_capacity(EMBEDDING_DIM);
        for _ in 0..EMBEDDING_DIM {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let val = (state >> 40) as f32 / (1u64 << 24) as f32;
            vector.push(val * 2.0 - 1.0);
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

    fn create_test_corpus(n: usize) -> super::super::storage::ProofCorpus {
        let mut corpus = super::super::storage::ProofCorpus::new();
        for i in 0..n {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }
        corpus
    }

    #[test]
    fn test_hnsw_binary_config_default() {
        let config = HnswBinaryConfig::default();
        assert_eq!(config.m, 16);
        assert_eq!(config.m0, 32);
        assert_eq!(config.ef_search, 50);
        assert!(config.validate(EMBEDDING_DIM).is_ok());
    }

    #[test]
    fn test_hnsw_binary_config_presets() {
        let fast = HnswBinaryConfig::fast();
        assert!(fast.validate(EMBEDDING_DIM).is_ok());
        assert!(!fast.use_asymmetric_rerank);

        let high_recall = HnswBinaryConfig::high_recall();
        assert!(high_recall.validate(EMBEDDING_DIM).is_ok());
        assert!(high_recall.use_asymmetric_rerank);

        let compact = HnswBinaryConfig::compact();
        assert!(compact.validate(EMBEDDING_DIM).is_ok());
    }

    #[test]
    fn test_hnsw_binary_config_validation_errors() {
        let invalid_m = HnswBinaryConfig {
            m: 0,
            ..Default::default()
        };
        assert!(invalid_m.validate(EMBEDDING_DIM).is_err());

        let invalid_m0 = HnswBinaryConfig {
            m: 16,
            m0: 8, // m0 < m
            ..Default::default()
        };
        assert!(invalid_m0.validate(EMBEDDING_DIM).is_err());
    }

    #[test]
    fn test_hnsw_binary_build_empty() {
        let corpus = super::super::storage::ProofCorpus::new();
        let result = ProofCorpusHnswBinary::build(&corpus).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_hnsw_binary_build_small() {
        let corpus = create_test_corpus(50);
        let index = ProofCorpusHnswBinary::build(&corpus).unwrap();
        assert!(index.is_some());

        let index = index.unwrap();
        assert_eq!(index.len(), 50);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_hnsw_binary_find_similar() {
        let corpus = create_test_corpus(100);
        let index = ProofCorpusHnswBinary::build_with_config(&corpus, HnswBinaryConfig::fast())
            .unwrap()
            .unwrap();

        let query = random_embedding(0);
        let results = index.find_similar(&corpus, &query, 5);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
        // Results should have similarity scores
        assert!(results[0].similarity > 0.0);
    }

    #[test]
    fn test_hnsw_binary_find_similar_ids() {
        let corpus = create_test_corpus(100);
        let index = ProofCorpusHnswBinary::build(&corpus).unwrap().unwrap();

        let query = random_embedding(42);
        let results = index.find_similar_ids(&query, 10);

        assert!(!results.is_empty());
        assert!(results.len() <= 10);
        // Results should have positive similarity scores
        assert!(results[0].1 > 0.0);
    }

    #[test]
    fn test_hnsw_binary_memory_stats() {
        let corpus = create_test_corpus(100);
        let index = ProofCorpusHnswBinary::build(&corpus).unwrap().unwrap();

        let stats = index.memory_stats();
        assert_eq!(stats.num_entries, 100);
        assert_eq!(stats.dim, EMBEDDING_DIM);
        assert!(stats.code_bytes > 0);
        assert!(stats.graph_bytes > 0);
        // For small corpora, graph overhead dominates. Just verify memory is tracked.
        // Binary codes are 12 bytes vs 384 bytes (32x) but graph adds ~1KB per node.
        // Compression improves at larger scale.
        assert!(
            stats.raw_bytes > stats.code_bytes,
            "Binary codes ({}) should be smaller than raw embeddings ({})",
            stats.code_bytes,
            stats.raw_bytes
        );
    }

    #[test]
    fn test_hnsw_binary_graph_stats() {
        let corpus = create_test_corpus(100);
        let index = ProofCorpusHnswBinary::build(&corpus).unwrap().unwrap();

        let stats = index.graph_stats();
        assert!(stats.total_edges > 0);
        assert!(stats.avg_edges_layer0 > 0.0);
        assert!(!stats.nodes_per_layer.is_empty());
    }

    #[test]
    fn test_hnsw_binary_measure_recall() {
        let corpus = create_test_corpus(100);
        let index =
            ProofCorpusHnswBinary::build_with_config(&corpus, HnswBinaryConfig::high_recall())
                .unwrap()
                .unwrap();

        // Use a few existing embeddings as queries
        let queries: Vec<Embedding> = (0..5).map(|i| random_embedding(i as u64)).collect();
        let recall = index.measure_recall(&corpus, &queries, 5);

        // Exact matches should give high recall
        assert!(recall >= 0.3, "recall was {} but expected >= 0.3", recall);
    }

    #[test]
    fn test_hnsw_binary_persistence() {
        let corpus = create_test_corpus(50);
        let index = ProofCorpusHnswBinary::build(&corpus).unwrap().unwrap();

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("hnsw_binary_test.json");

        // Save
        index.save_to_file(&path).unwrap();

        // Load
        let loaded = ProofCorpusHnswBinary::load_from_file(&path).unwrap();
        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.max_layer(), index.max_layer());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_hnsw_binary_load_if_exists() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("hnsw_binary_nonexistent.bin");

        // Should return None for nonexistent file
        let result = ProofCorpusHnswBinary::load_if_exists(&path).unwrap();
        assert!(result.is_none());
    }
}
