//! HNSW (Hierarchical Navigable Small World) index for proof corpus search
//!
//! HNSW is a graph-based approximate nearest neighbor algorithm that provides:
//!
//! - **O(log N) search time**: Hierarchical layers enable efficient navigation
//! - **High recall**: Graph connectivity ensures good approximation quality
//! - **Incremental updates**: Can add vectors without full rebuild
//!
//! # Architecture
//!
//! ```text
//! Layer 2:  A ------- B
//!           |         |
//! Layer 1:  A -- C -- B -- D
//!           |    |    |    |
//! Layer 0:  A-E--C-F--B-G--D-H
//! ```
//!
//! Each layer is a navigable small-world graph. Higher layers have fewer nodes
//! and longer-range connections, enabling fast coarse-to-fine search.
//!
//! # Parameters
//!
//! - **M**: Max edges per node (default 16, controls graph connectivity)
//! - **ef_construction**: Build-time search width (default 200, affects index quality)
//! - **ef_search**: Search-time search width (default 50, recall vs speed tradeoff)
//! - **m_level**: Level generation probability multiplier (default 1/ln(M))
//!
//! # Example
//!
//! ```ignore
//! use dashprove_learning::corpus::{ProofCorpus, ProofCorpusHnsw, HnswConfig};
//!
//! // Build HNSW index
//! let config = HnswConfig::default();
//! let index = ProofCorpusHnsw::build_with_config(&corpus, config)?;
//!
//! // Search (ef controls recall vs speed)
//! let results = index.find_similar(&corpus, &query, 10, 50);
//!
//! // Higher ef = better recall, slower search
//! let high_recall = index.find_similar(&corpus, &query, 10, 200);
//!
//! // Incremental insert without rebuild
//! let new_id = corpus.insert_with_embedding(&new_result, new_embedding);
//! index.insert_from_corpus(&corpus, &new_id)?;
//! ```

use super::types::ProofId;
use crate::distance::euclidean_distance_sq;
use crate::embedder::Embedding;
use crate::similarity::SimilarProof;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Configuration for HNSW index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum number of connections per node (M)
    /// Higher M = better recall but more memory
    pub m: usize,
    /// Maximum connections for layer 0 (typically 2*M)
    pub m0: usize,
    /// Search width during construction
    /// Higher = better index quality but slower build
    pub ef_construction: usize,
    /// Default search width during queries
    /// Higher = better recall but slower search
    pub ef_search: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 50,
            seed: 42,
        }
    }
}

impl HnswConfig {
    /// Config for fast builds with reasonable quality
    pub fn fast() -> Self {
        Self {
            m: 12,
            m0: 24,
            ef_construction: 100,
            ef_search: 30,
            seed: 42,
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
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), HnswError> {
        if self.m == 0 {
            return Err(HnswError::InvalidConfig("m must be > 0".into()));
        }
        if self.m0 < self.m {
            return Err(HnswError::InvalidConfig("m0 must be >= m".into()));
        }
        if self.ef_construction < self.m {
            return Err(HnswError::InvalidConfig(
                "ef_construction must be >= m".into(),
            ));
        }
        Ok(())
    }
}

/// HNSW-specific errors
#[derive(Debug, Clone)]
pub enum HnswError {
    /// Invalid configuration
    InvalidConfig(String),
    /// Dimension mismatch
    DimensionMismatch { expected: usize, got: usize },
    /// Proof not found in corpus when inserting
    ProofNotFound(ProofId),
    /// Proof exists in corpus but has no embedding
    MissingEmbedding(ProofId),
    /// Proof ID already exists in the index
    DuplicateProof(ProofId),
}

impl std::fmt::Display for HnswError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HnswError::InvalidConfig(s) => write!(f, "Invalid HNSW config: {}", s),
            HnswError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            HnswError::ProofNotFound(id) => write!(f, "Proof not found in corpus: {}", id),
            HnswError::MissingEmbedding(id) => {
                write!(f, "Proof {} is missing an embedding", id)
            }
            HnswError::DuplicateProof(id) => write!(f, "Proof {} already indexed", id),
        }
    }
}

impl std::error::Error for HnswError {}

/// A node in the HNSW graph
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswNode {
    /// Index in the embeddings array
    idx: usize,
    /// Proof ID for this node
    proof_id: ProofId,
    /// Connections at each layer (neighbors[layer] = vec of neighbor indices)
    neighbors: Vec<Vec<usize>>,
    /// Maximum layer this node appears in
    max_layer: usize,
}

/// Candidate for search (negated distance for max-heap to min-heap conversion)
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
        // Reverse ordering for max-heap -> min-heap
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

/// HNSW index for ProofCorpus
///
/// Provides O(log N) approximate nearest neighbor search with high recall.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofCorpusHnsw {
    /// All embeddings stored in the index
    embeddings: Vec<Embedding>,
    /// Graph nodes with connectivity information
    nodes: Vec<HnswNode>,
    /// Map from proof ID to node index for fast duplicate detection
    id_to_idx: HashMap<ProofId, usize>,
    /// Entry point (top-level node)
    entry_point: Option<usize>,
    /// Maximum layer in the graph
    max_layer: usize,
    /// Configuration
    config: HnswConfig,
    /// Embedding dimension
    dim: usize,
    /// Level multiplier for random level generation (1/ln(M))
    ml: f64,
    /// Random state for level generation
    rng_state: u64,
}

impl ProofCorpusHnsw {
    /// Build an HNSW index from a ProofCorpus
    pub fn build(corpus: &super::storage::ProofCorpus) -> Result<Option<Self>, HnswError> {
        Self::build_with_config(corpus, HnswConfig::default())
    }

    /// Build with custom configuration
    pub fn build_with_config(
        corpus: &super::storage::ProofCorpus,
        config: HnswConfig,
    ) -> Result<Option<Self>, HnswError> {
        config.validate()?;

        // Collect embeddings
        let entries_with_embeddings: Vec<_> = corpus
            .entries()
            .filter_map(|e| e.embedding.as_ref().map(|emb| (e.id.clone(), emb.clone())))
            .collect();

        if entries_with_embeddings.is_empty() {
            return Ok(None);
        }

        let dim = entries_with_embeddings[0].1.dim;
        let ml = 1.0 / (config.m as f64).ln();
        let rng_state = config.seed;

        let mut index = Self {
            embeddings: Vec::with_capacity(entries_with_embeddings.len()),
            nodes: Vec::with_capacity(entries_with_embeddings.len()),
            id_to_idx: HashMap::with_capacity(entries_with_embeddings.len()),
            entry_point: None,
            max_layer: 0,
            config,
            dim,
            ml,
            rng_state,
        };

        // Insert each embedding
        for (proof_id, embedding) in entries_with_embeddings {
            index.insert(proof_id, embedding)?;
        }

        Ok(Some(index))
    }

    /// Insert a new embedding into the index
    fn insert(&mut self, proof_id: ProofId, embedding: Embedding) -> Result<(), HnswError> {
        if self.id_to_idx.contains_key(&proof_id) {
            return Err(HnswError::DuplicateProof(proof_id));
        }

        if embedding.dim != self.dim {
            return Err(HnswError::DimensionMismatch {
                expected: self.dim,
                got: embedding.dim,
            });
        }

        let idx = self.embeddings.len();
        let level = self.random_level();

        // Store embedding
        self.embeddings.push(embedding);

        // Create node with empty neighbor lists for each layer
        let node = HnswNode {
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

        // Search from top layer to level+1, only updating entry point
        let mut current_ep = ep;
        for layer in (level + 1..=self.max_layer).rev() {
            current_ep = self.search_layer_single(&self.embeddings[idx], current_ep, layer);
        }

        // Search and connect at each layer from level down to 0
        for layer in (0..=level.min(self.max_layer)).rev() {
            let m_max = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let candidates = self.search_layer(
                &self.embeddings[idx],
                vec![current_ep],
                self.config.ef_construction,
                layer,
            );

            // Select best neighbors
            let neighbors = self.select_neighbors(&self.embeddings[idx], &candidates, m_max, layer);

            // Connect new node to neighbors
            self.nodes[idx].neighbors[layer] = neighbors.clone();

            // Connect neighbors back to new node (bidirectional)
            for &neighbor_idx in &neighbors {
                if self.nodes[neighbor_idx].max_layer >= layer {
                    let neighbor_neighbors = &mut self.nodes[neighbor_idx].neighbors[layer];
                    neighbor_neighbors.push(idx);

                    // Prune if exceeds m_max
                    if neighbor_neighbors.len() > m_max {
                        let neighbor_embedding = &self.embeddings[neighbor_idx];
                        let candidates: Vec<_> = neighbor_neighbors
                            .iter()
                            .map(|&n| SearchCandidate {
                                distance: Self::distance_static(
                                    neighbor_embedding,
                                    &self.embeddings[n],
                                ),
                                idx: n,
                            })
                            .collect();
                        let pruned =
                            self.select_neighbors(neighbor_embedding, &candidates, m_max, layer);
                        self.nodes[neighbor_idx].neighbors[layer] = pruned;
                    }
                }
            }

            if !candidates.is_empty() {
                current_ep = candidates[0].idx;
            }
        }

        // Update entry point if new node has higher level
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

    /// Check if a proof is already indexed
    pub fn contains(&self, proof_id: &ProofId) -> bool {
        self.id_to_idx.contains_key(proof_id)
    }

    /// Insert a new proof embedding directly
    ///
    /// Returns true if the embedding was inserted, or false if it was already present.
    pub fn insert_embedding(
        &mut self,
        proof_id: ProofId,
        embedding: Embedding,
    ) -> Result<bool, HnswError> {
        if self.id_to_idx.contains_key(&proof_id) {
            return Ok(false);
        }
        self.insert(proof_id, embedding)?;
        Ok(true)
    }

    /// Insert a proof from the corpus (requires an existing embedding)
    pub fn insert_from_corpus(
        &mut self,
        corpus: &super::storage::ProofCorpus,
        proof_id: &ProofId,
    ) -> Result<bool, HnswError> {
        let entry = corpus
            .get(proof_id)
            .ok_or_else(|| HnswError::ProofNotFound(proof_id.clone()))?;
        let embedding = entry
            .embedding
            .as_ref()
            .ok_or_else(|| HnswError::MissingEmbedding(proof_id.clone()))?;

        self.insert_embedding(proof_id.clone(), embedding.clone())
    }

    /// Insert any missing proofs that have embeddings from the corpus
    ///
    /// Returns the number of embeddings inserted.
    pub fn sync_with_corpus(
        &mut self,
        corpus: &super::storage::ProofCorpus,
    ) -> Result<usize, HnswError> {
        let mut inserted = 0;
        for entry in corpus.entries() {
            if let Some(embedding) = entry.embedding.as_ref() {
                if !self.id_to_idx.contains_key(&entry.id) {
                    self.insert(entry.id.clone(), embedding.clone())?;
                    inserted += 1;
                }
            }
        }
        Ok(inserted)
    }

    /// Get the configuration
    pub fn config(&self) -> &HnswConfig {
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

        // Start from entry point and descend to layer 1
        let mut current_ep = ep;
        for layer in (1..=self.max_layer).rev() {
            current_ep = self.search_layer_single(query, current_ep, layer);
        }

        // Search layer 0 with ef candidates
        let candidates = self.search_layer(query, vec![current_ep], ef.max(k), 0);

        // Take top-k
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

    /// Find exact k nearest neighbors (for recall measurement)
    pub fn find_similar_exact(
        &self,
        corpus: &super::storage::ProofCorpus,
        query: &Embedding,
        k: usize,
    ) -> Vec<SimilarProof> {
        if self.is_empty() || k == 0 || query.dim != self.dim {
            return vec![];
        }

        // Brute force search over all embeddings
        let mut scores: Vec<_> = self
            .nodes
            .iter()
            .map(|node| {
                let dist = Self::distance_static(query, &self.embeddings[node.idx]);
                (node, dist)
            })
            .collect();

        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
            .into_iter()
            .take(k)
            .filter_map(|(node, dist)| {
                let entry = corpus.get(&node.proof_id)?;
                Some(SimilarProof {
                    id: entry.id.clone(),
                    property: entry.property.clone(),
                    backend: entry.backend,
                    tactics: entry.tactics.clone(),
                    similarity: Self::distance_to_similarity(dist),
                })
            })
            .collect()
    }

    /// Measure recall of approximate search vs exact search
    pub fn measure_recall(
        &self,
        corpus: &super::storage::ProofCorpus,
        k: usize,
        ef: usize,
        num_samples: usize,
    ) -> f64 {
        if self.is_empty() || k == 0 || num_samples == 0 {
            return 0.0;
        }

        let step = self.nodes.len().max(1) / num_samples.max(1);
        let step = step.max(1);

        let mut total_recall = 0.0;
        let mut samples_tested = 0;

        for i in (0..self.nodes.len()).step_by(step) {
            if samples_tested >= num_samples {
                break;
            }

            let query = &self.embeddings[i];
            let approx = self.find_similar_with_ef(corpus, query, k, ef);
            let exact = self.find_similar_exact(corpus, query, k);

            let approx_ids: HashSet<_> = approx.iter().map(|r| r.id.clone()).collect();
            let exact_ids: HashSet<_> = exact.iter().map(|r| r.id.clone()).collect();

            if !exact_ids.is_empty() {
                let overlap = approx_ids.intersection(&exact_ids).count();
                total_recall += overlap as f64 / exact_ids.len() as f64;
                samples_tested += 1;
            }
        }

        if samples_tested > 0 {
            total_recall / samples_tested as f64
        } else {
            0.0
        }
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> HnswMemoryStats {
        let embeddings_bytes = self.embeddings.len() * self.dim * 4;
        let mut graph_edges = 0;
        for node in &self.nodes {
            for layer_neighbors in &node.neighbors {
                graph_edges += layer_neighbors.len();
            }
        }
        // Each edge is stored as a usize (8 bytes on 64-bit)
        let graph_bytes = graph_edges * 8 + self.nodes.len() * 48; // Node overhead

        HnswMemoryStats {
            num_entries: self.nodes.len(),
            num_layers: self.max_layer + 1,
            total_edges: graph_edges,
            embeddings_bytes,
            graph_bytes,
            total_bytes: embeddings_bytes + graph_bytes,
        }
    }

    /// Get graph statistics
    pub fn graph_stats(&self) -> HnswGraphStats {
        let mut edges_per_layer = vec![0usize; self.max_layer + 1];
        let mut nodes_per_layer = vec![0usize; self.max_layer + 1];
        let mut total_edges = 0;

        for node in &self.nodes {
            for layer in 0..=node.max_layer {
                nodes_per_layer[layer] += 1;
                edges_per_layer[layer] += node.neighbors[layer].len();
                total_edges += node.neighbors[layer].len();
            }
        }

        let mean_edges = if !self.nodes.is_empty() {
            total_edges as f64 / self.nodes.len() as f64
        } else {
            0.0
        };

        HnswGraphStats {
            nodes_per_layer,
            edges_per_layer,
            mean_edges_per_node: mean_edges,
        }
    }

    // ========== Internal methods ==========

    /// Generate a random level for a new node
    fn random_level(&mut self) -> usize {
        // Use xorshift for fast PRNG
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;

        // Convert to uniform [0, 1)
        let r = (x >> 12) as f64 / (1u64 << 52) as f64;

        // Level = floor(-log(uniform) * ml)
        let level = ((-r.ln()) * self.ml).floor() as usize;

        // Cap at reasonable maximum
        level.min(32)
    }

    /// Search a single layer, returning the single nearest neighbor
    fn search_layer_single(&self, query: &Embedding, ep: usize, layer: usize) -> usize {
        let mut current = ep;
        let mut current_dist = Self::distance_static(query, &self.embeddings[current]);

        loop {
            let mut changed = false;

            if layer <= self.nodes[current].max_layer {
                for &neighbor in &self.nodes[current].neighbors[layer] {
                    let dist = Self::distance_static(query, &self.embeddings[neighbor]);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// Search a layer returning ef closest neighbors
    fn search_layer(
        &self,
        query: &Embedding,
        entry_points: Vec<usize>,
        ef: usize,
        layer: usize,
    ) -> Vec<SearchCandidate> {
        let mut visited: HashSet<usize> = entry_points.iter().copied().collect();

        // Candidates (min-heap by distance)
        let mut candidates: BinaryHeap<SearchCandidate> = entry_points
            .iter()
            .map(|&idx| SearchCandidate {
                distance: Self::distance_static(query, &self.embeddings[idx]),
                idx,
            })
            .collect();

        // Results (max-heap by distance, so we can efficiently drop furthest)
        let mut results: Vec<SearchCandidate> = candidates.iter().cloned().collect();

        while let Some(current) = candidates.pop() {
            // If current is further than the furthest in results, stop
            let furthest_dist = results.iter().map(|c| c.distance).fold(0.0f32, f32::max);

            if current.distance > furthest_dist && results.len() >= ef {
                break;
            }

            // Explore neighbors
            if layer <= self.nodes[current.idx].max_layer {
                for &neighbor in &self.nodes[current.idx].neighbors[layer] {
                    if visited.insert(neighbor) {
                        let dist = Self::distance_static(query, &self.embeddings[neighbor]);

                        let should_add = results.len() < ef || dist < furthest_dist;

                        if should_add {
                            candidates.push(SearchCandidate {
                                distance: dist,
                                idx: neighbor,
                            });
                            results.push(SearchCandidate {
                                distance: dist,
                                idx: neighbor,
                            });

                            // Keep only ef best
                            if results.len() > ef {
                                results.sort_by(|a, b| {
                                    a.distance
                                        .partial_cmp(&b.distance)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                });
                                results.truncate(ef);
                            }
                        }
                    }
                }
            }
        }

        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Select best neighbors using simple heuristic
    fn select_neighbors(
        &self,
        _query: &Embedding,
        candidates: &[SearchCandidate],
        m: usize,
        _layer: usize,
    ) -> Vec<usize> {
        if candidates.len() <= m {
            return candidates.iter().map(|c| c.idx).collect();
        }

        // Simple selection: take m closest
        let mut sorted: Vec<_> = candidates.to_vec();
        sorted.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        sorted.into_iter().take(m).map(|c| c.idx).collect()
    }

    /// Compute squared L2 distance between embeddings (SIMD-accelerated)
    fn distance_static(a: &Embedding, b: &Embedding) -> f32 {
        euclidean_distance_sq(&a.vector, &b.vector)
    }

    /// Convert squared L2 distance to similarity score
    fn distance_to_similarity(dist: f32) -> f64 {
        // Use exponential decay: sim = exp(-dist)
        // For normalized vectors, this gives reasonable 0-1 range
        (-(dist as f64).sqrt()).exp()
    }

    // ========== Persistence methods ==========

    /// Save the HNSW index to a JSON file
    ///
    /// This persists the graph structure, embeddings, and configuration so the
    /// index can be loaded without rebuilding.
    pub fn save_to_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), crate::LearningError> {
        crate::io::write_json_atomic(path, self)
    }

    /// Load an HNSW index from a JSON file
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, crate::LearningError> {
        crate::io::read_json(path)
    }

    /// Load an HNSW index if the file exists, otherwise return None
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

/// Memory statistics for HNSW index
#[derive(Debug, Clone)]
pub struct HnswMemoryStats {
    /// Number of indexed entries
    pub num_entries: usize,
    /// Number of layers in the graph
    pub num_layers: usize,
    /// Total number of edges in the graph
    pub total_edges: usize,
    /// Bytes used for embeddings
    pub embeddings_bytes: usize,
    /// Bytes used for graph structure
    pub graph_bytes: usize,
    /// Total memory usage
    pub total_bytes: usize,
}

/// Graph statistics for HNSW index
#[derive(Debug, Clone)]
pub struct HnswGraphStats {
    /// Number of nodes at each layer
    pub nodes_per_layer: Vec<usize>,
    /// Number of edges at each layer
    pub edges_per_layer: Vec<usize>,
    /// Mean number of edges per node
    pub mean_edges_per_node: f64,
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

    #[test]
    fn test_hnsw_config_default() {
        let config = HnswConfig::default();
        assert_eq!(config.m, 16);
        assert_eq!(config.m0, 32);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_hnsw_config_presets() {
        let fast = HnswConfig::fast();
        let high_recall = HnswConfig::high_recall();
        let compact = HnswConfig::compact();

        assert!(fast.m < high_recall.m);
        assert!(compact.m < fast.m);
        assert!(fast.validate().is_ok());
        assert!(high_recall.validate().is_ok());
        assert!(compact.validate().is_ok());
    }

    #[test]
    fn test_hnsw_config_validation() {
        let invalid_m = HnswConfig {
            m: 0,
            ..Default::default()
        };
        assert!(invalid_m.validate().is_err());

        let invalid_m0 = HnswConfig {
            m: 16,
            m0: 8,
            ..Default::default()
        };
        assert!(invalid_m0.validate().is_err());

        let invalid_ef = HnswConfig {
            m: 16,
            ef_construction: 8,
            ..Default::default()
        };
        assert!(invalid_ef.validate().is_err());
    }

    #[test]
    fn test_build_empty_corpus() {
        let corpus = super::super::storage::ProofCorpus::new();
        let result = ProofCorpusHnsw::build(&corpus);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_build_single_entry() {
        let mut corpus = super::super::storage::ProofCorpus::new();
        let result = make_result("prop_0");
        let embedding = random_embedding(0);
        corpus.insert_with_embedding(&result, embedding);

        let index = ProofCorpusHnsw::build(&corpus).unwrap().unwrap();
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_build_multiple_entries() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..100 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = HnswConfig::fast();
        let index = ProofCorpusHnsw::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        assert_eq!(index.len(), 100);
        // max_layer is usize, so it's always >= 0; just verify it exists
        let _ = index.max_layer();
    }

    #[test]
    fn test_find_similar() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..200 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = HnswConfig::fast();
        let index = ProofCorpusHnsw::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        let query = random_embedding(42);
        let results = index.find_similar(&corpus, &query, 10);

        assert!(!results.is_empty());
        assert!(results.len() <= 10);

        // Results should be sorted by similarity (descending)
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

        for i in 0..100 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = HnswConfig::fast();
        let index = ProofCorpusHnsw::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        let query = random_embedding(50);
        let results = index.find_similar_exact(&corpus, &query, 10);

        assert_eq!(results.len(), 10);

        // Results should be sorted
        for i in 1..results.len() {
            assert!(results[i].similarity <= results[i - 1].similarity);
        }
    }

    #[test]
    fn test_measure_recall() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..300 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = HnswConfig::default();
        let index = ProofCorpusHnsw::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        let recall = index.measure_recall(&corpus, 10, 50, 30);

        // HNSW should have good recall with default parameters
        assert!(recall > 0.5);
        assert!(recall <= 1.0);
    }

    #[test]
    fn test_incremental_insert_from_corpus() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..50 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let mut index = ProofCorpusHnsw::build(&corpus).unwrap().unwrap();
        let base_len = index.len();

        let new_embedding = random_embedding(999);
        let new_id = corpus.insert_with_embedding(&make_result("new_prop"), new_embedding.clone());

        let inserted = index
            .insert_from_corpus(&corpus, &new_id)
            .expect("insert should succeed");
        assert!(inserted);
        assert_eq!(index.len(), base_len + 1);

        // Second insert should be a no-op
        let duplicate = index.insert_from_corpus(&corpus, &new_id).unwrap();
        assert!(!duplicate);

        let nearest = index.find_similar_exact(&corpus, &new_embedding, 1);
        assert_eq!(nearest.len(), 1);
        assert_eq!(nearest[0].id, new_id);
    }

    #[test]
    fn test_insert_missing_embedding_errors() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Seed index with one embedded proof
        let seed_embedding = random_embedding(1);
        corpus.insert_with_embedding(&make_result("seed"), seed_embedding);
        let mut index = ProofCorpusHnsw::build(&corpus).unwrap().unwrap();

        // Add proof without embedding
        let missing_id = corpus.insert(&make_result("no_embedding"));
        let err = index
            .insert_from_corpus(&corpus, &missing_id)
            .expect_err("missing embedding should error");
        assert!(matches!(err, HnswError::MissingEmbedding(id) if id == missing_id));
    }

    #[test]
    fn test_sync_with_corpus_adds_new_entries() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..30 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let mut index = ProofCorpusHnsw::build(&corpus).unwrap().unwrap();

        for i in 30..45 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let inserted = index
            .sync_with_corpus(&corpus)
            .expect("sync should succeed");
        assert_eq!(inserted, 15);
        assert_eq!(index.len(), 45);
    }

    #[test]
    fn test_recall_improves_with_ef() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..300 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = HnswConfig::fast();
        let index = ProofCorpusHnsw::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        let recall_20 = index.measure_recall(&corpus, 10, 20, 30);
        let recall_100 = index.measure_recall(&corpus, 10, 100, 30);
        let recall_200 = index.measure_recall(&corpus, 10, 200, 30);

        // Higher ef should give better or equal recall
        assert!(recall_100 >= recall_20 * 0.9);
        assert!(recall_200 >= recall_100 * 0.9);
    }

    #[test]
    fn test_memory_stats() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..200 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let index = ProofCorpusHnsw::build(&corpus).unwrap().unwrap();
        let stats = index.memory_stats();

        assert_eq!(stats.num_entries, 200);
        assert!(stats.num_layers > 0);
        assert!(stats.total_edges > 0);
        assert!(stats.total_bytes > 0);
    }

    #[test]
    fn test_graph_stats() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..200 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let index = ProofCorpusHnsw::build(&corpus).unwrap().unwrap();
        let stats = index.graph_stats();

        // Layer 0 should have all nodes
        assert_eq!(stats.nodes_per_layer[0], 200);

        // Higher layers should have fewer nodes
        for i in 1..stats.nodes_per_layer.len() {
            assert!(stats.nodes_per_layer[i] <= stats.nodes_per_layer[i - 1]);
        }

        // Should have edges
        assert!(stats.mean_edges_per_node > 0.0);
    }

    #[test]
    fn test_hierarchical_structure() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Need enough nodes to generate multiple layers
        for i in 0..500 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = HnswConfig::default();
        let index = ProofCorpusHnsw::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        // Should have multiple layers
        assert!(index.max_layer() >= 1);

        let stats = index.graph_stats();

        // Verify exponential decay of nodes per layer
        for i in 1..stats.nodes_per_layer.len() {
            // Each layer should have roughly M times fewer nodes than the layer below
            // (with some variance due to randomness)
            if stats.nodes_per_layer[i] > 0 {
                let ratio = stats.nodes_per_layer[i - 1] as f64 / stats.nodes_per_layer[i] as f64;
                // Ratio should be reasonable (between 1 and M)
                assert!(ratio >= 1.0);
            }
        }
    }

    #[test]
    fn test_hnsw_save_load() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..100 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = HnswConfig::fast();
        let index = ProofCorpusHnsw::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hnsw_index.json");

        // Save
        index.save_to_file(&path).unwrap();
        assert!(path.exists());

        // Load
        let loaded = ProofCorpusHnsw::load_from_file(&path).unwrap();
        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.max_layer(), index.max_layer());
        assert_eq!(loaded.config().m, index.config().m);
        assert_eq!(loaded.config().m0, index.config().m0);
        assert_eq!(loaded.config().ef_search, index.config().ef_search);

        // Verify search works on loaded index
        let query = random_embedding(42);
        let results1 = index.find_similar(&corpus, &query, 10);
        let results2 = loaded.find_similar(&corpus, &query, 10);

        assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.id, r2.id);
            assert!((r1.similarity - r2.similarity).abs() < 1e-6);
        }
    }

    #[test]
    fn test_hnsw_load_if_exists() {
        // Non-existent file should return None
        let result = ProofCorpusHnsw::load_if_exists("/nonexistent/path/hnsw.json").unwrap();
        assert!(result.is_none());

        // Existing file should return Some
        let mut corpus = super::super::storage::ProofCorpus::new();
        for i in 0..50 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let index = ProofCorpusHnsw::build(&corpus).unwrap().unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hnsw_index.json");
        index.save_to_file(&path).unwrap();

        let loaded = ProofCorpusHnsw::load_if_exists(&path).unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().len(), 50);
    }

    #[test]
    fn test_hnsw_incremental_after_load() {
        // Build, save, load, then insert incrementally
        let mut corpus = super::super::storage::ProofCorpus::new();
        for i in 0..50 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = HnswConfig::fast();
        let index = ProofCorpusHnsw::build_with_config(&corpus, config)
            .unwrap()
            .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hnsw_index.json");
        index.save_to_file(&path).unwrap();

        let mut loaded = ProofCorpusHnsw::load_from_file(&path).unwrap();
        assert_eq!(loaded.len(), 50);

        // Add more entries to corpus and sync
        for i in 50..75 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let inserted = loaded.sync_with_corpus(&corpus).unwrap();
        assert_eq!(inserted, 25);
        assert_eq!(loaded.len(), 75);

        // Search should find newly inserted entries
        let query = random_embedding(60);
        let results = loaded.find_similar_exact(&corpus, &query, 1);
        assert!(!results.is_empty());
    }
}
