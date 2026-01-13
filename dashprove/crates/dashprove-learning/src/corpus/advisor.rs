//! Index selection advisor for choosing optimal vector search indices
//!
//! This module provides intelligent recommendations for which index type to use
//! based on corpus characteristics, memory budgets, and performance requirements.
//!
//! # Available Index Types
//!
//! | Index | Memory | Build Time | Query Time | Recall |
//! |-------|--------|------------|------------|--------|
//! | Exact | O(n) | O(1) | O(n) | 100% |
//! | LSH | O(n) | O(n) | O(1) | ~70-90% |
//! | PQ-LSH | O(n/48) | O(n·k) | O(1) | ~60-80% |
//! | OPQ-LSH | O(n/48) | O(n·k·i) | O(1) | ~65-85% |
//! | IVFPQ | O(n/48) | O(n·k) | O(sqrt(n)) | ~70-90% |
//! | IVFOPQ | O(n/48) | O(n·k·i) | O(sqrt(n)) | ~75-92% |
//! | HNSW | O(n) | O(n·log n) | O(log n) | ~95-99% |
//! | HNSW-PQ | O(n/48) | O(n·log n + k) | O(log n) | ~80-95% |
//! | HNSW-OPQ | O(n/48) | O(n·log n + k·i) | O(log n) | ~85-96% |
//! | HNSW-Binary | O(n/32) | O(n·log n) | O(log n) | ~60-80% |
//!
//! # Usage
//!
//! ```ignore
//! use dashprove_learning::corpus::{IndexAdvisor, IndexRequirements};
//!
//! let advisor = IndexAdvisor::new();
//!
//! // Define requirements
//! let requirements = IndexRequirements {
//!     corpus_size: 100_000,
//!     embedding_dim: 96,
//!     max_memory_mb: 50.0,
//!     min_recall: 0.85,
//!     max_build_time_secs: Some(60.0),
//!     max_query_time_ms: Some(10.0),
//! };
//!
//! // Get recommendation
//! let recommendation = advisor.recommend(&requirements);
//! println!("Recommended: {:?}", recommendation.index_type);
//! println!("Expected recall: {:.0}%", recommendation.expected_recall * 100.0);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

/// Available index types for vector search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexType {
    /// Exact brute-force search (100% recall, O(n) query time)
    Exact,
    /// Locality-Sensitive Hashing (fast approximate search)
    Lsh,
    /// Product Quantization + LSH (memory efficient)
    PqLsh,
    /// Optimized Product Quantization + LSH (better recall)
    OpqLsh,
    /// Inverted File + Product Quantization (billion-scale)
    IvfPq,
    /// Inverted File + Optimized Product Quantization (better recall)
    IvfOpq,
    /// Hierarchical Navigable Small World (high recall graph-based)
    Hnsw,
    /// HNSW + Product Quantization (memory efficient, high recall)
    HnswPq,
    /// HNSW + Optimized Product Quantization (best recall/memory tradeoff)
    HnswOpq,
    /// HNSW + Binary Quantization (extreme compression, fastest build)
    HnswBinary,
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexType::Exact => write!(f, "Exact"),
            IndexType::Lsh => write!(f, "LSH"),
            IndexType::PqLsh => write!(f, "PQ-LSH"),
            IndexType::OpqLsh => write!(f, "OPQ-LSH"),
            IndexType::IvfPq => write!(f, "IVFPQ"),
            IndexType::IvfOpq => write!(f, "IVFOPQ"),
            IndexType::Hnsw => write!(f, "HNSW"),
            IndexType::HnswPq => write!(f, "HNSW-PQ"),
            IndexType::HnswOpq => write!(f, "HNSW-OPQ"),
            IndexType::HnswBinary => write!(f, "HNSW-Binary"),
        }
    }
}

impl IndexType {
    /// All available index types
    pub fn all() -> &'static [IndexType] {
        &[
            IndexType::Exact,
            IndexType::Lsh,
            IndexType::PqLsh,
            IndexType::OpqLsh,
            IndexType::IvfPq,
            IndexType::IvfOpq,
            IndexType::Hnsw,
            IndexType::HnswPq,
            IndexType::HnswOpq,
            IndexType::HnswBinary,
        ]
    }

    /// Whether this index requires training data
    pub fn requires_training(&self) -> bool {
        matches!(
            self,
            IndexType::PqLsh
                | IndexType::OpqLsh
                | IndexType::IvfPq
                | IndexType::IvfOpq
                | IndexType::HnswPq
                | IndexType::HnswOpq
        )
    }

    /// Minimum corpus size for this index type
    pub fn min_corpus_size(&self) -> usize {
        match self {
            IndexType::Exact => 1,
            IndexType::Lsh => 100,
            IndexType::PqLsh | IndexType::OpqLsh => 256, // Need enough for PQ codebook training
            IndexType::IvfPq | IndexType::IvfOpq => 1000, // Need enough for IVF clustering
            IndexType::Hnsw => 100,
            IndexType::HnswPq | IndexType::HnswOpq => 256,
            IndexType::HnswBinary => 100,
        }
    }
}

/// Requirements for index selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRequirements {
    /// Expected corpus size (number of embeddings)
    pub corpus_size: usize,
    /// Embedding dimensionality
    pub embedding_dim: usize,
    /// Maximum memory budget in MB (None = unlimited)
    pub max_memory_mb: Option<f64>,
    /// Minimum acceptable recall@10 (0.0 to 1.0)
    pub min_recall: f64,
    /// Maximum acceptable build time in seconds (None = unlimited)
    pub max_build_time_secs: Option<f64>,
    /// Maximum acceptable query time in milliseconds (None = unlimited)
    pub max_query_time_ms: Option<f64>,
    /// Whether build time is more important than recall
    pub prioritize_build_time: bool,
    /// Whether query latency is critical (real-time applications)
    pub low_latency_required: bool,
    /// Whether incremental updates are needed (vs full rebuilds)
    pub incremental_updates: bool,
}

impl Default for IndexRequirements {
    fn default() -> Self {
        Self {
            corpus_size: 1000,
            embedding_dim: 96,
            max_memory_mb: None,
            min_recall: 0.8,
            max_build_time_secs: None,
            max_query_time_ms: None,
            prioritize_build_time: false,
            low_latency_required: false,
            incremental_updates: false,
        }
    }
}

impl IndexRequirements {
    /// Create requirements for a small corpus (<1K embeddings)
    pub fn for_small_corpus(size: usize) -> Self {
        Self {
            corpus_size: size,
            min_recall: 0.95, // High recall is cheap for small corpora
            ..Default::default()
        }
    }

    /// Create requirements for a medium corpus (1K-100K embeddings)
    pub fn for_medium_corpus(size: usize) -> Self {
        Self {
            corpus_size: size,
            min_recall: 0.85,
            max_memory_mb: Some(100.0),
            ..Default::default()
        }
    }

    /// Create requirements for a large corpus (100K-1M embeddings)
    pub fn for_large_corpus(size: usize) -> Self {
        Self {
            corpus_size: size,
            min_recall: 0.80,
            max_memory_mb: Some(500.0),
            max_query_time_ms: Some(50.0),
            ..Default::default()
        }
    }

    /// Create requirements for a very large corpus (>1M embeddings)
    pub fn for_billion_scale(size: usize) -> Self {
        Self {
            corpus_size: size,
            min_recall: 0.75,
            max_memory_mb: Some(2000.0),
            max_query_time_ms: Some(100.0),
            low_latency_required: true,
            ..Default::default()
        }
    }

    /// Create requirements prioritizing fast builds
    pub fn fast_build(size: usize) -> Self {
        Self {
            corpus_size: size,
            min_recall: 0.70,
            prioritize_build_time: true,
            max_build_time_secs: Some(60.0),
            ..Default::default()
        }
    }
}

/// Estimated characteristics of an index for given requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEstimates {
    /// Estimated memory usage in MB
    pub memory_mb: f64,
    /// Estimated build time in seconds
    pub build_time_secs: f64,
    /// Estimated query time in milliseconds
    pub query_time_ms: f64,
    /// Expected recall@10 (0.0 to 1.0)
    pub expected_recall: f64,
}

/// A recommendation from the advisor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRecommendation {
    /// The recommended index type
    pub index_type: IndexType,
    /// Estimated performance characteristics
    pub estimates: IndexEstimates,
    /// Overall suitability score (0.0 to 1.0)
    pub score: f64,
    /// Reasons for this recommendation
    pub reasons: Vec<String>,
    /// Warnings or caveats
    pub warnings: Vec<String>,
}

impl IndexRecommendation {
    /// Check if this recommendation meets all requirements
    pub fn meets_requirements(&self, reqs: &IndexRequirements) -> bool {
        // Check memory
        if let Some(max_mem) = reqs.max_memory_mb {
            if self.estimates.memory_mb > max_mem {
                return false;
            }
        }

        // Check build time
        if let Some(max_build) = reqs.max_build_time_secs {
            if self.estimates.build_time_secs > max_build {
                return false;
            }
        }

        // Check query time
        if let Some(max_query) = reqs.max_query_time_ms {
            if self.estimates.query_time_ms > max_query {
                return false;
            }
        }

        // Check recall
        if self.estimates.expected_recall < reqs.min_recall {
            return false;
        }

        true
    }
}

/// Index selection advisor
#[derive(Debug, Clone, Default)]
pub struct IndexAdvisor {
    /// Calibration factors from actual benchmarks (optional)
    calibration: Option<CalibrationData>,
}

/// Calibration data from actual benchmark runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationData {
    /// Memory factor multiplier
    pub memory_factor: f64,
    /// Build time factor multiplier
    pub build_time_factor: f64,
    /// Query time factor multiplier
    pub query_time_factor: f64,
}

impl Default for CalibrationData {
    fn default() -> Self {
        Self {
            memory_factor: 1.0,
            build_time_factor: 1.0,
            query_time_factor: 1.0,
        }
    }
}

impl IndexAdvisor {
    /// Create a new advisor with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an advisor with calibration data from benchmarks
    pub fn with_calibration(calibration: CalibrationData) -> Self {
        Self {
            calibration: Some(calibration),
        }
    }

    /// Estimate index characteristics for a given type and requirements
    pub fn estimate(&self, index_type: IndexType, reqs: &IndexRequirements) -> IndexEstimates {
        let n = reqs.corpus_size as f64;
        let d = reqs.embedding_dim as f64;

        // Base embedding size in bytes: d * 4 (f32)
        let embedding_bytes = d * 4.0;

        // Apply calibration if available
        let cal = self.calibration.as_ref().cloned().unwrap_or_default();

        let (memory_mb, build_time_secs, query_time_ms, expected_recall) = match index_type {
            IndexType::Exact => {
                // Memory: n * d * 4 bytes
                let mem = (n * embedding_bytes) / (1024.0 * 1024.0);
                // Build: O(1) - just storing
                let build = n * 0.00001; // ~10µs per embedding
                                         // Query: O(n) - scan all
                let query = n * 0.001; // ~1µs per comparison
                (mem, build, query, 1.0)
            }
            IndexType::Lsh => {
                // Memory: embeddings + hash tables (~1.2x)
                let mem = (n * embedding_bytes * 1.2) / (1024.0 * 1024.0);
                // Build: O(n) - hash all embeddings
                let build = n * 0.0001; // ~100µs per embedding
                                        // Query: O(hash_tables * bucket_size) ≈ O(1) amortized
                let query = 0.5 + n.ln() * 0.1;
                let recall = (0.75 + 0.1 * (n.ln() / 10.0).min(1.0)).min(0.90);
                (mem, build, query, recall)
            }
            IndexType::PqLsh => {
                // Memory: PQ codes (~48x compression) + LSH tables
                let pq_compression = 48.0;
                let mem = (n * embedding_bytes / pq_compression * 1.3) / (1024.0 * 1024.0);
                // Build: O(n * k-means iterations)
                let build = n * 0.001 + 256.0 * 10.0 * 0.1; // Training overhead
                                                            // Query: O(1) amortized
                let query = 1.0 + n.ln() * 0.1;
                (mem, build, query, 0.70)
            }
            IndexType::OpqLsh => {
                // Memory: OPQ codes + rotation matrix
                let pq_compression = 48.0;
                let rotation_bytes = d * d * 4.0;
                let mem =
                    (n * embedding_bytes / pq_compression + rotation_bytes) / (1024.0 * 1024.0);
                // Build: O(n * k-means iterations * OPQ iterations)
                let build = n * 0.001 + 256.0 * 10.0 * 5.0 * 0.1; // More training
                                                                  // Query: O(1) amortized
                let query = 1.2 + n.ln() * 0.1;
                (mem, build, query, 0.75)
            }
            IndexType::IvfPq => {
                // Memory: PQ codes + centroids
                let pq_compression = 48.0;
                let n_lists = (n.sqrt() * 4.0).clamp(16.0, 4096.0);
                let centroid_bytes = n_lists * d * 4.0;
                let mem =
                    (n * embedding_bytes / pq_compression + centroid_bytes) / (1024.0 * 1024.0);
                // Build: O(n * k-means for IVF + PQ)
                let build = n * 0.002 + n_lists * 20.0 * 0.1;
                // Query: O(nprobe * n/n_lists) ≈ O(sqrt(n))
                let nprobe = 16.0;
                let query = nprobe * (n / n_lists) * 0.001 + 2.0;
                let recall = (0.80 + 0.05 * (nprobe / 16.0).min(1.0)).min(0.90);
                (mem, build, query, recall)
            }
            IndexType::IvfOpq => {
                // Memory: OPQ codes + centroids + rotation
                let pq_compression = 48.0;
                let n_lists = (n.sqrt() * 4.0).clamp(16.0, 4096.0);
                let centroid_bytes = n_lists * d * 4.0;
                let rotation_bytes = d * d * 4.0;
                let mem = (n * embedding_bytes / pq_compression + centroid_bytes + rotation_bytes)
                    / (1024.0 * 1024.0);
                // Build: more training for OPQ
                let build = n * 0.003 + n_lists * 20.0 * 0.1 + 256.0 * 10.0 * 5.0 * 0.1;
                // Query: same as IVFPQ
                let nprobe = 16.0;
                let query = nprobe * (n / n_lists) * 0.001 + 2.5;
                let recall = (0.82 + 0.05 * (nprobe / 16.0).min(1.0)).min(0.92);
                (mem, build, query, recall)
            }
            IndexType::Hnsw => {
                // Memory: embeddings + graph edges (~M * 2 * 8 bytes per node)
                let m = 16.0;
                let graph_bytes = n * m * 2.0 * 8.0;
                let mem = (n * embedding_bytes + graph_bytes) / (1024.0 * 1024.0);
                // Build: O(n * log n * ef_construction)
                let ef_construction = 200.0;
                let build = n * n.ln() * ef_construction * 0.00001;
                // Query: O(log n * ef_search)
                let ef_search = 50.0;
                let query = n.ln() * ef_search * 0.01;
                let recall = (0.95 + 0.03 * (ef_search / 100.0).min(1.0)).min(0.99);
                (mem, build, query, recall)
            }
            IndexType::HnswPq => {
                // Memory: PQ codes + graph
                let pq_compression = 48.0;
                let m = 16.0;
                let graph_bytes = n * m * 2.0 * 8.0;
                let mem = (n * embedding_bytes / pq_compression + graph_bytes) / (1024.0 * 1024.0);
                // Build: HNSW build (needs full embeddings) + PQ training
                let ef_construction = 200.0;
                let build = n * n.ln() * ef_construction * 0.00001 + 256.0 * 10.0 * 0.1;
                // Query: O(log n) with PQ distance
                let ef_search = 50.0;
                let query = n.ln() * ef_search * 0.015;
                let recall = (0.85 + 0.05 * (ef_search / 100.0).min(1.0)).min(0.95);
                (mem, build, query, recall)
            }
            IndexType::HnswOpq => {
                // Memory: OPQ codes + graph + rotation
                let pq_compression = 48.0;
                let m = 16.0;
                let graph_bytes = n * m * 2.0 * 8.0;
                let rotation_bytes = d * d * 4.0;
                let mem = (n * embedding_bytes / pq_compression + graph_bytes + rotation_bytes)
                    / (1024.0 * 1024.0);
                // Build: HNSW + OPQ training
                let ef_construction = 200.0;
                let build = n * n.ln() * ef_construction * 0.00001 + 256.0 * 10.0 * 5.0 * 0.1;
                // Query: O(log n) with OPQ distance
                let ef_search = 50.0;
                let query = n.ln() * ef_search * 0.018;
                let recall = (0.88 + 0.05 * (ef_search / 100.0).min(1.0)).min(0.96);
                (mem, build, query, recall)
            }
            IndexType::HnswBinary => {
                // Memory: binary codes (~32x compression) + graph
                let binary_compression = 32.0;
                let m = 16.0;
                let graph_bytes = n * m * 2.0 * 8.0;
                let mem =
                    (n * embedding_bytes / binary_compression + graph_bytes) / (1024.0 * 1024.0);
                // Build: HNSW build only (no training!)
                let ef_construction = 200.0;
                let build = n * n.ln() * ef_construction * 0.000008; // Faster - no training
                                                                     // Query: O(log n) with Hamming distance (very fast)
                let ef_search = 50.0;
                let query = n.ln() * ef_search * 0.008; // Hamming is faster than PQ
                let recall = (0.65 + 0.10 * (ef_search / 100.0).min(1.0)).min(0.80);
                (mem, build, query, recall)
            }
        };

        IndexEstimates {
            memory_mb: memory_mb * cal.memory_factor,
            build_time_secs: build_time_secs * cal.build_time_factor,
            query_time_ms: query_time_ms * cal.query_time_factor,
            expected_recall,
        }
    }

    /// Score an index type for the given requirements
    fn score_index(&self, index_type: IndexType, reqs: &IndexRequirements) -> (f64, Vec<String>) {
        let estimates = self.estimate(index_type, reqs);
        let mut score = 0.0;
        let mut reasons = Vec::new();

        // Check minimum corpus size
        if reqs.corpus_size < index_type.min_corpus_size() {
            return (
                0.0,
                vec![format!(
                    "Corpus too small (need {} for {})",
                    index_type.min_corpus_size(),
                    index_type
                )],
            );
        }

        // Score memory efficiency (higher = better)
        if let Some(max_mem) = reqs.max_memory_mb {
            if estimates.memory_mb > max_mem {
                return (
                    0.0,
                    vec![format!(
                        "Exceeds memory budget ({:.1} MB > {:.1} MB)",
                        estimates.memory_mb, max_mem
                    )],
                );
            }
            let mem_efficiency = 1.0 - (estimates.memory_mb / max_mem).min(1.0);
            score += mem_efficiency * 0.2;
            reasons.push(format!(
                "Memory: {:.1} MB ({:.0}% of budget)",
                estimates.memory_mb,
                estimates.memory_mb / max_mem * 100.0
            ));
        } else {
            // Without budget, prefer memory-efficient indices slightly
            let mem_penalty = (estimates.memory_mb / 1000.0).min(1.0) * 0.1;
            score += (1.0 - mem_penalty) * 0.1;
        }

        // Score build time
        if let Some(max_build) = reqs.max_build_time_secs {
            if estimates.build_time_secs > max_build {
                return (
                    0.0,
                    vec![format!(
                        "Exceeds build time budget ({:.1}s > {:.1}s)",
                        estimates.build_time_secs, max_build
                    )],
                );
            }
            let build_efficiency = 1.0 - (estimates.build_time_secs / max_build).min(1.0);
            let build_weight = if reqs.prioritize_build_time {
                0.3
            } else {
                0.15
            };
            score += build_efficiency * build_weight;
            reasons.push(format!("Build: {:.1}s", estimates.build_time_secs));
        }

        // Score query time
        if let Some(max_query) = reqs.max_query_time_ms {
            if estimates.query_time_ms > max_query {
                return (
                    0.0,
                    vec![format!(
                        "Exceeds query time budget ({:.1}ms > {:.1}ms)",
                        estimates.query_time_ms, max_query
                    )],
                );
            }
            let query_efficiency = 1.0 - (estimates.query_time_ms / max_query).min(1.0);
            let query_weight = if reqs.low_latency_required { 0.3 } else { 0.15 };
            score += query_efficiency * query_weight;
            reasons.push(format!("Query: {:.2}ms", estimates.query_time_ms));
        }

        // Score recall (most important)
        if estimates.expected_recall < reqs.min_recall {
            return (
                0.0,
                vec![format!(
                    "Below minimum recall ({:.0}% < {:.0}%)",
                    estimates.expected_recall * 100.0,
                    reqs.min_recall * 100.0
                )],
            );
        }
        let recall_margin = estimates.expected_recall - reqs.min_recall;
        score += recall_margin * 0.4; // Recall is weighted heavily
        reasons.push(format!("Recall: {:.0}%", estimates.expected_recall * 100.0));

        // Bonus for incremental updates if needed
        if reqs.incremental_updates {
            match index_type {
                IndexType::Hnsw
                | IndexType::HnswPq
                | IndexType::HnswOpq
                | IndexType::HnswBinary => {
                    score += 0.1;
                    reasons.push("Supports incremental updates".to_string());
                }
                IndexType::Lsh => {
                    score += 0.05;
                    reasons.push("Partial incremental support".to_string());
                }
                _ => {
                    score -= 0.05;
                    reasons.push("No incremental update support".to_string());
                }
            }
        }

        (score.clamp(0.0, 1.0), reasons)
    }

    /// Get the best recommendation for the given requirements
    pub fn recommend(&self, reqs: &IndexRequirements) -> IndexRecommendation {
        let recommendations = self.recommend_all(reqs);
        recommendations.into_iter().next().unwrap_or_else(|| {
            // Fallback to exact search if nothing else works
            let estimates = self.estimate(IndexType::Exact, reqs);
            IndexRecommendation {
                index_type: IndexType::Exact,
                estimates,
                score: 0.1,
                reasons: vec!["Fallback: exact search always works".to_string()],
                warnings: vec!["No index met all requirements".to_string()],
            }
        })
    }

    /// Get all recommendations sorted by score
    pub fn recommend_all(&self, reqs: &IndexRequirements) -> Vec<IndexRecommendation> {
        let mut recommendations: Vec<IndexRecommendation> = IndexType::all()
            .iter()
            .filter_map(|&index_type| {
                let (score, reasons) = self.score_index(index_type, reqs);
                if score > 0.0 {
                    let estimates = self.estimate(index_type, reqs);
                    let warnings = self.get_warnings(index_type, reqs, &estimates);
                    Some(IndexRecommendation {
                        index_type,
                        estimates,
                        score,
                        reasons,
                        warnings,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending
        recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        recommendations
    }

    /// Get warnings for a recommendation
    fn get_warnings(
        &self,
        index_type: IndexType,
        reqs: &IndexRequirements,
        estimates: &IndexEstimates,
    ) -> Vec<String> {
        let mut warnings = Vec::new();

        // Training data warnings
        if index_type.requires_training() && reqs.corpus_size < 1000 {
            warnings.push(format!(
                "Small corpus ({}) may result in poor codebook quality",
                reqs.corpus_size
            ));
        }

        // Memory warnings
        if estimates.memory_mb > 1000.0 {
            warnings.push(format!(
                "Large memory footprint: {:.0} MB",
                estimates.memory_mb
            ));
        }

        // Build time warnings
        if estimates.build_time_secs > 300.0 {
            warnings.push(format!(
                "Long build time: {:.0} seconds",
                estimates.build_time_secs
            ));
        }

        // Recall warnings
        if estimates.expected_recall < 0.8 {
            warnings.push(format!(
                "Low expected recall: {:.0}%",
                estimates.expected_recall * 100.0
            ));
        }

        // Binary quantization warning
        if index_type == IndexType::HnswBinary {
            warnings.push("Binary quantization trades recall for speed".to_string());
        }

        warnings
    }

    /// Quick recommendation based on corpus size alone
    pub fn quick_recommend(corpus_size: usize) -> IndexType {
        match corpus_size {
            0..=500 => IndexType::Exact,
            501..=5_000 => IndexType::Lsh,
            5_001..=50_000 => IndexType::Hnsw,
            50_001..=500_000 => IndexType::HnswPq,
            _ => IndexType::IvfPq,
        }
    }

    /// Compare two index types for the given requirements
    pub fn compare(
        &self,
        index_a: IndexType,
        index_b: IndexType,
        reqs: &IndexRequirements,
    ) -> IndexComparison {
        let est_a = self.estimate(index_a, reqs);
        let est_b = self.estimate(index_b, reqs);

        IndexComparison {
            index_a,
            index_b,
            estimates_a: est_a.clone(),
            estimates_b: est_b.clone(),
            memory_ratio: est_a.memory_mb / est_b.memory_mb,
            build_time_ratio: est_a.build_time_secs / est_b.build_time_secs,
            query_time_ratio: est_a.query_time_ms / est_b.query_time_ms,
            recall_diff: est_a.expected_recall - est_b.expected_recall,
        }
    }
}

/// Comparison between two index types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexComparison {
    /// First index type
    pub index_a: IndexType,
    /// Second index type
    pub index_b: IndexType,
    /// Estimates for first index
    pub estimates_a: IndexEstimates,
    /// Estimates for second index
    pub estimates_b: IndexEstimates,
    /// Memory ratio (a/b): <1 means A uses less memory
    pub memory_ratio: f64,
    /// Build time ratio (a/b): <1 means A builds faster
    pub build_time_ratio: f64,
    /// Query time ratio (a/b): <1 means A queries faster
    pub query_time_ratio: f64,
    /// Recall difference (a-b): >0 means A has better recall
    pub recall_diff: f64,
}

impl IndexComparison {
    /// Get a summary of which index is better for what
    pub fn summary(&self) -> String {
        let mut better_a = Vec::new();
        let mut better_b = Vec::new();

        if self.memory_ratio < 1.0 {
            better_a.push("memory");
        } else if self.memory_ratio > 1.0 {
            better_b.push("memory");
        }

        if self.build_time_ratio < 1.0 {
            better_a.push("build time");
        } else if self.build_time_ratio > 1.0 {
            better_b.push("build time");
        }

        if self.query_time_ratio < 1.0 {
            better_a.push("query speed");
        } else if self.query_time_ratio > 1.0 {
            better_b.push("query speed");
        }

        if self.recall_diff > 0.0 {
            better_a.push("recall");
        } else if self.recall_diff < 0.0 {
            better_b.push("recall");
        }

        let a_str = if better_a.is_empty() {
            "nothing".to_string()
        } else {
            better_a.join(", ")
        };
        let b_str = if better_b.is_empty() {
            "nothing".to_string()
        } else {
            better_b.join(", ")
        };

        format!(
            "{} is better for: {}; {} is better for: {}",
            self.index_a, a_str, self.index_b, b_str
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_type_display() {
        assert_eq!(format!("{}", IndexType::Exact), "Exact");
        assert_eq!(format!("{}", IndexType::Hnsw), "HNSW");
        assert_eq!(format!("{}", IndexType::HnswBinary), "HNSW-Binary");
    }

    #[test]
    fn test_index_type_all() {
        let all = IndexType::all();
        assert_eq!(all.len(), 10);
        assert!(all.contains(&IndexType::Exact));
        assert!(all.contains(&IndexType::HnswBinary));
    }

    #[test]
    fn test_index_requires_training() {
        assert!(!IndexType::Exact.requires_training());
        assert!(!IndexType::Lsh.requires_training());
        assert!(!IndexType::Hnsw.requires_training());
        assert!(!IndexType::HnswBinary.requires_training());
        assert!(IndexType::PqLsh.requires_training());
        assert!(IndexType::HnswPq.requires_training());
        assert!(IndexType::IvfPq.requires_training());
    }

    #[test]
    fn test_min_corpus_size() {
        assert_eq!(IndexType::Exact.min_corpus_size(), 1);
        assert_eq!(IndexType::Lsh.min_corpus_size(), 100);
        assert_eq!(IndexType::PqLsh.min_corpus_size(), 256);
        assert_eq!(IndexType::IvfPq.min_corpus_size(), 1000);
    }

    #[test]
    fn test_default_requirements() {
        let reqs = IndexRequirements::default();
        assert_eq!(reqs.corpus_size, 1000);
        assert_eq!(reqs.embedding_dim, 96);
        assert_eq!(reqs.min_recall, 0.8);
    }

    #[test]
    fn test_requirements_presets() {
        let small = IndexRequirements::for_small_corpus(500);
        assert_eq!(small.corpus_size, 500);
        assert_eq!(small.min_recall, 0.95);

        let large = IndexRequirements::for_large_corpus(100_000);
        assert_eq!(large.corpus_size, 100_000);
        assert!(large.max_memory_mb.is_some());

        let fast = IndexRequirements::fast_build(50_000);
        assert!(fast.prioritize_build_time);
    }

    #[test]
    fn test_advisor_estimate_exact() {
        let advisor = IndexAdvisor::new();
        let reqs = IndexRequirements {
            corpus_size: 1000,
            embedding_dim: 96,
            ..Default::default()
        };

        let est = advisor.estimate(IndexType::Exact, &reqs);
        assert!(est.memory_mb > 0.0);
        assert_eq!(est.expected_recall, 1.0); // Exact is always 100%
        assert!(est.query_time_ms > 0.0);
    }

    #[test]
    fn test_advisor_estimate_hnsw_better_recall_than_binary() {
        let advisor = IndexAdvisor::new();
        let reqs = IndexRequirements {
            corpus_size: 10_000,
            embedding_dim: 96,
            ..Default::default()
        };

        let hnsw = advisor.estimate(IndexType::Hnsw, &reqs);
        let binary = advisor.estimate(IndexType::HnswBinary, &reqs);

        assert!(hnsw.expected_recall > binary.expected_recall);
    }

    #[test]
    fn test_advisor_estimate_compression_indices_use_less_memory() {
        let advisor = IndexAdvisor::new();
        let reqs = IndexRequirements {
            corpus_size: 100_000,
            embedding_dim: 96,
            ..Default::default()
        };

        let exact = advisor.estimate(IndexType::Exact, &reqs);
        let hnsw_pq = advisor.estimate(IndexType::HnswPq, &reqs);
        let ivfpq = advisor.estimate(IndexType::IvfPq, &reqs);

        assert!(hnsw_pq.memory_mb < exact.memory_mb);
        assert!(ivfpq.memory_mb < exact.memory_mb);
    }

    #[test]
    fn test_advisor_recommend_small_corpus() {
        let advisor = IndexAdvisor::new();
        let reqs = IndexRequirements::for_small_corpus(100);

        let rec = advisor.recommend(&reqs);
        // For small corpus with high recall requirement, Exact or HNSW should be preferred
        assert!(
            rec.index_type == IndexType::Exact
                || rec.index_type == IndexType::Hnsw
                || rec.index_type == IndexType::Lsh
        );
    }

    #[test]
    fn test_advisor_recommend_large_corpus() {
        let advisor = IndexAdvisor::new();
        let reqs = IndexRequirements::for_large_corpus(500_000);

        let rec = advisor.recommend(&reqs);
        // For large corpus with memory constraint, compressed indices should be preferred
        assert!(
            rec.index_type == IndexType::HnswPq
                || rec.index_type == IndexType::HnswOpq
                || rec.index_type == IndexType::IvfPq
                || rec.index_type == IndexType::IvfOpq
        );
    }

    #[test]
    fn test_advisor_recommend_fast_build() {
        let advisor = IndexAdvisor::new();
        let reqs = IndexRequirements::fast_build(50_000);

        let rec = advisor.recommend(&reqs);
        // For fast build, HNSW-Binary (no training) or LSH should be preferred
        // But since recall requirement is relaxed, many options work
        assert!(rec.estimates.build_time_secs < 60.0 || !rec.warnings.is_empty());
    }

    #[test]
    fn test_advisor_recommend_all() {
        let advisor = IndexAdvisor::new();
        let reqs = IndexRequirements {
            corpus_size: 10_000,
            embedding_dim: 96,
            min_recall: 0.7, // Low recall to get more options
            ..Default::default()
        };

        let recs = advisor.recommend_all(&reqs);
        assert!(!recs.is_empty());
        // Should be sorted by score descending
        for i in 1..recs.len() {
            assert!(recs[i - 1].score >= recs[i].score);
        }
    }

    #[test]
    fn test_advisor_meets_requirements() {
        let advisor = IndexAdvisor::new();
        let reqs = IndexRequirements {
            corpus_size: 10_000,
            embedding_dim: 96,
            max_memory_mb: Some(100.0),
            min_recall: 0.8,
            ..Default::default()
        };

        let rec = advisor.recommend(&reqs);
        assert!(rec.meets_requirements(&reqs));
    }

    #[test]
    fn test_advisor_quick_recommend() {
        assert_eq!(IndexAdvisor::quick_recommend(100), IndexType::Exact);
        assert_eq!(IndexAdvisor::quick_recommend(1000), IndexType::Lsh);
        assert_eq!(IndexAdvisor::quick_recommend(10_000), IndexType::Hnsw);
        assert_eq!(IndexAdvisor::quick_recommend(100_000), IndexType::HnswPq);
        assert_eq!(IndexAdvisor::quick_recommend(1_000_000), IndexType::IvfPq);
    }

    #[test]
    fn test_advisor_compare() {
        let advisor = IndexAdvisor::new();
        let reqs = IndexRequirements {
            corpus_size: 50_000,
            embedding_dim: 96,
            ..Default::default()
        };

        let cmp = advisor.compare(IndexType::Hnsw, IndexType::HnswPq, &reqs);
        // HNSW uses more memory but has better recall
        assert!(cmp.memory_ratio > 1.0); // HNSW uses more memory
        assert!(cmp.recall_diff > 0.0); // HNSW has better recall

        let summary = cmp.summary();
        assert!(summary.contains("HNSW"));
        assert!(summary.contains("HNSW-PQ"));
    }

    #[test]
    fn test_advisor_with_calibration() {
        let cal = CalibrationData {
            memory_factor: 1.2,     // 20% more memory than estimated
            build_time_factor: 0.8, // 20% faster than estimated
            query_time_factor: 1.0,
        };
        let advisor = IndexAdvisor::with_calibration(cal);

        let reqs = IndexRequirements::default();
        let est = advisor.estimate(IndexType::Hnsw, &reqs);

        // Calibration should affect estimates
        let default_advisor = IndexAdvisor::new();
        let default_est = default_advisor.estimate(IndexType::Hnsw, &reqs);

        assert!(est.memory_mb > default_est.memory_mb);
        assert!(est.build_time_secs < default_est.build_time_secs);
    }

    #[test]
    fn test_recommendation_warnings() {
        let advisor = IndexAdvisor::new();
        let reqs = IndexRequirements {
            corpus_size: 500, // Small corpus
            embedding_dim: 96,
            min_recall: 0.6, // Allow low recall
            ..Default::default()
        };

        let recs = advisor.recommend_all(&reqs);
        // Find HNSW-Binary recommendation
        let binary_rec = recs.iter().find(|r| r.index_type == IndexType::HnswBinary);
        if let Some(rec) = binary_rec {
            // Should have warning about recall tradeoff
            assert!(rec
                .warnings
                .iter()
                .any(|w| w.contains("recall") || w.contains("Binary")));
        }
    }

    #[test]
    fn test_recommendation_reasons() {
        let advisor = IndexAdvisor::new();
        let reqs = IndexRequirements::default();

        let rec = advisor.recommend(&reqs);
        assert!(!rec.reasons.is_empty());
        // Should have reason about recall
        assert!(rec.reasons.iter().any(|r| r.contains("Recall")));
    }

    #[test]
    fn test_no_recommendation_fallback() {
        let advisor = IndexAdvisor::new();
        // Very restrictive requirements
        let reqs = IndexRequirements {
            corpus_size: 10_000,
            embedding_dim: 96,
            max_memory_mb: Some(0.001), // Impossibly small
            min_recall: 0.99,
            max_build_time_secs: Some(0.001),
            max_query_time_ms: Some(0.001),
            ..Default::default()
        };

        let rec = advisor.recommend(&reqs);
        // Should fall back to Exact
        assert_eq!(rec.index_type, IndexType::Exact);
        assert!(!rec.warnings.is_empty());
    }

    #[test]
    fn test_corpus_too_small_for_index() {
        let advisor = IndexAdvisor::new();
        let reqs = IndexRequirements {
            corpus_size: 50, // Too small for IVF
            embedding_dim: 96,
            min_recall: 0.7,
            ..Default::default()
        };

        let recs = advisor.recommend_all(&reqs);
        // IVFPQ and IVFOPQ should not be recommended (need 1000+ entries)
        assert!(!recs.iter().any(|r| r.index_type == IndexType::IvfPq));
        assert!(!recs.iter().any(|r| r.index_type == IndexType::IvfOpq));
    }

    #[test]
    fn test_incremental_updates_preference() {
        let advisor = IndexAdvisor::new();
        let reqs = IndexRequirements {
            corpus_size: 10_000,
            embedding_dim: 96,
            min_recall: 0.7,
            incremental_updates: true,
            ..Default::default()
        };

        let rec = advisor.recommend(&reqs);
        // HNSW variants support incremental updates
        assert!(matches!(
            rec.index_type,
            IndexType::Hnsw
                | IndexType::HnswPq
                | IndexType::HnswOpq
                | IndexType::HnswBinary
                | IndexType::Lsh
        ));
    }
}
