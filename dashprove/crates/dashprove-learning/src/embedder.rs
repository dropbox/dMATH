//! Proof and property embeddings for vector similarity search
//!
//! This module provides vector embeddings for properties and proofs, enabling
//! efficient similarity search in vector space. The embeddings are based on
//! structural features extracted from the USL AST.
//!
//! # Architecture
//!
//! The embedding system has two components:
//! 1. **Structural embeddings**: Dense vector representation of AST features
//! 2. **Keyword embeddings**: Bag-of-words style encoding of identifiers
//!
//! These are combined into a unified embedding vector that can be compared
//! using standard distance metrics (cosine similarity, L2 distance).
//!
//! # Example
//!
//! ```ignore
//! use dashprove_learning::embedder::{Embedding, PropertyEmbedder};
//! use dashprove_usl::ast::Property;
//!
//! let embedder = PropertyEmbedder::new();
//! let embedding = embedder.embed(&property);
//!
//! // Compare two embeddings
//! let similarity = embedding.cosine_similarity(&other_embedding);
//! ```

use crate::distance::{dot_product, euclidean_distance, vector_norm};
use crate::ordered_float::OrderedF64;
use crate::similarity::PropertyFeatures;
use dashprove_usl::ast::Property;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Default embedding dimension for structural features
pub const STRUCTURAL_DIM: usize = 32;

/// Default embedding dimension for keyword features (bag-of-words)
pub const KEYWORD_DIM: usize = 64;

/// Total embedding dimension
pub const EMBEDDING_DIM: usize = STRUCTURAL_DIM + KEYWORD_DIM;

/// Property category definitions for hierarchical encoding
///
/// The 30 PropertyType variants are grouped into 8 categories for
/// indices 0-7 (coarse), with sub-type indices 22-26 (fine-grained).
///
/// Categories:
/// 0. Theorem Proving: Theorem, Contract, Invariant, Refinement
/// 1. Model Checking: Temporal, Probabilistic
/// 2. Neural Networks: NeuralRobustness, NeuralReachability, AdversarialRobustness
/// 3. Security: SecurityProtocol, PlatformApi
/// 4. Memory Safety: MemorySafety, UndefinedBehavior, DataRace, MemoryLeak
/// 5. Testing: Fuzzing, PropertyBased, MutationTesting
/// 6. Static Analysis: Lint, ApiCompatibility, SecurityVulnerability, DependencyPolicy, SupplyChain, UnsafeAudit
/// 7. AI/ML: ModelOptimization, ModelCompression, DataQuality, Fairness, Interpretability, LLMGuardrails
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropertyCategory {
    /// Coarse category index (0-7) for indices 0-7
    pub category: usize,
    /// Fine-grained sub-type index (0-5) within category
    pub subtype: usize,
}

impl PropertyCategory {
    /// Map a property type string to its category encoding
    pub fn from_property_type(property_type: &str) -> Self {
        match property_type {
            // Category 0: Theorem Proving
            "theorem" => Self {
                category: 0,
                subtype: 0,
            },
            "contract" => Self {
                category: 0,
                subtype: 1,
            },
            "invariant" => Self {
                category: 0,
                subtype: 2,
            },
            "refinement" => Self {
                category: 0,
                subtype: 3,
            },

            // Category 1: Model Checking
            "temporal" => Self {
                category: 1,
                subtype: 0,
            },
            "probabilistic" => Self {
                category: 1,
                subtype: 1,
            },

            // Category 2: Neural Networks
            "neural_robustness" => Self {
                category: 2,
                subtype: 0,
            },
            "neural_reachability" => Self {
                category: 2,
                subtype: 1,
            },
            "adversarial_robustness" => Self {
                category: 2,
                subtype: 2,
            },

            // Category 3: Security
            "security" | "security_protocol" => Self {
                category: 3,
                subtype: 0,
            },
            "platform_api" => Self {
                category: 3,
                subtype: 1,
            },

            // Category 4: Memory Safety
            "memory_safety" => Self {
                category: 4,
                subtype: 0,
            },
            "undefined_behavior" => Self {
                category: 4,
                subtype: 1,
            },
            "data_race" => Self {
                category: 4,
                subtype: 2,
            },
            "memory_leak" => Self {
                category: 4,
                subtype: 3,
            },

            // Category 5: Testing
            "fuzzing" => Self {
                category: 5,
                subtype: 0,
            },
            "property_based" => Self {
                category: 5,
                subtype: 1,
            },
            "mutation_testing" => Self {
                category: 5,
                subtype: 2,
            },

            // Category 6: Static Analysis
            "lint" => Self {
                category: 6,
                subtype: 0,
            },
            "api_compatibility" => Self {
                category: 6,
                subtype: 1,
            },
            "security_vulnerability" => Self {
                category: 6,
                subtype: 2,
            },
            "dependency_policy" => Self {
                category: 6,
                subtype: 3,
            },
            "supply_chain" => Self {
                category: 6,
                subtype: 4,
            },
            "unsafe_audit" => Self {
                category: 6,
                subtype: 5,
            },

            // Category 7: AI/ML
            "model_optimization" => Self {
                category: 7,
                subtype: 0,
            },
            "model_compression" => Self {
                category: 7,
                subtype: 1,
            },
            "data_quality" => Self {
                category: 7,
                subtype: 2,
            },
            "fairness" => Self {
                category: 7,
                subtype: 3,
            },
            "interpretability" => Self {
                category: 7,
                subtype: 4,
            },
            "llm_guardrails" => Self {
                category: 7,
                subtype: 5,
            },

            // Semantic maps to theorem (it's theorem-like verification)
            "semantic" => Self {
                category: 0,
                subtype: 0,
            },

            // Unknown defaults to theorem category with unknown subtype
            _ => Self {
                category: 0,
                subtype: 4,
            },
        }
    }

    /// Maximum subtype index per category (for normalization)
    pub fn max_subtype_for_category(category: usize) -> usize {
        match category {
            0 => 4, // Theorem Proving: 4 subtypes + unknown
            1 => 1, // Model Checking: 2 subtypes
            2 => 2, // Neural Networks: 3 subtypes
            3 => 1, // Security: 2 subtypes
            4 => 3, // Memory Safety: 4 subtypes
            5 => 2, // Testing: 3 subtypes
            6 => 5, // Static Analysis: 6 subtypes
            7 => 5, // AI/ML: 6 subtypes
            _ => 0,
        }
    }
}

/// A dense vector embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// The embedding vector
    pub vector: Vec<f32>,
    /// Dimension of this embedding
    pub dim: usize,
}

impl Default for Embedding {
    fn default() -> Self {
        Self::zeros(EMBEDDING_DIM)
    }
}

impl Embedding {
    /// Create a new embedding from a vector
    pub fn new(vector: Vec<f32>) -> Self {
        let dim = vector.len();
        Self { vector, dim }
    }

    /// Create a zero embedding of given dimension
    pub fn zeros(dim: usize) -> Self {
        Self {
            vector: vec![0.0; dim],
            dim,
        }
    }

    /// Compute L2 (Euclidean) distance to another embedding
    pub fn l2_distance(&self, other: &Embedding) -> f32 {
        if self.dim != other.dim {
            return f32::MAX;
        }

        euclidean_distance(&self.vector, &other.vector)
    }

    /// Compute cosine similarity to another embedding
    ///
    /// Returns a value in [-1, 1] where 1 means identical direction,
    /// 0 means orthogonal, and -1 means opposite direction.
    ///
    /// Uses SIMD-accelerated dot product and norm calculations when available.
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        if self.dim != other.dim {
            return 0.0;
        }

        let dot = dot_product(&self.vector, &other.vector);
        let norm_a = vector_norm(&self.vector);
        let norm_b = vector_norm(&other.vector);

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Compute normalized similarity (0.0 to 1.0) from cosine similarity
    ///
    /// Maps cosine similarity from [-1, 1] to [0, 1] for consistency
    /// with existing similarity interfaces.
    pub fn normalized_similarity(&self, other: &Embedding) -> f64 {
        let cos = self.cosine_similarity(other);
        // Map [-1, 1] to [0, 1]
        ((cos + 1.0) / 2.0) as f64
    }

    /// L2 normalize the embedding to unit length
    ///
    /// Uses SIMD-accelerated norm calculation when available.
    pub fn normalize(&mut self) {
        let norm = vector_norm(&self.vector);
        if norm > 1e-10 {
            for x in &mut self.vector {
                *x /= norm;
            }
        }
    }

    /// Return a normalized copy
    pub fn normalized(&self) -> Self {
        let mut result = self.clone();
        result.normalize();
        result
    }
}

/// Embedder for property/proof structural features
#[derive(Debug, Clone, Default)]
pub struct PropertyEmbedder {
    /// Vocabulary for keyword hashing (built from observed keywords)
    keyword_vocab: HashMap<String, usize>,
    /// Next available vocabulary index
    next_vocab_idx: usize,
}

impl PropertyEmbedder {
    /// Create a new property embedder
    pub fn new() -> Self {
        Self {
            keyword_vocab: HashMap::new(),
            next_vocab_idx: 0,
        }
    }

    /// Create embedder with pre-initialized vocabulary
    pub fn with_vocabulary(vocab: HashMap<String, usize>) -> Self {
        let next_idx = vocab.values().max().map(|m| m + 1).unwrap_or(0);
        Self {
            keyword_vocab: vocab,
            next_vocab_idx: next_idx,
        }
    }

    /// Get current vocabulary (for persistence)
    pub fn vocabulary(&self) -> &HashMap<String, usize> {
        &self.keyword_vocab
    }

    /// Embed a property into vector space
    pub fn embed(&mut self, property: &Property) -> Embedding {
        let features = crate::similarity::extract_features(property);
        self.embed_features(&features)
    }

    /// Embed a property without updating vocabulary (for query embeddings)
    pub fn embed_query(&self, property: &Property) -> Embedding {
        let features = crate::similarity::extract_features(property);
        self.embed_features_readonly(&features)
    }

    /// Embed pre-extracted features
    pub fn embed_features(&mut self, features: &PropertyFeatures) -> Embedding {
        let structural = self.embed_structural(features);
        let keywords = self.embed_keywords_mutable(&features.keywords);
        self.combine_embeddings(structural, keywords)
    }

    /// Embed features without modifying vocabulary (for queries)
    pub fn embed_features_readonly(&self, features: &PropertyFeatures) -> Embedding {
        let structural = self.embed_structural(features);
        let keywords = self.embed_keywords_readonly(&features.keywords);
        self.combine_embeddings(structural, keywords)
    }

    /// Embed structural features into a fixed-size vector
    ///
    /// Index layout:
    /// - 0-7: Property category one-hot encoding (8 categories)
    /// - 8-21: Feature values (depth, quantifiers, etc.)
    /// - 22-26: Property subtype encoding (fine-grained within category)
    /// - 27-31: Category-specific features
    fn embed_structural(&self, features: &PropertyFeatures) -> Vec<f32> {
        let mut vec = vec![0.0f32; STRUCTURAL_DIM];

        // Get hierarchical property category
        let category = PropertyCategory::from_property_type(&features.property_type);

        // Property category one-hot encoding (indices 0-7)
        vec[category.category] = 1.0;

        // Depth (log-scaled, index 8)
        vec[8] = (features.depth as f32 + 1.0).ln();

        // Quantifier depth (log-scaled, index 9)
        vec[9] = (features.quantifier_depth as f32 + 1.0).ln();

        // Implication count (log-scaled, index 10)
        vec[10] = (features.implication_count as f32 + 1.0).ln();

        // Arithmetic ops (log-scaled, index 11)
        vec[11] = (features.arithmetic_ops as f32 + 1.0).ln();

        // Function calls (log-scaled, index 12)
        vec[12] = (features.function_calls as f32 + 1.0).ln();

        // Variable count (log-scaled, index 13)
        vec[13] = (features.variable_count as f32 + 1.0).ln();

        // Temporal flag (index 14)
        vec[14] = if features.has_temporal { 1.0 } else { 0.0 };

        // Type refs count (log-scaled, index 15)
        vec[15] = (features.type_refs.len() as f32 + 1.0).ln();

        // Keywords count (log-scaled, index 16)
        vec[16] = (features.keywords.len() as f32 + 1.0).ln();

        // Derived complexity metrics
        // Total AST complexity estimate (index 17)
        let complexity = features.depth
            + features.quantifier_depth * 2
            + features.implication_count
            + features.arithmetic_ops
            + features.function_calls;
        vec[17] = (complexity as f32 + 1.0).ln();

        // Has quantifiers flag (index 18)
        vec[18] = if features.quantifier_depth > 0 {
            1.0
        } else {
            0.0
        };

        // Has implications flag (index 19)
        vec[19] = if features.implication_count > 0 {
            1.0
        } else {
            0.0
        };

        // Has functions flag (index 20)
        vec[20] = if features.function_calls > 0 {
            1.0
        } else {
            0.0
        };

        // Has arithmetic flag (index 21)
        vec[21] = if features.arithmetic_ops > 0 {
            1.0
        } else {
            0.0
        };

        // Property subtype encoding (indices 22-26)
        // Normalized subtype value within category [0, 1]
        let max_subtype = PropertyCategory::max_subtype_for_category(category.category);
        if max_subtype > 0 {
            vec[22] = category.subtype as f32 / max_subtype as f32;
        }

        // Category-specific feature indicators (indices 23-26)
        // These provide additional discrimination within categories

        // Index 23: Is this a "core" subtype (first in category)?
        vec[23] = if category.subtype == 0 { 1.0 } else { 0.0 };

        // Index 24: Category domain signal
        // - Formal methods (categories 0, 1): logic-heavy
        // - Runtime analysis (categories 4, 5): execution-focused
        // - ML/AI (categories 2, 7): data-driven
        // - Security/Static (categories 3, 6): policy-focused
        vec[24] = match category.category {
            0 | 1 => 0.0,  // Formal methods - low domain index
            2 | 7 => 1.0,  // ML/AI - high domain index
            3 | 6 => 0.5,  // Security/static analysis - middle
            4 | 5 => 0.25, // Runtime analysis - low-middle
            _ => 0.0,
        };

        // Index 25: Verification vs Analysis signal
        // Verification produces proofs; analysis produces reports
        vec[25] = match category.category {
            0..=2 => 1.0, // Verification-oriented
            3..=7 => 0.0, // Analysis-oriented
            _ => 0.5,
        };

        // Index 26: Automated vs Interactive signal
        // Some tools are push-button; others need human guidance
        vec[26] = match category.category {
            5 | 6 => 1.0,  // Testing/linting - fully automated
            4 | 7 => 0.75, // Memory safety/ML - mostly automated
            1..=3 => 0.5,  // Model checking/neural/security - semi-automated
            0 => 0.25,     // Theorem proving - often interactive
            _ => 0.5,
        };

        // Indices 27-31 remain reserved for future features
        // (e.g., learned embeddings from external models)

        vec
    }

    /// Embed keywords using feature hashing (mutable - learns new vocabulary)
    fn embed_keywords_mutable(&mut self, keywords: &[String]) -> Vec<f32> {
        let mut vec = vec![0.0f32; KEYWORD_DIM];

        for keyword in keywords {
            // Get or create vocabulary index
            let idx = *self
                .keyword_vocab
                .entry(keyword.clone())
                .or_insert_with(|| {
                    let idx = self.next_vocab_idx;
                    self.next_vocab_idx += 1;
                    idx
                });

            // Hash to embedding dimension
            let hash_idx = idx % KEYWORD_DIM;
            vec[hash_idx] += 1.0;
        }

        // TF normalization (divide by max)
        let max_val = vec.iter().cloned().fold(0.0f32, f32::max);
        if max_val > 0.0 {
            for x in &mut vec {
                *x /= max_val;
            }
        }

        vec
    }

    /// Embed keywords using feature hashing (readonly - uses existing vocabulary only)
    fn embed_keywords_readonly(&self, keywords: &[String]) -> Vec<f32> {
        let mut vec = vec![0.0f32; KEYWORD_DIM];

        for keyword in keywords {
            // Only use existing vocabulary entries
            if let Some(&idx) = self.keyword_vocab.get(keyword) {
                let hash_idx = idx % KEYWORD_DIM;
                vec[hash_idx] += 1.0;
            } else {
                // For unknown keywords, use string hash
                let hash_idx = string_hash(keyword) % KEYWORD_DIM;
                vec[hash_idx] += 0.5; // Lower weight for unknown keywords
            }
        }

        // TF normalization
        let max_val = vec.iter().cloned().fold(0.0f32, f32::max);
        if max_val > 0.0 {
            for x in &mut vec {
                *x /= max_val;
            }
        }

        vec
    }

    /// Combine structural and keyword embeddings
    fn combine_embeddings(&self, structural: Vec<f32>, keywords: Vec<f32>) -> Embedding {
        let mut vector = structural;
        vector.extend(keywords);
        Embedding::new(vector)
    }
}

/// Simple string hash for keyword fallback
fn string_hash(s: &str) -> usize {
    // DJB2 hash
    let mut hash: usize = 5381;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(byte as usize);
    }
    hash
}

/// Builder for creating an embedding index from a corpus
#[derive(Debug, Default)]
pub struct EmbeddingIndexBuilder {
    embeddings: Vec<(String, Embedding)>,
    embedder: PropertyEmbedder,
}

impl EmbeddingIndexBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a property with its ID to the index
    pub fn add(&mut self, id: String, property: &Property) -> &mut Self {
        let embedding = self.embedder.embed(property);
        self.embeddings.push((id, embedding));
        self
    }

    /// Add pre-extracted features
    pub fn add_features(&mut self, id: String, features: &PropertyFeatures) -> &mut Self {
        let embedding = self.embedder.embed_features(features);
        self.embeddings.push((id, embedding));
        self
    }

    /// Build the embedding index
    pub fn build(self) -> EmbeddingIndex {
        EmbeddingIndex {
            entries: self.embeddings,
            embedder: self.embedder,
        }
    }
}

/// A simple embedding index for similarity search
///
/// This is a basic linear scan index. For larger corpora,
/// consider using an approximate nearest neighbor (ANN) index.
#[derive(Debug)]
pub struct EmbeddingIndex {
    entries: Vec<(String, Embedding)>,
    embedder: PropertyEmbedder,
}

impl EmbeddingIndex {
    /// Create a new empty index
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            embedder: PropertyEmbedder::new(),
        }
    }

    /// Create index with pre-built embedder
    pub fn with_embedder(embedder: PropertyEmbedder) -> Self {
        Self {
            entries: Vec::new(),
            embedder,
        }
    }

    /// Number of entries in the index
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add an entry to the index
    pub fn insert(&mut self, id: String, property: &Property) {
        let embedding = self.embedder.embed(property);
        self.entries.push((id, embedding));
    }

    /// Add entry with pre-extracted features
    pub fn insert_features(&mut self, id: String, features: &PropertyFeatures) {
        let embedding = self.embedder.embed_features(features);
        self.entries.push((id, embedding));
    }

    /// Add entry with pre-computed embedding
    pub fn insert_embedding(&mut self, id: String, embedding: Embedding) {
        self.entries.push((id, embedding));
    }

    /// Find k nearest neighbors to a query property
    pub fn find_nearest(&self, property: &Property, k: usize) -> Vec<(String, f64)> {
        let query = self.embedder.embed_query(property);
        self.find_nearest_embedding(&query, k)
    }

    /// Find k nearest neighbors to a query embedding
    ///
    /// Uses a min-heap to efficiently find top-k results in O(n log k) time.
    pub fn find_nearest_embedding(&self, query: &Embedding, k: usize) -> Vec<(String, f64)> {
        if self.entries.is_empty() || k == 0 {
            return vec![];
        }

        // Use min-heap for efficient top-k selection
        // BinaryHeap is max-heap by default, so we use Reverse to get min-heap
        let mut heap: BinaryHeap<std::cmp::Reverse<(OrderedF64, String)>> =
            BinaryHeap::with_capacity(k + 1);

        for (id, emb) in &self.entries {
            let sim = query.normalized_similarity(emb);
            let ordered = OrderedF64(sim);

            if heap.len() < k {
                heap.push(std::cmp::Reverse((ordered, id.clone())));
            } else if let Some(std::cmp::Reverse((min_score, _))) = heap.peek() {
                if ordered > *min_score {
                    heap.pop();
                    heap.push(std::cmp::Reverse((ordered, id.clone())));
                }
            }
        }

        // Extract results and sort by descending score
        let mut results: Vec<_> = heap
            .into_iter()
            .map(|std::cmp::Reverse((score, id))| (id, score.0))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Get the embedder (for creating query embeddings)
    pub fn embedder(&self) -> &PropertyEmbedder {
        &self.embedder
    }

    /// Get mutable embedder
    pub fn embedder_mut(&mut self) -> &mut PropertyEmbedder {
        &mut self.embedder
    }
}

impl Default for EmbeddingIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::similarity::PropertyFeatures;
    use dashprove_usl::ast::{Expr, Invariant, Temporal, TemporalExpr, Theorem, Type};
    use std::collections::HashMap;

    fn approx_eq_f32(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    fn approx_eq_f64(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    fn make_invariant(name: &str, body: Expr) -> Property {
        Property::Invariant(Invariant {
            name: name.to_string(),
            body,
        })
    }

    fn make_theorem(name: &str, body: Expr) -> Property {
        Property::Theorem(Theorem {
            name: name.to_string(),
            body,
        })
    }

    fn make_temporal(name: &str, body: TemporalExpr) -> Property {
        Property::Temporal(Temporal {
            name: name.to_string(),
            body,
            fairness: vec![],
        })
    }

    #[test]
    fn test_embedding_zeros() {
        let emb = Embedding::zeros(10);
        assert_eq!(emb.dim, 10);
        assert_eq!(emb.vector.len(), 10);
        assert!(emb.vector.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_embedding_l2_distance() {
        let a = Embedding::new(vec![1.0, 0.0, 0.0]);
        let b = Embedding::new(vec![0.0, 1.0, 0.0]);

        let dist = a.l2_distance(&b);
        // sqrt(1 + 1) = sqrt(2) ≈ 1.414
        assert!((dist - 1.414).abs() < 0.01);

        // Self distance should be 0
        assert!((a.l2_distance(&a)).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_cosine_similarity() {
        let a = Embedding::new(vec![1.0, 0.0, 0.0]);
        let b = Embedding::new(vec![0.0, 1.0, 0.0]);
        let c = Embedding::new(vec![1.0, 0.0, 0.0]);
        let d = Embedding::new(vec![-1.0, 0.0, 0.0]);
        let e = Embedding::new(vec![1.0, 2.0]);

        // Orthogonal vectors
        assert!((a.cosine_similarity(&b)).abs() < 1e-6);

        // Same direction
        assert!((a.cosine_similarity(&c) - 1.0).abs() < 1e-6);

        // Opposite direction
        assert!((a.cosine_similarity(&d) - (-1.0)).abs() < 1e-6);

        // Mismatched dimensions fall back to 0.0
        assert_eq!(a.cosine_similarity(&e), 0.0);
    }

    #[test]
    fn test_embedding_dimension_mismatch_handling() {
        let a = Embedding::new(vec![1.0, 2.0]);
        let b = Embedding::new(vec![3.0, 4.0, 5.0]);

        assert_eq!(a.l2_distance(&b), f32::MAX);
        assert!(approx_eq_f64(a.normalized_similarity(&b), 0.5));
    }

    #[test]
    fn test_embedding_normalized_similarity() {
        let a = Embedding::new(vec![1.0, 0.0, 0.0]);
        let b = Embedding::new(vec![0.0, 1.0, 0.0]);
        let c = Embedding::new(vec![1.0, 0.0, 0.0]);
        let d = Embedding::new(vec![-1.0, 0.0, 0.0]);

        // Orthogonal: cosine=0 -> normalized=0.5
        assert!((a.normalized_similarity(&b) - 0.5).abs() < 1e-6);

        // Same: cosine=1 -> normalized=1.0
        assert!((a.normalized_similarity(&c) - 1.0).abs() < 1e-6);

        // Opposite: cosine=-1 -> normalized=0.0
        assert!((a.normalized_similarity(&d) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_normalize() {
        let mut emb = Embedding::new(vec![3.0, 4.0]);
        emb.normalize();

        // Should be unit length
        let norm: f32 = emb.vector.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);

        // Components should be 3/5, 4/5
        assert!((emb.vector[0] - 0.6).abs() < 1e-6);
        assert!((emb.vector[1] - 0.8).abs() < 1e-6);

        // Zero vector should remain unchanged
        let mut zero = Embedding::zeros(3);
        zero.normalize();
        assert!(zero.vector.iter().all(|&x| x == 0.0));

        // normalized() should leave the original untouched
        let original = Embedding::new(vec![2.0, 0.0]);
        let normalized = original.normalized();
        assert_eq!(original.vector, vec![2.0, 0.0]);
        assert!(approx_eq_f32(normalized.vector[0], 1.0));
        assert!(approx_eq_f32(normalized.vector[1], 0.0));
    }

    #[test]
    fn test_property_embedder_simple() {
        let mut embedder = PropertyEmbedder::new();

        let prop = make_invariant("test", Expr::Bool(true));
        let emb = embedder.embed(&prop);

        assert_eq!(emb.dim, EMBEDDING_DIM);
        assert_eq!(emb.vector.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_property_embedder_type_distinction() {
        let mut embedder = PropertyEmbedder::new();

        let inv = make_invariant("test", Expr::Bool(true));
        let thm = make_theorem("test", Expr::Bool(true));

        let emb_inv = embedder.embed(&inv);
        let emb_thm = embedder.embed(&thm);

        // Different types should have different embeddings
        assert!(emb_inv.l2_distance(&emb_thm) > 0.0);

        // But same property should be identical
        let emb_inv2 = embedder.embed(&inv);
        assert!(emb_inv.l2_distance(&emb_inv2) < 1e-6);
    }

    #[test]
    fn test_property_embedder_similar_properties() {
        let mut embedder = PropertyEmbedder::new();

        // Two similar invariants (same structure, different names)
        let inv1 = make_invariant("array_bounds", Expr::Bool(true));
        let inv2 = make_invariant("bounds_check", Expr::Bool(true));

        // A very different temporal property
        let temp = make_temporal(
            "liveness",
            TemporalExpr::Always(Box::new(TemporalExpr::Eventually(Box::new(
                TemporalExpr::Atom(Expr::Var("done".to_string())),
            )))),
        );

        let emb1 = embedder.embed(&inv1);
        let emb2 = embedder.embed(&inv2);
        let emb_temp = embedder.embed(&temp);

        // Similar invariants should be more similar to each other than to temporal
        let sim_inv_inv = emb1.cosine_similarity(&emb2);
        let sim_inv_temp = emb1.cosine_similarity(&emb_temp);

        assert!(sim_inv_inv > sim_inv_temp);
    }

    #[test]
    fn test_property_embedder_quantified() {
        let mut embedder = PropertyEmbedder::new();

        let simple = make_invariant("simple", Expr::Bool(true));
        let quantified = make_invariant(
            "quantified",
            Expr::ForAll {
                var: "x".to_string(),
                ty: Some(Type::Named("Int".to_string())),
                body: Box::new(Expr::Bool(true)),
            },
        );

        let emb_simple = embedder.embed(&simple);
        let emb_quant = embedder.embed(&quantified);

        // Quantified should have different embedding
        assert!(emb_simple.l2_distance(&emb_quant) > 0.0);
    }

    #[test]
    fn test_embed_structural_feature_mapping() {
        let embedder = PropertyEmbedder::new();
        let features = PropertyFeatures {
            property_type: "security".to_string(),
            depth: 3,
            quantifier_depth: 2,
            implication_count: 4,
            arithmetic_ops: 1,
            function_calls: 2,
            variable_count: 5,
            has_temporal: true,
            type_refs: vec!["User".to_string(), "Action".to_string()],
            keywords: vec!["key".to_string()],
        };

        let structural = embedder.embed_structural(&features);
        assert_eq!(structural.len(), STRUCTURAL_DIM);
        // "security" maps to category 3 (Security)
        assert!(approx_eq_f32(structural[3], 1.0)); // security category one-hot
        assert!(approx_eq_f32(
            structural[8],
            (features.depth as f32 + 1.0).ln()
        ));
        assert!(approx_eq_f32(
            structural[9],
            (features.quantifier_depth as f32 + 1.0).ln()
        ));
        assert!(approx_eq_f32(
            structural[10],
            (features.implication_count as f32 + 1.0).ln()
        ));
        assert!(approx_eq_f32(structural[14], 1.0));
        assert!(approx_eq_f32(
            structural[15],
            (features.type_refs.len() as f32 + 1.0).ln()
        ));
        assert!(approx_eq_f32(
            structural[16],
            (features.keywords.len() as f32 + 1.0).ln()
        ));
        assert!(approx_eq_f32(structural[21], 1.0));
        // Indices 22-26 now have category-specific features
        // 22: subtype normalized (security is subtype 0, max 1) = 0.0
        assert!(approx_eq_f32(structural[22], 0.0));
        // 23: core subtype flag (subtype == 0) = 1.0
        assert!(approx_eq_f32(structural[23], 1.0));
        // 24: domain signal (security = 0.5)
        assert!(approx_eq_f32(structural[24], 0.5));
        // 25: verification vs analysis (security = 0.0)
        assert!(approx_eq_f32(structural[25], 0.0));
        // 26: automated vs interactive (security = 0.5)
        assert!(approx_eq_f32(structural[26], 0.5));
        // 27-31 remain zero
        assert!(structural.iter().skip(27).all(|x| *x == 0.0));
    }

    #[test]
    fn test_keyword_embedding_normalization_and_growth() {
        let mut embedder = PropertyEmbedder::new();
        let features = PropertyFeatures {
            keywords: vec!["alpha".to_string(), "alpha".to_string(), "beta".to_string()],
            ..Default::default()
        };

        let keywords = embedder.embed_keywords_mutable(&features.keywords);

        assert_eq!(embedder.vocabulary().len(), 2);
        assert!(approx_eq_f32(keywords[0], 1.0)); // "alpha" inserted twice, normalized to 1.0
        assert!(approx_eq_f32(keywords[1], 0.5)); // "beta" occurs once
    }

    #[test]
    fn test_embed_keywords_readonly_unknown_weight() {
        let mut vocab = HashMap::new();
        vocab.insert("known".to_string(), 10);
        let embedder = PropertyEmbedder::with_vocabulary(vocab);

        let known_bucket = 10 % KEYWORD_DIM;
        let mut unknown = "unknown".to_string();
        let mut unknown_bucket = string_hash(&unknown) % KEYWORD_DIM;
        if unknown_bucket == known_bucket {
            for i in 0..100 {
                let candidate = format!("unknown_{i}");
                let bucket = string_hash(&candidate) % KEYWORD_DIM;
                if bucket != known_bucket {
                    unknown = candidate;
                    unknown_bucket = bucket;
                    break;
                }
            }
        }

        let keywords = embedder.embed_keywords_readonly(&["known".to_string(), unknown.clone()]);

        assert!(approx_eq_f32(keywords[known_bucket], 1.0));
        assert!(approx_eq_f32(keywords[unknown_bucket], 0.5));
    }

    #[test]
    fn test_with_vocabulary_advances_index() {
        let mut vocab = HashMap::new();
        vocab.insert("existing".to_string(), 2);
        vocab.insert("latest".to_string(), 5);
        let mut embedder = PropertyEmbedder::with_vocabulary(vocab);

        let features = PropertyFeatures {
            keywords: vec!["new_keyword".to_string()],
            ..Default::default()
        };

        let _ = embedder.embed_features(&features);

        assert_eq!(embedder.vocabulary().get("new_keyword"), Some(&6));
    }

    #[test]
    fn test_embedding_index_basic() {
        let mut index = EmbeddingIndex::new();

        let prop1 = make_invariant("inv1", Expr::Bool(true));
        let prop2 = make_invariant("inv2", Expr::Bool(false));
        let prop3 = make_theorem("thm1", Expr::Bool(true));

        index.insert("id1".to_string(), &prop1);
        index.insert("id2".to_string(), &prop2);
        index.insert("id3".to_string(), &prop3);

        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_embedding_index_find_nearest() {
        let mut index = EmbeddingIndex::new();

        // Add some properties
        let inv1 = make_invariant("array_bounds", Expr::Bool(true));
        let inv2 = make_invariant("bounds_check", Expr::Bool(true));
        let thm1 = make_theorem("theorem_one", Expr::Bool(true));
        let temp1 = make_temporal(
            "liveness",
            TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
        );

        index.insert("inv1".to_string(), &inv1);
        index.insert("inv2".to_string(), &inv2);
        index.insert("thm1".to_string(), &thm1);
        index.insert("temp1".to_string(), &temp1);

        // Query with similar invariant
        let query = make_invariant("array_check", Expr::Bool(true));
        let results = index.find_nearest(&query, 2);

        assert_eq!(results.len(), 2);
        // The two invariants should be more similar to query than the theorem/temporal
        // (they share "invariant" type and similar keywords)
        let result_ids: Vec<_> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(result_ids.contains(&"inv1") || result_ids.contains(&"inv2"));
    }

    #[test]
    fn test_embedding_index_nearest_embedding_ordering() {
        let mut index = EmbeddingIndex::new();
        index.insert_embedding("a".to_string(), Embedding::new(vec![1.0, 0.0]));
        index.insert_embedding("b".to_string(), Embedding::new(vec![0.0, 1.0]));

        let query = Embedding::new(vec![1.0, 0.0]);
        let results = index.find_nearest_embedding(&query, 5);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 >= results[1].1);
    }

    #[test]
    fn test_embedding_index_builder() {
        let mut builder = EmbeddingIndexBuilder::new();

        let prop1 = make_invariant("test1", Expr::Bool(true));
        let prop2 = make_invariant("test2", Expr::Bool(false));

        builder
            .add("id1".to_string(), &prop1)
            .add("id2".to_string(), &prop2);

        let index = builder.build();
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_vocabulary_persistence() {
        let mut embedder = PropertyEmbedder::new();

        let prop = make_invariant("array_bounds_check", Expr::Var("size".to_string()));
        let _emb = embedder.embed(&prop);

        // Vocabulary should contain the keywords
        let vocab = embedder.vocabulary();
        assert!(!vocab.is_empty());
        assert!(
            vocab.contains_key("array")
                || vocab.contains_key("bounds")
                || vocab.contains_key("check")
        );
    }

    #[test]
    fn test_embed_query_readonly() {
        let mut embedder = PropertyEmbedder::new();

        // First, build vocabulary
        let train_prop = make_invariant("array_bounds", Expr::Bool(true));
        let _train_emb = embedder.embed(&train_prop);

        let vocab_size_before = embedder.vocabulary().len();

        // Query with new keywords - should NOT modify vocabulary
        let query_prop = make_invariant("completely_new_keywords", Expr::Bool(true));
        let _query_emb = embedder.embed_query(&query_prop);

        let vocab_size_after = embedder.vocabulary().len();
        assert_eq!(vocab_size_before, vocab_size_after);
    }

    // ==========================================================================
    // PropertyCategory tests
    // ==========================================================================

    #[test]
    fn test_property_category_theorem_proving() {
        // Category 0: Theorem Proving
        assert_eq!(
            PropertyCategory::from_property_type("theorem"),
            PropertyCategory {
                category: 0,
                subtype: 0
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("contract"),
            PropertyCategory {
                category: 0,
                subtype: 1
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("invariant"),
            PropertyCategory {
                category: 0,
                subtype: 2
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("refinement"),
            PropertyCategory {
                category: 0,
                subtype: 3
            }
        );
    }

    #[test]
    fn test_property_category_model_checking() {
        // Category 1: Model Checking
        assert_eq!(
            PropertyCategory::from_property_type("temporal"),
            PropertyCategory {
                category: 1,
                subtype: 0
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("probabilistic"),
            PropertyCategory {
                category: 1,
                subtype: 1
            }
        );
    }

    #[test]
    fn test_property_category_neural_networks() {
        // Category 2: Neural Networks
        assert_eq!(
            PropertyCategory::from_property_type("neural_robustness"),
            PropertyCategory {
                category: 2,
                subtype: 0
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("neural_reachability"),
            PropertyCategory {
                category: 2,
                subtype: 1
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("adversarial_robustness"),
            PropertyCategory {
                category: 2,
                subtype: 2
            }
        );
    }

    #[test]
    fn test_property_category_security() {
        // Category 3: Security
        assert_eq!(
            PropertyCategory::from_property_type("security"),
            PropertyCategory {
                category: 3,
                subtype: 0
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("security_protocol"),
            PropertyCategory {
                category: 3,
                subtype: 0
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("platform_api"),
            PropertyCategory {
                category: 3,
                subtype: 1
            }
        );
    }

    #[test]
    fn test_property_category_memory_safety() {
        // Category 4: Memory Safety
        assert_eq!(
            PropertyCategory::from_property_type("memory_safety"),
            PropertyCategory {
                category: 4,
                subtype: 0
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("undefined_behavior"),
            PropertyCategory {
                category: 4,
                subtype: 1
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("data_race"),
            PropertyCategory {
                category: 4,
                subtype: 2
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("memory_leak"),
            PropertyCategory {
                category: 4,
                subtype: 3
            }
        );
    }

    #[test]
    fn test_property_category_testing() {
        // Category 5: Testing
        assert_eq!(
            PropertyCategory::from_property_type("fuzzing"),
            PropertyCategory {
                category: 5,
                subtype: 0
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("property_based"),
            PropertyCategory {
                category: 5,
                subtype: 1
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("mutation_testing"),
            PropertyCategory {
                category: 5,
                subtype: 2
            }
        );
    }

    #[test]
    fn test_property_category_static_analysis() {
        // Category 6: Static Analysis
        assert_eq!(
            PropertyCategory::from_property_type("lint"),
            PropertyCategory {
                category: 6,
                subtype: 0
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("api_compatibility"),
            PropertyCategory {
                category: 6,
                subtype: 1
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("security_vulnerability"),
            PropertyCategory {
                category: 6,
                subtype: 2
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("dependency_policy"),
            PropertyCategory {
                category: 6,
                subtype: 3
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("supply_chain"),
            PropertyCategory {
                category: 6,
                subtype: 4
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("unsafe_audit"),
            PropertyCategory {
                category: 6,
                subtype: 5
            }
        );
    }

    #[test]
    fn test_property_category_ai_ml() {
        // Category 7: AI/ML
        assert_eq!(
            PropertyCategory::from_property_type("model_optimization"),
            PropertyCategory {
                category: 7,
                subtype: 0
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("model_compression"),
            PropertyCategory {
                category: 7,
                subtype: 1
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("data_quality"),
            PropertyCategory {
                category: 7,
                subtype: 2
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("fairness"),
            PropertyCategory {
                category: 7,
                subtype: 3
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("interpretability"),
            PropertyCategory {
                category: 7,
                subtype: 4
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("llm_guardrails"),
            PropertyCategory {
                category: 7,
                subtype: 5
            }
        );
    }

    #[test]
    fn test_property_category_semantic_maps_to_theorem() {
        // "semantic" should map to theorem category
        assert_eq!(
            PropertyCategory::from_property_type("semantic"),
            PropertyCategory {
                category: 0,
                subtype: 0
            }
        );
    }

    #[test]
    fn test_property_category_unknown_defaults() {
        // Unknown types default to category 0, subtype 4
        assert_eq!(
            PropertyCategory::from_property_type("unknown_type"),
            PropertyCategory {
                category: 0,
                subtype: 4
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type(""),
            PropertyCategory {
                category: 0,
                subtype: 4
            }
        );
        assert_eq!(
            PropertyCategory::from_property_type("foobar"),
            PropertyCategory {
                category: 0,
                subtype: 4
            }
        );
    }

    #[test]
    fn test_max_subtype_for_category() {
        assert_eq!(PropertyCategory::max_subtype_for_category(0), 4); // Theorem Proving
        assert_eq!(PropertyCategory::max_subtype_for_category(1), 1); // Model Checking
        assert_eq!(PropertyCategory::max_subtype_for_category(2), 2); // Neural Networks
        assert_eq!(PropertyCategory::max_subtype_for_category(3), 1); // Security
        assert_eq!(PropertyCategory::max_subtype_for_category(4), 3); // Memory Safety
        assert_eq!(PropertyCategory::max_subtype_for_category(5), 2); // Testing
        assert_eq!(PropertyCategory::max_subtype_for_category(6), 5); // Static Analysis
        assert_eq!(PropertyCategory::max_subtype_for_category(7), 5); // AI/ML
        assert_eq!(PropertyCategory::max_subtype_for_category(8), 0); // Out of range
    }

    #[test]
    fn test_embed_structural_hierarchical_encoding() {
        let embedder = PropertyEmbedder::new();

        // Test theorem (category 0, subtype 0)
        let theorem_features = PropertyFeatures {
            property_type: "theorem".to_string(),
            ..Default::default()
        };
        let theorem_vec = embedder.embed_structural(&theorem_features);
        assert!(approx_eq_f32(theorem_vec[0], 1.0)); // category 0
        assert!(approx_eq_f32(theorem_vec[22], 0.0)); // subtype 0/4 = 0.0
        assert!(approx_eq_f32(theorem_vec[23], 1.0)); // core subtype
        assert!(approx_eq_f32(theorem_vec[25], 1.0)); // verification-oriented

        // Test contract (category 0, subtype 1)
        let contract_features = PropertyFeatures {
            property_type: "contract".to_string(),
            ..Default::default()
        };
        let contract_vec = embedder.embed_structural(&contract_features);
        assert!(approx_eq_f32(contract_vec[0], 1.0)); // same category as theorem
        assert!(approx_eq_f32(contract_vec[22], 0.25)); // subtype 1/4 = 0.25
        assert!(approx_eq_f32(contract_vec[23], 0.0)); // not core subtype

        // Test temporal (category 1, subtype 0)
        let temporal_features = PropertyFeatures {
            property_type: "temporal".to_string(),
            ..Default::default()
        };
        let temporal_vec = embedder.embed_structural(&temporal_features);
        assert!(approx_eq_f32(temporal_vec[1], 1.0)); // category 1
        assert!(approx_eq_f32(temporal_vec[22], 0.0)); // subtype 0/1 = 0.0
        assert!(approx_eq_f32(temporal_vec[23], 1.0)); // core subtype

        // Test fuzzing (category 5, subtype 0)
        let fuzzing_features = PropertyFeatures {
            property_type: "fuzzing".to_string(),
            ..Default::default()
        };
        let fuzzing_vec = embedder.embed_structural(&fuzzing_features);
        assert!(approx_eq_f32(fuzzing_vec[5], 1.0)); // category 5
        assert!(approx_eq_f32(fuzzing_vec[25], 0.0)); // analysis-oriented
        assert!(approx_eq_f32(fuzzing_vec[26], 1.0)); // fully automated
    }

    #[test]
    fn test_embed_structural_category_domain_signals() {
        let embedder = PropertyEmbedder::new();

        // Test domain signal for formal methods (categories 0, 1)
        let theorem_vec = embedder.embed_structural(&PropertyFeatures {
            property_type: "theorem".to_string(),
            ..Default::default()
        });
        assert!(approx_eq_f32(theorem_vec[24], 0.0)); // formal methods

        let temporal_vec = embedder.embed_structural(&PropertyFeatures {
            property_type: "temporal".to_string(),
            ..Default::default()
        });
        assert!(approx_eq_f32(temporal_vec[24], 0.0)); // formal methods

        // Test domain signal for ML/AI (categories 2, 7)
        let neural_vec = embedder.embed_structural(&PropertyFeatures {
            property_type: "neural_robustness".to_string(),
            ..Default::default()
        });
        assert!(approx_eq_f32(neural_vec[24], 1.0)); // ML/AI

        let fairness_vec = embedder.embed_structural(&PropertyFeatures {
            property_type: "fairness".to_string(),
            ..Default::default()
        });
        assert!(approx_eq_f32(fairness_vec[24], 1.0)); // ML/AI

        // Test domain signal for security/static (categories 3, 6)
        let security_vec = embedder.embed_structural(&PropertyFeatures {
            property_type: "security".to_string(),
            ..Default::default()
        });
        assert!(approx_eq_f32(security_vec[24], 0.5)); // security/static

        let lint_vec = embedder.embed_structural(&PropertyFeatures {
            property_type: "lint".to_string(),
            ..Default::default()
        });
        assert!(approx_eq_f32(lint_vec[24], 0.5)); // security/static
    }

    #[test]
    fn test_hierarchical_encoding_similarity_within_category() {
        let mut embedder = PropertyEmbedder::new();

        // Properties in the same category should be more similar than across categories
        let theorem = make_theorem("theorem1", Expr::Bool(true));
        let invariant = make_invariant("invariant1", Expr::Bool(true));
        let temporal = make_temporal("temporal1", TemporalExpr::Atom(Expr::Bool(true)));

        let emb_theorem = embedder.embed(&theorem);
        let emb_invariant = embedder.embed(&invariant);
        let emb_temporal = embedder.embed(&temporal);

        // Theorem and invariant are both category 0 (theorem proving)
        let sim_within_category = emb_theorem.cosine_similarity(&emb_invariant);
        // Theorem and temporal are different categories (0 vs 1)
        let sim_across_categories = emb_theorem.cosine_similarity(&emb_temporal);

        // Within-category similarity should be higher
        assert!(
            sim_within_category > sim_across_categories,
            "Same category should have higher similarity: within={}, across={}",
            sim_within_category,
            sim_across_categories
        );
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify that zeros() creates a vector of the specified dimension
    #[kani::proof]
    fn verify_zeros_dimension() {
        let dim: usize = kani::any();
        kani::assume(dim <= 256); // Limit for tractability

        let emb = Embedding::zeros(dim);
        assert_eq!(emb.dim, dim, "Embedding dim should match requested");
        assert_eq!(emb.vector.len(), dim, "Vector length should match dim");
    }

    /// Verify that zeros() creates all-zero vectors
    #[kani::proof]
    fn verify_zeros_are_zero() {
        let dim: usize = kani::any();
        kani::assume(dim > 0 && dim <= 8); // Small for tractability

        let emb = Embedding::zeros(dim);
        for i in 0..dim {
            assert!(emb.vector[i] == 0.0, "All elements should be zero");
        }
    }

    /// Verify cosine similarity returns 0.0 for mismatched dimensions
    #[kani::proof]
    fn verify_cosine_mismatched_dims_returns_zero() {
        let dim_a: usize = kani::any();
        let dim_b: usize = kani::any();
        kani::assume(dim_a >= 1 && dim_a <= 8);
        kani::assume(dim_b >= 1 && dim_b <= 8);
        kani::assume(dim_a != dim_b);

        let a = Embedding::zeros(dim_a);
        let b = Embedding::zeros(dim_b);

        let sim = a.cosine_similarity(&b);
        assert!(sim == 0.0, "Mismatched dims should return 0.0");
    }

    /// Verify l2_distance returns MAX for mismatched dimensions
    #[kani::proof]
    fn verify_l2_mismatched_dims_returns_max() {
        let dim_a: usize = kani::any();
        let dim_b: usize = kani::any();
        kani::assume(dim_a >= 1 && dim_a <= 8);
        kani::assume(dim_b >= 1 && dim_b <= 8);
        kani::assume(dim_a != dim_b);

        let a = Embedding::zeros(dim_a);
        let b = Embedding::zeros(dim_b);

        let dist = a.l2_distance(&b);
        assert!(dist == f32::MAX, "Mismatched dims should return MAX");
    }

    /// Verify l2_distance with same embedding returns 0
    #[kani::proof]
    fn verify_l2_self_distance_is_zero() {
        let dim: usize = kani::any();
        kani::assume(dim >= 1 && dim <= 4);

        let emb = Embedding::zeros(dim);
        let dist = emb.l2_distance(&emb);
        assert!(dist == 0.0, "Self-distance should be 0");
    }

    /// Verify normalized_similarity maps cosine [-1, 1] to [0, 1]
    #[kani::proof]
    fn verify_normalized_similarity_bounds() {
        // Test with concrete small embeddings
        let a = Embedding::new(vec![1.0, 0.0]);
        let b = Embedding::new(vec![0.0, 1.0]);

        let sim = a.normalized_similarity(&b);
        assert!(sim >= 0.0, "normalized_similarity should be >= 0");
        assert!(sim <= 1.0, "normalized_similarity should be <= 1");
    }

    /// Verify normalized_similarity of identical vectors is 1.0
    #[kani::proof]
    fn verify_normalized_similarity_identical_is_one() {
        let a = Embedding::new(vec![1.0, 2.0, 3.0]);
        let sim = a.normalized_similarity(&a);
        // (1.0 + 1.0) / 2.0 = 1.0
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Identical vectors should have normalized_similarity 1.0"
        );
    }

    /// Verify normalized_similarity of opposite vectors is 0.0
    #[kani::proof]
    fn verify_normalized_similarity_opposite_is_zero() {
        let a = Embedding::new(vec![1.0, 0.0]);
        let b = Embedding::new(vec![-1.0, 0.0]);
        let sim = a.normalized_similarity(&b);
        // (-1.0 + 1.0) / 2.0 = 0.0
        assert!(
            sim.abs() < 1e-6,
            "Opposite vectors should have normalized_similarity 0.0"
        );
    }

    /// Verify EMBEDDING_DIM equals STRUCTURAL_DIM + KEYWORD_DIM
    #[kani::proof]
    fn verify_embedding_dim_constant() {
        assert_eq!(
            EMBEDDING_DIM,
            STRUCTURAL_DIM + KEYWORD_DIM,
            "EMBEDDING_DIM should equal STRUCTURAL_DIM + KEYWORD_DIM"
        );
    }

    /// Verify default embedding has correct dimension
    #[kani::proof]
    fn verify_default_embedding_dimension() {
        let emb = Embedding::default();
        assert_eq!(emb.dim, EMBEDDING_DIM, "Default should have EMBEDDING_DIM");
        assert_eq!(
            emb.vector.len(),
            EMBEDDING_DIM,
            "Default vector should have EMBEDDING_DIM length"
        );
    }

    /// Verify string_hash is deterministic
    #[kani::proof]
    fn verify_string_hash_deterministic() {
        // Use a concrete string to verify determinism
        let h1 = string_hash("test");
        let h2 = string_hash("test");
        assert_eq!(h1, h2, "Same input should produce same hash");
    }

    /// Verify DJB2 hash starts with 5381
    #[kani::proof]
    fn verify_string_hash_empty_is_5381() {
        let h = string_hash("");
        assert_eq!(h, 5381, "Empty string should hash to 5381");
    }
}
