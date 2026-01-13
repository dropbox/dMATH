//! Semantic similarity checking for text comparisons
//!
//! This module provides the core semantic similarity checking functionality,
//! comparing texts using embedding-based similarity with configurable thresholds.

use crate::embedding::{EmbeddingBackend, LocalEmbedding, TextEmbedding, TfIdfEmbedder};
use crate::error::SemanticResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for semantic similarity checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    /// Default similarity threshold (0.0 to 1.0)
    pub default_threshold: f64,
    /// Similarity metric to use
    pub metric: SimilarityMetric,
    /// Embedding backend to use
    pub backend: BackendConfig,
    /// Whether to store source texts in results
    pub store_sources: bool,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            default_threshold: 0.8,
            metric: SimilarityMetric::Cosine,
            backend: BackendConfig::Local { dimension: 256 },
            store_sources: true,
        }
    }
}

/// Similarity metric to use for comparison
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SimilarityMetric {
    /// Cosine similarity (angle-based)
    Cosine,
    /// Euclidean distance converted to similarity
    Euclidean,
    /// Dot product (for normalized vectors, equals cosine)
    DotProduct,
}

/// Configuration for embedding backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackendConfig {
    /// Local hash-based embeddings
    Local {
        /// Embedding dimension
        dimension: usize,
    },
    /// TF-IDF embeddings (requires fitting)
    TfIdf {
        /// Maximum vocabulary size
        max_vocab_size: usize,
    },
}

/// Result of a similarity check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    /// Whether the similarity check passed (similarity >= threshold)
    pub passed: bool,
    /// The computed similarity score (0.0 to 1.0)
    pub similarity: f64,
    /// The threshold that was used
    pub threshold: f64,
    /// Source text (if stored)
    pub source: Option<String>,
    /// Target text (if stored)
    pub target: Option<String>,
    /// Additional details
    pub details: Option<String>,
}

impl SimilarityResult {
    /// Create a new passing result
    pub fn pass(similarity: f64, threshold: f64) -> Self {
        Self {
            passed: true,
            similarity,
            threshold,
            source: None,
            target: None,
            details: None,
        }
    }

    /// Create a new failing result
    pub fn fail(similarity: f64, threshold: f64) -> Self {
        Self {
            passed: false,
            similarity,
            threshold,
            source: None,
            target: None,
            details: None,
        }
    }

    /// Add source text
    pub fn with_source(mut self, source: String) -> Self {
        self.source = Some(source);
        self
    }

    /// Add target text
    pub fn with_target(mut self, target: String) -> Self {
        self.target = Some(target);
        self
    }

    /// Add details
    pub fn with_details(mut self, details: String) -> Self {
        self.details = Some(details);
        self
    }
}

/// Semantic similarity checker
pub struct SemanticChecker {
    /// Configuration
    config: SemanticConfig,
    /// Embedding backend
    backend: Arc<dyn EmbeddingBackend>,
}

impl SemanticChecker {
    /// Create a new semantic checker with local embeddings
    pub fn new(config: SemanticConfig) -> Self {
        let backend: Arc<dyn EmbeddingBackend> = match &config.backend {
            BackendConfig::Local { dimension } => Arc::new(LocalEmbedding::new(*dimension)),
            BackendConfig::TfIdf { .. } => {
                // TF-IDF requires fitting, so default to local
                Arc::new(LocalEmbedding::default())
            }
        };

        Self { config, backend }
    }

    /// Create a checker with a pre-configured TF-IDF embedder
    pub fn with_tfidf(config: SemanticConfig, embedder: TfIdfEmbedder) -> Self {
        Self {
            config,
            backend: Arc::new(embedder),
        }
    }

    /// Create a checker with a custom backend
    pub fn with_backend(config: SemanticConfig, backend: Arc<dyn EmbeddingBackend>) -> Self {
        Self { config, backend }
    }

    /// Get the configuration
    pub fn config(&self) -> &SemanticConfig {
        &self.config
    }

    /// Check if two texts are semantically similar
    pub async fn check_similarity(
        &self,
        source: &str,
        target: &str,
        threshold: Option<f64>,
    ) -> SemanticResult<SimilarityResult> {
        let threshold = threshold.unwrap_or(self.config.default_threshold);

        // Generate embeddings
        let emb_source = self.backend.embed(source).await?;
        let emb_target = self.backend.embed(target).await?;

        // Compute similarity
        let similarity = self.compute_similarity(&emb_source, &emb_target);
        let passed = similarity >= threshold;

        let mut result = if passed {
            SimilarityResult::pass(similarity, threshold)
        } else {
            SimilarityResult::fail(similarity, threshold)
        };

        if self.config.store_sources {
            result = result
                .with_source(source.to_string())
                .with_target(target.to_string());
        }

        Ok(result)
    }

    /// Check similarity between pre-computed embeddings
    pub fn check_embedding_similarity(
        &self,
        source: &TextEmbedding,
        target: &TextEmbedding,
        threshold: Option<f64>,
    ) -> SimilarityResult {
        let threshold = threshold.unwrap_or(self.config.default_threshold);
        let similarity = self.compute_similarity(source, target);
        let passed = similarity >= threshold;

        if passed {
            SimilarityResult::pass(similarity, threshold)
        } else {
            SimilarityResult::fail(similarity, threshold)
        }
    }

    /// Compute similarity between two embeddings using configured metric
    pub fn compute_similarity(&self, a: &TextEmbedding, b: &TextEmbedding) -> f64 {
        match self.config.metric {
            SimilarityMetric::Cosine => a.cosine_similarity(b) as f64,
            SimilarityMetric::Euclidean => {
                // Convert distance to similarity: 1 / (1 + distance)
                let dist = a.l2_distance(b) as f64;
                1.0 / (1.0 + dist)
            }
            SimilarityMetric::DotProduct => {
                // For normalized vectors, dot product equals cosine similarity
                a.vector
                    .iter()
                    .zip(b.vector.iter())
                    .map(|(x, y)| (*x as f64) * (*y as f64))
                    .sum()
            }
        }
    }

    /// Generate embedding for text
    pub async fn embed(&self, text: &str) -> SemanticResult<TextEmbedding> {
        self.backend.embed(text).await
    }

    /// Batch embed multiple texts
    pub async fn embed_batch(&self, texts: &[&str]) -> SemanticResult<Vec<TextEmbedding>> {
        self.backend.embed_batch(texts).await
    }

    /// Find most similar text from a list
    pub async fn find_most_similar(
        &self,
        query: &str,
        candidates: &[&str],
    ) -> SemanticResult<Option<(usize, f64)>> {
        if candidates.is_empty() {
            return Ok(None);
        }

        let query_emb = self.backend.embed(query).await?;
        let candidate_embs = self.backend.embed_batch(candidates).await?;

        let mut best_idx = 0;
        let mut best_sim = f64::NEG_INFINITY;

        for (idx, emb) in candidate_embs.iter().enumerate() {
            let sim = self.compute_similarity(&query_emb, emb);
            if sim > best_sim {
                best_sim = sim;
                best_idx = idx;
            }
        }

        Ok(Some((best_idx, best_sim)))
    }

    /// Find all similar texts above threshold
    pub async fn find_similar_above_threshold(
        &self,
        query: &str,
        candidates: &[&str],
        threshold: Option<f64>,
    ) -> SemanticResult<Vec<(usize, f64)>> {
        let threshold = threshold.unwrap_or(self.config.default_threshold);

        let query_emb = self.backend.embed(query).await?;
        let candidate_embs = self.backend.embed_batch(candidates).await?;

        let mut results: Vec<(usize, f64)> = candidate_embs
            .iter()
            .enumerate()
            .filter_map(|(idx, emb)| {
                let sim = self.compute_similarity(&query_emb, emb);
                if sim >= threshold {
                    Some((idx, sim))
                } else {
                    None
                }
            })
            .collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }
}

/// Builder for SemanticChecker with fluent API
#[derive(Default)]
pub struct SemanticCheckerBuilder {
    config: SemanticConfig,
    backend: Option<Arc<dyn EmbeddingBackend>>,
}

impl SemanticCheckerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the default threshold
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.config.default_threshold = threshold;
        self
    }

    /// Set the similarity metric
    pub fn metric(mut self, metric: SimilarityMetric) -> Self {
        self.config.metric = metric;
        self
    }

    /// Use local embeddings with specified dimension
    pub fn local_embeddings(mut self, dimension: usize) -> Self {
        self.config.backend = BackendConfig::Local { dimension };
        self.backend = Some(Arc::new(LocalEmbedding::new(dimension)));
        self
    }

    /// Use TF-IDF embeddings
    pub fn tfidf_embeddings(mut self, embedder: TfIdfEmbedder) -> Self {
        self.config.backend = BackendConfig::TfIdf {
            max_vocab_size: embedder.vocab_size(),
        };
        self.backend = Some(Arc::new(embedder));
        self
    }

    /// Use a custom embedding backend
    pub fn custom_backend(mut self, backend: Arc<dyn EmbeddingBackend>) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Whether to store source texts in results
    pub fn store_sources(mut self, store: bool) -> Self {
        self.config.store_sources = store;
        self
    }

    /// Build the SemanticChecker
    pub fn build(self) -> SemanticChecker {
        if let Some(backend) = self.backend {
            SemanticChecker::with_backend(self.config, backend)
        } else {
            SemanticChecker::new(self.config)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_semantic_checker_default() {
        let checker = SemanticChecker::new(SemanticConfig::default());

        let result = checker
            .check_similarity("hello world", "hello world", None)
            .await
            .unwrap();

        assert!(result.passed);
        assert!((result.similarity - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_semantic_checker_different_texts() {
        let checker = SemanticChecker::new(SemanticConfig::default());

        let result = checker
            .check_similarity(
                "The quick brown fox jumps over the lazy dog",
                "A completely unrelated sentence about programming",
                None,
            )
            .await
            .unwrap();

        // Different texts should have lower similarity
        assert!(result.similarity < 1.0);
    }

    #[tokio::test]
    async fn test_semantic_checker_threshold() {
        let checker = SemanticChecker::new(SemanticConfig::default());

        // Very low threshold - should pass
        let result = checker
            .check_similarity("hello", "goodbye", Some(0.0))
            .await
            .unwrap();
        assert!(result.passed);

        // Very high threshold - should fail for different texts
        let result = checker
            .check_similarity("hello world", "goodbye universe", Some(0.99))
            .await
            .unwrap();
        assert!(!result.passed);
    }

    #[tokio::test]
    async fn test_semantic_checker_find_most_similar() {
        let checker = SemanticChecker::new(SemanticConfig::default());

        let candidates = vec![
            "programming in python",
            "hello there friend",
            "greeting world hello",
        ];

        let result = checker
            .find_most_similar("hello world", &candidates)
            .await
            .unwrap();

        assert!(result.is_some());
        let (idx, sim) = result.unwrap();
        // "greeting world hello" should be most similar due to word overlap
        assert_eq!(idx, 2);
        assert!(sim > 0.0);
    }

    #[tokio::test]
    async fn test_semantic_checker_find_similar_above_threshold() {
        let checker = SemanticChecker::new(SemanticConfig::default());

        let candidates = vec![
            "hello world greeting",
            "completely different text",
            "world hello there",
        ];

        let results = checker
            .find_similar_above_threshold("hello world", &candidates, Some(0.5))
            .await
            .unwrap();

        // At least one candidate should be above threshold
        assert!(!results.is_empty());
        // Results should be sorted by similarity descending
        if results.len() >= 2 {
            assert!(results[0].1 >= results[1].1);
        }
    }

    #[tokio::test]
    async fn test_semantic_checker_builder() {
        let checker = SemanticCheckerBuilder::new()
            .threshold(0.9)
            .metric(SimilarityMetric::Cosine)
            .local_embeddings(128)
            .store_sources(false)
            .build();

        assert_eq!(checker.config().default_threshold, 0.9);
        assert_eq!(checker.config().metric, SimilarityMetric::Cosine);
        assert!(!checker.config().store_sources);
    }

    #[tokio::test]
    async fn test_similarity_result_builder() {
        let result = SimilarityResult::pass(0.95, 0.8)
            .with_source("source text".to_string())
            .with_target("target text".to_string())
            .with_details("additional info".to_string());

        assert!(result.passed);
        assert_eq!(result.similarity, 0.95);
        assert_eq!(result.source, Some("source text".to_string()));
        assert_eq!(result.target, Some("target text".to_string()));
        assert_eq!(result.details, Some("additional info".to_string()));
    }

    #[tokio::test]
    async fn test_euclidean_metric() {
        let config = SemanticConfig {
            metric: SimilarityMetric::Euclidean,
            ..Default::default()
        };
        let checker = SemanticChecker::new(config);

        let result = checker
            .check_similarity("hello world", "hello world", None)
            .await
            .unwrap();

        // Same text should have distance 0, similarity 1
        assert!(result.similarity > 0.99);
    }

    #[test]
    fn test_similarity_config_default() {
        let config = SemanticConfig::default();
        assert_eq!(config.default_threshold, 0.8);
        assert_eq!(config.metric, SimilarityMetric::Cosine);
        assert!(config.store_sources);
    }

    #[tokio::test]
    async fn test_batch_embed() {
        let checker = SemanticChecker::new(SemanticConfig::default());

        let texts = vec!["hello world", "goodbye universe", "test text"];
        let embeddings = checker.embed_batch(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].dimension(), embeddings[1].dimension());
    }

    // Mutation-killing tests for compute_similarity
    #[test]
    fn test_compute_similarity_euclidean_formula() {
        // Euclidean: 1 / (1 + distance)
        let config = SemanticConfig {
            metric: SimilarityMetric::Euclidean,
            ..Default::default()
        };
        let checker = SemanticChecker::new(config);

        // Create embeddings with known distance
        let a = TextEmbedding::new(vec![0.0, 0.0]);
        let b = TextEmbedding::new(vec![3.0, 4.0]);
        // Distance = 5, similarity = 1/(1+5) = 1/6 ≈ 0.1667

        let sim = checker.compute_similarity(&a, &b);
        assert!(
            (sim - (1.0 / 6.0)).abs() < 0.001,
            "Euclidean similarity should be 1/(1+5) = 1/6, got {}",
            sim
        );
    }

    #[test]
    fn test_compute_similarity_euclidean_zero_distance() {
        let config = SemanticConfig {
            metric: SimilarityMetric::Euclidean,
            ..Default::default()
        };
        let checker = SemanticChecker::new(config);

        let a = TextEmbedding::new(vec![1.0, 2.0, 3.0]);
        // Same embedding - distance = 0, similarity = 1/(1+0) = 1.0
        let sim = checker.compute_similarity(&a, &a);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "Zero distance should give similarity 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_compute_similarity_dot_product() {
        let config = SemanticConfig {
            metric: SimilarityMetric::DotProduct,
            ..Default::default()
        };
        let checker = SemanticChecker::new(config);

        // a = [1, 2, 3], b = [4, 5, 6]
        // dot product = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let a = TextEmbedding::new(vec![1.0, 2.0, 3.0]);
        let b = TextEmbedding::new(vec![4.0, 5.0, 6.0]);

        let sim = checker.compute_similarity(&a, &b);
        assert!(
            (sim - 32.0).abs() < 0.001,
            "Dot product should be 32, got {}",
            sim
        );
    }

    #[test]
    fn test_check_embedding_similarity_threshold() {
        let config = SemanticConfig::default();
        let checker = SemanticChecker::new(config);

        // Create identical embeddings
        let a = TextEmbedding::new(vec![1.0, 0.0, 0.0]);

        // With identical embeddings, similarity = 1.0
        // Test >= threshold boundary
        let result = checker.check_embedding_similarity(&a, &a, Some(1.0));
        assert!(
            result.passed,
            "Similarity 1.0 should pass threshold 1.0 (>= check)"
        );

        let result2 = checker.check_embedding_similarity(&a, &a, Some(1.001));
        assert!(
            !result2.passed,
            "Similarity 1.0 should fail threshold 1.001"
        );
    }

    #[tokio::test]
    async fn test_find_most_similar_empty_candidates() {
        let checker = SemanticChecker::new(SemanticConfig::default());
        let candidates: Vec<&str> = vec![];

        let result = checker
            .find_most_similar("query", &candidates)
            .await
            .unwrap();
        assert!(result.is_none(), "Empty candidates should return None");
    }

    #[tokio::test]
    async fn test_find_most_similar_comparison() {
        let checker = SemanticChecker::new(SemanticConfig::default());

        // candidates where the best match is not the first one
        let candidates = vec![
            "completely unrelated text about cooking",
            "hello world greeting", // Most similar to "hello world"
            "different topic entirely",
        ];

        let result = checker
            .find_most_similar("hello world", &candidates)
            .await
            .unwrap();
        assert!(result.is_some());

        let (idx, sim) = result.unwrap();
        // Should find the most similar (index 1)
        assert_eq!(idx, 1, "Index 1 should be most similar");
        assert!(sim > 0.5, "Should have decent similarity");
    }

    #[tokio::test]
    async fn test_find_similar_above_threshold_boundary() {
        let checker = SemanticChecker::new(SemanticConfig::default());

        // Same text should have similarity 1.0
        let candidates = vec!["hello world", "different text"];

        // Threshold at exactly 1.0 - only identical text should pass
        let results = checker
            .find_similar_above_threshold("hello world", &candidates, Some(1.0))
            .await
            .unwrap();

        // Only "hello world" (similarity 1.0) should pass >= 1.0 threshold
        assert!(
            results
                .iter()
                .any(|(idx, sim)| *idx == 0 && (*sim - 1.0).abs() < 0.001),
            "Identical text should be included at threshold 1.0"
        );
    }

    // Mutation-killing tests for builder methods
    #[tokio::test]
    async fn test_builder_tfidf_embeddings() {
        use crate::embedding::{EmbeddingConfig, TfIdfEmbedder};

        let mut embedder = TfIdfEmbedder::new(EmbeddingConfig::default());
        embedder.fit(&["hello world", "test text"]);

        let checker = SemanticCheckerBuilder::new()
            .tfidf_embeddings(embedder)
            .build();

        // Should use TF-IDF backend
        let result = checker
            .check_similarity("hello world", "hello world", None)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_builder_custom_backend() {
        let backend: Arc<dyn EmbeddingBackend> = Arc::new(LocalEmbedding::new(64));

        let checker = SemanticCheckerBuilder::new()
            .custom_backend(backend)
            .threshold(0.5)
            .build();

        // Should use custom backend
        let result = checker.check_similarity("hello", "hello", None).await;
        assert!(result.is_ok());
        assert!(result.unwrap().passed);
    }

    #[test]
    fn test_builder_methods_return_self() {
        // Verify builder methods return Self for chaining
        let builder = SemanticCheckerBuilder::new();

        // Each method should return the builder
        let _builder = builder
            .threshold(0.5)
            .metric(SimilarityMetric::DotProduct)
            .local_embeddings(64)
            .store_sources(true);

        // If we can build, methods returned Self correctly
        let _checker = _builder.build();
    }
}

// Kani formal verification proofs
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify SimilarityResult::pass() always has passed=true
    #[kani::proof]
    fn verify_similarity_result_pass_is_true() {
        let sim: f64 = kani::any();
        let threshold: f64 = kani::any();
        kani::assume(sim.is_finite());
        kani::assume(threshold.is_finite());

        let result = SimilarityResult::pass(sim, threshold);

        kani::assert(result.passed, "pass() must have passed=true");
        kani::assert(result.similarity == sim, "similarity must be preserved");
        kani::assert(result.threshold == threshold, "threshold must be preserved");
    }

    /// Verify SimilarityResult::fail() always has passed=false
    #[kani::proof]
    fn verify_similarity_result_fail_is_false() {
        let sim: f64 = kani::any();
        let threshold: f64 = kani::any();
        kani::assume(sim.is_finite());
        kani::assume(threshold.is_finite());

        let result = SimilarityResult::fail(sim, threshold);

        kani::assert(!result.passed, "fail() must have passed=false");
        kani::assert(result.similarity == sim, "similarity must be preserved");
        kani::assert(result.threshold == threshold, "threshold must be preserved");
    }

    /// Verify SemanticConfig::default() has valid threshold in [0, 1]
    #[kani::proof]
    fn verify_semantic_config_default_threshold() {
        let config = SemanticConfig::default();

        kani::assert(
            config.default_threshold >= 0.0 && config.default_threshold <= 1.0,
            "Default threshold must be in [0, 1]",
        );
    }

    /// Verify with_source preserves source
    #[kani::proof]
    fn verify_with_source_preserves() {
        let result = SimilarityResult::pass(0.5, 0.5).with_source("test".to_string());

        kani::assert(result.source.is_some(), "source must be set");
        kani::assert(result.source.unwrap() == "test", "source must match input");
    }

    /// Verify with_target preserves target
    #[kani::proof]
    fn verify_with_target_preserves() {
        let result = SimilarityResult::pass(0.5, 0.5).with_target("target".to_string());

        kani::assert(result.target.is_some(), "target must be set");
        kani::assert(
            result.target.unwrap() == "target",
            "target must match input",
        );
    }

    /// Verify with_details preserves details
    #[kani::proof]
    fn verify_with_details_preserves() {
        let result = SimilarityResult::pass(0.5, 0.5).with_details("details".to_string());

        kani::assert(result.details.is_some(), "details must be set");
        kani::assert(
            result.details.unwrap() == "details",
            "details must match input",
        );
    }

    /// Verify Euclidean similarity formula: 1 / (1 + dist) is always in (0, 1]
    /// for non-negative distance
    #[kani::proof]
    fn verify_euclidean_similarity_range() {
        let dist: f64 = kani::any();
        kani::assume(dist >= 0.0);
        kani::assume(dist.is_finite());
        kani::assume(dist <= 1e10); // Reasonable upper bound

        let similarity = 1.0 / (1.0 + dist);

        kani::assert(similarity > 0.0, "Euclidean similarity must be > 0");
        kani::assert(similarity <= 1.0, "Euclidean similarity must be <= 1");
    }

    /// Verify Euclidean similarity: distance 0 yields similarity 1.0
    #[kani::proof]
    fn verify_euclidean_zero_distance_is_one() {
        let similarity = 1.0 / (1.0 + 0.0);
        kani::assert(similarity == 1.0, "Zero distance must give similarity 1.0");
    }

    /// Verify SemanticCheckerBuilder::threshold preserves value
    #[kani::proof]
    fn verify_builder_threshold_preserved() {
        let threshold: f64 = kani::any();
        kani::assume(threshold.is_finite());
        kani::assume(threshold >= 0.0 && threshold <= 1.0);

        let checker = SemanticCheckerBuilder::new().threshold(threshold).build();

        kani::assert(
            checker.config().default_threshold == threshold,
            "Builder must preserve threshold",
        );
    }

    /// Verify SemanticCheckerBuilder::metric preserves value
    #[kani::proof]
    fn verify_builder_metric_preserved() {
        let checker_cosine = SemanticCheckerBuilder::new()
            .metric(SimilarityMetric::Cosine)
            .build();
        kani::assert(
            checker_cosine.config().metric == SimilarityMetric::Cosine,
            "Cosine metric must be preserved",
        );

        let checker_euclidean = SemanticCheckerBuilder::new()
            .metric(SimilarityMetric::Euclidean)
            .build();
        kani::assert(
            checker_euclidean.config().metric == SimilarityMetric::Euclidean,
            "Euclidean metric must be preserved",
        );

        let checker_dot = SemanticCheckerBuilder::new()
            .metric(SimilarityMetric::DotProduct)
            .build();
        kani::assert(
            checker_dot.config().metric == SimilarityMetric::DotProduct,
            "DotProduct metric must be preserved",
        );
    }

    /// Verify SemanticCheckerBuilder::store_sources preserves value
    #[kani::proof]
    fn verify_builder_store_sources_preserved() {
        let checker_true = SemanticCheckerBuilder::new().store_sources(true).build();
        kani::assert(
            checker_true.config().store_sources,
            "store_sources=true must be preserved",
        );

        let checker_false = SemanticCheckerBuilder::new().store_sources(false).build();
        kani::assert(
            !checker_false.config().store_sources,
            "store_sources=false must be preserved",
        );
    }
}
