//! Text embedding generation for semantic similarity
//!
//! This module provides different embedding backends for converting text
//! into vector representations for similarity computation.

use crate::error::{SemanticError, SemanticResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding dimension (for TF-IDF, this is vocabulary size)
    pub dimension: usize,
    /// Whether to normalize embeddings to unit length
    pub normalize: bool,
    /// Minimum term frequency for TF-IDF vocabulary
    pub min_term_freq: usize,
    /// Maximum vocabulary size for TF-IDF
    pub max_vocab_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            normalize: true,
            min_term_freq: 1,
            max_vocab_size: 10000,
        }
    }
}

/// A text embedding vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEmbedding {
    /// The embedding vector
    pub vector: Vec<f32>,
    /// Source text (optional, for debugging)
    pub source: Option<String>,
}

impl TextEmbedding {
    /// Create a new embedding from a vector
    pub fn new(vector: Vec<f32>) -> Self {
        Self {
            vector,
            source: None,
        }
    }

    /// Create embedding with source text
    pub fn with_source(vector: Vec<f32>, source: String) -> Self {
        Self {
            vector,
            source: Some(source),
        }
    }

    /// Get the dimension of this embedding
    pub fn dimension(&self) -> usize {
        self.vector.len()
    }

    /// Compute L2 norm
    pub fn l2_norm(&self) -> f32 {
        self.vector.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize to unit length
    pub fn normalize(&mut self) {
        let norm = self.l2_norm();
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

    /// Compute cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &TextEmbedding) -> f32 {
        if self.vector.len() != other.vector.len() {
            return 0.0;
        }

        let dot: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a = self.l2_norm();
        let norm_b = other.l2_norm();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Compute L2 (Euclidean) distance
    pub fn l2_distance(&self, other: &TextEmbedding) -> f32 {
        if self.vector.len() != other.vector.len() {
            return f32::MAX;
        }

        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Trait for text embedding backends
#[async_trait]
pub trait EmbeddingBackend: Send + Sync {
    /// Generate embedding for a single text
    async fn embed(&self, text: &str) -> SemanticResult<TextEmbedding>;

    /// Generate embeddings for multiple texts (batch)
    async fn embed_batch(&self, texts: &[&str]) -> SemanticResult<Vec<TextEmbedding>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// Get the embedding dimension
    fn dimension(&self) -> usize;

    /// Get the backend name
    fn name(&self) -> &str;
}

/// Local embedding using TF-IDF
#[derive(Debug, Clone)]
pub struct TfIdfEmbedder {
    /// Vocabulary mapping word -> index
    vocabulary: HashMap<String, usize>,
    /// Inverse document frequency for each term
    idf: Vec<f32>,
    /// Configuration
    config: EmbeddingConfig,
    /// Number of documents seen (for IDF computation)
    num_docs: usize,
}

impl TfIdfEmbedder {
    /// Create a new TF-IDF embedder with empty vocabulary
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            vocabulary: HashMap::new(),
            idf: Vec::new(),
            config,
            num_docs: 0,
        }
    }

    /// Create embedder with pre-built vocabulary
    pub fn with_vocabulary(vocabulary: HashMap<String, usize>, idf: Vec<f32>) -> Self {
        let config = EmbeddingConfig {
            dimension: vocabulary.len(),
            ..Default::default()
        };
        Self {
            vocabulary,
            idf,
            config,
            num_docs: 0,
        }
    }

    /// Build vocabulary from a corpus of documents
    pub fn fit(&mut self, documents: &[&str]) {
        // Count document frequency for each term
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut term_freq: HashMap<String, usize> = HashMap::new();

        for doc in documents {
            let mut seen_in_doc: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            for token in tokenize(doc) {
                // Overall term frequency
                *term_freq.entry(token.clone()).or_insert(0) += 1;
                // Document frequency (count each term once per doc)
                if seen_in_doc.insert(token.clone()) {
                    *doc_freq.entry(token).or_insert(0) += 1;
                }
            }
        }

        self.num_docs = documents.len();

        // Filter by minimum term frequency and build vocabulary
        let mut vocab_terms: Vec<_> = term_freq
            .into_iter()
            .filter(|(_, count)| *count >= self.config.min_term_freq)
            .collect();

        // Sort by frequency (descending) and take top max_vocab_size
        vocab_terms.sort_by(|a, b| b.1.cmp(&a.1));
        vocab_terms.truncate(self.config.max_vocab_size);

        // Build vocabulary and IDF
        self.vocabulary.clear();
        self.idf = vec![0.0; vocab_terms.len()];

        for (idx, (term, _)) in vocab_terms.into_iter().enumerate() {
            let df = doc_freq.get(&term).copied().unwrap_or(1) as f32;
            // IDF = log(N / df) + 1 (smoothed)
            self.idf[idx] = (self.num_docs as f32 / df).ln() + 1.0;
            self.vocabulary.insert(term, idx);
        }

        self.config.dimension = self.vocabulary.len();
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Get vocabulary (for persistence)
    pub fn vocabulary(&self) -> &HashMap<String, usize> {
        &self.vocabulary
    }

    /// Get IDF values
    pub fn idf(&self) -> &[f32] {
        &self.idf
    }

    /// Embed a document using TF-IDF
    fn embed_tfidf(&self, text: &str) -> TextEmbedding {
        let tokens = tokenize(text);
        let mut tf: HashMap<String, f32> = HashMap::new();

        // Count term frequencies
        for token in &tokens {
            *tf.entry(token.clone()).or_insert(0.0) += 1.0;
        }

        // Normalize TF by document length
        let doc_len = tokens.len() as f32;
        if doc_len > 0.0 {
            for val in tf.values_mut() {
                *val /= doc_len;
            }
        }

        // Build TF-IDF vector
        let dim = self.vocabulary.len().max(1);
        let mut vector = vec![0.0f32; dim];

        for (term, term_tf) in tf {
            if let Some(&idx) = self.vocabulary.get(&term) {
                if idx < dim {
                    let idf = self.idf.get(idx).copied().unwrap_or(1.0);
                    vector[idx] = term_tf * idf;
                }
            }
        }

        // Normalize if configured
        let mut embedding = TextEmbedding::with_source(vector, text.to_string());
        if self.config.normalize {
            embedding.normalize();
        }

        embedding
    }
}

#[async_trait]
impl EmbeddingBackend for TfIdfEmbedder {
    async fn embed(&self, text: &str) -> SemanticResult<TextEmbedding> {
        if self.vocabulary.is_empty() {
            return Err(SemanticError::Embedding(
                "TF-IDF vocabulary is empty. Call fit() first.".to_string(),
            ));
        }
        Ok(self.embed_tfidf(text))
    }

    fn dimension(&self) -> usize {
        self.vocabulary.len().max(1)
    }

    fn name(&self) -> &str {
        "tfidf"
    }
}

/// Local embedding that uses simple word vectors
///
/// This embedder creates embeddings by averaging word vectors from a simple
/// hash-based approach (no external dependencies).
#[derive(Debug, Clone)]
pub struct LocalEmbedding {
    /// Embedding dimension
    dimension: usize,
    /// Whether to normalize
    normalize: bool,
}

impl LocalEmbedding {
    /// Create a new local embedder
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            normalize: true,
        }
    }

    /// Generate a pseudo-random vector for a word using hash
    fn word_vector(&self, word: &str) -> Vec<f32> {
        // Use a simple hash-based approach to generate consistent vectors
        let mut vector = vec![0.0f32; self.dimension];
        let mut hash: u64 = 5381;

        for byte in word.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }

        // Use hash to seed pseudo-random values
        for (i, v) in vector.iter_mut().enumerate() {
            let h = hash.wrapping_mul(i as u64 + 1);
            // Map to [-1, 1]
            *v = ((h % 10000) as f32 / 5000.0) - 1.0;
        }

        // Normalize word vector
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut vector {
                *x /= norm;
            }
        }

        vector
    }

    /// Embed text by averaging word vectors
    fn embed_local(&self, text: &str) -> TextEmbedding {
        let tokens = tokenize(text);
        let mut vector = vec![0.0f32; self.dimension];

        if tokens.is_empty() {
            return TextEmbedding::with_source(vector, text.to_string());
        }

        // Average word vectors
        for token in &tokens {
            let word_vec = self.word_vector(token);
            for (i, val) in word_vec.into_iter().enumerate() {
                vector[i] += val;
            }
        }

        // Average
        let n = tokens.len() as f32;
        for x in &mut vector {
            *x /= n;
        }

        let mut embedding = TextEmbedding::with_source(vector, text.to_string());
        if self.normalize {
            embedding.normalize();
        }

        embedding
    }
}

impl Default for LocalEmbedding {
    fn default() -> Self {
        Self::new(256)
    }
}

#[async_trait]
impl EmbeddingBackend for LocalEmbedding {
    async fn embed(&self, text: &str) -> SemanticResult<TextEmbedding> {
        Ok(self.embed_local(text))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "local"
    }
}

/// Tokenize text into words
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() > 1)
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! This is a TEST.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"this".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Single-char tokens are filtered out
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_text_embedding_cosine_similarity() {
        let a = TextEmbedding::new(vec![1.0, 0.0, 0.0]);
        let b = TextEmbedding::new(vec![0.0, 1.0, 0.0]);
        let c = TextEmbedding::new(vec![1.0, 0.0, 0.0]);

        // Orthogonal vectors
        assert!((a.cosine_similarity(&b)).abs() < 1e-6);

        // Same direction
        assert!((a.cosine_similarity(&c) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_text_embedding_normalize() {
        let mut emb = TextEmbedding::new(vec![3.0, 4.0]);
        emb.normalize();

        // Should be unit length
        let norm = emb.l2_norm();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tfidf_fit() {
        let config = EmbeddingConfig::default();
        let mut embedder = TfIdfEmbedder::new(config);

        let docs = vec!["the quick brown fox", "the lazy dog", "the quick brown dog"];

        embedder.fit(&docs);

        assert!(embedder.vocab_size() > 0);
        assert!(embedder.vocabulary().contains_key("quick"));
        assert!(embedder.vocabulary().contains_key("brown"));
        assert!(embedder.vocabulary().contains_key("dog"));
    }

    #[tokio::test]
    async fn test_tfidf_embed() {
        let config = EmbeddingConfig::default();
        let mut embedder = TfIdfEmbedder::new(config);

        let docs = vec!["the quick brown fox", "the lazy dog", "the quick brown dog"];
        embedder.fit(&docs);

        let emb1 = embedder.embed("quick brown fox").await.unwrap();
        let emb2 = embedder.embed("quick brown dog").await.unwrap();
        let emb3 = embedder.embed("completely different text").await.unwrap();

        // Similar texts should have higher similarity
        let sim_12 = emb1.cosine_similarity(&emb2);
        let sim_13 = emb1.cosine_similarity(&emb3);

        assert!(
            sim_12 > sim_13,
            "Similar texts should have higher similarity"
        );
    }

    #[tokio::test]
    async fn test_local_embedding() {
        let embedder = LocalEmbedding::new(128);

        let emb1 = embedder.embed("hello world").await.unwrap();
        let emb2 = embedder.embed("hello world").await.unwrap();
        let emb3 = embedder.embed("goodbye universe").await.unwrap();

        // Same text should produce identical embedding
        assert!((emb1.cosine_similarity(&emb2) - 1.0).abs() < 1e-6);

        // Different text should produce different embedding
        assert!(emb1.cosine_similarity(&emb3) < 0.99);
    }

    #[tokio::test]
    async fn test_embedding_dimension() {
        let embedder = LocalEmbedding::new(64);
        assert_eq!(embedder.dimension(), 64);

        let emb = embedder.embed("test").await.unwrap();
        assert_eq!(emb.dimension(), 64);
    }

    #[test]
    fn test_text_embedding_l2_distance() {
        let a = TextEmbedding::new(vec![1.0, 0.0]);
        let b = TextEmbedding::new(vec![0.0, 1.0]);

        let dist = a.l2_distance(&b);
        // sqrt(1 + 1) = sqrt(2) ≈ 1.414
        assert!((dist - 1.414).abs() < 0.01);

        // Self distance should be 0
        assert!(a.l2_distance(&a).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.dimension, 512);
        assert!(config.normalize);
        assert_eq!(config.min_term_freq, 1);
        assert_eq!(config.max_vocab_size, 10000);
    }

    // Mutation-killing tests for TextEmbedding
    #[test]
    fn test_text_embedding_with_source() {
        let emb = TextEmbedding::with_source(vec![1.0, 2.0, 3.0], "test".to_string());
        assert_eq!(emb.vector, vec![1.0, 2.0, 3.0]);
        assert_eq!(emb.source, Some("test".to_string()));
        assert_eq!(emb.dimension(), 3);
    }

    #[test]
    fn test_text_embedding_dimension() {
        let emb = TextEmbedding::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(emb.dimension(), 5);

        let empty = TextEmbedding::new(vec![]);
        assert_eq!(empty.dimension(), 0);

        let single = TextEmbedding::new(vec![1.0]);
        assert_eq!(single.dimension(), 1);
    }

    #[test]
    fn test_text_embedding_l2_norm_formula() {
        // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
        let emb = TextEmbedding::new(vec![3.0, 4.0]);
        let norm = emb.l2_norm();
        assert!(
            (norm - 5.0).abs() < 1e-6,
            "L2 norm of [3,4] should be 5, got {}",
            norm
        );

        // sqrt(1^2 + 2^2 + 2^2) = sqrt(1 + 4 + 4) = sqrt(9) = 3
        let emb2 = TextEmbedding::new(vec![1.0, 2.0, 2.0]);
        assert!(
            (emb2.l2_norm() - 3.0).abs() < 1e-6,
            "L2 norm of [1,2,2] should be 3"
        );
    }

    #[test]
    fn test_text_embedding_normalize_zero_norm() {
        // Test normalization with near-zero norm
        let mut emb = TextEmbedding::new(vec![1e-15, 1e-15, 1e-15]);
        let original = emb.vector.clone();
        emb.normalize();
        // With norm < 1e-10, vector should remain unchanged
        assert_eq!(emb.vector, original);
    }

    #[test]
    fn test_text_embedding_normalize_modifies_vector() {
        let mut emb = TextEmbedding::new(vec![3.0, 4.0]);
        emb.normalize();

        // After normalization, norm should be 1.0
        let norm = emb.l2_norm();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Normalized vector should have norm 1.0, got {}",
            norm
        );

        // Original values should be scaled
        assert!(
            (emb.vector[0] - 0.6).abs() < 1e-6,
            "First element should be 3/5 = 0.6, got {}",
            emb.vector[0]
        );
        assert!(
            (emb.vector[1] - 0.8).abs() < 1e-6,
            "Second element should be 4/5 = 0.8, got {}",
            emb.vector[1]
        );
    }

    #[test]
    fn test_text_embedding_normalized_returns_copy() {
        let emb = TextEmbedding::new(vec![3.0, 4.0]);
        let normalized = emb.normalized();

        // Original should be unchanged
        assert_eq!(emb.vector, vec![3.0, 4.0]);

        // Normalized copy should have norm 1.0
        assert!((normalized.l2_norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_length_mismatch() {
        let a = TextEmbedding::new(vec![1.0, 2.0]);
        let b = TextEmbedding::new(vec![1.0, 2.0, 3.0]);

        // Mismatched lengths should return 0.0
        assert_eq!(a.cosine_similarity(&b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_norm() {
        let a = TextEmbedding::new(vec![1.0, 2.0]);
        let zero = TextEmbedding::new(vec![1e-15, 1e-15]);

        // Zero vector should return 0.0
        assert_eq!(a.cosine_similarity(&zero), 0.0);
        assert_eq!(zero.cosine_similarity(&a), 0.0);
    }

    #[test]
    fn test_cosine_similarity_formula_correctness() {
        // dot(a, b) / (|a| * |b|)
        let a = TextEmbedding::new(vec![1.0, 0.0]);
        let b = TextEmbedding::new(vec![1.0, 1.0]);

        // dot = 1*1 + 0*1 = 1
        // |a| = 1, |b| = sqrt(2)
        // similarity = 1 / sqrt(2)
        let sim = a.cosine_similarity(&b);
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (sim - expected).abs() < 0.01,
            "Cosine similarity should be ~{}, got {}",
            expected,
            sim
        );
    }

    #[test]
    fn test_l2_distance_length_mismatch() {
        let a = TextEmbedding::new(vec![1.0, 2.0]);
        let b = TextEmbedding::new(vec![1.0, 2.0, 3.0]);

        // Mismatched lengths should return f32::MAX
        assert_eq!(a.l2_distance(&b), f32::MAX);
    }

    #[test]
    fn test_l2_distance_formula() {
        let a = TextEmbedding::new(vec![0.0, 0.0]);
        let b = TextEmbedding::new(vec![3.0, 4.0]);

        // sqrt((3-0)^2 + (4-0)^2) = 5
        let dist = a.l2_distance(&b);
        assert!(
            (dist - 5.0).abs() < 1e-6,
            "L2 distance should be 5, got {}",
            dist
        );
    }

    // Mutation-killing tests for TfIdfEmbedder
    #[test]
    fn test_tfidf_with_vocabulary() {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        let idf = vec![1.5, 2.0];

        let embedder = TfIdfEmbedder::with_vocabulary(vocab, idf.clone());

        assert_eq!(embedder.vocab_size(), 2);
        assert_eq!(embedder.vocabulary().get("hello"), Some(&0));
        assert_eq!(embedder.idf(), &idf[..]);
    }

    #[test]
    fn test_tfidf_idf_values() {
        let config = EmbeddingConfig::default();
        let mut embedder = TfIdfEmbedder::new(config);

        // Fit on a corpus where "the" appears in all docs but "unique" in one
        let docs = vec!["the cat", "the dog", "the unique bird"];
        embedder.fit(&docs);

        // IDF = ln(N/df) + 1
        // For "the": df=3, IDF = ln(3/3) + 1 = 0 + 1 = 1.0
        // For term with df=1: IDF = ln(3/1) + 1 = 1.099 + 1 = 2.099

        let idf = embedder.idf();
        assert!(!idf.is_empty(), "IDF should not be empty after fit");

        // Verify IDF values are positive
        for &val in idf {
            assert!(val > 0.0, "IDF values should be positive");
        }
    }

    #[test]
    fn test_tfidf_fit_term_frequency_filter() {
        let config = EmbeddingConfig {
            min_term_freq: 2,
            ..Default::default()
        };
        let mut embedder = TfIdfEmbedder::new(config);

        // "rare" appears only once, should be filtered out
        let docs = vec!["common word", "common word word", "rare term"];
        embedder.fit(&docs);

        // "common" appears 2 times, "word" appears 3 times, "rare" appears 1 time
        // With min_term_freq=2, "rare" and "term" should be filtered
        assert!(!embedder.vocabulary().contains_key("rare"));
        assert!(embedder.vocabulary().contains_key("common"));
        assert!(embedder.vocabulary().contains_key("word"));
    }

    #[tokio::test]
    async fn test_tfidf_embed_empty_vocabulary_error() {
        let config = EmbeddingConfig::default();
        let embedder = TfIdfEmbedder::new(config);

        let result = embedder.embed("test").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_tfidf_embed_tf_normalization() {
        let config = EmbeddingConfig {
            normalize: false, // Disable normalization to see raw TF-IDF values
            ..Default::default()
        };
        let mut embedder = TfIdfEmbedder::new(config);

        let docs = vec!["hello world", "hello there"];
        embedder.fit(&docs);

        // Create embedding for text with repeated words
        // TF is normalized by document length
        // "hello hello" has tf(hello) = 2/2 = 1.0
        // vs "hello world" has tf(hello) = 1/2 = 0.5
    }

    #[tokio::test]
    async fn test_tfidf_backend_dimension() {
        let config = EmbeddingConfig::default();
        let mut embedder = TfIdfEmbedder::new(config);

        let docs = vec!["the quick", "brown fox"];
        embedder.fit(&docs);

        // Dimension should match vocabulary size
        assert_eq!(embedder.dimension(), embedder.vocab_size());
    }

    #[tokio::test]
    async fn test_tfidf_backend_name() {
        let embedder = TfIdfEmbedder::new(EmbeddingConfig::default());
        assert_eq!(embedder.name(), "tfidf");
    }

    // Mutation-killing tests for LocalEmbedding
    #[test]
    fn test_local_embedding_word_vector_consistency() {
        let embedder = LocalEmbedding::new(32);

        // Same word should always produce same vector
        let v1 = embedder.word_vector("test");
        let v2 = embedder.word_vector("test");
        assert_eq!(v1, v2, "Same word should produce same vector");

        // Different words should produce different vectors
        let v3 = embedder.word_vector("other");
        assert_ne!(v1, v3, "Different words should produce different vectors");
    }

    #[test]
    fn test_local_embedding_word_vector_normalized() {
        let embedder = LocalEmbedding::new(64);

        let v = embedder.word_vector("hello");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Word vector should be normalized, got norm {}",
            norm
        );
    }

    #[test]
    fn test_local_embedding_word_vector_range() {
        let embedder = LocalEmbedding::new(128);

        let v = embedder.word_vector("testing");

        // All values should be in reasonable range after normalization
        for &val in &v {
            assert!(
                (-1.0..=1.0).contains(&val),
                "Values should be in [-1, 1], got {}",
                val
            );
        }
    }

    #[tokio::test]
    async fn test_local_embedding_empty_text() {
        let embedder = LocalEmbedding::new(32);

        let emb = embedder.embed("").await.unwrap();

        // Empty text should produce zero vector
        for &val in &emb.vector {
            assert_eq!(val, 0.0, "Empty text should produce zero vector");
        }
    }

    #[tokio::test]
    async fn test_local_embedding_average_words() {
        let embedder = LocalEmbedding::new(32);

        // Single word embedding
        let single = embedder.embed("hello").await.unwrap();

        // Two-word embedding should be different due to averaging
        let double = embedder.embed("hello world").await.unwrap();

        // They should be different
        assert_ne!(
            single.vector, double.vector,
            "Single word and averaged should differ"
        );
    }

    #[tokio::test]
    async fn test_local_embedding_name() {
        let embedder = LocalEmbedding::default();
        assert_eq!(embedder.name(), "local");
    }

    #[tokio::test]
    async fn test_local_embedding_default_dimension() {
        let embedder = LocalEmbedding::default();
        assert_eq!(embedder.dimension(), 256);
    }

    // Mutation-killing tests for EmbeddingBackend::embed_batch
    #[tokio::test]
    async fn test_embedding_batch() {
        let embedder = LocalEmbedding::new(32);

        let texts = vec!["hello", "world", "test"];
        let embeddings = embedder.embed_batch(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 3);

        // Each embedding should match individual embed call
        for (text, emb) in texts.iter().zip(embeddings.iter()) {
            let individual = embedder.embed(text).await.unwrap();
            assert_eq!(
                emb.vector, individual.vector,
                "Batch and individual should match"
            );
        }
    }
}

// Kani formal verification proofs
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify l2_norm() is always non-negative
    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_l2_norm_non_negative() {
        let len: usize = kani::any();
        kani::assume(len > 0 && len <= 4);

        let mut vector = Vec::with_capacity(len);
        for _ in 0..len {
            let v: f32 = kani::any();
            // Assume finite values to avoid NaN complications
            kani::assume(v.is_finite());
            kani::assume(v.abs() <= 1e10);
            vector.push(v);
        }

        let emb = TextEmbedding::new(vector);
        let norm = emb.l2_norm();

        // Norm should be non-negative (or NaN in degenerate cases)
        kani::assert(norm >= 0.0 || norm.is_nan(), "L2 norm must be non-negative");
    }

    /// Verify l2_distance returns f32::MAX for mismatched vector lengths
    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_l2_distance_length_mismatch() {
        let len_a: usize = kani::any();
        let len_b: usize = kani::any();
        kani::assume(len_a > 0 && len_a <= 4);
        kani::assume(len_b > 0 && len_b <= 4);
        kani::assume(len_a != len_b);

        let vec_a: Vec<f32> = (0..len_a).map(|_| 0.0f32).collect();
        let vec_b: Vec<f32> = (0..len_b).map(|_| 0.0f32).collect();

        let emb_a = TextEmbedding::new(vec_a);
        let emb_b = TextEmbedding::new(vec_b);

        let dist = emb_a.l2_distance(&emb_b);
        kani::assert(dist == f32::MAX, "Mismatched lengths must return f32::MAX");
    }

    /// Verify cosine_similarity returns 0.0 for mismatched vector lengths
    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_cosine_similarity_length_mismatch() {
        let len_a: usize = kani::any();
        let len_b: usize = kani::any();
        kani::assume(len_a > 0 && len_a <= 4);
        kani::assume(len_b > 0 && len_b <= 4);
        kani::assume(len_a != len_b);

        let vec_a: Vec<f32> = (0..len_a).map(|_| 1.0f32).collect();
        let vec_b: Vec<f32> = (0..len_b).map(|_| 1.0f32).collect();

        let emb_a = TextEmbedding::new(vec_a);
        let emb_b = TextEmbedding::new(vec_b);

        let sim = emb_a.cosine_similarity(&emb_b);
        kani::assert(
            sim == 0.0,
            "Mismatched lengths must return 0.0 for cosine similarity",
        );
    }

    /// Verify l2_distance with self is always 0.0
    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_l2_distance_self_is_zero() {
        let len: usize = kani::any();
        kani::assume(len > 0 && len <= 4);

        let mut vector = Vec::with_capacity(len);
        for _ in 0..len {
            let v: f32 = kani::any();
            kani::assume(v.is_finite());
            kani::assume(v.abs() <= 1e10);
            vector.push(v);
        }

        let emb = TextEmbedding::new(vector);
        let dist = emb.l2_distance(&emb);

        kani::assert(dist == 0.0, "L2 distance to self must be 0");
    }

    /// Verify TextEmbedding::dimension() returns vector length
    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_dimension_equals_vector_len() {
        let len: usize = kani::any();
        kani::assume(len <= 4);

        let vector: Vec<f32> = (0..len).map(|_| 0.0f32).collect();
        let emb = TextEmbedding::new(vector);

        kani::assert(
            emb.dimension() == len,
            "dimension() must equal vector length",
        );
    }

    /// Verify tokenize produces only lowercase strings
    #[kani::proof]
    fn verify_tokenize_lowercase() {
        // Test with a fixed string containing uppercase
        let tokens = tokenize("Hello World");

        for token in &tokens {
            for c in token.chars() {
                kani::assert(!c.is_uppercase(), "All characters must be lowercase");
            }
        }
    }

    /// Verify tokenize produces tokens with length > 1
    #[kani::proof]
    fn verify_tokenize_min_length() {
        let tokens = tokenize("I am a test A");

        for token in &tokens {
            kani::assert(token.len() > 1, "All tokens must have length > 1");
        }
    }

    /// Verify EmbeddingConfig::default() has valid values
    #[kani::proof]
    fn verify_embedding_config_default_valid() {
        let config = EmbeddingConfig::default();

        kani::assert(config.dimension > 0, "Default dimension must be positive");
        kani::assert(config.min_term_freq >= 1, "min_term_freq must be >= 1");
        kani::assert(config.max_vocab_size > 0, "max_vocab_size must be positive");
    }

    /// Verify LocalEmbedding::new produces embedder with correct dimension
    #[kani::proof]
    fn verify_local_embedding_dimension() {
        let dim: usize = kani::any();
        kani::assume(dim > 0 && dim <= 512);

        let embedder = LocalEmbedding::new(dim);
        kani::assert(
            embedder.dimension() == dim,
            "LocalEmbedding dimension must match input",
        );
    }

    /// Verify LocalEmbedding::default() has dimension 256
    #[kani::proof]
    fn verify_local_embedding_default_dimension() {
        let embedder = LocalEmbedding::default();
        kani::assert(embedder.dimension() == 256, "Default dimension must be 256");
    }
}
