//! Embedding models for vector search

use crate::types::{DocumentChunk, EmbeddedChunk};
use crate::{KnowledgeError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use tracing::{debug, warn};

/// Supported embedding models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EmbeddingModel {
    /// OpenAI text-embedding-3-small (1536 dims)
    #[default]
    OpenAISmall,
    /// OpenAI text-embedding-3-large (3072 dims)
    OpenAILarge,
    /// Local sentence-transformers (384 dims)
    SentenceTransformers,
    /// Cohere embed-english-v3 (1024 dims)
    CohereV3,
    /// Voyage code-2 (1536 dims, optimized for code)
    VoyageCode,
}

impl EmbeddingModel {
    /// Get the embedding dimension for this model
    pub fn dimensions(&self) -> usize {
        match self {
            Self::OpenAISmall => 1536,
            Self::OpenAILarge => 3072,
            Self::SentenceTransformers => 384,
            Self::CohereV3 => 1024,
            Self::VoyageCode => 1536,
        }
    }

    /// Get the model name/ID
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::OpenAISmall => "text-embedding-3-small",
            Self::OpenAILarge => "text-embedding-3-large",
            Self::SentenceTransformers => "all-MiniLM-L6-v2",
            Self::CohereV3 => "embed-english-v3.0",
            Self::VoyageCode => "voyage-code-2",
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CachedEmbedding {
    model: String,
    dimensions: usize,
    embedding: Vec<f32>,
}

/// Embedder for generating vector embeddings
pub struct Embedder {
    model: EmbeddingModel,
    api_key: Option<String>,
    cache_dir: Option<PathBuf>,
}

impl Embedder {
    /// Create a new embedder
    pub fn new(model: EmbeddingModel) -> Self {
        let api_key = match model {
            EmbeddingModel::OpenAISmall | EmbeddingModel::OpenAILarge => {
                std::env::var("OPENAI_API_KEY").ok()
            }
            EmbeddingModel::CohereV3 => std::env::var("COHERE_API_KEY").ok(),
            EmbeddingModel::VoyageCode => std::env::var("VOYAGE_API_KEY").ok(),
            EmbeddingModel::SentenceTransformers => None, // Local model
        };

        Self {
            model,
            api_key,
            cache_dir: None,
        }
    }

    /// Set cache directory for embeddings
    pub fn with_cache(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = Some(cache_dir);
        self
    }

    /// Embed a single text
    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(cached) = self.load_cached_embedding(text) {
            debug!(
                model = self.model.model_id(),
                "Embedding cache hit, using cached vector"
            );
            return Ok(cached);
        }

        let embedding = match self.model {
            EmbeddingModel::OpenAISmall | EmbeddingModel::OpenAILarge => {
                self.embed_openai(text).await
            }
            EmbeddingModel::SentenceTransformers => self.embed_local(text),
            EmbeddingModel::CohereV3 => self.embed_cohere(text).await,
            EmbeddingModel::VoyageCode => self.embed_voyage(text).await,
        }?;

        if self.cache_dir.is_some() {
            self.store_cached_embedding(text, &embedding);
        }

        Ok(embedding)
    }

    /// Embed multiple texts (batched)
    pub async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // For now, embed one at a time
        // In production, batch requests for efficiency
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.embed_text(text).await?);
        }
        Ok(embeddings)
    }

    /// Embed a document chunk
    pub async fn embed_chunk(&self, chunk: &DocumentChunk) -> Result<EmbeddedChunk> {
        let embedding = self.embed_text(&chunk.content).await?;
        Ok(EmbeddedChunk {
            chunk: chunk.clone(),
            embedding,
        })
    }

    /// Embed multiple chunks
    pub async fn embed_chunks(&self, chunks: &[DocumentChunk]) -> Result<Vec<EmbeddedChunk>> {
        let mut embedded = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            embedded.push(self.embed_chunk(chunk).await?);
        }
        Ok(embedded)
    }

    // OpenAI embedding implementation
    async fn embed_openai(&self, text: &str) -> Result<Vec<f32>> {
        let api_key = self
            .api_key
            .as_ref()
            .ok_or_else(|| KnowledgeError::EmbeddingError("OPENAI_API_KEY not set".to_string()))?;

        let client = reqwest::Client::new();
        let response = client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&serde_json::json!({
                "model": self.model.model_id(),
                "input": text,
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(KnowledgeError::EmbeddingError(format!(
                "OpenAI API error: {}",
                response.status()
            )));
        }

        let data: OpenAIEmbeddingResponse = response.json().await?;
        Ok(data.data[0].embedding.clone())
    }

    // Local sentence-transformers embedding (placeholder)
    fn embed_local(&self, text: &str) -> Result<Vec<f32>> {
        // In production, this would use a local model via Python interop or ONNX
        // For now, return a placeholder based on text hash
        let dims = self.model.dimensions();
        let mut embedding = vec![0.0f32; dims];

        // Simple hash-based placeholder
        for (i, c) in text.chars().enumerate() {
            let idx = (i + c as usize) % dims;
            embedding[idx] += (c as u32 as f32 / 1000.0).sin();
        }

        // Normalize
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for x in &mut embedding {
                *x /= magnitude;
            }
        }

        Ok(embedding)
    }

    // Cohere embedding implementation
    async fn embed_cohere(&self, text: &str) -> Result<Vec<f32>> {
        let api_key = self
            .api_key
            .as_ref()
            .ok_or_else(|| KnowledgeError::EmbeddingError("COHERE_API_KEY not set".to_string()))?;

        let client = reqwest::Client::new();
        let response = client
            .post("https://api.cohere.ai/v1/embed")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&serde_json::json!({
                "model": self.model.model_id(),
                "texts": [text],
                "input_type": "search_document",
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(KnowledgeError::EmbeddingError(format!(
                "Cohere API error: {}",
                response.status()
            )));
        }

        let data: CohereEmbeddingResponse = response.json().await?;
        Ok(data.embeddings[0].clone())
    }

    // Voyage embedding implementation
    async fn embed_voyage(&self, text: &str) -> Result<Vec<f32>> {
        let api_key = self
            .api_key
            .as_ref()
            .ok_or_else(|| KnowledgeError::EmbeddingError("VOYAGE_API_KEY not set".to_string()))?;

        let client = reqwest::Client::new();
        let response = client
            .post("https://api.voyageai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&serde_json::json!({
                "model": self.model.model_id(),
                "input": [text],
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(KnowledgeError::EmbeddingError(format!(
                "Voyage API error: {}",
                response.status()
            )));
        }

        let data: VoyageEmbeddingResponse = response.json().await?;
        Ok(data.data[0].embedding.clone())
    }

    /// Return cache file path for a given text if caching is enabled
    fn cache_file_path(&self, text: &str) -> Option<PathBuf> {
        let base = self.cache_dir.as_ref()?;
        let mut hasher = Sha256::new();
        hasher.update(self.model.model_id().as_bytes());
        hasher.update(text.as_bytes());
        let hash = format!("{:x}", hasher.finalize());
        Some(
            base.join(self.model.model_id())
                .join(format!("{hash}.json")),
        )
    }

    /// Try to load an embedding from cache
    fn load_cached_embedding(&self, text: &str) -> Option<Vec<f32>> {
        let path = self.cache_file_path(text)?;
        let data = fs::read_to_string(&path).ok()?;
        let cached: CachedEmbedding = match serde_json::from_str(&data) {
            Ok(entry) => entry,
            Err(err) => {
                warn!(?err, path = %path.display(), "Failed to parse cached embedding");
                return None;
            }
        };

        if cached.model != self.model.model_id() {
            debug!(
                expected = self.model.model_id(),
                cached = cached.model,
                "Cache entry model mismatch, ignoring"
            );
            return None;
        }

        if cached.embedding.len() != self.model.dimensions()
            || cached.dimensions != self.model.dimensions()
        {
            warn!(
                expected = self.model.dimensions(),
                found = cached.embedding.len(),
                "Cached embedding dimensions mismatch"
            );
            return None;
        }

        Some(cached.embedding)
    }

    /// Store an embedding on disk for future reuse
    fn store_cached_embedding(&self, text: &str, embedding: &[f32]) {
        let Some(path) = self.cache_file_path(text) else {
            return;
        };

        if let Some(parent) = path.parent() {
            if let Err(err) = fs::create_dir_all(parent) {
                warn!(
                    ?err,
                    dir = %parent.display(),
                    "Failed to create embedding cache directory"
                );
                return;
            }
        }

        let entry = CachedEmbedding {
            model: self.model.model_id().to_string(),
            dimensions: self.model.dimensions(),
            embedding: embedding.to_vec(),
        };

        match serde_json::to_string(&entry) {
            Ok(serialized) => {
                if let Err(err) = fs::write(&path, serialized) {
                    warn!(?err, path = %path.display(), "Failed to write embedding cache");
                } else {
                    debug!(path = %path.display(), "Stored embedding in cache");
                }
            }
            Err(err) => warn!(?err, "Failed to serialize embedding cache entry"),
        }
    }
}

// API response types
#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbedding>,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbedding {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct CohereEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct VoyageEmbeddingResponse {
    data: Vec<VoyageEmbedding>,
}

#[derive(Debug, Deserialize)]
struct VoyageEmbedding {
    embedding: Vec<f32>,
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_dimensions() {
        assert_eq!(EmbeddingModel::OpenAISmall.dimensions(), 1536);
        assert_eq!(EmbeddingModel::SentenceTransformers.dimensions(), 384);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c)).abs() < 0.001);
    }

    #[test]
    fn test_local_embedding() {
        let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);
        let embedding = embedder.embed_local("test text").unwrap();
        assert_eq!(embedding.len(), 384);

        // Check normalization
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_embedding_cache_written() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Embedder::new(EmbeddingModel::SentenceTransformers)
            .with_cache(dir.path().to_path_buf());
        let text = "cache write test";
        let path = embedder.cache_file_path(text).expect("cache path");
        assert!(!path.exists(), "cache should start empty");

        let embedding = tokio_test::block_on(embedder.embed_text(text)).unwrap();
        assert_eq!(embedding.len(), 384);
        assert!(path.exists(), "embedding cache file should be created");
    }

    #[test]
    fn test_embedding_cache_hit_returns_cached_vector() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Embedder::new(EmbeddingModel::SentenceTransformers)
            .with_cache(dir.path().to_path_buf());
        let text = "cache hit test";
        let path = embedder.cache_file_path(text).expect("cache path");

        let cached = CachedEmbedding {
            model: embedder.model.model_id().to_string(),
            dimensions: embedder.model.dimensions(),
            embedding: vec![1.0f32; embedder.model.dimensions()],
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, serde_json::to_string(&cached).unwrap()).unwrap();

        let embedding = tokio_test::block_on(embedder.embed_text(text)).unwrap();
        assert_eq!(embedding, cached.embedding, "should return cached value");
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Cosine similarity is symmetric: sim(a, b) == sim(b, a)
        #[test]
        fn test_cosine_similarity_symmetric(
            a in prop::collection::vec(-1.0f32..1.0, 10..50),
            b in prop::collection::vec(-1.0f32..1.0, 10..50)
        ) {
            // Make vectors same length
            let len = a.len().min(b.len());
            let a = &a[..len];
            let b = &b[..len];

            let sim_ab = cosine_similarity(a, b);
            let sim_ba = cosine_similarity(b, a);

            prop_assert!((sim_ab - sim_ba).abs() < 1e-6,
                "Cosine similarity should be symmetric: {} vs {}", sim_ab, sim_ba);
        }

        /// Cosine similarity with itself is 1.0 (for non-zero vectors)
        #[test]
        fn test_cosine_similarity_self(
            // Use bounded range to avoid overflow with extreme values
            v in prop::collection::vec(-100.0f32..100.0, 10..50)
                .prop_filter("non-zero vector", |v| v.iter().any(|x| *x != 0.0))
        ) {
            let sim = cosine_similarity(&v, &v);
            // Handle NaN case (from numerical instability with very small values)
            prop_assert!(!sim.is_nan() && (sim - 1.0).abs() < 1e-4,
                "Cosine similarity with self should be 1.0, got {}", sim);
        }

        /// Cosine similarity is bounded between -1 and 1
        #[test]
        fn test_cosine_similarity_bounded(
            a in prop::collection::vec(-10.0f32..10.0, 10..50),
            b in prop::collection::vec(-10.0f32..10.0, 10..50)
        ) {
            let len = a.len().min(b.len());
            let a = &a[..len];
            let b = &b[..len];

            let sim = cosine_similarity(a, b);
            prop_assert!((-1.0 - 1e-6..=1.0 + 1e-6).contains(&sim),
                "Cosine similarity should be in [-1, 1], got {}", sim);
        }

        /// Zero vector has zero similarity with anything
        #[test]
        fn test_cosine_similarity_zero_vector(
            v in prop::collection::vec(-10.0f32..10.0, 10..50)
        ) {
            let zero = vec![0.0f32; v.len()];
            let sim = cosine_similarity(&zero, &v);
            prop_assert!(sim.abs() < 1e-6,
                "Zero vector should have 0 similarity, got {}", sim);
        }

        /// Orthogonal vectors have zero similarity
        #[test]
        fn test_cosine_similarity_orthogonal(
            val in 0.1f32..10.0
        ) {
            let a = vec![val, 0.0, 0.0];
            let b = vec![0.0, val, 0.0];
            let sim = cosine_similarity(&a, &b);
            prop_assert!(sim.abs() < 1e-6,
                "Orthogonal vectors should have 0 similarity, got {}", sim);
        }

        /// Different length vectors return 0 (edge case handling)
        #[test]
        fn test_cosine_similarity_different_lengths(
            len_a in 5usize..20,
            len_b in 21usize..40
        ) {
            let a = vec![1.0f32; len_a];
            let b = vec![1.0f32; len_b];
            let sim = cosine_similarity(&a, &b);
            prop_assert_eq!(sim, 0.0,
                "Different length vectors should return 0");
        }

        /// Local embeddings have correct dimensionality
        #[test]
        fn test_local_embedding_dimensions(
            text in "[a-zA-Z0-9 ]{10,500}"
        ) {
            let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);
            let embedding = embedder.embed_local(&text).unwrap();
            prop_assert_eq!(embedding.len(), 384,
                "SentenceTransformers should produce 384-dim embeddings");
        }

        /// Local embeddings are normalized (magnitude ~= 1)
        #[test]
        fn test_local_embedding_normalized(
            text in "[a-zA-Z0-9 ]{10,500}"
        ) {
            let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);
            let embedding = embedder.embed_local(&text).unwrap();

            let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assert!((magnitude - 1.0).abs() < 0.05,
                "Embedding should be normalized, magnitude = {}", magnitude);
        }

        /// Different texts produce different embeddings
        #[test]
        fn test_local_embedding_different_texts(
            text1 in "[a-z]{20,50}",
            text2 in "[A-Z]{20,50}"
        ) {
            let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);
            let emb1 = embedder.embed_local(&text1).unwrap();
            let emb2 = embedder.embed_local(&text2).unwrap();

            // Embeddings should be different (not identical)
            let same = emb1.iter().zip(emb2.iter()).all(|(a, b)| (a - b).abs() < 1e-10);
            prop_assert!(!same, "Different texts should produce different embeddings");
        }

        /// Same text produces identical embeddings (deterministic)
        #[test]
        fn test_local_embedding_deterministic(
            text in "[a-zA-Z0-9 ]{10,100}"
        ) {
            let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);
            let emb1 = embedder.embed_local(&text).unwrap();
            let emb2 = embedder.embed_local(&text).unwrap();

            for (a, b) in emb1.iter().zip(emb2.iter()) {
                prop_assert!((a - b).abs() < 1e-10,
                    "Same text should produce identical embeddings");
            }
        }

        /// Model IDs are non-empty strings
        #[test]
        fn test_model_ids_non_empty(
            model_idx in 0usize..5
        ) {
            let models = [
                EmbeddingModel::OpenAISmall,
                EmbeddingModel::OpenAILarge,
                EmbeddingModel::SentenceTransformers,
                EmbeddingModel::CohereV3,
                EmbeddingModel::VoyageCode,
            ];
            let model = models[model_idx];
            prop_assert!(!model.model_id().is_empty(),
                "Model ID should not be empty");
        }

        /// Model dimensions are positive
        #[test]
        fn test_model_dimensions_positive(
            model_idx in 0usize..5
        ) {
            let models = [
                EmbeddingModel::OpenAISmall,
                EmbeddingModel::OpenAILarge,
                EmbeddingModel::SentenceTransformers,
                EmbeddingModel::CohereV3,
                EmbeddingModel::VoyageCode,
            ];
            let model = models[model_idx];
            prop_assert!(model.dimensions() > 0,
                "Model dimensions should be positive");
        }
    }
}
