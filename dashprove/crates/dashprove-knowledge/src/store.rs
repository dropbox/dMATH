//! Knowledge and vector stores

use crate::embedding::cosine_similarity;
use crate::types::{
    ArxivPaper, ContentType, DocumentChunk, EmbeddedChunk, GithubRepo, KnowledgeQuery, ScoredChunk,
    ScoredPaper, ScoredRepo, SearchResult,
};
use crate::Result;
use dashprove_backends::BackendId;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::fs;
use tracing::debug;

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify VectorFilter matches when no filters are set
    #[kani::proof]
    fn verify_filter_matches_no_constraints() {
        let filter = VectorFilter {
            backend: None,
            content_type: None,
            tags: vec![],
        };
        let chunk = DocumentChunk {
            id: String::new(),
            document_id: String::new(),
            chunk_index: 0,
            content: String::new(),
            backend: Some(BackendId::Lean4),
            content_type: ContentType::General,
            section_path: vec![],
            token_count: 0,
        };
        kani::assert(
            filter.matches(&chunk),
            "empty filter should match any chunk",
        );
    }

    /// Verify VectorFilter matches when backend matches
    #[kani::proof]
    fn verify_filter_matches_backend() {
        let filter = VectorFilter {
            backend: Some(BackendId::Lean4),
            content_type: None,
            tags: vec![],
        };
        let chunk = DocumentChunk {
            id: String::new(),
            document_id: String::new(),
            chunk_index: 0,
            content: String::new(),
            backend: Some(BackendId::Lean4),
            content_type: ContentType::General,
            section_path: vec![],
            token_count: 0,
        };
        kani::assert(
            filter.matches(&chunk),
            "filter should match when backend matches",
        );
    }

    /// Verify VectorFilter rejects when backend differs
    #[kani::proof]
    fn verify_filter_rejects_different_backend() {
        let filter = VectorFilter {
            backend: Some(BackendId::Lean4),
            content_type: None,
            tags: vec![],
        };
        let chunk = DocumentChunk {
            id: String::new(),
            document_id: String::new(),
            chunk_index: 0,
            content: String::new(),
            backend: Some(BackendId::Coq),
            content_type: ContentType::General,
            section_path: vec![],
            token_count: 0,
        };
        kani::assert!(
            !filter.matches(&chunk),
            "filter should reject when backend differs"
        );
    }

    /// Verify VectorFilter rejects when chunk has no backend but filter requires one
    #[kani::proof]
    fn verify_filter_rejects_missing_backend() {
        let filter = VectorFilter {
            backend: Some(BackendId::Lean4),
            content_type: None,
            tags: vec![],
        };
        let chunk = DocumentChunk {
            id: String::new(),
            document_id: String::new(),
            chunk_index: 0,
            content: String::new(),
            backend: None,
            content_type: ContentType::General,
            section_path: vec![],
            token_count: 0,
        };
        kani::assert!(
            !filter.matches(&chunk),
            "filter should reject when chunk has no backend"
        );
    }

    /// Verify VectorFilter matches when content_type matches
    #[kani::proof]
    fn verify_filter_matches_content_type() {
        let filter = VectorFilter {
            backend: None,
            content_type: Some(ContentType::Tutorial),
            tags: vec![],
        };
        let chunk = DocumentChunk {
            id: String::new(),
            document_id: String::new(),
            chunk_index: 0,
            content: String::new(),
            backend: None,
            content_type: ContentType::Tutorial,
            section_path: vec![],
            token_count: 0,
        };
        kani::assert(
            filter.matches(&chunk),
            "filter should match when content_type matches",
        );
    }

    /// Verify VectorFilter rejects when content_type differs
    #[kani::proof]
    fn verify_filter_rejects_different_content_type() {
        let filter = VectorFilter {
            backend: None,
            content_type: Some(ContentType::Tutorial),
            tags: vec![],
        };
        let chunk = DocumentChunk {
            id: String::new(),
            document_id: String::new(),
            chunk_index: 0,
            content: String::new(),
            backend: None,
            content_type: ContentType::Api,
            section_path: vec![],
            token_count: 0,
        };
        kani::assert!(
            !filter.matches(&chunk),
            "filter should reject when content_type differs"
        );
    }

    /// Verify VectorFilter matches when both backend and content_type match
    #[kani::proof]
    fn verify_filter_matches_both() {
        let filter = VectorFilter {
            backend: Some(BackendId::Coq),
            content_type: Some(ContentType::Reference),
            tags: vec![],
        };
        let chunk = DocumentChunk {
            id: String::new(),
            document_id: String::new(),
            chunk_index: 0,
            content: String::new(),
            backend: Some(BackendId::Coq),
            content_type: ContentType::Reference,
            section_path: vec![],
            token_count: 0,
        };
        kani::assert(
            filter.matches(&chunk),
            "filter should match when both fields match",
        );
    }

    /// Verify VectorFilter rejects when only backend matches but content_type differs
    #[kani::proof]
    fn verify_filter_rejects_partial_match() {
        let filter = VectorFilter {
            backend: Some(BackendId::Coq),
            content_type: Some(ContentType::Reference),
            tags: vec![],
        };
        let chunk = DocumentChunk {
            id: String::new(),
            document_id: String::new(),
            chunk_index: 0,
            content: String::new(),
            backend: Some(BackendId::Coq),
            content_type: ContentType::Tutorial,
            section_path: vec![],
            token_count: 0,
        };
        kani::assert!(
            !filter.matches(&chunk),
            "filter should reject partial match"
        );
    }

    /// Verify VectorStore len returns correct count
    #[kani::proof]
    fn verify_vector_store_len_empty() {
        let store = VectorStore::new(10);
        kani::assert(store.len() == 0, "new store should have len 0");
        kani::assert(store.is_empty(), "new store should be empty");
    }

    /// Verify StoreStats fields are initialized correctly
    #[kani::proof]
    fn verify_store_stats_fields() {
        let stats = StoreStats {
            chunk_count: 5,
            paper_count: 3,
            repo_count: 2,
        };
        kani::assert(stats.chunk_count == 5, "chunk_count should be preserved");
        kani::assert(stats.paper_count == 3, "paper_count should be preserved");
        kani::assert(stats.repo_count == 2, "repo_count should be preserved");
    }
}

/// In-memory vector store for semantic search
pub struct VectorStore {
    /// Embedded chunks
    chunks: Vec<EmbeddedChunk>,
    /// Dimension of embeddings
    dimensions: usize,
}

impl VectorStore {
    /// Create a new vector store
    pub fn new(dimensions: usize) -> Self {
        Self {
            chunks: Vec::new(),
            dimensions,
        }
    }

    /// Add embedded chunks
    pub fn add_chunks(&mut self, chunks: Vec<EmbeddedChunk>) {
        for chunk in chunks {
            if chunk.embedding.len() == self.dimensions {
                self.chunks.push(chunk);
            }
        }
    }

    /// Search for similar chunks
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filter: Option<VectorFilter>,
    ) -> Vec<ScoredChunk> {
        let mut scored: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, chunk)| {
                if let Some(ref f) = filter {
                    f.matches(&chunk.chunk)
                } else {
                    true
                }
            })
            .map(|(idx, chunk)| {
                let score = cosine_similarity(query_embedding, &chunk.embedding);
                (idx, score)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(top_k)
            .map(|(idx, score)| ScoredChunk {
                chunk: self.chunks[idx].chunk.clone(),
                score,
            })
            .collect()
    }

    /// Get total chunk count
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Save to disk
    pub async fn save(&self, path: &PathBuf) -> Result<()> {
        let data = VectorStoreData {
            dimensions: self.dimensions,
            chunks: self.chunks.clone(),
        };
        let json = serde_json::to_string(&data)?;
        fs::write(path, json).await?;
        Ok(())
    }

    /// Load from disk
    pub async fn load(path: &PathBuf) -> Result<Self> {
        let json = fs::read_to_string(path).await?;
        let data: VectorStoreData = serde_json::from_str(&json)?;
        Ok(Self {
            dimensions: data.dimensions,
            chunks: data.chunks,
        })
    }
}

/// Filter for vector search
#[derive(Debug, Clone)]
pub struct VectorFilter {
    pub backend: Option<BackendId>,
    pub content_type: Option<ContentType>,
    pub tags: Vec<String>,
}

impl VectorFilter {
    pub fn matches(&self, chunk: &DocumentChunk) -> bool {
        if let Some(backend) = self.backend {
            if chunk.backend != Some(backend) {
                return false;
            }
        }
        if let Some(content_type) = self.content_type {
            if chunk.content_type != content_type {
                return false;
            }
        }
        true
    }
}

/// Serializable vector store data
#[derive(Debug, Serialize, Deserialize)]
struct VectorStoreData {
    dimensions: usize,
    chunks: Vec<EmbeddedChunk>,
}

/// Complete knowledge store with documents, papers, and repos
pub struct KnowledgeStore {
    /// Vector store for semantic search
    pub vector_store: VectorStore,
    /// ArXiv papers
    papers: Vec<ArxivPaper>,
    /// GitHub repositories
    repos: Vec<GithubRepo>,
    /// Storage directory
    store_dir: PathBuf,
}

impl KnowledgeStore {
    /// Create a new knowledge store
    pub fn new(store_dir: PathBuf, embedding_dimensions: usize) -> Self {
        Self {
            vector_store: VectorStore::new(embedding_dimensions),
            papers: Vec::new(),
            repos: Vec::new(),
            store_dir,
        }
    }

    /// Add papers to the store
    pub fn add_papers(&mut self, papers: Vec<ArxivPaper>) {
        self.papers.extend(papers);
    }

    /// Add repositories to the store
    pub fn add_repos(&mut self, repos: Vec<GithubRepo>) {
        self.repos.extend(repos);
    }

    /// Search the knowledge store
    pub fn search(&self, query: &KnowledgeQuery, query_embedding: &[f32]) -> SearchResult {
        // Search vector store
        let filter = VectorFilter {
            backend: query.backend,
            content_type: query.content_type,
            tags: query.tags.clone(),
        };

        let chunks = self
            .vector_store
            .search(query_embedding, query.limit, Some(filter));

        // Search papers (simple text matching for now)
        let papers = if query.include_papers {
            self.search_papers(&query.text, query.limit)
        } else {
            vec![]
        };

        // Search repos
        let repos = if query.include_repos {
            self.search_repos(&query.text, query.limit)
        } else {
            vec![]
        };

        SearchResult {
            chunks,
            papers,
            repos,
        }
    }

    /// Simple text search for papers
    fn search_papers(&self, query: &str, limit: usize) -> Vec<ScoredPaper> {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<(usize, f32)> = self
            .papers
            .iter()
            .enumerate()
            .map(|(idx, paper)| {
                let text = format!(
                    "{} {} {}",
                    paper.title,
                    paper.abstract_text,
                    paper.categories.join(" ")
                )
                .to_lowercase();

                let score: f32 = query_terms
                    .iter()
                    .filter(|term| text.contains(*term))
                    .count() as f32
                    / query_terms.len().max(1) as f32;

                (idx, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(limit)
            .map(|(idx, score)| ScoredPaper {
                paper: self.papers[idx].clone(),
                score,
            })
            .collect()
    }

    /// Simple text search for repos
    fn search_repos(&self, query: &str, limit: usize) -> Vec<ScoredRepo> {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<(usize, f32)> = self
            .repos
            .iter()
            .enumerate()
            .map(|(idx, repo)| {
                let text = format!(
                    "{} {} {}",
                    repo.full_name,
                    repo.description.as_deref().unwrap_or(""),
                    repo.topics.join(" ")
                )
                .to_lowercase();

                let score: f32 = query_terms
                    .iter()
                    .filter(|term| text.contains(*term))
                    .count() as f32
                    / query_terms.len().max(1) as f32;

                // Boost by stars (logarithmic)
                let star_boost = (repo.stars as f32 + 1.0).log10() / 5.0;

                (idx, score + star_boost * 0.2)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(limit)
            .map(|(idx, score)| ScoredRepo {
                repo: self.repos[idx].clone(),
                score,
            })
            .collect()
    }

    /// Save the entire store
    pub async fn save(&self) -> Result<()> {
        fs::create_dir_all(&self.store_dir).await?;

        // Save vector store
        let vector_path = self.store_dir.join("vectors.json");
        self.vector_store.save(&vector_path).await?;

        // Save papers
        let papers_path = self.store_dir.join("papers.json");
        let papers_json = serde_json::to_string(&self.papers)?;
        fs::write(&papers_path, papers_json).await?;

        // Save repos
        let repos_path = self.store_dir.join("repos.json");
        let repos_json = serde_json::to_string(&self.repos)?;
        fs::write(&repos_path, repos_json).await?;

        debug!("Saved knowledge store to {:?}", self.store_dir);
        Ok(())
    }

    /// Load the store
    pub async fn load(store_dir: PathBuf) -> Result<Self> {
        let vector_path = store_dir.join("vectors.json");
        let vector_store = VectorStore::load(&vector_path).await?;

        let papers_path = store_dir.join("papers.json");
        let papers_json = fs::read_to_string(&papers_path).await?;
        let papers: Vec<ArxivPaper> = serde_json::from_str(&papers_json)?;

        let repos_path = store_dir.join("repos.json");
        let repos_json = fs::read_to_string(&repos_path).await?;
        let repos: Vec<GithubRepo> = serde_json::from_str(&repos_json)?;

        Ok(Self {
            vector_store,
            papers,
            repos,
            store_dir,
        })
    }

    /// Get statistics
    pub fn stats(&self) -> StoreStats {
        StoreStats {
            chunk_count: self.vector_store.len(),
            paper_count: self.papers.len(),
            repo_count: self.repos.len(),
        }
    }
}

/// Store statistics
#[derive(Debug, Clone)]
pub struct StoreStats {
    pub chunk_count: usize,
    pub paper_count: usize,
    pub repo_count: usize,
}

/// Semantic search engine for ArXiv papers
///
/// Provides vector-based semantic search over paper abstracts and extracted text.
/// Papers are embedded and searched using cosine similarity.
pub struct PaperSearchEngine {
    /// Embedded papers (using abstract + title as text)
    papers: Vec<EmbeddedPaper>,
    /// Dimension of embeddings
    dimensions: usize,
}

/// A paper with its embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedPaper {
    /// The paper
    pub paper: ArxivPaper,
    /// Embedding of title + abstract
    pub abstract_embedding: Vec<f32>,
    /// Embedding of extracted full text (if available)
    pub fulltext_embedding: Option<Vec<f32>>,
}

impl PaperSearchEngine {
    /// Create a new paper search engine
    pub fn new(dimensions: usize) -> Self {
        Self {
            papers: Vec::new(),
            dimensions,
        }
    }

    /// Add an embedded paper
    pub fn add_paper(&mut self, paper: EmbeddedPaper) {
        if paper.abstract_embedding.len() == self.dimensions {
            self.papers.push(paper);
        }
    }

    /// Add multiple embedded papers
    pub fn add_papers(&mut self, papers: Vec<EmbeddedPaper>) {
        for paper in papers {
            self.add_paper(paper);
        }
    }

    /// Search for papers semantically similar to the query
    ///
    /// # Arguments
    /// * `query_embedding` - The embedding of the query text
    /// * `top_k` - Maximum number of results to return
    /// * `use_fulltext` - Whether to also search full text embeddings
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        use_fulltext: bool,
    ) -> Vec<ScoredPaper> {
        if query_embedding.len() != self.dimensions {
            return vec![];
        }

        let mut scored: Vec<(usize, f32)> = self
            .papers
            .iter()
            .enumerate()
            .map(|(idx, embedded)| {
                // Score against abstract embedding
                let abstract_score =
                    cosine_similarity(query_embedding, &embedded.abstract_embedding);

                // Optionally score against full text embedding
                let fulltext_score = if use_fulltext {
                    embedded
                        .fulltext_embedding
                        .as_ref()
                        .map(|ft| cosine_similarity(query_embedding, ft))
                        .unwrap_or(0.0)
                } else {
                    0.0
                };

                // Take the maximum of the two scores
                let score = abstract_score.max(fulltext_score);
                (idx, score)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(top_k)
            .map(|(idx, score)| ScoredPaper {
                paper: self.papers[idx].paper.clone(),
                score,
            })
            .collect()
    }

    /// Search by category filter
    pub fn search_by_category(
        &self,
        query_embedding: &[f32],
        categories: &[String],
        top_k: usize,
    ) -> Vec<ScoredPaper> {
        if query_embedding.len() != self.dimensions || categories.is_empty() {
            return self.search(query_embedding, top_k, false);
        }

        let mut scored: Vec<(usize, f32)> = self
            .papers
            .iter()
            .enumerate()
            .filter(|(_, embedded)| {
                // Check if paper is in any of the specified categories
                embedded
                    .paper
                    .categories
                    .iter()
                    .any(|c| categories.contains(c))
            })
            .map(|(idx, embedded)| {
                let score = cosine_similarity(query_embedding, &embedded.abstract_embedding);
                (idx, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(top_k)
            .map(|(idx, score)| ScoredPaper {
                paper: self.papers[idx].paper.clone(),
                score,
            })
            .collect()
    }

    /// Find papers similar to a given paper
    pub fn find_similar(&self, arxiv_id: &str, top_k: usize) -> Vec<ScoredPaper> {
        // Find the paper's embedding
        let source_embedding = self
            .papers
            .iter()
            .find(|p| p.paper.arxiv_id == arxiv_id)
            .map(|p| &p.abstract_embedding);

        match source_embedding {
            Some(embedding) => {
                let mut results = self.search(embedding, top_k + 1, false);
                // Remove the source paper from results
                results.retain(|r| r.paper.arxiv_id != arxiv_id);
                results.truncate(top_k);
                results
            }
            None => vec![],
        }
    }

    /// Get total paper count
    pub fn len(&self) -> usize {
        self.papers.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.papers.is_empty()
    }

    /// Save to disk
    pub async fn save(&self, path: &PathBuf) -> Result<()> {
        let data = PaperSearchEngineData {
            dimensions: self.dimensions,
            papers: self.papers.clone(),
        };
        let json = serde_json::to_string(&data)?;
        fs::write(path, json).await?;
        Ok(())
    }

    /// Load from disk
    pub async fn load(path: &PathBuf) -> Result<Self> {
        let json = fs::read_to_string(path).await?;
        let data: PaperSearchEngineData = serde_json::from_str(&json)?;
        Ok(Self {
            dimensions: data.dimensions,
            papers: data.papers,
        })
    }

    /// Get all papers (without embeddings)
    pub fn get_papers(&self) -> Vec<&ArxivPaper> {
        self.papers.iter().map(|p| &p.paper).collect()
    }

    /// Get a paper by ArXiv ID
    pub fn get_paper(&self, arxiv_id: &str) -> Option<&ArxivPaper> {
        self.papers
            .iter()
            .find(|p| p.paper.arxiv_id == arxiv_id)
            .map(|p| &p.paper)
    }
}

/// Serializable paper search engine data
#[derive(Debug, Serialize, Deserialize)]
struct PaperSearchEngineData {
    dimensions: usize,
    papers: Vec<EmbeddedPaper>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_store_search() {
        let mut store = VectorStore::new(3);

        store.chunks.push(EmbeddedChunk {
            chunk: DocumentChunk {
                id: "1".to_string(),
                document_id: "doc1".to_string(),
                chunk_index: 0,
                content: "test content".to_string(),
                backend: None,
                content_type: ContentType::General,
                section_path: vec![],
                token_count: 2,
            },
            embedding: vec![1.0, 0.0, 0.0],
        });

        store.chunks.push(EmbeddedChunk {
            chunk: DocumentChunk {
                id: "2".to_string(),
                document_id: "doc2".to_string(),
                chunk_index: 0,
                content: "other content".to_string(),
                backend: None,
                content_type: ContentType::General,
                section_path: vec![],
                token_count: 2,
            },
            embedding: vec![0.0, 1.0, 0.0],
        });

        // Query similar to first chunk
        let results = store.search(&[1.0, 0.0, 0.0], 1, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, "1");
        assert!((results[0].score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_filter() {
        let filter = VectorFilter {
            backend: Some(BackendId::Lean4),
            content_type: None,
            tags: vec![],
        };

        let chunk_match = DocumentChunk {
            id: "1".to_string(),
            document_id: "doc".to_string(),
            chunk_index: 0,
            content: "".to_string(),
            backend: Some(BackendId::Lean4),
            content_type: ContentType::General,
            section_path: vec![],
            token_count: 0,
        };

        let chunk_no_match = DocumentChunk {
            id: "2".to_string(),
            document_id: "doc".to_string(),
            chunk_index: 0,
            content: "".to_string(),
            backend: Some(BackendId::Coq),
            content_type: ContentType::General,
            section_path: vec![],
            token_count: 0,
        };

        assert!(filter.matches(&chunk_match));
        assert!(!filter.matches(&chunk_no_match));
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    fn make_chunk(id: &str, doc_id: &str, backend: Option<BackendId>) -> DocumentChunk {
        DocumentChunk {
            id: id.to_string(),
            document_id: doc_id.to_string(),
            chunk_index: 0,
            content: format!("Content for {}", id),
            backend,
            content_type: ContentType::General,
            section_path: vec![],
            token_count: 10,
        }
    }

    fn make_embedded_chunk(id: &str, embedding: Vec<f32>) -> EmbeddedChunk {
        EmbeddedChunk {
            chunk: make_chunk(id, "doc", None),
            embedding,
        }
    }

    proptest! {
        /// Vector store respects top_k limit
        #[test]
        fn test_vector_store_respects_top_k(
            chunk_count in 5usize..20,
            top_k in 1usize..10
        ) {
            let dims = 10;
            let mut store = VectorStore::new(dims);

            for i in 0..chunk_count {
                let mut embedding = vec![0.0f32; dims];
                embedding[i % dims] = 1.0;
                store.chunks.push(make_embedded_chunk(&format!("chunk{}", i), embedding));
            }

            let query = vec![1.0f32; dims];
            let results = store.search(&query, top_k, None);

            prop_assert!(results.len() <= top_k,
                "Results ({}) should not exceed top_k ({})", results.len(), top_k);
            prop_assert!(results.len() <= chunk_count,
                "Results ({}) should not exceed chunk_count ({})", results.len(), chunk_count);
        }

        /// Results are sorted by score descending
        #[test]
        fn test_vector_store_results_sorted(
            chunk_count in 3usize..15,
        ) {
            let dims = 5;
            let mut store = VectorStore::new(dims);

            for i in 0..chunk_count {
                let embedding: Vec<f32> = (0..dims)
                    .map(|d| if d == i % dims { 1.0 } else { 0.0 })
                    .collect();
                store.chunks.push(make_embedded_chunk(&format!("chunk{}", i), embedding));
            }

            let query = vec![1.0f32; dims];
            let results = store.search(&query, chunk_count, None);

            for window in results.windows(2) {
                prop_assert!(window[0].score >= window[1].score,
                    "Results should be sorted by score descending: {} < {}",
                    window[0].score, window[1].score);
            }
        }

        /// Empty store returns empty results
        #[test]
        fn test_empty_store_empty_results(
            top_k in 1usize..100
        ) {
            let store = VectorStore::new(10);
            let query = vec![1.0f32; 10];
            let results = store.search(&query, top_k, None);

            prop_assert!(results.is_empty(),
                "Empty store should return empty results");
        }

        /// Adding chunks increases store size
        #[test]
        fn test_store_size_increases(
            initial_count in 1usize..10,
            add_count in 1usize..10
        ) {
            let dims = 5;
            let mut store = VectorStore::new(dims);

            // Add initial chunks
            for i in 0..initial_count {
                store.chunks.push(make_embedded_chunk(&format!("init{}", i), vec![1.0; dims]));
            }
            let initial_size = store.len();

            // Add more chunks
            let mut new_chunks = Vec::new();
            for i in 0..add_count {
                new_chunks.push(make_embedded_chunk(&format!("new{}", i), vec![1.0; dims]));
            }
            store.add_chunks(new_chunks);

            prop_assert_eq!(store.len(), initial_size + add_count,
                "Store size should increase by add_count");
        }

        /// Wrong dimension embeddings are rejected
        #[test]
        fn test_wrong_dimension_rejected(
            store_dims in 5usize..20,
            wrong_dims in 21usize..40
        ) {
            let mut store = VectorStore::new(store_dims);
            let wrong_embedding = vec![1.0f32; wrong_dims];

            store.add_chunks(vec![make_embedded_chunk("wrong", wrong_embedding)]);

            prop_assert!(store.is_empty(),
                "Wrong dimension embeddings should be rejected");
        }

        /// Backend filter works correctly
        #[test]
        fn test_backend_filter(
            lean_count in 1usize..5,
            coq_count in 1usize..5
        ) {
            let dims = 3;
            let mut store = VectorStore::new(dims);

            // Add Lean4 chunks
            for i in 0..lean_count {
                store.chunks.push(EmbeddedChunk {
                    chunk: make_chunk(&format!("lean{}", i), "doc", Some(BackendId::Lean4)),
                    embedding: vec![1.0, 0.0, 0.0],
                });
            }

            // Add Coq chunks
            for i in 0..coq_count {
                store.chunks.push(EmbeddedChunk {
                    chunk: make_chunk(&format!("coq{}", i), "doc", Some(BackendId::Coq)),
                    embedding: vec![0.0, 1.0, 0.0],
                });
            }

            // Filter for Lean4 only
            let filter = VectorFilter {
                backend: Some(BackendId::Lean4),
                content_type: None,
                tags: vec![],
            };

            let query = vec![0.5, 0.5, 0.0];
            let results = store.search(&query, 100, Some(filter));

            prop_assert_eq!(results.len(), lean_count,
                "Should only return Lean4 chunks");
            for result in &results {
                prop_assert_eq!(result.chunk.backend, Some(BackendId::Lean4),
                    "All results should be Lean4");
            }
        }

        /// Content type filter works correctly
        #[test]
        fn test_content_type_filter(
            tutorial_count in 1usize..5,
            api_count in 1usize..5
        ) {
            let dims = 3;
            let mut store = VectorStore::new(dims);

            // Add Tutorial chunks
            for i in 0..tutorial_count {
                store.chunks.push(EmbeddedChunk {
                    chunk: DocumentChunk {
                        id: format!("tut{}", i),
                        document_id: "doc".to_string(),
                        chunk_index: 0,
                        content: "tutorial content".to_string(),
                        backend: None,
                        content_type: ContentType::Tutorial,
                        section_path: vec![],
                        token_count: 10,
                    },
                    embedding: vec![1.0, 0.0, 0.0],
                });
            }

            // Add API chunks
            for i in 0..api_count {
                store.chunks.push(EmbeddedChunk {
                    chunk: DocumentChunk {
                        id: format!("api{}", i),
                        document_id: "doc".to_string(),
                        chunk_index: 0,
                        content: "api content".to_string(),
                        backend: None,
                        content_type: ContentType::Api,
                        section_path: vec![],
                        token_count: 10,
                    },
                    embedding: vec![0.0, 1.0, 0.0],
                });
            }

            // Filter for Tutorial only
            let filter = VectorFilter {
                backend: None,
                content_type: Some(ContentType::Tutorial),
                tags: vec![],
            };

            let query = vec![0.5, 0.5, 0.0];
            let results = store.search(&query, 100, Some(filter));

            prop_assert_eq!(results.len(), tutorial_count,
                "Should only return Tutorial chunks");
            for result in &results {
                prop_assert_eq!(result.chunk.content_type, ContentType::Tutorial,
                    "All results should be Tutorial type");
            }
        }

        /// Store stats are accurate
        #[test]
        fn test_store_stats_accurate(
            chunk_count in 0usize..20
        ) {
            let dims = 5;
            let mut store = VectorStore::new(dims);

            for i in 0..chunk_count {
                store.chunks.push(make_embedded_chunk(&format!("c{}", i), vec![1.0; dims]));
            }

            prop_assert_eq!(store.len(), chunk_count);
            prop_assert_eq!(store.is_empty(), chunk_count == 0);
        }

        /// Identical query returns highest score for matching vector
        #[test]
        fn test_identical_query_highest_score(
            other_count in 1usize..10
        ) {
            let dims = 5;
            let mut store = VectorStore::new(dims);

            // Add a specific target vector
            let target = vec![1.0, 0.0, 0.0, 0.0, 0.0];
            store.chunks.push(make_embedded_chunk("target", target.clone()));

            // Add other random-ish vectors
            for i in 0..other_count {
                let other: Vec<f32> = (0..dims)
                    .map(|d| if d == (i + 1) % dims { 1.0 } else { 0.0 })
                    .collect();
                store.chunks.push(make_embedded_chunk(&format!("other{}", i), other));
            }

            let results = store.search(&target, 1, None);

            prop_assert_eq!(results.len(), 1);
            prop_assert_eq!(&results[0].chunk.id, "target",
                "Identical vector should be top result");
            prop_assert!((results[0].score - 1.0).abs() < 1e-5,
                "Score should be ~1.0 for identical vector");
        }
    }
}

#[cfg(test)]
mod paper_search_tests {
    use super::*;
    use chrono::Utc;

    fn make_test_paper(id: &str, title: &str, abstract_text: &str) -> ArxivPaper {
        ArxivPaper {
            arxiv_id: id.to_string(),
            title: title.to_string(),
            authors: vec!["Test Author".to_string()],
            abstract_text: abstract_text.to_string(),
            primary_category: "cs.LO".to_string(),
            categories: vec!["cs.LO".to_string(), "cs.PL".to_string()],
            published: Utc::now(),
            updated: Utc::now(),
            pdf_url: format!("https://arxiv.org/pdf/{}.pdf", id),
            comment: None,
            local_path: None,
            extracted_text: None,
            tags: vec![],
        }
    }

    fn make_embedded_paper(id: &str, embedding: Vec<f32>) -> EmbeddedPaper {
        EmbeddedPaper {
            paper: make_test_paper(id, &format!("Paper {}", id), "Abstract text"),
            abstract_embedding: embedding,
            fulltext_embedding: None,
        }
    }

    #[test]
    fn test_paper_search_engine_new() {
        let engine = PaperSearchEngine::new(384);
        assert!(engine.is_empty());
        assert_eq!(engine.len(), 0);
    }

    #[test]
    fn test_paper_search_engine_add_paper() {
        let mut engine = PaperSearchEngine::new(3);
        let paper = make_embedded_paper("2401.00001", vec![1.0, 0.0, 0.0]);
        engine.add_paper(paper);
        assert_eq!(engine.len(), 1);
    }

    #[test]
    fn test_paper_search_engine_rejects_wrong_dims() {
        let mut engine = PaperSearchEngine::new(3);
        let paper = make_embedded_paper("2401.00001", vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        engine.add_paper(paper);
        assert!(engine.is_empty()); // Should reject wrong dimension
    }

    #[test]
    fn test_paper_search_engine_search() {
        let mut engine = PaperSearchEngine::new(3);
        engine.add_paper(make_embedded_paper("paper1", vec![1.0, 0.0, 0.0]));
        engine.add_paper(make_embedded_paper("paper2", vec![0.0, 1.0, 0.0]));
        engine.add_paper(make_embedded_paper("paper3", vec![0.0, 0.0, 1.0]));

        // Query similar to paper1
        let results = engine.search(&[1.0, 0.0, 0.0], 1, false);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].paper.arxiv_id, "paper1");
        assert!((results[0].score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_paper_search_engine_search_sorted() {
        let mut engine = PaperSearchEngine::new(3);
        engine.add_paper(make_embedded_paper("paper1", vec![1.0, 0.0, 0.0]));
        engine.add_paper(make_embedded_paper("paper2", vec![0.7, 0.7, 0.0]));
        engine.add_paper(make_embedded_paper("paper3", vec![0.5, 0.5, 0.7]));

        // Query similar to paper1
        let results = engine.search(&[1.0, 0.0, 0.0], 3, false);
        assert_eq!(results.len(), 3);
        // Results should be sorted by score descending
        for window in results.windows(2) {
            assert!(window[0].score >= window[1].score);
        }
    }

    #[test]
    fn test_paper_search_engine_find_similar() {
        let mut engine = PaperSearchEngine::new(3);
        engine.add_paper(make_embedded_paper("paper1", vec![1.0, 0.0, 0.0]));
        engine.add_paper(make_embedded_paper("paper2", vec![0.9, 0.1, 0.0])); // Similar to paper1
        engine.add_paper(make_embedded_paper("paper3", vec![0.0, 0.0, 1.0])); // Different

        let results = engine.find_similar("paper1", 2);
        assert_eq!(results.len(), 2);
        // paper1 should not be in results
        assert!(results.iter().all(|r| r.paper.arxiv_id != "paper1"));
        // paper2 should be first (most similar)
        assert_eq!(results[0].paper.arxiv_id, "paper2");
    }

    #[test]
    fn test_paper_search_engine_find_similar_nonexistent() {
        let mut engine = PaperSearchEngine::new(3);
        engine.add_paper(make_embedded_paper("paper1", vec![1.0, 0.0, 0.0]));

        let results = engine.find_similar("nonexistent", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_paper_search_engine_get_paper() {
        let mut engine = PaperSearchEngine::new(3);
        engine.add_paper(make_embedded_paper("2401.00001", vec![1.0, 0.0, 0.0]));

        let paper = engine.get_paper("2401.00001");
        assert!(paper.is_some());
        assert_eq!(paper.unwrap().arxiv_id, "2401.00001");

        let missing = engine.get_paper("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_paper_search_engine_get_papers() {
        let mut engine = PaperSearchEngine::new(3);
        engine.add_paper(make_embedded_paper("paper1", vec![1.0, 0.0, 0.0]));
        engine.add_paper(make_embedded_paper("paper2", vec![0.0, 1.0, 0.0]));

        let papers = engine.get_papers();
        assert_eq!(papers.len(), 2);
    }

    #[test]
    fn test_paper_search_engine_search_by_category() {
        let mut engine = PaperSearchEngine::new(3);

        let mut paper1 = make_embedded_paper("paper1", vec![1.0, 0.0, 0.0]);
        paper1.paper.categories = vec!["cs.LO".to_string()];

        let mut paper2 = make_embedded_paper("paper2", vec![0.9, 0.1, 0.0]);
        paper2.paper.categories = vec!["cs.AI".to_string()];

        engine.add_paper(paper1);
        engine.add_paper(paper2);

        // Search only in cs.LO
        let results = engine.search_by_category(&[1.0, 0.0, 0.0], &["cs.LO".to_string()], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].paper.arxiv_id, "paper1");
    }

    #[test]
    fn test_paper_search_engine_fulltext_search() {
        let mut engine = PaperSearchEngine::new(3);

        let mut paper = make_embedded_paper("paper1", vec![1.0, 0.0, 0.0]);
        paper.fulltext_embedding = Some(vec![0.0, 1.0, 0.0]);
        engine.add_paper(paper);

        // Query that matches fulltext better than abstract
        let query = vec![0.0, 1.0, 0.0];

        // Without fulltext search, score should be 0 (orthogonal to abstract embedding)
        let results_no_fulltext = engine.search(&query, 1, false);
        assert!((results_no_fulltext[0].score).abs() < 0.001);

        // With fulltext search, score should be 1.0 (matches fulltext)
        let results_with_fulltext = engine.search(&query, 1, true);
        assert!((results_with_fulltext[0].score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_paper_search_engine_wrong_query_dims() {
        let mut engine = PaperSearchEngine::new(3);
        engine.add_paper(make_embedded_paper("paper1", vec![1.0, 0.0, 0.0]));

        // Query with wrong dimensions
        let results = engine.search(&[1.0, 0.0, 0.0, 0.0, 0.0], 10, false);
        assert!(results.is_empty());
    }
}
