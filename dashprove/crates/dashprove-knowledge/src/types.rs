//! Core types for the knowledge base

use chrono::{DateTime, Utc};
use dashprove_backends::BackendId;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify KnowledgeQuery::new sets default limit
    #[kani::proof]
    fn verify_query_new_default_limit() {
        let query = KnowledgeQuery::new("test");
        kani::assert(query.limit == 10, "default limit should be 10");
    }

    /// Verify KnowledgeQuery::new includes papers by default
    #[kani::proof]
    fn verify_query_new_include_papers() {
        let query = KnowledgeQuery::new("test");
        kani::assert(query.include_papers, "should include papers by default");
    }

    /// Verify KnowledgeQuery::new includes repos by default
    #[kani::proof]
    fn verify_query_new_include_repos() {
        let query = KnowledgeQuery::new("test");
        kani::assert(query.include_repos, "should include repos by default");
    }

    /// Verify KnowledgeQuery::new sets text correctly
    #[kani::proof]
    fn verify_query_new_text() {
        let query = KnowledgeQuery::new("search query");
        kani::assert(query.text == "search query", "text should be set correctly");
    }

    /// Verify KnowledgeQuery::new has None for backend filter
    #[kani::proof]
    fn verify_query_new_no_backend() {
        let query = KnowledgeQuery::new("test");
        kani::assert(query.backend.is_none(), "backend should be None by default");
    }

    /// Verify KnowledgeQuery::new has None for content_type filter
    #[kani::proof]
    fn verify_query_new_no_content_type() {
        let query = KnowledgeQuery::new("test");
        kani::assert(
            query.content_type.is_none(),
            "content_type should be None by default",
        );
    }

    /// Verify KnowledgeQuery::new has empty tags
    #[kani::proof]
    fn verify_query_new_empty_tags() {
        let query = KnowledgeQuery::new("test");
        kani::assert(query.tags.is_empty(), "tags should be empty by default");
    }

    /// Verify with_backend sets the backend
    #[kani::proof]
    fn verify_with_backend() {
        let query = KnowledgeQuery::new("test").with_backend(BackendId::Lean4);
        kani::assert(
            query.backend == Some(BackendId::Lean4),
            "backend should be set to Lean4",
        );
    }

    /// Verify with_content_type sets the content_type
    #[kani::proof]
    fn verify_with_content_type() {
        let query = KnowledgeQuery::new("test").with_content_type(ContentType::Tutorial);
        kani::assert(
            query.content_type == Some(ContentType::Tutorial),
            "content_type should be set to Tutorial",
        );
    }

    /// Verify with_limit sets the limit
    #[kani::proof]
    fn verify_with_limit() {
        let query = KnowledgeQuery::new("test").with_limit(25);
        kani::assert(query.limit == 25, "limit should be set to 25");
    }

    /// Verify builder chain preserves text
    #[kani::proof]
    fn verify_builder_preserves_text() {
        let query = KnowledgeQuery::new("original")
            .with_backend(BackendId::Coq)
            .with_limit(5);
        kani::assert(
            query.text == "original",
            "text should be preserved through chain",
        );
    }

    /// Verify builder chain preserves backend
    #[kani::proof]
    fn verify_builder_preserves_backend() {
        let query = KnowledgeQuery::new("test")
            .with_backend(BackendId::TlaPlus)
            .with_limit(5);
        kani::assert(
            query.backend == Some(BackendId::TlaPlus),
            "backend should be preserved through chain",
        );
    }

    /// Verify ContentType default is General
    #[kani::proof]
    fn verify_content_type_default() {
        let ct: ContentType = Default::default();
        kani::assert(
            ct == ContentType::General,
            "default ContentType should be General",
        );
    }

    /// Verify DocumentMetadata default word_count is 0
    #[kani::proof]
    fn verify_doc_metadata_default_word_count() {
        let meta: DocumentMetadata = Default::default();
        kani::assert(meta.word_count == 0, "default word_count should be 0");
    }

    /// Verify DocumentMetadata default sections is empty
    #[kani::proof]
    fn verify_doc_metadata_default_sections() {
        let meta: DocumentMetadata = Default::default();
        kani::assert(meta.sections.is_empty(), "default sections should be empty");
    }

    /// Verify DocumentMetadata default tags is empty
    #[kani::proof]
    fn verify_doc_metadata_default_tags() {
        let meta: DocumentMetadata = Default::default();
        kani::assert(meta.tags.is_empty(), "default tags should be empty");
    }

    /// Verify DocumentChunk default chunk_index is 0
    #[kani::proof]
    fn verify_doc_chunk_default_index() {
        let chunk: DocumentChunk = Default::default();
        kani::assert(chunk.chunk_index == 0, "default chunk_index should be 0");
    }

    /// Verify DocumentChunk default token_count is 0
    #[kani::proof]
    fn verify_doc_chunk_default_token_count() {
        let chunk: DocumentChunk = Default::default();
        kani::assert(chunk.token_count == 0, "default token_count should be 0");
    }
}

/// A document in the knowledge base
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique document ID
    pub id: String,
    /// Source URL or path
    pub source: String,
    /// Backend this document relates to (if any)
    pub backend: Option<BackendId>,
    /// Document title
    pub title: String,
    /// Document content (markdown)
    pub content: String,
    /// Content type
    pub content_type: ContentType,
    /// When the document was fetched
    pub fetched_at: DateTime<Utc>,
    /// Additional metadata
    pub metadata: DocumentMetadata,
}

/// Document content type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ContentType {
    /// Reference documentation
    Reference,
    /// Tutorial content
    Tutorial,
    /// API documentation
    Api,
    /// Code examples
    Example,
    /// Academic paper
    Paper,
    /// Error messages/diagnostics
    Errors,
    /// General information
    #[default]
    General,
}

/// Additional document metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Section hierarchy
    pub sections: Vec<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Code blocks in the document
    pub code_blocks: Vec<CodeBlock>,
    /// Original format before conversion
    pub original_format: Option<String>,
    /// Word count
    pub word_count: usize,
}

/// A code block extracted from documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeBlock {
    /// Programming language
    pub language: String,
    /// Code content
    pub code: String,
    /// Description/context
    pub description: Option<String>,
}

/// A chunk of a document for embedding
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentChunk {
    /// Unique chunk ID
    pub id: String,
    /// Parent document ID
    pub document_id: String,
    /// Chunk index within document
    pub chunk_index: usize,
    /// Chunk content
    pub content: String,
    /// Backend (inherited from document)
    pub backend: Option<BackendId>,
    /// Content type (inherited from document)
    pub content_type: ContentType,
    /// Section path
    pub section_path: Vec<String>,
    /// Token count (approximate)
    pub token_count: usize,
}

/// An embedding vector with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedChunk {
    /// The chunk
    pub chunk: DocumentChunk,
    /// Embedding vector
    pub embedding: Vec<f32>,
}

/// An ArXiv paper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArxivPaper {
    /// ArXiv ID (e.g., "2401.12345")
    pub arxiv_id: String,
    /// Paper title
    pub title: String,
    /// Authors
    pub authors: Vec<String>,
    /// Abstract
    pub abstract_text: String,
    /// Primary category
    pub primary_category: String,
    /// All categories
    pub categories: Vec<String>,
    /// Published date
    pub published: DateTime<Utc>,
    /// Updated date
    pub updated: DateTime<Utc>,
    /// PDF URL
    pub pdf_url: String,
    /// Comment (version info, page count, etc.)
    pub comment: Option<String>,
    /// Local path to downloaded PDF
    pub local_path: Option<PathBuf>,
    /// Extracted text content
    pub extracted_text: Option<String>,
    /// Relevance tags (assigned by analysis)
    pub tags: Vec<String>,
}

/// A GitHub repository
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GithubRepo {
    /// Full repository name (owner/repo)
    pub full_name: String,
    /// Repository description
    pub description: Option<String>,
    /// Star count
    pub stars: usize,
    /// Fork count
    pub forks: usize,
    /// Primary language
    pub language: Option<String>,
    /// Topics/tags
    pub topics: Vec<String>,
    /// Repository URL
    pub url: String,
    /// Last update time
    pub updated_at: DateTime<Utc>,
    /// Created time
    pub created_at: DateTime<Utc>,
    /// README content (if fetched)
    pub readme: Option<String>,
    /// Whether this is relevant to DashProve
    pub relevance_score: f32,
}

/// Search result from the knowledge base
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Matching chunks
    pub chunks: Vec<ScoredChunk>,
    /// Matching papers
    pub papers: Vec<ScoredPaper>,
    /// Matching repositories
    pub repos: Vec<ScoredRepo>,
}

/// A chunk with relevance score
#[derive(Debug, Clone)]
pub struct ScoredChunk {
    pub chunk: DocumentChunk,
    pub score: f32,
}

/// A paper with relevance score
#[derive(Debug, Clone)]
pub struct ScoredPaper {
    pub paper: ArxivPaper,
    pub score: f32,
}

/// A repository with relevance score
#[derive(Debug, Clone)]
pub struct ScoredRepo {
    pub repo: GithubRepo,
    pub score: f32,
}

/// Query for searching the knowledge base
#[derive(Debug, Clone, Default)]
pub struct KnowledgeQuery {
    /// Search text
    pub text: String,
    /// Filter by backend
    pub backend: Option<BackendId>,
    /// Filter by content type
    pub content_type: Option<ContentType>,
    /// Filter by tags
    pub tags: Vec<String>,
    /// Maximum results
    pub limit: usize,
    /// Include papers in results
    pub include_papers: bool,
    /// Include repos in results
    pub include_repos: bool,
}

impl KnowledgeQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            limit: 10,
            include_papers: true,
            include_repos: true,
            ..Default::default()
        }
    }

    pub fn with_backend(mut self, backend: BackendId) -> Self {
        self.backend = Some(backend);
        self
    }

    pub fn with_content_type(mut self, content_type: ContentType) -> Self {
        self.content_type = Some(content_type);
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}
