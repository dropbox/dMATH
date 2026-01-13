use serde::{Deserialize, Serialize};

use crate::routes::types::{default_k, BackendIdParam};

// ============ Corpus Search Types ============

/// Request to search the corpus
#[derive(Debug, Deserialize)]
pub struct CorpusSearchQuery {
    /// Search query (property source or keywords)
    pub query: String,
    /// Number of results to return
    #[serde(default = "default_k")]
    pub k: usize,
}

/// Response from corpus search
#[derive(Debug, Serialize, Deserialize)]
pub struct CorpusSearchResponse {
    /// List of similar proofs
    pub results: Vec<SimilarProofResponse>,
    /// Total number of proofs in corpus
    pub total_corpus_size: usize,
}

/// A similar proof in search results
#[derive(Debug, Serialize, Deserialize)]
pub struct SimilarProofResponse {
    /// Unique proof identifier
    pub proof_id: String,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f64,
    /// Name of the property
    pub property_name: String,
    /// Backend that proved this property
    pub backend: BackendIdParam,
    /// Tactics used in the proof
    pub tactics: Vec<String>,
}
