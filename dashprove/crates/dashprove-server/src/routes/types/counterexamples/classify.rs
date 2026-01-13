use serde::{Deserialize, Serialize};

use crate::routes::types::StructuredCounterexampleRequest;

// ============ Counterexample Classify Types ============

/// Request to classify a counterexample against stored cluster patterns
#[derive(Debug, Deserialize)]
pub struct CounterexampleClassifyRequest {
    /// The counterexample to classify
    pub counterexample: StructuredCounterexampleRequest,
}

/// Response from counterexample classification
#[derive(Debug, Serialize, Deserialize)]
pub struct CounterexampleClassifyResponse {
    /// Best matching cluster label (if found)
    pub cluster_label: Option<String>,
    /// Similarity score to the best matching cluster (0.0 to 1.0)
    pub similarity: Option<f64>,
    /// Total cluster patterns in corpus
    pub total_patterns: usize,
}
