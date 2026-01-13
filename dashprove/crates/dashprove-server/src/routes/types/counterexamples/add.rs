use serde::{Deserialize, Serialize};

use crate::routes::types::{BackendIdParam, StructuredCounterexampleRequest};

// ============ Counterexample Add Types ============

/// Request to add a counterexample to the corpus
#[derive(Debug, Deserialize)]
pub struct CounterexampleAddRequest {
    /// The counterexample to add
    pub counterexample: StructuredCounterexampleRequest,
    /// Property name this counterexample violates
    pub property_name: String,
    /// Backend that generated this counterexample
    pub backend: BackendIdParam,
    /// Optional cluster label
    #[serde(default)]
    pub cluster_label: Option<String>,
}

/// Response after adding a counterexample
#[derive(Debug, Serialize, Deserialize)]
pub struct CounterexampleAddResponse {
    /// The ID assigned to the counterexample
    pub id: String,
    /// Total counterexamples now in corpus
    pub total_corpus_size: usize,
}
