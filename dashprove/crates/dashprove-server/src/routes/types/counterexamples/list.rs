use serde::{Deserialize, Serialize};

use crate::routes::types::{BackendIdParam, CounterexampleEntryResponse};

// ============ Counterexample List Types ============

/// Query parameters for listing counterexamples
#[derive(Debug, Deserialize)]
pub struct CounterexampleListQuery {
    /// Number of results per page
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Offset for pagination
    #[serde(default)]
    pub offset: usize,
    /// Optional backend filter
    pub backend: Option<BackendIdParam>,
    /// Optional property name filter (case-insensitive substring match)
    pub property_name: Option<String>,
    /// Optional start date filter (YYYY-MM-DD)
    pub from: Option<String>,
    /// Optional end date filter (YYYY-MM-DD)
    pub to: Option<String>,
}

pub fn default_limit() -> usize {
    50
}

/// Response for listing counterexamples
#[derive(Debug, Serialize, Deserialize)]
pub struct CounterexampleListResponse {
    /// Counterexample entries
    pub counterexamples: Vec<CounterexampleEntryResponse>,
    /// Total number of counterexamples in corpus (for pagination)
    pub total: usize,
    /// Current offset
    pub offset: usize,
    /// Current limit
    pub limit: usize,
}
