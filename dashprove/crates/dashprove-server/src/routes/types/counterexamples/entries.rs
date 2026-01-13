use dashprove_learning::CounterexampleEntry;
use serde::{Deserialize, Serialize};

use crate::routes::types::BackendIdParam;

// ============ Counterexample Entry Response Types ============

/// Full counterexample entry in responses
#[derive(Debug, Serialize, Deserialize)]
pub struct CounterexampleEntryResponse {
    /// Counterexample ID
    pub id: String,
    /// Property name
    pub property_name: String,
    /// Backend that generated it
    pub backend: BackendIdParam,
    /// Cluster label if classified
    pub cluster_label: Option<String>,
    /// When it was recorded (ISO 8601)
    pub recorded_at: String,
    /// Features extracted from the counterexample
    pub features: CounterexampleFeaturesResponse,
}

/// Counterexample features in response
#[derive(Debug, Serialize, Deserialize)]
pub struct CounterexampleFeaturesResponse {
    /// Variable names in witness
    pub witness_vars: Vec<String>,
    /// Variable names in trace
    pub trace_vars: Vec<String>,
    /// IDs of failed checks
    pub failed_check_ids: Vec<String>,
    /// Keywords from failed checks
    pub failed_check_keywords: Vec<String>,
    /// Length of execution trace
    pub trace_length: usize,
    /// Action names in trace
    pub action_names: Vec<String>,
    /// Keywords extracted
    pub keywords: Vec<String>,
}

impl From<&CounterexampleEntry> for CounterexampleEntryResponse {
    fn from(entry: &CounterexampleEntry) -> Self {
        CounterexampleEntryResponse {
            id: entry.id.0.clone(),
            property_name: entry.property_name.clone(),
            backend: entry.backend.into(),
            cluster_label: entry.cluster_label.clone(),
            recorded_at: entry.recorded_at.to_rfc3339(),
            features: CounterexampleFeaturesResponse {
                witness_vars: entry.features.witness_vars.clone(),
                trace_vars: entry.features.trace_vars.clone(),
                failed_check_ids: entry.features.failed_check_ids.clone(),
                failed_check_keywords: entry.features.failed_check_keywords.clone(),
                trace_length: entry.features.trace_length,
                action_names: entry.features.action_names.clone(),
                keywords: entry.features.keywords.clone(),
            },
        }
    }
}
