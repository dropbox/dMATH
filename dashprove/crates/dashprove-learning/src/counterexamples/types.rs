//! Core counterexample types
//!
//! Contains the fundamental types for counterexample storage and retrieval.

use chrono::{DateTime, Utc};
use dashprove_backends::traits::{BackendId, StructuredCounterexample};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a stored counterexample
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CounterexampleId(pub String);

impl fmt::Display for CounterexampleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl CounterexampleId {
    /// Generate a new unique ID based on property name and timestamp
    pub fn generate(property_name: &str) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        CounterexampleId(format!("cx_{}_{}", property_name, timestamp))
    }
}

/// Extracted features from a counterexample for similarity search
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CounterexampleFeatures {
    /// Variable names present in witness
    pub witness_vars: Vec<String>,
    /// Variable names present in trace states
    pub trace_vars: Vec<String>,
    /// Failed check IDs
    pub failed_check_ids: Vec<String>,
    /// Failed check descriptions (normalized/tokenized)
    pub failed_check_keywords: Vec<String>,
    /// Number of states in trace
    pub trace_length: usize,
    /// Action names from trace
    pub action_names: Vec<String>,
    /// Keywords extracted from the counterexample
    pub keywords: Vec<String>,
}

impl CounterexampleFeatures {
    /// Extract features from a counterexample
    pub fn extract(cx: &StructuredCounterexample) -> Self {
        let witness_vars: Vec<String> = cx.witness.keys().cloned().collect();

        let trace_vars: Vec<String> = cx
            .trace
            .iter()
            .flat_map(|s| s.variables.keys().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let failed_check_ids: Vec<String> = cx
            .failed_checks
            .iter()
            .map(|c| c.check_id.clone())
            .collect();

        let failed_check_keywords: Vec<String> = cx
            .failed_checks
            .iter()
            .flat_map(|c| {
                c.description
                    .split(|ch: char| ch.is_whitespace() || ch == '_' || ch == '-' || ch == '.')
                    .filter(|s| s.len() > 2)
                    .map(|s| s.to_lowercase())
            })
            .collect();

        let action_names: Vec<String> = cx
            .trace
            .iter()
            .filter_map(|s| s.action.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Build keywords from all sources
        let mut keywords: Vec<String> = Vec::new();
        keywords.extend(witness_vars.iter().cloned());
        keywords.extend(trace_vars.iter().cloned());
        keywords.extend(failed_check_keywords.iter().cloned());
        keywords.extend(action_names.iter().cloned());

        // Deduplicate and sort
        let keywords: Vec<String> = keywords
            .into_iter()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        Self {
            witness_vars,
            trace_vars,
            failed_check_ids,
            failed_check_keywords,
            trace_length: cx.trace.len(),
            action_names,
            keywords,
        }
    }
}

/// A stored counterexample entry in the corpus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterexampleEntry {
    /// Unique identifier
    pub id: CounterexampleId,
    /// The property that was disproven
    pub property_name: String,
    /// Which backend generated this counterexample
    pub backend: BackendId,
    /// The structured counterexample
    pub counterexample: StructuredCounterexample,
    /// Extracted features for similarity search
    pub features: CounterexampleFeatures,
    /// Cluster label if this counterexample was part of a cluster
    pub cluster_label: Option<String>,
    /// When this counterexample was recorded
    #[serde(default = "default_recorded_at")]
    pub recorded_at: DateTime<Utc>,
}

pub(crate) fn default_recorded_at() -> DateTime<Utc> {
    Utc::now()
}

/// A similar counterexample found in corpus
#[derive(Debug, Clone)]
pub struct SimilarCounterexample {
    /// The counterexample entry
    pub id: CounterexampleId,
    /// Property name
    pub property_name: String,
    /// Backend that found it
    pub backend: BackendId,
    /// Cluster label (if any)
    pub cluster_label: Option<String>,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f64,
}

/// A stored cluster pattern for classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterPattern {
    /// Pattern label/description
    pub label: String,
    /// Representative counterexample features
    pub representative_features: CounterexampleFeatures,
    /// Number of counterexamples originally in this cluster
    pub original_count: usize,
    /// Similarity threshold used for this cluster
    pub similarity_threshold: f64,
}
