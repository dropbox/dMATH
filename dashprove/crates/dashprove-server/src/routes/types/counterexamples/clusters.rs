use dashprove_learning::ClusterPattern;
use serde::{Deserialize, Serialize};

use crate::routes::types::{CounterexampleFeaturesResponse, StructuredCounterexampleRequest};

// ============ Counterexample Clusters Types ============

/// Request to record cluster patterns from clustering results
#[derive(Debug, Deserialize)]
pub struct CounterexampleClustersRequest {
    /// Cluster patterns to record
    pub patterns: Vec<ClusterPatternRequest>,
    /// Similarity threshold used for clustering
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f64,
}

pub fn default_similarity_threshold() -> f64 {
    0.7
}

/// A cluster pattern to record
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClusterPatternRequest {
    /// Cluster label
    pub label: String,
    /// Representative counterexample for this cluster
    pub representative: StructuredCounterexampleRequest,
    /// Number of counterexamples in this cluster
    pub count: usize,
}

/// Response after recording cluster patterns
#[derive(Debug, Serialize, Deserialize)]
pub struct CounterexampleClustersResponse {
    /// Number of patterns recorded
    pub patterns_recorded: usize,
    /// Total cluster patterns now in corpus
    pub total_patterns: usize,
}

/// Cluster pattern in response
#[derive(Debug, Serialize, Deserialize)]
pub struct ClusterPatternResponse {
    /// Cluster label
    pub label: String,
    /// Number of counterexamples originally in this cluster
    pub original_count: usize,
    /// Similarity threshold used
    pub similarity_threshold: f64,
    /// Features of the representative counterexample
    pub features: CounterexampleFeaturesResponse,
}

impl From<&ClusterPattern> for ClusterPatternResponse {
    fn from(p: &ClusterPattern) -> Self {
        ClusterPatternResponse {
            label: p.label.clone(),
            original_count: p.original_count,
            similarity_threshold: p.similarity_threshold,
            features: CounterexampleFeaturesResponse {
                witness_vars: p.representative_features.witness_vars.clone(),
                trace_vars: p.representative_features.trace_vars.clone(),
                failed_check_ids: p.representative_features.failed_check_ids.clone(),
                failed_check_keywords: p.representative_features.failed_check_keywords.clone(),
                trace_length: p.representative_features.trace_length,
                action_names: p.representative_features.action_names.clone(),
                keywords: p.representative_features.keywords.clone(),
            },
        }
    }
}
