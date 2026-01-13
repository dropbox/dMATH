use std::collections::HashMap;

use dashprove_backends::{CounterexampleValue, FailedCheck, StructuredCounterexample, TraceState};
use dashprove_learning::SimilarCounterexample;
use serde::{Deserialize, Serialize};

use crate::routes::types::{default_k, BackendIdParam};

// ============ Counterexample Search Types ============

/// Query parameters for counterexample search
#[derive(Debug, Deserialize)]
pub struct CounterexampleSearchQuery {
    /// Number of results to return
    #[serde(default = "default_k")]
    pub k: usize,
}

/// Request body for counterexample search (POST with JSON body)
#[derive(Debug, Deserialize)]
pub struct CounterexampleSearchRequest {
    /// The counterexample to search for similar ones
    pub counterexample: StructuredCounterexampleRequest,
    /// Number of results to return
    #[serde(default = "default_k")]
    pub k: usize,
}

/// Query parameters for counterexample text search (GET with query params)
#[derive(Debug, Deserialize)]
pub struct CounterexampleTextSearchQuery {
    /// Text query (keywords to search for in counterexamples)
    pub query: String,
    /// Number of results to return
    #[serde(default = "default_k")]
    pub k: usize,
}

/// Structured counterexample for API requests
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StructuredCounterexampleRequest {
    /// Witness values (variable name -> value)
    #[serde(default)]
    pub witness: HashMap<String, serde_json::Value>,
    /// Failed checks
    #[serde(default)]
    pub failed_checks: Vec<FailedCheckRequest>,
    /// Execution trace
    #[serde(default)]
    pub trace: Vec<TraceStateRequest>,
    /// Raw counterexample output
    #[serde(default)]
    pub raw: Option<String>,
    /// Whether this counterexample has been minimized
    #[serde(default)]
    pub minimized: bool,
}

/// A failed check in the counterexample
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FailedCheckRequest {
    /// Check ID or name
    pub id: String,
    /// Description of the failure
    #[serde(default)]
    pub description: Option<String>,
    /// Location in source
    #[serde(default)]
    pub location: Option<String>,
}

/// A state in the execution trace
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TraceStateRequest {
    /// State number
    #[serde(default)]
    pub state_num: usize,
    /// Action name that led to this state
    #[serde(default)]
    pub action: Option<String>,
    /// Variable values in this state
    #[serde(default)]
    pub values: HashMap<String, serde_json::Value>,
}

/// Convert a JSON value to a CounterexampleValue
pub fn json_to_counterexample_value(v: serde_json::Value) -> CounterexampleValue {
    match v {
        serde_json::Value::Bool(b) => CounterexampleValue::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                CounterexampleValue::Int {
                    value: i as i128,
                    type_hint: None,
                }
            } else if let Some(f) = n.as_f64() {
                CounterexampleValue::Float { value: f }
            } else {
                CounterexampleValue::String(n.to_string())
            }
        }
        serde_json::Value::String(s) => CounterexampleValue::String(s),
        other => CounterexampleValue::String(other.to_string()),
    }
}

impl From<StructuredCounterexampleRequest> for StructuredCounterexample {
    fn from(req: StructuredCounterexampleRequest) -> Self {
        let witness = req
            .witness
            .into_iter()
            .map(|(k, v)| (k, json_to_counterexample_value(v)))
            .collect();

        let failed_checks = req
            .failed_checks
            .into_iter()
            .map(|fc| FailedCheck {
                check_id: fc.id,
                description: fc.description.unwrap_or_default(),
                location: None,
                function: None,
            })
            .collect();

        let trace = req
            .trace
            .into_iter()
            .map(|ts| TraceState {
                state_num: ts.state_num as u32,
                action: ts.action,
                variables: ts
                    .values
                    .into_iter()
                    .map(|(k, v)| (k, json_to_counterexample_value(v)))
                    .collect(),
            })
            .collect();

        StructuredCounterexample {
            witness,
            failed_checks,
            playback_test: None,
            trace,
            raw: req.raw,
            minimized: req.minimized,
        }
    }
}

/// Response from counterexample search
#[derive(Debug, Serialize, Deserialize)]
pub struct CounterexampleSearchResponse {
    /// Similar counterexamples found
    pub results: Vec<SimilarCounterexampleResponse>,
    /// Total counterexamples in corpus
    pub total_corpus_size: usize,
}

/// A similar counterexample in search results
#[derive(Debug, Serialize, Deserialize)]
pub struct SimilarCounterexampleResponse {
    /// Counterexample ID
    pub id: String,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f64,
    /// Property that was violated
    pub property_name: String,
    /// Backend that generated it
    pub backend: BackendIdParam,
    /// Cluster label if classified
    pub cluster_label: Option<String>,
}

impl From<SimilarCounterexample> for SimilarCounterexampleResponse {
    fn from(sc: SimilarCounterexample) -> Self {
        SimilarCounterexampleResponse {
            id: sc.id.0,
            similarity: sc.similarity,
            property_name: sc.property_name,
            backend: sc.backend.into(),
            cluster_label: sc.cluster_label,
        }
    }
}
