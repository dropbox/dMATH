//! Request and response types for proof search endpoint

use dashprove_backends::BackendId;
use serde::{Deserialize, Serialize};

/// Request for proof search
#[derive(Debug, Clone, Deserialize)]
pub struct ProofSearchRequest {
    /// USL property source to prove
    pub property: String,
    /// Target backend for the proof
    pub backend: BackendIdInput,
    /// Maximum search iterations
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,
    /// Validation threshold (0.0 to 1.0)
    #[serde(default = "default_threshold")]
    pub validation_threshold: f64,
    /// Additional hints to guide search
    #[serde(default)]
    pub hints: Vec<String>,
    /// Preferred tactics to try first
    #[serde(default)]
    pub preferred_tactics: Vec<String>,
    /// Backends to propagate hints to
    #[serde(default)]
    pub propagate_to: Vec<BackendIdInput>,
}

fn default_max_iterations() -> u32 {
    4
}

fn default_threshold() -> f64 {
    0.75
}

/// Backend ID input for API requests
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendIdInput {
    Lean4,
    TlaPlus,
    Kani,
    Coq,
    Alloy,
    Isabelle,
    Dafny,
}

impl From<BackendIdInput> for BackendId {
    fn from(input: BackendIdInput) -> Self {
        match input {
            BackendIdInput::Lean4 => BackendId::Lean4,
            BackendIdInput::TlaPlus => BackendId::TlaPlus,
            BackendIdInput::Kani => BackendId::Kani,
            BackendIdInput::Coq => BackendId::Coq,
            BackendIdInput::Alloy => BackendId::Alloy,
            BackendIdInput::Isabelle => BackendId::Isabelle,
            BackendIdInput::Dafny => BackendId::Dafny,
        }
    }
}

impl From<BackendId> for BackendIdInput {
    fn from(id: BackendId) -> Self {
        match id {
            BackendId::Lean4 => BackendIdInput::Lean4,
            BackendId::TlaPlus => BackendIdInput::TlaPlus,
            BackendId::Kani => BackendIdInput::Kani,
            BackendId::Coq => BackendIdInput::Coq,
            BackendId::Alloy => BackendIdInput::Alloy,
            BackendId::Isabelle => BackendIdInput::Isabelle,
            BackendId::Dafny => BackendIdInput::Dafny,
            _ => BackendIdInput::Lean4, // Default fallback
        }
    }
}

/// Response from proof search
#[derive(Debug, Clone, Serialize)]
pub struct ProofSearchResponse {
    /// Whether a proof was found
    pub found: bool,
    /// Best proof found (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof: Option<ProofResult>,
    /// Search steps taken
    pub steps: Vec<SearchStep>,
    /// Hints propagated to other backends
    pub propagated_hints: Vec<PropagatedHint>,
    /// Current tactic policy weights
    pub policy_weights: Vec<(String, f64)>,
}

/// A proof result from the search
#[derive(Debug, Clone, Serialize)]
pub struct ProofResult {
    /// The proof code
    pub code: String,
    /// Confidence in the proof (0.0 - 1.0)
    pub confidence: f64,
    /// Tactics used in the proof
    pub tactics_used: Vec<String>,
}

/// A single search step
#[derive(Debug, Clone, Serialize)]
pub struct SearchStep {
    /// Iteration number
    pub iteration: u32,
    /// Tactic tried
    pub tactic: String,
    /// Whether the tactic succeeded
    pub succeeded: bool,
    /// Reward assigned
    pub reward: f64,
}

/// A hint propagated to another backend
#[derive(Debug, Clone, Serialize)]
pub struct PropagatedHint {
    /// Source backend
    pub source: BackendIdInput,
    /// Target backend
    pub target: BackendIdInput,
    /// The hint
    pub hint: String,
    /// Confidence in the hint
    pub confidence: f64,
}
