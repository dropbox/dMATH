use dashprove_ai::Confidence;
use serde::{Deserialize, Serialize};

// ============ Tactic/Sketch Types ============

/// Request to suggest tactics
#[derive(Debug, Deserialize)]
pub struct TacticSuggestRequest {
    /// The property to get suggestions for (USL source)
    pub property: String,
    /// Number of suggestions
    #[serde(default = "default_n")]
    pub n: usize,
}

pub fn default_n() -> usize {
    5
}

/// Response with tactic suggestions
#[derive(Debug, Serialize, Deserialize)]
pub struct TacticSuggestResponse {
    /// List of suggested tactics
    pub suggestions: Vec<TacticSuggestionResponse>,
}

/// A single tactic suggestion
#[derive(Debug, Serialize, Deserialize)]
pub struct TacticSuggestionResponse {
    /// The tactic name
    pub tactic: String,
    /// Confidence level (high, medium, low, speculative)
    pub confidence: String,
    /// Source of the suggestion (pattern, corpus, heuristic)
    pub source: String,
    /// Explanation for why this tactic is suggested
    pub rationale: String,
}

impl From<dashprove_ai::TacticSuggestion> for TacticSuggestionResponse {
    fn from(s: dashprove_ai::TacticSuggestion) -> Self {
        let confidence = match s.confidence {
            Confidence::High => "high",
            Confidence::Medium => "medium",
            Confidence::Low => "low",
            Confidence::Speculative => "speculative",
        }
        .to_string();

        let source = format!("{:?}", s.source);

        TacticSuggestionResponse {
            tactic: s.tactic,
            confidence,
            source,
            rationale: s.rationale,
        }
    }
}

/// Request to elaborate a proof sketch
#[derive(Debug, Deserialize)]
pub struct SketchElaborateRequest {
    /// The property to create a sketch for (USL source)
    pub property: String,
    /// Hints for elaboration
    #[serde(default)]
    pub hints: Vec<String>,
}

/// Response with elaborated sketch
#[derive(Debug, Serialize, Deserialize)]
pub struct SketchElaborateResponse {
    /// The proof sketch structure
    pub sketch: ProofSketchResponse,
    /// Generated LEAN code
    pub lean_code: String,
}

/// Proof sketch in response
#[derive(Debug, Serialize, Deserialize)]
pub struct ProofSketchResponse {
    /// Name of the property being proved
    pub property_name: String,
    /// Steps in the proof sketch
    pub steps: Vec<SketchStepResponse>,
    /// Whether all steps are completed
    pub is_complete: bool,
}

/// A single sketch step
#[derive(Debug, Serialize, Deserialize)]
pub struct SketchStepResponse {
    /// Human-readable description of this step
    pub description: String,
    /// Tactic to apply (if known)
    pub tactic: Option<String>,
    /// Whether this step is completed
    pub completed: bool,
}
