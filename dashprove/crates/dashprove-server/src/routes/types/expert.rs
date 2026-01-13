use serde::{Deserialize, Serialize};

use crate::routes::types::BackendIdParam;

// ============ Expert API Types ============

/// Request for backend recommendation
#[derive(Debug, Deserialize)]
pub struct ExpertBackendRequest {
    /// Optional specification content
    pub spec: Option<String>,
    /// Property types (comma-separated: safety, liveness, temporal, etc.)
    pub property_types: Option<String>,
    /// Code language (e.g., "rust")
    pub code_lang: Option<String>,
    /// Optional tags for context
    pub tags: Option<Vec<String>>,
}

/// Response for backend recommendation
#[derive(Debug, Serialize, Deserialize)]
pub struct ExpertBackendResponse {
    /// Recommended backend
    pub backend: BackendIdParam,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Rationale for the recommendation
    pub rationale: String,
    /// Relevant capabilities for this use case
    pub capabilities: Vec<String>,
    /// Known limitations
    pub limitations: Vec<String>,
    /// Alternative backends
    pub alternatives: Vec<ExpertBackendAlternative>,
}

/// An alternative backend suggestion
#[derive(Debug, Serialize, Deserialize)]
pub struct ExpertBackendAlternative {
    /// Backend ID
    pub backend: BackendIdParam,
    /// Why this is a good alternative
    pub rationale: String,
    /// When to prefer this over the primary
    pub prefer_when: String,
    /// Confidence score
    pub confidence: f32,
}

/// Request for error explanation
#[derive(Debug, Deserialize)]
pub struct ExpertErrorRequest {
    /// The error message to explain
    pub message: String,
    /// Optional backend context
    pub backend: Option<BackendIdParam>,
}

/// Response for error explanation
#[derive(Debug, Serialize, Deserialize)]
pub struct ExpertErrorResponse {
    /// Original error message
    pub original_error: String,
    /// Human-readable explanation
    pub explanation: String,
    /// Root cause analysis
    pub root_cause: String,
    /// Suggested fixes
    pub suggested_fixes: Vec<ExpertSuggestedFix>,
}

/// A suggested fix for an error
#[derive(Debug, Serialize, Deserialize)]
pub struct ExpertSuggestedFix {
    /// Description of the fix
    pub description: String,
    /// Optional code example
    pub code_example: Option<String>,
    /// Confidence that this fix will work
    pub confidence: f32,
}

/// Request for tactic suggestion
#[derive(Debug, Deserialize)]
pub struct ExpertTacticRequest {
    /// The proof goal description
    pub goal: String,
    /// Target backend
    pub backend: BackendIdParam,
    /// Optional additional context
    pub context: Option<String>,
}

/// Response for tactic suggestion
#[derive(Debug, Serialize, Deserialize)]
pub struct ExpertTacticResponse {
    /// List of tactic suggestions
    pub suggestions: Vec<ExpertTacticSuggestion>,
}

/// A tactic suggestion
#[derive(Debug, Serialize, Deserialize)]
pub struct ExpertTacticSuggestion {
    /// The suggested tactic
    pub tactic: String,
    /// Backend this applies to
    pub backend: BackendIdParam,
    /// When to use this tactic
    pub when_to_use: String,
    /// Expected effect
    pub expected_effect: String,
    /// Optional usage example
    pub example: Option<String>,
    /// Confidence score
    pub confidence: f32,
    /// Alternative tactics
    pub alternatives: Vec<String>,
}

/// Request for compilation guidance
#[derive(Debug, Deserialize)]
pub struct ExpertCompileRequest {
    /// The specification to compile
    pub spec: String,
    /// Target backend
    pub backend: BackendIdParam,
}

/// Response for compilation guidance
#[derive(Debug, Serialize, Deserialize)]
pub struct ExpertCompileResponse {
    /// Summary of input
    pub input_summary: String,
    /// Target backend
    pub target_backend: BackendIdParam,
    /// Step-by-step compilation guidance
    pub steps: Vec<ExpertCompilationStep>,
    /// Common pitfalls to avoid
    pub pitfalls: Vec<String>,
    /// Best practices
    pub best_practices: Vec<String>,
}

/// A compilation step
#[derive(Debug, Serialize, Deserialize)]
pub struct ExpertCompilationStep {
    /// Step number
    pub step_number: usize,
    /// Step description
    pub description: String,
    /// Optional code example
    pub code_example: Option<String>,
    /// What to verify after this step
    pub verification: Option<String>,
}
