use serde::{Deserialize, Serialize};

use crate::routes::types::BackendIdParam;

// ============ Request/Response Types ============

/// Request to verify a specification
#[derive(Debug, Deserialize)]
pub struct VerifyRequest {
    /// USL specification source code
    pub spec: String,
    /// Optional: specific backend to use
    pub backend: Option<BackendIdParam>,
    /// Enable ML-based backend selection (overrides `backend` if true)
    #[serde(default)]
    pub use_ml: bool,
    /// Minimum confidence threshold for ML predictions (0.0-1.0)
    #[serde(default = "default_ml_confidence")]
    pub ml_min_confidence: f64,
}

pub fn default_ml_confidence() -> f64 {
    0.5
}

/// Response from verify endpoint
#[derive(Debug, Serialize, Deserialize)]
pub struct VerifyResponse {
    /// Whether parsing and type-checking succeeded
    pub valid: bool,
    /// Number of properties found
    pub property_count: usize,
    /// Compilation outputs per backend
    pub compilations: Vec<CompilationResult>,
    /// Errors (if any)
    pub errors: Vec<String>,
    /// ML prediction info (when use_ml is true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ml_prediction: Option<MlPredictionInfo>,
}

/// ML prediction information in verification response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlPredictionInfo {
    /// Predicted best backend
    pub predicted_backend: BackendIdParam,
    /// Prediction confidence (0.0-1.0)
    pub confidence: f64,
    /// Whether the prediction was used (confidence >= threshold)
    pub used: bool,
    /// Alternative backends with their confidence scores
    pub alternatives: Vec<(BackendIdParam, f64)>,
}

/// Compilation result for a single backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationResult {
    /// Which backend this compilation is for
    pub backend: BackendIdParam,
    /// The compiled code for this backend
    pub code: String,
}
