//! Prediction result types for the strategy module.
//!
//! This module contains the result types returned by the strategy predictor
//! for backend selection, tactic suggestions, and time estimation.

use serde::{Deserialize, Serialize};

use dashprove_backends::BackendId;

/// Backend prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendPrediction {
    /// Predicted best backend
    pub backend: BackendId,
    /// Confidence (0.0-1.0)
    pub confidence: f64,
    /// Alternative backends with their probabilities
    pub alternatives: Vec<(BackendId, f64)>,
}

/// Tactic prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TacticPrediction {
    /// Predicted tactic name
    pub tactic: String,
    /// Confidence (0.0-1.0)
    pub confidence: f64,
    /// Position in sequence (0 = first)
    pub position: usize,
}

/// Time prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimePrediction {
    /// Expected verification time in seconds
    pub expected_seconds: f64,
    /// Confidence in the prediction (0.0-1.0)
    pub confidence: f64,
    /// Expected range (min, max) seconds
    pub range: (f64, f64),
}
