//! ReluVal configuration
//!
//! ReluVal uses interval arithmetic specifically optimized for ReLU networks,
//! with efficient symbolic propagation.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// ReluVal configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReluValConfig {
    /// Path to the neural network model
    pub model_path: Option<PathBuf>,
    /// Epsilon for local robustness verification
    pub epsilon: f64,
    /// Interval refinement mode
    pub refinement_mode: RefinementMode,
    /// Maximum iterations for refinement
    pub max_iterations: usize,
    /// Precision threshold for convergence
    pub precision: f64,
    /// Timeout for verification
    pub timeout: Duration,
    /// Path to ReluVal binary
    pub reluval_path: Option<PathBuf>,
}

/// Refinement mode for ReluVal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RefinementMode {
    /// Bisection refinement (default)
    #[default]
    Bisection,
    /// Gradient-guided refinement
    Gradient,
    /// Layer-wise refinement
    LayerWise,
    /// Smear refinement
    Smear,
}

impl RefinementMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bisection => "bisection",
            Self::Gradient => "gradient",
            Self::LayerWise => "layerwise",
            Self::Smear => "smear",
        }
    }
}

impl Default for ReluValConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            epsilon: 0.01,
            refinement_mode: RefinementMode::default(),
            max_iterations: 1000,
            precision: 1e-6,
            timeout: Duration::from_secs(300),
            reluval_path: None,
        }
    }
}
