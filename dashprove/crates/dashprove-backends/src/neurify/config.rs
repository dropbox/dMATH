//! Neurify configuration
//!
//! Neurify uses symbolic interval propagation for neural network verification,
//! with efficient gradient-based refinement.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Neurify configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeurifyConfig {
    /// Path to the neural network model
    pub model_path: Option<PathBuf>,
    /// Epsilon for local robustness verification
    pub epsilon: f64,
    /// Splitting method
    pub split_method: SplitMethod,
    /// Maximum number of splits
    pub max_splits: usize,
    /// Use symbolic interval propagation
    pub use_symbolic: bool,
    /// Timeout for verification
    pub timeout: Duration,
    /// Path to Neurify binary
    pub neurify_path: Option<PathBuf>,
}

/// Splitting method for Neurify
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SplitMethod {
    /// Gradient-based splitting (default)
    #[default]
    Gradient,
    /// Input-based splitting
    Input,
    /// ReLU-based splitting
    ReLU,
    /// Adaptive splitting
    Adaptive,
}

impl SplitMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Gradient => "gradient",
            Self::Input => "input",
            Self::ReLU => "relu",
            Self::Adaptive => "adaptive",
        }
    }
}

impl Default for NeurifyConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            epsilon: 0.01,
            split_method: SplitMethod::default(),
            max_splits: 5000,
            use_symbolic: true,
            timeout: Duration::from_secs(300),
            neurify_path: None,
        }
    }
}
