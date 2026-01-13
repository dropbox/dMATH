//! NNV backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Verification method for NNV
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VerificationMethod {
    /// Star set reachability (exact but slow)
    #[default]
    Star,
    /// Zonotope abstraction (fast but approximate)
    Zonotope,
    /// Abstract interpretation with polytopes
    Polytope,
    /// Hybrid approach (star + zonotope)
    Hybrid,
}

impl VerificationMethod {
    /// Get the NNV method string
    pub fn as_str(&self) -> &'static str {
        match self {
            VerificationMethod::Star => "star",
            VerificationMethod::Zonotope => "zonotope",
            VerificationMethod::Polytope => "polytope",
            VerificationMethod::Hybrid => "hybrid",
        }
    }
}

/// NNV backend configuration
#[derive(Debug, Clone)]
pub struct NnvConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Verification method
    pub method: VerificationMethod,
    /// Epsilon bound for perturbation
    pub epsilon: f64,
    /// Number of parallel workers
    pub num_workers: usize,
    /// Verification timeout
    pub timeout: Duration,
    /// Model path override
    pub model_path: Option<PathBuf>,
    /// Use GPU acceleration if available
    pub use_gpu: bool,
}

impl Default for NnvConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            method: VerificationMethod::Star,
            epsilon: 0.01,
            num_workers: 1,
            timeout: Duration::from_secs(300),
            model_path: None,
            use_gpu: false,
        }
    }
}

impl NnvConfig {
    /// Create config with zonotope method (faster)
    pub fn fast() -> Self {
        Self {
            method: VerificationMethod::Zonotope,
            ..Default::default()
        }
    }

    /// Create config with hybrid method
    pub fn hybrid() -> Self {
        Self {
            method: VerificationMethod::Hybrid,
            ..Default::default()
        }
    }
}
