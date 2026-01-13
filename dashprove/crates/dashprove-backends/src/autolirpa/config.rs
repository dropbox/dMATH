//! AutoLiRPA backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Bound method for AutoLiRPA
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundMethod {
    /// Interval Bound Propagation (fastest, loosest)
    #[default]
    IBP,
    /// CROWN (backward mode)
    CROWN,
    /// IBP + CROWN (hybrid)
    IBPCrown,
    /// Forward + CROWN (more precise)
    ForwardCrown,
    /// Alpha-CROWN (tightest, slowest)
    AlphaCrown,
}

impl BoundMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            BoundMethod::IBP => "IBP",
            BoundMethod::CROWN => "CROWN",
            BoundMethod::IBPCrown => "IBP+CROWN",
            BoundMethod::ForwardCrown => "Forward+CROWN",
            BoundMethod::AlphaCrown => "alpha-CROWN",
        }
    }
}

/// AutoLiRPA backend configuration
#[derive(Debug, Clone)]
pub struct AutoLirpaConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Bound computation method
    pub bound_method: BoundMethod,
    /// Epsilon bound for perturbation
    pub epsilon: f64,
    /// Verification timeout
    pub timeout: Duration,
    /// Model path override
    pub model_path: Option<PathBuf>,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// Number of optimization iterations (for alpha-CROWN)
    pub opt_iterations: usize,
}

impl Default for AutoLirpaConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            bound_method: BoundMethod::IBP,
            epsilon: 0.01,
            timeout: Duration::from_secs(300),
            model_path: None,
            use_gpu: false,
            opt_iterations: 20,
        }
    }
}

impl AutoLirpaConfig {
    /// Create config with CROWN (more precise)
    pub fn crown() -> Self {
        Self {
            bound_method: BoundMethod::CROWN,
            ..Default::default()
        }
    }

    /// Create config with alpha-CROWN (tightest bounds)
    pub fn alpha_crown() -> Self {
        Self {
            bound_method: BoundMethod::AlphaCrown,
            opt_iterations: 100,
            ..Default::default()
        }
    }
}
