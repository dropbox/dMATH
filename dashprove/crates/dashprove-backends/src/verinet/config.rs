//! VeriNet backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Splitting strategy for VeriNet
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplittingStrategy {
    /// Input splitting
    #[default]
    Input,
    /// ReLU splitting (more precise)
    ReLU,
    /// Adaptive splitting
    Adaptive,
}

impl SplittingStrategy {
    /// Get the VeriNet strategy string
    pub fn as_str(&self) -> &'static str {
        match self {
            SplittingStrategy::Input => "input",
            SplittingStrategy::ReLU => "relu",
            SplittingStrategy::Adaptive => "adaptive",
        }
    }
}

/// VeriNet backend configuration
#[derive(Debug, Clone)]
pub struct VeriNetConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Splitting strategy
    pub strategy: SplittingStrategy,
    /// Epsilon bound for perturbation
    pub epsilon: f64,
    /// Maximum splitting depth
    pub max_depth: usize,
    /// Verification timeout
    pub timeout: Duration,
    /// Model path override
    pub model_path: Option<PathBuf>,
    /// Use GPU acceleration
    pub use_gpu: bool,
}

impl Default for VeriNetConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            strategy: SplittingStrategy::Input,
            epsilon: 0.01,
            max_depth: 15,
            timeout: Duration::from_secs(300),
            model_path: None,
            use_gpu: false,
        }
    }
}

impl VeriNetConfig {
    /// Create config optimized for completeness
    pub fn complete() -> Self {
        Self {
            strategy: SplittingStrategy::ReLU,
            max_depth: 20,
            ..Default::default()
        }
    }

    /// Create config optimized for speed
    pub fn fast() -> Self {
        Self {
            strategy: SplittingStrategy::Input,
            max_depth: 10,
            ..Default::default()
        }
    }
}
