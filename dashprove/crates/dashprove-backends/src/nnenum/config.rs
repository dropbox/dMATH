//! Nnenum backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Enumeration strategy for nnenum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnumerationStrategy {
    /// Depth-first enumeration
    #[default]
    DepthFirst,
    /// Breadth-first enumeration
    BreadthFirst,
    /// Best-first (heuristic-guided)
    BestFirst,
    /// Mixed strategy
    Mixed,
}

impl EnumerationStrategy {
    /// Get the nnenum strategy string
    pub fn as_str(&self) -> &'static str {
        match self {
            EnumerationStrategy::DepthFirst => "depth_first",
            EnumerationStrategy::BreadthFirst => "breadth_first",
            EnumerationStrategy::BestFirst => "best_first",
            EnumerationStrategy::Mixed => "mixed",
        }
    }
}

/// Nnenum backend configuration
#[derive(Debug, Clone)]
pub struct NnenumConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Enumeration strategy
    pub strategy: EnumerationStrategy,
    /// Epsilon bound for perturbation
    pub epsilon: f64,
    /// Number of parallel processes
    pub num_processes: usize,
    /// Verification timeout
    pub timeout: Duration,
    /// Model path override
    pub model_path: Option<PathBuf>,
    /// Enable exact arithmetic (slower but complete)
    pub exact_arithmetic: bool,
    /// Maximum number of LP solver calls
    pub max_lp_calls: Option<usize>,
}

impl Default for NnenumConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            strategy: EnumerationStrategy::DepthFirst,
            epsilon: 0.01,
            num_processes: 1,
            timeout: Duration::from_secs(300),
            model_path: None,
            exact_arithmetic: false,
            max_lp_calls: None,
        }
    }
}

impl NnenumConfig {
    /// Create config optimized for speed
    pub fn fast() -> Self {
        Self {
            strategy: EnumerationStrategy::BestFirst,
            exact_arithmetic: false,
            max_lp_calls: Some(10000),
            ..Default::default()
        }
    }

    /// Create config for complete verification
    pub fn complete() -> Self {
        Self {
            strategy: EnumerationStrategy::DepthFirst,
            exact_arithmetic: true,
            max_lp_calls: None,
            ..Default::default()
        }
    }
}
