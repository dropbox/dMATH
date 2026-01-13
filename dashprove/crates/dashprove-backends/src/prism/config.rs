//! Configuration types for PRISM backend

use std::path::PathBuf;
use std::time::Duration;

/// State variable extracted from USL spec
#[derive(Debug, Clone)]
pub struct StateVar {
    /// Variable name
    pub name: String,
    /// Minimum value
    pub min: i32,
    /// Maximum value
    pub max: i32,
    /// Initial value
    pub init: i32,
}

/// Configuration for PRISM backend
#[derive(Debug, Clone)]
pub struct PrismConfig {
    /// Path to PRISM binary
    pub prism_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Engine to use (hybrid, sparse, mtbdd, explicit)
    pub engine: PrismEngine,
    /// Numerical precision
    pub precision: f64,
    /// Maximum iterations for iterative methods
    pub max_iters: Option<usize>,
}

/// PRISM computation engine
#[derive(Debug, Clone, Copy, Default)]
pub enum PrismEngine {
    /// Hybrid engine (default) - combines symbolic/explicit
    #[default]
    Hybrid,
    /// Sparse matrix engine
    Sparse,
    /// Pure MTBDD engine (symbolic)
    Mtbdd,
    /// Explicit state engine
    Explicit,
}

impl PrismEngine {
    pub fn as_arg(&self) -> &'static str {
        match self {
            Self::Hybrid => "-h",
            Self::Sparse => "-s",
            Self::Mtbdd => "-m",
            Self::Explicit => "-ex",
        }
    }
}

impl Default for PrismConfig {
    fn default() -> Self {
        Self {
            prism_path: None,
            timeout: Duration::from_secs(300),
            engine: PrismEngine::default(),
            precision: 1e-6,
            max_iters: None,
        }
    }
}
