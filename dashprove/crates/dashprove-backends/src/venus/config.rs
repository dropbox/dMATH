//! Venus backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Solver backend for Venus
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverBackend {
    /// Gurobi LP solver (commercial, fast)
    #[default]
    Gurobi,
    /// GLPK solver (open source)
    GLPK,
    /// CBC solver (open source)
    CBC,
}

impl SolverBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            SolverBackend::Gurobi => "gurobi",
            SolverBackend::GLPK => "glpk",
            SolverBackend::CBC => "cbc",
        }
    }
}

/// Venus backend configuration
#[derive(Debug, Clone)]
pub struct VenusConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// LP solver backend
    pub solver: SolverBackend,
    /// Epsilon bound for perturbation
    pub epsilon: f64,
    /// Number of parallel workers
    pub num_workers: usize,
    /// Verification timeout
    pub timeout: Duration,
    /// Model path override
    pub model_path: Option<PathBuf>,
    /// Enable branch and bound
    pub use_bnb: bool,
}

impl Default for VenusConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            solver: SolverBackend::Gurobi,
            epsilon: 0.01,
            num_workers: 1,
            timeout: Duration::from_secs(300),
            model_path: None,
            use_bnb: true,
        }
    }
}

impl VenusConfig {
    pub fn with_glpk() -> Self {
        Self {
            solver: SolverBackend::GLPK,
            ..Default::default()
        }
    }
}
