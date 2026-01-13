//! MNBaB configuration
//!
//! MNBaB (Multi-Neuron Branch and Bound) uses multi-neuron relaxation
//! combined with branch-and-bound search for neural network verification.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// MNBaB configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MNBaBConfig {
    /// Path to the neural network model
    pub model_path: Option<PathBuf>,
    /// Epsilon for local robustness verification
    pub epsilon: f64,
    /// Branching strategy
    pub branching_strategy: BranchingStrategy,
    /// Number of neurons to consider for multi-neuron relaxation
    pub multi_neuron_count: usize,
    /// Maximum number of branches
    pub max_branches: usize,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// Timeout for verification
    pub timeout: Duration,
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
}

/// Branching strategy for MNBaB
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BranchingStrategy {
    /// Score-based branching (default)
    #[default]
    Score,
    /// FSB (Filtered Smart Branching)
    FSB,
    /// Babsr (BaB with smart refinement)
    Babsr,
    /// Input space branching
    Input,
}

impl BranchingStrategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Score => "score",
            Self::FSB => "fsb",
            Self::Babsr => "babsr",
            Self::Input => "input",
        }
    }
}

impl Default for MNBaBConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            epsilon: 0.01,
            branching_strategy: BranchingStrategy::default(),
            multi_neuron_count: 3,
            max_branches: 10000,
            use_gpu: false,
            timeout: Duration::from_secs(300),
            python_path: None,
        }
    }
}
