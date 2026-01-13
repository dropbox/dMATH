//! LIME backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Task type for LIME explanations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LimeTaskType {
    /// Classification tasks
    #[default]
    Classification,
    /// Regression tasks
    Regression,
}

impl LimeTaskType {
    pub fn as_str(&self) -> &'static str {
        match self {
            LimeTaskType::Classification => "classification",
            LimeTaskType::Regression => "regression",
        }
    }
}

/// Kernel width strategy
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum KernelWidth {
    /// Use automatic kernel width from LIME
    #[default]
    Auto,
    /// Use a fixed width value
    Fixed(f64),
}

impl KernelWidth {
    pub fn to_python(&self) -> String {
        match self {
            KernelWidth::Auto => "None".to_string(),
            KernelWidth::Fixed(v) => v.to_string(),
        }
    }
}

/// LIME backend configuration
#[derive(Debug, Clone)]
pub struct LimeConfig {
    /// Python interpreter path
    pub python_path: Option<PathBuf>,
    /// Task type
    pub task_type: LimeTaskType,
    /// Number of features to explain
    pub num_features: usize,
    /// Number of samples to draw for local surrogate
    pub num_samples: usize,
    /// Kernel width strategy
    pub kernel_width: KernelWidth,
    /// Whether to discretize continuous features
    pub discretize_continuous: bool,
    /// Minimum local fidelity required
    pub fidelity_threshold: f64,
    /// Timeout for verification
    pub timeout: Duration,
}

impl Default for LimeConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            task_type: LimeTaskType::Classification,
            num_features: 6,
            num_samples: 1500,
            kernel_width: KernelWidth::Auto,
            discretize_continuous: true,
            fidelity_threshold: 0.65,
            timeout: Duration::from_secs(300),
        }
    }
}

impl LimeConfig {
    /// Configure LIME for regression tasks
    pub fn regression() -> Self {
        Self {
            task_type: LimeTaskType::Regression,
            fidelity_threshold: 0.6,
            ..Default::default()
        }
    }
}
