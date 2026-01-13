//! SHAP backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Explainer type for SHAP
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShapExplainer {
    /// KernelExplainer - model agnostic
    #[default]
    Kernel,
    /// TreeExplainer - optimized for tree models
    Tree,
    /// LinearExplainer - linear models
    Linear,
    /// DeepExplainer - deep learning models
    Deep,
    /// GradientExplainer - differentiable models
    Gradient,
}

impl ShapExplainer {
    pub fn as_str(&self) -> &'static str {
        match self {
            ShapExplainer::Kernel => "kernel",
            ShapExplainer::Tree => "tree",
            ShapExplainer::Linear => "linear",
            ShapExplainer::Deep => "deep",
            ShapExplainer::Gradient => "gradient",
        }
    }
}

/// Model/task type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShapModelType {
    /// Binary classification
    #[default]
    Classification,
    /// Regression
    Regression,
}

impl ShapModelType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ShapModelType::Classification => "classification",
            ShapModelType::Regression => "regression",
        }
    }
}

/// SHAP backend configuration
#[derive(Debug, Clone)]
pub struct ShapConfig {
    /// Optional Python interpreter path
    pub python_path: Option<PathBuf>,
    /// Explainer type
    pub explainer: ShapExplainer,
    /// Task type
    pub model_type: ShapModelType,
    /// Number of evaluation samples
    pub sample_size: usize,
    /// Background sample size
    pub background_size: usize,
    /// Maximum features to include in report
    pub max_features: usize,
    /// Minimum mean absolute SHAP value for verification
    pub importance_threshold: f64,
    /// Whether to compute stability with two runs
    pub evaluate_stability: bool,
    /// Timeout for verification run
    pub timeout: Duration,
}

impl Default for ShapConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            explainer: ShapExplainer::Kernel,
            model_type: ShapModelType::Classification,
            sample_size: 400,
            background_size: 80,
            max_features: 6,
            importance_threshold: 0.01,
            evaluate_stability: true,
            timeout: Duration::from_secs(300),
        }
    }
}

impl ShapConfig {
    /// Use the optimized tree explainer
    pub fn tree() -> Self {
        Self {
            explainer: ShapExplainer::Tree,
            ..Default::default()
        }
    }

    /// Use the deep explainer for neural networks
    pub fn deep() -> Self {
        Self {
            explainer: ShapExplainer::Deep,
            ..Default::default()
        }
    }

    /// Configure for regression tasks
    pub fn regression() -> Self {
        Self {
            model_type: ShapModelType::Regression,
            ..Default::default()
        }
    }
}
