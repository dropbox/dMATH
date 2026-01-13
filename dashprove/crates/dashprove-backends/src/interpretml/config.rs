//! InterpretML backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// InterpretML explainer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpretExplainer {
    /// Explainable Boosting Machine (EBM)
    #[default]
    ExplainableBoosting,
    /// Glassbox linear model
    Linear,
    /// Decision tree glassbox model
    DecisionTree,
}

impl InterpretExplainer {
    pub fn as_str(&self) -> &'static str {
        match self {
            InterpretExplainer::ExplainableBoosting => "ebm",
            InterpretExplainer::Linear => "linear",
            InterpretExplainer::DecisionTree => "tree",
        }
    }
}

/// Task type for InterpretML
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpretTask {
    /// Classification tasks
    #[default]
    Classification,
    /// Regression tasks
    Regression,
}

impl InterpretTask {
    pub fn as_str(&self) -> &'static str {
        match self {
            InterpretTask::Classification => "classification",
            InterpretTask::Regression => "regression",
        }
    }
}

/// InterpretML backend configuration
#[derive(Debug, Clone)]
pub struct InterpretMlConfig {
    /// Optional Python path
    pub python_path: Option<PathBuf>,
    /// Explainer to use
    pub explainer: InterpretExplainer,
    /// Task type
    pub task: InterpretTask,
    /// Maximum bins for EBM
    pub max_bins: u32,
    /// Maximum interactions
    pub max_interactions: usize,
    /// Validation fraction for data split
    pub validation_fraction: f64,
    /// Minimum mean importance required
    pub importance_threshold: f64,
    /// Timeout for verification
    pub timeout: Duration,
}

impl Default for InterpretMlConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            explainer: InterpretExplainer::ExplainableBoosting,
            task: InterpretTask::Classification,
            max_bins: 16,
            max_interactions: 2,
            validation_fraction: 0.2,
            importance_threshold: 0.01,
            timeout: Duration::from_secs(300),
        }
    }
}

impl InterpretMlConfig {
    /// Configure for regression tasks
    pub fn regression() -> Self {
        Self {
            task: InterpretTask::Regression,
            ..Default::default()
        }
    }

    /// Use a linear glassbox model
    pub fn linear() -> Self {
        Self {
            explainer: InterpretExplainer::Linear,
            ..Default::default()
        }
    }
}
