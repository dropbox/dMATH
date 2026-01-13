//! Deepchecks backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Validation suite type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SuiteType {
    /// Data integrity suite
    #[default]
    DataIntegrity,
    /// Train-test validation suite
    TrainTestValidation,
    /// Model evaluation suite
    ModelEvaluation,
    /// Full validation suite
    FullSuite,
}

impl SuiteType {
    pub fn as_str(&self) -> &'static str {
        match self {
            SuiteType::DataIntegrity => "data_integrity",
            SuiteType::TrainTestValidation => "train_test_validation",
            SuiteType::ModelEvaluation => "model_evaluation",
            SuiteType::FullSuite => "full_suite",
        }
    }
}

/// Task type for model validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TaskType {
    /// Binary classification
    #[default]
    BinaryClassification,
    /// Multiclass classification
    MulticlassClassification,
    /// Regression
    Regression,
    /// Object detection
    ObjectDetection,
    /// Semantic segmentation
    SemanticSegmentation,
}

impl TaskType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskType::BinaryClassification => "binary",
            TaskType::MulticlassClassification => "multiclass",
            TaskType::Regression => "regression",
            TaskType::ObjectDetection => "object_detection",
            TaskType::SemanticSegmentation => "semantic_segmentation",
        }
    }
}

/// Severity threshold for checks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeverityThreshold {
    /// Low severity
    Low,
    /// Medium severity
    #[default]
    Medium,
    /// High severity only
    High,
}

impl SeverityThreshold {
    pub fn as_str(&self) -> &'static str {
        match self {
            SeverityThreshold::Low => "low",
            SeverityThreshold::Medium => "medium",
            SeverityThreshold::High => "high",
        }
    }
}

/// Deepchecks backend configuration
#[derive(Debug, Clone)]
pub struct DeepchecksConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Suite type to run
    pub suite_type: SuiteType,
    /// Task type for model validation
    pub task_type: TaskType,
    /// Severity threshold
    pub severity_threshold: SeverityThreshold,
    /// Verification timeout
    pub timeout: Duration,
    /// Run with conditions
    pub with_conditions: bool,
    /// Show only failed checks
    pub show_only_failed: bool,
    /// Number of samples for analysis
    pub n_samples: usize,
}

impl Default for DeepchecksConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            suite_type: SuiteType::DataIntegrity,
            task_type: TaskType::BinaryClassification,
            severity_threshold: SeverityThreshold::Medium,
            timeout: Duration::from_secs(300),
            with_conditions: true,
            show_only_failed: false,
            n_samples: 1000,
        }
    }
}

impl DeepchecksConfig {
    /// Create config for train-test validation
    pub fn train_test() -> Self {
        Self {
            suite_type: SuiteType::TrainTestValidation,
            ..Default::default()
        }
    }

    /// Create config for model evaluation
    pub fn model_evaluation(task_type: TaskType) -> Self {
        Self {
            suite_type: SuiteType::ModelEvaluation,
            task_type,
            ..Default::default()
        }
    }

    /// Create config for full validation
    pub fn full_suite() -> Self {
        Self {
            suite_type: SuiteType::FullSuite,
            ..Default::default()
        }
    }

    /// Create config for regression tasks
    pub fn regression() -> Self {
        Self {
            task_type: TaskType::Regression,
            ..Default::default()
        }
    }
}
