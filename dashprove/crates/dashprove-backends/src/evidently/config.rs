//! Evidently backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Report type for Evidently
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReportType {
    /// Data drift report
    #[default]
    DataDrift,
    /// Data quality report
    DataQuality,
    /// Regression performance report
    RegressionPerformance,
    /// Classification performance report
    ClassificationPerformance,
    /// Target drift report
    TargetDrift,
}

impl ReportType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ReportType::DataDrift => "data_drift",
            ReportType::DataQuality => "data_quality",
            ReportType::RegressionPerformance => "regression_performance",
            ReportType::ClassificationPerformance => "classification_performance",
            ReportType::TargetDrift => "target_drift",
        }
    }
}

/// Statistical test method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StatTestMethod {
    /// Kolmogorov-Smirnov test
    #[default]
    KolmogorovSmirnov,
    /// Chi-squared test
    ChiSquared,
    /// Population Stability Index
    PSI,
    /// Jensen-Shannon divergence
    JensenShannon,
    /// Wasserstein distance
    Wasserstein,
}

impl StatTestMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            StatTestMethod::KolmogorovSmirnov => "ks",
            StatTestMethod::ChiSquared => "chi2",
            StatTestMethod::PSI => "psi",
            StatTestMethod::JensenShannon => "jensenshannon",
            StatTestMethod::Wasserstein => "wasserstein",
        }
    }
}

/// Output format for reports
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// JSON output
    #[default]
    JSON,
    /// HTML report
    HTML,
    /// Dictionary (Python dict)
    Dict,
}

impl OutputFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            OutputFormat::JSON => "json",
            OutputFormat::HTML => "html",
            OutputFormat::Dict => "dict",
        }
    }
}

/// Evidently backend configuration
#[derive(Debug, Clone)]
pub struct EvidentlyConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Report type
    pub report_type: ReportType,
    /// Statistical test method
    pub stat_test_method: StatTestMethod,
    /// Output format
    pub output_format: OutputFormat,
    /// Drift threshold
    pub drift_threshold: f64,
    /// Verification timeout
    pub timeout: Duration,
    /// Number of samples for analysis
    pub n_samples: usize,
    /// Stattest threshold
    pub stattest_threshold: f64,
}

impl Default for EvidentlyConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            report_type: ReportType::DataDrift,
            stat_test_method: StatTestMethod::KolmogorovSmirnov,
            output_format: OutputFormat::JSON,
            drift_threshold: 0.1,
            timeout: Duration::from_secs(300),
            n_samples: 1000,
            stattest_threshold: 0.05,
        }
    }
}

impl EvidentlyConfig {
    /// Create config for data quality report
    pub fn data_quality() -> Self {
        Self {
            report_type: ReportType::DataQuality,
            ..Default::default()
        }
    }

    /// Create config for classification performance
    pub fn classification_performance() -> Self {
        Self {
            report_type: ReportType::ClassificationPerformance,
            ..Default::default()
        }
    }

    /// Create config for regression performance
    pub fn regression_performance() -> Self {
        Self {
            report_type: ReportType::RegressionPerformance,
            ..Default::default()
        }
    }

    /// Create config for target drift detection
    pub fn target_drift() -> Self {
        Self {
            report_type: ReportType::TargetDrift,
            ..Default::default()
        }
    }

    /// Create config with PSI statistical test
    pub fn with_psi() -> Self {
        Self {
            stat_test_method: StatTestMethod::PSI,
            ..Default::default()
        }
    }
}
