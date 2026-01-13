//! Fairlearn backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Fairness metric to evaluate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FairnessMetric {
    /// Demographic parity difference
    #[default]
    DemographicParity,
    /// Equalized odds difference
    EqualizedOdds,
    /// Equal opportunity difference
    EqualOpportunity,
    /// True positive rate parity
    TruePositiveRateParity,
    /// False positive rate parity
    FalsePositiveRateParity,
}

impl FairnessMetric {
    pub fn as_str(&self) -> &'static str {
        match self {
            FairnessMetric::DemographicParity => "demographic_parity",
            FairnessMetric::EqualizedOdds => "equalized_odds",
            FairnessMetric::EqualOpportunity => "equal_opportunity",
            FairnessMetric::TruePositiveRateParity => "true_positive_rate_parity",
            FairnessMetric::FalsePositiveRateParity => "false_positive_rate_parity",
        }
    }
}

/// Mitigation method for bias reduction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MitigationMethod {
    /// No mitigation (assessment only)
    #[default]
    None,
    /// Exponentiated gradient reduction
    ExponentiatedGradient,
    /// Grid search reduction
    GridSearch,
    /// Threshold optimizer
    ThresholdOptimizer,
}

impl MitigationMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            MitigationMethod::None => "none",
            MitigationMethod::ExponentiatedGradient => "exponentiated_gradient",
            MitigationMethod::GridSearch => "grid_search",
            MitigationMethod::ThresholdOptimizer => "threshold_optimizer",
        }
    }
}

/// Constraint type for mitigation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FairnessConstraint {
    /// Demographic parity constraint
    #[default]
    DemographicParity,
    /// Equalized odds constraint
    EqualizedOdds,
    /// True positive rate parity
    TruePositiveRateParity,
    /// False positive rate parity
    FalsePositiveRateParity,
    /// Bounded group loss
    BoundedGroupLoss,
}

impl FairnessConstraint {
    pub fn as_str(&self) -> &'static str {
        match self {
            FairnessConstraint::DemographicParity => "demographic_parity",
            FairnessConstraint::EqualizedOdds => "equalized_odds",
            FairnessConstraint::TruePositiveRateParity => "true_positive_rate_parity",
            FairnessConstraint::FalsePositiveRateParity => "false_positive_rate_parity",
            FairnessConstraint::BoundedGroupLoss => "bounded_group_loss",
        }
    }
}

/// Fairlearn backend configuration
#[derive(Debug, Clone)]
pub struct FairlearnConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Fairness metric to evaluate
    pub fairness_metric: FairnessMetric,
    /// Mitigation method
    pub mitigation_method: MitigationMethod,
    /// Fairness constraint for mitigation
    pub fairness_constraint: FairnessConstraint,
    /// Fairness threshold (acceptable difference)
    pub fairness_threshold: f64,
    /// Verification timeout
    pub timeout: Duration,
    /// Number of samples for analysis
    pub n_samples: usize,
}

impl Default for FairlearnConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            fairness_metric: FairnessMetric::DemographicParity,
            mitigation_method: MitigationMethod::None,
            fairness_constraint: FairnessConstraint::DemographicParity,
            fairness_threshold: 0.1,
            timeout: Duration::from_secs(300),
            n_samples: 1000,
        }
    }
}

impl FairlearnConfig {
    /// Create config for equalized odds evaluation
    pub fn equalized_odds() -> Self {
        Self {
            fairness_metric: FairnessMetric::EqualizedOdds,
            fairness_constraint: FairnessConstraint::EqualizedOdds,
            ..Default::default()
        }
    }

    /// Create config with exponentiated gradient mitigation
    pub fn with_mitigation() -> Self {
        Self {
            mitigation_method: MitigationMethod::ExponentiatedGradient,
            ..Default::default()
        }
    }

    /// Create config for threshold optimization
    pub fn threshold_optimizer() -> Self {
        Self {
            mitigation_method: MitigationMethod::ThresholdOptimizer,
            ..Default::default()
        }
    }

    /// Create config with strict fairness threshold
    pub fn strict(threshold: f64) -> Self {
        Self {
            fairness_threshold: threshold,
            ..Default::default()
        }
    }
}
