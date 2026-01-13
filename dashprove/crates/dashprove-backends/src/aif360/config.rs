//! AIF360 backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Bias metric to evaluate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BiasMetric {
    /// Statistical parity difference
    #[default]
    StatisticalParityDifference,
    /// Disparate impact
    DisparateImpact,
    /// Average odds difference
    AverageOddsDifference,
    /// Equal opportunity difference
    EqualOpportunityDifference,
    /// Theil index
    TheilIndex,
}

impl BiasMetric {
    pub fn as_str(&self) -> &'static str {
        match self {
            BiasMetric::StatisticalParityDifference => "statistical_parity_difference",
            BiasMetric::DisparateImpact => "disparate_impact",
            BiasMetric::AverageOddsDifference => "average_odds_difference",
            BiasMetric::EqualOpportunityDifference => "equal_opportunity_difference",
            BiasMetric::TheilIndex => "theil_index",
        }
    }
}

/// Mitigation algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AIF360MitigationAlgorithm {
    /// No mitigation
    #[default]
    None,
    /// Reweighing (pre-processing)
    Reweighing,
    /// Disparate impact remover (pre-processing)
    DisparateImpactRemover,
    /// Learning fair representations (pre-processing)
    LFR,
    /// Prejudice remover (in-processing)
    PrejudiceRemover,
    /// Adversarial debiasing (in-processing)
    AdversarialDebiasing,
    /// Calibrated equalized odds (post-processing)
    CalibratedEqOdds,
    /// Reject option classification (post-processing)
    RejectOptionClassification,
}

impl AIF360MitigationAlgorithm {
    pub fn as_str(&self) -> &'static str {
        match self {
            AIF360MitigationAlgorithm::None => "none",
            AIF360MitigationAlgorithm::Reweighing => "reweighing",
            AIF360MitigationAlgorithm::DisparateImpactRemover => "disparate_impact_remover",
            AIF360MitigationAlgorithm::LFR => "lfr",
            AIF360MitigationAlgorithm::PrejudiceRemover => "prejudice_remover",
            AIF360MitigationAlgorithm::AdversarialDebiasing => "adversarial_debiasing",
            AIF360MitigationAlgorithm::CalibratedEqOdds => "calibrated_eq_odds",
            AIF360MitigationAlgorithm::RejectOptionClassification => "reject_option_classification",
        }
    }
}

/// AIF360 backend configuration
#[derive(Debug, Clone)]
pub struct AIF360Config {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Bias metric to evaluate
    pub bias_metric: BiasMetric,
    /// Mitigation algorithm
    pub mitigation_algorithm: AIF360MitigationAlgorithm,
    /// Fairness threshold
    pub fairness_threshold: f64,
    /// Disparate impact threshold (80% rule)
    pub disparate_impact_threshold: f64,
    /// Verification timeout
    pub timeout: Duration,
    /// Number of samples for analysis
    pub n_samples: usize,
}

impl Default for AIF360Config {
    fn default() -> Self {
        Self {
            python_path: None,
            bias_metric: BiasMetric::StatisticalParityDifference,
            mitigation_algorithm: AIF360MitigationAlgorithm::None,
            fairness_threshold: 0.1,
            disparate_impact_threshold: 0.8,
            timeout: Duration::from_secs(300),
            n_samples: 1000,
        }
    }
}

impl AIF360Config {
    /// Create config for disparate impact evaluation
    pub fn disparate_impact() -> Self {
        Self {
            bias_metric: BiasMetric::DisparateImpact,
            ..Default::default()
        }
    }

    /// Create config with reweighing mitigation
    pub fn with_reweighing() -> Self {
        Self {
            mitigation_algorithm: AIF360MitigationAlgorithm::Reweighing,
            ..Default::default()
        }
    }

    /// Create config for equalized odds
    pub fn equalized_odds() -> Self {
        Self {
            bias_metric: BiasMetric::AverageOddsDifference,
            ..Default::default()
        }
    }

    /// Create config with calibrated equalized odds post-processing
    pub fn calibrated_eq_odds() -> Self {
        Self {
            mitigation_algorithm: AIF360MitigationAlgorithm::CalibratedEqOdds,
            ..Default::default()
        }
    }
}
