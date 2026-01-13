//! Aequitas backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Fairness metric for Aequitas
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AequitasMetric {
    /// Predictive parity
    #[default]
    PredictiveParity,
    /// False positive rate parity
    FPRParity,
    /// False negative rate parity
    FNRParity,
    /// False discovery rate parity
    FDRParity,
    /// False omission rate parity
    FORParity,
    /// Treatment equality
    TreatmentEquality,
    /// Impact parity
    ImpactParity,
}

impl AequitasMetric {
    pub fn as_str(&self) -> &'static str {
        match self {
            AequitasMetric::PredictiveParity => "predictive_parity",
            AequitasMetric::FPRParity => "fpr_parity",
            AequitasMetric::FNRParity => "fnr_parity",
            AequitasMetric::FDRParity => "fdr_parity",
            AequitasMetric::FORParity => "for_parity",
            AequitasMetric::TreatmentEquality => "treatment_equality",
            AequitasMetric::ImpactParity => "impact_parity",
        }
    }
}

/// Reference group selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReferenceGroup {
    /// Majority group
    #[default]
    Majority,
    /// Minority group
    Minority,
    /// Global average
    Global,
}

impl ReferenceGroup {
    pub fn as_str(&self) -> &'static str {
        match self {
            ReferenceGroup::Majority => "majority",
            ReferenceGroup::Minority => "minority",
            ReferenceGroup::Global => "global",
        }
    }
}

/// Aequitas backend configuration
#[derive(Debug, Clone)]
pub struct AequitasConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Fairness metric to evaluate
    pub fairness_metric: AequitasMetric,
    /// Reference group for comparison
    pub reference_group: ReferenceGroup,
    /// Disparity tolerance (e.g., 0.8 for 80% rule)
    pub disparity_tolerance: f64,
    /// Significance threshold
    pub significance_threshold: f64,
    /// Verification timeout
    pub timeout: Duration,
    /// Number of samples for analysis
    pub n_samples: usize,
}

impl Default for AequitasConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            fairness_metric: AequitasMetric::PredictiveParity,
            reference_group: ReferenceGroup::Majority,
            disparity_tolerance: 0.8,
            significance_threshold: 0.05,
            timeout: Duration::from_secs(300),
            n_samples: 1000,
        }
    }
}

impl AequitasConfig {
    /// Create config for FPR parity evaluation
    pub fn fpr_parity() -> Self {
        Self {
            fairness_metric: AequitasMetric::FPRParity,
            ..Default::default()
        }
    }

    /// Create config for FNR parity evaluation
    pub fn fnr_parity() -> Self {
        Self {
            fairness_metric: AequitasMetric::FNRParity,
            ..Default::default()
        }
    }

    /// Create config for treatment equality
    pub fn treatment_equality() -> Self {
        Self {
            fairness_metric: AequitasMetric::TreatmentEquality,
            ..Default::default()
        }
    }

    /// Create config with strict disparity tolerance
    pub fn strict(tolerance: f64) -> Self {
        Self {
            disparity_tolerance: tolerance,
            ..Default::default()
        }
    }
}
