//! Alibi backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Alibi explainer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlibiExplainer {
    /// Anchor explanations for tabular data
    #[default]
    AnchorTabular,
    /// Anchor explanations for text data
    AnchorText,
    /// Counterfactual explanations
    Counterfactual,
}

impl AlibiExplainer {
    pub fn as_str(&self) -> &'static str {
        match self {
            AlibiExplainer::AnchorTabular => "anchor_tabular",
            AlibiExplainer::AnchorText => "anchor_text",
            AlibiExplainer::Counterfactual => "counterfactual",
        }
    }
}

/// Alibi backend configuration
#[derive(Debug, Clone)]
pub struct AlibiConfig {
    /// Python interpreter path
    pub python_path: Option<PathBuf>,
    /// Explainer type
    pub explainer: AlibiExplainer,
    /// Number of samples to use
    pub sample_size: usize,
    /// Precision threshold required
    pub precision_threshold: f64,
    /// Coverage threshold required
    pub coverage_threshold: f64,
    /// Timeout for verification
    pub timeout: Duration,
}

impl Default for AlibiConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            explainer: AlibiExplainer::AnchorTabular,
            sample_size: 800,
            precision_threshold: 0.8,
            coverage_threshold: 0.6,
            timeout: Duration::from_secs(300),
        }
    }
}

impl AlibiConfig {
    /// Configure counterfactual explanations
    pub fn counterfactual() -> Self {
        Self {
            explainer: AlibiExplainer::Counterfactual,
            precision_threshold: 0.75,
            ..Default::default()
        }
    }
}
