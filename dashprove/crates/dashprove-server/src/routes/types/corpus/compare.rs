use std::collections::HashMap;

use dashprove_learning::HistoryComparison;
use serde::{Deserialize, Serialize};

use crate::routes::types::{default_corpus_type, CorpusType, OutputFormat};

// ============ Corpus Compare Types ============

/// Query parameters for corpus compare endpoint
#[derive(Debug, Deserialize)]
pub struct CorpusCompareQuery {
    /// Which corpus to compare
    #[serde(default = "default_corpus_type")]
    pub corpus: CorpusType,
    /// Baseline period start date (YYYY-MM-DD)
    pub baseline_from: String,
    /// Baseline period end date (YYYY-MM-DD)
    pub baseline_to: String,
    /// Comparison period start date (YYYY-MM-DD)
    pub compare_from: String,
    /// Comparison period end date (YYYY-MM-DD)
    pub compare_to: String,
    /// Output format (json or html)
    #[serde(default)]
    pub format: OutputFormat,
}

/// Response for corpus comparison
#[derive(Debug, Serialize, Deserialize)]
pub struct CorpusCompareResponse {
    /// Baseline period label
    pub baseline_label: String,
    /// Comparison period label
    pub comparison_label: String,
    /// Total in baseline
    pub baseline_count: usize,
    /// Total in comparison
    pub comparison_count: usize,
    /// Absolute change
    pub count_delta: i64,
    /// Percentage change
    pub count_percent_change: Option<f64>,
    /// Backend-level changes
    pub backend_deltas: HashMap<String, i64>,
    /// Backends new in comparison period
    pub new_backends: Vec<String>,
    /// Backends removed in comparison period
    pub removed_backends: Vec<String>,
    /// Number of periods in baseline
    pub baseline_period_count: usize,
    /// Number of periods in comparison
    pub comparison_period_count: usize,
    /// Average per period in baseline
    pub baseline_avg_per_period: f64,
    /// Average per period in comparison
    pub comparison_avg_per_period: f64,
    /// Growth rate per period
    pub growth_rate_per_period: Option<f64>,
    /// Compound growth rate
    pub compound_growth_rate: Option<f64>,
}

impl From<HistoryComparison> for CorpusCompareResponse {
    fn from(c: HistoryComparison) -> Self {
        CorpusCompareResponse {
            baseline_label: c.baseline_label,
            comparison_label: c.comparison_label,
            baseline_count: c.baseline_count,
            comparison_count: c.comparison_count,
            count_delta: c.count_delta,
            count_percent_change: c.count_percent_change,
            backend_deltas: c
                .backend_deltas
                .into_iter()
                .map(|(k, v)| (format!("{:?}", k), v))
                .collect(),
            new_backends: c
                .new_backends
                .into_iter()
                .map(|b| format!("{:?}", b))
                .collect(),
            removed_backends: c
                .removed_backends
                .into_iter()
                .map(|b| format!("{:?}", b))
                .collect(),
            baseline_period_count: c.baseline_period_count,
            comparison_period_count: c.comparison_period_count,
            baseline_avg_per_period: c.baseline_avg_per_period,
            comparison_avg_per_period: c.comparison_avg_per_period,
            growth_rate_per_period: c.growth_rate_per_period,
            compound_growth_rate: c.compound_growth_rate,
        }
    }
}
