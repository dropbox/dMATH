use std::collections::HashMap;

use dashprove_learning::{CorpusHistory, ProofHistory, TimePeriod};
use serde::{Deserialize, Serialize};

// ============ Corpus History Types ============

/// Output format for endpoints that support multiple formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// JSON output format (default)
    #[default]
    Json,
    /// HTML output format with interactive visualizations
    Html,
}

/// Query parameters for corpus history endpoint
#[derive(Debug, Deserialize)]
pub struct CorpusHistoryQuery {
    /// Which corpus to get history for (proofs or counterexamples)
    #[serde(default = "default_corpus_type")]
    pub corpus: CorpusType,
    /// Time period granularity
    #[serde(default)]
    pub period: TimePeriodParam,
    /// Filter: start date (YYYY-MM-DD)
    pub from: Option<String>,
    /// Filter: end date (YYYY-MM-DD)
    pub to: Option<String>,
    /// Output format (json or html)
    #[serde(default)]
    pub format: OutputFormat,
}

pub fn default_corpus_type() -> CorpusType {
    CorpusType::Counterexamples
}

/// Corpus type parameter
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CorpusType {
    /// Successful proof corpus
    Proofs,
    /// Counterexample corpus (default)
    #[default]
    Counterexamples,
}

/// Time period parameter for API
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TimePeriodParam {
    /// Daily aggregation (default)
    #[default]
    Day,
    /// Weekly aggregation
    Week,
    /// Monthly aggregation
    Month,
}

impl From<TimePeriodParam> for TimePeriod {
    fn from(p: TimePeriodParam) -> Self {
        match p {
            TimePeriodParam::Day => TimePeriod::Day,
            TimePeriodParam::Week => TimePeriod::Week,
            TimePeriodParam::Month => TimePeriod::Month,
        }
    }
}

impl From<TimePeriod> for TimePeriodParam {
    fn from(p: TimePeriod) -> Self {
        match p {
            TimePeriod::Day => TimePeriodParam::Day,
            TimePeriod::Week => TimePeriodParam::Week,
            TimePeriod::Month => TimePeriodParam::Month,
        }
    }
}

/// Response for corpus history
#[derive(Debug, Serialize, Deserialize)]
pub struct CorpusHistoryResponse {
    /// Total count in corpus
    pub total_count: usize,
    /// First recorded timestamp (ISO 8601)
    pub first_recorded: Option<String>,
    /// Last recorded timestamp (ISO 8601)
    pub last_recorded: Option<String>,
    /// Period granularity used
    pub period_type: TimePeriodParam,
    /// Stats per period
    pub periods: Vec<PeriodStatsResponse>,
    /// Cumulative totals per period
    pub cumulative_counts: Vec<usize>,
    /// Count by backend
    pub by_backend: HashMap<String, usize>,
}

/// Period statistics in response
#[derive(Debug, Serialize, Deserialize)]
pub struct PeriodStatsResponse {
    /// Period key (e.g., "2024-01-15")
    pub period: String,
    /// Start timestamp (ISO 8601)
    pub start: String,
    /// Total count in this period
    pub count: usize,
    /// Count by backend
    pub by_backend: HashMap<String, usize>,
}

impl From<CorpusHistory> for CorpusHistoryResponse {
    fn from(h: CorpusHistory) -> Self {
        CorpusHistoryResponse {
            total_count: h.total_count,
            first_recorded: h.first_recorded.map(|dt| dt.to_rfc3339()),
            last_recorded: h.last_recorded.map(|dt| dt.to_rfc3339()),
            period_type: h.period_type.into(),
            periods: h
                .periods
                .into_iter()
                .map(|p| PeriodStatsResponse {
                    period: p.period,
                    start: p.start.to_rfc3339(),
                    count: p.count,
                    by_backend: p
                        .by_backend
                        .into_iter()
                        .map(|(k, v)| (format!("{:?}", k), v))
                        .collect(),
                })
                .collect(),
            cumulative_counts: h.cumulative_counts,
            by_backend: h
                .by_backend
                .into_iter()
                .map(|(k, v)| (format!("{:?}", k), v))
                .collect(),
        }
    }
}

impl CorpusHistoryResponse {
    /// Convert from ProofHistory
    pub fn from_proof_history(h: ProofHistory) -> Self {
        CorpusHistoryResponse {
            total_count: h.total_count,
            first_recorded: h.first_recorded.map(|dt| dt.to_rfc3339()),
            last_recorded: h.last_recorded.map(|dt| dt.to_rfc3339()),
            period_type: h.period_type.into(),
            periods: h
                .periods
                .into_iter()
                .map(|p| PeriodStatsResponse {
                    period: p.period,
                    start: p.start.to_rfc3339(),
                    count: p.count,
                    by_backend: p
                        .by_backend
                        .into_iter()
                        .map(|(k, v)| (format!("{:?}", k), v))
                        .collect(),
                })
                .collect(),
            cumulative_counts: h.cumulative_counts,
            by_backend: h
                .by_backend
                .into_iter()
                .map(|(k, v)| (format!("{:?}", k), v))
                .collect(),
        }
    }
}
