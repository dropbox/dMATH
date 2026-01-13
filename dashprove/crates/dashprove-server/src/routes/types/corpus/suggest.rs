use dashprove_learning::PeriodSuggestion;
use serde::{Deserialize, Serialize};

use crate::routes::types::{default_corpus_type, CorpusType, TimePeriodParam};

// ============ Corpus Suggest-Compare Types ============

/// Query parameters for suggest-compare endpoint
#[derive(Debug, Deserialize)]
pub struct CorpusSuggestQuery {
    /// Which corpus to suggest comparisons for
    #[serde(default = "default_corpus_type")]
    pub corpus: CorpusType,
}

/// Response for period suggestions
#[derive(Debug, Serialize, Deserialize)]
pub struct CorpusSuggestResponse {
    /// Available period suggestions
    pub suggestions: Vec<PeriodSuggestionResponse>,
}

/// A single period suggestion
#[derive(Debug, Serialize, Deserialize)]
pub struct PeriodSuggestionResponse {
    /// Type of suggestion
    pub suggestion_type: String,
    /// Human-readable description
    pub description: String,
    /// Baseline start date (YYYY-MM-DD)
    pub baseline_start: String,
    /// Baseline end date (YYYY-MM-DD)
    pub baseline_end: String,
    /// Comparison start date (YYYY-MM-DD)
    pub comparison_start: String,
    /// Comparison end date (YYYY-MM-DD)
    pub comparison_end: String,
    /// Recommended period granularity
    pub recommended_period: TimePeriodParam,
    /// Whether there's enough data
    pub has_sufficient_data: bool,
    /// Expected number of periods
    pub expected_period_count: usize,
    /// API query string for compare endpoint
    pub api_query: String,
}

impl From<PeriodSuggestion> for PeriodSuggestionResponse {
    fn from(s: PeriodSuggestion) -> Self {
        let args = s.cli_args();
        let api_query = format!(
            "baseline_from={}&baseline_to={}&compare_from={}&compare_to={}",
            args.baseline_from, args.baseline_to, args.compare_from, args.compare_to
        );
        PeriodSuggestionResponse {
            suggestion_type: s.suggestion_type.to_string(),
            description: s.description,
            baseline_start: args.baseline_from,
            baseline_end: args.baseline_to,
            comparison_start: args.compare_from,
            comparison_end: args.compare_to,
            recommended_period: s.recommended_period.into(),
            has_sufficient_data: s.has_sufficient_data,
            expected_period_count: s.expected_period_count,
            api_query,
        }
    }
}
