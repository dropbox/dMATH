//! Period comparison suggestions
//!
//! Provides types and functions for suggesting meaningful comparison periods
//! based on available data history.

use super::history::TimePeriod;
use chrono::{DateTime, Datelike, Duration, TimeZone, Utc};
use serde::{Deserialize, Serialize};

/// Type of suggested comparison period
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionType {
    /// Compare last week vs this week
    WeekOverWeek,
    /// Compare last month vs this month
    MonthOverMonth,
    /// Compare previous N days vs last N days
    RollingDays(u32),
    /// Compare previous quarter vs this quarter
    QuarterOverQuarter,
    /// Compare same period last year vs this year
    YearOverYear,
}

impl std::fmt::Display for SuggestionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SuggestionType::WeekOverWeek => write!(f, "Week over Week"),
            SuggestionType::MonthOverMonth => write!(f, "Month over Month"),
            SuggestionType::RollingDays(n) => write!(f, "Last {} days vs previous {} days", n, n),
            SuggestionType::QuarterOverQuarter => write!(f, "Quarter over Quarter"),
            SuggestionType::YearOverYear => write!(f, "Year over Year"),
        }
    }
}

/// A suggested comparison period with calculated date ranges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodSuggestion {
    /// Type of suggestion
    pub suggestion_type: SuggestionType,
    /// Human-readable description
    pub description: String,
    /// Start of baseline period
    pub baseline_start: DateTime<Utc>,
    /// End of baseline period
    pub baseline_end: DateTime<Utc>,
    /// Start of comparison period
    pub comparison_start: DateTime<Utc>,
    /// End of comparison period
    pub comparison_end: DateTime<Utc>,
    /// Recommended TimePeriod for grouping
    pub recommended_period: TimePeriod,
    /// Whether there is enough data for this comparison
    pub has_sufficient_data: bool,
    /// Expected number of periods in each range
    pub expected_period_count: usize,
}

impl PeriodSuggestion {
    /// Get baseline date range as tuple (start, end)
    pub fn baseline_range(&self) -> (DateTime<Utc>, DateTime<Utc>) {
        (self.baseline_start, self.baseline_end)
    }

    /// Get comparison date range as tuple (start, end)
    pub fn comparison_range(&self) -> (DateTime<Utc>, DateTime<Utc>) {
        (self.comparison_start, self.comparison_end)
    }

    /// Format baseline label for display
    pub fn baseline_label(&self) -> String {
        format!(
            "{} to {}",
            self.baseline_start.format("%Y-%m-%d"),
            self.baseline_end.format("%Y-%m-%d")
        )
    }

    /// Format comparison label for display
    pub fn comparison_label(&self) -> String {
        format!(
            "{} to {}",
            self.comparison_start.format("%Y-%m-%d"),
            self.comparison_end.format("%Y-%m-%d")
        )
    }

    /// Format dates as YYYY-MM-DD strings for CLI arguments
    pub fn cli_args(&self) -> PeriodSuggestionCliArgs {
        PeriodSuggestionCliArgs {
            baseline_from: self.baseline_start.format("%Y-%m-%d").to_string(),
            baseline_to: self.baseline_end.format("%Y-%m-%d").to_string(),
            compare_from: self.comparison_start.format("%Y-%m-%d").to_string(),
            compare_to: self.comparison_end.format("%Y-%m-%d").to_string(),
            period: self.recommended_period.to_string(),
        }
    }
}

/// CLI argument values for a period suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodSuggestionCliArgs {
    /// --baseline-from value
    pub baseline_from: String,
    /// --baseline-to value
    pub baseline_to: String,
    /// --compare-from value
    pub compare_from: String,
    /// --compare-to value
    pub compare_to: String,
    /// --period value
    pub period: String,
}

impl PeriodSuggestionCliArgs {
    /// Format as a CLI command snippet
    pub fn to_cli_command(&self) -> String {
        format!(
            "--baseline-from {} --baseline-to {} --compare-from {} --compare-to {} --period {}",
            self.baseline_from, self.baseline_to, self.compare_from, self.compare_to, self.period
        )
    }
}

/// Generate period suggestions based on available data range
pub fn suggest_comparison_periods(
    first_recorded: Option<DateTime<Utc>>,
    last_recorded: Option<DateTime<Utc>>,
    reference_date: Option<DateTime<Utc>>,
) -> Vec<PeriodSuggestion> {
    let now = reference_date.unwrap_or_else(Utc::now);
    let first = first_recorded.unwrap_or(now - Duration::days(365));
    let last = last_recorded.unwrap_or(now);

    // Calculate data span in days
    let data_span_days = (last - first).num_days();

    let mut suggestions = Vec::new();

    // Week over Week: compare last week vs this week
    // Need at least 14 days of data
    if data_span_days >= 14 {
        let this_week_end = now;
        let this_week_start = now - Duration::days(6);
        let last_week_end = this_week_start - Duration::days(1);
        let last_week_start = last_week_end - Duration::days(6);

        suggestions.push(PeriodSuggestion {
            suggestion_type: SuggestionType::WeekOverWeek,
            description: "Compare last week's activity to this week".to_string(),
            baseline_start: last_week_start,
            baseline_end: last_week_end,
            comparison_start: this_week_start,
            comparison_end: this_week_end,
            recommended_period: TimePeriod::Day,
            has_sufficient_data: first <= last_week_start,
            expected_period_count: 7,
        });
    }

    // Month over Month: compare last month vs this month
    // Need at least 60 days of data
    if data_span_days >= 60 {
        let this_month_start = Utc
            .with_ymd_and_hms(now.year(), now.month(), 1, 0, 0, 0)
            .unwrap();
        let this_month_end = now;

        // Last month
        let last_month_end = this_month_start - Duration::days(1);
        let last_month_start = Utc
            .with_ymd_and_hms(last_month_end.year(), last_month_end.month(), 1, 0, 0, 0)
            .unwrap();

        suggestions.push(PeriodSuggestion {
            suggestion_type: SuggestionType::MonthOverMonth,
            description: "Compare last month's activity to this month".to_string(),
            baseline_start: last_month_start,
            baseline_end: last_month_end,
            comparison_start: this_month_start,
            comparison_end: this_month_end,
            recommended_period: TimePeriod::Day,
            has_sufficient_data: first <= last_month_start,
            expected_period_count: 30,
        });
    }

    // Rolling 7 days: compare last 7 days vs previous 7 days
    if data_span_days >= 14 {
        let end = now;
        let mid = now - Duration::days(7);
        let start = mid - Duration::days(7);

        suggestions.push(PeriodSuggestion {
            suggestion_type: SuggestionType::RollingDays(7),
            description: "Compare previous 7 days to last 7 days".to_string(),
            baseline_start: start,
            baseline_end: mid - Duration::days(1),
            comparison_start: mid,
            comparison_end: end,
            recommended_period: TimePeriod::Day,
            has_sufficient_data: first <= start,
            expected_period_count: 7,
        });
    }

    // Rolling 30 days: compare last 30 days vs previous 30 days
    if data_span_days >= 60 {
        let end = now;
        let mid = now - Duration::days(30);
        let start = mid - Duration::days(30);

        suggestions.push(PeriodSuggestion {
            suggestion_type: SuggestionType::RollingDays(30),
            description: "Compare previous 30 days to last 30 days".to_string(),
            baseline_start: start,
            baseline_end: mid - Duration::days(1),
            comparison_start: mid,
            comparison_end: end,
            recommended_period: TimePeriod::Week,
            has_sufficient_data: first <= start,
            expected_period_count: 4,
        });
    }

    // Quarter over Quarter: compare last quarter vs this quarter
    // Need at least 180 days of data
    if data_span_days >= 180 {
        let current_quarter = (now.month() - 1) / 3;
        let this_quarter_start_month = current_quarter * 3 + 1;
        let this_quarter_start = Utc
            .with_ymd_and_hms(now.year(), this_quarter_start_month, 1, 0, 0, 0)
            .unwrap();
        let this_quarter_end = now;

        // Previous quarter
        let prev_quarter_end = this_quarter_start - Duration::days(1);
        let prev_quarter = if current_quarter == 0 {
            3
        } else {
            current_quarter - 1
        };
        let prev_year = if current_quarter == 0 {
            now.year() - 1
        } else {
            now.year()
        };
        let prev_quarter_start_month = prev_quarter * 3 + 1;
        let prev_quarter_start = Utc
            .with_ymd_and_hms(prev_year, prev_quarter_start_month, 1, 0, 0, 0)
            .unwrap();

        suggestions.push(PeriodSuggestion {
            suggestion_type: SuggestionType::QuarterOverQuarter,
            description: "Compare last quarter's activity to this quarter".to_string(),
            baseline_start: prev_quarter_start,
            baseline_end: prev_quarter_end,
            comparison_start: this_quarter_start,
            comparison_end: this_quarter_end,
            recommended_period: TimePeriod::Week,
            has_sufficient_data: first <= prev_quarter_start,
            expected_period_count: 13,
        });
    }

    // Year over Year: compare same period last year
    // Need at least 365 days of data
    if data_span_days >= 365 {
        let days_into_year = now.ordinal() as i64;
        let this_year_start = Utc.with_ymd_and_hms(now.year(), 1, 1, 0, 0, 0).unwrap();
        let this_year_end = now;

        let last_year_start = Utc.with_ymd_and_hms(now.year() - 1, 1, 1, 0, 0, 0).unwrap();
        let last_year_end = last_year_start + Duration::days(days_into_year - 1);

        suggestions.push(PeriodSuggestion {
            suggestion_type: SuggestionType::YearOverYear,
            description: format!(
                "Compare same period last year (Jan 1 - {}) to this year",
                now.format("%b %d")
            ),
            baseline_start: last_year_start,
            baseline_end: last_year_end,
            comparison_start: this_year_start,
            comparison_end: this_year_end,
            recommended_period: TimePeriod::Month,
            has_sufficient_data: first <= last_year_start,
            expected_period_count: now.month() as usize,
        });
    }

    suggestions
}

/// Format suggestions as a text summary
pub fn format_suggestions(suggestions: &[PeriodSuggestion]) -> String {
    if suggestions.is_empty() {
        return "No comparison suggestions available (insufficient data history).".to_string();
    }

    let mut lines = vec!["Available period comparisons:".to_string()];
    lines.push("=".repeat(50));

    for (i, suggestion) in suggestions.iter().enumerate() {
        let status = if suggestion.has_sufficient_data {
            "✓"
        } else {
            "⚠ insufficient data"
        };
        lines.push(format!(
            "\n{}. {} {}",
            i + 1,
            suggestion.suggestion_type,
            status
        ));
        lines.push(format!("   {}", suggestion.description));
        lines.push(format!(
            "   Baseline: {} to {}",
            suggestion.baseline_start.format("%Y-%m-%d"),
            suggestion.baseline_end.format("%Y-%m-%d")
        ));
        lines.push(format!(
            "   Compare:  {} to {}",
            suggestion.comparison_start.format("%Y-%m-%d"),
            suggestion.comparison_end.format("%Y-%m-%d")
        ));
        lines.push(format!("   Period:   {}", suggestion.recommended_period));

        // Show CLI command
        let args = suggestion.cli_args();
        lines.push(format!("   CLI:      {}", args.to_cli_command()));
    }

    lines.join("\n")
}
