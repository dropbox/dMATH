//! Time period types and corpus history

use chrono::{DateTime, Datelike, TimeZone, Utc};
use dashprove_backends::traits::BackendId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Time period granularity for corpus history
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TimePeriod {
    /// Group by day (YYYY-MM-DD)
    #[default]
    Day,
    /// Group by week (YYYY-Www)
    Week,
    /// Group by month (YYYY-MM)
    Month,
}

impl TimePeriod {
    /// Get the period key for a given timestamp
    pub fn key_for(&self, dt: DateTime<Utc>) -> String {
        match self {
            TimePeriod::Day => dt.format("%Y-%m-%d").to_string(),
            TimePeriod::Week => dt.format("%Y-W%W").to_string(),
            TimePeriod::Month => dt.format("%Y-%m").to_string(),
        }
    }

    /// Get the start timestamp for a period containing the given datetime
    pub fn start_for(&self, dt: DateTime<Utc>) -> DateTime<Utc> {
        match self {
            TimePeriod::Day => Utc
                .with_ymd_and_hms(dt.year(), dt.month(), dt.day(), 0, 0, 0)
                .unwrap(),
            TimePeriod::Week => {
                // Find the Monday of this week
                let weekday = dt.weekday().num_days_from_monday();
                let monday = dt - chrono::Duration::days(weekday as i64);
                Utc.with_ymd_and_hms(monday.year(), monday.month(), monday.day(), 0, 0, 0)
                    .unwrap()
            }
            TimePeriod::Month => Utc
                .with_ymd_and_hms(dt.year(), dt.month(), 1, 0, 0, 0)
                .unwrap(),
        }
    }
}

impl std::str::FromStr for TimePeriod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "day" | "daily" | "d" => Ok(TimePeriod::Day),
            "week" | "weekly" | "w" => Ok(TimePeriod::Week),
            "month" | "monthly" | "m" => Ok(TimePeriod::Month),
            _ => Err(format!(
                "Unknown time period: '{}'. Expected: day, week, month",
                s
            )),
        }
    }
}

impl std::fmt::Display for TimePeriod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimePeriod::Day => write!(f, "day"),
            TimePeriod::Week => write!(f, "week"),
            TimePeriod::Month => write!(f, "month"),
        }
    }
}

/// Statistics for a single time period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodStats {
    /// Period identifier (e.g., "2024-06-15", "2024-W24", "2024-06")
    pub period: String,
    /// Start timestamp of this period
    pub start: DateTime<Utc>,
    /// Total count in this period
    pub count: usize,
    /// Count by backend
    pub by_backend: HashMap<BackendId, usize>,
}

/// Corpus history over time
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CorpusHistory {
    /// Total counterexamples in corpus
    pub total_count: usize,
    /// First recorded timestamp
    pub first_recorded: Option<DateTime<Utc>>,
    /// Last recorded timestamp
    pub last_recorded: Option<DateTime<Utc>>,
    /// Period granularity
    pub period_type: TimePeriod,
    /// Stats per period (chronologically sorted)
    pub periods: Vec<PeriodStats>,
    /// Cumulative totals per period
    pub cumulative_counts: Vec<usize>,
    /// Total count by backend
    pub by_backend: HashMap<BackendId, usize>,
    /// Total count by property name
    pub by_property: HashMap<String, usize>,
    /// Count by cluster label
    pub by_cluster: HashMap<String, usize>,
}

impl CorpusHistory {
    /// Generate a simple text summary
    pub fn summary(&self) -> String {
        if self.total_count == 0 {
            return "Corpus is empty.".to_string();
        }

        let mut lines = vec![format!("Total counterexamples: {}", self.total_count)];

        if let (Some(first), Some(last)) = (&self.first_recorded, &self.last_recorded) {
            lines.push(format!(
                "Time range: {} to {}",
                first.format("%Y-%m-%d %H:%M"),
                last.format("%Y-%m-%d %H:%M")
            ));
        }

        lines.push("\nBy backend:".to_string());
        let mut backend_list: Vec<_> = self.by_backend.iter().collect();
        backend_list.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        for (backend, count) in backend_list {
            lines.push(format!("  {:?}: {}", backend, count));
        }

        if !self.by_cluster.is_empty() {
            lines.push("\nBy cluster:".to_string());
            let mut cluster_list: Vec<_> = self.by_cluster.iter().collect();
            cluster_list.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
            for (cluster, count) in cluster_list.iter().take(10) {
                lines.push(format!("  {}: {}", cluster, count));
            }
            if cluster_list.len() > 10 {
                lines.push(format!(
                    "  ... and {} more clusters",
                    cluster_list.len() - 10
                ));
            }
        }

        lines.push(format!(
            "\nPer {} ({} periods):",
            self.period_type,
            self.periods.len()
        ));
        for (i, period) in self.periods.iter().enumerate() {
            let cumulative = self.cumulative_counts.get(i).copied().unwrap_or(0);
            lines.push(format!(
                "  {}: {} (+{} cumulative: {})",
                period.period, period.count, period.count, cumulative
            ));
        }

        lines.join("\n")
    }

    /// Generate HTML visualization with charts
    pub fn to_html(&self, title: &str) -> String {
        use crate::templates::{
            render_history_html, ExtraStatCard, HistoryChartConfig, PeriodData,
        };

        let config = HistoryChartConfig {
            title: title.to_string(),
            entity_name: "Counterexamples".to_string(),
            total: self.total_count,
            period_count: self.periods.len(),
            backend_count: self.by_backend.len(),
            period_type: self.period_type.to_string(),
            cumulative_color: "rgba(220, 38, 38, 1)".to_string(),
            download_prefix: "corpus".to_string(),
        };

        let periods: Vec<PeriodData> = self
            .periods
            .iter()
            .enumerate()
            .map(|(i, p)| PeriodData {
                period: p.period.clone(),
                count: p.count,
                cumulative: self.cumulative_counts.get(i).copied().unwrap_or(0),
                backends: p
                    .by_backend
                    .iter()
                    .map(|(b, c)| (format!("{:?}", b), *c))
                    .collect(),
            })
            .collect();

        let backends: HashMap<String, usize> = self
            .by_backend
            .iter()
            .map(|(b, c)| (format!("{:?}", b), *c))
            .collect();

        let extra_stats = vec![ExtraStatCard {
            value: self.by_cluster.len(),
            label: "Clusters".to_string(),
        }];

        render_history_html(&config, &periods, &backends, &extra_stats)
    }
}
