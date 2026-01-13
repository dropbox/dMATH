//! Core types for proof corpus

use crate::counterexamples::{PeriodStats, TimePeriod};
use crate::similarity::PropertyFeatures;
use chrono::{DateTime, Utc};
use dashprove_backends::traits::BackendId;
use dashprove_usl::ast::Property;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// Unique identifier for a stored proof
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProofId(pub String);

impl fmt::Display for ProofId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl ProofId {
    /// Generate a new unique ID based on property name and timestamp
    pub fn generate(property_name: &str) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        ProofId(format!("{}_{}", property_name, timestamp))
    }
}

/// A stored proof entry in the corpus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofEntry {
    /// Unique identifier
    pub id: ProofId,
    /// The property that was proven
    pub property: Property,
    /// Which backend generated this proof
    pub backend: BackendId,
    /// Tactics that were used
    pub tactics: Vec<String>,
    /// Time to complete verification
    pub time_taken: Duration,
    /// Raw proof output from the backend
    pub proof_output: Option<String>,
    /// Extracted features for similarity search
    pub features: PropertyFeatures,
    /// When this proof was recorded
    #[serde(default = "default_recorded_at")]
    pub recorded_at: DateTime<Utc>,
    /// Optional vector embedding for similarity search
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<crate::embedder::Embedding>,
}

pub fn default_recorded_at() -> DateTime<Utc> {
    Utc::now()
}

/// History of proof corpus over time
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProofHistory {
    /// Total proofs in corpus
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
    /// Total count by tactic name
    pub by_tactic: HashMap<String, usize>,
}

impl ProofHistory {
    /// Generate a simple text summary
    pub fn summary(&self) -> String {
        if self.total_count == 0 {
            return "Proof corpus is empty.".to_string();
        }

        let mut lines = vec![format!("Total proofs: {}", self.total_count)];

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

        if !self.by_tactic.is_empty() {
            lines.push("\nTop tactics:".to_string());
            let mut tactic_list: Vec<_> = self.by_tactic.iter().collect();
            tactic_list.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
            for (tactic, count) in tactic_list.iter().take(10) {
                lines.push(format!("  {}: {}", tactic, count));
            }
            if tactic_list.len() > 10 {
                lines.push(format!("  ... and {} more tactics", tactic_list.len() - 10));
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
            render_history_html_with_tactics, ExtraStatCard, HistoryChartConfig, PeriodData,
            TacticChartData,
        };

        let config = HistoryChartConfig {
            title: title.to_string(),
            entity_name: "Proofs".to_string(),
            total: self.total_count,
            period_count: self.periods.len(),
            backend_count: self.by_backend.len(),
            period_type: self.period_type.to_string(),
            cumulative_color: "rgba(234, 88, 12, 1)".to_string(),
            download_prefix: "proof_corpus".to_string(),
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

        let backends: std::collections::HashMap<String, usize> = self
            .by_backend
            .iter()
            .map(|(b, c)| (format!("{:?}", b), *c))
            .collect();

        let extra_stats = vec![ExtraStatCard {
            value: self.by_property.len(),
            label: "Properties".to_string(),
        }];

        // Include tactic breakdown chart if tactics exist
        let tactics = if self.by_tactic.is_empty() {
            None
        } else {
            Some(TacticChartData::new(self.by_tactic.clone()))
        };

        render_history_html_with_tactics(
            &config,
            &periods,
            &backends,
            &extra_stats,
            tactics.as_ref(),
        )
    }
}
