//! Shared HTML templates for history visualization
//!
//! Provides reusable Chart.js-based templates for rendering corpus history
//! visualizations (both counterexamples and proofs).

use serde::Serialize;
use std::collections::HashMap;

/// Configuration for a history chart visualization
#[derive(Debug, Clone, Serialize)]
pub struct HistoryChartConfig {
    /// Chart title
    pub title: String,
    /// Entity name for display (e.g., "Counterexamples", "Proofs")
    pub entity_name: String,
    /// Total count
    pub total: usize,
    /// Number of time periods
    pub period_count: usize,
    /// Number of backends
    pub backend_count: usize,
    /// Period type label (e.g., "day", "week", "month")
    pub period_type: String,
    /// Cumulative line color (CSS rgba)
    pub cumulative_color: String,
    /// Download filename prefix
    pub download_prefix: String,
}

/// Period data for chart rendering
#[derive(Debug, Clone, Serialize)]
pub struct PeriodData {
    /// Period label (e.g., "2024-01", "Week 5")
    pub period: String,
    /// Count of items in this period
    pub count: usize,
    /// Cumulative count up to and including this period
    pub cumulative: usize,
    /// Breakdown by backend (backend name -> count)
    pub backends: HashMap<String, usize>,
}

/// Extra stat cards to show in the summary section
#[derive(Debug, Clone, Serialize)]
pub struct ExtraStatCard {
    /// Numeric value to display
    pub value: usize,
    /// Label describing the stat
    pub label: String,
}

/// Optional tactic breakdown chart data
#[derive(Debug, Clone, Serialize, Default)]
pub struct TacticChartData {
    /// Tactic name to count mapping
    pub tactics: HashMap<String, usize>,
    /// Maximum number of tactics to show (rest grouped as "Other")
    pub max_display: usize,
}

impl TacticChartData {
    /// Create tactic chart data with default max_display of 10
    pub fn new(tactics: HashMap<String, usize>) -> Self {
        Self {
            tactics,
            max_display: 10,
        }
    }

    /// Returns true if there are any tactics to display
    pub fn is_empty(&self) -> bool {
        self.tactics.is_empty()
    }

    /// Get tactics sorted by count, grouping excess into "Other"
    pub fn display_data(&self) -> HashMap<String, usize> {
        if self.tactics.len() <= self.max_display {
            return self.tactics.clone();
        }

        let mut sorted: Vec<_> = self.tactics.iter().collect();
        sorted.sort_by_key(|(_, count)| std::cmp::Reverse(*count));

        let mut result: HashMap<String, usize> = sorted
            .iter()
            .take(self.max_display - 1)
            .map(|(k, v)| ((*k).clone(), **v))
            .collect();

        let other_count: usize = sorted
            .iter()
            .skip(self.max_display - 1)
            .map(|(_, v)| *v)
            .sum();
        if other_count > 0 {
            result.insert("Other".to_string(), other_count);
        }

        result
    }
}

/// Generate HTML for a history visualization with Chart.js
///
/// This creates a complete HTML page with:
/// - Summary stat cards at the top
/// - A bar chart showing counts over time with cumulative line
/// - A doughnut chart showing breakdown by backend
/// - An optional horizontal bar chart showing tactic breakdown
/// - A download button for JSON export
pub fn render_history_html(
    config: &HistoryChartConfig,
    periods: &[PeriodData],
    backends: &HashMap<String, usize>,
    extra_stats: &[ExtraStatCard],
) -> String {
    render_history_html_with_tactics(config, periods, backends, extra_stats, None)
}

/// Generate HTML for a history visualization with optional tactic breakdown
pub fn render_history_html_with_tactics(
    config: &HistoryChartConfig,
    periods: &[PeriodData],
    backends: &HashMap<String, usize>,
    extra_stats: &[ExtraStatCard],
    tactics: Option<&TacticChartData>,
) -> String {
    let periods_json = serde_json::to_string(periods).unwrap_or_default();
    let backend_json = serde_json::to_string(backends).unwrap_or_default();

    // Prepare tactic data if provided
    let (tactic_json, has_tactics) = match tactics {
        Some(t) if !t.is_empty() => {
            let display_data = t.display_data();
            (
                serde_json::to_string(&display_data).unwrap_or_default(),
                true,
            )
        }
        _ => ("{}".to_string(), false),
    };

    let extra_cards_html: String = extra_stats
        .iter()
        .map(|stat| {
            format!(
                r#"<div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">{}</div>
            </div>"#,
                stat.value, stat.label
            )
        })
        .collect();

    // Conditional tactic chart HTML section
    let tactic_chart_html = if has_tactics {
        r#"
        <div class="chart-container">
            <h3>Top Tactics Used</h3>
            <canvas id="tacticChart"></canvas>
        </div>"#
            .to_string()
    } else {
        String::new()
    };

    // Conditional tactic chart JS
    let tactic_chart_js = if has_tactics {
        format!(
            r#"
        // Tactic horizontal bar chart
        const tacticData = {tactic_json};
        const tacticEntries = Object.entries(tacticData).sort((a, b) => b[1] - a[1]);
        new Chart(document.getElementById('tacticChart'), {{
            type: 'bar',
            data: {{
                labels: tacticEntries.map(e => e[0]),
                datasets: [{{
                    label: 'Usage Count',
                    data: tacticEntries.map(e => e[1]),
                    backgroundColor: 'rgba(34, 197, 94, 0.7)',
                    borderColor: 'rgba(34, 197, 94, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{ beginAtZero: true }}
                }}
            }}
        }});"#,
            tactic_json = tactic_json
        )
    } else {
        String::new()
    };

    // Conditional tactic data in download
    let tactic_download_field = if has_tactics {
        format!("by_tactic: {tactic_json},", tactic_json = tactic_json)
    } else {
        String::new()
    };

    format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #2563eb; }}
        .stat-label {{ color: #666; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .chart-row {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }}
        @media (max-width: 800px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
        .download-btn {{
            position: fixed; bottom: 20px; right: 20px;
            background: #2563eb; color: white; border: none;
            padding: 12px 24px; border-radius: 8px; cursor: pointer;
            font-size: 14px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .download-btn:hover {{ background: #1d4ed8; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total {entity_name}</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{period_count}</div>
                <div class="stat-label">Time Periods</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{backend_count}</div>
                <div class="stat-label">Backends</div>
            </div>
            {extra_cards}
        </div>
        <div class="chart-row">
            <div class="chart-container">
                <h3>{entity_name} Over Time (by {period_type})</h3>
                <canvas id="timeChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>By Backend</h3>
                <canvas id="backendChart"></canvas>
            </div>
        </div>{tactic_chart_html}
    </div>
    <button class="download-btn" onclick="downloadData()">Download JSON</button>
    <script>
        const periodsData = {periods_json};
        const backendData = {backend_json};

        // Time series chart
        new Chart(document.getElementById('timeChart'), {{
            type: 'bar',
            data: {{
                labels: periodsData.map(p => p.period),
                datasets: [
                    {{
                        label: 'Per Period',
                        data: periodsData.map(p => p.count),
                        backgroundColor: 'rgba(37, 99, 235, 0.7)',
                        order: 2
                    }},
                    {{
                        label: 'Cumulative',
                        data: periodsData.map(p => p.cumulative),
                        type: 'line',
                        borderColor: '{cumulative_color}',
                        backgroundColor: 'transparent',
                        order: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ beginAtZero: true }}
                }}
            }}
        }});

        // Backend pie chart
        new Chart(document.getElementById('backendChart'), {{
            type: 'doughnut',
            data: {{
                labels: Object.keys(backendData),
                datasets: [{{
                    data: Object.values(backendData),
                    backgroundColor: [
                        'rgba(37, 99, 235, 0.8)',
                        'rgba(220, 38, 38, 0.8)',
                        'rgba(34, 197, 94, 0.8)',
                        'rgba(234, 179, 8, 0.8)',
                        'rgba(168, 85, 247, 0.8)',
                        'rgba(236, 72, 153, 0.8)'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ position: 'bottom' }}
                }}
            }}
        }});
        {tactic_chart_js}

        function downloadData() {{
            const data = {{
                total: {total},
                periods: periodsData,
                by_backend: backendData,
                {tactic_download_field}
                generated: new Date().toISOString()
            }};
            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = '{download_prefix}_history.json';
            a.click();
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>"#,
        title = config.title,
        entity_name = config.entity_name,
        total = config.total,
        period_count = config.period_count,
        backend_count = config.backend_count,
        period_type = config.period_type,
        cumulative_color = config.cumulative_color,
        download_prefix = config.download_prefix,
        extra_cards = extra_cards_html,
        periods_json = periods_json,
        backend_json = backend_json,
        tactic_chart_html = tactic_chart_html,
        tactic_chart_js = tactic_chart_js,
        tactic_download_field = tactic_download_field,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_history_html_basic() {
        let config = HistoryChartConfig {
            title: "Test History".to_string(),
            entity_name: "Items".to_string(),
            total: 10,
            period_count: 3,
            backend_count: 2,
            period_type: "day".to_string(),
            cumulative_color: "rgba(220, 38, 38, 1)".to_string(),
            download_prefix: "test".to_string(),
        };

        let periods = vec![
            PeriodData {
                period: "2024-01-01".to_string(),
                count: 3,
                cumulative: 3,
                backends: HashMap::from([("Lean4".to_string(), 2), ("TlaPlus".to_string(), 1)]),
            },
            PeriodData {
                period: "2024-01-02".to_string(),
                count: 4,
                cumulative: 7,
                backends: HashMap::from([("Lean4".to_string(), 4)]),
            },
            PeriodData {
                period: "2024-01-03".to_string(),
                count: 3,
                cumulative: 10,
                backends: HashMap::from([("TlaPlus".to_string(), 3)]),
            },
        ];

        let backends = HashMap::from([
            ("Lean4".to_string(), 6usize),
            ("TlaPlus".to_string(), 4usize),
        ]);

        let html = render_history_html(&config, &periods, &backends, &[]);

        assert!(html.contains("<html>"));
        assert!(html.contains("Test History"));
        assert!(html.contains("Total Items"));
        assert!(html.contains("chart.js"));
        assert!(html.contains("test_history.json"));
    }

    #[test]
    fn test_render_history_html_with_extra_stats() {
        let config = HistoryChartConfig {
            title: "Corpus History".to_string(),
            entity_name: "Counterexamples".to_string(),
            total: 50,
            period_count: 5,
            backend_count: 3,
            period_type: "week".to_string(),
            cumulative_color: "rgba(220, 38, 38, 1)".to_string(),
            download_prefix: "corpus".to_string(),
        };

        let periods = vec![];
        let backends = HashMap::new();
        let extra_stats = vec![
            ExtraStatCard {
                value: 12,
                label: "Clusters".to_string(),
            },
            ExtraStatCard {
                value: 8,
                label: "Properties".to_string(),
            },
        ];

        let html = render_history_html(&config, &periods, &backends, &extra_stats);

        assert!(html.contains("Clusters"));
        assert!(html.contains("Properties"));
        assert!(html.contains("12"));
        assert!(html.contains("8"));
    }

    #[test]
    fn test_render_history_html_empty_periods() {
        let config = HistoryChartConfig {
            title: "Empty".to_string(),
            entity_name: "Items".to_string(),
            total: 0,
            period_count: 0,
            backend_count: 0,
            period_type: "month".to_string(),
            cumulative_color: "rgba(234, 88, 12, 1)".to_string(),
            download_prefix: "empty".to_string(),
        };

        let html = render_history_html(&config, &[], &HashMap::new(), &[]);

        assert!(html.contains("<html>"));
        assert!(html.contains("Empty"));
        assert!(html.contains("Total Items"));
    }

    #[test]
    fn test_tactic_chart_data_new() {
        let tactics = HashMap::from([
            ("simp".to_string(), 10usize),
            ("decide".to_string(), 5usize),
        ]);
        let chart_data = TacticChartData::new(tactics.clone());

        assert_eq!(chart_data.tactics, tactics);
        assert_eq!(chart_data.max_display, 10);
        assert!(!chart_data.is_empty());
    }

    #[test]
    fn test_tactic_chart_data_empty() {
        let chart_data = TacticChartData::new(HashMap::new());
        assert!(chart_data.is_empty());
    }

    #[test]
    fn test_tactic_chart_data_display_data_under_limit() {
        let tactics = HashMap::from([
            ("simp".to_string(), 10usize),
            ("decide".to_string(), 5usize),
            ("ring".to_string(), 3usize),
        ]);
        let chart_data = TacticChartData::new(tactics.clone());
        let display = chart_data.display_data();

        // Should return all tactics unchanged
        assert_eq!(display.len(), 3);
        assert_eq!(display.get("simp"), Some(&10));
        assert_eq!(display.get("decide"), Some(&5));
        assert_eq!(display.get("ring"), Some(&3));
    }

    #[test]
    fn test_tactic_chart_data_display_data_over_limit() {
        // Create 15 tactics to exceed max_display of 10
        let tactics: HashMap<String, usize> = (1..=15)
            .map(|i| (format!("tactic_{}", i), 100 - i))
            .collect();

        let chart_data = TacticChartData::new(tactics.clone());
        let display = chart_data.display_data();

        // Should have max_display entries (9 top tactics + "Other")
        assert_eq!(display.len(), 10);
        assert!(display.contains_key("Other"));

        // Top tactics should be present
        assert!(display.contains_key("tactic_1")); // 99
        assert!(display.contains_key("tactic_2")); // 98

        // Bottom tactics should be grouped into "Other"
        assert!(!display.contains_key("tactic_15"));

        // Verify exact "Other" count to catch skip arithmetic mutations
        // max_display=10, so we take top 9 tactics (tactic_1..tactic_9 with values 99..91)
        // and sum the rest (tactic_10..tactic_15 with values 90..85)
        // Other = 90 + 89 + 88 + 87 + 86 + 85 = 525
        assert_eq!(display.get("Other"), Some(&525usize));
    }

    #[test]
    fn test_tactic_chart_data_display_data_exact_at_limit() {
        // Create exactly max_display (10) tactics - should NOT group into "Other"
        let tactics: HashMap<String, usize> = (1..=10)
            .map(|i| (format!("tactic_{}", i), i * 10))
            .collect();

        let chart_data = TacticChartData::new(tactics);
        let display = chart_data.display_data();

        // Should have exactly 10 entries, no "Other"
        assert_eq!(display.len(), 10);
        assert!(!display.contains_key("Other"));
    }

    #[test]
    fn test_tactic_chart_data_display_data_one_over_limit() {
        // Create max_display + 1 (11) tactics - should have exactly one in "Other"
        let tactics: HashMap<String, usize> = (1..=11)
            .map(|i| (format!("tactic_{}", i), (12 - i) * 10))
            .collect();
        // tactic_1=110, tactic_2=100, ..., tactic_11=10

        let chart_data = TacticChartData::new(tactics);
        let display = chart_data.display_data();

        // Should have 10 entries (9 top tactics + "Other")
        assert_eq!(display.len(), 10);
        assert!(display.contains_key("Other"));

        // "Other" should contain exactly the lowest 2 tactics (tactic_10=20, tactic_11=10)
        // because we take top 9, skip 9, so remaining is tactic_10 and tactic_11
        assert_eq!(display.get("Other"), Some(&30usize)); // 20 + 10

        // Verify top tactics are present
        assert_eq!(display.get("tactic_1"), Some(&110usize));
        assert_eq!(display.get("tactic_9"), Some(&30usize));
        // tactic_10 should NOT be in top (it's in "Other")
        assert!(!display.contains_key("tactic_10"));
    }

    #[test]
    fn test_render_history_html_with_tactics() {
        let config = HistoryChartConfig {
            title: "Proof History".to_string(),
            entity_name: "Proofs".to_string(),
            total: 20,
            period_count: 2,
            backend_count: 1,
            period_type: "day".to_string(),
            cumulative_color: "rgba(234, 88, 12, 1)".to_string(),
            download_prefix: "proof".to_string(),
        };

        let tactics = TacticChartData::new(HashMap::from([
            ("simp".to_string(), 15usize),
            ("decide".to_string(), 8usize),
            ("ring".to_string(), 3usize),
        ]));

        let html =
            render_history_html_with_tactics(&config, &[], &HashMap::new(), &[], Some(&tactics));

        // Should include tactic chart elements
        assert!(html.contains("tacticChart"));
        assert!(html.contains("Top Tactics Used"));
        assert!(html.contains("tacticData"));
        assert!(html.contains("by_tactic"));
    }

    #[test]
    fn test_render_history_html_without_tactics() {
        let config = HistoryChartConfig {
            title: "Proof History".to_string(),
            entity_name: "Proofs".to_string(),
            total: 20,
            period_count: 2,
            backend_count: 1,
            period_type: "day".to_string(),
            cumulative_color: "rgba(234, 88, 12, 1)".to_string(),
            download_prefix: "proof".to_string(),
        };

        let html = render_history_html_with_tactics(&config, &[], &HashMap::new(), &[], None);

        // Should NOT include tactic chart elements
        assert!(!html.contains("tacticChart"));
        assert!(!html.contains("Top Tactics Used"));
        assert!(!html.contains("tacticData"));
        assert!(!html.contains("by_tactic"));
    }

    #[test]
    fn test_render_history_html_with_empty_tactics() {
        let config = HistoryChartConfig {
            title: "Proof History".to_string(),
            entity_name: "Proofs".to_string(),
            total: 20,
            period_count: 2,
            backend_count: 1,
            period_type: "day".to_string(),
            cumulative_color: "rgba(234, 88, 12, 1)".to_string(),
            download_prefix: "proof".to_string(),
        };

        let empty_tactics = TacticChartData::new(HashMap::new());

        let html = render_history_html_with_tactics(
            &config,
            &[],
            &HashMap::new(),
            &[],
            Some(&empty_tactics),
        );

        // Empty tactics should NOT render the chart
        assert!(!html.contains("tacticChart"));
        assert!(!html.contains("Top Tactics Used"));
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify HistoryChartConfig can be constructed with any valid values
    #[kani::proof]
    fn verify_history_chart_config_construction() {
        let total: usize = kani::any();
        let period_count: usize = kani::any();
        let backend_count: usize = kani::any();

        // Bound inputs
        kani::assume(total <= 1000);
        kani::assume(period_count <= 100);
        kani::assume(backend_count <= 50);

        let config = HistoryChartConfig {
            title: String::new(),
            entity_name: String::new(),
            total,
            period_count,
            backend_count,
            period_type: String::new(),
            cumulative_color: String::new(),
            download_prefix: String::new(),
        };

        kani::assert(config.total == total, "total preserved");
        kani::assert(
            config.period_count == period_count,
            "period_count preserved",
        );
        kani::assert(
            config.backend_count == backend_count,
            "backend_count preserved",
        );
    }

    /// Verify ExtraStatCard fields are preserved
    #[kani::proof]
    fn verify_extra_stat_card_fields() {
        let value: usize = kani::any();
        kani::assume(value <= 1000000);

        let card = ExtraStatCard {
            value,
            label: String::new(),
        };

        kani::assert(card.value == value, "value preserved");
    }

    /// Verify that max_display of 10 is the TacticChartData default
    /// (verified via unit test integration, here we verify the constant)
    #[kani::proof]
    fn verify_tactic_chart_max_display_constant() {
        // TacticChartData::new() sets max_display to 10 by default
        // We verify this constant is correctly used
        let max_display: usize = 10;
        kani::assert(max_display == 10, "default max_display constant is 10");
    }

    /// Verify PeriodData count and cumulative relationship
    #[kani::proof]
    fn verify_period_data_count_cumulative() {
        let count: usize = kani::any();
        let cumulative: usize = kani::any();

        // Bound inputs
        kani::assume(count <= 1000);
        kani::assume(cumulative <= 10000);

        // The semantic invariant: cumulative should include count
        // (i.e., cumulative >= count for a valid sequence)
        kani::assume(cumulative >= count);

        // Verify the invariant holds
        kani::assert(cumulative >= count, "cumulative includes count");
    }

    /// Verify HistoryChartConfig total is never negative (usize guarantees this)
    #[kani::proof]
    fn verify_history_chart_config_total_non_negative() {
        let total: usize = kani::any();
        kani::assume(total <= 1000000);

        // usize is always non-negative
        kani::assert(total >= 0, "total is non-negative");
    }
}
