//! History comparison types and analysis
//!
//! Provides types and methods for comparing corpus history across time periods.

use super::history::{CorpusHistory, PeriodStats};
use super::suggestions::PeriodSuggestion;
use chrono::{DateTime, Utc};
use dashprove_backends::traits::BackendId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Stats for a comparison period (baseline or comparison)
pub(crate) struct ComparisonPeriodData<'a> {
    pub count: usize,
    pub by_backend: &'a HashMap<BackendId, usize>,
    pub period_count: usize,
}

/// Comparison between two time periods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryComparison {
    /// Label for the baseline period (e.g., "2024-01-01 to 2024-01-31")
    pub baseline_label: String,
    /// Label for the comparison period
    pub comparison_label: String,
    /// Total count in baseline period
    pub baseline_count: usize,
    /// Total count in comparison period
    pub comparison_count: usize,
    /// Absolute change (comparison - baseline)
    pub count_delta: i64,
    /// Percentage change ((comparison - baseline) / baseline * 100)
    pub count_percent_change: Option<f64>,
    /// Backend count changes (backend -> delta)
    pub backend_deltas: HashMap<BackendId, i64>,
    /// New backends that appeared in comparison period
    pub new_backends: Vec<BackendId>,
    /// Backends that disappeared in comparison period
    pub removed_backends: Vec<BackendId>,
    /// Number of periods in baseline
    pub baseline_period_count: usize,
    /// Number of periods in comparison
    pub comparison_period_count: usize,
    /// Average count per period in baseline
    pub baseline_avg_per_period: f64,
    /// Average count per period in comparison
    pub comparison_avg_per_period: f64,
    /// Growth rate per period (ratio of comparison avg to baseline avg)
    pub growth_rate_per_period: Option<f64>,
    /// Compound growth rate (useful for projections)
    pub compound_growth_rate: Option<f64>,
}

impl HistoryComparison {
    /// Create a comparison from period stats
    pub(crate) fn from_period_stats(
        baseline: ComparisonPeriodData<'_>,
        comparison: ComparisonPeriodData<'_>,
        baseline_label: &str,
        comparison_label: &str,
    ) -> Self {
        let count_delta = comparison.count as i64 - baseline.count as i64;
        let count_percent_change = if baseline.count > 0 {
            Some((count_delta as f64 / baseline.count as f64) * 100.0)
        } else if comparison.count > 0 {
            Some(f64::INFINITY)
        } else {
            None
        };

        // Calculate backend deltas
        let mut backend_deltas: HashMap<BackendId, i64> = HashMap::new();
        let mut new_backends = Vec::new();
        let mut removed_backends = Vec::new();

        // All backends from both periods
        let all_backends: std::collections::HashSet<_> = baseline
            .by_backend
            .keys()
            .chain(comparison.by_backend.keys())
            .collect();

        for backend in all_backends {
            let b_count = baseline.by_backend.get(backend).copied().unwrap_or(0);
            let c_count = comparison.by_backend.get(backend).copied().unwrap_or(0);
            let delta = c_count as i64 - b_count as i64;

            if delta != 0 {
                backend_deltas.insert(*backend, delta);
            }

            if b_count == 0 && c_count > 0 {
                new_backends.push(*backend);
            } else if b_count > 0 && c_count == 0 {
                removed_backends.push(*backend);
            }
        }

        // Calculate averages per period
        let baseline_avg = if baseline.period_count == 0 {
            0.0
        } else {
            baseline.count as f64 / baseline.period_count as f64
        };

        let comparison_avg = if comparison.period_count == 0 {
            0.0
        } else {
            comparison.count as f64 / comparison.period_count as f64
        };

        // Calculate growth rate per period (ratio of averages)
        let growth_rate_per_period = if baseline_avg > 0.0 && comparison_avg > 0.0 {
            Some(comparison_avg / baseline_avg)
        } else if baseline_avg == 0.0 && comparison_avg > 0.0 {
            Some(f64::INFINITY)
        } else {
            None
        };

        // Calculate compound growth rate
        // If we have period counts, calculate the average growth rate per period
        // Formula: (comparison_avg / baseline_avg)^(1/n) - 1 where n is the number of periods between
        let compound_growth_rate = if baseline_avg > 0.0
            && comparison_avg > 0.0
            && baseline.period_count > 0
            && comparison.period_count > 0
        {
            let ratio = comparison_avg / baseline_avg;
            // Assume 1 period between baseline and comparison for compound rate calculation
            // This gives a per-period growth multiplier
            let growth = ratio - 1.0;
            Some(growth)
        } else {
            None
        };

        Self {
            baseline_label: baseline_label.to_string(),
            comparison_label: comparison_label.to_string(),
            baseline_count: baseline.count,
            comparison_count: comparison.count,
            count_delta,
            count_percent_change,
            backend_deltas,
            new_backends,
            removed_backends,
            baseline_period_count: baseline.period_count,
            comparison_period_count: comparison.period_count,
            baseline_avg_per_period: baseline_avg,
            comparison_avg_per_period: comparison_avg,
            growth_rate_per_period,
            compound_growth_rate,
        }
    }

    /// Create a comparison from two CorpusHistory instances (counterexamples)
    pub fn from_corpus_histories(
        baseline: &CorpusHistory,
        comparison: &CorpusHistory,
        baseline_label: &str,
        comparison_label: &str,
    ) -> Self {
        Self::from_period_stats(
            ComparisonPeriodData {
                count: baseline.total_count,
                by_backend: &baseline.by_backend,
                period_count: baseline.periods.len(),
            },
            ComparisonPeriodData {
                count: comparison.total_count,
                by_backend: &comparison.by_backend,
                period_count: comparison.periods.len(),
            },
            baseline_label,
            comparison_label,
        )
    }

    /// Create a comparison from two ProofHistory instances
    pub fn from_proof_histories(
        baseline: &crate::corpus::ProofHistory,
        comparison: &crate::corpus::ProofHistory,
        baseline_label: &str,
        comparison_label: &str,
    ) -> Self {
        Self::from_period_stats(
            ComparisonPeriodData {
                count: baseline.total_count,
                by_backend: &baseline.by_backend,
                period_count: baseline.periods.len(),
            },
            ComparisonPeriodData {
                count: comparison.total_count,
                by_backend: &comparison.by_backend,
                period_count: comparison.periods.len(),
            },
            baseline_label,
            comparison_label,
        )
    }

    /// Generate a text summary of the comparison
    pub fn summary(&self) -> String {
        let mut lines = vec![format!(
            "Comparison: {} vs {}",
            self.baseline_label, self.comparison_label
        )];
        lines.push("=".repeat(50));

        // Total counts
        lines.push(format!(
            "\nTotal count: {} -> {} ({:+})",
            self.baseline_count, self.comparison_count, self.count_delta
        ));

        if let Some(pct) = self.count_percent_change {
            if pct.is_infinite() {
                lines.push("  Change: +∞% (from zero baseline)".to_string());
            } else {
                lines.push(format!("  Change: {:+.1}%", pct));
            }
        }

        // Period averages
        lines.push(format!(
            "\nPeriods: {} -> {}",
            self.baseline_period_count, self.comparison_period_count
        ));
        lines.push(format!(
            "Average per period: {:.1} -> {:.1}",
            self.baseline_avg_per_period, self.comparison_avg_per_period
        ));

        // Growth rate
        if let Some(rate) = self.growth_rate_per_period {
            if rate.is_infinite() {
                lines.push("Growth rate: +∞ (from zero baseline)".to_string());
            } else {
                lines.push(format!("Growth rate: {:.2}x per period", rate));
            }
        }

        if let Some(rate) = self.compound_growth_rate {
            let pct = rate * 100.0;
            lines.push(format!("Growth rate change: {:+.1}% per period", pct));

            // Add projections
            if self.comparison_avg_per_period > 0.0 {
                let proj_3 = self.project_count(3);
                let proj_6 = self.project_count(6);
                let proj_12 = self.project_count(12);
                lines.push("\nProjections (based on current growth rate):".to_string());
                lines.push(format!(
                    "  Next 3 periods: ~{} entries",
                    proj_3.map_or("N/A".to_string(), |v| format!("{:.0}", v))
                ));
                lines.push(format!(
                    "  Next 6 periods: ~{} entries",
                    proj_6.map_or("N/A".to_string(), |v| format!("{:.0}", v))
                ));
                lines.push(format!(
                    "  Next 12 periods: ~{} entries",
                    proj_12.map_or("N/A".to_string(), |v| format!("{:.0}", v))
                ));
            }
        }

        // Backend changes
        if !self.backend_deltas.is_empty() {
            lines.push("\nBy backend:".to_string());
            let mut deltas: Vec<_> = self.backend_deltas.iter().collect();
            deltas.sort_by_key(|(_, delta)| std::cmp::Reverse(**delta));
            for (backend, delta) in deltas {
                lines.push(format!("  {:?}: {:+}", backend, delta));
            }
        }

        if !self.new_backends.is_empty() {
            lines.push(format!("\nNew backends: {:?}", self.new_backends));
        }
        if !self.removed_backends.is_empty() {
            lines.push(format!("Removed backends: {:?}", self.removed_backends));
        }

        lines.join("\n")
    }

    fn render_html_with_extras(
        &self,
        extra_css: &str,
        extra_sections: &str,
        extra_scripts: &str,
    ) -> String {
        let percent_change_str = match self.count_percent_change {
            Some(pct) if pct.is_infinite() => "+∞% (from zero)".to_string(),
            Some(pct) => format!("{:+.1}%", pct),
            None => "N/A".to_string(),
        };

        let delta_class = if self.count_delta > 0 {
            "positive"
        } else if self.count_delta < 0 {
            "negative"
        } else {
            "neutral"
        };

        // Build backend comparison data
        let mut all_backends: Vec<&BackendId> = self
            .backend_deltas
            .keys()
            .chain(self.new_backends.iter())
            .chain(self.removed_backends.iter())
            .collect();
        all_backends.sort_by_key(|b| format!("{:?}", b));
        all_backends.dedup();

        let backend_labels: Vec<String> = all_backends.iter().map(|b| format!("{:?}", b)).collect();
        let backend_deltas: Vec<i64> = all_backends
            .iter()
            .map(|b| self.backend_deltas.get(*b).copied().unwrap_or(0))
            .collect();

        let backend_labels_json = serde_json::to_string(&backend_labels).unwrap_or_default();
        let backend_deltas_json = serde_json::to_string(&backend_deltas).unwrap_or_default();

        // New/removed backends lists
        let new_backends_html = if self.new_backends.is_empty() {
            String::new()
        } else {
            let backends: Vec<String> = self
                .new_backends
                .iter()
                .map(|b| format!("{:?}", b))
                .collect();
            format!(
                r#"<div class="alert new">New backends: {}</div>"#,
                backends.join(", ")
            )
        };

        let removed_backends_html = if self.removed_backends.is_empty() {
            String::new()
        } else {
            let backends: Vec<String> = self
                .removed_backends
                .iter()
                .map(|b| format!("{:?}", b))
                .collect();
            format!(
                r#"<div class="alert removed">Removed backends: {}</div>"#,
                backends.join(", ")
            )
        };

        // Projections data
        let growth_rate_str = match self.growth_rate_per_period {
            Some(rate) if rate.is_infinite() => "+∞ (from zero)".to_string(),
            Some(rate) => format!("{:.2}x", rate),
            None => "N/A".to_string(),
        };

        let compound_rate_str = match self.compound_growth_rate {
            Some(rate) => format!("{:+.1}%", rate * 100.0),
            None => "N/A".to_string(),
        };

        // Generate projections for the chart
        let proj_3 = self.project_count(3).unwrap_or(0.0);
        let proj_6 = self.project_count(6).unwrap_or(0.0);
        let proj_12 = self.project_count(12).unwrap_or(0.0);
        let has_projections =
            self.compound_growth_rate.is_some() && self.comparison_avg_per_period > 0.0;

        let projections_section = if has_projections {
            format!(
                r#"
        <div class="chart-container">
            <h3>Growth Rate & Projections</h3>
            <div class="growth-stats">
                <div class="growth-stat">
                    <span class="growth-label">Growth Rate</span>
                    <span class="growth-value">{growth_rate}</span>
                </div>
                <div class="growth-stat">
                    <span class="growth-label">Per-Period Change</span>
                    <span class="growth-value">{compound_rate}</span>
                </div>
            </div>
            <canvas id="projectionsChart"></canvas>
        </div>"#,
                growth_rate = growth_rate_str,
                compound_rate = compound_rate_str,
            )
        } else {
            String::new()
        };

        let projections_chart_js = if has_projections {
            format!(
                r#"
        // Projections line chart
        new Chart(document.getElementById('projectionsChart'), {{{{
            type: 'line',
            data: {{{{
                labels: ['Current', '+3 periods', '+6 periods', '+12 periods'],
                datasets: [{{{{
                    label: 'Projected Cumulative',
                    data: [{current}, {proj3:.0}, {proj6:.0}, {proj12:.0}],
                    borderColor: 'rgba(37, 99, 235, 1)',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    fill: true,
                    tension: 0.3
                }}}}]
            }}}},
            options: {{{{
                responsive: true,
                plugins: {{{{
                    legend: {{{{ display: false }}}},
                    tooltip: {{{{
                        callbacks: {{{{
                            label: function(context) {{{{
                                return 'Projected: ~' + Math.round(context.raw);
                            }}}}
                        }}}}
                    }}}}
                }}}},
                scales: {{{{
                    y: {{{{
                        beginAtZero: true,
                        title: {{{{ display: true, text: 'Cumulative count' }}}}
                    }}}}
                }}}}
            }}}}
        }});"#,
                current = self.comparison_count,
                proj3 = self.comparison_count as f64 + proj_3,
                proj6 = self.comparison_count as f64 + proj_6,
                proj12 = self.comparison_count as f64 + proj_12,
            )
        } else {
            String::new()
        };

        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Period Comparison</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #333; margin-bottom: 10px; }}
        .subtitle {{ color: #666; margin-bottom: 30px; }}
        .comparison-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
        .stat-label {{ color: #666; font-size: 0.9em; margin-bottom: 8px; }}
        .stat-value {{ font-size: 2em; font-weight: bold; }}
        .baseline {{ color: #6b7280; }}
        .comparison {{ color: #2563eb; }}
        .delta {{ }}
        .delta.positive {{ color: #059669; }}
        .delta.negative {{ color: #dc2626; }}
        .delta.neutral {{ color: #6b7280; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .alert {{ padding: 12px 16px; border-radius: 6px; margin-bottom: 15px; }}
        .alert.new {{ background: #d1fae5; color: #065f46; }}
        .alert.removed {{ background: #fee2e2; color: #991b1b; }}
        .averages {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
        .avg-card {{ background: white; padding: 15px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .avg-card h4 {{ margin: 0 0 8px 0; color: #666; font-size: 0.9em; }}
        .avg-card .value {{ font-size: 1.5em; font-weight: bold; }}
        .download-btn {{
            position: fixed; bottom: 20px; right: 20px;
            background: #2563eb; color: white; border: none;
            padding: 12px 24px; border-radius: 8px; cursor: pointer;
            font-size: 14px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .download-btn:hover {{ background: #1d4ed8; }}
        .growth-stats {{ display: flex; gap: 40px; margin-bottom: 20px; }}
        .growth-stat {{ display: flex; flex-direction: column; }}
        .growth-label {{ color: #666; font-size: 0.85em; margin-bottom: 4px; }}
        .growth-value {{ font-size: 1.5em; font-weight: bold; color: #2563eb; }}
        {extra_css}
    </style>
</head>
<body>
    <div class="container">
        <h1>Period Comparison</h1>
        <p class="subtitle">{baseline_label} vs {comparison_label}</p>

        {new_backends_html}
        {removed_backends_html}

        <div class="comparison-grid">
            <div class="stat-card">
                <div class="stat-label">Baseline</div>
                <div class="stat-value baseline">{baseline_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Comparison</div>
                <div class="stat-value comparison">{comparison_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Change</div>
                <div class="stat-value delta {delta_class}">{count_delta:+} ({percent_change})</div>
            </div>
        </div>

        <div class="averages">
            <div class="avg-card">
                <h4>Baseline ({baseline_periods} periods)</h4>
                <div class="value">{baseline_avg:.1} per period</div>
            </div>
            <div class="avg-card">
                <h4>Comparison ({comparison_periods} periods)</h4>
                <div class="value">{comparison_avg:.1} per period</div>
            </div>
        </div>

        <div class="chart-container">
            <h3>Change by Backend</h3>
            <canvas id="backendChart"></canvas>
        </div>

        {projections_section}
        {extra_sections}
    </div>

    <button class="download-btn" onclick="downloadData()">Download JSON</button>

    <script>
        const backendLabels = {backend_labels_json};
        const backendDeltas = {backend_deltas_json};

        // Backend comparison bar chart
        new Chart(document.getElementById('backendChart'), {{
            type: 'bar',
            data: {{
                labels: backendLabels,
                datasets: [{{
                    label: 'Change',
                    data: backendDeltas,
                    backgroundColor: backendDeltas.map(d =>
                        d > 0 ? 'rgba(5, 150, 105, 0.7)' :
                        d < 0 ? 'rgba(220, 38, 38, 0.7)' :
                        'rgba(107, 114, 128, 0.7)'
                    ),
                    borderColor: backendDeltas.map(d =>
                        d > 0 ? 'rgba(5, 150, 105, 1)' :
                        d < 0 ? 'rgba(220, 38, 38, 1)' :
                        'rgba(107, 114, 128, 1)'
                    ),
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{ display: true, text: 'Change (comparison - baseline)' }}
                    }}
                }}
            }}
        }});

        {projections_chart_js}
        {extra_scripts}

        function downloadData() {{
            const data = {{
                baseline_label: "{baseline_label}",
                comparison_label: "{comparison_label}",
                baseline_count: {baseline_count},
                comparison_count: {comparison_count},
                count_delta: {count_delta},
                percent_change: "{percent_change}",
                baseline_periods: {baseline_periods},
                comparison_periods: {comparison_periods},
                baseline_avg_per_period: {baseline_avg},
                comparison_avg_per_period: {comparison_avg},
                backend_changes: Object.fromEntries(backendLabels.map((l, i) => [l, backendDeltas[i]])),
                generated: new Date().toISOString()
            }};
            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'period_comparison.json';
            a.click();
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>"#,
            baseline_label = self.baseline_label,
            comparison_label = self.comparison_label,
            baseline_count = self.baseline_count,
            comparison_count = self.comparison_count,
            count_delta = self.count_delta,
            percent_change = percent_change_str,
            delta_class = delta_class,
            baseline_periods = self.baseline_period_count,
            comparison_periods = self.comparison_period_count,
            baseline_avg = self.baseline_avg_per_period,
            comparison_avg = self.comparison_avg_per_period,
            backend_labels_json = backend_labels_json,
            backend_deltas_json = backend_deltas_json,
            new_backends_html = new_backends_html,
            removed_backends_html = removed_backends_html,
            projections_section = projections_section,
            projections_chart_js = projections_chart_js,
            extra_css = extra_css,
            extra_sections = extra_sections,
            extra_scripts = extra_scripts,
        )
    }

    /// Generate an HTML visualization of the comparison
    pub fn to_html(&self) -> String {
        self.render_html_with_extras("", "", "")
    }

    /// Generate an HTML visualization with side-by-side period charts
    ///
    /// This method takes the original history data to display period-by-period
    /// comparisons in addition to the aggregate statistics.
    pub fn to_html_with_periods(
        &self,
        baseline_periods: &[PeriodStats],
        comparison_periods: &[PeriodStats],
    ) -> String {
        let baseline_labels: Vec<String> =
            baseline_periods.iter().map(|p| p.period.clone()).collect();
        let baseline_counts: Vec<usize> = baseline_periods.iter().map(|p| p.count).collect();
        let comparison_labels: Vec<String> = comparison_periods
            .iter()
            .map(|p| p.period.clone())
            .collect();
        let comparison_counts: Vec<usize> = comparison_periods.iter().map(|p| p.count).collect();

        let baseline_labels_json = serde_json::to_string(&baseline_labels).unwrap_or_default();
        let baseline_counts_json = serde_json::to_string(&baseline_counts).unwrap_or_default();
        let comparison_labels_json = serde_json::to_string(&comparison_labels).unwrap_or_default();
        let comparison_counts_json = serde_json::to_string(&comparison_counts).unwrap_or_default();

        // Build cumulative data
        let mut baseline_cumulative: Vec<usize> = Vec::new();
        let mut running_sum = 0;
        for count in &baseline_counts {
            running_sum += count;
            baseline_cumulative.push(running_sum);
        }

        let mut comparison_cumulative: Vec<usize> = Vec::new();
        let mut running_sum = 0;
        for count in &comparison_counts {
            running_sum += count;
            comparison_cumulative.push(running_sum);
        }

        let baseline_cumulative_json =
            serde_json::to_string(&baseline_cumulative).unwrap_or_default();
        let comparison_cumulative_json =
            serde_json::to_string(&comparison_cumulative).unwrap_or_default();

        let periods_section = format!(
            r#"
        <div class="chart-container">
            <h3>Side-by-Side Period Comparison</h3>
            <div class="period-charts">
                <div class="period-chart">
                    <h4>Baseline: {baseline_label}</h4>
                    <canvas id="baselineChart"></canvas>
                </div>
                <div class="period-chart">
                    <h4>Comparison: {comparison_label}</h4>
                    <canvas id="comparisonChart"></canvas>
                </div>
            </div>
        </div>

        <div class="chart-container">
            <h3>Cumulative Growth Comparison</h3>
            <canvas id="cumulativeComparisonChart"></canvas>
        </div>"#,
            baseline_label = self.baseline_label,
            comparison_label = self.comparison_label,
        );

        let periods_chart_js = format!(
            r#"
        // Baseline period chart
        new Chart(document.getElementById('baselineChart'), {{
            type: 'bar',
            data: {{
                labels: {baseline_labels},
                datasets: [{{
                    label: 'Count',
                    data: {baseline_counts},
                    backgroundColor: 'rgba(107, 114, 128, 0.7)',
                    borderColor: 'rgba(107, 114, 128, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{ display: true, text: 'Count per period' }}
                    }}
                }}
            }}
        }});

        // Comparison period chart
        new Chart(document.getElementById('comparisonChart'), {{
            type: 'bar',
            data: {{
                labels: {comparison_labels},
                datasets: [{{
                    label: 'Count',
                    data: {comparison_counts},
                    backgroundColor: 'rgba(37, 99, 235, 0.7)',
                    borderColor: 'rgba(37, 99, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{ display: true, text: 'Count per period' }}
                    }}
                }}
            }}
        }});

        // Cumulative comparison chart
        new Chart(document.getElementById('cumulativeComparisonChart'), {{
            type: 'line',
            data: {{
                labels: {comparison_labels},
                datasets: [
                    {{
                        label: 'Baseline cumulative',
                        data: {baseline_cumulative},
                        borderColor: 'rgba(107, 114, 128, 1)',
                        backgroundColor: 'rgba(107, 114, 128, 0.1)',
                        fill: true,
                        tension: 0.3
                    }},
                    {{
                        label: 'Comparison cumulative',
                        data: {comparison_cumulative},
                        borderColor: 'rgba(37, 99, 235, 1)',
                        backgroundColor: 'rgba(37, 99, 235, 0.1)',
                        fill: true,
                        tension: 0.3
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: true }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{ display: true, text: 'Cumulative count' }}
                    }}
                }}
            }}
        }});"#,
            baseline_labels = baseline_labels_json,
            baseline_counts = baseline_counts_json,
            comparison_labels = comparison_labels_json,
            comparison_counts = comparison_counts_json,
            baseline_cumulative = baseline_cumulative_json,
            comparison_cumulative = comparison_cumulative_json,
        );

        let additional_css = r#"
        .period-charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .period-chart { min-height: 300px; }
        .period-chart h4 { margin: 0 0 15px 0; color: #666; }"#;

        self.render_html_with_extras(additional_css, &periods_section, &periods_chart_js)
    }

    /// Generate HTML with suggested comparison periods appended
    pub fn to_html_with_suggestions(
        &self,
        suggestions: &[PeriodSuggestion],
        corpus_name: &str,
        data_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    ) -> String {
        let suggestion_css = r#"
        .suggestion-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .suggestion-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 12px; }
        .suggestion-card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; background: #f9fafb; display: flex; flex-direction: column; gap: 10px; }
        .suggestion-header { display: flex; justify-content: space-between; align-items: center; }
        .suggestion-badge { background: #e0ecff; color: #1d4ed8; padding: 4px 8px; border-radius: 999px; font-size: 0.85em; font-weight: 600; }
        .suggestion-status { font-size: 0.85em; font-weight: 600; }
        .suggestion-status.ready { color: #065f46; }
        .suggestion-status.missing { color: #92400e; }
        .suggestion-desc { margin: 0; color: #374151; }
        .suggestion-dates { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.95em; color: #4b5563; }
        .suggestion-label { display: block; color: #6b7280; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.02em; }
        .suggestion-value { font-weight: 600; color: #111827; }
        .suggestion-meta { display: flex; gap: 16px; align-items: baseline; font-size: 0.95em; color: #4b5563; }
        .cli-snippet { background: #111827; color: #e5e7eb; padding: 8px 10px; border-radius: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.9em; overflow-x: auto; }
        .suggestion-empty { padding: 12px 16px; background: #fff7ed; color: #92400e; border: 1px solid #fed7aa; border-radius: 8px; }
        .suggestion-subtitle { color: #6b7280; margin: 4px 0 0 0; }"#;

        let range_note = data_range.map(|(start, end)| {
            format!(
                "Recorded {} from {} to {}.",
                corpus_name,
                start.format("%Y-%m-%d"),
                end.format("%Y-%m-%d")
            )
        });

        let suggestion_cards = if suggestions.is_empty() {
            String::from(
                r#"<div class="suggestion-empty">No comparison suggestions available (insufficient data history).</div>"#,
            )
        } else {
            suggestions
                .iter()
                .map(|suggestion| {
                    let status_class = if suggestion.has_sufficient_data {
                        "ready"
                    } else {
                        "missing"
                    };
                    let status_label = if suggestion.has_sufficient_data {
                        "Ready"
                    } else {
                        "Needs more data"
                    };
                    let status_hint = if suggestion.has_sufficient_data {
                        "Comparison spans available history".to_string()
                    } else {
                        format!(
                            "Requires data back to {}",
                            suggestion.baseline_start.format("%Y-%m-%d")
                        )
                    };
                    let cli_args = suggestion.cli_args();
                    let cli_command = format!(
                        "dashprove corpus compare --corpus {} {}",
                        corpus_name,
                        cli_args.to_cli_command()
                    );

                    format!(
                        r#"<div class="suggestion-card">
    <div class="suggestion-header">
        <span class="suggestion-badge">{suggestion_type}</span>
        <span class="suggestion-status {status_class}">{status_label}</span>
    </div>
    <p class="suggestion-desc">{description}</p>
    <div class="suggestion-dates">
        <div>
            <span class="suggestion-label">Baseline</span>
            <span class="suggestion-value">{baseline_start} to {baseline_end}</span>
        </div>
        <div>
            <span class="suggestion-label">Comparison</span>
            <span class="suggestion-value">{comparison_start} to {comparison_end}</span>
        </div>
    </div>
    <div class="suggestion-meta">
        <div><span class="suggestion-label">Period</span> <span class="suggestion-value">{period}</span></div>
        <div><span class="suggestion-label">Expected periods</span> <span class="suggestion-value">{expected_periods}</span></div>
        <div><span class="suggestion-label">Data</span> <span class="suggestion-value">{status_hint}</span></div>
    </div>
    <div class="cli-snippet">{cli_command}</div>
</div>"#,
                        suggestion_type = suggestion.suggestion_type,
                        status_class = status_class,
                        status_label = status_label,
                        description = suggestion.description,
                        baseline_start = suggestion.baseline_start.format("%Y-%m-%d"),
                        baseline_end = suggestion.baseline_end.format("%Y-%m-%d"),
                        comparison_start = suggestion.comparison_start.format("%Y-%m-%d"),
                        comparison_end = suggestion.comparison_end.format("%Y-%m-%d"),
                        period = suggestion.recommended_period,
                        expected_periods = suggestion.expected_period_count,
                        status_hint = status_hint,
                        cli_command = cli_command,
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        let suggestions_section = format!(
            r#"<div class="suggestion-container">
    <h3>Suggested Comparisons</h3>
    {}
    <div class="suggestion-grid">
        {cards}
    </div>
</div>"#,
            range_note
                .map(|note| format!(r#"<p class="suggestion-subtitle">{}</p>"#, note))
                .unwrap_or_default(),
            cards = suggestion_cards
        );

        self.render_html_with_extras(suggestion_css, &suggestions_section, "")
    }

    /// Project the total count for a given number of future periods
    ///
    /// Uses the compound growth rate to estimate future values.
    /// Returns None if growth rate cannot be calculated.
    pub fn project_count(&self, periods_ahead: usize) -> Option<f64> {
        let growth_rate = self.compound_growth_rate?;

        // Use comparison avg as base and apply growth rate
        let base = self.comparison_avg_per_period;
        if base <= 0.0 {
            return None;
        }

        // Project each period with compound growth
        let mut total = 0.0;
        for i in 1..=periods_ahead {
            let period_count = base * (1.0 + growth_rate).powi(i as i32);
            total += period_count;
        }

        Some(total)
    }

    /// Project the average count per period for a given number of future periods
    ///
    /// Returns the expected average per period at `periods_ahead` periods into the future.
    pub fn project_avg_at(&self, periods_ahead: usize) -> Option<f64> {
        let growth_rate = self.compound_growth_rate?;

        let base = self.comparison_avg_per_period;
        if base <= 0.0 {
            return None;
        }

        // Compound growth formula: base * (1 + rate)^n
        Some(base * (1.0 + growth_rate).powi(periods_ahead as i32))
    }

    /// Get projections for common future periods (3, 6, 12)
    ///
    /// Returns a struct with pre-calculated projections for easy access.
    pub fn projections(&self) -> GrowthProjections {
        GrowthProjections {
            periods_3_total: self.project_count(3),
            periods_6_total: self.project_count(6),
            periods_12_total: self.project_count(12),
            periods_3_avg: self.project_avg_at(3),
            periods_6_avg: self.project_avg_at(6),
            periods_12_avg: self.project_avg_at(12),
            growth_rate_per_period: self.growth_rate_per_period,
            compound_growth_rate: self.compound_growth_rate,
        }
    }
}

/// Pre-calculated growth projections
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GrowthProjections {
    /// Projected total count over next 3 periods
    pub periods_3_total: Option<f64>,
    /// Projected total count over next 6 periods
    pub periods_6_total: Option<f64>,
    /// Projected total count over next 12 periods
    pub periods_12_total: Option<f64>,
    /// Projected average per period at 3 periods ahead
    pub periods_3_avg: Option<f64>,
    /// Projected average per period at 6 periods ahead
    pub periods_6_avg: Option<f64>,
    /// Projected average per period at 12 periods ahead
    pub periods_12_avg: Option<f64>,
    /// Growth rate multiplier per period
    pub growth_rate_per_period: Option<f64>,
    /// Compound growth rate (delta)
    pub compound_growth_rate: Option<f64>,
}

impl GrowthProjections {
    /// Format projections as a text summary
    pub fn summary(&self) -> String {
        let mut lines = vec![];

        if let Some(rate) = self.growth_rate_per_period {
            if rate.is_infinite() {
                lines.push("Growth rate: +∞ (from zero baseline)".to_string());
            } else {
                lines.push(format!("Growth rate: {:.2}x per period", rate));
            }
        }

        if let Some(rate) = self.compound_growth_rate {
            let pct = rate * 100.0;
            lines.push(format!("Per-period growth: {:+.1}%", pct));
        }

        if self.periods_3_total.is_some() {
            lines.push("\nProjected totals:".to_string());
            if let Some(v) = self.periods_3_total {
                lines.push(format!("  Next 3 periods: ~{:.0}", v));
            }
            if let Some(v) = self.periods_6_total {
                lines.push(format!("  Next 6 periods: ~{:.0}", v));
            }
            if let Some(v) = self.periods_12_total {
                lines.push(format!("  Next 12 periods: ~{:.0}", v));
            }
        }

        if self.periods_3_avg.is_some() {
            lines.push("\nProjected averages per period:".to_string());
            if let Some(v) = self.periods_3_avg {
                lines.push(format!("  At 3 periods: ~{:.1}", v));
            }
            if let Some(v) = self.periods_6_avg {
                lines.push(format!("  At 6 periods: ~{:.1}", v));
            }
            if let Some(v) = self.periods_12_avg {
                lines.push(format!("  At 12 periods: ~{:.1}", v));
            }
        }

        if lines.is_empty() {
            "No growth projections available (insufficient data)".to_string()
        } else {
            lines.join("\n")
        }
    }
}
