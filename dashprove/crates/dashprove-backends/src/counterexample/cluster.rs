//! Counterexample clustering
//!
//! This module provides tools for grouping similar counterexamples:
//! - `CounterexampleCluster`: A cluster of similar counterexamples
//! - `CounterexampleClusters`: Collection of clusters
//! - `CounterexampleStats`: Statistics about a counterexample

use super::types::StructuredCounterexample;
use crate::traits::{html_download_buttons, DOWNLOAD_BUTTON_CSS};
use crate::util::mermaid_escape;

// ============================================================================
// Counterexample Clustering
// ============================================================================

/// Represents a cluster of similar counterexamples
#[derive(Debug, Clone)]
pub struct CounterexampleCluster {
    /// Representative counterexample for this cluster
    pub representative: StructuredCounterexample,
    /// All counterexamples in this cluster (including representative)
    pub members: Vec<StructuredCounterexample>,
    /// Cluster label/summary describing common failure pattern
    pub label: String,
    /// Similarity threshold used for clustering
    pub similarity_threshold: f64,
}

impl CounterexampleCluster {
    /// Create a new cluster with a single counterexample
    pub fn new(cx: StructuredCounterexample, threshold: f64) -> Self {
        let label = Self::generate_label(&cx);
        Self {
            representative: cx.clone(),
            members: vec![cx],
            label,
            similarity_threshold: threshold,
        }
    }

    /// Try to add a counterexample to this cluster
    /// Returns true if added, false if not similar enough
    pub fn try_add(&mut self, cx: StructuredCounterexample) -> bool {
        let similarity = self.similarity(&cx);
        if similarity >= self.similarity_threshold {
            self.members.push(cx);
            true
        } else {
            false
        }
    }

    /// Calculate similarity between a counterexample and this cluster's representative
    /// Returns value from 0.0 (no similarity) to 1.0 (identical)
    pub fn similarity(&self, cx: &StructuredCounterexample) -> f64 {
        let mut score = 0.0;
        let mut factors = 0.0;

        // Factor 1: Same failed checks (high weight)
        let checks_a: Vec<_> = self
            .representative
            .failed_checks
            .iter()
            .map(|c| &c.check_id)
            .collect();
        let checks_b: Vec<_> = cx.failed_checks.iter().map(|c| &c.check_id).collect();
        let check_sim = self.ref_set_similarity(&checks_a, &checks_b);
        score += check_sim * 3.0;
        factors += 3.0;

        // Factor 2: Similar trace length
        let len_a = self.representative.trace.len() as f64;
        let len_b = cx.trace.len() as f64;
        if len_a > 0.0 || len_b > 0.0 {
            let len_sim = len_a.min(len_b) / len_a.max(len_b);
            score += len_sim;
            factors += 1.0;
        }

        // Factor 3: Similar witness variables
        let witness_keys_a: std::collections::HashSet<_> =
            self.representative.witness.keys().collect();
        let witness_keys_b: std::collections::HashSet<_> = cx.witness.keys().collect();
        if !witness_keys_a.is_empty() || !witness_keys_b.is_empty() {
            let witness_sim = self.key_set_similarity(&witness_keys_a, &witness_keys_b);
            score += witness_sim * 2.0;
            factors += 2.0;
        }

        // Factor 4: Similar trace actions
        let actions_a: Vec<_> = self
            .representative
            .trace
            .iter()
            .filter_map(|s| s.action.as_ref())
            .collect();
        let actions_b: Vec<_> = cx.trace.iter().filter_map(|s| s.action.as_ref()).collect();
        if !actions_a.is_empty() || !actions_b.is_empty() {
            let action_sim = self.sequence_similarity(&actions_a, &actions_b);
            score += action_sim * 2.0;
            factors += 2.0;
        }

        if factors > 0.0 {
            score / factors
        } else {
            0.0
        }
    }

    fn ref_set_similarity<T: std::hash::Hash + Eq>(&self, a: &[T], b: &[T]) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        let set_a: std::collections::HashSet<_> = a.iter().collect();
        let set_b: std::collections::HashSet<_> = b.iter().collect();
        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();
        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn key_set_similarity<T: std::hash::Hash + Eq>(
        &self,
        a: &std::collections::HashSet<T>,
        b: &std::collections::HashSet<T>,
    ) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        let intersection = a.intersection(b).count();
        let union = a.union(b).count();
        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn sequence_similarity<T: PartialEq>(&self, a: &[T], b: &[T]) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        // Simple prefix matching for sequences
        let common = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
        let max_len = a.len().max(b.len());
        if max_len == 0 {
            1.0
        } else {
            common as f64 / max_len as f64
        }
    }

    fn generate_label(cx: &StructuredCounterexample) -> String {
        if !cx.failed_checks.is_empty() {
            let check_strs: Vec<_> = cx
                .failed_checks
                .iter()
                .map(|c| c.description.as_str())
                .collect();
            format!("Fails: {}", check_strs.join(", "))
        } else if !cx.trace.is_empty() {
            let first_action = cx
                .trace
                .first()
                .and_then(|s| s.action.as_ref())
                .map(|a| a.as_str())
                .unwrap_or("unknown");
            format!("Trace starting with: {}", first_action)
        } else {
            "Unknown pattern".to_string()
        }
    }

    /// Get the size of this cluster
    pub fn size(&self) -> usize {
        self.members.len()
    }
}

/// Collection of counterexample clusters
#[derive(Debug, Clone)]
pub struct CounterexampleClusters {
    /// List of clusters
    pub clusters: Vec<CounterexampleCluster>,
    /// Similarity threshold used for clustering (0.0 to 1.0)
    pub similarity_threshold: f64,
}

impl CounterexampleClusters {
    /// Create clustering from a list of counterexamples
    /// Uses greedy clustering with the given similarity threshold (0.0 to 1.0)
    pub fn from_counterexamples(
        counterexamples: Vec<StructuredCounterexample>,
        similarity_threshold: f64,
    ) -> Self {
        let mut clusters: Vec<CounterexampleCluster> = Vec::new();

        for cx in counterexamples {
            let mut added = false;
            for cluster in &mut clusters {
                if cluster.try_add(cx.clone()) {
                    added = true;
                    break;
                }
            }
            if !added {
                clusters.push(CounterexampleCluster::new(cx, similarity_threshold));
            }
        }

        // Sort clusters by size (largest first)
        clusters.sort_by_key(|c| std::cmp::Reverse(c.size()));

        Self {
            clusters,
            similarity_threshold,
        }
    }

    /// Get the number of clusters
    pub fn num_clusters(&self) -> usize {
        self.clusters.len()
    }

    /// Get total number of counterexamples across all clusters
    pub fn total_counterexamples(&self) -> usize {
        self.clusters.iter().map(|c| c.size()).sum()
    }

    /// Get the largest cluster
    pub fn largest_cluster(&self) -> Option<&CounterexampleCluster> {
        self.clusters.first()
    }

    /// Summary of the clustering
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Clustered {} counterexamples into {} clusters (threshold: {:.2})\n",
            self.total_counterexamples(),
            self.num_clusters(),
            self.similarity_threshold
        ));

        for (i, cluster) in self.clusters.iter().enumerate() {
            out.push_str(&format!(
                "\nCluster {} ({} members): {}\n",
                i + 1,
                cluster.size(),
                cluster.label
            ));
        }

        out
    }
}

impl std::fmt::Display for CounterexampleClusters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

impl CounterexampleClusters {
    /// Export clusters as a Mermaid pie chart showing cluster distribution
    ///
    /// Shows the relative size of each cluster and their labels.
    /// Useful for embedding in markdown documentation.
    pub fn to_mermaid(&self) -> String {
        let mut mermaid = String::new();

        if self.clusters.is_empty() {
            mermaid.push_str("flowchart TB\n");
            mermaid.push_str("    empty[\"No clusters\"]\n");
            return mermaid;
        }

        // Use pie chart for cluster distribution
        mermaid.push_str("pie showData\n");
        mermaid.push_str(&format!(
            "    title Counterexample Clusters ({} total, threshold: {:.2})\n",
            self.total_counterexamples(),
            self.similarity_threshold
        ));

        for (i, cluster) in self.clusters.iter().enumerate() {
            let label = if cluster.label.len() > 30 {
                format!("{}...", &cluster.label[..27])
            } else {
                cluster.label.clone()
            };
            // Escape quotes in label
            let safe_label = label.replace('"', "'");
            mermaid.push_str(&format!(
                "    \"Cluster {} ({})\" : {}\n",
                i + 1,
                safe_label,
                cluster.size()
            ));
        }

        mermaid
    }

    /// Export clusters as a Mermaid flowchart showing cluster structure
    ///
    /// Shows each cluster as a node with member count and sample failures.
    /// Useful for understanding cluster relationships.
    pub fn to_mermaid_flowchart(&self) -> String {
        let mut mermaid = String::new();

        mermaid.push_str("flowchart TB\n");
        mermaid.push_str(&format!(
            "    %% {} counterexamples in {} clusters (threshold: {:.2})\n",
            self.total_counterexamples(),
            self.num_clusters(),
            self.similarity_threshold
        ));

        if self.clusters.is_empty() {
            mermaid.push_str("    empty[\"No clusters\"]\n");
            return mermaid;
        }

        // Root node
        mermaid.push_str(&format!(
            "    root{{\"All Counterexamples<br/>({} total)\"}}\n",
            self.total_counterexamples()
        ));

        // Each cluster as a node
        for (i, cluster) in self.clusters.iter().enumerate() {
            let cluster_id = format!("c{}", i);
            let label = if cluster.label.len() > 25 {
                format!("{}...", &cluster.label[..22])
            } else {
                cluster.label.clone()
            };
            let safe_label = mermaid_escape(&label);

            // Get sample failed checks
            let checks: Vec<_> = cluster
                .representative
                .failed_checks
                .iter()
                .take(2)
                .map(|c| {
                    if c.description.len() > 20 {
                        format!("{}...", &c.description[..17])
                    } else {
                        c.description.clone()
                    }
                })
                .collect();
            let checks_str = if checks.is_empty() {
                String::new()
            } else {
                format!("<br/>{}", checks.join(", "))
            };

            mermaid.push_str(&format!(
                "    {}[\"Cluster {}<br/>{}<br/>({} members){}\"]\n",
                cluster_id,
                i + 1,
                safe_label,
                cluster.size(),
                mermaid_escape(&checks_str)
            ));

            mermaid.push_str(&format!("    root --> {}\n", cluster_id));
        }

        // Style definitions
        mermaid.push_str("\n    %% Style definitions\n");
        mermaid.push_str("    classDef root fill:#e8f5e9,stroke:#4caf50,stroke-width:3px\n");
        mermaid.push_str("    classDef cluster fill:#e3f2fd,stroke:#1976d2,stroke-width:2px\n");
        mermaid.push_str("    class root root\n");
        for i in 0..self.clusters.len() {
            mermaid.push_str(&format!("    class c{} cluster\n", i));
        }

        mermaid
    }

    /// Export clusters as a standalone HTML page with embedded visualization
    ///
    /// Generates a complete HTML document with summary statistics,
    /// cluster details table, and interactive Mermaid diagrams.
    pub fn to_html(&self, title: Option<&str>) -> String {
        let title = title.unwrap_or("Counterexample Clusters Visualization");
        let mermaid_pie = self.to_mermaid();
        let mermaid_flowchart = self.to_mermaid_flowchart();
        let download_pie = html_download_buttons(&mermaid_pie, None);
        let download_flowchart = html_download_buttons(&mermaid_flowchart, None);

        // Build cluster details table
        let mut clusters_html = String::from(
            "<table class=\"clusters-table\"><tr><th>#</th><th>Label</th><th>Members</th><th>Sample Failures</th></tr>",
        );
        for (i, cluster) in self.clusters.iter().enumerate() {
            let failures: Vec<_> = cluster
                .representative
                .failed_checks
                .iter()
                .take(2)
                .map(|c| c.description.clone())
                .collect();
            let failures_str = if failures.is_empty() {
                "(none)".to_string()
            } else {
                failures.join(", ")
            };
            clusters_html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td><code>{}</code></td></tr>",
                i + 1,
                cluster.label,
                cluster.size(),
                failures_str
            ));
        }
        clusters_html.push_str("</table>");

        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #1976d2;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #666;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .stat {{
            background: #e8f5e9;
            padding: 10px 15px;
            border-radius: 4px;
            border-left: 4px solid #4caf50;
        }}
        .stat.clusters {{
            background: #e3f2fd;
            border-left-color: #1976d2;
        }}
        .stat.threshold {{
            background: #fff3e0;
            border-left-color: #ff9800;
        }}
        .clusters-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .clusters-table th, .clusters-table td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .clusters-table th {{
            background: #f5f5f5;
        }}
        .diagram {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        .diagram-tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }}
        .diagram-tab {{
            padding: 8px 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            background: #f5f5f5;
        }}
        .diagram-tab.active {{
            background: #1976d2;
            color: white;
            border-color: #1976d2;
        }}
        .diagram-content {{
            display: none;
        }}
        .diagram-content.active {{
            display: block;
        }}
        .mermaid {{
            text-align: center;
        }}
        {download_css}
    </style>
</head>
<body>
    <h1>{title}</h1>

    <div class="card">
        <h2>Summary</h2>
        <div class="stats">
            <div class="stat">
                <strong>Total Counterexamples:</strong> {total}
            </div>
            <div class="stat clusters">
                <strong>Clusters:</strong> {cluster_count}
            </div>
            <div class="stat threshold">
                <strong>Similarity Threshold:</strong> {threshold:.2}
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Cluster Details</h2>
        {clusters_html}
    </div>

    <div class="diagram">
        <h2>Visualization</h2>
        <div class="diagram-tabs">
            <button class="diagram-tab active" onclick="showDiagram('pie')">Distribution</button>
            <button class="diagram-tab" onclick="showDiagram('flowchart')">Structure</button>
        </div>
        <div id="pie" class="diagram-content active">
            {download_pie}
            <pre class="mermaid">
{mermaid_pie}
            </pre>
        </div>
        <div id="flowchart" class="diagram-content">
            {download_flowchart}
            <pre class="mermaid">
{mermaid_flowchart}
            </pre>
        </div>
    </div>

    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});

        function showDiagram(type) {{
            document.querySelectorAll('.diagram-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.diagram-tab').forEach(el => el.classList.remove('active'));
            document.getElementById(type).classList.add('active');
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>"#,
            title = title,
            total = self.total_counterexamples(),
            cluster_count = self.num_clusters(),
            threshold = self.similarity_threshold,
            clusters_html = clusters_html,
            mermaid_pie = mermaid_pie,
            mermaid_flowchart = mermaid_flowchart,
            download_pie = download_pie,
            download_flowchart = download_flowchart,
            download_css = DOWNLOAD_BUTTON_CSS,
        )
    }
}

/// Statistics about a counterexample
#[derive(Debug, Clone, Default)]
pub struct CounterexampleStats {
    /// Number of states in the trace
    pub num_trace_states: usize,
    /// Number of witness variables
    pub num_witness_variables: usize,
    /// Number of variables in the trace
    pub num_trace_variables: usize,
    /// Number of variables that change during the trace
    pub num_changing_variables: usize,
    /// Number of variables that remain constant
    pub num_constant_variables: usize,
    /// Number of failed checks/assertions
    pub num_failed_checks: usize,
    /// Whether a Kani playback test is available
    pub has_playback_test: bool,
    /// Whether raw counterexample output is preserved
    pub has_raw_counterexample: bool,
    /// Whether the counterexample has been minimized
    pub is_minimized: bool,
}

impl std::fmt::Display for CounterexampleStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Counterexample Statistics:")?;
        writeln!(f, "  Trace states: {}", self.num_trace_states)?;
        writeln!(f, "  Witness variables: {}", self.num_witness_variables)?;
        writeln!(
            f,
            "  Trace variables: {} ({} changing, {} constant)",
            self.num_trace_variables, self.num_changing_variables, self.num_constant_variables
        )?;
        writeln!(f, "  Failed checks: {}", self.num_failed_checks)?;
        writeln!(f, "  Has playback test: {}", self.has_playback_test)?;
        writeln!(f, "  Has raw text: {}", self.has_raw_counterexample)?;
        writeln!(f, "  Is minimized: {}", self.is_minimized)?;
        Ok(())
    }
}
