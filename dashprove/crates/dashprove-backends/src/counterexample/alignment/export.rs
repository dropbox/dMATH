//! Export methods for MultiTraceAlignment (DOT, Mermaid, HTML)

use super::types::MultiTraceAlignment;
use crate::counterexample::types::TraceState;
use crate::traits::{html_download_buttons, DOWNLOAD_BUTTON_CSS};
use crate::util::mermaid_escape;

impl MultiTraceAlignment {
    /// Export the multi-trace alignment as a DOT graph
    ///
    /// Creates a graph showing multiple traces side by side with:
    /// - Each trace in its own subgraph/cluster
    /// - States as nodes with variable values
    /// - Edges showing transitions between states
    /// - Divergence points highlighted with colored edges between traces
    pub fn to_dot(&self) -> String {
        let mut dot = String::new();

        dot.push_str("digraph MultiTraceAlignment {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box, fontname=\"Courier\"];\n");
        dot.push_str("  edge [fontname=\"Courier\"];\n\n");

        // Add title
        dot.push_str(&format!(
            "  // Multi-Trace Alignment: {} traces, {} states\n",
            self.trace_count(),
            self.rows.len()
        ));

        // Track divergent states for highlighting
        let divergent_state_nums: std::collections::HashSet<u32> = self
            .divergence_points
            .iter()
            .map(|dp| dp.state_num)
            .collect();

        // Create a subgraph for each trace
        for (trace_idx, label) in self.trace_labels.iter().enumerate() {
            let cluster_id = format!("cluster_trace_{}", trace_idx);
            let safe_label = label.replace("\"", "\\\"");

            dot.push_str(&format!("  subgraph {} {{\n", cluster_id));
            dot.push_str(&format!("    label=\"{}\";\n", safe_label));
            dot.push_str("    style=dashed;\n");
            dot.push_str(&format!(
                "    color=\"{}\";\n",
                Self::trace_color(trace_idx)
            ));
            dot.push_str("    fontcolor=black;\n\n");

            // Add nodes for this trace
            let mut prev_node_id: Option<String> = None;
            for row in &self.rows {
                if let Some(state) = &row.states[trace_idx] {
                    let node_id = format!("t{}_s{}", trace_idx, row.state_num);
                    let is_divergent = divergent_state_nums.contains(&row.state_num);

                    // Build node label with variables
                    let mut label_parts = vec![format!("State {}", row.state_num)];
                    if let Some(action) = &state.action {
                        label_parts.push(format!("Action: {}", action));
                    }
                    label_parts.push("---".to_string());

                    // Add variable values (limited to avoid huge nodes)
                    let mut vars: Vec<_> = state.variables.iter().collect();
                    vars.sort_by_key(|(k, _)| *k);
                    for (var, val) in vars.iter().take(5) {
                        label_parts.push(format!("{} = {}", var, val));
                    }
                    if state.variables.len() > 5 {
                        label_parts.push(format!("... ({} more)", state.variables.len() - 5));
                    }

                    let node_label = label_parts.join("\\n");
                    let border_color = if is_divergent { "red" } else { "black" };

                    dot.push_str(&format!(
                        "    {} [label=\"{}\", color=\"{}\"];\n",
                        node_id, node_label, border_color
                    ));

                    // Add edge from previous state
                    if let Some(prev) = &prev_node_id {
                        dot.push_str(&format!(
                            "    {} -> {} [color=\"{}\"];\n",
                            prev,
                            node_id,
                            Self::trace_color(trace_idx)
                        ));
                    }

                    prev_node_id = Some(node_id);
                }
            }

            dot.push_str("  }\n\n");
        }

        // Add invisible edges to align states across traces horizontally
        dot.push_str("  // Alignment constraints\n");
        dot.push_str("  {\n    rank=same;\n");
        for row in &self.rows {
            let present: Vec<String> = (0..self.trace_count())
                .filter(|&i| row.states[i].is_some())
                .map(|i| format!("t{}_s{}", i, row.state_num))
                .collect();

            if present.len() > 1 {
                // Add invisible edges to align
                dot.push_str(&format!("    {{ {} }};\n", present.join("; ")));
            }
        }
        dot.push_str("  }\n\n");

        // Add divergence indicators as dotted edges between traces
        if !self.divergence_points.is_empty() {
            dot.push_str("  // Divergence points\n");
            for dp in &self.divergence_points {
                // Find which traces have this state
                let present_traces: Vec<usize> = (0..self.trace_count())
                    .filter(|&i| {
                        self.rows
                            .iter()
                            .any(|r| r.state_num == dp.state_num && r.states[i].is_some())
                    })
                    .collect();

                // Add edges between adjacent traces at this divergence point
                for window in present_traces.windows(2) {
                    if let [t1, t2] = window {
                        let val1 = dp
                            .values
                            .get(*t1)
                            .and_then(|v| v.as_ref())
                            .map(|v| format!("{}", v))
                            .unwrap_or_else(|| "-".to_string());
                        let val2 = dp
                            .values
                            .get(*t2)
                            .and_then(|v| v.as_ref())
                            .map(|v| format!("{}", v))
                            .unwrap_or_else(|| "-".to_string());

                        dot.push_str(&format!(
                            "  t{}_s{} -> t{}_s{} [style=dotted, color=red, constraint=false, label=\"{}\\n{} vs {}\"];\n",
                            t1, dp.state_num, t2, dp.state_num, dp.variable, val1, val2
                        ));
                    }
                }
            }
        }

        dot.push_str("}\n");
        dot
    }

    /// Get a color for a trace index
    fn trace_color(idx: usize) -> &'static str {
        const COLORS: [&str; 8] = [
            "blue", "green", "purple", "orange", "brown", "cyan", "magenta", "olive",
        ];
        COLORS[idx % COLORS.len()]
    }

    /// Get Mermaid color for a trace index (hex colors for compatibility)
    fn mermaid_trace_color(idx: usize) -> &'static str {
        const COLORS: [&str; 8] = [
            "#2196f3", "#4caf50", "#9c27b0", "#ff9800", "#795548", "#00bcd4", "#e91e63", "#607d8b",
        ];
        COLORS[idx % COLORS.len()]
    }

    /// Export multi-trace alignment as a Mermaid diagram for markdown embedding
    ///
    /// Creates a flowchart showing multiple traces side by side with divergence highlighting.
    /// Each trace is shown in its own subgraph with state nodes and transition edges.
    pub fn to_mermaid(&self) -> String {
        let mut mermaid = String::new();

        mermaid.push_str("flowchart TB\n");
        mermaid.push_str(&format!(
            "    %% Multi-Trace Alignment: {} traces, {} states\n",
            self.trace_count(),
            self.rows.len()
        ));

        // Track divergent states for highlighting
        let divergent_state_nums: std::collections::HashSet<u32> = self
            .divergence_points
            .iter()
            .map(|dp| dp.state_num)
            .collect();

        // Create a subgraph for each trace
        for (trace_idx, label) in self.trace_labels.iter().enumerate() {
            let safe_label = mermaid_escape(label);
            mermaid.push_str(&format!(
                "    subgraph T{}[\"{}\"]\n",
                trace_idx, safe_label
            ));
            mermaid.push_str("        direction TB\n");

            let mut prev_node_id: Option<String> = None;
            for row in &self.rows {
                if let Some(state) = &row.states[trace_idx] {
                    let node_id = format!("t{}_s{}", trace_idx, row.state_num);
                    let is_divergent = divergent_state_nums.contains(&row.state_num);

                    // Build compact node label
                    let label = Self::mermaid_state_label(state, is_divergent);

                    // Use hexagon for divergent states, rectangle for normal
                    if is_divergent {
                        mermaid.push_str(&format!(
                            "        {}{{{{\"{}\"}}}}:::divergent{}\n",
                            node_id,
                            mermaid_escape(&label),
                            trace_idx
                        ));
                    } else {
                        mermaid.push_str(&format!(
                            "        {}[\"{}\"]\n",
                            node_id,
                            mermaid_escape(&label)
                        ));
                    }

                    // Add edge from previous state
                    if let Some(prev) = &prev_node_id {
                        let edge_label = state
                            .action
                            .as_ref()
                            .map(|a| {
                                if a.len() > 12 {
                                    format!("{}...", &a[..9])
                                } else {
                                    a.clone()
                                }
                            })
                            .unwrap_or_default();
                        if edge_label.is_empty() {
                            mermaid.push_str(&format!("        {} --> {}\n", prev, node_id));
                        } else {
                            mermaid.push_str(&format!(
                                "        {} -->|\"{}\"| {}\n",
                                prev,
                                mermaid_escape(&edge_label),
                                node_id
                            ));
                        }
                    }

                    prev_node_id = Some(node_id);
                }
            }
            mermaid.push_str("    end\n\n");
        }

        // Add divergence cross-links between traces
        if !self.divergence_points.is_empty() {
            mermaid.push_str("    %% Divergence connections\n");
            for dp in &self.divergence_points {
                // Find which traces have this state
                let present_traces: Vec<usize> = (0..self.trace_count())
                    .filter(|&i| {
                        self.rows
                            .iter()
                            .any(|r| r.state_num == dp.state_num && r.states[i].is_some())
                    })
                    .collect();

                // Connect adjacent traces at divergence point
                for window in present_traces.windows(2) {
                    if let [t1, t2] = window {
                        mermaid.push_str(&format!(
                            "    t{}_s{} -.-|\"{}\"| t{}_s{}\n",
                            t1, dp.state_num, dp.variable, t2, dp.state_num
                        ));
                    }
                }
            }
        }

        // Add style definitions
        mermaid.push_str("\n    %% Style definitions\n");
        for (trace_idx, _) in self.trace_labels.iter().enumerate() {
            let color = Self::mermaid_trace_color(trace_idx);
            mermaid.push_str(&format!(
                "    classDef divergent{} fill:#ffebee,stroke:{},stroke-width:3px\n",
                trace_idx, color
            ));
        }

        mermaid
    }

    fn mermaid_state_label(state: &TraceState, is_divergent: bool) -> String {
        let mut label = format!("S{}", state.state_num);
        if is_divergent {
            label.push_str(" !");
        }
        let var_count = state.variables.len();
        if var_count <= 2 {
            for (var, val) in &state.variables {
                let val_str = format!("{}", val);
                if val_str.len() <= 15 {
                    label.push_str(&format!("<br/>{}: {}", var, val_str));
                } else {
                    label.push_str(&format!("<br/>{}...", var));
                }
            }
        } else {
            label.push_str(&format!("<br/>({} vars)", var_count));
        }
        label
    }

    /// Export multi-trace alignment as a standalone HTML page with embedded visualization
    ///
    /// Generates a complete HTML document that can be opened in a browser.
    /// Includes summary statistics, divergence points table, and interactive Mermaid diagram.
    pub fn to_html(&self, title: Option<&str>) -> String {
        let title = title.unwrap_or("Multi-Trace Alignment Visualization");
        let mermaid_code = self.to_mermaid();
        let dot_code = self.to_dot();
        let download_buttons = html_download_buttons(&mermaid_code, Some(&dot_code));

        // Build trace labels list
        let traces_html = self
            .trace_labels
            .iter()
            .enumerate()
            .map(|(i, label)| {
                format!(
                    "<span class=\"trace-label\" style=\"border-color: {}\">{}</span>",
                    Self::mermaid_trace_color(i),
                    label
                )
            })
            .collect::<Vec<_>>()
            .join(" ");

        // Build divergence points table
        let divergence_html = if self.divergence_points.is_empty() {
            "<p>No divergence points - all traces are identical.</p>".to_string()
        } else {
            let mut html = String::from(
                "<table class=\"divergence-table\"><tr><th>State</th><th>Variable</th>",
            );
            for label in &self.trace_labels {
                html.push_str(&format!("<th>{}</th>", label));
            }
            html.push_str("</tr>");

            for dp in &self.divergence_points {
                html.push_str(&format!(
                    "<tr><td>{}</td><td><code>{}</code></td>",
                    dp.state_num, dp.variable
                ));
                for val in &dp.values {
                    let val_str = val
                        .as_ref()
                        .map(|v| format!("<code>{}</code>", v))
                        .unwrap_or_else(|| "<em>-</em>".to_string());
                    html.push_str(&format!("<td>{}</td>", val_str));
                }
                html.push_str("</tr>");
            }
            html.push_str("</table>");
            html
        };

        // Find first divergence state
        let first_div = self
            .first_divergence()
            .map(|s| format!("State {}", s))
            .unwrap_or_else(|| "None".to_string());

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
            border-bottom: 2px solid #2196f3;
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
            margin-bottom: 15px;
        }}
        .stat {{
            background: #e8f5e9;
            padding: 10px 15px;
            border-radius: 4px;
            border-left: 4px solid #4caf50;
        }}
        .stat.traces {{
            background: #e3f2fd;
            border-left-color: #2196f3;
        }}
        .stat.divergent {{
            background: #ffebee;
            border-left-color: #f44336;
        }}
        .trace-label {{
            display: inline-block;
            padding: 4px 12px;
            margin: 2px;
            border-radius: 4px;
            background: #f5f5f5;
            border-left: 4px solid;
        }}
        .divergence-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .divergence-table th, .divergence-table td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .divergence-table th {{
            background: #f5f5f5;
        }}
        .divergence-table tr:hover {{
            background: #fff8e1;
        }}
        .diagram {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        .mermaid {{
            text-align: center;
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background: #fafafa;
            border-radius: 4px;
        }}
        .legend-item {{
            display: inline-flex;
            align-items: center;
            margin-right: 20px;
        }}
        .legend-box {{
            width: 20px;
            height: 20px;
            margin-right: 8px;
            border-radius: 4px;
        }}
        .legend-normal {{ background: white; border: 2px solid #333; }}
        .legend-divergent {{ background: #ffebee; border: 3px solid #f44336; }}
        {download_css}
        footer {{
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 12px;
        }}
        code {{
            background: #f5f5f5;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>

    <div class="card">
        <h2>Summary</h2>
        <div class="stats">
            <div class="stat traces">
                <strong>{trace_count}</strong> traces
            </div>
            <div class="stat">
                <strong>{state_count}</strong> states
            </div>
            <div class="stat divergent">
                <strong>{divergence_count}</strong> divergence points
            </div>
            <div class="stat">
                First divergence: <strong>{first_div}</strong>
            </div>
        </div>
        <div>
            <strong>Traces:</strong> {traces_html}
        </div>
    </div>

    <div class="card">
        <h2>Divergence Points</h2>
        {divergence_html}
    </div>

    <div class="diagram">
        <h2>Trace Diagram</h2>
        {download_buttons}
        <div class="mermaid">
{mermaid}
        </div>

        <div class="legend">
            <span class="legend-item">
                <span class="legend-box legend-normal"></span>
                Normal state
            </span>
            <span class="legend-item">
                <span class="legend-box legend-divergent"></span>
                Divergent state
            </span>
        </div>
    </div>

    <footer>
        Generated by DashProve Counterexample Visualization
    </footer>

    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
    </script>
</body>
</html>"#,
            title = title,
            trace_count = self.trace_count(),
            state_count = self.rows.len(),
            divergence_count = self.divergence_points.len(),
            first_div = first_div,
            traces_html = traces_html,
            divergence_html = divergence_html,
            mermaid = mermaid_code,
            download_buttons = download_buttons,
            download_css = DOWNLOAD_BUTTON_CSS,
        )
    }
}
