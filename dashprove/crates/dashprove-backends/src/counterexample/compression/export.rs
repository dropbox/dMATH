//! Export methods for CompressedTrace (DOT, Mermaid, HTML)

use super::types::{CompressedTrace, TraceSegment};
use crate::counterexample::types::TraceState;
use crate::traits::{html_download_buttons, DOWNLOAD_BUTTON_CSS};
use crate::util::mermaid_escape;

impl CompressedTrace {
    /// Export compressed trace as DOT graph format
    /// Nodes are states, edges are transitions between states
    pub fn to_dot(&self) -> String {
        let mut dot = String::new();
        dot.push_str("digraph CompressedTrace {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box, style=rounded];\n");
        dot.push_str("  edge [fontsize=10];\n\n");

        let mut node_id = 0usize;
        let mut prev_node: Option<usize> = None;

        for (seg_idx, segment) in self.segments.iter().enumerate() {
            match segment {
                TraceSegment::Single(state) => {
                    let label = Self::dot_state_label(state);
                    dot.push_str(&format!(
                        "  s{} [label=\"{}\"];\n",
                        node_id,
                        Self::escape_dot(&label)
                    ));
                    if let Some(prev) = prev_node {
                        let edge_label = state
                            .action
                            .as_ref()
                            .map(|a| Self::escape_dot(a))
                            .unwrap_or_default();
                        dot.push_str(&format!(
                            "  s{} -> s{} [label=\"{}\"];\n",
                            prev, node_id, edge_label
                        ));
                    }
                    prev_node = Some(node_id);
                    node_id += 1;
                }
                TraceSegment::Repeated { pattern, count } => {
                    // Create a subgraph for the repeated pattern
                    dot.push_str(&format!("  subgraph cluster_{} {{\n", seg_idx));
                    dot.push_str(&format!("    label=\"Repeated {}x\";\n", count));
                    dot.push_str("    style=dashed;\n");
                    dot.push_str("    color=blue;\n");

                    let cluster_start = node_id;
                    for (i, state) in pattern.iter().enumerate() {
                        let label = Self::dot_state_label(state);
                        dot.push_str(&format!(
                            "    s{} [label=\"{}\"];\n",
                            node_id,
                            Self::escape_dot(&label)
                        ));
                        if i > 0 {
                            let edge_label = state
                                .action
                                .as_ref()
                                .map(|a| Self::escape_dot(a))
                                .unwrap_or_default();
                            dot.push_str(&format!(
                                "    s{} -> s{} [label=\"{}\"];\n",
                                node_id - 1,
                                node_id,
                                edge_label
                            ));
                        }
                        node_id += 1;
                    }

                    // Add loop-back edge for repetition
                    if !pattern.is_empty() {
                        dot.push_str(&format!(
                            "    s{} -> s{} [style=dashed, color=blue, constraint=false];\n",
                            node_id - 1,
                            cluster_start
                        ));
                    }

                    dot.push_str("  }\n");

                    // Connect from previous segment
                    if let Some(prev) = prev_node {
                        let edge_label = pattern
                            .first()
                            .and_then(|s| s.action.as_ref())
                            .map(|a| Self::escape_dot(a))
                            .unwrap_or_default();
                        dot.push_str(&format!(
                            "  s{} -> s{} [label=\"{}\"];\n",
                            prev, cluster_start, edge_label
                        ));
                    }
                    prev_node = Some(node_id - 1);
                }
            }
        }

        dot.push_str("}\n");
        dot
    }

    fn dot_state_label(state: &TraceState) -> String {
        let mut label = format!("State {}", state.state_num);
        if state.variables.len() <= 3 {
            for (var, val) in &state.variables {
                label.push_str(&format!("\\n{}: {}", var, val));
            }
        } else {
            label.push_str(&format!("\\n({} vars)", state.variables.len()));
        }
        label
    }

    fn escape_dot(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    }

    /// Export compressed trace as Mermaid flowchart diagram
    ///
    /// Creates a flowchart showing compressed segments with repeated patterns
    /// shown in subgraphs with loop indicators. Useful for embedding in markdown.
    pub fn to_mermaid(&self) -> String {
        let mut mermaid = String::new();

        mermaid.push_str("flowchart TB\n");
        mermaid.push_str(&format!(
            "    %% Compressed Trace ({} -> {} segments, {:.0}% compression)\n",
            self.original_length,
            self.segments.len(),
            self.compression_ratio() * 100.0
        ));

        if self.segments.is_empty() {
            mermaid.push_str("    empty[\"Empty trace\"]\n");
            return mermaid;
        }

        let mut node_id = 0usize;
        let mut prev_node: Option<String> = None;

        for (seg_idx, segment) in self.segments.iter().enumerate() {
            match segment {
                TraceSegment::Single(state) => {
                    let current_id = format!("s{}", node_id);
                    let label = Self::mermaid_state_label(state);
                    mermaid.push_str(&format!(
                        "    {}[\"{}\"]\n",
                        current_id,
                        mermaid_escape(&label)
                    ));

                    if let Some(ref prev) = prev_node {
                        let edge_label = state
                            .action
                            .as_ref()
                            .map(|a| {
                                if a.len() > 15 {
                                    format!("{}...", &a[..12])
                                } else {
                                    a.clone()
                                }
                            })
                            .unwrap_or_default();
                        if edge_label.is_empty() {
                            mermaid.push_str(&format!("    {} --> {}\n", prev, current_id));
                        } else {
                            mermaid.push_str(&format!(
                                "    {} -->|\"{}\"| {}\n",
                                prev,
                                mermaid_escape(&edge_label),
                                current_id
                            ));
                        }
                    }
                    prev_node = Some(current_id);
                    node_id += 1;
                }
                TraceSegment::Repeated { pattern, count } => {
                    // Create a subgraph for the repeated pattern
                    let subgraph_id = format!("repeat_{}", seg_idx);
                    mermaid.push_str(&format!(
                        "\n    subgraph {}[\"Repeat {}x\"]\n",
                        subgraph_id, count
                    ));
                    mermaid.push_str("        direction TB\n");

                    let cluster_start = format!("s{}", node_id);
                    let mut cluster_prev: Option<String> = None;

                    for state in pattern {
                        let current_id = format!("s{}", node_id);
                        let label = Self::mermaid_state_label(state);
                        mermaid.push_str(&format!(
                            "        {}[\"{}\"]\n",
                            current_id,
                            mermaid_escape(&label)
                        ));

                        if let Some(ref prev) = cluster_prev {
                            let edge_label = state
                                .action
                                .as_ref()
                                .map(|a| {
                                    if a.len() > 15 {
                                        format!("{}...", &a[..12])
                                    } else {
                                        a.clone()
                                    }
                                })
                                .unwrap_or_default();
                            if edge_label.is_empty() {
                                mermaid.push_str(&format!("        {} --> {}\n", prev, current_id));
                            } else {
                                mermaid.push_str(&format!(
                                    "        {} -->|\"{}\"| {}\n",
                                    prev,
                                    mermaid_escape(&edge_label),
                                    current_id
                                ));
                            }
                        }
                        cluster_prev = Some(current_id);
                        node_id += 1;
                    }

                    // Add loop-back edge for repetition
                    if let Some(ref last) = cluster_prev {
                        if pattern.len() > 1 {
                            mermaid.push_str(&format!(
                                "        {} -.\"loop\".- {}\n",
                                last, cluster_start
                            ));
                        }
                    }

                    mermaid.push_str("    end\n");

                    // Connect from previous segment
                    if let Some(ref prev) = prev_node {
                        let edge_label = pattern
                            .first()
                            .and_then(|s| s.action.as_ref())
                            .map(|a| {
                                if a.len() > 15 {
                                    format!("{}...", &a[..12])
                                } else {
                                    a.clone()
                                }
                            })
                            .unwrap_or_default();
                        if edge_label.is_empty() {
                            mermaid.push_str(&format!("    {} --> {}\n", prev, cluster_start));
                        } else {
                            mermaid.push_str(&format!(
                                "    {} -->|\"{}\"| {}\n",
                                prev,
                                mermaid_escape(&edge_label),
                                cluster_start
                            ));
                        }
                    }
                    prev_node = cluster_prev;
                }
            }
        }

        // Add style definitions
        mermaid.push_str("\n    %% Style definitions\n");
        mermaid.push_str("    classDef repeated fill:#e3f2fd,stroke:#1976d2,stroke-width:2px\n");

        mermaid
    }

    fn mermaid_state_label(state: &TraceState) -> String {
        let mut label = format!("S{}", state.state_num);
        if let Some(action) = &state.action {
            if !action.is_empty() {
                let action_display = if action.len() > 20 {
                    format!("{}...", &action[..17])
                } else {
                    action.clone()
                };
                label.push_str(&format!("<br/>{}", action_display));
            }
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

    /// Export compressed trace as a standalone HTML page with embedded visualization
    ///
    /// Generates a complete HTML document that can be opened in a browser.
    /// Includes summary statistics, segment breakdown, and interactive Mermaid diagram.
    pub fn to_html(&self, title: Option<&str>) -> String {
        let title = title.unwrap_or("Compressed Trace Visualization");
        let mermaid_code = self.to_mermaid();
        let dot_code = self.to_dot();
        let download_buttons = html_download_buttons(&mermaid_code, Some(&dot_code));

        // Build segment details
        let mut segments_html = String::from(
            "<table class=\"segments-table\"><tr><th>#</th><th>Type</th><th>Details</th></tr>",
        );
        for (i, segment) in self.segments.iter().enumerate() {
            match segment {
                TraceSegment::Single(state) => {
                    let action = state.action.as_deref().unwrap_or("(no action)");
                    segments_html.push_str(&format!(
                        "<tr><td>{}</td><td>Single</td><td>State {}: {}</td></tr>",
                        i + 1,
                        state.state_num,
                        action
                    ));
                }
                TraceSegment::Repeated { pattern, count } => {
                    let pattern_desc = if pattern.len() == 1 {
                        format!("1 state repeated {}x", count)
                    } else {
                        format!("{} states repeated {}x", pattern.len(), count)
                    };
                    segments_html.push_str(&format!(
                        "<tr class=\"repeated\"><td>{}</td><td>Repeated</td><td>{}</td></tr>",
                        i + 1,
                        pattern_desc
                    ));
                }
            }
        }
        segments_html.push_str("</table>");

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
            max-width: 1200px;
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
        .stat.compression {{
            background: #e3f2fd;
            border-left-color: #1976d2;
        }}
        .stat.segments {{
            background: #fff3e0;
            border-left-color: #ff9800;
        }}
        .segments-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .segments-table th, .segments-table td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .segments-table th {{
            background: #f5f5f5;
        }}
        .segments-table tr.repeated {{
            background: #e3f2fd;
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
            font-size: 0.9em;
        }}
        .legend-item {{
            display: inline-flex;
            align-items: center;
            margin-right: 20px;
            margin-bottom: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            margin-right: 8px;
            border: 2px solid;
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
                <strong>Original States:</strong> {original_length}
            </div>
            <div class="stat segments">
                <strong>Segments:</strong> {segment_count}
            </div>
            <div class="stat compression">
                <strong>Compression:</strong> {compression:.0}%
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Segments</h2>
        {segments_html}
    </div>

    <div class="diagram">
        <h2>Trace Diagram</h2>
        {download_buttons}
        <pre class="mermaid">
{mermaid_code}
        </pre>
        <div class="legend">
            <strong>Legend:</strong>
            <span class="legend-item"><span class="legend-color" style="background: #fff; border-color: #333;"></span> Single State</span>
            <span class="legend-item"><span class="legend-color" style="background: #e3f2fd; border-color: #1976d2;"></span> Repeated Pattern</span>
            <span class="legend-item">Dashed arrows indicate loop repetition</span>
        </div>
    </div>

    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>"#,
            title = title,
            original_length = self.original_length,
            segment_count = self.segments.len(),
            compression = self.compression_ratio() * 100.0,
            segments_html = segments_html,
            mermaid_code = mermaid_code,
            download_buttons = download_buttons,
            download_css = DOWNLOAD_BUTTON_CSS,
        )
    }
}
