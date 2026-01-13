//! Export methods for AbstractedTrace (DOT, Mermaid, HTML)

use super::{AbstractedState, AbstractedTrace, TraceAbstractionSegment};
use crate::counterexample::types::TraceState;
use crate::traits::{html_download_buttons, DOWNLOAD_BUTTON_CSS};
use crate::util::mermaid_escape;

impl AbstractedTrace {
    /// Export abstracted trace as DOT graph format
    /// Shows concrete states as boxes and abstracted segments as dashed boxes with details
    pub fn to_dot(&self) -> String {
        let mut dot = String::new();
        dot.push_str("digraph AbstractedTrace {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box, style=rounded];\n");
        dot.push_str("  edge [fontsize=10];\n\n");

        dot.push_str(&format!(
            "  // Original: {} states, compressed to {} segments ({:.1}%)\n\n",
            self.original_length,
            self.segments.len(),
            self.compression_ratio * 100.0
        ));

        let mut prev_node: Option<usize> = None;

        for (seg_idx, segment) in self.segments.iter().enumerate() {
            match segment {
                TraceAbstractionSegment::Concrete(state) => {
                    let label = Self::dot_state_label(state);
                    dot.push_str(&format!(
                        "  s{} [label=\"{}\"];\n",
                        seg_idx,
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
                            prev, seg_idx, edge_label
                        ));
                    }
                    prev_node = Some(seg_idx);
                }
                TraceAbstractionSegment::Abstracted(abstracted) => {
                    // Abstracted states shown as dashed boxes with summary
                    let mut label =
                        format!("{}\\n({} states)", abstracted.description, abstracted.count);

                    // Add variable summaries (up to 3)
                    let var_count = abstracted.variables.len();
                    if var_count > 0 {
                        let mut vars: Vec<_> = abstracted.variables.iter().collect();
                        vars.sort_by(|a, b| a.0.cmp(b.0));
                        for (var, val) in vars.iter().take(3) {
                            let val_str = format!("{}", val);
                            if val_str.len() <= 20 {
                                label.push_str(&format!("\\n{}: {}", var, val_str));
                            } else {
                                label.push_str(&format!("\\n{}: ...", var));
                            }
                        }
                        if var_count > 3 {
                            label.push_str(&format!("\\n(+{} more)", var_count - 3));
                        }
                    }

                    dot.push_str(&format!(
                        "  s{} [label=\"{}\", style=\"dashed,rounded\", color=blue];\n",
                        seg_idx,
                        Self::escape_dot(&label)
                    ));

                    if let Some(prev) = prev_node {
                        let edge_label = abstracted
                            .common_action
                            .as_ref()
                            .map(|a| Self::escape_dot(a))
                            .unwrap_or_default();
                        dot.push_str(&format!(
                            "  s{} -> s{} [label=\"{}\"];\n",
                            prev, seg_idx, edge_label
                        ));
                    }
                    prev_node = Some(seg_idx);
                }
            }
        }

        dot.push_str("}\n");
        dot
    }

    fn dot_state_label(state: &TraceState) -> String {
        let mut label = format!("State {}", state.state_num);
        let var_count = state.variables.len();
        if var_count <= 3 {
            for (var, val) in &state.variables {
                let val_str = format!("{}", val);
                if val_str.len() <= 20 {
                    label.push_str(&format!("\\n{}: {}", var, val_str));
                } else {
                    label.push_str(&format!("\\n{}: ...", var));
                }
            }
        } else {
            label.push_str(&format!("\\n({} vars)", var_count));
        }
        label
    }

    fn escape_dot(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    }

    /// Export abstracted trace as a Mermaid diagram for markdown embedding
    ///
    /// Creates a flowchart showing concrete states and abstracted segments.
    /// Abstracted segments are shown with dashed borders and summary information.
    pub fn to_mermaid(&self) -> String {
        let mut mermaid = String::new();

        mermaid.push_str("flowchart TB\n");
        mermaid.push_str(&format!(
            "    %% Abstracted Trace: {} states -> {} segments ({:.1}% compression)\n",
            self.original_length,
            self.segments.len(),
            self.compression_ratio * 100.0
        ));

        for (seg_idx, segment) in self.segments.iter().enumerate() {
            match segment {
                TraceAbstractionSegment::Concrete(state) => {
                    let label = Self::mermaid_state_label(state);
                    mermaid.push_str(&format!(
                        "    s{}[\"{}\"]\n",
                        seg_idx,
                        mermaid_escape(&label)
                    ));

                    if seg_idx > 0 {
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
                            mermaid.push_str(&format!("    s{} --> s{}\n", seg_idx - 1, seg_idx));
                        } else {
                            mermaid.push_str(&format!(
                                "    s{} -->|\"{}\"| s{}\n",
                                seg_idx - 1,
                                mermaid_escape(&edge_label),
                                seg_idx
                            ));
                        }
                    }
                }
                TraceAbstractionSegment::Abstracted(abs) => {
                    let label = Self::mermaid_abstracted_label(abs);
                    // Use stadium shape for abstracted segments
                    mermaid.push_str(&format!(
                        "    s{}([\"{}\"]\n):::abstracted\n",
                        seg_idx,
                        mermaid_escape(&label)
                    ));

                    if seg_idx > 0 {
                        let edge_label = abs
                            .common_action
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
                            mermaid.push_str(&format!("    s{} --> s{}\n", seg_idx - 1, seg_idx));
                        } else {
                            mermaid.push_str(&format!(
                                "    s{} -->|\"{}\"| s{}\n",
                                seg_idx - 1,
                                mermaid_escape(&edge_label),
                                seg_idx
                            ));
                        }
                    }
                }
            }
        }

        // Add style definitions
        mermaid.push_str("\n    %% Style definitions\n");
        mermaid.push_str(
            "    classDef abstracted fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,stroke-dasharray: 5 5\n",
        );

        mermaid
    }

    fn mermaid_state_label(state: &TraceState) -> String {
        let mut label = format!("S{}", state.state_num);
        let var_count = state.variables.len();
        if var_count <= 3 {
            for (var, val) in &state.variables {
                let val_str = format!("{}", val);
                if val_str.len() <= 20 {
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

    fn mermaid_abstracted_label(abs: &AbstractedState) -> String {
        let mut label = format!("{}<br/>({} states)", abs.description, abs.count);
        let var_count = abs.variables.len();
        if var_count > 0 {
            let mut vars: Vec<_> = abs.variables.iter().collect();
            vars.sort_by(|a, b| a.0.cmp(b.0));
            for (var, val) in vars.iter().take(3) {
                let val_str = format!("{}", val);
                if val_str.len() <= 20 {
                    label.push_str(&format!("<br/>{}: {}", var, val_str));
                } else {
                    label.push_str(&format!("<br/>{}...", var));
                }
            }
            if var_count > 3 {
                label.push_str(&format!("<br/>(+{} more)", var_count - 3));
            }
        }
        label
    }

    /// Export abstracted trace as a standalone HTML page with embedded visualization
    ///
    /// Generates a complete HTML document that can be opened in a browser.
    /// Includes summary statistics, segment breakdown, and interactive Mermaid diagram.
    pub fn to_html(&self, title: Option<&str>) -> String {
        let title = title.unwrap_or("Abstracted Trace Visualization");
        let mermaid_code = self.to_mermaid();
        let dot_code = self.to_dot();
        let download_buttons = html_download_buttons(&mermaid_code, Some(&dot_code));

        // Count concrete vs abstracted segments
        let concrete_count = self
            .segments
            .iter()
            .filter(|s| matches!(s, TraceAbstractionSegment::Concrete(_)))
            .count();
        let abstracted_count = self.segments.len() - concrete_count;
        let abstracted_states: usize = self
            .segments
            .iter()
            .filter_map(|s| match s {
                TraceAbstractionSegment::Abstracted(a) => Some(a.count),
                _ => None,
            })
            .sum();

        // Build segment details table
        let mut segments_html = String::from(
            "<table class=\"segments-table\"><tr><th>#</th><th>Type</th><th>Description</th><th>Details</th></tr>",
        );
        for (i, segment) in self.segments.iter().enumerate() {
            match segment {
                TraceAbstractionSegment::Concrete(state) => {
                    let vars: Vec<String> = state
                        .variables
                        .iter()
                        .take(3)
                        .map(|(k, v)| format!("{}: {}", k, v))
                        .collect();
                    let details = if state.variables.len() > 3 {
                        format!("{} (+{} more)", vars.join(", "), state.variables.len() - 3)
                    } else {
                        vars.join(", ")
                    };
                    segments_html.push_str(&format!(
                        "<tr><td>{}</td><td>Concrete</td><td>State {}</td><td><code>{}</code></td></tr>",
                        i + 1,
                        state.state_num,
                        details
                    ));
                }
                TraceAbstractionSegment::Abstracted(abs) => {
                    let vars: Vec<String> = abs
                        .variables
                        .iter()
                        .take(3)
                        .map(|(k, v)| format!("{}: {}", k, v))
                        .collect();
                    let details = if abs.variables.len() > 3 {
                        format!("{} (+{} more)", vars.join(", "), abs.variables.len() - 3)
                    } else {
                        vars.join(", ")
                    };
                    segments_html.push_str(&format!(
                        "<tr class=\"abstracted\"><td>{}</td><td>Abstracted</td><td>{} ({} states)</td><td><code>{}</code></td></tr>",
                        i + 1,
                        abs.description,
                        abs.count,
                        details
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
        .stat.abstracted {{
            background: #e3f2fd;
            border-left-color: #1976d2;
        }}
        .stat.compression {{
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
        .segments-table tr.abstracted {{
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
        .legend-concrete {{ background: white; border: 2px solid #333; }}
        .legend-abstracted {{ background: #e3f2fd; border: 2px dashed #1976d2; }}
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
            <div class="stat">
                <strong>{original}</strong> original states
            </div>
            <div class="stat">
                <strong>{segments}</strong> segments
            </div>
            <div class="stat abstracted">
                <strong>{abstracted}</strong> abstracted ({abstracted_states} states)
            </div>
            <div class="stat">
                <strong>{concrete}</strong> concrete
            </div>
            <div class="stat compression">
                <strong>{compression:.1}%</strong> compression
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
        <div class="mermaid">
{mermaid}
        </div>

        <div class="legend">
            <span class="legend-item">
                <span class="legend-box legend-concrete"></span>
                Concrete state
            </span>
            <span class="legend-item">
                <span class="legend-box legend-abstracted"></span>
                Abstracted segment
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
            original = self.original_length,
            segments = self.segments.len(),
            abstracted = abstracted_count,
            abstracted_states = abstracted_states,
            concrete = concrete_count,
            compression = self.compression_ratio * 100.0,
            segments_html = segments_html,
            mermaid = mermaid_code,
            download_buttons = download_buttons,
            download_css = DOWNLOAD_BUTTON_CSS,
        )
    }
}
