//! Export methods for StructuredCounterexample (DOT, Mermaid, HTML)

use crate::counterexample::types::{StructuredCounterexample, TraceState};
use crate::traits::{html_download_buttons, DOWNLOAD_BUTTON_CSS};
use crate::util::mermaid_escape;

impl StructuredCounterexample {
    /// Export trace as DOT graph format
    pub fn to_dot(&self) -> String {
        let mut dot = String::new();
        dot.push_str("digraph Counterexample {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box, style=rounded];\n");
        dot.push_str("  edge [fontsize=10];\n\n");

        // Add witness node if present
        if !self.witness.is_empty() {
            dot.push_str("  witness [shape=note, label=\"Witness\\n");
            for (var, val) in &self.witness {
                let val_str = format!("{}", val);
                dot.push_str(&format!("{}: {}\\n", var, Self::dot_escape(&val_str)));
            }
            dot.push_str("\"];\n\n");
        }

        // Add trace states
        for (i, state) in self.trace.iter().enumerate() {
            let label = self.dot_state_label(state);
            dot.push_str(&format!(
                "  s{} [label=\"{}\"];\n",
                i,
                Self::dot_escape(&label)
            ));

            if i > 0 {
                let edge_label = state
                    .action
                    .as_ref()
                    .map(|a| Self::dot_escape(a))
                    .unwrap_or_default();
                dot.push_str(&format!(
                    "  s{} -> s{} [label=\"{}\"];\n",
                    i - 1,
                    i,
                    edge_label
                ));
            }
        }

        // Highlight cycle if detected
        if let Some((cycle_start, _cycle_len)) = self.detect_cycle() {
            let cycle_end = self.trace.len() - 1;
            if cycle_end > cycle_start {
                dot.push_str(&format!(
                    "  s{} -> s{} [style=dashed, color=red, label=\"cycle\"];\n",
                    cycle_end, cycle_start
                ));
            }
        }

        // Add failed checks
        if !self.failed_checks.is_empty() {
            dot.push_str("\n  failures [shape=octagon, color=red, label=\"Failed Checks\\n");
            for check in &self.failed_checks {
                dot.push_str(&format!("{}\\n", Self::dot_escape(&check.description)));
            }
            dot.push_str("\"];\n");

            // Connect last state to failures
            if !self.trace.is_empty() {
                dot.push_str(&format!(
                    "  s{} -> failures [style=bold, color=red];\n",
                    self.trace.len() - 1
                ));
            }
        }

        dot.push_str("}\n");
        dot
    }

    fn dot_state_label(&self, state: &TraceState) -> String {
        let mut label = format!("State {}", state.state_num);
        let var_count = state.variables.len();
        if var_count <= 4 {
            for (var, val) in &state.variables {
                let val_str = format!("{}", val);
                if val_str.len() <= 30 {
                    label.push_str(&format!("\\n{}: {}", var, val_str));
                } else {
                    label.push_str(&format!("\\n{}: ...", var));
                }
            }
        } else {
            label.push_str(&format!("\\n({} variables)", var_count));
        }
        label
    }

    fn dot_escape(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    }

    /// Export counterexample as a Mermaid diagram for markdown embedding
    ///
    /// Creates a flowchart showing the trace states and failed checks.
    /// Useful for embedding in GitHub markdown, documentation, etc.
    pub fn to_mermaid(&self) -> String {
        let mut mermaid = String::new();

        mermaid.push_str("flowchart TB\n");
        mermaid.push_str("    %% Counterexample trace\n");

        // Add witness node if present
        if !self.witness.is_empty() {
            mermaid.push_str("    witness>\"Witness");
            for (var, val) in self.witness.iter().take(5) {
                mermaid.push_str(&format!(
                    "<br/>{}: {}",
                    var,
                    mermaid_escape(&format!("{}", val))
                ));
            }
            if self.witness.len() > 5 {
                mermaid.push_str(&format!("<br/>...({} more)", self.witness.len() - 5));
            }
            mermaid.push_str("\"]:::witness\n\n");
        }

        // Add trace states
        for (i, state) in self.trace.iter().enumerate() {
            let label = self.mermaid_state_label(state);
            mermaid.push_str(&format!("    s{}[\"{}\"]\n", i, mermaid_escape(&label)));

            if i > 0 {
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
                    mermaid.push_str(&format!("    s{} --> s{}\n", i - 1, i));
                } else {
                    mermaid.push_str(&format!(
                        "    s{} -->|\"{}\"| s{}\n",
                        i - 1,
                        mermaid_escape(&edge_label),
                        i
                    ));
                }
            }
        }

        // Highlight cycle if detected
        if let Some((cycle_start, _cycle_len)) = self.detect_cycle() {
            let cycle_end = self.trace.len() - 1;
            if cycle_end > cycle_start {
                mermaid.push_str(&format!(
                    "    s{} -.\"cycle\".- s{}:::cycle\n",
                    cycle_end, cycle_start
                ));
            }
        }

        // Add failed checks
        if !self.failed_checks.is_empty() {
            mermaid.push_str("\n    failures{{\"Failed Checks");
            for check in self.failed_checks.iter().take(3) {
                mermaid.push_str(&format!("<br/>{}", mermaid_escape(&check.description)));
            }
            if self.failed_checks.len() > 3 {
                mermaid.push_str(&format!("<br/>...({} more)", self.failed_checks.len() - 3));
            }
            mermaid.push_str("\"}}:::error\n");

            // Connect last state to failures
            if !self.trace.is_empty() {
                mermaid.push_str(&format!("    s{} ==> failures\n", self.trace.len() - 1));
            }
        }

        // Add style definitions
        mermaid.push_str("\n    %% Style definitions\n");
        mermaid.push_str("    classDef witness fill:#e3f2fd,stroke:#1976d2,stroke-width:2px\n");
        mermaid.push_str("    classDef error fill:#ffebee,stroke:#c62828,stroke-width:3px\n");
        mermaid
            .push_str("    classDef cycle stroke:#ff9800,stroke-width:2px,stroke-dasharray: 5 5\n");

        mermaid
    }

    fn mermaid_state_label(&self, state: &TraceState) -> String {
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

    /// Export counterexample as a standalone HTML page with embedded visualization
    ///
    /// Generates a complete HTML document that can be opened in a browser.
    /// Includes summary information, witness values, trace, and failed checks.
    /// Includes download buttons for Mermaid and DOT formats.
    pub fn to_html(&self, title: Option<&str>) -> String {
        let title = title.unwrap_or("Counterexample Visualization");
        let mermaid_code = self.to_mermaid();
        let dot_code = self.to_dot();
        let download_buttons = html_download_buttons(&mermaid_code, Some(&dot_code));

        // Build witness table
        let witness_html = if self.witness.is_empty() {
            "<p>No witness values</p>".to_string()
        } else {
            let mut html = String::from(
                "<table class=\"witness-table\"><tr><th>Variable</th><th>Value</th></tr>",
            );
            for (var, val) in &self.witness {
                html.push_str(&format!(
                    "<tr><td><code>{}</code></td><td><code>{}</code></td></tr>",
                    var, val
                ));
            }
            html.push_str("</table>");
            html
        };

        // Build failed checks list
        let failures_html = if self.failed_checks.is_empty() {
            String::new()
        } else {
            let mut html = String::from("<div class=\"failures\"><h3>Failed Checks</h3><ul>");
            for check in &self.failed_checks {
                html.push_str(&format!("<li><strong>{}</strong>", check.description));
                if let Some(loc) = &check.location {
                    html.push_str(&format!(" <span class=\"location\">at {}</span>", loc));
                }
                html.push_str("</li>");
            }
            html.push_str("</ul></div>");
            html
        };

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
            border-bottom: 2px solid #c62828;
            padding-bottom: 10px;
        }}
        h2, h3 {{
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
            background: #e3f2fd;
            padding: 10px 15px;
            border-radius: 4px;
            border-left: 4px solid #1976d2;
        }}
        .stat.error {{
            background: #ffebee;
            border-left-color: #c62828;
        }}
        .witness-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .witness-table th, .witness-table td {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        .witness-table th {{
            background: #f5f5f5;
        }}
        .witness-table code {{
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .failures {{
            margin-top: 15px;
            padding: 15px;
            background: #ffebee;
            border-radius: 4px;
            border-left: 4px solid #c62828;
        }}
        .failures h3 {{
            margin-top: 0;
            color: #c62828;
        }}
        .failures ul {{
            margin-bottom: 0;
        }}
        .location {{
            color: #666;
            font-size: 0.9em;
        }}
        .diagram {{
            overflow-x: auto;
        }}
        .mermaid {{
            text-align: center;
        }}
        footer {{
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 12px;
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
                <strong>{trace_len}</strong> trace states
            </div>
            <div class="stat">
                <strong>{witness_len}</strong> witness values
            </div>
            <div class="stat error">
                <strong>{failure_count}</strong> failed checks
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Witness Values</h2>
        {witness_html}
    </div>

    <div class="card">
        <h2>Execution Trace</h2>
        {download_buttons}
        <div class="diagram">
            <div class="mermaid">
{mermaid}
            </div>
        </div>
        {failures_html}
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
            trace_len = self.trace.len(),
            witness_len = self.witness.len(),
            failure_count = self.failed_checks.len(),
            witness_html = witness_html,
            failures_html = failures_html,
            mermaid = mermaid_code,
            download_css = DOWNLOAD_BUTTON_CSS,
            download_buttons = download_buttons,
        )
    }
}
