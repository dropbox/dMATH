//! Counterexample comparison and diff functionality
//!
//! This module provides tools for comparing counterexamples and traces:
//! - `TraceDiff`: Differences between two traces
//! - `StateLevelDiff`: Differences within a single state
//! - `CounterexampleDiff`: High-level differences between counterexamples

use super::types::{CounterexampleValue, FailedCheck, StructuredCounterexample, TraceState};
use crate::traits::{html_download_buttons, DOWNLOAD_BUTTON_CSS};
use std::collections::HashMap;

// ==================== Counterexample Comparison ====================

/// State-level differences between two traces
#[derive(Debug, Clone, Default)]
pub struct TraceDiff {
    /// States only in first trace (by state_num)
    pub states_only_in_first: Vec<u32>,
    /// States only in second trace (by state_num)
    pub states_only_in_second: Vec<u32>,
    /// States in both with differences: state_num -> (vars_only_first, vars_only_second, value_diffs)
    pub state_diffs: HashMap<u32, StateLevelDiff>,
    /// States in both that are identical
    pub identical_states: Vec<u32>,
}

/// Differences within a single state
#[derive(Debug, Clone, Default)]
pub struct StateLevelDiff {
    /// Variables only in first state's version
    pub vars_only_in_first: HashMap<String, CounterexampleValue>,
    /// Variables only in second state's version
    pub vars_only_in_second: HashMap<String, CounterexampleValue>,
    /// Variables with different values (first_value, second_value)
    pub value_diffs: HashMap<String, (CounterexampleValue, CounterexampleValue)>,
    /// Action differs (first_action, second_action)
    pub action_diff: Option<(Option<String>, Option<String>)>,
}

impl StateLevelDiff {
    /// Check if there are no differences at this state level
    pub fn is_empty(&self) -> bool {
        self.vars_only_in_first.is_empty()
            && self.vars_only_in_second.is_empty()
            && self.value_diffs.is_empty()
            && self.action_diff.is_none()
    }
}

impl TraceDiff {
    /// Check if traces are equivalent
    pub fn is_equivalent(&self) -> bool {
        self.states_only_in_first.is_empty()
            && self.states_only_in_second.is_empty()
            && self.state_diffs.values().all(|d| d.is_empty())
    }

    /// Get a summary of trace differences
    pub fn summary(&self) -> String {
        if self.is_equivalent() {
            return "Traces are equivalent".to_string();
        }

        let mut parts = Vec::new();

        if !self.states_only_in_first.is_empty() {
            parts.push(format!(
                "{} states only in first",
                self.states_only_in_first.len()
            ));
        }

        if !self.states_only_in_second.is_empty() {
            parts.push(format!(
                "{} states only in second",
                self.states_only_in_second.len()
            ));
        }

        let differing_states: Vec<_> = self
            .state_diffs
            .iter()
            .filter(|(_, d)| !d.is_empty())
            .collect();
        if !differing_states.is_empty() {
            parts.push(format!("{} states differ", differing_states.len()));
        }

        parts.join(", ")
    }

    /// Export the trace diff as a DOT graph for visualization
    ///
    /// Creates a side-by-side comparison showing:
    /// - States unique to each trace (colored differently)
    /// - Shared states with value differences highlighted
    /// - Transitions between states
    pub fn to_dot(
        &self,
        trace1: &[TraceState],
        trace2: &[TraceState],
        label1: &str,
        label2: &str,
    ) -> String {
        let mut dot = String::new();

        dot.push_str("digraph TraceDiff {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box, fontname=\"Courier\"];\n");
        dot.push_str("  edge [fontname=\"Courier\"];\n\n");

        // Add summary comment
        dot.push_str(&format!("  // Trace Diff: {}\n", self.summary()));
        dot.push_str(&format!(
            "  // {} identical, {} different, {} only in first, {} only in second\n\n",
            self.identical_states.len(),
            self.state_diffs.values().filter(|d| !d.is_empty()).count(),
            self.states_only_in_first.len(),
            self.states_only_in_second.len()
        ));

        // Build state lookup maps
        let map1: HashMap<u32, &TraceState> = trace1.iter().map(|s| (s.state_num, s)).collect();
        let map2: HashMap<u32, &TraceState> = trace2.iter().map(|s| (s.state_num, s)).collect();

        // Collect all state numbers in order
        let mut all_states: Vec<u32> = Vec::new();
        for s in trace1 {
            if !all_states.contains(&s.state_num) {
                all_states.push(s.state_num);
            }
        }
        for s in trace2 {
            if !all_states.contains(&s.state_num) {
                all_states.push(s.state_num);
            }
        }
        all_states.sort();

        // Subgraph for first trace
        let safe_label1 = label1.replace("\"", "\\\"");
        dot.push_str("  subgraph cluster_trace1 {\n");
        dot.push_str(&format!("    label=\"{}\";\n", safe_label1));
        dot.push_str("    style=filled;\n");
        dot.push_str("    color=lightblue;\n\n");

        let mut prev1: Option<u32> = None;
        for state_num in &all_states {
            if let Some(state) = map1.get(state_num) {
                let node_id = format!("t1_s{}", state_num);
                let (color, style) = self.node_style_for_state(*state_num, true);
                let label = self.format_state_label(state, *state_num, true);

                dot.push_str(&format!(
                    "    {} [label=\"{}\", color=\"{}\", style=\"{}\"];\n",
                    node_id, label, color, style
                ));

                if let Some(prev) = prev1 {
                    dot.push_str(&format!("    t1_s{} -> {};\n", prev, node_id));
                }
                prev1 = Some(*state_num);
            }
        }
        dot.push_str("  }\n\n");

        // Subgraph for second trace
        let safe_label2 = label2.replace("\"", "\\\"");
        dot.push_str("  subgraph cluster_trace2 {\n");
        dot.push_str(&format!("    label=\"{}\";\n", safe_label2));
        dot.push_str("    style=filled;\n");
        dot.push_str("    color=lightgreen;\n\n");

        let mut prev2: Option<u32> = None;
        for state_num in &all_states {
            if let Some(state) = map2.get(state_num) {
                let node_id = format!("t2_s{}", state_num);
                let (color, style) = self.node_style_for_state(*state_num, false);
                let label = self.format_state_label(state, *state_num, false);

                dot.push_str(&format!(
                    "    {} [label=\"{}\", color=\"{}\", style=\"{}\"];\n",
                    node_id, label, color, style
                ));

                if let Some(prev) = prev2 {
                    dot.push_str(&format!("    t2_s{} -> {};\n", prev, node_id));
                }
                prev2 = Some(*state_num);
            }
        }
        dot.push_str("  }\n\n");

        // Add cross-edges for state comparisons
        dot.push_str("  // State comparison edges\n");
        for state_num in &self.identical_states {
            if map1.contains_key(state_num) && map2.contains_key(state_num) {
                dot.push_str(&format!(
                    "  t1_s{} -> t2_s{} [style=dashed, color=gray, constraint=false, label=\"=\"];\n",
                    state_num, state_num
                ));
            }
        }

        // Add edges showing differences
        for (state_num, state_diff) in &self.state_diffs {
            if !state_diff.is_empty()
                && map1.contains_key(state_num)
                && map2.contains_key(state_num)
            {
                let diff_count = state_diff.value_diffs.len()
                    + state_diff.vars_only_in_first.len()
                    + state_diff.vars_only_in_second.len();
                dot.push_str(&format!(
                    "  t1_s{} -> t2_s{} [style=bold, color=red, constraint=false, label=\"{} diff(s)\"];\n",
                    state_num, state_num, diff_count
                ));
            }
        }

        // Legend
        dot.push_str("\n  // Legend\n");
        dot.push_str("  subgraph cluster_legend {\n");
        dot.push_str("    label=\"Legend\";\n");
        dot.push_str("    style=dotted;\n");
        dot.push_str("    legend_identical [label=\"Identical\", color=black, style=solid];\n");
        dot.push_str(
            "    legend_different [label=\"Has Differences\", color=red, style=bold, penwidth=2];\n",
        );
        dot.push_str(
            "    legend_unique [label=\"Unique to Trace\", color=orange, style=dashed];\n",
        );
        dot.push_str("  }\n");

        dot.push_str("}\n");
        dot
    }

    /// Get node style based on whether state is identical, different, or unique
    fn node_style_for_state(
        &self,
        state_num: u32,
        is_first_trace: bool,
    ) -> (&'static str, &'static str) {
        if self.identical_states.contains(&state_num) {
            ("black", "solid")
        } else if self.state_diffs.contains_key(&state_num) {
            ("red", "bold")
        } else if (is_first_trace && self.states_only_in_first.contains(&state_num))
            || (!is_first_trace && self.states_only_in_second.contains(&state_num))
        {
            ("orange", "dashed")
        } else {
            ("black", "solid")
        }
    }

    /// Format state label for DOT output, highlighting differences
    fn format_state_label(
        &self,
        state: &TraceState,
        state_num: u32,
        is_first_trace: bool,
    ) -> String {
        let mut parts = vec![format!("State {}", state_num)];

        if let Some(action) = &state.action {
            parts.push(format!("Action: {}", action));
        }
        parts.push("---".to_string());

        // Show variables, marking those that differ
        let diff_vars: std::collections::HashSet<String> = self
            .state_diffs
            .get(&state_num)
            .map(|d| {
                d.value_diffs
                    .keys()
                    .chain(d.vars_only_in_first.keys())
                    .chain(d.vars_only_in_second.keys())
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        let mut vars: Vec<_> = state.variables.iter().collect();
        vars.sort_by_key(|(k, _)| *k);

        for (var, val) in vars.iter().take(5) {
            let marker = if diff_vars.contains(*var) { "*" } else { "" };
            parts.push(format!("{}{} = {}", marker, var, val));
        }

        if state.variables.len() > 5 {
            parts.push(format!("... ({} more)", state.variables.len() - 5));
        }

        // Show unique variables for this trace
        if let Some(diff) = self.state_diffs.get(&state_num) {
            let unique = if is_first_trace {
                &diff.vars_only_in_first
            } else {
                &diff.vars_only_in_second
            };
            if !unique.is_empty() {
                parts.push(format!("(+{} unique vars)", unique.len()));
            }
        }

        parts.join("\\n")
    }

    /// Export the trace diff as a Mermaid diagram for markdown embedding
    ///
    /// Creates a side-by-side comparison using Mermaid's flowchart syntax.
    /// This is useful for embedding in GitHub markdown, documentation, etc.
    pub fn to_mermaid(
        &self,
        trace1: &[TraceState],
        trace2: &[TraceState],
        label1: &str,
        label2: &str,
    ) -> String {
        let mut mermaid = String::new();

        mermaid.push_str("flowchart TB\n");
        mermaid.push_str(&format!("    %% Trace Diff: {}\n", self.summary()));

        // Define subgraphs for each trace
        let safe_label1 = label1.replace("\"", "'");
        let safe_label2 = label2.replace("\"", "'");

        mermaid.push_str(&format!("    subgraph T1[\"{}\"]\n", safe_label1));
        mermaid.push_str("        direction TB\n");

        // Build nodes for trace 1
        let mut prev1: Option<u32> = None;
        for state in trace1 {
            let node_id = format!("t1_s{}", state.state_num);
            let (style_class, shape_start, shape_end) =
                self.mermaid_node_style(state.state_num, true);
            let label = self.format_mermaid_label(state, state.state_num, true);

            mermaid.push_str(&format!(
                "        {}{}\"{}\"{}:::{}\n",
                node_id, shape_start, label, shape_end, style_class
            ));

            if let Some(prev) = prev1 {
                mermaid.push_str(&format!("        t1_s{} --> {}\n", prev, node_id));
            }
            prev1 = Some(state.state_num);
        }
        mermaid.push_str("    end\n\n");

        mermaid.push_str(&format!("    subgraph T2[\"{}\"]\n", safe_label2));
        mermaid.push_str("        direction TB\n");

        // Build nodes for trace 2
        let mut prev2: Option<u32> = None;
        for state in trace2 {
            let node_id = format!("t2_s{}", state.state_num);
            let (style_class, shape_start, shape_end) =
                self.mermaid_node_style(state.state_num, false);
            let label = self.format_mermaid_label(state, state.state_num, false);

            mermaid.push_str(&format!(
                "        {}{}\"{}\"{}:::{}\n",
                node_id, shape_start, label, shape_end, style_class
            ));

            if let Some(prev) = prev2 {
                mermaid.push_str(&format!("        t2_s{} --> {}\n", prev, node_id));
            }
            prev2 = Some(state.state_num);
        }
        mermaid.push_str("    end\n\n");

        // Add cross-links for state comparisons
        mermaid.push_str("    %% State comparison links\n");
        for state_num in &self.identical_states {
            mermaid.push_str(&format!(
                "    t1_s{} -.\"==\".- t2_s{}\n",
                state_num, state_num
            ));
        }

        // Add links showing differences
        for (state_num, state_diff) in &self.state_diffs {
            if !state_diff.is_empty() {
                let diff_count = state_diff.value_diffs.len()
                    + state_diff.vars_only_in_first.len()
                    + state_diff.vars_only_in_second.len();
                mermaid.push_str(&format!(
                    "    t1_s{} -.\"{} diff(s)\".- t2_s{}\n",
                    state_num, diff_count, state_num
                ));
            }
        }

        // Add style definitions
        mermaid.push_str("\n    %% Style definitions\n");
        mermaid.push_str("    classDef identical fill:#e8f5e9,stroke:#4caf50,stroke-width:2px\n");
        mermaid.push_str("    classDef different fill:#ffebee,stroke:#f44336,stroke-width:3px\n");
        mermaid.push_str("    classDef unique fill:#fff3e0,stroke:#ff9800,stroke-width:2px,stroke-dasharray: 5 5\n");

        mermaid
    }

    /// Get Mermaid node style based on whether state is identical, different, or unique
    fn mermaid_node_style(
        &self,
        state_num: u32,
        is_first_trace: bool,
    ) -> (&'static str, &'static str, &'static str) {
        if self.identical_states.contains(&state_num) {
            ("identical", "[", "]") // Rectangle
        } else if self.state_diffs.contains_key(&state_num) {
            ("different", "{{", "}}") // Hexagon for emphasis
        } else if (is_first_trace && self.states_only_in_first.contains(&state_num))
            || (!is_first_trace && self.states_only_in_second.contains(&state_num))
        {
            ("unique", "([", "])") // Stadium shape
        } else {
            ("identical", "[", "]")
        }
    }

    /// Format state label for Mermaid output (shorter than DOT)
    fn format_mermaid_label(
        &self,
        state: &TraceState,
        state_num: u32,
        is_first_trace: bool,
    ) -> String {
        let mut parts = vec![format!("S{}", state_num)];

        if let Some(action) = &state.action {
            let short_action = if action.len() > 20 {
                format!("{}...", &action[..17])
            } else {
                action.clone()
            };
            parts.push(short_action);
        }

        // Show key variable changes for different states
        if let Some(diff) = self.state_diffs.get(&state_num) {
            if !diff.is_empty() {
                let diff_count = diff.value_diffs.len()
                    + diff.vars_only_in_first.len()
                    + diff.vars_only_in_second.len();
                parts.push(format!("({} vars differ)", diff_count));
            }
        }

        // Show unique indicator
        if (is_first_trace && self.states_only_in_first.contains(&state_num))
            || (!is_first_trace && self.states_only_in_second.contains(&state_num))
        {
            parts.push("(unique)".to_string());
        }

        // Escape special mermaid characters
        parts
            .join("<br/>")
            .replace("\"", "'")
            .replace("[", "(")
            .replace("]", ")")
    }

    /// Export the trace diff as a standalone HTML page with embedded visualization
    ///
    /// Generates a complete HTML document that can be opened in a browser.
    /// Includes both a DOT-based SVG (if graphviz is available) and a Mermaid diagram.
    pub fn to_html(
        &self,
        trace1: &[TraceState],
        trace2: &[TraceState],
        label1: &str,
        label2: &str,
        title: Option<&str>,
    ) -> String {
        let title = title.unwrap_or("Trace Diff Visualization");
        let mermaid_code = self.to_mermaid(trace1, trace2, label1, label2);
        let dot_code = self.to_dot(trace1, trace2, label1, label2);
        let download_buttons = html_download_buttons(&mermaid_code, Some(&dot_code));
        let summary = self.summary();

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
            border-bottom: 2px solid #4caf50;
            padding-bottom: 10px;
        }}
        .summary {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary h2 {{
            margin-top: 0;
            color: #666;
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
        .stat.diff {{
            background: #ffebee;
            border-left-color: #f44336;
        }}
        .stat.unique {{
            background: #fff3e0;
            border-left-color: #ff9800;
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
        .legend-identical {{ background: #e8f5e9; border: 2px solid #4caf50; }}
        .legend-different {{ background: #ffebee; border: 3px solid #f44336; }}
        .legend-unique {{ background: #fff3e0; border: 2px dashed #ff9800; }}
        {download_css}
        footer {{
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p>{summary}</p>
        <div class="stats">
            <div class="stat">
                <strong>{identical}</strong> identical states
            </div>
            <div class="stat diff">
                <strong>{different}</strong> states with differences
            </div>
            <div class="stat unique">
                <strong>{only_first}</strong> only in {label1}
            </div>
            <div class="stat unique">
                <strong>{only_second}</strong> only in {label2}
            </div>
        </div>
    </div>

    <div class="diagram">
        {download_buttons}
        <div class="mermaid">
{mermaid}
        </div>

        <div class="legend">
            <span class="legend-item">
                <span class="legend-box legend-identical"></span>
                Identical state
            </span>
            <span class="legend-item">
                <span class="legend-box legend-different"></span>
                State with differences
            </span>
            <span class="legend-item">
                <span class="legend-box legend-unique"></span>
                Unique to one trace
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
            summary = summary,
            identical = self.identical_states.len(),
            different = self.state_diffs.values().filter(|d| !d.is_empty()).count(),
            only_first = self.states_only_in_first.len(),
            only_second = self.states_only_in_second.len(),
            label1 = label1,
            label2 = label2,
            mermaid = mermaid_code,
            download_buttons = download_buttons,
            download_css = DOWNLOAD_BUTTON_CSS,
        )
    }
}

/// Differences between two counterexamples
#[derive(Debug, Clone, Default)]
pub struct CounterexampleDiff {
    /// Variables that exist only in the first counterexample
    pub only_in_first: HashMap<String, CounterexampleValue>,
    /// Variables that exist only in the second counterexample
    pub only_in_second: HashMap<String, CounterexampleValue>,
    /// Variables present in both but with different values (first, second)
    pub value_differences: HashMap<String, (CounterexampleValue, CounterexampleValue)>,
    /// Difference in number of trace states
    pub trace_length_diff: Option<(usize, usize)>,
    /// Failed checks only in first
    pub checks_only_in_first: Vec<FailedCheck>,
    /// Failed checks only in second
    pub checks_only_in_second: Vec<FailedCheck>,
    /// State-level trace differences (when traces are compared in detail)
    pub trace_diff: Option<TraceDiff>,
}

impl CounterexampleDiff {
    /// Check if counterexamples are semantically equivalent
    pub fn is_equivalent(&self) -> bool {
        self.only_in_first.is_empty()
            && self.only_in_second.is_empty()
            && self.value_differences.is_empty()
            && self.trace_length_diff.is_none()
            && self.checks_only_in_first.is_empty()
            && self.checks_only_in_second.is_empty()
            && self.trace_diff.as_ref().is_none_or(|t| t.is_equivalent())
    }

    /// Get a summary of the differences
    pub fn summary(&self) -> String {
        if self.is_equivalent() {
            return "Counterexamples are equivalent".to_string();
        }

        let mut parts = Vec::new();

        if !self.only_in_first.is_empty() {
            parts.push(format!(
                "{} variables only in first",
                self.only_in_first.len()
            ));
        }

        if !self.only_in_second.is_empty() {
            parts.push(format!(
                "{} variables only in second",
                self.only_in_second.len()
            ));
        }

        if !self.value_differences.is_empty() {
            parts.push(format!("{} variables differ", self.value_differences.len()));
        }

        if let Some((first_len, second_len)) = self.trace_length_diff {
            parts.push(format!(
                "trace lengths differ ({} vs {})",
                first_len, second_len
            ));
        }

        if !self.checks_only_in_first.is_empty() || !self.checks_only_in_second.is_empty() {
            parts.push(format!(
                "failed checks differ (+{}, -{})",
                self.checks_only_in_second.len(),
                self.checks_only_in_first.len()
            ));
        }

        if let Some(ref trace_diff) = self.trace_diff {
            if !trace_diff.is_equivalent() {
                parts.push(trace_diff.summary());
            }
        }

        parts.join(", ")
    }
}

impl StructuredCounterexample {
    /// Compare this counterexample with another and return the differences
    pub fn diff(&self, other: &StructuredCounterexample) -> CounterexampleDiff {
        let mut diff = CounterexampleDiff::default();

        // Compare witness values
        for (var, value) in &self.witness {
            match other.witness.get(var) {
                Some(other_value) if value != other_value => {
                    diff.value_differences
                        .insert(var.clone(), (value.clone(), other_value.clone()));
                }
                None => {
                    diff.only_in_first.insert(var.clone(), value.clone());
                }
                _ => {} // Equal
            }
        }

        for (var, value) in &other.witness {
            if !self.witness.contains_key(var) {
                diff.only_in_second.insert(var.clone(), value.clone());
            }
        }

        // Compare trace lengths
        if self.trace.len() != other.trace.len() {
            diff.trace_length_diff = Some((self.trace.len(), other.trace.len()));
        }

        // Compare failed checks
        for check in &self.failed_checks {
            if !other
                .failed_checks
                .iter()
                .any(|c| c.check_id == check.check_id)
            {
                diff.checks_only_in_first.push(check.clone());
            }
        }

        for check in &other.failed_checks {
            if !self
                .failed_checks
                .iter()
                .any(|c| c.check_id == check.check_id)
            {
                diff.checks_only_in_second.push(check.clone());
            }
        }

        diff
    }

    /// Check if this counterexample is semantically equivalent to another
    pub fn is_equivalent_to(&self, other: &StructuredCounterexample) -> bool {
        self.diff(other).is_equivalent()
    }

    /// Check if this counterexample is semantically equivalent to another,
    /// treating sets as equal regardless of element order (e.g., {1,2} == {2,1})
    pub fn is_semantically_equivalent_to(&self, other: &StructuredCounterexample) -> bool {
        self.diff_semantic(other).is_equivalent()
    }

    /// Compare this counterexample with another using semantic equality for values
    /// Sets, records, and functions are compared regardless of element/field order
    pub fn diff_semantic(&self, other: &StructuredCounterexample) -> CounterexampleDiff {
        let mut diff = CounterexampleDiff::default();

        // Compare witness values using semantic equality
        for (var, value) in &self.witness {
            match other.witness.get(var) {
                Some(other_value) if !value.semantically_equal(other_value) => {
                    diff.value_differences
                        .insert(var.clone(), (value.clone(), other_value.clone()));
                }
                None => {
                    diff.only_in_first.insert(var.clone(), value.clone());
                }
                _ => {} // Semantically equal
            }
        }

        for (var, value) in &other.witness {
            if !self.witness.contains_key(var) {
                diff.only_in_second.insert(var.clone(), value.clone());
            }
        }

        // Compare trace lengths
        if self.trace.len() != other.trace.len() {
            diff.trace_length_diff = Some((self.trace.len(), other.trace.len()));
        }

        // Compare failed checks
        for check in &self.failed_checks {
            if !other
                .failed_checks
                .iter()
                .any(|c| c.check_id == check.check_id)
            {
                diff.checks_only_in_first.push(check.clone());
            }
        }

        for check in &other.failed_checks {
            if !self
                .failed_checks
                .iter()
                .any(|c| c.check_id == check.check_id)
            {
                diff.checks_only_in_second.push(check.clone());
            }
        }

        diff
    }

    /// Compare this counterexample with another, including detailed state-level trace comparison
    pub fn diff_detailed(&self, other: &StructuredCounterexample) -> CounterexampleDiff {
        let mut diff = self.diff(other);
        diff.trace_diff = Some(self.diff_traces(other));
        diff
    }

    /// Compare traces at the state level, aligning by state_num
    pub fn diff_traces(&self, other: &StructuredCounterexample) -> TraceDiff {
        let mut trace_diff = TraceDiff::default();

        // Index states by state_num for alignment
        let self_states: HashMap<u32, &TraceState> =
            self.trace.iter().map(|s| (s.state_num, s)).collect();
        let other_states: HashMap<u32, &TraceState> =
            other.trace.iter().map(|s| (s.state_num, s)).collect();

        // Find states only in first
        for state_num in self_states.keys() {
            if !other_states.contains_key(state_num) {
                trace_diff.states_only_in_first.push(*state_num);
            }
        }
        trace_diff.states_only_in_first.sort();

        // Find states only in second
        for state_num in other_states.keys() {
            if !self_states.contains_key(state_num) {
                trace_diff.states_only_in_second.push(*state_num);
            }
        }
        trace_diff.states_only_in_second.sort();

        // Compare states that exist in both
        for (state_num, self_state) in &self_states {
            if let Some(other_state) = other_states.get(state_num) {
                let state_diff = Self::diff_states(self_state, other_state);
                if state_diff.is_empty() {
                    trace_diff.identical_states.push(*state_num);
                } else {
                    trace_diff.state_diffs.insert(*state_num, state_diff);
                }
            }
        }
        trace_diff.identical_states.sort();

        trace_diff
    }

    /// Compare two individual states
    pub fn diff_states(first: &TraceState, second: &TraceState) -> StateLevelDiff {
        let mut diff = StateLevelDiff::default();

        // Compare variables
        for (var, value) in &first.variables {
            match second.variables.get(var) {
                Some(other_value) if value != other_value => {
                    diff.value_diffs
                        .insert(var.clone(), (value.clone(), other_value.clone()));
                }
                None => {
                    diff.vars_only_in_first.insert(var.clone(), value.clone());
                }
                _ => {} // Equal
            }
        }

        for (var, value) in &second.variables {
            if !first.variables.contains_key(var) {
                diff.vars_only_in_second.insert(var.clone(), value.clone());
            }
        }

        // Compare actions
        if first.action != second.action {
            diff.action_diff = Some((first.action.clone(), second.action.clone()));
        }

        diff
    }

    /// Align two traces by state number and return paired states
    /// Returns: (aligned_pairs, only_in_first, only_in_second)
    /// where aligned_pairs contains (state_num, Option<first_state>, Option<second_state>)
    pub fn align_traces<'a>(
        &'a self,
        other: &'a StructuredCounterexample,
    ) -> Vec<(u32, Option<&'a TraceState>, Option<&'a TraceState>)> {
        let self_states: HashMap<u32, &TraceState> =
            self.trace.iter().map(|s| (s.state_num, s)).collect();
        let other_states: HashMap<u32, &TraceState> =
            other.trace.iter().map(|s| (s.state_num, s)).collect();

        // Collect all state numbers
        let mut all_state_nums: Vec<u32> = self_states
            .keys()
            .chain(other_states.keys())
            .copied()
            .collect();
        all_state_nums.sort();
        all_state_nums.dedup();

        all_state_nums
            .into_iter()
            .map(|num| {
                (
                    num,
                    self_states.get(&num).copied(),
                    other_states.get(&num).copied(),
                )
            })
            .collect()
    }

    /// Format aligned traces showing differences side by side
    pub fn format_aligned_traces(&self, other: &StructuredCounterexample) -> String {
        let aligned = self.align_traces(other);
        let mut output = String::new();

        for (state_num, first, second) in aligned {
            output.push_str(&format!("\n=== State {} ===\n", state_num));

            match (first, second) {
                (Some(f), Some(s)) => {
                    let state_diff = Self::diff_states(f, s);
                    if state_diff.is_empty() {
                        output.push_str("  (identical)\n");
                        // Show action if present
                        if let Some(ref action) = f.action {
                            output.push_str(&format!("  Action: {}\n", action));
                        }
                    } else {
                        // Show action difference
                        if let Some((ref a1, ref a2)) = state_diff.action_diff {
                            output.push_str(&format!(
                                "  Action: {:?} vs {:?}\n",
                                a1.as_deref().unwrap_or("(none)"),
                                a2.as_deref().unwrap_or("(none)")
                            ));
                        } else if let Some(ref action) = f.action {
                            output.push_str(&format!("  Action: {}\n", action));
                        }

                        // Show variable differences
                        if !state_diff.value_diffs.is_empty() {
                            output.push_str("  Value differences:\n");
                            let mut diffs: Vec<_> = state_diff.value_diffs.iter().collect();
                            diffs.sort_by(|a, b| a.0.cmp(b.0));
                            for (var, (v1, v2)) in diffs {
                                output.push_str(&format!("    {} : {} vs {}\n", var, v1, v2));
                            }
                        }

                        if !state_diff.vars_only_in_first.is_empty() {
                            output.push_str("  Only in first:\n");
                            let mut vars: Vec<_> = state_diff.vars_only_in_first.iter().collect();
                            vars.sort_by(|a, b| a.0.cmp(b.0));
                            for (var, val) in vars {
                                output.push_str(&format!("    {} = {}\n", var, val));
                            }
                        }

                        if !state_diff.vars_only_in_second.is_empty() {
                            output.push_str("  Only in second:\n");
                            let mut vars: Vec<_> = state_diff.vars_only_in_second.iter().collect();
                            vars.sort_by(|a, b| a.0.cmp(b.0));
                            for (var, val) in vars {
                                output.push_str(&format!("    {} = {}\n", var, val));
                            }
                        }
                    }
                }
                (Some(f), None) => {
                    output.push_str("  (only in first trace)\n");
                    if let Some(ref action) = f.action {
                        output.push_str(&format!("  Action: {}\n", action));
                    }
                    let mut vars: Vec<_> = f.variables.iter().collect();
                    vars.sort_by(|a, b| a.0.cmp(b.0));
                    for (var, val) in vars {
                        output.push_str(&format!("    {} = {}\n", var, val));
                    }
                }
                (None, Some(s)) => {
                    output.push_str("  (only in second trace)\n");
                    if let Some(ref action) = s.action {
                        output.push_str(&format!("  Action: {}\n", action));
                    }
                    let mut vars: Vec<_> = s.variables.iter().collect();
                    vars.sort_by(|a, b| a.0.cmp(b.0));
                    for (var, val) in vars {
                        output.push_str(&format!("    {} = {}\n", var, val));
                    }
                }
                (None, None) => {
                    // Should not happen
                    output.push_str("  (missing from both - error)\n");
                }
            }
        }

        output
    }
}
