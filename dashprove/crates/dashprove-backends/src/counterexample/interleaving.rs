//! Trace interleaving analysis (swimlane diagrams)
//!
//! This module provides tools for analyzing multi-actor traces:
//! - `ActorPatternConfig`: Configuration for detecting actors in traces
//! - `TraceLane`: A single actor's lane in the swimlane diagram
//! - `TraceInterleaving`: Full swimlane representation of a trace

use super::types::{CounterexampleValue, StructuredCounterexample, TraceState};
use crate::traits::{html_download_buttons, DOWNLOAD_BUTTON_CSS};
use crate::util::mermaid_escape;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ==================== Trace Interleaving Analysis ====================

/// Configuration for actor pattern detection in interleaving analysis
#[derive(Debug, Clone, Default)]
pub struct ActorPatternConfig {
    /// Custom regex patterns for extracting actor from action strings
    /// Each pattern should have a capture group for the actor name
    pub action_patterns: Vec<String>,
    /// Custom regex patterns for extracting actor from variable names
    pub variable_patterns: Vec<String>,
    /// Whether to use default patterns in addition to custom ones
    pub use_default_patterns: bool,
}

impl ActorPatternConfig {
    /// Create a config with only default patterns
    pub fn default_patterns() -> Self {
        Self {
            action_patterns: Vec::new(),
            variable_patterns: Vec::new(),
            use_default_patterns: true,
        }
    }

    /// Create a config with custom patterns only
    pub fn custom(action_patterns: Vec<String>, variable_patterns: Vec<String>) -> Self {
        Self {
            action_patterns,
            variable_patterns,
            use_default_patterns: false,
        }
    }

    /// Create a config with both custom and default patterns
    pub fn combined(action_patterns: Vec<String>, variable_patterns: Vec<String>) -> Self {
        Self {
            action_patterns,
            variable_patterns,
            use_default_patterns: true,
        }
    }
}

/// A lane of related states within an interleaved trace (e.g., a single actor/thread)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceLane {
    /// Actor identifier inferred from actions or variable prefixes
    pub actor: String,
    /// States belonging to this actor in their original order
    pub states: Vec<TraceState>,
}

/// Result of attempting to group a trace into interleaved lanes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceInterleaving {
    /// Ordered list of discovered lanes
    pub lanes: Vec<TraceLane>,
    /// States that could not be assigned to any lane
    pub unassigned_states: Vec<TraceState>,
    /// Actor assignment per original state index (None if unassigned)
    pub assignments: Vec<Option<String>>,
    /// Original trace preserved for formatting
    pub original_trace: Vec<TraceState>,
}

impl TraceInterleaving {
    /// Fraction of states that were assigned to a lane
    pub fn coverage(&self) -> f64 {
        if self.original_trace.is_empty() {
            return 0.0;
        }
        let assigned = self.assignments.iter().filter(|a| a.is_some()).count();
        assigned as f64 / self.original_trace.len() as f64
    }

    /// Count how often the active lane changes across the trace
    pub fn lane_switches(&self) -> usize {
        let mut switches = 0usize;
        let mut last_lane: Option<&str> = None;

        for lane in self.assignments.iter().flatten() {
            if let Some(prev) = last_lane {
                if prev != lane.as_str() {
                    switches += 1;
                }
            }
            last_lane = Some(lane.as_str());
        }

        switches
    }

    /// Format a human-readable view of the interleaving lanes
    pub fn format_lane_view(&self) -> String {
        let mut output = String::new();
        let coverage_pct = self.coverage() * 100.0;

        output.push_str(&format!(
            "Interleaving coverage: {:.1}% ({} of {} states assigned)\n",
            coverage_pct,
            self.assignments.iter().filter(|a| a.is_some()).count(),
            self.original_trace.len()
        ));
        output.push_str(&format!(
            "Lane switches: {}{}\n",
            self.lane_switches(),
            if self.original_trace.len() > 1 {
                format!(" over {} transitions", self.original_trace.len() - 1)
            } else {
                String::new()
            }
        ));

        for lane in &self.lanes {
            output.push_str(&format!(
                "\nLane {} ({} states):\n",
                lane.actor,
                lane.states.len()
            ));
            for state in &lane.states {
                let action = state.action.as_deref().unwrap_or("(no action)");
                output.push_str(&format!("  [{}] {}\n", state.state_num, action));
            }
        }

        if !self.unassigned_states.is_empty() {
            output.push_str("\nUnassigned states:\n");
            for state in &self.unassigned_states {
                let action = state.action.as_deref().unwrap_or("(no action)");
                output.push_str(&format!("  [{}] {}\n", state.state_num, action));
            }
        }

        output
    }

    /// Render the interleaving as a Mermaid sequence diagram for visualization
    pub fn to_mermaid_sequence_diagram(&self) -> String {
        let mut output = String::from("sequenceDiagram\n");

        if self.original_trace.is_empty() {
            output.push_str("  Note over Trace: empty trace\n");
            return output;
        }

        // Declare participants for each lane plus an Unassigned bucket
        for lane in &self.lanes {
            let id = sanitize_mermaid_id(&lane.actor);
            output.push_str(&format!("  participant {} as {}\n", id, lane.actor));
        }
        if !self.unassigned_states.is_empty() {
            output.push_str("  participant Unassigned as Unassigned\n");
        }

        // Emit self-messages to preserve ordering on each lane
        for (idx, assignment) in self.assignments.iter().enumerate() {
            let state = &self.original_trace[idx];
            let label = state
                .action
                .as_deref()
                .filter(|s| !s.is_empty())
                .unwrap_or("State");
            match assignment {
                Some(actor) => {
                    let id = sanitize_mermaid_id(actor);
                    output.push_str(&format!("  {id}->>{id}: [{}] {label}\n", state.state_num));
                }
                None => {
                    output.push_str(&format!(
                        "  Unassigned->>Unassigned: [{}] {label}\n",
                        state.state_num
                    ));
                }
            }
        }

        if !self.unassigned_states.is_empty() {
            output.push_str("  Note over Unassigned: States without clear actor\n");
        }

        output
    }

    /// Export trace interleaving as a Mermaid flowchart diagram
    ///
    /// Creates a flowchart showing lanes as swimlanes with state nodes.
    /// Each lane is a subgraph containing its assigned states in order.
    /// Useful for embedding in GitHub markdown, documentation, etc.
    pub fn to_mermaid(&self) -> String {
        let mut mermaid = String::new();

        mermaid.push_str("flowchart TB\n");
        mermaid.push_str(&format!(
            "    %% Trace Interleaving ({} lanes, {} states, {:.0}% coverage)\n",
            self.lanes.len(),
            self.original_trace.len(),
            self.coverage() * 100.0
        ));

        if self.original_trace.is_empty() {
            mermaid.push_str("    empty[\"Empty trace\"]\n");
            return mermaid;
        }

        // Create subgraphs for each lane
        for lane in &self.lanes {
            let lane_id = sanitize_mermaid_id(&lane.actor);
            mermaid.push_str(&format!("\n    subgraph {}[\"{}\"]\n", lane_id, lane.actor));
            mermaid.push_str("        direction TB\n");

            // Add nodes for each state in this lane
            for (state_idx, state) in lane.states.iter().enumerate() {
                let node_id = format!("{}_{}", lane_id, state_idx);
                let label = Self::mermaid_state_label(state);
                mermaid.push_str(&format!(
                    "        {}[\"{}\"]\n",
                    node_id,
                    mermaid_escape(&label)
                ));

                // Connect to previous state in same lane
                if state_idx > 0 {
                    let prev_node_id = format!("{}_{}", lane_id, state_idx - 1);
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
                        mermaid.push_str(&format!("        {} --> {}\n", prev_node_id, node_id));
                    } else {
                        mermaid.push_str(&format!(
                            "        {} -->|\"{}\"| {}\n",
                            prev_node_id,
                            mermaid_escape(&edge_label),
                            node_id
                        ));
                    }
                }
            }
            mermaid.push_str("    end\n");
        }

        // Add unassigned states if present
        if !self.unassigned_states.is_empty() {
            mermaid.push_str("\n    subgraph Unassigned[\"Unassigned States\"]\n");
            mermaid.push_str("        direction TB\n");
            for (state_idx, state) in self.unassigned_states.iter().enumerate() {
                let node_id = format!("unassigned_{}", state_idx);
                let label = Self::mermaid_state_label(state);
                mermaid.push_str(&format!(
                    "        {}[\"{}\"]\n",
                    node_id,
                    mermaid_escape(&label)
                ));
            }
            mermaid.push_str("    end\n");
        }

        // Add style definitions for each lane
        mermaid.push_str("\n    %% Lane style definitions\n");
        let lane_colors = ["#e3f2fd", "#e8f5e9", "#fff3e0", "#fce4ec", "#f3e5f5"];
        let lane_borders = ["#1976d2", "#4caf50", "#ff9800", "#e91e63", "#9c27b0"];
        for (lane_idx, lane) in self.lanes.iter().enumerate() {
            let lane_id = sanitize_mermaid_id(&lane.actor);
            let color = lane_colors[lane_idx % lane_colors.len()];
            let border = lane_borders[lane_idx % lane_borders.len()];
            mermaid.push_str(&format!(
                "    style {} fill:{},stroke:{},stroke-width:2px\n",
                lane_id, color, border
            ));
        }
        if !self.unassigned_states.is_empty() {
            mermaid.push_str("    style Unassigned fill:#f5f5f5,stroke:#9e9e9e,stroke-width:1px,stroke-dasharray: 5 5\n");
        }

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

    /// Export trace interleaving as a standalone HTML page with embedded visualization
    ///
    /// Generates a complete HTML document that can be opened in a browser.
    /// Includes summary statistics, lane breakdown, and interactive Mermaid diagrams.
    pub fn to_html(&self, title: Option<&str>) -> String {
        let title = title.unwrap_or("Trace Interleaving Visualization");
        let mermaid_flowchart = self.to_mermaid();
        let mermaid_sequence = self.to_mermaid_sequence_diagram();
        let download_flowchart = html_download_buttons(&mermaid_flowchart, None);
        let download_sequence = html_download_buttons(&mermaid_sequence, None);

        // Build lane details table
        let mut lanes_html = String::from(
            "<table class=\"lanes-table\"><tr><th>Actor</th><th>States</th><th>Actions</th></tr>",
        );
        for lane in &self.lanes {
            let actions: Vec<String> = lane
                .states
                .iter()
                .filter_map(|s| s.action.as_ref())
                .filter(|a| !a.is_empty())
                .take(3)
                .cloned()
                .collect();
            let actions_display = if actions.is_empty() {
                "(no actions)".to_string()
            } else if lane.states.iter().filter_map(|s| s.action.as_ref()).count() > 3 {
                format!(
                    "{} (+{} more)",
                    actions.join(", "),
                    lane.states.len().saturating_sub(3)
                )
            } else {
                actions.join(", ")
            };
            lanes_html.push_str(&format!(
                "<tr><td><strong>{}</strong></td><td>{}</td><td><code>{}</code></td></tr>",
                lane.actor,
                lane.states.len(),
                actions_display
            ));
        }
        if !self.unassigned_states.is_empty() {
            lanes_html.push_str(&format!(
                "<tr class=\"unassigned\"><td><em>Unassigned</em></td><td>{}</td><td>-</td></tr>",
                self.unassigned_states.len()
            ));
        }
        lanes_html.push_str("</table>");

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
        .stat.lanes {{
            background: #e3f2fd;
            border-left-color: #1976d2;
        }}
        .stat.coverage {{
            background: #fff3e0;
            border-left-color: #ff9800;
        }}
        .stat.switches {{
            background: #fce4ec;
            border-left-color: #e91e63;
        }}
        .lanes-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .lanes-table th, .lanes-table td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .lanes-table th {{
            background: #f5f5f5;
        }}
        .lanes-table tr.unassigned {{
            background: #f5f5f5;
            font-style: italic;
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
            <div class="stat lanes">
                <strong>Lanes:</strong> {lane_count}
            </div>
            <div class="stat">
                <strong>Total States:</strong> {state_count}
            </div>
            <div class="stat coverage">
                <strong>Coverage:</strong> {coverage:.0}%
            </div>
            <div class="stat switches">
                <strong>Lane Switches:</strong> {switches}
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Lanes</h2>
        {lanes_html}
    </div>

    <div class="diagram">
        <h2>Trace Diagram</h2>
        <div class="diagram-tabs">
            <button class="diagram-tab active" onclick="showDiagram('flowchart')">Flowchart</button>
            <button class="diagram-tab" onclick="showDiagram('sequence')">Sequence</button>
        </div>
        <div id="flowchart" class="diagram-content active">
            {download_flowchart}
            <pre class="mermaid">
{mermaid_flowchart}
            </pre>
        </div>
        <div id="sequence" class="diagram-content">
            {download_sequence}
            <pre class="mermaid">
{mermaid_sequence}
            </pre>
        </div>
        <div class="legend">
            <strong>Legend:</strong>
            <span class="legend-item"><span class="legend-color" style="background: #e3f2fd; border-color: #1976d2;"></span> Lane 1</span>
            <span class="legend-item"><span class="legend-color" style="background: #e8f5e9; border-color: #4caf50;"></span> Lane 2</span>
            <span class="legend-item"><span class="legend-color" style="background: #fff3e0; border-color: #ff9800;"></span> Lane 3</span>
            <span class="legend-item"><span class="legend-color" style="background: #f5f5f5; border-color: #9e9e9e; border-style: dashed;"></span> Unassigned</span>
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
            lane_count = self.lanes.len(),
            state_count = self.original_trace.len(),
            coverage = self.coverage() * 100.0,
            switches = self.lane_switches(),
            lanes_html = lanes_html,
            mermaid_flowchart = mermaid_flowchart,
            mermaid_sequence = mermaid_sequence,
            download_flowchart = download_flowchart,
            download_sequence = download_sequence,
            download_css = DOWNLOAD_BUTTON_CSS,
        )
    }
}

/// Sanitize a string for use as a Mermaid participant identifier
fn sanitize_mermaid_id(actor: &str) -> String {
    let mut result = String::new();
    for ch in actor.chars() {
        if ch.is_ascii_alphanumeric() {
            result.push(ch);
        } else {
            result.push('_');
        }
    }
    if result.is_empty() {
        "actor".to_string()
    } else {
        result
    }
}

impl StructuredCounterexample {
    /// Attempt to split the trace into interleaved lanes based on action or variable prefixes
    pub fn detect_interleavings(&self) -> TraceInterleaving {
        self.detect_interleavings_with_config(&ActorPatternConfig::default_patterns())
    }

    /// Attempt to split the trace into interleaved lanes using custom actor patterns
    pub fn detect_interleavings_with_config(
        &self,
        config: &ActorPatternConfig,
    ) -> TraceInterleaving {
        let mut lanes: Vec<TraceLane> = Vec::new();
        let mut lane_index: HashMap<String, usize> = HashMap::new();
        let mut assignments = Vec::with_capacity(self.trace.len());
        let mut unassigned = Vec::new();

        for state in &self.trace {
            if let Some(actor) = Self::extract_actor_from_state_with_config(state, config) {
                let idx = *lane_index.entry(actor.clone()).or_insert_with(|| {
                    lanes.push(TraceLane {
                        actor: actor.clone(),
                        states: Vec::new(),
                    });
                    lanes.len() - 1
                });
                lanes[idx].states.push(state.clone());
                assignments.push(Some(lanes[idx].actor.clone()));
            } else {
                assignments.push(None);
                unassigned.push(state.clone());
            }
        }

        TraceInterleaving {
            lanes,
            unassigned_states: unassigned,
            assignments,
            original_trace: self.trace.clone(),
        }
    }

    /// Extract an actor identifier from a trace state using action or variable prefixes
    #[allow(dead_code)]
    fn extract_actor_from_state(state: &TraceState) -> Option<String> {
        Self::extract_actor_from_state_with_config(state, &ActorPatternConfig::default_patterns())
    }

    /// Extract an actor identifier from a trace state using configurable patterns
    fn extract_actor_from_state_with_config(
        state: &TraceState,
        config: &ActorPatternConfig,
    ) -> Option<String> {
        if let Some(actor) = state
            .action
            .as_deref()
            .and_then(|a| Self::extract_actor_from_action_with_config(a, config))
        {
            return Some(actor);
        }

        Self::extract_actor_from_variables_with_config(&state.variables, config)
    }

    /// Default action patterns for actor extraction
    fn default_action_patterns() -> Vec<&'static str> {
        vec![
            r"^\s*\[?([A-Za-z][A-Za-z0-9_]*)\]?\s*[:/]",
            r"^\s*([A-Za-z][A-Za-z0-9_]*)::",
            r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*->",
            r"^\s*([A-Za-z][A-Za-z0-9_]*)\.",
        ]
    }

    /// Extract actor from the action description (e.g., "Proc1: step", "workerA/Send")
    #[allow(dead_code)]
    fn extract_actor_from_action(action: &str) -> Option<String> {
        Self::extract_actor_from_action_with_config(action, &ActorPatternConfig::default_patterns())
    }

    /// Extract actor from action using configurable patterns
    fn extract_actor_from_action_with_config(
        action: &str,
        config: &ActorPatternConfig,
    ) -> Option<String> {
        // Try custom patterns first
        for pattern in &config.action_patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                if let Some(caps) = re.captures(action) {
                    if let Some(mat) = caps.get(1) {
                        return Some(mat.as_str().to_string());
                    }
                }
            }
        }

        // Try default patterns if enabled
        if config.use_default_patterns {
            for pattern in Self::default_action_patterns() {
                if let Ok(re) = regex::Regex::new(pattern) {
                    if let Some(caps) = re.captures(action) {
                        if let Some(mat) = caps.get(1) {
                            return Some(mat.as_str().to_string());
                        }
                    }
                }
            }
        }

        None
    }

    /// Extract actor from variable prefixes (e.g., "proc1.x", "threadA::y")
    #[allow(dead_code)]
    fn extract_actor_from_variables(
        variables: &HashMap<String, CounterexampleValue>,
    ) -> Option<String> {
        Self::extract_actor_from_variables_with_config(
            variables,
            &ActorPatternConfig::default_patterns(),
        )
    }

    /// Extract actor from variable prefixes using configurable patterns
    fn extract_actor_from_variables_with_config(
        variables: &HashMap<String, CounterexampleValue>,
        config: &ActorPatternConfig,
    ) -> Option<String> {
        let mut counts: HashMap<String, usize> = HashMap::new();

        for var in variables.keys() {
            // Try custom variable patterns first
            for pattern in &config.variable_patterns {
                if let Ok(re) = regex::Regex::new(pattern) {
                    if let Some(caps) = re.captures(var) {
                        if let Some(mat) = caps.get(1) {
                            *counts.entry(mat.as_str().to_string()).or_insert(0) += 1;
                        }
                    }
                }
            }

            // Try default patterns if enabled and no custom match found
            if config.use_default_patterns
                && !config.variable_patterns.iter().any(|p| {
                    regex::Regex::new(p)
                        .ok()
                        .and_then(|re| re.captures(var))
                        .is_some()
                })
            {
                if let Some(prefix) = var.split('.').next() {
                    if prefix.len() >= 2
                        && prefix
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || c == '_')
                    {
                        *counts.entry(prefix.to_string()).or_insert(0) += 1;
                        continue;
                    }
                }

                if let Some((prefix, _)) = var.split_once("::") {
                    if prefix.len() >= 2
                        && prefix
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || c == '_')
                    {
                        *counts.entry(prefix.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }

        counts
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)))
            .map(|(prefix, _)| prefix)
    }
}
