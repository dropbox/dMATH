//! Table formatting for MultiTraceAlignment

use super::types::MultiTraceAlignment;

impl MultiTraceAlignment {
    /// Format the alignment as a human-readable table
    pub fn format_table(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "Multi-Trace Alignment ({} traces, {} states)\n",
            self.trace_count(),
            self.rows.len()
        ));
        output.push_str(&format!("Traces: {}\n\n", self.trace_labels.join(", ")));

        for row in &self.rows {
            output.push_str(&format!("=== State {} ===\n", row.state_num));

            // Collect all variables across all traces at this state
            let mut all_vars: std::collections::HashSet<String> = std::collections::HashSet::new();
            for state in row.states.iter().flatten() {
                all_vars.extend(state.variables.keys().cloned());
            }
            let mut sorted_vars: Vec<_> = all_vars.into_iter().collect();
            sorted_vars.sort();

            // Check presence in each trace
            let presence: Vec<bool> = row.states.iter().map(|s| s.is_some()).collect();
            let all_present = presence.iter().all(|&p| p);

            if !all_present {
                output.push_str("  Present in: ");
                let present_labels: Vec<_> = self
                    .trace_labels
                    .iter()
                    .zip(presence.iter())
                    .filter(|(_, &p)| p)
                    .map(|(l, _)| l.as_str())
                    .collect();
                output.push_str(&present_labels.join(", "));
                output.push('\n');
            }

            // Show variables
            for var in &sorted_vars {
                let values: Vec<_> = row
                    .states
                    .iter()
                    .map(|s| {
                        s.as_ref()
                            .and_then(|st| st.variables.get(var))
                            .map(|v| format!("{}", v))
                            .unwrap_or_else(|| "-".to_string())
                    })
                    .collect();

                // Check if all values are the same
                let first = &values[0];
                let all_same = values.iter().all(|v| v == first);

                if all_same {
                    output.push_str(&format!("  {}: {}\n", var, first));
                } else {
                    output.push_str(&format!("  {} [differs]:\n", var));
                    for (i, val) in values.iter().enumerate() {
                        output.push_str(&format!("    {}: {}\n", self.trace_labels[i], val));
                    }
                }
            }
            output.push('\n');
        }

        // Summary of divergence points
        if !self.divergence_points.is_empty() {
            output.push_str("=== Divergence Summary ===\n");
            for dp in &self.divergence_points {
                output.push_str(&format!(
                    "  State {}, var '{}': ",
                    dp.state_num, dp.variable
                ));
                let vals: Vec<_> = dp
                    .values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        format!(
                            "{}={}",
                            self.trace_labels[i],
                            v.as_ref()
                                .map(|val| format!("{}", val))
                                .unwrap_or_else(|| "-".to_string())
                        )
                    })
                    .collect();
                output.push_str(&vals.join(", "));
                output.push('\n');
            }
        }

        output
    }
}
