//! TLC state trace parsing

use crate::traits::{FailedCheck, StructuredCounterexample, TraceState};
use regex::Regex;

use super::values::parse_tla_value;

/// Parse TLC state trace into structured counterexample
pub fn parse_trace(output: &str) -> StructuredCounterexample {
    let mut ce = StructuredCounterexample::new();
    let mut current_state: Option<TraceState> = None;
    let mut raw_lines = Vec::new();
    let mut in_trace = false;

    // Regex to match state header: "State N: <action>"
    let state_re = Regex::new(r"^State\s+(\d+):\s*(.*)$").expect("Invalid state regex");
    // Regex to match variable assignment: "var = value"
    let var_re = Regex::new(r"^(\w+)\s*=\s*(.+)$").expect("Invalid var regex");

    // Look for violated invariant/property info
    for line in output.lines() {
        if line.contains("Invariant") && line.contains("is violated") {
            // Extract invariant name
            if let Some(start) = line.find("Invariant ") {
                let rest = &line[start + 10..];
                if let Some(end) = rest.find(" is violated") {
                    let inv_name = &rest[..end];
                    ce.failed_checks.push(FailedCheck {
                        check_id: inv_name.to_string(),
                        description: format!("Invariant {} is violated", inv_name),
                        location: None,
                        function: None,
                    });
                }
            }
        }
        if line.contains("Temporal properties were violated") {
            ce.failed_checks.push(FailedCheck {
                check_id: "temporal".to_string(),
                description: "Temporal properties were violated".to_string(),
                location: None,
                function: None,
            });
        }
    }

    // Parse state trace
    let mut consecutive_empty_lines = 0;
    for line in output.lines() {
        // Detect start of state trace
        if line.contains("State ") && line.contains(":") {
            in_trace = true;
            consecutive_empty_lines = 0;
        }

        if !in_trace {
            continue;
        }

        let trimmed = line.trim();

        // Track consecutive empty lines - two in a row ends the trace
        if trimmed.is_empty() {
            consecutive_empty_lines += 1;
            if consecutive_empty_lines >= 2 && !ce.trace.is_empty() {
                break;
            }
            raw_lines.push(line.to_string());
            continue;
        } else {
            consecutive_empty_lines = 0;
        }

        // End trace if we hit summary lines (e.g., "5 states generated...")
        if trimmed.contains("states generated") || trimmed.contains("Finished in") {
            break;
        }

        raw_lines.push(line.to_string());

        // Check if this is a state header line
        if let Some(caps) = state_re.captures(trimmed) {
            // Save previous state if any
            if let Some(state) = current_state.take() {
                ce.trace.push(state);
            }

            let state_num: u32 = caps.get(1).unwrap().as_str().parse().unwrap_or(0);
            let action_text = caps.get(2).map(|m| m.as_str().trim().to_string());

            let mut new_state = TraceState::new(state_num);
            // Parse action from angle brackets: <Initial predicate> or <Next line...>
            if let Some(ref action) = action_text {
                if action.starts_with('<') && action.ends_with('>') {
                    new_state.action = Some(action[1..action.len() - 1].to_string());
                } else if !action.is_empty() {
                    new_state.action = Some(action.clone());
                }
            }
            current_state = Some(new_state);
        } else if let Some(ref mut state) = current_state {
            // Try to parse as variable assignment
            if let Some(caps) = var_re.captures(trimmed) {
                let var_name = caps.get(1).unwrap().as_str().to_string();
                let var_value = caps.get(2).unwrap().as_str().trim().to_string();
                let value = parse_tla_value(&var_value);
                state.variables.insert(var_name, value);
            }
        }
    }

    // Don't forget the last state
    if let Some(state) = current_state {
        ce.trace.push(state);
    }

    // Store raw trace if we captured anything
    if !raw_lines.is_empty() {
        ce.raw = Some(raw_lines.join("\n"));
    }

    ce
}
