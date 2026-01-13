//! Counterexample trace minimization
//!
//! This module provides tools for minimizing counterexample traces
//! by removing redundant states.

use super::cluster::CounterexampleStats;
use super::types::{StructuredCounterexample, TraceState};
use std::collections::HashMap;

// ==================== Counterexample Minimization ====================

impl StructuredCounterexample {
    /// Create a minimized version of this counterexample
    /// Removes redundant information while preserving the essential failure
    pub fn minimize(&self) -> Self {
        let mut minimized = self.clone();

        // Remove constant variables from trace (they don't contribute to understanding)
        let changing = self.changing_variables();
        if !changing.is_empty() && !self.trace.is_empty() {
            minimized.trace = self
                .trace
                .iter()
                .map(|state| {
                    let mut filtered_vars = HashMap::new();
                    for (var, value) in &state.variables {
                        // Keep only changing variables or variables in witness
                        if changing.contains(var) || self.witness.contains_key(var) {
                            filtered_vars.insert(var.clone(), value.clone());
                        }
                    }
                    TraceState {
                        state_num: state.state_num,
                        action: state.action.clone(),
                        variables: filtered_vars,
                    }
                })
                .collect();
        }

        // Remove unchanged intermediate states
        minimized.trace = Self::remove_unchanged_states(&minimized.trace);

        // Remove raw counterexample text if structured data is available
        if minimized.has_structured_data() {
            minimized.raw = None;
        }

        minimized.minimized = true;
        minimized
    }

    /// Remove states that have no changes from previous state
    fn remove_unchanged_states(trace: &[TraceState]) -> Vec<TraceState> {
        if trace.is_empty() {
            return Vec::new();
        }

        let mut result = vec![trace[0].clone()]; // Always keep initial state

        for i in 1..trace.len() {
            let prev = &trace[i - 1];
            let curr = &trace[i];

            // Check if anything changed
            let has_changes = curr
                .variables
                .iter()
                .any(|(var, value)| prev.variables.get(var) != Some(value))
                || prev
                    .variables
                    .keys()
                    .any(|var| !curr.variables.contains_key(var));

            if has_changes {
                result.push(curr.clone());
            }
        }

        // Always include the last state if it's different from what we have
        if let Some(last) = trace.last() {
            if result.last().map(|s| s.state_num) != Some(last.state_num) {
                result.push(last.clone());
            }
        }

        result
    }

    /// Minimize trace to only the specified number of states
    /// Keeps first, last, and evenly distributed intermediate states
    pub fn minimize_trace_length(&self, max_states: usize) -> Self {
        if self.trace.len() <= max_states || max_states < 2 {
            return self.clone();
        }

        let mut minimized = self.clone();
        let n = self.trace.len();

        // Calculate which indices to keep
        let mut keep_indices = vec![0]; // Always keep first

        if max_states > 2 {
            let step = (n - 1) as f64 / (max_states - 1) as f64;
            for i in 1..(max_states - 1) {
                let idx = (i as f64 * step).round() as usize;
                if !keep_indices.contains(&idx) {
                    keep_indices.push(idx);
                }
            }
        }

        keep_indices.push(n - 1); // Always keep last

        minimized.trace = keep_indices
            .into_iter()
            .map(|i| self.trace[i].clone())
            .collect();

        minimized.minimized = true;
        minimized
    }

    /// Keep only specified variables in witness and trace
    pub fn keep_only_variables(&self, variables: &[String]) -> Self {
        let var_set: std::collections::HashSet<_> = variables.iter().collect();

        let mut minimized = self.clone();

        // Filter witness
        minimized.witness = self
            .witness
            .iter()
            .filter(|(k, _)| var_set.contains(k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // Filter trace
        minimized.trace = self
            .trace
            .iter()
            .map(|state| TraceState {
                state_num: state.state_num,
                action: state.action.clone(),
                variables: state
                    .variables
                    .iter()
                    .filter(|(k, _)| var_set.contains(k))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            })
            .collect();

        minimized.minimized = true;
        minimized
    }

    /// Remove specified variables from witness and trace
    pub fn remove_variables(&self, variables: &[String]) -> Self {
        let var_set: std::collections::HashSet<_> = variables.iter().collect();

        let mut minimized = self.clone();

        // Filter witness
        minimized.witness = self
            .witness
            .iter()
            .filter(|(k, _)| !var_set.contains(k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // Filter trace
        minimized.trace = self
            .trace
            .iter()
            .map(|state| TraceState {
                state_num: state.state_num,
                action: state.action.clone(),
                variables: state
                    .variables
                    .iter()
                    .filter(|(k, _)| !var_set.contains(k))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            })
            .collect();

        minimized.minimized = true;
        minimized
    }

    /// Get statistics about the counterexample
    pub fn statistics(&self) -> CounterexampleStats {
        let all_vars = self.trace_variables();
        let changing_vars = self.changing_variables();

        CounterexampleStats {
            num_trace_states: self.trace.len(),
            num_witness_variables: self.witness.len(),
            num_trace_variables: all_vars.len(),
            num_changing_variables: changing_vars.len(),
            num_constant_variables: all_vars.len() - changing_vars.len(),
            num_failed_checks: self.failed_checks.len(),
            has_playback_test: self.playback_test.is_some(),
            has_raw_counterexample: self.raw.is_some(),
            is_minimized: self.minimized,
        }
    }
}
