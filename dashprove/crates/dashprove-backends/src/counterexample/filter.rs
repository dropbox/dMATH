//! Trace filtering functionality
//!
//! This module provides tools for filtering counterexample traces:
//! - `TraceFilterOptions`: Configuration for filtering trace states

use super::types::{StructuredCounterexample, TraceState};
use std::collections::HashMap;

// ==================== Trace Filtering ====================

/// Options for filtering trace data
#[derive(Debug, Clone, Default)]
pub struct TraceFilterOptions {
    /// Only include these variables (if empty, include all)
    pub include_variables: Vec<String>,
    /// Exclude these variables
    pub exclude_variables: Vec<String>,
    /// Only include states where these variables changed
    pub changed_variables: Vec<String>,
    /// Skip states with no changes (after the initial state)
    pub skip_unchanged_states: bool,
    /// Maximum number of states to include
    pub max_states: Option<usize>,
    /// Regex pattern for including variables (matches full variable name)
    pub include_pattern: Option<String>,
    /// Regex pattern for excluding variables (matches full variable name)
    pub exclude_pattern: Option<String>,
}

impl TraceFilterOptions {
    /// Create options that include only specific variables
    pub fn only(variables: Vec<String>) -> Self {
        Self {
            include_variables: variables,
            ..Default::default()
        }
    }

    /// Create options that exclude specific variables
    pub fn excluding(variables: Vec<String>) -> Self {
        Self {
            exclude_variables: variables,
            ..Default::default()
        }
    }

    /// Only include states where the given variables changed
    pub fn changed(variables: Vec<String>) -> Self {
        Self {
            changed_variables: variables,
            ..Default::default()
        }
    }

    /// Create options that include variables matching a regex pattern
    pub fn matching(pattern: &str) -> Self {
        Self {
            include_pattern: Some(pattern.to_string()),
            ..Default::default()
        }
    }

    /// Create options that exclude variables matching a regex pattern
    pub fn excluding_pattern(pattern: &str) -> Self {
        Self {
            exclude_pattern: Some(pattern.to_string()),
            ..Default::default()
        }
    }

    /// Check if a variable should be included based on patterns
    fn matches_include_pattern(&self, var: &str) -> bool {
        if let Some(ref pattern) = self.include_pattern {
            if let Ok(re) = regex::Regex::new(pattern) {
                return re.is_match(var);
            }
        }
        true // No pattern means include all
    }

    /// Check if a variable should be excluded based on patterns
    fn matches_exclude_pattern(&self, var: &str) -> bool {
        if let Some(ref pattern) = self.exclude_pattern {
            if let Ok(re) = regex::Regex::new(pattern) {
                return re.is_match(var);
            }
        }
        false // No pattern means exclude nothing
    }
}

impl StructuredCounterexample {
    /// Create a filtered view of the trace
    pub fn filter_trace(&self, options: &TraceFilterOptions) -> Vec<TraceState> {
        let mut filtered = Vec::new();
        let mut prev_state: Option<&TraceState> = None;

        for (i, state) in self.trace.iter().enumerate() {
            // Check max states limit
            if let Some(max) = options.max_states {
                if filtered.len() >= max {
                    break;
                }
            }

            // Filter variables in this state
            let mut filtered_vars = HashMap::new();
            for (var, value) in &state.variables {
                // Check include list (if specified)
                if !options.include_variables.is_empty() && !options.include_variables.contains(var)
                {
                    continue;
                }

                // Check exclude list
                if options.exclude_variables.contains(var) {
                    continue;
                }

                // Check regex include pattern
                if options.include_pattern.is_some() && !options.matches_include_pattern(var) {
                    continue;
                }

                // Check regex exclude pattern
                if options.matches_exclude_pattern(var) {
                    continue;
                }

                filtered_vars.insert(var.clone(), value.clone());
            }

            // Check if we should skip this state based on changes
            if i > 0 && options.skip_unchanged_states {
                if let Some(prev) = prev_state {
                    let has_changes = filtered_vars
                        .iter()
                        .any(|(var, value)| prev.variables.get(var) != Some(value));

                    if !has_changes {
                        continue;
                    }
                }
            }

            // Check changed_variables filter
            if !options.changed_variables.is_empty() && i > 0 {
                if let Some(prev) = prev_state {
                    let relevant_change = options.changed_variables.iter().any(|var| {
                        let old = prev.variables.get(var);
                        let new = state.variables.get(var);
                        old != new
                    });

                    if !relevant_change {
                        continue;
                    }
                }
            }

            let filtered_state = TraceState {
                state_num: state.state_num,
                action: state.action.clone(),
                variables: filtered_vars,
            };

            filtered.push(filtered_state);
            prev_state = Some(state);
        }

        filtered
    }

    /// Get a copy of this counterexample with a filtered trace
    pub fn with_filtered_trace(&self, options: &TraceFilterOptions) -> Self {
        Self {
            witness: self.witness.clone(),
            failed_checks: self.failed_checks.clone(),
            playback_test: self.playback_test.clone(),
            trace: self.filter_trace(options),
            raw: self.raw.clone(),
            minimized: self.minimized,
        }
    }

    /// Get all variable names that appear in the trace
    pub fn trace_variables(&self) -> Vec<String> {
        let mut vars: std::collections::HashSet<String> = std::collections::HashSet::new();
        for state in &self.trace {
            for var in state.variables.keys() {
                vars.insert(var.clone());
            }
        }
        let mut sorted: Vec<_> = vars.into_iter().collect();
        sorted.sort();
        sorted
    }

    /// Get variables that change during the trace
    pub fn changing_variables(&self) -> Vec<String> {
        let mut changing = std::collections::HashSet::new();

        for i in 1..self.trace.len() {
            let prev = &self.trace[i - 1];
            let curr = &self.trace[i];

            for (var, value) in &curr.variables {
                if prev.variables.get(var) != Some(value) {
                    changing.insert(var.clone());
                }
            }
        }

        let mut sorted: Vec<_> = changing.into_iter().collect();
        sorted.sort();
        sorted
    }
}
