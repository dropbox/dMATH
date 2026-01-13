//! Trace abstraction types and core logic

use crate::counterexample::types::{CounterexampleValue, StructuredCounterexample, TraceState};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ==================== Trace Abstraction ====================

/// An abstracted representation of similar states in a trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractedState {
    /// Summary description of the abstracted states
    pub description: String,
    /// Number of original states this abstraction represents
    pub count: usize,
    /// Representative variable values (may be ranges or patterns)
    pub variables: HashMap<String, AbstractedValue>,
    /// Original state indices that were abstracted
    pub original_indices: Vec<usize>,
    /// Common action pattern (if any)
    pub common_action: Option<String>,
}

/// An abstracted value that may represent a range or pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AbstractedValue {
    /// A single concrete value (no abstraction needed)
    Concrete(CounterexampleValue),
    /// A range of integer values [min, max]
    IntRange {
        /// Minimum value in the range
        min: i128,
        /// Maximum value in the range
        max: i128,
    },
    /// A set of distinct values seen
    ValueSet(Vec<CounterexampleValue>),
    /// A pattern description for complex values
    Pattern(String),
}

impl std::fmt::Display for AbstractedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AbstractedValue::Concrete(v) => write!(f, "{}", v),
            AbstractedValue::IntRange { min, max } => write!(f, "[{}, {}]", min, max),
            AbstractedValue::ValueSet(vs) => {
                let items: Vec<_> = vs.iter().map(|v| v.to_string()).collect();
                write!(f, "one of {{{}}}", items.join(", "))
            }
            AbstractedValue::Pattern(p) => write!(f, "<{}>", p),
        }
    }
}

/// A trace with some states abstracted into summaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractedTrace {
    /// Sequence of abstracted and concrete segments
    pub segments: Vec<TraceAbstractionSegment>,
    /// Original trace length
    pub original_length: usize,
    /// Compression ratio achieved
    pub compression_ratio: f64,
}

/// A segment in an abstracted trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceAbstractionSegment {
    /// A concrete state from the original trace
    Concrete(TraceState),
    /// An abstraction of multiple similar states
    Abstracted(AbstractedState),
}

impl std::fmt::Display for AbstractedTrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Abstracted Trace ({} -> {} segments, {:.1}% compression)",
            self.original_length,
            self.segments.len(),
            self.compression_ratio * 100.0
        )?;

        for (i, segment) in self.segments.iter().enumerate() {
            match segment {
                TraceAbstractionSegment::Concrete(state) => {
                    write!(f, "  [{}] State {}", i + 1, state.state_num)?;
                    if let Some(ref action) = state.action {
                        write!(f, " <{}>", action)?;
                    }
                    writeln!(f)?;
                }
                TraceAbstractionSegment::Abstracted(abs) => {
                    writeln!(
                        f,
                        "  [{}] {} ({} states)",
                        i + 1,
                        abs.description,
                        abs.count
                    )?;
                    if let Some(ref action) = abs.common_action {
                        writeln!(f, "      Common action: {}", action)?;
                    }
                    for (var, val) in &abs.variables {
                        writeln!(f, "      {}: {}", var, val)?;
                    }
                }
            }
        }

        Ok(())
    }
}

impl StructuredCounterexample {
    /// Abstract consecutive similar states in the trace
    /// Groups states that have the same action and similar variable patterns
    pub fn abstract_trace(&self, min_group_size: usize) -> AbstractedTrace {
        if self.trace.len() <= min_group_size {
            return AbstractedTrace {
                segments: self
                    .trace
                    .iter()
                    .map(|s| TraceAbstractionSegment::Concrete(s.clone()))
                    .collect(),
                original_length: self.trace.len(),
                compression_ratio: 0.0,
            };
        }

        let mut segments = Vec::new();
        let mut i = 0;

        while i < self.trace.len() {
            // Try to find a group of similar states starting at i
            let group_end = self.find_similar_state_group(i);
            let group_size = group_end - i;

            if group_size >= min_group_size {
                // Abstract this group
                let abstracted = self.abstract_state_group(i, group_end);
                segments.push(TraceAbstractionSegment::Abstracted(abstracted));
                i = group_end;
            } else {
                // Keep as concrete
                segments.push(TraceAbstractionSegment::Concrete(self.trace[i].clone()));
                i += 1;
            }
        }

        let compression_ratio = if self.trace.is_empty() {
            0.0
        } else {
            1.0 - (segments.len() as f64 / self.trace.len() as f64)
        };

        AbstractedTrace {
            segments,
            original_length: self.trace.len(),
            compression_ratio,
        }
    }

    /// Find the end index of a group of similar states starting at `start`
    fn find_similar_state_group(&self, start: usize) -> usize {
        if start >= self.trace.len() {
            return start;
        }

        let first = &self.trace[start];
        let mut end = start + 1;

        while end < self.trace.len() {
            let current = &self.trace[end];

            // Check if states are similar enough to group
            if !Self::states_similar_for_abstraction(first, current) {
                break;
            }
            end += 1;
        }

        end
    }

    /// Check if two states are similar enough to be abstracted together
    fn states_similar_for_abstraction(a: &TraceState, b: &TraceState) -> bool {
        // Same action (if present)
        if a.action.is_some() && b.action.is_some() {
            let action_a = a.action.as_ref().unwrap();
            let action_b = b.action.as_ref().unwrap();
            // Actions should be the same or follow the same pattern
            if !Self::actions_similar(action_a, action_b) {
                return false;
            }
        }

        // Same variable names
        if a.variables.len() != b.variables.len() {
            return false;
        }

        // All variable names must match
        for var in a.variables.keys() {
            if !b.variables.contains_key(var) {
                return false;
            }
        }

        true
    }

    /// Check if two actions are similar (same base pattern)
    fn actions_similar(a: &str, b: &str) -> bool {
        // Strip numbers from actions and compare
        let stripped_a = Self::strip_numbers(a);
        let stripped_b = Self::strip_numbers(b);
        stripped_a == stripped_b
    }

    /// Strip numbers from a string for pattern matching
    fn strip_numbers(s: &str) -> String {
        s.chars().filter(|c| !c.is_ascii_digit()).collect()
    }

    /// Create an abstraction of a group of states
    fn abstract_state_group(&self, start: usize, end: usize) -> AbstractedState {
        let states: Vec<_> = (start..end).map(|i| &self.trace[i]).collect();
        let count = states.len();

        // Determine common action pattern
        let common_action = states.first().and_then(|s| s.action.clone());

        // Abstract each variable
        let mut variables = HashMap::new();
        if let Some(first) = states.first() {
            for var in first.variables.keys() {
                let values: Vec<_> = states.iter().filter_map(|s| s.variables.get(var)).collect();

                let abstracted = Self::abstract_values(&values);
                variables.insert(var.clone(), abstracted);
            }
        }

        // Generate description
        let description = if let Some(ref action) = common_action {
            let stripped = Self::strip_numbers(action);
            format!("{} (repeated)", stripped.trim())
        } else {
            format!("States {}-{}", start + 1, end)
        };

        AbstractedState {
            description,
            count,
            variables,
            original_indices: (start..end).collect(),
            common_action,
        }
    }

    /// Abstract a collection of values into a single abstracted value
    pub fn abstract_values(values: &[&CounterexampleValue]) -> AbstractedValue {
        if values.is_empty() {
            return AbstractedValue::Pattern("empty".to_string());
        }

        // Check if all values are the same
        let first = values[0];
        if values.iter().all(|v| *v == first) {
            return AbstractedValue::Concrete(first.clone());
        }

        // Try to create an integer range
        let ints: Vec<i128> = values
            .iter()
            .filter_map(|v| match v {
                CounterexampleValue::Int { value, .. } => Some(*value),
                CounterexampleValue::UInt { value, .. } => (*value).try_into().ok(),
                _ => None,
            })
            .collect();

        if ints.len() == values.len() {
            let min = *ints.iter().min().unwrap();
            let max = *ints.iter().max().unwrap();
            return AbstractedValue::IntRange { min, max };
        }

        // Return as value set if small enough
        if values.len() <= 5 {
            // Dedup manually since CounterexampleValue doesn't implement Hash
            let mut unique: Vec<CounterexampleValue> = Vec::new();
            for v in values {
                if !unique.iter().any(|u| u == *v) {
                    unique.push((*v).clone());
                }
            }
            return AbstractedValue::ValueSet(unique);
        }

        // Default to pattern description
        AbstractedValue::Pattern(format!("{} distinct values", values.len()))
    }
}
