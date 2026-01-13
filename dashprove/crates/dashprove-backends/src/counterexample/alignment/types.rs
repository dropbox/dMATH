//! Multi-trace alignment types

use crate::counterexample::types::{CounterexampleValue, TraceState};

/// A row in a multi-trace alignment table
#[derive(Debug, Clone)]
pub struct MultiTraceAlignmentRow {
    /// State number (used as alignment key)
    pub state_num: u32,
    /// States from each trace (None if trace doesn't have this state number)
    pub states: Vec<Option<TraceState>>,
}

/// Result of aligning multiple counterexample traces
#[derive(Debug, Clone)]
pub struct MultiTraceAlignment {
    /// Column labels (identifiers for each trace)
    pub trace_labels: Vec<String>,
    /// Aligned rows by state number
    pub rows: Vec<MultiTraceAlignmentRow>,
    /// Variables that differ across traces at each state
    pub divergence_points: Vec<DivergencePoint>,
}

/// A point where traces diverge in their values
#[derive(Debug, Clone)]
pub struct DivergencePoint {
    /// State number where divergence occurs
    pub state_num: u32,
    /// Variable that differs
    pub variable: String,
    /// Values in each trace (None if variable not present)
    pub values: Vec<Option<CounterexampleValue>>,
}

impl MultiTraceAlignment {
    /// Get the number of traces being aligned
    pub fn trace_count(&self) -> usize {
        self.trace_labels.len()
    }

    /// Find all state numbers where at least one trace differs from others
    pub fn divergent_states(&self) -> Vec<u32> {
        self.divergence_points
            .iter()
            .map(|dp| dp.state_num)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }

    /// Find the first state where traces diverge
    pub fn first_divergence(&self) -> Option<u32> {
        self.divergence_points.iter().map(|dp| dp.state_num).min()
    }
}
