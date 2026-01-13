//! Multi-trace alignment algorithm

use super::types::{DivergencePoint, MultiTraceAlignment, MultiTraceAlignmentRow};
use crate::counterexample::types::{StructuredCounterexample, TraceState};
use std::collections::HashMap;

/// Align multiple counterexample traces by state number
pub fn align_multiple_traces(
    traces: &[&StructuredCounterexample],
    labels: Option<Vec<String>>,
) -> MultiTraceAlignment {
    let trace_labels = labels.unwrap_or_else(|| {
        traces
            .iter()
            .enumerate()
            .map(|(i, _)| format!("Trace {}", i + 1))
            .collect()
    });

    // Collect all state numbers
    let mut all_state_nums: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for trace in traces {
        for state in &trace.trace {
            all_state_nums.insert(state.state_num);
        }
    }

    // Build state lookup maps
    let trace_maps: Vec<HashMap<u32, &TraceState>> = traces
        .iter()
        .map(|t| t.trace.iter().map(|s| (s.state_num, s)).collect())
        .collect();

    // Build alignment rows
    let mut rows = Vec::new();
    let mut divergence_points = Vec::new();

    for state_num in all_state_nums {
        let states: Vec<Option<TraceState>> = trace_maps
            .iter()
            .map(|m| m.get(&state_num).map(|s| (*s).clone()))
            .collect();

        // Find divergence points at this state
        let present_states: Vec<&TraceState> = states.iter().filter_map(|s| s.as_ref()).collect();
        if !present_states.is_empty() {
            // Collect all variables
            let mut all_vars: std::collections::HashSet<String> = std::collections::HashSet::new();
            for state in &present_states {
                all_vars.extend(state.variables.keys().cloned());
            }

            for var in all_vars {
                let values: Vec<Option<crate::counterexample::types::CounterexampleValue>> = states
                    .iter()
                    .map(|s| s.as_ref().and_then(|st| st.variables.get(&var).cloned()))
                    .collect();

                // Check if values differ
                let first_some = values.iter().find(|v| v.is_some());
                if let Some(first_val) = first_some {
                    let differs = values.iter().any(|v| v != first_val);
                    if differs {
                        divergence_points.push(DivergencePoint {
                            state_num,
                            variable: var,
                            values,
                        });
                    }
                }
            }
        }

        rows.push(MultiTraceAlignmentRow { state_num, states });
    }

    MultiTraceAlignment {
        trace_labels,
        rows,
        divergence_points,
    }
}
