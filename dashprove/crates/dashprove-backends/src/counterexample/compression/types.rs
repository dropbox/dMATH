//! Trace pattern compression types

use crate::counterexample::types::TraceState;
use serde::{Deserialize, Serialize};

/// A compressed representation of a trace pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedTrace {
    /// Segments of the compressed trace
    pub segments: Vec<TraceSegment>,
    /// Original trace length before compression
    pub original_length: usize,
    /// Compressed length (number of actual unique patterns)
    pub compressed_length: usize,
}

impl CompressedTrace {
    /// Calculate compression ratio (0.0 = no compression, 1.0 = maximum compression)
    pub fn compression_ratio(&self) -> f64 {
        if self.original_length == 0 {
            return 0.0;
        }
        1.0 - (self.compressed_length as f64 / self.original_length as f64)
    }

    /// Get total number of states represented
    pub fn total_states(&self) -> usize {
        self.segments.iter().map(|s| s.total_states()).sum()
    }

    /// Expand the compressed trace back to full trace
    pub fn expand(&self) -> Vec<TraceState> {
        let mut result = Vec::new();
        let mut state_num = 1u32;

        for segment in &self.segments {
            match segment {
                TraceSegment::Single(state) => {
                    let mut expanded = state.clone();
                    expanded.state_num = state_num;
                    result.push(expanded);
                    state_num += 1;
                }
                TraceSegment::Repeated { pattern, count } => {
                    for _ in 0..*count {
                        for state in pattern {
                            let mut expanded = state.clone();
                            expanded.state_num = state_num;
                            result.push(expanded);
                            state_num += 1;
                        }
                    }
                }
            }
        }

        result
    }
}

impl std::fmt::Display for CompressedTrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Compressed Trace ({} states -> {} segments, {:.1}% compression)",
            self.original_length,
            self.segments.len(),
            self.compression_ratio() * 100.0
        )?;

        for (i, segment) in self.segments.iter().enumerate() {
            match segment {
                TraceSegment::Single(state) => {
                    writeln!(f, "  [{}] State {}", i + 1, state.state_num)?;
                    if let Some(ref action) = state.action {
                        writeln!(f, "      Action: {}", action)?;
                    }
                }
                TraceSegment::Repeated { pattern, count } => {
                    writeln!(
                        f,
                        "  [{}] Repeat {} times ({} states each):",
                        i + 1,
                        count,
                        pattern.len()
                    )?;
                    for (j, state) in pattern.iter().enumerate() {
                        write!(f, "      Pattern[{}]", j + 1)?;
                        if let Some(ref action) = state.action {
                            write!(f, " <{}>", action)?;
                        }
                        writeln!(f)?;
                    }
                }
            }
        }

        Ok(())
    }
}

/// A segment in a compressed trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceSegment {
    /// A single non-repeated state
    Single(TraceState),
    /// A repeated pattern
    Repeated {
        /// The pattern of states that repeats
        pattern: Vec<TraceState>,
        /// Number of times the pattern repeats
        count: usize,
    },
}

impl TraceSegment {
    /// Get total number of states represented by this segment
    pub fn total_states(&self) -> usize {
        match self {
            TraceSegment::Single(_) => 1,
            TraceSegment::Repeated { pattern, count } => pattern.len() * count,
        }
    }
}
