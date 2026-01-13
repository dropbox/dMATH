//! Trace compression algorithm

use super::types::{CompressedTrace, TraceSegment};
use crate::counterexample::types::{StructuredCounterexample, TraceState};

impl StructuredCounterexample {
    /// Detect and compress repeated patterns in the trace
    /// Returns a compressed representation that can be more easily understood
    pub fn compress_trace(&self) -> CompressedTrace {
        if self.trace.is_empty() {
            return CompressedTrace {
                segments: Vec::new(),
                original_length: 0,
                compressed_length: 0,
            };
        }

        let original_length = self.trace.len();
        let mut segments = Vec::new();
        let mut i = 0;

        while i < self.trace.len() {
            // Try to find repeating patterns starting at position i
            // Try pattern lengths from 1 to half remaining trace
            let max_pattern_len = (self.trace.len() - i) / 2;
            let mut best_pattern: Option<(usize, usize)> = None; // (pattern_len, repeat_count)

            for pattern_len in 1..=max_pattern_len.min(10) {
                // Limit pattern search to reasonable length
                let repeat_count = self.count_repetitions(i, pattern_len);
                if repeat_count >= 2 {
                    // Prefer longer patterns that repeat multiple times
                    let score = pattern_len * repeat_count;
                    if best_pattern.is_none()
                        || score > best_pattern.unwrap().0 * best_pattern.unwrap().1
                    {
                        best_pattern = Some((pattern_len, repeat_count));
                    }
                }
            }

            if let Some((pattern_len, repeat_count)) = best_pattern {
                // Extract the pattern
                let pattern: Vec<TraceState> = self.trace[i..i + pattern_len].to_vec();
                segments.push(TraceSegment::Repeated {
                    pattern,
                    count: repeat_count,
                });
                i += pattern_len * repeat_count;
            } else {
                // No pattern found, add as single state
                segments.push(TraceSegment::Single(self.trace[i].clone()));
                i += 1;
            }
        }

        let compressed_length = segments
            .iter()
            .map(|s| match s {
                TraceSegment::Single(_) => 1,
                TraceSegment::Repeated { pattern, .. } => pattern.len(),
            })
            .sum();

        CompressedTrace {
            segments,
            original_length,
            compressed_length,
        }
    }

    /// Count how many times a pattern of given length repeats starting at position start
    fn count_repetitions(&self, start: usize, pattern_len: usize) -> usize {
        let mut count = 1;
        let mut pos = start + pattern_len;

        while pos + pattern_len <= self.trace.len() {
            let matches = (0..pattern_len).all(|j| {
                self.states_match_for_compression(&self.trace[start + j], &self.trace[pos + j])
            });

            if matches {
                count += 1;
                pos += pattern_len;
            } else {
                break;
            }
        }

        count
    }

    /// Check if two states match for compression purposes
    /// Compares variables values (ignoring state_num which will differ)
    pub(crate) fn states_match_for_compression(&self, a: &TraceState, b: &TraceState) -> bool {
        // Check if actions match
        if a.action != b.action {
            return false;
        }

        // Check if same variables
        if a.variables.len() != b.variables.len() {
            return false;
        }

        // Check all variable values match
        a.variables.iter().all(|(var, val_a)| {
            b.variables
                .get(var)
                .is_some_and(|val_b| val_a.semantically_equal(val_b))
        })
    }

    /// Detect if the trace contains a cycle (returns to a previous state)
    /// Returns the cycle info if found: (cycle_start_index, cycle_length)
    pub fn detect_cycle(&self) -> Option<(usize, usize)> {
        // Compare each state with subsequent states to find repetition
        for i in 0..self.trace.len() {
            for j in (i + 1)..self.trace.len() {
                if self.states_match_for_compression(&self.trace[i], &self.trace[j]) {
                    // Found a cycle: states at i and j match
                    return Some((i, j - i));
                }
            }
        }
        None
    }

    /// Format the trace with cycle detection annotation
    pub fn format_trace_with_cycles(&self) -> String {
        let mut output = String::new();

        if let Some((cycle_start, cycle_len)) = self.detect_cycle() {
            output.push_str(&format!(
                "=== Trace with Cycle (starts at state {}, length {}) ===\n",
                cycle_start + 1,
                cycle_len
            ));
        } else {
            output.push_str("=== Trace (no cycles detected) ===\n");
        }

        // Show compressed view
        let compressed = self.compress_trace();
        if compressed.compression_ratio() > 0.1 {
            output.push_str(&format!("{}", compressed));
        } else {
            // Not much compression, show regular trace
            output.push_str(&self.format_trace_with_diffs());
        }

        output
    }
}
