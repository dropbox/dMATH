//! Fix suggestions from counterexamples
//!
//! This module provides tools for generating fix suggestions:
//! - `TraceSuggestion`: A suggested fix based on trace analysis
//! - `SuggestionKind`: Type of suggestion (invariant, action, etc)
//! - `SuggestionSeverity`: How critical the suggestion is

use super::compression::TraceSegment;
use super::types::{CounterexampleValue, StructuredCounterexample};
use serde::{Deserialize, Serialize};

// ============================================================================
// Pattern Suggestions
// ============================================================================

/// A suggested pattern for filtering or understanding traces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSuggestion {
    /// Type of suggestion
    pub kind: SuggestionKind,
    /// Human-readable description
    pub description: String,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Severity/priority level (derived from kind and confidence)
    pub severity: SuggestionSeverity,
    /// Suggested filter or action to take
    pub suggested_action: String,
}

impl TraceSuggestion {
    /// Create a new suggestion with auto-calculated severity
    pub fn new(
        kind: SuggestionKind,
        description: String,
        confidence: f64,
        suggested_action: String,
    ) -> Self {
        let severity = Self::calculate_severity(&kind, confidence);
        Self {
            kind,
            description,
            confidence,
            severity,
            suggested_action,
        }
    }

    /// Calculate severity based on kind and confidence
    fn calculate_severity(kind: &SuggestionKind, confidence: f64) -> SuggestionSeverity {
        // Different suggestion kinds have different base severities
        let kind_weight = match kind {
            // Interleaving actors often explain concurrency issues - higher priority
            SuggestionKind::InterleavingActors => 1.2,
            // Repeating patterns indicate potential cycles/livelocks - higher priority
            SuggestionKind::RepeatingPattern => 1.1,
            // Monotonic variables can indicate progress/termination issues
            SuggestionKind::MonotonicVariable => 1.0,
            // Focus state range helps find the "interesting" part
            SuggestionKind::FocusStateRange => 0.9,
            // Invariant variables are noise reduction - useful but not critical
            SuggestionKind::InvariantVariable => 0.7,
            // Filter variables is a general noise reduction suggestion
            SuggestionKind::FilterVariables => 0.6,
        };

        let effective_confidence = (confidence * kind_weight).min(1.0);
        SuggestionSeverity::from_confidence(effective_confidence)
    }

    /// Format as a single-line summary
    pub fn format_summary(&self) -> String {
        format!("[{}] {}: {}", self.severity, self.kind, self.description)
    }

    /// Format with full details including suggested action
    pub fn format_detailed(&self) -> String {
        format!(
            "[{}] {} (confidence: {:.0}%)\n  {}\n  Action: {}",
            self.severity,
            self.kind,
            self.confidence * 100.0,
            self.description,
            self.suggested_action
        )
    }
}

/// Types of pattern suggestions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionKind {
    /// Suggest filtering out certain variables
    FilterVariables,
    /// Suggest focusing on certain state range
    FocusStateRange,
    /// Detected repeating pattern
    RepeatingPattern,
    /// Detected interleaving actors
    InterleavingActors,
    /// Detected monotonic variable
    MonotonicVariable,
    /// Detected invariant variable (never changes)
    InvariantVariable,
}

impl std::fmt::Display for SuggestionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SuggestionKind::FilterVariables => write!(f, "Filter Variables"),
            SuggestionKind::FocusStateRange => write!(f, "Focus State Range"),
            SuggestionKind::RepeatingPattern => write!(f, "Repeating Pattern"),
            SuggestionKind::InterleavingActors => write!(f, "Interleaving Actors"),
            SuggestionKind::MonotonicVariable => write!(f, "Monotonic Variable"),
            SuggestionKind::InvariantVariable => write!(f, "Invariant Variable"),
        }
    }
}

/// Severity level for a suggestion - how important it is to act on
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SuggestionSeverity {
    /// Low priority - informational, may help understanding
    Low,
    /// Medium priority - suggested action could simplify analysis
    Medium,
    /// High priority - strongly recommended action
    High,
    /// Critical - this insight is essential for understanding the counterexample
    Critical,
}

impl std::fmt::Display for SuggestionSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SuggestionSeverity::Low => write!(f, "Low"),
            SuggestionSeverity::Medium => write!(f, "Medium"),
            SuggestionSeverity::High => write!(f, "High"),
            SuggestionSeverity::Critical => write!(f, "Critical"),
        }
    }
}

impl SuggestionSeverity {
    /// Convert confidence to severity level
    pub fn from_confidence(confidence: f64) -> Self {
        if confidence >= 0.9 {
            SuggestionSeverity::Critical
        } else if confidence >= 0.7 {
            SuggestionSeverity::High
        } else if confidence >= 0.5 {
            SuggestionSeverity::Medium
        } else {
            SuggestionSeverity::Low
        }
    }

    /// Get numeric value for sorting (higher = more important)
    pub fn priority_value(&self) -> u8 {
        match self {
            SuggestionSeverity::Low => 1,
            SuggestionSeverity::Medium => 2,
            SuggestionSeverity::High => 3,
            SuggestionSeverity::Critical => 4,
        }
    }
}

impl StructuredCounterexample {
    /// Analyze the trace and suggest patterns or filters that might help understanding
    pub fn suggest_patterns(&self) -> Vec<TraceSuggestion> {
        let mut suggestions = Vec::new();

        if self.trace.is_empty() {
            return suggestions;
        }

        // Check for invariant variables (never change)
        self.suggest_invariant_variables(&mut suggestions);

        // Check for monotonic variables
        self.suggest_monotonic_variables(&mut suggestions);

        // Check for repeating patterns
        self.suggest_repeating_patterns(&mut suggestions);

        // Check for interleaving actors
        self.suggest_interleaving_actors(&mut suggestions);

        // Check for interesting state ranges
        self.suggest_state_ranges(&mut suggestions);

        // Sort by severity (primary) then confidence (secondary)
        suggestions.sort_by(|a, b| match b.severity.cmp(&a.severity) {
            std::cmp::Ordering::Equal => b
                .confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal),
            ord => ord,
        });

        suggestions
    }

    fn suggest_invariant_variables(&self, suggestions: &mut Vec<TraceSuggestion>) {
        if self.trace.len() < 2 {
            return;
        }

        // Get all variables from first state
        let first = &self.trace[0];
        let mut invariant_vars = Vec::new();

        for (var, first_val) in &first.variables {
            let is_invariant = self.trace[1..]
                .iter()
                .all(|state| state.variables.get(var) == Some(first_val));

            if is_invariant {
                invariant_vars.push(var.clone());
            }
        }

        if !invariant_vars.is_empty() {
            let var_list = invariant_vars.join(", ");
            let count = invariant_vars.len();
            suggestions.push(TraceSuggestion::new(
                SuggestionKind::InvariantVariable,
                format!(
                    "{} variable(s) never change: {}",
                    count,
                    if count <= 3 {
                        var_list.clone()
                    } else {
                        format!("{} and {} more", invariant_vars[..3].join(", "), count - 3)
                    }
                ),
                0.8,
                format!(
                    "Consider filtering out invariant variables with exclude_variables: [{}]",
                    var_list
                ),
            ));
        }
    }

    fn suggest_monotonic_variables(&self, suggestions: &mut Vec<TraceSuggestion>) {
        if self.trace.len() < 3 {
            return;
        }

        // Get variables from first state
        let first = &self.trace[0];

        for var in first.variables.keys() {
            // Extract integer values
            let values: Vec<i128> = self
                .trace
                .iter()
                .filter_map(|s| {
                    s.variables.get(var).and_then(|v| match v {
                        CounterexampleValue::Int { value, .. } => Some(*value),
                        CounterexampleValue::UInt { value, .. } => (*value).try_into().ok(),
                        _ => None,
                    })
                })
                .collect();

            if values.len() < 3 {
                continue;
            }

            // Check if monotonically increasing
            let is_increasing = values.windows(2).all(|w| w[1] >= w[0]);
            let is_strictly_increasing = values.windows(2).all(|w| w[1] > w[0]);

            // Check if monotonically decreasing
            let is_decreasing = values.windows(2).all(|w| w[1] <= w[0]);
            let is_strictly_decreasing = values.windows(2).all(|w| w[1] < w[0]);

            if is_strictly_increasing {
                suggestions.push(TraceSuggestion::new(
                    SuggestionKind::MonotonicVariable,
                    format!(
                        "Variable '{}' is strictly increasing ({} -> {})",
                        var,
                        values.first().unwrap(),
                        values.last().unwrap()
                    ),
                    0.9,
                    format!(
                        "Variable '{}' may be a counter or timestamp - useful for ordering",
                        var
                    ),
                ));
            } else if is_increasing && !is_strictly_increasing {
                suggestions.push(TraceSuggestion::new(
                    SuggestionKind::MonotonicVariable,
                    format!(
                        "Variable '{}' is non-decreasing ({} -> {})",
                        var,
                        values.first().unwrap(),
                        values.last().unwrap()
                    ),
                    0.7,
                    format!(
                        "Variable '{}' may be a monotonic counter with some stable phases",
                        var
                    ),
                ));
            } else if is_strictly_decreasing {
                suggestions.push(TraceSuggestion::new(
                    SuggestionKind::MonotonicVariable,
                    format!(
                        "Variable '{}' is strictly decreasing ({} -> {})",
                        var,
                        values.first().unwrap(),
                        values.last().unwrap()
                    ),
                    0.9,
                    format!(
                        "Variable '{}' may be a countdown or remaining resource",
                        var
                    ),
                ));
            } else if is_decreasing && !is_strictly_decreasing {
                suggestions.push(TraceSuggestion::new(
                    SuggestionKind::MonotonicVariable,
                    format!(
                        "Variable '{}' is non-increasing ({} -> {})",
                        var,
                        values.first().unwrap(),
                        values.last().unwrap()
                    ),
                    0.7,
                    format!(
                        "Variable '{}' may be a countdown with some stable phases",
                        var
                    ),
                ));
            }
        }
    }

    fn suggest_repeating_patterns(&self, suggestions: &mut Vec<TraceSuggestion>) {
        let compressed = self.compress_trace();
        if compressed.compression_ratio() > 0.3 {
            let repeated_count = compressed
                .segments
                .iter()
                .filter(|s| matches!(s, TraceSegment::Repeated { .. }))
                .count();

            if repeated_count > 0 {
                suggestions.push(TraceSuggestion::new(
                    SuggestionKind::RepeatingPattern,
                    format!(
                        "Trace has {:.0}% repetition ({} repeated patterns)",
                        compressed.compression_ratio() * 100.0,
                        repeated_count
                    ),
                    compressed.compression_ratio(),
                    "Use compress_trace() to see repeated patterns more clearly".to_string(),
                ));
            }
        }
    }

    fn suggest_interleaving_actors(&self, suggestions: &mut Vec<TraceSuggestion>) {
        let interleaving = self.detect_interleavings();

        if interleaving.lanes.len() >= 2 && interleaving.coverage() > 0.5 {
            let actor_names: Vec<_> = interleaving.lanes.iter().map(|l| l.actor.clone()).collect();
            suggestions.push(TraceSuggestion::new(
                SuggestionKind::InterleavingActors,
                format!(
                    "Detected {} interleaved actors: {} ({:.0}% coverage, {} switches)",
                    interleaving.lanes.len(),
                    actor_names.join(", "),
                    interleaving.coverage() * 100.0,
                    interleaving.lane_switches()
                ),
                interleaving.coverage(),
                "Use detect_interleavings() to view per-actor traces separately".to_string(),
            ));
        }
    }

    fn suggest_state_ranges(&self, suggestions: &mut Vec<TraceSuggestion>) {
        if self.trace.len() < 5 {
            return;
        }

        // Look for state ranges with high activity
        let window_size = std::cmp::min(5, self.trace.len() / 3);
        if window_size < 2 {
            return;
        }

        let mut max_changes = 0usize;
        let mut max_change_start = 0usize;

        for start in 0..self.trace.len().saturating_sub(window_size) {
            let mut changes = 0usize;
            for i in start..start + window_size - 1 {
                let diff = Self::diff_states(&self.trace[i], &self.trace[i + 1]);
                changes += diff.value_diffs.len();
            }
            if changes > max_changes {
                max_changes = changes;
                max_change_start = start;
            }
        }

        if max_changes > window_size * 2 {
            // High activity window
            suggestions.push(TraceSuggestion::new(
                SuggestionKind::FocusStateRange,
                format!(
                    "High activity in states {}-{} ({} variable changes)",
                    max_change_start + 1,
                    max_change_start + window_size,
                    max_changes
                ),
                0.6,
                format!(
                    "Focus on states {}-{} where most changes occur",
                    max_change_start + 1,
                    max_change_start + window_size
                ),
            ));
        }
    }
}
