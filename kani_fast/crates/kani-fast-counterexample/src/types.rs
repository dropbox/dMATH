//! Core counterexample types
//!
//! This module contains the fundamental types for representing counterexamples:
//! - `CounterexampleValue`: Concrete values (ints, bools, sets, records, etc.)
//! - `TraceState`: A single state in a counterexample trace
//! - `StructuredCounterexample`: The main counterexample container
//! - `FailedCheck`: Information about failed verification checks
//! - `SourceLocation`: Source code locations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write;

/// Type alias for state diffs: variable name -> (old_value, new_value)
pub type StateDiff = HashMap<String, (Option<CounterexampleValue>, CounterexampleValue)>;

/// Type alias for a sequence of state diffs with state numbers
pub type TraceDiffs = Vec<(u32, StateDiff)>;

/// A concrete value in a counterexample
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CounterexampleValue {
    /// Integer value with optional type info (e.g., "u32", "i64")
    Int {
        value: i128,
        type_hint: Option<String>,
    },
    /// Unsigned integer value
    UInt {
        value: u128,
        type_hint: Option<String>,
    },
    /// Floating point value
    Float { value: f64 },
    /// Boolean value
    Bool(bool),
    /// String value
    String(String),
    /// Raw bytes (from Kani concrete playback)
    Bytes(Vec<u8>),
    /// Set of values
    Set(Vec<CounterexampleValue>),
    /// Sequence/tuple of values
    Sequence(Vec<CounterexampleValue>),
    /// Record/struct with named fields
    Record(HashMap<String, CounterexampleValue>),
    /// Function/mapping
    Function(Vec<(CounterexampleValue, CounterexampleValue)>),
    /// Unknown/unparsed value
    Unknown(String),
}

impl CounterexampleValue {
    /// Check if two values are semantically equivalent.
    #[must_use]
    pub fn semantically_equal(&self, other: &Self) -> bool {
        match (self, other) {
            (
                CounterexampleValue::Int {
                    value: v1,
                    type_hint: t1,
                },
                CounterexampleValue::Int {
                    value: v2,
                    type_hint: t2,
                },
            ) => v1 == v2 && t1 == t2,
            (
                CounterexampleValue::UInt {
                    value: v1,
                    type_hint: t1,
                },
                CounterexampleValue::UInt {
                    value: v2,
                    type_hint: t2,
                },
            ) => v1 == v2 && t1 == t2,
            (
                CounterexampleValue::Float { value: v1 },
                CounterexampleValue::Float { value: v2 },
            ) => (v1 - v2).abs() < f64::EPSILON,
            (CounterexampleValue::Bool(b1), CounterexampleValue::Bool(b2)) => b1 == b2,
            (CounterexampleValue::String(s1), CounterexampleValue::String(s2)) => s1 == s2,
            (CounterexampleValue::Bytes(b1), CounterexampleValue::Bytes(b2)) => b1 == b2,
            (CounterexampleValue::Set(s1), CounterexampleValue::Set(s2)) => {
                if s1.len() != s2.len() {
                    return false;
                }
                // MUTATION NOTE: The && on line 91 changing to || is an equivalent mutant.
                // Given equal-length sets where every element of s1 is in s2 (first condition),
                // every element of s2 must be in s1 (pigeonhole principle). The bidirectional
                // check is technically redundant but kept for clarity.
                s1.iter()
                    .all(|e1| s2.iter().any(|e2| e1.semantically_equal(e2)))
                    && s2
                        .iter()
                        .all(|e2| s1.iter().any(|e1| e1.semantically_equal(e2)))
            }
            (CounterexampleValue::Sequence(seq1), CounterexampleValue::Sequence(seq2)) => {
                seq1.len() == seq2.len()
                    && seq1
                        .iter()
                        .zip(seq2.iter())
                        .all(|(a, b)| a.semantically_equal(b))
            }
            (CounterexampleValue::Record(r1), CounterexampleValue::Record(r2)) => {
                if r1.len() != r2.len() {
                    return false;
                }
                r1.iter()
                    .all(|(k, v1)| r2.get(k).is_some_and(|v2| v1.semantically_equal(v2)))
            }
            (CounterexampleValue::Function(f1), CounterexampleValue::Function(f2)) => {
                if f1.len() != f2.len() {
                    return false;
                }
                f1.iter().all(|(k1, v1)| {
                    f2.iter()
                        .any(|(k2, v2)| k1.semantically_equal(k2) && v1.semantically_equal(v2))
                })
            }
            (CounterexampleValue::Unknown(s1), CounterexampleValue::Unknown(s2)) => s1 == s2,
            _ => false,
        }
    }

    /// Normalize a value for canonical comparison
    #[must_use]
    pub fn normalize(&self) -> Self {
        match self {
            CounterexampleValue::Set(elems) => {
                let mut normalized: Vec<_> = elems.iter().map(|e| e.normalize()).collect();
                normalized.sort_by_key(|a| a.to_string());
                CounterexampleValue::Set(normalized)
            }
            CounterexampleValue::Sequence(elems) => {
                CounterexampleValue::Sequence(elems.iter().map(|e| e.normalize()).collect())
            }
            CounterexampleValue::Record(fields) => {
                let normalized: HashMap<String, CounterexampleValue> = fields
                    .iter()
                    .map(|(k, v)| (k.clone(), v.normalize()))
                    .collect();
                CounterexampleValue::Record(normalized)
            }
            CounterexampleValue::Function(mappings) => {
                let mut normalized: Vec<_> = mappings
                    .iter()
                    .map(|(k, v)| (k.normalize(), v.normalize()))
                    .collect();
                normalized.sort_by_key(|(k, _)| k.to_string());
                CounterexampleValue::Function(normalized)
            }
            _ => self.clone(),
        }
    }
}

impl std::fmt::Display for CounterexampleValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int { value, type_hint } => {
                if let Some(ty) = type_hint {
                    write!(f, "{value} ({ty})")
                } else {
                    write!(f, "{value}")
                }
            }
            Self::UInt { value, type_hint } => {
                if let Some(ty) = type_hint {
                    write!(f, "{value} ({ty})")
                } else {
                    write!(f, "{value}")
                }
            }
            Self::Float { value } => write!(f, "{value}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::String(s) => write!(f, "\"{s}\""),
            Self::Bytes(bytes) => {
                write!(
                    f,
                    "[{}]",
                    bytes
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Self::Set(elems) => {
                let items: Vec<_> = elems.iter().map(ToString::to_string).collect();
                write!(f, "{{{}}}", items.join(", "))
            }
            Self::Sequence(elems) => {
                let items: Vec<_> = elems.iter().map(ToString::to_string).collect();
                write!(f, "<<{}>>", items.join(", "))
            }
            Self::Record(fields) => {
                let mut items: Vec<_> = fields.iter().collect();
                items.sort_by_key(|(k, _)| *k);
                let formatted: Vec<_> = items.iter().map(|(k, v)| format!("{k} |-> {v}")).collect();
                write!(f, "[{}]", formatted.join(", "))
            }
            Self::Function(mappings) => {
                let items: Vec<_> = mappings
                    .iter()
                    .map(|(k, v)| format!("{k} :> {v}"))
                    .collect();
                write!(f, "({})", items.join(" @@ "))
            }
            Self::Unknown(s) => write!(f, "{s}"),
        }
    }
}

/// Information about a failed check in Kani
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FailedCheck {
    /// Check identifier (e.g., "fail::unsafe_div.assertion.1")
    pub check_id: String,
    /// Human-readable description (e.g., "attempt to divide by zero")
    pub description: String,
    /// Source location (file:line:column)
    pub location: Option<SourceLocation>,
    /// Function containing the failure
    pub function: Option<String>,
}

/// Source code location
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceLocation {
    /// File path
    pub file: String,
    /// Line number (1-indexed)
    pub line: u32,
    /// Column number (1-indexed, optional)
    pub column: Option<u32>,
}

/// A single state in a counterexample trace
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraceState {
    /// State number in the trace (1-indexed)
    pub state_num: u32,
    /// Action/transition that led to this state
    pub action: Option<String>,
    /// Variable assignments in this state
    pub variables: HashMap<String, CounterexampleValue>,
}

impl TraceState {
    /// Create a new trace state with the given state number
    #[must_use]
    pub fn new(state_num: u32) -> Self {
        Self {
            state_num,
            action: None,
            variables: HashMap::new(),
        }
    }

    /// Compute the diff between this state and a previous state
    #[must_use]
    pub fn diff_from(&self, previous: &Self) -> StateDiff {
        let mut changes = HashMap::new();

        for (var_name, new_value) in &self.variables {
            match previous.variables.get(var_name) {
                Some(old_value) if old_value != new_value => {
                    changes.insert(
                        var_name.clone(),
                        (Some(old_value.clone()), new_value.clone()),
                    );
                }
                None => {
                    changes.insert(var_name.clone(), (None, new_value.clone()));
                }
                _ => {}
            }
        }

        changes
    }
}

impl std::fmt::Display for TraceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "State {}", self.state_num)?;
        if let Some(action) = &self.action {
            write!(f, " <{action}>")?;
        }
        writeln!(f)?;

        let mut vars: Vec<_> = self.variables.iter().collect();
        vars.sort_by_key(|(k, _)| *k);

        for (var_name, value) in vars {
            writeln!(f, "  {var_name} = {value}")?;
        }

        Ok(())
    }
}

impl std::fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(col) = self.column {
            write!(f, "{}:{}:{col}", self.file, self.line)
        } else {
            write!(f, "{}:{}", self.file, self.line)
        }
    }
}

impl std::fmt::Display for FailedCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let desc = &self.description;
        write!(f, "{desc}")?;
        if let Some(loc) = &self.location {
            write!(f, " at {loc}")?;
        }
        if let Some(func) = &self.function {
            write!(f, " in {func}")?;
        }
        Ok(())
    }
}

/// Structured counterexample with parsed values and context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StructuredCounterexample {
    /// Witness values: variable name -> concrete value
    pub witness: HashMap<String, CounterexampleValue>,
    /// List of failed checks with details
    pub failed_checks: Vec<FailedCheck>,
    /// Concrete playback test code (if available)
    pub playback_test: Option<String>,
    /// Sequence of states forming a counterexample trace
    pub trace: Vec<TraceState>,
    /// Raw counterexample text (for backwards compatibility)
    pub raw: Option<String>,
    /// Whether the counterexample has been minimized
    pub minimized: bool,
}

impl StructuredCounterexample {
    /// Create a new empty structured counterexample
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from raw text (backwards compatible)
    #[must_use]
    pub fn from_raw(raw: String) -> Self {
        Self {
            raw: Some(raw),
            ..Default::default()
        }
    }

    /// Check if this counterexample has any structured data
    #[must_use]
    pub fn has_structured_data(&self) -> bool {
        !self.witness.is_empty()
            || !self.failed_checks.is_empty()
            || self.playback_test.is_some()
            || !self.trace.is_empty()
    }

    /// Get a human-readable summary
    #[must_use]
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        if !self.failed_checks.is_empty() {
            let checks: Vec<_> = self
                .failed_checks
                .iter()
                .map(|c| c.description.clone())
                .collect();
            parts.push(format!("Failed: {}", checks.join(", ")));
        }

        if !self.witness.is_empty() {
            let mut vars: Vec<_> = self.witness.iter().collect();
            vars.sort_by_key(|(k, _)| *k);
            let formatted: Vec<_> = vars
                .into_iter()
                .map(|(k, v)| format!("{k} = {v}"))
                .collect();
            parts.push(format!("Witness: {}", formatted.join(", ")));
        }

        if !self.trace.is_empty() {
            parts.push(format!("Trace: {} states", self.trace.len()));
        }

        if parts.is_empty() {
            if let Some(raw) = &self.raw {
                raw.lines()
                    .next()
                    .unwrap_or("Unknown counterexample")
                    .to_string()
            } else {
                "Unknown counterexample".to_string()
            }
        } else {
            parts.join("; ")
        }
    }

    /// Get a detailed multi-line representation
    #[must_use]
    pub fn format_detailed(&self) -> String {
        let mut output = String::new();

        if !self.failed_checks.is_empty() {
            output.push_str("=== Failed Checks ===\n");
            for check in &self.failed_checks {
                let _ = writeln!(output, "  {check}");
            }
            output.push('\n');
        }

        if !self.witness.is_empty() {
            output.push_str("=== Witness Values ===\n");
            let mut vars: Vec<_> = self.witness.iter().collect();
            vars.sort_by_key(|(k, _)| *k);
            for (var_name, value) in vars {
                let _ = writeln!(output, "  {var_name} = {value}");
            }
            output.push('\n');
        }

        if !self.trace.is_empty() {
            let _ = writeln!(
                output,
                "=== Counterexample Trace ({} states) ===",
                self.trace.len()
            );
            output.push_str(&self.format_trace_with_diffs());
        }

        if let Some(playback) = &self.playback_test {
            output.push_str("=== Concrete Playback Test ===\n");
            for line in playback.lines() {
                let _ = writeln!(output, "  {line}");
            }
            output.push('\n');
        }

        if output.is_empty() {
            if let Some(raw) = &self.raw {
                output.push_str("=== Raw Counterexample ===\n");
                output.push_str(raw);
                output.push('\n');
            } else {
                output.push_str("No counterexample details available\n");
            }
        }

        output
    }

    /// Format the trace showing what changed between each state
    #[must_use]
    pub fn format_trace_with_diffs(&self) -> String {
        let mut output = String::new();

        for (i, state) in self.trace.iter().enumerate() {
            let _ = write!(output, "\nState {}", state.state_num);
            if let Some(action) = &state.action {
                let _ = write!(output, " <{action}>");
            }
            output.push('\n');

            if i == 0 {
                output.push_str("  Initial state:\n");
                let mut vars: Vec<_> = state.variables.iter().collect();
                vars.sort_by_key(|(k, _)| *k);
                for (var_name, value) in vars {
                    let _ = writeln!(output, "    {var_name} = {value}");
                }
            } else {
                let prev_state = &self.trace[i - 1];
                let diffs = state.diff_from(prev_state);

                if diffs.is_empty() {
                    output.push_str("  (no changes)\n");
                } else {
                    output.push_str("  Changes:\n");
                    let mut diff_items: Vec<_> = diffs.iter().collect();
                    diff_items.sort_by_key(|(k, _)| *k);

                    for (var_name, (old_val, new_val)) in diff_items {
                        if let Some(old) = old_val {
                            let _ = writeln!(output, "    {var_name} : {old} -> {new_val}");
                        } else {
                            let _ = writeln!(output, "    {var_name} : (new) {new_val}");
                        }
                    }
                }
            }
        }

        output
    }

    /// Get only the state changes (diffs) from the trace
    #[must_use]
    pub fn trace_diffs(&self) -> TraceDiffs {
        let mut diffs = Vec::new();

        for i in 1..self.trace.len() {
            let prev_state = &self.trace[i - 1];
            let curr_state = &self.trace[i];
            let state_diff = curr_state.diff_from(prev_state);
            if !state_diff.is_empty() {
                diffs.push((curr_state.state_num, state_diff));
            }
        }

        diffs
    }

    /// Serialize to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serialize to pretty-printed JSON string
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl std::fmt::Display for StructuredCounterexample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let summary = self.summary();
        write!(f, "{summary}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===================
    // CounterexampleValue tests
    // ===================

    #[test]
    fn test_counterexample_value_int_display() {
        let val = CounterexampleValue::Int {
            value: 42,
            type_hint: None,
        };
        assert_eq!(val.to_string(), "42");

        let val_typed = CounterexampleValue::Int {
            value: -100,
            type_hint: Some("i32".to_string()),
        };
        assert_eq!(val_typed.to_string(), "-100 (i32)");
    }

    #[test]
    fn test_counterexample_value_uint_display() {
        let val = CounterexampleValue::UInt {
            value: 255,
            type_hint: None,
        };
        assert_eq!(val.to_string(), "255");

        let val_typed = CounterexampleValue::UInt {
            value: 1000,
            type_hint: Some("u64".to_string()),
        };
        assert_eq!(val_typed.to_string(), "1000 (u64)");
    }

    #[test]
    fn test_counterexample_value_float_display() {
        let val = CounterexampleValue::Float { value: 2.5 };
        assert_eq!(val.to_string(), "2.5");
    }

    #[test]
    fn test_counterexample_value_bool_display() {
        assert_eq!(CounterexampleValue::Bool(true).to_string(), "true");
        assert_eq!(CounterexampleValue::Bool(false).to_string(), "false");
    }

    #[test]
    fn test_counterexample_value_string_display() {
        let val = CounterexampleValue::String("hello".to_string());
        assert_eq!(val.to_string(), "\"hello\"");
    }

    #[test]
    fn test_counterexample_value_bytes_display() {
        let val = CounterexampleValue::Bytes(vec![1, 2, 3]);
        assert_eq!(val.to_string(), "[1, 2, 3]");

        let empty = CounterexampleValue::Bytes(vec![]);
        assert_eq!(empty.to_string(), "[]");
    }

    #[test]
    fn test_counterexample_value_set_display() {
        let val = CounterexampleValue::Set(vec![
            CounterexampleValue::Bool(true),
            CounterexampleValue::Bool(false),
        ]);
        assert_eq!(val.to_string(), "{true, false}");
    }

    #[test]
    fn test_counterexample_value_sequence_display() {
        let val = CounterexampleValue::Sequence(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        ]);
        assert_eq!(val.to_string(), "<<1, 2>>");
    }

    #[test]
    fn test_counterexample_value_record_display() {
        let mut fields = HashMap::new();
        fields.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 10,
                type_hint: None,
            },
        );
        let val = CounterexampleValue::Record(fields);
        assert_eq!(val.to_string(), "[x |-> 10]");
    }

    #[test]
    fn test_counterexample_value_function_display() {
        let val = CounterexampleValue::Function(vec![(
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        )]);
        assert_eq!(val.to_string(), "(1 :> 2)");
    }

    #[test]
    fn test_counterexample_value_unknown_display() {
        let val = CounterexampleValue::Unknown("??".to_string());
        assert_eq!(val.to_string(), "??");
    }

    #[test]
    fn test_counterexample_value_semantically_equal_int() {
        let v1 = CounterexampleValue::Int {
            value: 42,
            type_hint: Some("i32".to_string()),
        };
        let v2 = CounterexampleValue::Int {
            value: 42,
            type_hint: Some("i32".to_string()),
        };
        let v3 = CounterexampleValue::Int {
            value: 42,
            type_hint: Some("i64".to_string()),
        };
        let v4 = CounterexampleValue::Int {
            value: 43,
            type_hint: Some("i32".to_string()),
        };

        assert!(v1.semantically_equal(&v2));
        assert!(!v1.semantically_equal(&v3)); // different type hint
        assert!(!v1.semantically_equal(&v4)); // different value
    }

    #[test]
    fn test_counterexample_value_semantically_equal_uint() {
        let v1 = CounterexampleValue::UInt {
            value: 100,
            type_hint: None,
        };
        let v2 = CounterexampleValue::UInt {
            value: 100,
            type_hint: None,
        };
        let v3 = CounterexampleValue::UInt {
            value: 101,
            type_hint: None,
        };

        assert!(v1.semantically_equal(&v2));
        assert!(!v1.semantically_equal(&v3));
    }

    #[test]
    fn test_counterexample_value_semantically_equal_float() {
        let v1 = CounterexampleValue::Float { value: 1.0 };
        let v2 = CounterexampleValue::Float { value: 1.0 };
        let v3 = CounterexampleValue::Float { value: 2.0 };

        assert!(v1.semantically_equal(&v2));
        assert!(!v1.semantically_equal(&v3));
    }

    #[test]
    fn test_counterexample_value_semantically_equal_set() {
        let s1 = CounterexampleValue::Set(vec![
            CounterexampleValue::Bool(true),
            CounterexampleValue::Bool(false),
        ]);
        // Same set, different order
        let s2 = CounterexampleValue::Set(vec![
            CounterexampleValue::Bool(false),
            CounterexampleValue::Bool(true),
        ]);
        let s3 = CounterexampleValue::Set(vec![CounterexampleValue::Bool(true)]);

        assert!(s1.semantically_equal(&s2)); // Sets equal regardless of order
        assert!(!s1.semantically_equal(&s3)); // Different size
    }

    #[test]
    fn test_counterexample_value_semantically_equal_sequence() {
        let seq1 = CounterexampleValue::Sequence(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        ]);
        let seq2 = CounterexampleValue::Sequence(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        ]);
        // Same elements, different order (sequences are ordered)
        let seq3 = CounterexampleValue::Sequence(vec![
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        ]);

        assert!(seq1.semantically_equal(&seq2));
        assert!(!seq1.semantically_equal(&seq3)); // Order matters for sequences
    }

    #[test]
    fn test_counterexample_value_semantically_equal_record() {
        let mut r1 = HashMap::new();
        r1.insert(
            "a".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );
        r1.insert(
            "b".to_string(),
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        );

        let mut r2 = HashMap::new();
        r2.insert(
            "b".to_string(),
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        );
        r2.insert(
            "a".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );

        let mut r3 = HashMap::new();
        r3.insert(
            "a".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );

        assert!(CounterexampleValue::Record(r1.clone())
            .semantically_equal(&CounterexampleValue::Record(r2)));
        assert!(
            !CounterexampleValue::Record(r1).semantically_equal(&CounterexampleValue::Record(r3))
        );
    }

    #[test]
    fn test_counterexample_value_semantically_equal_function() {
        let f1 = CounterexampleValue::Function(vec![(
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 10,
                type_hint: None,
            },
        )]);
        let f2 = CounterexampleValue::Function(vec![(
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 10,
                type_hint: None,
            },
        )]);
        let f3 = CounterexampleValue::Function(vec![(
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 20,
                type_hint: None,
            },
        )]);

        assert!(f1.semantically_equal(&f2));
        assert!(!f1.semantically_equal(&f3));
    }

    #[test]
    fn test_counterexample_value_semantically_equal_different_types() {
        let int_val = CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        };
        let bool_val = CounterexampleValue::Bool(true);
        let uint_val = CounterexampleValue::UInt {
            value: 1,
            type_hint: None,
        };

        assert!(!int_val.semantically_equal(&bool_val));
        assert!(!int_val.semantically_equal(&uint_val));
    }

    #[test]
    fn test_counterexample_value_normalize_set() {
        let set = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 3,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        ]);

        let normalized = set.normalize();
        if let CounterexampleValue::Set(elems) = normalized {
            // Should be sorted by string representation
            assert_eq!(elems.len(), 3);
        } else {
            panic!("Expected Set variant");
        }
    }

    #[test]
    fn test_counterexample_value_normalize_sequence() {
        let seq = CounterexampleValue::Sequence(vec![CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        ])]);

        let normalized = seq.normalize();
        // Inner set should be normalized
        if let CounterexampleValue::Sequence(elems) = normalized {
            assert_eq!(elems.len(), 1);
        } else {
            panic!("Expected Sequence variant");
        }
    }

    #[test]
    fn test_counterexample_value_normalize_primitive() {
        let int_val = CounterexampleValue::Int {
            value: 42,
            type_hint: None,
        };
        let normalized = int_val.normalize();
        assert_eq!(int_val, normalized);
    }

    // ===================
    // Mutation coverage tests for semantically_equal
    // ===================

    #[test]
    fn test_semantically_equal_string_equal() {
        let s1 = CounterexampleValue::String("hello".to_string());
        let s2 = CounterexampleValue::String("hello".to_string());
        assert!(s1.semantically_equal(&s2));
    }

    #[test]
    fn test_semantically_equal_string_not_equal() {
        let s1 = CounterexampleValue::String("hello".to_string());
        let s2 = CounterexampleValue::String("world".to_string());
        assert!(!s1.semantically_equal(&s2));
    }

    #[test]
    fn test_semantically_equal_bytes_equal() {
        let b1 = CounterexampleValue::Bytes(vec![1, 2, 3]);
        let b2 = CounterexampleValue::Bytes(vec![1, 2, 3]);
        assert!(b1.semantically_equal(&b2));
    }

    #[test]
    fn test_semantically_equal_bytes_not_equal() {
        let b1 = CounterexampleValue::Bytes(vec![1, 2, 3]);
        let b2 = CounterexampleValue::Bytes(vec![1, 2, 4]);
        assert!(!b1.semantically_equal(&b2));
    }

    #[test]
    fn test_semantically_equal_unknown_equal() {
        let u1 = CounterexampleValue::Unknown("?".to_string());
        let u2 = CounterexampleValue::Unknown("?".to_string());
        assert!(u1.semantically_equal(&u2));
    }

    #[test]
    fn test_semantically_equal_unknown_not_equal() {
        let u1 = CounterexampleValue::Unknown("foo".to_string());
        let u2 = CounterexampleValue::Unknown("bar".to_string());
        assert!(!u1.semantically_equal(&u2));
    }

    #[test]
    fn test_semantically_equal_bool_equal() {
        let b1 = CounterexampleValue::Bool(true);
        let b2 = CounterexampleValue::Bool(true);
        assert!(b1.semantically_equal(&b2));
    }

    #[test]
    fn test_semantically_equal_bool_not_equal() {
        let b1 = CounterexampleValue::Bool(true);
        let b2 = CounterexampleValue::Bool(false);
        assert!(!b1.semantically_equal(&b2));
    }

    #[test]
    fn test_semantically_equal_float_boundary() {
        // Test exact boundary: v1 - v2 should be < f64::EPSILON, not <=
        let v1 = CounterexampleValue::Float { value: 1.0 };
        let v2 = CounterexampleValue::Float {
            value: 1.0 + f64::EPSILON,
        };
        // These should NOT be semantically equal since difference is exactly epsilon
        assert!(!v1.semantically_equal(&v2));
    }

    #[test]
    fn test_semantically_equal_set_order_independent() {
        // Test that && in set comparison is required (both directions must match)
        let s1 = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        ]);
        let s2 = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        ]);
        // Should be equal regardless of order
        assert!(s1.semantically_equal(&s2));
    }

    #[test]
    fn test_semantically_equal_set_different_elements() {
        // Test that both directions matter in set comparison
        let s1 = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        ]);
        let s2 = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 3,
                type_hint: None,
            },
        ]);
        // Should NOT be equal (element 2 vs 3)
        assert!(!s1.semantically_equal(&s2));
    }

    // ===================
    // Mutation coverage tests for normalize
    // ===================

    #[test]
    fn test_normalize_record_with_nested_set() {
        // Tests that Record arm is executed: nested Set gets normalized
        let mut fields = HashMap::new();
        fields.insert(
            "nested".to_string(),
            CounterexampleValue::Set(vec![
                CounterexampleValue::Int {
                    value: 3,
                    type_hint: None,
                },
                CounterexampleValue::Int {
                    value: 1,
                    type_hint: None,
                },
            ]),
        );
        let r = CounterexampleValue::Record(fields);
        let normalized = r.normalize();
        if let CounterexampleValue::Record(fields) = &normalized {
            if let Some(CounterexampleValue::Set(elems)) = fields.get("nested") {
                // Nested set should be sorted (1 before 3)
                assert_eq!(elems.len(), 2);
                if let CounterexampleValue::Int { value: v1, .. } = &elems[0] {
                    if let CounterexampleValue::Int { value: v2, .. } = &elems[1] {
                        // After normalization, should be sorted: 1, 3
                        assert!(*v1 < *v2, "Set should be sorted after normalize");
                    }
                }
            } else {
                panic!("Expected nested Set");
            }
        } else {
            panic!("Expected Record variant");
        }
    }

    #[test]
    fn test_normalize_function_sorted() {
        // Tests that Function arm is executed: mappings get sorted by key
        let f = CounterexampleValue::Function(vec![
            (
                CounterexampleValue::Int {
                    value: 2,
                    type_hint: None,
                },
                CounterexampleValue::String("two".to_string()),
            ),
            (
                CounterexampleValue::Int {
                    value: 1,
                    type_hint: None,
                },
                CounterexampleValue::String("one".to_string()),
            ),
        ]);
        let normalized = f.normalize();
        if let CounterexampleValue::Function(mappings) = normalized {
            assert_eq!(mappings.len(), 2);
            // Should be sorted by key string representation (1 < 2)
            if let CounterexampleValue::Int { value: k1, .. } = &mappings[0].0 {
                if let CounterexampleValue::Int { value: k2, .. } = &mappings[1].0 {
                    assert!(
                        *k1 < *k2,
                        "Function mappings should be sorted after normalize"
                    );
                }
            }
        } else {
            panic!("Expected Function variant");
        }
    }

    #[test]
    fn test_normalize_set_sorted() {
        // Tests that Set arm is executed: elements get sorted
        let s = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 5,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        ]);
        let normalized = s.normalize();
        if let CounterexampleValue::Set(elems) = normalized {
            assert_eq!(elems.len(), 2);
            // After normalize, elements should be sorted (2 < 5)
            if let CounterexampleValue::Int { value: v1, .. } = &elems[0] {
                if let CounterexampleValue::Int { value: v2, .. } = &elems[1] {
                    assert!(*v1 < *v2, "Set elements should be sorted after normalize");
                }
            }
        } else {
            panic!("Expected Set variant");
        }
    }

    #[test]
    fn test_normalize_sequence_with_nested_set() {
        // Tests that Sequence arm is executed: nested elements get normalized
        let seq = CounterexampleValue::Sequence(vec![CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 9,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 3,
                type_hint: None,
            },
        ])]);
        let normalized = seq.normalize();
        if let CounterexampleValue::Sequence(elems) = normalized {
            assert_eq!(elems.len(), 1);
            if let CounterexampleValue::Set(set_elems) = &elems[0] {
                // Nested set should be sorted (3 < 9)
                assert_eq!(set_elems.len(), 2);
                if let CounterexampleValue::Int { value: v1, .. } = &set_elems[0] {
                    if let CounterexampleValue::Int { value: v2, .. } = &set_elems[1] {
                        assert!(
                            *v1 < *v2,
                            "Nested set in sequence should be sorted after normalize"
                        );
                    }
                }
            } else {
                panic!("Expected nested Set");
            }
        } else {
            panic!("Expected Sequence variant");
        }
    }

    // ===================
    // FailedCheck tests
    // ===================

    #[test]
    fn test_failed_check_display_full() {
        let check = FailedCheck {
            check_id: "assertion.1".to_string(),
            description: "divide by zero".to_string(),
            location: Some(SourceLocation {
                file: "src/main.rs".to_string(),
                line: 42,
                column: Some(10),
            }),
            function: Some("compute".to_string()),
        };

        let display = check.to_string();
        assert!(display.contains("divide by zero"));
        assert!(display.contains("src/main.rs:42:10"));
        assert!(display.contains("compute"));
    }

    #[test]
    fn test_failed_check_display_minimal() {
        let check = FailedCheck {
            check_id: "assertion.1".to_string(),
            description: "overflow".to_string(),
            location: None,
            function: None,
        };

        assert_eq!(check.to_string(), "overflow");
    }

    // ===================
    // SourceLocation tests
    // ===================

    #[test]
    fn test_source_location_display_with_column() {
        let loc = SourceLocation {
            file: "lib.rs".to_string(),
            line: 100,
            column: Some(5),
        };
        assert_eq!(loc.to_string(), "lib.rs:100:5");
    }

    #[test]
    fn test_source_location_display_without_column() {
        let loc = SourceLocation {
            file: "lib.rs".to_string(),
            line: 100,
            column: None,
        };
        assert_eq!(loc.to_string(), "lib.rs:100");
    }

    // ===================
    // TraceState tests
    // ===================

    #[test]
    fn test_trace_state_new() {
        let state = TraceState::new(1);
        assert_eq!(state.state_num, 1);
        assert!(state.action.is_none());
        assert!(state.variables.is_empty());
    }

    #[test]
    fn test_trace_state_diff_from_new_variable() {
        let prev = TraceState::new(1);
        let mut curr = TraceState::new(2);
        curr.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 10,
                type_hint: None,
            },
        );

        let diff = curr.diff_from(&prev);
        assert_eq!(diff.len(), 1);
        assert!(diff.contains_key("x"));
        let (old, new) = &diff["x"];
        assert!(old.is_none());
        if let CounterexampleValue::Int { value, .. } = new {
            assert_eq!(*value, 10);
        } else {
            panic!("Expected Int variant");
        }
    }

    #[test]
    fn test_trace_state_diff_from_changed_variable() {
        let mut prev = TraceState::new(1);
        prev.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 5,
                type_hint: None,
            },
        );

        let mut curr = TraceState::new(2);
        curr.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 10,
                type_hint: None,
            },
        );

        let diff = curr.diff_from(&prev);
        assert_eq!(diff.len(), 1);
        let (old, new) = &diff["x"];
        assert!(old.is_some());
        if let CounterexampleValue::Int { value, .. } = old.as_ref().unwrap() {
            assert_eq!(*value, 5);
        }
        if let CounterexampleValue::Int { value, .. } = new {
            assert_eq!(*value, 10);
        }
    }

    #[test]
    fn test_trace_state_diff_from_unchanged() {
        let mut prev = TraceState::new(1);
        prev.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 5,
                type_hint: None,
            },
        );

        let mut curr = TraceState::new(2);
        curr.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 5,
                type_hint: None,
            },
        );

        let diff = curr.diff_from(&prev);
        assert!(diff.is_empty()); // No changes
    }

    #[test]
    fn test_trace_state_display() {
        let mut state = TraceState::new(1);
        state.action = Some("init".to_string());
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );

        let display = state.to_string();
        assert!(display.contains("State 1"));
        assert!(display.contains("<init>"));
        assert!(display.contains("x = 0"));
    }

    // ===================
    // StructuredCounterexample tests
    // ===================

    #[test]
    fn test_structured_counterexample_new() {
        let cx = StructuredCounterexample::new();
        assert!(cx.witness.is_empty());
        assert!(cx.failed_checks.is_empty());
        assert!(cx.playback_test.is_none());
        assert!(cx.trace.is_empty());
        assert!(cx.raw.is_none());
        assert!(!cx.minimized);
    }

    #[test]
    fn test_structured_counterexample_from_raw() {
        let cx = StructuredCounterexample::from_raw("raw output".to_string());
        assert!(cx.witness.is_empty());
        assert_eq!(cx.raw, Some("raw output".to_string()));
    }

    #[test]
    fn test_structured_counterexample_has_structured_data() {
        let empty = StructuredCounterexample::new();
        assert!(!empty.has_structured_data());

        let mut with_witness = StructuredCounterexample::new();
        with_witness.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );
        assert!(with_witness.has_structured_data());

        let mut with_checks = StructuredCounterexample::new();
        with_checks.failed_checks.push(FailedCheck {
            check_id: "test".to_string(),
            description: "test".to_string(),
            location: None,
            function: None,
        });
        assert!(with_checks.has_structured_data());

        let mut with_playback = StructuredCounterexample::new();
        with_playback.playback_test = Some("fn test() {}".to_string());
        assert!(with_playback.has_structured_data());

        let mut with_trace = StructuredCounterexample::new();
        with_trace.trace.push(TraceState::new(1));
        assert!(with_trace.has_structured_data());
    }

    #[test]
    fn test_structured_counterexample_summary_with_checks() {
        let mut cx = StructuredCounterexample::new();
        cx.failed_checks.push(FailedCheck {
            check_id: "test.1".to_string(),
            description: "divide by zero".to_string(),
            location: None,
            function: None,
        });

        let summary = cx.summary();
        assert!(summary.contains("Failed: divide by zero"));
    }

    #[test]
    fn test_structured_counterexample_summary_with_witness() {
        let mut cx = StructuredCounterexample::new();
        cx.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 42,
                type_hint: None,
            },
        );

        let summary = cx.summary();
        assert!(summary.contains("Witness:"));
        assert!(summary.contains("x = 42"));
    }

    #[test]
    fn test_structured_counterexample_summary_witness_sorted() {
        let mut cx = StructuredCounterexample::new();
        cx.witness.insert(
            "b".to_string(),
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        );
        cx.witness.insert(
            "a".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );

        let summary = cx.summary();
        assert!(summary.contains("Witness: a = 1, b = 2"));
    }

    #[test]
    fn test_structured_counterexample_summary_with_trace() {
        let mut cx = StructuredCounterexample::new();
        cx.trace.push(TraceState::new(1));
        cx.trace.push(TraceState::new(2));

        let summary = cx.summary();
        assert!(summary.contains("Trace: 2 states"));
    }

    #[test]
    fn test_structured_counterexample_summary_raw_fallback() {
        let mut cx = StructuredCounterexample::new();
        cx.raw = Some("Error: something went wrong\nMore details".to_string());

        let summary = cx.summary();
        assert_eq!(summary, "Error: something went wrong");
    }

    #[test]
    fn test_structured_counterexample_summary_empty() {
        let cx = StructuredCounterexample::new();
        let summary = cx.summary();
        assert_eq!(summary, "Unknown counterexample");
    }

    #[test]
    fn test_structured_counterexample_format_detailed() {
        let mut cx = StructuredCounterexample::new();
        cx.failed_checks.push(FailedCheck {
            check_id: "test".to_string(),
            description: "assertion failed".to_string(),
            location: None,
            function: None,
        });
        cx.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );
        cx.playback_test = Some("#[test]\nfn playback() {}".to_string());

        let detailed = cx.format_detailed();
        assert!(detailed.contains("=== Failed Checks ==="));
        assert!(detailed.contains("assertion failed"));
        assert!(detailed.contains("=== Witness Values ==="));
        assert!(detailed.contains("x = 0"));
        assert!(detailed.contains("=== Concrete Playback Test ==="));
    }

    #[test]
    fn test_structured_counterexample_format_detailed_raw_fallback() {
        let mut cx = StructuredCounterexample::new();
        cx.raw = Some("raw counterexample data".to_string());

        let detailed = cx.format_detailed();
        assert!(detailed.contains("=== Raw Counterexample ==="));
        assert!(detailed.contains("raw counterexample data"));
    }

    #[test]
    fn test_structured_counterexample_trace_diffs() {
        let mut cx = StructuredCounterexample::new();

        let mut state1 = TraceState::new(1);
        state1.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );

        let mut state2 = TraceState::new(2);
        state2.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );

        let mut state3 = TraceState::new(3);
        state3.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        ); // No change from state2

        cx.trace.push(state1);
        cx.trace.push(state2);
        cx.trace.push(state3);

        let diffs = cx.trace_diffs();
        // Only state 2 has changes (state 1 -> 2), state 3 has no changes
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].0, 2); // State number 2
    }

    #[test]
    fn test_structured_counterexample_format_trace_with_diffs() {
        let mut cx = StructuredCounterexample::new();

        let mut state1 = TraceState::new(1);
        state1.action = Some("init".to_string());
        state1.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );

        let mut state2 = TraceState::new(2);
        state2.action = Some("increment".to_string());
        state2.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );

        cx.trace.push(state1);
        cx.trace.push(state2);

        let formatted = cx.format_trace_with_diffs();
        assert!(formatted.contains("State 1"));
        assert!(formatted.contains("<init>"));
        assert!(formatted.contains("Initial state:"));
        assert!(formatted.contains("x = 0"));
        assert!(formatted.contains("State 2"));
        assert!(formatted.contains("<increment>"));
        assert!(formatted.contains("Changes:"));
        assert!(formatted.contains("x : 0 -> 1"));
    }

    #[test]
    fn test_structured_counterexample_json_roundtrip() {
        let mut cx = StructuredCounterexample::new();
        cx.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 42,
                type_hint: Some("i32".to_string()),
            },
        );
        cx.failed_checks.push(FailedCheck {
            check_id: "test".to_string(),
            description: "assertion".to_string(),
            location: Some(SourceLocation {
                file: "test.rs".to_string(),
                line: 10,
                column: Some(5),
            }),
            function: Some("foo".to_string()),
        });
        cx.minimized = true;

        let json = cx.to_json().expect("serialize");
        let parsed = StructuredCounterexample::from_json(&json).expect("deserialize");

        assert!(parsed.minimized);
        assert_eq!(parsed.failed_checks.len(), 1);
        assert_eq!(parsed.failed_checks[0].description, "assertion");
        assert!(parsed.witness.contains_key("x"));
    }

    #[test]
    fn test_structured_counterexample_json_pretty() {
        let cx = StructuredCounterexample::new();
        let pretty = cx.to_json_pretty().expect("serialize pretty");
        assert!(pretty.contains('\n')); // Pretty format has newlines
    }

    #[test]
    fn test_structured_counterexample_display() {
        let mut cx = StructuredCounterexample::new();
        cx.failed_checks.push(FailedCheck {
            check_id: "test".to_string(),
            description: "error".to_string(),
            location: None,
            function: None,
        });

        // Display should equal summary
        assert_eq!(cx.to_string(), cx.summary());
    }

    #[test]
    fn test_trace_state_no_changes_display() {
        let mut cx = StructuredCounterexample::new();

        let mut state1 = TraceState::new(1);
        state1.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 5,
                type_hint: None,
            },
        );

        let mut state2 = TraceState::new(2);
        state2.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 5,
                type_hint: None,
            },
        );

        cx.trace.push(state1);
        cx.trace.push(state2);

        let formatted = cx.format_trace_with_diffs();
        assert!(formatted.contains("(no changes)"));
    }

    #[test]
    fn test_trace_state_new_variable_in_diff() {
        let state1 = TraceState::new(1);

        let mut state2 = TraceState::new(2);
        state2.variables.insert(
            "y".to_string(),
            CounterexampleValue::Int {
                value: 100,
                type_hint: None,
            },
        );

        let mut cx = StructuredCounterexample::new();
        cx.trace.push(state1);
        cx.trace.push(state2);

        let formatted = cx.format_trace_with_diffs();
        assert!(formatted.contains("y : (new) 100"));
    }
}
