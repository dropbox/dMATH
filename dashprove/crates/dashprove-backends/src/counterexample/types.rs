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

/// Type alias for state diffs: variable name -> (old_value, new_value)
pub type StateDiff = HashMap<String, (Option<CounterexampleValue>, CounterexampleValue)>;

/// Type alias for a sequence of state diffs with state numbers
pub type TraceDiffs = Vec<(u32, StateDiff)>;

/// A concrete value in a counterexample
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CounterexampleValue {
    /// Integer value with optional type info (e.g., "u32", "i64")
    Int {
        /// The integer value
        value: i128,
        /// Optional type annotation (e.g., "i32", "i64")
        type_hint: Option<String>,
    },
    /// Unsigned integer value
    UInt {
        /// The unsigned integer value
        value: u128,
        /// Optional type annotation (e.g., "u32", "u64")
        type_hint: Option<String>,
    },
    /// Floating point value
    Float {
        /// The floating point value
        value: f64,
    },
    /// Boolean value
    Bool(bool),
    /// String value
    String(String),
    /// Raw bytes (from Kani concrete playback)
    Bytes(Vec<u8>),
    /// Set of values (e.g., TLA+ {1, 2, 3})
    Set(Vec<CounterexampleValue>),
    /// Sequence/tuple of values (e.g., TLA+ <<1, 2, 3>>)
    Sequence(Vec<CounterexampleValue>),
    /// Record/struct with named fields (e.g., TLA+ [a |-> 1, b |-> 2])
    Record(HashMap<String, CounterexampleValue>),
    /// Function/mapping (e.g., TLA+ (1 :> "a" @@ 2 :> "b"))
    Function(Vec<(CounterexampleValue, CounterexampleValue)>),
    /// Unknown/unparsed value
    Unknown(String),
}

impl CounterexampleValue {
    /// Check if two values are semantically equivalent.
    /// This differs from `PartialEq` in that:
    /// - Sets are compared regardless of element order
    /// - Records are compared regardless of field order
    /// - Functions are compared regardless of mapping order
    /// - Sequences preserve order (they are positional)
    #[must_use]
    pub fn semantically_equal(&self, other: &Self) -> bool {
        match (self, other) {
            // Primitives use standard equality
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

            // Sets: order doesn't matter
            (CounterexampleValue::Set(s1), CounterexampleValue::Set(s2)) => {
                if s1.len() != s2.len() {
                    return false;
                }
                // Each element in s1 must have a semantic match in s2
                s1.iter()
                    .all(|e1| s2.iter().any(|e2| e1.semantically_equal(e2)))
                    && s2
                        .iter()
                        .all(|e2| s1.iter().any(|e1| e1.semantically_equal(e2)))
            }

            // Sequences: order matters
            (CounterexampleValue::Sequence(seq1), CounterexampleValue::Sequence(seq2)) => {
                seq1.len() == seq2.len()
                    && seq1
                        .iter()
                        .zip(seq2.iter())
                        .all(|(a, b)| a.semantically_equal(b))
            }

            // Records: field order doesn't matter
            (CounterexampleValue::Record(r1), CounterexampleValue::Record(r2)) => {
                if r1.len() != r2.len() {
                    return false;
                }
                r1.iter()
                    .all(|(k, v1)| r2.get(k).is_some_and(|v2| v1.semantically_equal(v2)))
            }

            // Functions: mapping order doesn't matter
            (CounterexampleValue::Function(f1), CounterexampleValue::Function(f2)) => {
                if f1.len() != f2.len() {
                    return false;
                }
                // Each mapping in f1 must have a semantic match in f2
                f1.iter().all(|(k1, v1)| {
                    f2.iter()
                        .any(|(k2, v2)| k1.semantically_equal(k2) && v1.semantically_equal(v2))
                })
            }

            // Unknown values: string comparison
            (CounterexampleValue::Unknown(s1), CounterexampleValue::Unknown(s2)) => s1 == s2,

            // Different types are not equal
            _ => false,
        }
    }

    /// Normalize a value for canonical comparison
    /// - Sorts set elements
    /// - Sorts record fields by key
    /// - Sorts function mappings by key
    #[must_use]
    pub fn normalize(&self) -> Self {
        match self {
            CounterexampleValue::Set(elems) => {
                let mut normalized: Vec<_> = elems.iter().map(|e| e.normalize()).collect();
                // Sort by display representation for deterministic order
                normalized.sort_by_key(|a| a.to_string());
                CounterexampleValue::Set(normalized)
            }
            CounterexampleValue::Sequence(elems) => {
                CounterexampleValue::Sequence(elems.iter().map(|e| e.normalize()).collect())
            }
            CounterexampleValue::Record(fields) => {
                // HashMap doesn't preserve order, but normalization ensures consistent iteration
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
                // Sort by key representation
                normalized.sort_by(|a, b| a.0.to_string().cmp(&b.0.to_string()));
                CounterexampleValue::Function(normalized)
            }
            // Primitives are already normalized
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
                // Sort fields by key for deterministic output
                let mut items: Vec<_> = fields.iter().collect();
                items.sort_by(|a, b| a.0.cmp(b.0));
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

/// A single state in a counterexample trace (e.g., from TLA+ TLC output)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraceState {
    /// State number in the trace (1-indexed)
    pub state_num: u32,
    /// Action/transition that led to this state (e.g., "Initial predicate", "Next line 12...")
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
    /// Returns a map of variable names to (old_value, new_value) tuples
    /// Only includes variables that changed
    #[must_use]
    pub fn diff_from(&self, previous: &Self) -> StateDiff {
        let mut changes = HashMap::new();

        // Find variables that changed or are new
        for (var_name, new_value) in &self.variables {
            match previous.variables.get(var_name) {
                Some(old_value) if old_value != new_value => {
                    changes.insert(
                        var_name.clone(),
                        (Some(old_value.clone()), new_value.clone()),
                    );
                }
                None => {
                    // New variable
                    changes.insert(var_name.clone(), (None, new_value.clone()));
                }
                _ => {} // No change
            }
        }

        changes
    }
}

impl std::fmt::Display for TraceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "State {}", self.state_num)?;
        if let Some(ref action) = self.action {
            write!(f, " <{action}>")?;
        }
        writeln!(f)?;

        // Sort variables for consistent output
        let mut vars: Vec<_> = self.variables.iter().collect();
        vars.sort_by(|a, b| a.0.cmp(b.0));

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
        write!(f, "{}", self.description)?;
        if let Some(ref loc) = self.location {
            write!(f, " at {loc}")?;
        }
        if let Some(ref func) = self.function {
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
    /// Sequence of states forming a counterexample trace (e.g., from TLA+/TLC)
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
            let joined = checks.join(", ");
            parts.push(format!("Failed: {joined}"));
        }

        if !self.witness.is_empty() {
            let vars: Vec<_> = self
                .witness
                .iter()
                .map(|(k, v)| format!("{k} = {v}"))
                .collect();
            let joined = vars.join(", ");
            parts.push(format!("Witness: {joined}"));
        }

        if !self.trace.is_empty() {
            let len = self.trace.len();
            parts.push(format!("Trace: {len} states"));
        }

        if parts.is_empty() {
            if let Some(ref raw) = self.raw {
                // Return first line of raw
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

    /// Get a detailed multi-line representation of the counterexample
    #[must_use]
    pub fn format_detailed(&self) -> String {
        let mut output = String::new();

        // Failed checks section
        if !self.failed_checks.is_empty() {
            output.push_str("=== Failed Checks ===\n");
            for check in &self.failed_checks {
                output.push_str(&format!("  {check}\n"));
            }
            output.push('\n');
        }

        // Witness values section
        if !self.witness.is_empty() {
            output.push_str("=== Witness Values ===\n");
            let mut vars: Vec<_> = self.witness.iter().collect();
            vars.sort_by(|a, b| a.0.cmp(b.0));
            for (var_name, value) in vars {
                output.push_str(&format!("  {var_name} = {value}\n"));
            }
            output.push('\n');
        }

        // Trace section with state diffs
        if !self.trace.is_empty() {
            let len = self.trace.len();
            output.push_str(&format!("=== Counterexample Trace ({len} states) ===\n"));
            output.push_str(&self.format_trace_with_diffs());
        }

        // Playback test section
        if let Some(ref playback) = self.playback_test {
            output.push_str("=== Concrete Playback Test ===\n");
            for line in playback.lines() {
                output.push_str(&format!("  {line}\n"));
            }
            output.push('\n');
        }

        if output.is_empty() {
            if let Some(ref raw) = self.raw {
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
            // State header
            let state_num = state.state_num;
            output.push_str(&format!("\nState {state_num}"));
            if let Some(ref action) = state.action {
                output.push_str(&format!(" <{action}>"));
            }
            output.push('\n');

            if i == 0 {
                // First state: show all variables
                output.push_str("  Initial state:\n");
                let mut vars: Vec<_> = state.variables.iter().collect();
                vars.sort_by(|a, b| a.0.cmp(b.0));
                for (var_name, value) in vars {
                    output.push_str(&format!("    {var_name} = {value}\n"));
                }
            } else {
                // Subsequent states: show diffs
                let prev_state = &self.trace[i - 1];
                let diffs = state.diff_from(prev_state);

                if diffs.is_empty() {
                    output.push_str("  (no changes)\n");
                } else {
                    output.push_str("  Changes:\n");
                    let mut diff_items: Vec<_> = diffs.iter().collect();
                    diff_items.sort_by(|a, b| a.0.cmp(b.0));

                    for (var_name, (old_val, new_val)) in diff_items {
                        if let Some(old) = old_val {
                            output.push_str(&format!("    {var_name} : {old} -> {new_val}\n"));
                        } else {
                            output.push_str(&format!("    {var_name} : (new) {new_val}\n"));
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
    ///
    /// # Errors
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serialize to pretty-printed JSON string
    ///
    /// # Errors
    /// Returns an error if serialization fails.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string
    ///
    /// # Errors
    /// Returns an error if the JSON is invalid or doesn't match the expected structure.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize to JSON Value (for programmatic manipulation)
    ///
    /// # Errors
    /// Returns an error if serialization fails.
    pub fn to_json_value(&self) -> Result<serde_json::Value, serde_json::Error> {
        serde_json::to_value(self)
    }
}

impl std::fmt::Display for StructuredCounterexample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ============================================================================
// Kani Proof Harnesses
// ============================================================================
// These formal verification harnesses prove properties about counterexample types
// using bounded model checking. Run with: cargo kani -p dashprove-backends

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // NOTE: Proofs for TraceState::new(), StructuredCounterexample::new(), and
    // HashMap::new() are NOT included here because HashMap's RandomState uses
    // CCRandomGenerateBytes for hash randomization, which Kani doesn't support.
    // See: https://github.com/model-checking/kani/issues/2423
    // These properties are thoroughly covered by regular unit tests instead.

    /// Prove that semantic equality is reflexive: x.semantically_equal(x) == true
    #[kani::proof]
    fn verify_semantic_equality_reflexive_bool() {
        let val = CounterexampleValue::Bool(kani::any());
        kani::assert(val.semantically_equal(&val), "Bool should be reflexive");
    }

    /// Prove that semantic equality for Int is reflexive
    #[kani::proof]
    fn verify_semantic_equality_reflexive_int() {
        let value: i128 = kani::any();
        let val = CounterexampleValue::Int {
            value,
            type_hint: None,
        };
        kani::assert(val.semantically_equal(&val), "Int should be reflexive");
    }

    /// Prove that semantic equality for UInt is reflexive
    #[kani::proof]
    fn verify_semantic_equality_reflexive_uint() {
        let value: u128 = kani::any();
        let val = CounterexampleValue::UInt {
            value,
            type_hint: None,
        };
        kani::assert(val.semantically_equal(&val), "UInt should be reflexive");
    }

    /// Prove that different types are not equal
    #[kani::proof]
    fn verify_different_types_not_equal() {
        let bool_val = CounterexampleValue::Bool(true);
        let int_val = CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        };
        kani::assert(
            !bool_val.semantically_equal(&int_val),
            "Different types should not be equal",
        );
    }

    /// Prove that Bool(true) != Bool(false)
    #[kani::proof]
    fn verify_bool_values_distinct() {
        let true_val = CounterexampleValue::Bool(true);
        let false_val = CounterexampleValue::Bool(false);
        kani::assert(
            !true_val.semantically_equal(&false_val),
            "Bool(true) != Bool(false)",
        );
    }

    /// Prove that normalize on Bool returns semantically equal value
    #[kani::proof]
    fn verify_normalize_bool_is_clone() {
        let b: bool = kani::any();
        let val = CounterexampleValue::Bool(b);
        let normalized = val.normalize();
        kani::assert(
            val.semantically_equal(&normalized),
            "Normalized Bool should equal original",
        );
    }

    /// Prove that empty sets are equal
    #[kani::proof]
    fn verify_empty_sets_equal() {
        let set1 = CounterexampleValue::Set(vec![]);
        let set2 = CounterexampleValue::Set(vec![]);
        kani::assert(set1.semantically_equal(&set2), "Empty sets should be equal");
    }

    /// Prove that empty sequences are equal
    #[kani::proof]
    fn verify_empty_sequences_equal() {
        let seq1 = CounterexampleValue::Sequence(vec![]);
        let seq2 = CounterexampleValue::Sequence(vec![]);
        kani::assert(
            seq1.semantically_equal(&seq2),
            "Empty sequences should be equal",
        );
    }

    /// Prove that empty functions are equal
    #[kani::proof]
    fn verify_empty_functions_equal() {
        let func1 = CounterexampleValue::Function(vec![]);
        let func2 = CounterexampleValue::Function(vec![]);
        kani::assert(
            func1.semantically_equal(&func2),
            "Empty functions should be equal",
        );
    }

    /// Prove that set semantic equality is order independent
    #[kani::proof]
    fn verify_set_semantic_equality_order_independent() {
        let set1 = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        ]);
        let set2 = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        ]);
        kani::assert(
            set1.semantically_equal(&set2),
            "Set equality should ignore element order",
        );
    }

    /// Prove that sequence semantic equality respects element order
    #[kani::proof]
    fn verify_sequence_semantic_equality_ordered() {
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
                value: 2,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        ]);
        kani::assert(
            !seq1.semantically_equal(&seq2),
            "Sequence equality must respect element order",
        );
    }

    /// Prove that normalize sorts set elements deterministically
    #[kani::proof]
    fn verify_normalize_sorts_sets() {
        let set = CounterexampleValue::Set(vec![
            CounterexampleValue::String("b".to_string()),
            CounterexampleValue::String("a".to_string()),
        ]);
        let normalized = set.normalize();
        if let CounterexampleValue::Set(items) = normalized {
            kani::assert(items.len() == 2, "normalized set must keep length");
            let first = items[0].to_string();
            let second = items[1].to_string();
            kani::assert(
                first <= second,
                "normalized set order must be deterministic and sorted lexicographically",
            );
        } else {
            kani::assert(false, "normalize should keep set variant");
        }
    }
}
