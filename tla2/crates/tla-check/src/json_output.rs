//! JSON output format for AI agents and automated tooling.
//!
//! This module provides structured JSON output designed for machine consumption,
//! with explicit types, source locations, and state diffs.

use crate::{CheckResult, CheckStats, Trace, Value};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

/// Version of the JSON output format
pub const OUTPUT_VERSION: &str = "1.0";

/// Complete JSON output for model checking results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonOutput {
    /// Schema version
    pub version: String,
    /// Tool identifier
    pub tool: String,
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Input files and configuration
    pub input: InputInfo,
    /// Specification details
    pub specification: SpecInfo,
    /// Model checking result
    pub result: ResultInfo,
    /// Counterexample trace (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counterexample: Option<CounterexampleInfo>,
    /// Statistics
    pub statistics: StatisticsInfo,
    /// Diagnostic messages
    pub diagnostics: DiagnosticsInfo,
    /// Action coverage information
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub actions_detected: Vec<ActionInfo>,
}

/// Input file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputInfo {
    pub spec_file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_file: Option<String>,
    pub module: String,
    pub workers: usize,
}

/// Specification structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub init: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next: Option<String>,
    pub invariants: Vec<String>,
    pub properties: Vec<String>,
    pub constants: HashMap<String, JsonValue>,
    pub variables: Vec<String>,
}

/// Model checking result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultInfo {
    /// Status: "ok", "error", "timeout", "interrupted", "limit_reached"
    pub status: String,
    /// Error type if status is "error"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
    /// Structured error code for programmatic handling
    ///
    /// Error codes follow a prefix convention:
    /// - `TLC_` - Model checker errors (deadlock, invariant violation, etc.)
    /// - `CFG_` - Configuration file parsing errors
    /// - `TLA_` - TLA+ source parsing errors
    /// - `SYS_` - System/runtime errors
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    /// Human-readable error message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Details about violated property
    #[serde(skip_serializing_if = "Option::is_none")]
    pub violated_property: Option<ViolatedProperty>,
    /// Actionable suggestion for fixing the error
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<ErrorSuggestion>,
}

/// Error codes for structured error handling
pub mod error_codes {
    // Model checker errors (TLC_*)
    pub const TLC_DEADLOCK: &str = "TLC_DEADLOCK";
    pub const TLC_INVARIANT_VIOLATED: &str = "TLC_INVARIANT_VIOLATED";
    pub const TLC_PROPERTY_VIOLATED: &str = "TLC_PROPERTY_VIOLATED";
    pub const TLC_LIVENESS_VIOLATED: &str = "TLC_LIVENESS_VIOLATED";
    pub const TLC_EVAL_ERROR: &str = "TLC_EVAL_ERROR";
    pub const TLC_TYPE_MISMATCH: &str = "TLC_TYPE_MISMATCH";
    pub const TLC_UNDEFINED_VAR: &str = "TLC_UNDEFINED_VAR";
    pub const TLC_UNDEFINED_OP: &str = "TLC_UNDEFINED_OP";
    pub const TLC_LIMIT_REACHED: &str = "TLC_LIMIT_REACHED";

    // Configuration errors (CFG_*)
    pub const CFG_PARSE_ERROR: &str = "CFG_PARSE_ERROR";
    pub const CFG_MISSING_INIT: &str = "CFG_MISSING_INIT";
    pub const CFG_MISSING_NEXT: &str = "CFG_MISSING_NEXT";
    pub const CFG_UNSUPPORTED_SYNTAX: &str = "CFG_UNSUPPORTED_SYNTAX";

    // TLA+ parsing errors (TLA_*)
    pub const TLA_PARSE_ERROR: &str = "TLA_PARSE_ERROR";
    pub const TLA_LOWER_ERROR: &str = "TLA_LOWER_ERROR";

    // System errors (SYS_*)
    pub const SYS_IO_ERROR: &str = "SYS_IO_ERROR";
    pub const SYS_TIMEOUT: &str = "SYS_TIMEOUT";
}

/// Actionable suggestion for fixing an error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSuggestion {
    /// Brief description of the suggested action
    pub action: String,
    /// Example code or configuration fix
    #[serde(skip_serializing_if = "Option::is_none")]
    pub example: Option<String>,
    /// Alternative options if applicable
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub options: Vec<String>,
}

/// Information about a violated property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolatedProperty {
    pub name: String,
    /// Type: "invariant", "liveness", "assertion"
    #[serde(rename = "type")]
    pub prop_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<SourceLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,
}

/// Source code location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub line: usize,
    pub column: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_line: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_column: Option<usize>,
}

/// Counterexample trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterexampleInfo {
    pub length: usize,
    pub states: Vec<StateInfo>,
    /// For liveness violations: index where the cycle begins
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loop_start: Option<usize>,
}

/// A single state in a trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateInfo {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<String>,
    pub action: ActionRef,
    pub variables: HashMap<String, JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diff_from_previous: Option<StateDiff>,
}

/// Reference to an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRef {
    pub name: String,
    /// Type: "initial", "next"
    #[serde(rename = "type")]
    pub action_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<SourceLocation>,
}

/// Diff between two states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDiff {
    pub changed: HashMap<String, ValueChange>,
    pub unchanged: Vec<String>,
}

/// A value change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueChange {
    pub from: JsonValue,
    pub to: JsonValue,
}

/// Typed JSON value representation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum JsonValue {
    #[serde(rename = "bool")]
    Bool(bool),
    #[serde(rename = "int")]
    Int(i64),
    #[serde(rename = "string")]
    String(String),
    #[serde(rename = "set")]
    Set(Vec<JsonValue>),
    #[serde(rename = "seq")]
    Seq(Vec<JsonValue>),
    #[serde(rename = "record")]
    Record(HashMap<String, JsonValue>),
    #[serde(rename = "function")]
    Function {
        domain: Vec<JsonValue>,
        mapping: Vec<(JsonValue, JsonValue)>,
    },
    #[serde(rename = "tuple")]
    Tuple(Vec<JsonValue>),
    #[serde(rename = "model_value")]
    ModelValue(String),
    #[serde(rename = "undefined")]
    Undefined,
}

/// Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsInfo {
    pub states_found: usize,
    pub states_initial: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub states_distinct: Option<usize>,
    pub transitions: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_depth: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_queue_depth: Option<usize>,
    pub time_seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub states_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_mb: Option<f64>,
}

/// Diagnostic messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticsInfo {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<DiagnosticMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub info: Vec<DiagnosticMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub print_outputs: Vec<PrintOutput>,
}

/// A diagnostic message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticMessage {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<SourceLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

/// Print statement output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrintOutput {
    pub value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<SourceLocation>,
}

/// Action coverage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionInfo {
    pub name: String,
    pub occurrences: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub percentage: Option<f64>,
}

// ============================================================================
// Conversion implementations
// ============================================================================

impl JsonOutput {
    /// Create a new JSON output structure
    pub fn new(
        spec_file: &Path,
        config_file: Option<&Path>,
        module_name: &str,
        workers: usize,
    ) -> Self {
        let now = chrono::Utc::now();
        Self {
            version: OUTPUT_VERSION.to_string(),
            tool: "tla2".to_string(),
            timestamp: now.to_rfc3339(),
            input: InputInfo {
                spec_file: spec_file.display().to_string(),
                config_file: config_file.map(|p| p.display().to_string()),
                module: module_name.to_string(),
                workers,
            },
            specification: SpecInfo {
                init: None,
                next: None,
                invariants: Vec::new(),
                properties: Vec::new(),
                constants: HashMap::new(),
                variables: Vec::new(),
            },
            result: ResultInfo {
                status: "ok".to_string(),
                error_type: None,
                error_code: None,
                error_message: None,
                violated_property: None,
                suggestion: None,
            },
            counterexample: None,
            statistics: StatisticsInfo {
                states_found: 0,
                states_initial: 0,
                states_distinct: None,
                transitions: 0,
                max_depth: None,
                max_queue_depth: None,
                time_seconds: 0.0,
                states_per_second: None,
                memory_mb: None,
            },
            diagnostics: DiagnosticsInfo {
                warnings: Vec::new(),
                info: Vec::new(),
                print_outputs: Vec::new(),
            },
            actions_detected: Vec::new(),
        }
    }

    /// Set specification info
    pub fn with_spec_info(
        mut self,
        init: Option<&str>,
        next: Option<&str>,
        invariants: Vec<String>,
        properties: Vec<String>,
        variables: Vec<String>,
    ) -> Self {
        self.specification.init = init.map(String::from);
        self.specification.next = next.map(String::from);
        self.specification.invariants = invariants;
        self.specification.properties = properties;
        self.specification.variables = variables;
        self
    }

    /// Set result from CheckResult
    pub fn with_check_result(mut self, result: &CheckResult, elapsed: Duration) -> Self {
        match result {
            CheckResult::Success(stats) => {
                self.result.status = "ok".to_string();
                self.statistics = stats_to_json(stats, elapsed);
                // Use coverage stats if available, otherwise just list action names
                if let Some(ref coverage) = stats.coverage {
                    self.actions_detected = coverage
                        .action_order
                        .iter()
                        .filter_map(|name| coverage.actions.get(name))
                        .map(|a| ActionInfo {
                            name: a.name.clone(),
                            occurrences: a.transitions,
                            percentage: if coverage.total_transitions > 0 {
                                Some(
                                    a.transitions as f64 / coverage.total_transitions as f64
                                        * 100.0,
                                )
                            } else {
                                None
                            },
                        })
                        .collect();
                } else if !stats.detected_actions.is_empty() {
                    self.actions_detected = stats
                        .detected_actions
                        .iter()
                        .map(|name| ActionInfo {
                            name: name.clone(),
                            occurrences: 0,
                            percentage: None,
                        })
                        .collect();
                }
            }
            CheckResult::InvariantViolation {
                invariant,
                trace,
                stats,
            } => {
                self.result.status = "error".to_string();
                self.result.error_type = Some("invariant_violation".to_string());
                self.result.error_code = Some(error_codes::TLC_INVARIANT_VIOLATED.to_string());
                self.result.error_message = Some(format!("Invariant '{}' violated", invariant));
                self.result.violated_property = Some(ViolatedProperty {
                    name: invariant.clone(),
                    prop_type: "invariant".to_string(),
                    location: None,
                    expression: None,
                });
                self.result.suggestion = Some(ErrorSuggestion {
                    action: "Examine the counterexample trace to understand why the invariant fails".to_string(),
                    example: Some(format!(
                        "Check the definition of {} and verify it holds for the final state in the trace",
                        invariant
                    )),
                    options: vec![
                        "Add CONSTRAINT to limit state space".to_string(),
                        "Strengthen invariant preconditions".to_string(),
                        "Fix the Next action that causes the violation".to_string(),
                    ],
                });
                self.counterexample = Some(trace_to_counterexample(trace, None));
                self.statistics = stats_to_json(stats, elapsed);
            }
            CheckResult::PropertyViolation {
                property,
                trace,
                stats,
            } => {
                self.result.status = "error".to_string();
                self.result.error_type = Some("property_violation".to_string());
                self.result.error_code = Some(error_codes::TLC_PROPERTY_VIOLATED.to_string());
                self.result.error_message = Some(format!("Property '{}' violated", property));
                self.result.violated_property = Some(ViolatedProperty {
                    name: property.clone(),
                    prop_type: "property".to_string(),
                    location: None,
                    expression: None,
                });
                self.result.suggestion = Some(ErrorSuggestion {
                    action: "Review the counterexample trace to identify the property violation"
                        .to_string(),
                    example: None,
                    options: vec![],
                });
                self.counterexample = Some(trace_to_counterexample(trace, None));
                self.statistics = stats_to_json(stats, elapsed);
            }
            CheckResult::LivenessViolation {
                property,
                prefix,
                cycle,
                stats,
            } => {
                self.result.status = "error".to_string();
                self.result.error_type = Some("liveness_violation".to_string());
                self.result.error_code = Some(error_codes::TLC_LIVENESS_VIOLATED.to_string());
                self.result.error_message =
                    Some(format!("Liveness property '{}' violated", property));
                self.result.violated_property = Some(ViolatedProperty {
                    name: property.clone(),
                    prop_type: "liveness".to_string(),
                    location: None,
                    expression: None,
                });
                self.result.suggestion = Some(ErrorSuggestion {
                    action:
                        "The trace shows a cycle where the liveness property never becomes true"
                            .to_string(),
                    example: Some(
                        "Check for missing fairness constraints (WF_ or SF_)".to_string(),
                    ),
                    options: vec![
                        "Add weak fairness: WF_vars(Action)".to_string(),
                        "Add strong fairness: SF_vars(Action)".to_string(),
                        "Check for blocking conditions in the cycle".to_string(),
                    ],
                });
                // For liveness violations, combine prefix and cycle into one trace
                // with loop_start indicating where the cycle begins
                let loop_start = prefix.states.len();
                let mut combined_trace = prefix.clone();
                for state in &cycle.states {
                    combined_trace.states.push(state.clone());
                }
                self.counterexample =
                    Some(trace_to_counterexample(&combined_trace, Some(loop_start)));
                self.statistics = stats_to_json(stats, elapsed);
            }
            CheckResult::Deadlock { trace, stats } => {
                self.result.status = "error".to_string();
                self.result.error_type = Some("deadlock".to_string());
                self.result.error_code = Some(error_codes::TLC_DEADLOCK.to_string());
                self.result.error_message =
                    Some("Deadlock detected: no enabled actions".to_string());
                self.result.suggestion = Some(ErrorSuggestion {
                    action: "No action is enabled in the final state".to_string(),
                    example: Some("Use --no-deadlock if this is expected, or add TERMINAL in config for intentional final states".to_string()),
                    options: vec![
                        "Add --no-deadlock flag to disable deadlock checking".to_string(),
                        "Add TERMINAL state = \"value\" to config for expected final states".to_string(),
                        "Fix the Next relation to enable an action".to_string(),
                    ],
                });
                self.counterexample = Some(trace_to_counterexample(trace, None));
                self.statistics = stats_to_json(stats, elapsed);
            }
            CheckResult::Error { error, stats } => {
                self.result.status = "error".to_string();
                self.result.error_type = Some("runtime_error".to_string());
                self.result.error_code = Some(error_code_from_check_error(error));
                self.result.error_message = Some(error.to_string());
                self.result.suggestion = suggestion_from_check_error(error);
                self.statistics = stats_to_json(stats, elapsed);
            }
            CheckResult::LimitReached { limit_type, stats } => {
                self.result.status = "limit_reached".to_string();
                self.result.error_type = Some(
                    match limit_type {
                        crate::LimitType::States => "state_limit",
                        crate::LimitType::Depth => "depth_limit",
                    }
                    .to_string(),
                );
                self.result.error_code = Some(error_codes::TLC_LIMIT_REACHED.to_string());
                self.result.error_message = Some(format!("{:?} limit reached", limit_type));
                self.result.suggestion = Some(ErrorSuggestion {
                    action: "The model checking stopped due to a configured limit".to_string(),
                    example: Some(
                        "Use --max-states 0 or --max-depth 0 to disable limits".to_string(),
                    ),
                    options: vec![
                        "Increase --max-states to explore more states".to_string(),
                        "Increase --max-depth to explore deeper".to_string(),
                        "Add CONSTRAINT to reduce state space".to_string(),
                    ],
                });
                self.statistics = stats_to_json(stats, elapsed);
            }
        }
        self
    }

    /// Add an info diagnostic
    pub fn add_info(&mut self, code: &str, message: &str) {
        self.diagnostics.info.push(DiagnosticMessage {
            code: code.to_string(),
            message: message.to_string(),
            location: None,
            suggestion: None,
        });
    }

    /// Serialize to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Serialize to compact JSON string
    pub fn to_json_compact(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// Convert TLA+ Value to JsonValue
pub fn value_to_json(value: &Value) -> JsonValue {
    match value {
        Value::Bool(b) => JsonValue::Bool(*b),
        Value::SmallInt(n) => JsonValue::Int(*n),
        Value::Int(n) => JsonValue::Int(n.to_i64().unwrap_or(i64::MAX)),
        Value::String(s) => JsonValue::String(s.to_string()),
        Value::Set(elements) => {
            let items: Vec<JsonValue> = elements.iter().map(value_to_json).collect();
            JsonValue::Set(items)
        }
        Value::Tuple(elements) => {
            let items: Vec<JsonValue> = elements.iter().map(value_to_json).collect();
            JsonValue::Tuple(items)
        }
        Value::Seq(elements) => {
            let items: Vec<JsonValue> = elements.iter().map(value_to_json).collect();
            JsonValue::Seq(items)
        }
        Value::Record(fields) => {
            let map: HashMap<String, JsonValue> = fields
                .iter()
                .map(|(k, v)| (k.to_string(), value_to_json(v)))
                .collect();
            JsonValue::Record(map)
        }
        Value::Func(func) => {
            let domain: Vec<JsonValue> = func.domain_iter().map(value_to_json).collect();
            let mapping: Vec<(JsonValue, JsonValue)> = func
                .mapping_iter()
                .map(|(k, v)| (value_to_json(k), value_to_json(v)))
                .collect();
            JsonValue::Function { domain, mapping }
        }
        Value::IntFunc(func) => {
            let domain: Vec<JsonValue> = (func.min..=func.max).map(JsonValue::Int).collect();
            let mapping: Vec<(JsonValue, JsonValue)> = (func.min..=func.max)
                .zip(func.values.iter())
                .map(|(k, v)| (JsonValue::Int(k), value_to_json(v)))
                .collect();
            JsonValue::Function { domain, mapping }
        }
        Value::ModelValue(id) => JsonValue::ModelValue(id.to_string()),
        // For lazy values, try to materialize or represent as undefined
        Value::Interval(iv) => {
            // Use iter() which returns BigInt, convert to i64
            let items: Vec<JsonValue> = iv
                .iter()
                .map(|n| JsonValue::Int(n.to_i64().unwrap_or(i64::MAX)))
                .collect();
            JsonValue::Set(items)
        }
        // For complex lazy sets that might be expensive/infinite, represent as undefined
        Value::FuncSet(_)
        | Value::Subset(_)
        | Value::RecordSet(_)
        | Value::TupleSet(_)
        | Value::SetCup(_)
        | Value::SetCap(_)
        | Value::SetDiff(_)
        | Value::SetPred(_)
        | Value::KSubset(_)
        | Value::BigUnion(_)
        | Value::LazyFunc(_)
        | Value::Closure(_)
        | Value::StringSet
        | Value::AnySet
        | Value::SeqSet(_) => JsonValue::Undefined,
    }
}

/// Convert CheckStats to StatisticsInfo
fn stats_to_json(stats: &CheckStats, elapsed: Duration) -> StatisticsInfo {
    let time_secs = elapsed.as_secs_f64();
    StatisticsInfo {
        states_found: stats.states_found,
        states_initial: stats.initial_states,
        states_distinct: Some(stats.states_found),
        transitions: stats.transitions,
        max_depth: Some(stats.max_depth),
        max_queue_depth: Some(stats.max_queue_depth),
        time_seconds: time_secs,
        states_per_second: if time_secs > 0.0 {
            Some(stats.states_found as f64 / time_secs)
        } else {
            None
        },
        memory_mb: None,
    }
}

/// Convert Trace to CounterexampleInfo with state diffs
fn trace_to_counterexample(trace: &Trace, loop_start: Option<usize>) -> CounterexampleInfo {
    use crate::State;

    let mut states = Vec::new();
    let mut prev_state: Option<&State> = None;

    for (i, state) in trace.states.iter().enumerate() {
        let variables: HashMap<String, JsonValue> = state
            .vars()
            .map(|(k, v)| (k.to_string(), value_to_json(v)))
            .collect();

        // Compute diff from previous state
        let diff = if let Some(prev) = prev_state {
            let mut changed = HashMap::new();
            let mut unchanged = Vec::new();

            for (var, new_val) in state.vars() {
                if let Some(old_val) = prev.get(var) {
                    if old_val != new_val {
                        changed.insert(
                            var.to_string(),
                            ValueChange {
                                from: value_to_json(old_val),
                                to: value_to_json(new_val),
                            },
                        );
                    } else {
                        unchanged.push(var.to_string());
                    }
                } else {
                    // New variable (shouldn't happen normally, but handle it)
                    changed.insert(
                        var.to_string(),
                        ValueChange {
                            from: JsonValue::Undefined,
                            to: value_to_json(new_val),
                        },
                    );
                }
            }

            Some(StateDiff { changed, unchanged })
        } else {
            None
        };

        let action_type = if i == 0 { "initial" } else { "next" };
        let action_name = if i == 0 {
            "Init".to_string()
        } else {
            "Next".to_string()
        };

        states.push(StateInfo {
            index: i + 1, // 1-based indexing as per design
            fingerprint: Some(format!("{:016x}", state.fingerprint().0)),
            action: ActionRef {
                name: action_name,
                action_type: action_type.to_string(),
                location: None,
            },
            variables,
            diff_from_previous: diff,
        });

        prev_state = Some(state);
    }

    CounterexampleInfo {
        length: states.len(),
        states,
        loop_start: loop_start.map(|s| s + 1), // Convert to 1-based
    }
}

/// Map CheckError to a structured error code
fn error_code_from_check_error(error: &crate::CheckError) -> String {
    use crate::CheckError;
    match error {
        CheckError::MissingInit => error_codes::CFG_MISSING_INIT.to_string(),
        CheckError::MissingNext => error_codes::CFG_MISSING_NEXT.to_string(),
        CheckError::MissingInvariant(_) => error_codes::TLC_UNDEFINED_OP.to_string(),
        CheckError::MissingProperty(_) => error_codes::TLC_UNDEFINED_OP.to_string(),
        CheckError::EvalError(_) => error_codes::TLC_EVAL_ERROR.to_string(),
        CheckError::InitNotBoolean => error_codes::TLC_TYPE_MISMATCH.to_string(),
        CheckError::NextNotBoolean => error_codes::TLC_TYPE_MISMATCH.to_string(),
        CheckError::InvariantNotBoolean(_) => error_codes::TLC_TYPE_MISMATCH.to_string(),
        CheckError::PropertyNotBoolean(_) => error_codes::TLC_TYPE_MISMATCH.to_string(),
        CheckError::NoVariables => error_codes::TLC_EVAL_ERROR.to_string(),
        CheckError::InitCannotEnumerate(_) => error_codes::TLC_EVAL_ERROR.to_string(),
        CheckError::SpecificationError(_) => error_codes::CFG_PARSE_ERROR.to_string(),
        CheckError::LivenessError(_) => error_codes::TLC_LIVENESS_VIOLATED.to_string(),
        CheckError::FingerprintStorageOverflow { .. } => error_codes::TLC_LIMIT_REACHED.to_string(),
        CheckError::AssumeFalse { .. } => error_codes::TLC_EVAL_ERROR.to_string(),
    }
}

/// Generate actionable suggestion from CheckError
fn suggestion_from_check_error(error: &crate::CheckError) -> Option<ErrorSuggestion> {
    use crate::CheckError;
    match error {
        CheckError::MissingInit => Some(ErrorSuggestion {
            action: "Define an Init predicate in the spec and reference it in the config"
                .to_string(),
            example: Some("INIT Init\n\nIn spec: Init == x = 0".to_string()),
            options: vec![],
        }),
        CheckError::MissingNext => Some(ErrorSuggestion {
            action: "Define a Next action in the spec and reference it in the config".to_string(),
            example: Some("NEXT Next\n\nIn spec: Next == x' = x + 1".to_string()),
            options: vec![],
        }),
        CheckError::MissingInvariant(name) => Some(ErrorSuggestion {
            action: format!("Define the invariant '{}' in the spec", name),
            example: Some(format!("{} == x >= 0", name)),
            options: vec![],
        }),
        CheckError::MissingProperty(name) => Some(ErrorSuggestion {
            action: format!("Define the property '{}' in the spec", name),
            example: Some(format!("{} == <>[] done", name)),
            options: vec![],
        }),
        CheckError::EvalError(e) => {
            let (action, example) = match e {
                crate::error::EvalError::UndefinedVar { name, .. } => (
                    format!("Variable '{}' is not defined", name),
                    Some(format!("Ensure '{}' is declared: VARIABLE {}", name, name)),
                ),
                crate::error::EvalError::UndefinedOp { name, .. } => (
                    format!("Operator '{}' is not defined", name),
                    Some(format!(
                        "Define '{}' in spec or EXTENDS a module that defines it",
                        name
                    )),
                ),
                crate::error::EvalError::TypeError { expected, got, .. } => (
                    format!("Type error: expected {}, got {}", expected, got),
                    None,
                ),
                crate::error::EvalError::DivisionByZero { .. } => (
                    "Division by zero occurred".to_string(),
                    Some("Check that the divisor is never zero".to_string()),
                ),
                crate::error::EvalError::ChooseFailed { .. } => (
                    "CHOOSE found no satisfying value".to_string(),
                    Some(
                        "Ensure the set is non-empty and the predicate can be satisfied"
                            .to_string(),
                    ),
                ),
                _ => (
                    "Check the error details for more information".to_string(),
                    None,
                ),
            };
            Some(ErrorSuggestion {
                action,
                example,
                options: vec![],
            })
        }
        CheckError::InitNotBoolean => Some(ErrorSuggestion {
            action: "Init predicate must return a boolean value".to_string(),
            example: Some("Init == x = 0 /\\ y = 0".to_string()),
            options: vec![],
        }),
        CheckError::NextNotBoolean => Some(ErrorSuggestion {
            action: "Next relation must return a boolean value".to_string(),
            example: Some("Next == x' = x + 1".to_string()),
            options: vec![],
        }),
        CheckError::InvariantNotBoolean(name) => Some(ErrorSuggestion {
            action: format!("Invariant '{}' must return a boolean value", name),
            example: Some(format!("{} == x >= 0", name)),
            options: vec![],
        }),
        CheckError::PropertyNotBoolean(name) => Some(ErrorSuggestion {
            action: format!("Property '{}' must return a boolean value", name),
            example: Some(format!("{} == <>[] done", name)),
            options: vec![],
        }),
        CheckError::NoVariables => Some(ErrorSuggestion {
            action: "Spec must declare at least one variable".to_string(),
            example: Some("VARIABLE x, y".to_string()),
            options: vec![],
        }),
        CheckError::InitCannotEnumerate(reason) => Some(ErrorSuggestion {
            action: "Init predicate cannot be enumerated for initial states".to_string(),
            example: Some(format!(
                "Reason: {}\n\nUse explicit enumeration: x \\in {{0, 1, 2}}",
                reason
            )),
            options: vec![
                "Use set membership: x \\in {1, 2, 3}".to_string(),
                "Use range: x \\in 1..10".to_string(),
            ],
        }),
        CheckError::SpecificationError(msg) => Some(ErrorSuggestion {
            action: "Fix the SPECIFICATION formula".to_string(),
            example: Some(msg.clone()),
            options: vec![],
        }),
        CheckError::LivenessError(msg) => Some(ErrorSuggestion {
            action: "Liveness checking encountered an error".to_string(),
            example: Some(msg.clone()),
            options: vec![],
        }),
        CheckError::FingerprintStorageOverflow { dropped } => Some(ErrorSuggestion {
            action: format!(
                "Fingerprint storage overflowed, {} states were dropped",
                dropped
            ),
            example: Some(
                "Increase --mmap-fingerprints capacity or reduce state space".to_string(),
            ),
            options: vec![
                "Use larger --mmap-fingerprints value".to_string(),
                "Add CONSTRAINT to reduce state space".to_string(),
                "Use symmetry reduction".to_string(),
            ],
        }),
        CheckError::AssumeFalse { location } => Some(ErrorSuggestion {
            action: format!("ASSUME statement {} evaluated to FALSE", location),
            example: Some("Check that all ASSUME prerequisites are satisfied".to_string()),
            options: vec![
                "Verify ASSUME conditions match your model configuration".to_string(),
                "Remove or modify the failing ASSUME statement".to_string(),
            ],
        }),
    }
}

// ============================================================================
// GraphViz DOT format export
// ============================================================================

/// Convert a trace to GraphViz DOT format for visualization.
///
/// The DOT output creates a directed graph where:
/// - Each state is a node with variable values shown
/// - Each transition is an edge labeled with "Next" or "Init"
/// - For liveness violations, the cycle portion is highlighted
///
/// # Arguments
///
/// * `trace` - The counterexample trace to convert
/// * `loop_start` - Optional index where a liveness cycle begins
///
/// # Example output
///
/// ```dot
/// digraph trace {
///   rankdir=TB;
///   node [shape=record, fontname="Courier"];
///
///   s0 [label="State 1|x = 0\\ly = 0"];
///   s1 [label="State 2|x = 1\\ly = 0"];
///
///   s0 -> s1 [label="Next"];
/// }
/// ```
pub fn trace_to_dot(trace: &Trace, loop_start: Option<usize>) -> String {
    let mut dot = String::new();
    dot.push_str("digraph trace {\n");
    dot.push_str("  rankdir=TB;\n");
    dot.push_str("  node [shape=record, fontname=\"Courier\"];\n");
    dot.push_str("  edge [fontname=\"Helvetica\", fontsize=10];\n");
    dot.push('\n');

    // Generate nodes for each state
    for (i, state) in trace.states.iter().enumerate() {
        let state_num = i + 1;
        let fp = format!("{:08x}", state.fingerprint().0 & 0xFFFFFFFF);

        // Build variable list with proper escaping for DOT records
        let vars: Vec<String> = state
            .vars()
            .map(|(name, value)| {
                let value_str = format!("{}", value);
                // Escape special characters for DOT record labels
                let escaped = value_str
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('{', "\\{")
                    .replace('}', "\\}")
                    .replace('|', "\\|")
                    .replace('<', "\\<")
                    .replace('>', "\\>");
                format!("{} = {}", name, escaped)
            })
            .collect();
        let vars_label = vars.join("\\l") + "\\l"; // Left-align each line

        // Determine node style based on position
        let style = if i == 0 {
            // Initial state: green border
            ", style=bold, color=green"
        } else if i == trace.states.len() - 1 && loop_start.is_none() {
            // Final state (error state): red border
            ", style=bold, color=red"
        } else if loop_start.is_some() && i >= loop_start.unwrap() {
            // Part of liveness cycle: blue border
            ", style=bold, color=blue"
        } else {
            ""
        };

        dot.push_str(&format!(
            "  s{} [label=\"State {} ({})\\n|{}\"{}];\n",
            i, state_num, fp, vars_label, style
        ));
    }

    dot.push('\n');

    // Generate edges between consecutive states
    for i in 0..trace.states.len().saturating_sub(1) {
        let edge_label = if i == 0 { "Init" } else { "Next" };

        // Determine edge style
        let style = if let Some(ls) = loop_start {
            if i >= ls {
                // Cycle edge: blue, bold
                ", style=bold, color=blue"
            } else {
                ""
            }
        } else {
            ""
        };

        dot.push_str(&format!(
            "  s{} -> s{} [label=\"{}\"{}];\n",
            i,
            i + 1,
            edge_label,
            style
        ));
    }

    // For liveness cycles, add back-edge to show the loop
    if let Some(ls) = loop_start {
        if trace.states.len() > ls {
            let last_idx = trace.states.len() - 1;
            dot.push_str(&format!(
                "  s{} -> s{} [label=\"cycle\", style=dashed, color=blue, constraint=false];\n",
                last_idx, ls
            ));
        }
    }

    dot.push_str("}\n");
    dot
}

/// Convert a combined liveness trace (prefix + cycle) to DOT format.
///
/// This is a convenience function for liveness violations where
/// the prefix and cycle are provided separately.
pub fn liveness_trace_to_dot(prefix: &Trace, cycle: &Trace) -> String {
    // Combine prefix and cycle into single trace
    let mut combined = prefix.clone();
    for state in &cycle.states {
        combined.states.push(state.clone());
    }
    trace_to_dot(&combined, Some(prefix.states.len()))
}

// ============================================================================
// JSON Lines (JSONL) streaming output
// ============================================================================

/// JSONL event types for streaming output
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum JsonlEvent {
    /// Model checking started
    #[serde(rename = "start")]
    Start { spec: String, timestamp: String },
    /// Progress update
    #[serde(rename = "progress")]
    Progress {
        states: usize,
        depth: usize,
        time: f64,
    },
    /// Error detected
    #[serde(rename = "error")]
    Error {
        error_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        state_index: Option<usize>,
    },
    /// State in counterexample
    #[serde(rename = "state")]
    State {
        index: usize,
        action: String,
        variables: HashMap<String, JsonValue>,
        #[serde(skip_serializing_if = "Option::is_none")]
        diff: Option<HashMap<String, (JsonValue, JsonValue)>>,
    },
    /// Model checking complete
    #[serde(rename = "done")]
    Done { status: String, time: f64 },
}

impl JsonlEvent {
    /// Serialize to single JSON line
    pub fn to_jsonl(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_value_serialization() {
        let val = JsonValue::Int(42);
        let json = serde_json::to_string(&val).unwrap();
        assert!(json.contains("\"type\":\"int\""));
        assert!(json.contains("\"value\":42"));
    }

    #[test]
    fn test_json_output_basic() {
        let output = JsonOutput::new(
            Path::new("/tmp/test.tla"),
            Some(Path::new("/tmp/test.cfg")),
            "Test",
            1,
        );
        let json = output.to_json().unwrap();
        // Check basic structure
        assert!(json.contains("\"version\": \"1.0\""), "JSON: {}", json);
        assert!(json.contains("\"tool\": \"tla2\""), "JSON: {}", json);
    }

    #[test]
    fn test_value_to_json() {
        use std::sync::Arc;
        assert!(matches!(
            value_to_json(&Value::Bool(true)),
            JsonValue::Bool(true)
        ));
        assert!(matches!(
            value_to_json(&Value::String(Arc::from("hello"))),
            JsonValue::String(s) if s == "hello"
        ));
    }

    #[test]
    fn test_error_codes() {
        // Verify error codes are properly defined
        assert_eq!(error_codes::TLC_DEADLOCK, "TLC_DEADLOCK");
        assert_eq!(
            error_codes::TLC_INVARIANT_VIOLATED,
            "TLC_INVARIANT_VIOLATED"
        );
        assert_eq!(error_codes::TLC_LIVENESS_VIOLATED, "TLC_LIVENESS_VIOLATED");
        assert_eq!(error_codes::CFG_MISSING_INIT, "CFG_MISSING_INIT");
        assert_eq!(error_codes::TLC_EVAL_ERROR, "TLC_EVAL_ERROR");
    }

    #[test]
    fn test_error_suggestion_serialization() {
        let suggestion = ErrorSuggestion {
            action: "Test action".to_string(),
            example: Some("Example code".to_string()),
            options: vec!["Option 1".to_string(), "Option 2".to_string()],
        };
        let json = serde_json::to_string(&suggestion).unwrap();
        assert!(
            json.contains("\"action\":\"Test action\""),
            "JSON: {}",
            json
        );
        assert!(
            json.contains("\"example\":\"Example code\""),
            "JSON: {}",
            json
        );
        assert!(json.contains("\"options\":[\"Option 1\""), "JSON: {}", json);
    }

    #[test]
    fn test_result_info_with_error_code() {
        let result = ResultInfo {
            status: "error".to_string(),
            error_type: Some("deadlock".to_string()),
            error_code: Some(error_codes::TLC_DEADLOCK.to_string()),
            error_message: Some("Deadlock detected".to_string()),
            violated_property: None,
            suggestion: Some(ErrorSuggestion {
                action: "No action enabled".to_string(),
                example: Some("Add --no-deadlock".to_string()),
                options: vec![],
            }),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(
            json.contains("\"error_code\":\"TLC_DEADLOCK\""),
            "JSON: {}",
            json
        );
        assert!(json.contains("\"suggestion\":{"), "JSON: {}", json);
    }

    #[test]
    fn test_trace_to_dot_basic() {
        use crate::{State, Trace};

        // Create a simple 2-state trace using from_pairs
        let state1 = State::from_pairs([("x", Value::SmallInt(0)), ("y", Value::SmallInt(0))]);

        let state2 = State::from_pairs([("x", Value::SmallInt(1)), ("y", Value::SmallInt(0))]);

        let trace = Trace::from_states(vec![state1, state2]);
        let dot = trace_to_dot(&trace, None);

        // Verify basic DOT structure
        assert!(dot.starts_with("digraph trace {"), "DOT: {}", dot);
        assert!(dot.contains("rankdir=TB;"), "DOT: {}", dot);
        assert!(dot.contains("State 1"), "DOT: {}", dot);
        assert!(dot.contains("State 2"), "DOT: {}", dot);
        assert!(dot.contains("x = 0"), "DOT: {}", dot);
        assert!(dot.contains("x = 1"), "DOT: {}", dot);
        assert!(dot.contains("s0 -> s1"), "DOT: {}", dot);
        assert!(dot.ends_with("}\n"), "DOT: {}", dot);
    }

    #[test]
    fn test_trace_to_dot_with_special_chars() {
        use crate::{State, Trace};
        use std::sync::Arc;

        // Create state with values containing special DOT characters
        let state = State::from_pairs([("name", Value::String(Arc::from("test|value")))]);

        let trace = Trace::from_states(vec![state]);
        let dot = trace_to_dot(&trace, None);

        // Verify special characters are escaped
        assert!(
            dot.contains("test\\|value"),
            "Pipe should be escaped. DOT: {}",
            dot
        );
    }

    #[test]
    fn test_trace_to_dot_with_sets() {
        use crate::value::SortedSet;
        use crate::{State, Trace};

        // Create state with set value
        let state = State::from_pairs([(
            "s",
            Value::Set(SortedSet::from_iter([
                Value::SmallInt(1),
                Value::SmallInt(2),
                Value::SmallInt(3),
            ])),
        )]);

        let trace = Trace::from_states(vec![state]);
        let dot = trace_to_dot(&trace, None);

        // Set should be rendered with escaped braces
        assert!(
            dot.contains("s = \\{"),
            "Set braces should be escaped. DOT: {}",
            dot
        );
    }

    #[test]
    fn test_trace_to_dot_liveness_cycle() {
        use crate::{State, Trace};

        // Create a 4-state trace where states 2-3 form a cycle
        let states: Vec<State> = (0..4)
            .map(|i| State::from_pairs([("x", Value::SmallInt(i))]))
            .collect();

        let trace = Trace::from_states(states);
        let dot = trace_to_dot(&trace, Some(2)); // Cycle starts at index 2

        // Verify cycle styling
        assert!(
            dot.contains("color=blue"),
            "Cycle states should be blue. DOT: {}",
            dot
        );
        assert!(
            dot.contains("cycle"),
            "Should have cycle back-edge label. DOT: {}",
            dot
        );
        assert!(
            dot.contains("constraint=false"),
            "Cycle edge should have constraint=false. DOT: {}",
            dot
        );
    }

    #[test]
    fn test_trace_to_dot_empty_trace() {
        use crate::Trace;

        let trace = Trace::new();
        let dot = trace_to_dot(&trace, None);

        // Should still produce valid DOT, just with no nodes
        assert!(dot.starts_with("digraph trace {"), "DOT: {}", dot);
        assert!(dot.ends_with("}\n"), "DOT: {}", dot);
    }

    #[test]
    fn test_liveness_trace_to_dot() {
        use crate::{State, Trace};

        // Create prefix (2 states)
        let prefix_states: Vec<State> = (0..2)
            .map(|i| State::from_pairs([("x", Value::SmallInt(i))]))
            .collect();
        let prefix = Trace::from_states(prefix_states);

        // Create cycle (2 states)
        let cycle_states: Vec<State> = (2..4)
            .map(|i| State::from_pairs([("x", Value::SmallInt(i))]))
            .collect();
        let cycle = Trace::from_states(cycle_states);

        let dot = liveness_trace_to_dot(&prefix, &cycle);

        // Should have 4 states total
        assert!(dot.contains("State 1"), "DOT: {}", dot);
        assert!(dot.contains("State 4"), "DOT: {}", dot);

        // Should have cycle back-edge
        assert!(
            dot.contains("cycle"),
            "Should have cycle label. DOT: {}",
            dot
        );
    }
}
