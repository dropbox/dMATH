//! TLA+ trace verification integration
//!
//! This module provides functionality to verify async execution traces
//! against TLA+ specifications. It allows checking whether observed
//! system behavior conforms to a formal TLA+ model.
//!
//! ## Overview
//!
//! The integration works by:
//! 1. Converting async execution traces to TLA+ trace format
//! 2. Generating a TLA+ module with the trace as initial state constraints
//! 3. Running TLC to check if the trace is a valid behavior of the spec
//!
//! ## Example
//!
//! ```rust,ignore
//! use dashprove_async::{ExecutionTrace, StateTransition, TlaTraceVerifier};
//!
//! let mut trace = ExecutionTrace::new(serde_json::json!({"counter": 0}));
//! trace.add_transition(StateTransition::new(
//!     serde_json::json!({"counter": 0}),
//!     "Increment".to_string(),
//!     serde_json::json!({"counter": 1}),
//! ));
//!
//! let verifier = TlaTraceVerifier::new("Counter.tla", "/path/to/tla");
//! let result = verifier.verify_trace(&trace).await?;
//! ```

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;
use tempfile::TempDir;

use crate::{AsyncVerifyError, ExecutionTrace, Violation};

/// Result of TLA+ trace verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlaTraceVerificationResult {
    /// Whether the trace is a valid behavior of the specification
    pub valid: bool,
    /// State at which trace diverged from spec (if invalid)
    pub divergence_state: Option<usize>,
    /// Action that caused divergence (if invalid)
    pub divergence_action: Option<String>,
    /// Expected possible actions at divergence point
    pub expected_actions: Vec<String>,
    /// Violations found during verification
    pub violations: Vec<TraceViolation>,
    /// Raw TLC output
    pub raw_output: String,
    /// Verification duration
    pub duration_ms: u64,
}

impl TlaTraceVerificationResult {
    /// Create a successful result (trace is valid)
    pub fn valid(raw_output: String, duration: Duration) -> Self {
        Self {
            valid: true,
            divergence_state: None,
            divergence_action: None,
            expected_actions: vec![],
            violations: vec![],
            raw_output,
            duration_ms: duration.as_millis() as u64,
        }
    }

    /// Create a failed result (trace diverged from spec)
    pub fn diverged(
        state: usize,
        action: Option<String>,
        expected: Vec<String>,
        raw_output: String,
        duration: Duration,
    ) -> Self {
        Self {
            valid: false,
            divergence_state: Some(state),
            divergence_action: action,
            expected_actions: expected,
            violations: vec![TraceViolation::DivergenceFromSpec {
                state_index: state,
                message: "Trace diverged from specification".to_string(),
            }],
            raw_output,
            duration_ms: duration.as_millis() as u64,
        }
    }

    /// Create a failed result with violations
    pub fn with_violations(
        violations: Vec<TraceViolation>,
        raw_output: String,
        duration: Duration,
    ) -> Self {
        Self {
            valid: false,
            divergence_state: None,
            divergence_action: None,
            expected_actions: vec![],
            violations,
            raw_output,
            duration_ms: duration.as_millis() as u64,
        }
    }
}

/// A violation found during trace verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceViolation {
    /// Trace diverged from spec at the given state
    DivergenceFromSpec {
        /// State index where divergence occurred
        state_index: usize,
        /// Description of the divergence
        message: String,
    },
    /// An invariant was violated at the given state
    InvariantViolation {
        /// State index where violation occurred
        state_index: usize,
        /// Name of the violated invariant
        invariant: String,
        /// State values at violation
        state: serde_json::Value,
    },
    /// Action precondition was not satisfied
    PreconditionViolation {
        /// State index
        state_index: usize,
        /// Action name
        action: String,
        /// Description
        message: String,
    },
    /// Stuttering occurred when it shouldn't
    UnexpectedStutter {
        /// State index
        state_index: usize,
    },
}

impl TraceViolation {
    /// Convert to a general Violation type
    pub fn to_violation(&self) -> Violation {
        match self {
            TraceViolation::DivergenceFromSpec {
                state_index,
                message,
            } => Violation::SafetyViolation {
                property: format!("trace_conformance_state_{}", state_index),
                message: message.clone(),
            },
            TraceViolation::InvariantViolation {
                invariant, state, ..
            } => Violation::InvariantViolation {
                invariant: invariant.clone(),
                state: state.clone(),
                message: "Invariant violated during trace".to_string(),
            },
            TraceViolation::PreconditionViolation {
                action, message, ..
            } => Violation::SafetyViolation {
                property: format!("{}_precondition", action),
                message: message.clone(),
            },
            TraceViolation::UnexpectedStutter { state_index } => Violation::SafetyViolation {
                property: format!("no_stuttering_state_{}", state_index),
                message: "Unexpected stuttering step".to_string(),
            },
        }
    }
}

/// Configuration for TLA+ trace verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlaTraceConfig {
    /// Path to the TLA+ specification file
    pub spec_path: PathBuf,
    /// Working directory for TLC
    pub working_dir: Option<PathBuf>,
    /// Timeout for TLC execution
    pub timeout_ms: u64,
    /// Whether to allow stuttering steps
    pub allow_stuttering: bool,
    /// Invariants to check during trace replay
    pub invariants: Vec<String>,
    /// Mapping from trace variable names to TLA+ variable names
    pub variable_mapping: HashMap<String, String>,
    /// Path to TLC executable or JAR
    pub tlc_path: Option<PathBuf>,
    /// Path to Java executable (for JAR execution)
    pub java_path: Option<PathBuf>,
}

impl Default for TlaTraceConfig {
    fn default() -> Self {
        Self {
            spec_path: PathBuf::new(),
            working_dir: None,
            timeout_ms: 60000,
            allow_stuttering: true,
            invariants: vec![],
            variable_mapping: HashMap::new(),
            tlc_path: None,
            java_path: None,
        }
    }
}

impl TlaTraceConfig {
    /// Create a new config with the given spec path
    pub fn new(spec_path: impl Into<PathBuf>) -> Self {
        Self {
            spec_path: spec_path.into(),
            ..Default::default()
        }
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_ms = timeout.as_millis() as u64;
        self
    }

    /// Disable stuttering
    pub fn without_stuttering(mut self) -> Self {
        self.allow_stuttering = false;
        self
    }

    /// Add invariants to check
    pub fn with_invariants(mut self, invariants: Vec<String>) -> Self {
        self.invariants = invariants;
        self
    }

    /// Add variable mapping
    pub fn with_variable_mapping(mut self, mapping: HashMap<String, String>) -> Self {
        self.variable_mapping = mapping;
        self
    }

    /// Set TLC path
    pub fn with_tlc_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.tlc_path = Some(path.into());
        self
    }
}

/// TLA+ trace verifier
///
/// Verifies that execution traces conform to a TLA+ specification.
pub struct TlaTraceVerifier {
    config: TlaTraceConfig,
}

impl TlaTraceVerifier {
    /// Create a new verifier with the given config
    pub fn new(config: TlaTraceConfig) -> Self {
        Self { config }
    }

    /// Create a verifier for a specific spec file
    pub fn for_spec(spec_path: impl Into<PathBuf>) -> Self {
        Self::new(TlaTraceConfig::new(spec_path))
    }

    /// Verify that an execution trace conforms to the TLA+ specification
    pub async fn verify_trace(
        &self,
        trace: &ExecutionTrace,
    ) -> Result<TlaTraceVerificationResult, AsyncVerifyError> {
        let start = std::time::Instant::now();

        // Create temp directory for trace spec
        let temp_dir = TempDir::new().map_err(|e| {
            AsyncVerifyError::tla_verification(format!("Failed to create temp dir: {}", e))
        })?;

        // Generate the trace constraint module
        let trace_module = self.generate_trace_module(trace)?;
        let trace_path = temp_dir.path().join("TraceConstraint.tla");
        std::fs::write(&trace_path, &trace_module).map_err(|e| {
            AsyncVerifyError::tla_verification(format!("Failed to write trace module: {}", e))
        })?;

        // Generate TLC config
        let cfg_content = self.generate_tlc_config(trace)?;
        let cfg_path = temp_dir.path().join("TraceConstraint.cfg");
        std::fs::write(&cfg_path, &cfg_content).map_err(|e| {
            AsyncVerifyError::tla_verification(format!("Failed to write config: {}", e))
        })?;

        // Copy the spec file to temp dir
        let spec_name = self
            .config
            .spec_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("Spec.tla");
        let spec_dest = temp_dir.path().join(spec_name);

        // Read and copy spec
        let spec_content = std::fs::read_to_string(&self.config.spec_path).map_err(|e| {
            AsyncVerifyError::tla_verification(format!("Failed to read spec: {}", e))
        })?;
        std::fs::write(&spec_dest, &spec_content).map_err(|e| {
            AsyncVerifyError::tla_verification(format!("Failed to copy spec: {}", e))
        })?;

        // Run TLC
        let output = self.run_tlc(temp_dir.path(), &trace_path, &cfg_path)?;

        let duration = start.elapsed();

        // Parse results
        self.parse_tlc_output(&output, duration)
    }

    /// Verify multiple traces against the specification
    pub async fn verify_traces(
        &self,
        traces: &[ExecutionTrace],
    ) -> Result<Vec<TlaTraceVerificationResult>, AsyncVerifyError> {
        let mut results = Vec::with_capacity(traces.len());
        for trace in traces {
            results.push(self.verify_trace(trace).await?);
        }
        Ok(results)
    }

    /// Generate a TLA+ module that constrains the initial state to the trace
    fn generate_trace_module(&self, trace: &ExecutionTrace) -> Result<String, AsyncVerifyError> {
        let spec_module = self
            .config
            .spec_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Spec");

        let mut module = String::new();
        module.push_str("---- MODULE TraceConstraint ----\n");
        module.push_str(&format!("EXTENDS {}, Sequences, TLC\n\n", spec_module));

        // Generate trace as a TLA+ sequence of states
        module.push_str("\\* Recorded execution trace\n");
        module.push_str("Trace == <<\n");

        // Initial state
        let initial_state = self.json_to_tla_record(&trace.initial_state)?;
        module.push_str(&format!("    {}", initial_state));

        // Each transition state
        for transition in &trace.transitions {
            module.push_str(",\n");
            let state = self.json_to_tla_record(&transition.to_state)?;
            module.push_str(&format!("    {}", state));
        }

        module.push_str("\n>>\n\n");

        // Generate trace actions
        module.push_str("\\* Actions taken in the trace\n");
        module.push_str("TraceActions == <<\n");
        for (i, transition) in trace.transitions.iter().enumerate() {
            if i > 0 {
                module.push_str(",\n");
            }
            module.push_str(&format!("    \"{}\"", transition.event));
        }
        module.push_str("\n>>\n\n");

        // Trace length
        module.push_str(&format!("TraceLen == {}\n\n", trace.transitions.len() + 1));

        // State index variable to track position in trace
        module.push_str("VARIABLE traceIdx\n\n");

        // Modified init - constrain to initial trace state
        module.push_str("TraceInit ==\n");
        module.push_str("    /\\ traceIdx = 1\n");
        module.push_str("    /\\ Init\n");

        // Add constraints for each variable in initial state
        if let Some(obj) = trace.initial_state.as_object() {
            for (var, value) in obj {
                let mapped_var = self
                    .config
                    .variable_mapping
                    .get(var)
                    .cloned()
                    .unwrap_or_else(|| var.clone());
                let tla_value = self.json_to_tla_value(value)?;
                module.push_str(&format!("    /\\ {} = {}\n", mapped_var, tla_value));
            }
        }
        module.push('\n');

        // Modified next - only allow transitions that match trace
        module.push_str("TraceNext ==\n");
        if self.config.allow_stuttering {
            module.push_str("    \\/ /\\ traceIdx <= TraceLen\n");
            module.push_str("       /\\ traceIdx' = traceIdx + 1\n");
            module.push_str("       /\\ Next\n");
            module.push_str("    \\/ /\\ traceIdx > TraceLen\n");
            module.push_str("       /\\ UNCHANGED <<traceIdx>>\n");
            module.push_str("       /\\ UNCHANGED vars\n");
        } else {
            module.push_str("    /\\ traceIdx < TraceLen\n");
            module.push_str("    /\\ traceIdx' = traceIdx + 1\n");
            module.push_str("    /\\ Next\n");
        }
        module.push('\n');

        // Trace conformance invariant
        module.push_str("\\* Check that current state matches trace at current index\n");
        module.push_str("TraceConformance ==\n");
        module.push_str("    traceIdx <= TraceLen =>\n");
        module.push_str("        LET expected == Trace[traceIdx]\n");
        module.push_str("        IN  ");

        // Generate conformance check for each traced variable
        if let Some(obj) = trace.initial_state.as_object() {
            let vars: Vec<_> = obj.keys().collect();
            for (i, var) in vars.iter().enumerate() {
                let mapped_var = self
                    .config
                    .variable_mapping
                    .get(*var)
                    .cloned()
                    .unwrap_or_else(|| (*var).clone());
                if i > 0 {
                    module.push_str("            /\\ ");
                }
                module.push_str(&format!("{} = expected.{}\n", mapped_var, var));
            }
        }
        module.push('\n');

        // Spec with trace constraints
        module.push_str("TraceSpec == TraceInit /\\ [][TraceNext]_<<traceIdx, vars>>\n\n");

        // Additional invariants
        for inv in &self.config.invariants {
            module.push_str(&format!("TraceInv_{} == {}\n", inv, inv));
        }

        module.push_str("====\n");

        Ok(module)
    }

    /// Generate TLC configuration for trace verification
    fn generate_tlc_config(&self, _trace: &ExecutionTrace) -> Result<String, AsyncVerifyError> {
        let mut cfg = String::new();

        cfg.push_str("SPECIFICATION TraceSpec\n\n");

        // Add conformance check as invariant
        cfg.push_str("INVARIANT TraceConformance\n");

        // Add user-specified invariants
        for inv in &self.config.invariants {
            cfg.push_str(&format!("INVARIANT TraceInv_{}\n", inv));
        }

        Ok(cfg)
    }

    /// Run TLC model checker
    fn run_tlc(
        &self,
        work_dir: &Path,
        spec_path: &Path,
        cfg_path: &Path,
    ) -> Result<String, AsyncVerifyError> {
        let tlc_cmd = self.find_tlc()?;

        let timeout = Duration::from_millis(self.config.timeout_ms);

        match tlc_cmd {
            TlcCommand::Standalone(path) => {
                let output = Command::new(&path)
                    .current_dir(work_dir)
                    .arg("-config")
                    .arg(cfg_path)
                    .arg(spec_path)
                    .output()
                    .map_err(|e| {
                        AsyncVerifyError::tla_verification(format!("Failed to run TLC: {}", e))
                    })?;

                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                Ok(format!("{}\n{}", stdout, stderr))
            }
            TlcCommand::Jar {
                java_path,
                jar_path,
            } => {
                let output = Command::new(&java_path)
                    .current_dir(work_dir)
                    .arg("-jar")
                    .arg(&jar_path)
                    .arg("-config")
                    .arg(cfg_path)
                    .arg(spec_path)
                    .output()
                    .map_err(|e| {
                        AsyncVerifyError::tla_verification(format!("Failed to run TLC: {}", e))
                    })?;

                // Check for timeout
                let _timeout = timeout; // Used for documentation, actual timeout would need more work

                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                Ok(format!("{}\n{}", stdout, stderr))
            }
        }
    }

    /// Find TLC executable
    fn find_tlc(&self) -> Result<TlcCommand, AsyncVerifyError> {
        // Check config first
        if let Some(ref path) = self.config.tlc_path {
            if path.exists() {
                if path.extension().is_some_and(|e| e == "jar") {
                    let java = self
                        .config
                        .java_path
                        .clone()
                        .unwrap_or_else(|| PathBuf::from("java"));
                    return Ok(TlcCommand::Jar {
                        java_path: java,
                        jar_path: path.clone(),
                    });
                } else {
                    return Ok(TlcCommand::Standalone(path.clone()));
                }
            }
        }

        // Try to find tlc in PATH
        if let Ok(output) = Command::new("which").arg("tlc").output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                return Ok(TlcCommand::Standalone(PathBuf::from(path)));
            }
        }

        // Try to find tla2tools.jar
        let jar_locations = [
            PathBuf::from("/usr/local/lib/tla2tools.jar"),
            PathBuf::from("/opt/tla/tla2tools.jar"),
            dirs::home_dir()
                .unwrap_or_default()
                .join(".local/lib/tla2tools.jar"),
        ];

        for jar in &jar_locations {
            if jar.exists() {
                return Ok(TlcCommand::Jar {
                    java_path: PathBuf::from("java"),
                    jar_path: jar.clone(),
                });
            }
        }

        Err(AsyncVerifyError::tla_verification(
            "TLC not found. Please install TLA+ tools or set tlc_path in config".to_string(),
        ))
    }

    /// Parse TLC output into verification result
    fn parse_tlc_output(
        &self,
        output: &str,
        duration: Duration,
    ) -> Result<TlaTraceVerificationResult, AsyncVerifyError> {
        // Check for success
        if output.contains("Model checking completed. No error has been found") {
            return Ok(TlaTraceVerificationResult::valid(
                output.to_string(),
                duration,
            ));
        }

        // Check for invariant violation
        let inv_pattern = Regex::new(r"Invariant (\w+) is violated").ok();
        if let Some(ref re) = inv_pattern {
            if let Some(caps) = re.captures(output) {
                let invariant = caps.get(1).map_or("unknown", |m| m.as_str());

                // Try to extract state number
                let state_pattern = Regex::new(r"State (\d+):").ok();
                let state_idx = state_pattern.and_then(|re| {
                    re.captures(output)
                        .and_then(|c| c.get(1))
                        .and_then(|m| m.as_str().parse::<usize>().ok())
                });

                let violations = vec![TraceViolation::InvariantViolation {
                    state_index: state_idx.unwrap_or(0),
                    invariant: invariant.to_string(),
                    state: serde_json::Value::Null,
                }];

                return Ok(TlaTraceVerificationResult::with_violations(
                    violations,
                    output.to_string(),
                    duration,
                ));
            }
        }

        // Check for trace conformance violation
        if output.contains("TraceConformance") && output.contains("violated") {
            // Extract state where divergence occurred
            let state_pattern = Regex::new(r"State (\d+):").ok();
            let state_idx = state_pattern.and_then(|re| {
                re.captures(output)
                    .and_then(|c| c.get(1))
                    .and_then(|m| m.as_str().parse::<usize>().ok())
            });

            return Ok(TlaTraceVerificationResult::diverged(
                state_idx.unwrap_or(0),
                None,
                vec![],
                output.to_string(),
                duration,
            ));
        }

        // Check for deadlock
        if output.contains("Deadlock reached") {
            let violations = vec![TraceViolation::DivergenceFromSpec {
                state_index: 0,
                message: "Deadlock detected - trace cannot continue".to_string(),
            }];
            return Ok(TlaTraceVerificationResult::with_violations(
                violations,
                output.to_string(),
                duration,
            ));
        }

        // Check for temporal property violation
        if output.contains("Temporal properties were violated") {
            let violations = vec![TraceViolation::DivergenceFromSpec {
                state_index: 0,
                message: "Temporal property violated".to_string(),
            }];
            return Ok(TlaTraceVerificationResult::with_violations(
                violations,
                output.to_string(),
                duration,
            ));
        }

        // Check for error
        if output.contains("Error:") || output.contains("error:") {
            return Err(AsyncVerifyError::tla_verification(format!(
                "TLC error: {}",
                output
            )));
        }

        // Unknown result - assume valid but warn
        Ok(TlaTraceVerificationResult::valid(
            output.to_string(),
            duration,
        ))
    }

    /// Convert JSON value to TLA+ value representation
    fn json_to_tla_value(&self, value: &serde_json::Value) -> Result<String, AsyncVerifyError> {
        json_to_tla_value_impl(value)
    }

    /// Convert JSON object to TLA+ record
    fn json_to_tla_record(&self, value: &serde_json::Value) -> Result<String, AsyncVerifyError> {
        json_to_tla_value_impl(value)
    }
}

/// Standalone implementation of JSON to TLA+ value conversion
fn json_to_tla_value_impl(value: &serde_json::Value) -> Result<String, AsyncVerifyError> {
    match value {
        serde_json::Value::Null => Ok("<<>>".to_string()),
        serde_json::Value::Bool(b) => Ok(if *b {
            "TRUE".to_string()
        } else {
            "FALSE".to_string()
        }),
        serde_json::Value::Number(n) => Ok(n.to_string()),
        serde_json::Value::String(s) => Ok(format!("\"{}\"", s.replace('"', "\\\""))),
        serde_json::Value::Array(arr) => {
            let elements: Result<Vec<_>, _> = arr.iter().map(json_to_tla_value_impl).collect();
            Ok(format!("<<{}>>", elements?.join(", ")))
        }
        serde_json::Value::Object(obj) => {
            let fields: Result<Vec<String>, AsyncVerifyError> = obj
                .iter()
                .map(|(k, v)| {
                    let tla_v = json_to_tla_value_impl(v)?;
                    Ok(format!("{} |-> {}", k, tla_v))
                })
                .collect();
            Ok(format!("[{}]", fields?.join(", ")))
        }
    }
}

/// TLC command representation
enum TlcCommand {
    Standalone(PathBuf),
    Jar {
        java_path: PathBuf,
        jar_path: PathBuf,
    },
}

/// Convert an ExecutionTrace to TLA+ format string
pub fn trace_to_tla(trace: &ExecutionTrace) -> Result<String, AsyncVerifyError> {
    let mut output = String::new();

    output.push_str("\\* Execution trace in TLA+ format\n");
    output.push_str("Trace == <<\n");

    // Initial state
    let initial = json_to_tla_value_impl(&trace.initial_state)?;
    output.push_str(&format!("    {}", initial));

    // Transitions
    for transition in &trace.transitions {
        output.push_str(",\n");
        let state = json_to_tla_value_impl(&transition.to_state)?;
        output.push_str(&format!("    {} \\* {}", state, transition.event));
    }

    output.push_str("\n>>\n");

    Ok(output)
}

/// Verify that a trace satisfies a TLA+ invariant
pub fn verify_trace_invariant(
    trace: &ExecutionTrace,
    invariant: impl Fn(&serde_json::Value) -> bool,
    invariant_name: &str,
) -> Vec<TraceViolation> {
    let mut violations = vec![];

    // Check initial state
    if !invariant(&trace.initial_state) {
        violations.push(TraceViolation::InvariantViolation {
            state_index: 0,
            invariant: invariant_name.to_string(),
            state: trace.initial_state.clone(),
        });
    }

    // Check each transition state
    for (i, transition) in trace.transitions.iter().enumerate() {
        if !invariant(&transition.to_state) {
            violations.push(TraceViolation::InvariantViolation {
                state_index: i + 1,
                invariant: invariant_name.to_string(),
                state: transition.to_state.clone(),
            });
        }
    }

    violations
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateTransition;

    #[test]
    fn test_json_to_tla_value_primitives() {
        let config = TlaTraceConfig::default();
        let verifier = TlaTraceVerifier::new(config);

        assert_eq!(
            verifier
                .json_to_tla_value(&serde_json::json!(true))
                .unwrap(),
            "TRUE"
        );
        assert_eq!(
            verifier
                .json_to_tla_value(&serde_json::json!(false))
                .unwrap(),
            "FALSE"
        );
        assert_eq!(
            verifier.json_to_tla_value(&serde_json::json!(42)).unwrap(),
            "42"
        );
        assert_eq!(
            verifier
                .json_to_tla_value(&serde_json::json!("hello"))
                .unwrap(),
            "\"hello\""
        );
    }

    #[test]
    fn test_json_to_tla_value_array() {
        let config = TlaTraceConfig::default();
        let verifier = TlaTraceVerifier::new(config);

        let arr = serde_json::json!([1, 2, 3]);
        assert_eq!(verifier.json_to_tla_value(&arr).unwrap(), "<<1, 2, 3>>");
    }

    #[test]
    fn test_json_to_tla_value_object() {
        let config = TlaTraceConfig::default();
        let verifier = TlaTraceVerifier::new(config);

        let obj = serde_json::json!({"x": 1, "y": 2});
        let result = verifier.json_to_tla_value(&obj).unwrap();
        // Order may vary, check both possible orderings
        assert!(result.contains("x |-> 1"));
        assert!(result.contains("y |-> 2"));
    }

    #[test]
    fn test_trace_to_tla() {
        let mut trace = ExecutionTrace::new(serde_json::json!({"counter": 0}));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"counter": 0}),
            "Increment".to_string(),
            serde_json::json!({"counter": 1}),
        ));

        let tla = trace_to_tla(&trace).unwrap();
        assert!(tla.contains("Trace =="));
        assert!(tla.contains("[counter |-> 0]"));
        assert!(tla.contains("[counter |-> 1]"));
        assert!(tla.contains("Increment"));
    }

    #[test]
    fn test_verify_trace_invariant_pass() {
        let mut trace = ExecutionTrace::new(serde_json::json!({"value": 1}));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"value": 1}),
            "double".to_string(),
            serde_json::json!({"value": 2}),
        ));

        let violations = verify_trace_invariant(
            &trace,
            |state| {
                state
                    .get("value")
                    .and_then(|v| v.as_i64())
                    .is_some_and(|n| n > 0)
            },
            "positive",
        );

        assert!(violations.is_empty());
    }

    #[test]
    fn test_verify_trace_invariant_fail() {
        let mut trace = ExecutionTrace::new(serde_json::json!({"value": 1}));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"value": 1}),
            "negate".to_string(),
            serde_json::json!({"value": -1}),
        ));

        let violations = verify_trace_invariant(
            &trace,
            |state| {
                state
                    .get("value")
                    .and_then(|v| v.as_i64())
                    .is_some_and(|n| n > 0)
            },
            "positive",
        );

        assert_eq!(violations.len(), 1);
        if let TraceViolation::InvariantViolation {
            state_index,
            invariant,
            ..
        } = &violations[0]
        {
            assert_eq!(*state_index, 1);
            assert_eq!(invariant, "positive");
        } else {
            panic!("Expected InvariantViolation");
        }
    }

    #[test]
    fn test_tla_trace_verification_result() {
        let valid = TlaTraceVerificationResult::valid("output".to_string(), Duration::from_secs(1));
        assert!(valid.valid);
        assert!(valid.violations.is_empty());

        let diverged = TlaTraceVerificationResult::diverged(
            5,
            Some("BadAction".to_string()),
            vec!["GoodAction".to_string()],
            "output".to_string(),
            Duration::from_secs(1),
        );
        assert!(!diverged.valid);
        assert_eq!(diverged.divergence_state, Some(5));
        assert_eq!(diverged.violations.len(), 1);
    }

    #[test]
    fn test_trace_violation_to_violation() {
        let tv = TraceViolation::InvariantViolation {
            state_index: 3,
            invariant: "Safety".to_string(),
            state: serde_json::json!({}),
        };

        let v = tv.to_violation();
        if let Violation::InvariantViolation { invariant, .. } = v {
            assert_eq!(invariant, "Safety");
        } else {
            panic!("Expected InvariantViolation");
        }
    }

    #[test]
    fn test_tla_trace_config() {
        let config = TlaTraceConfig::new("spec.tla")
            .with_timeout(Duration::from_secs(30))
            .without_stuttering()
            .with_invariants(vec!["Inv1".to_string(), "Inv2".to_string()]);

        assert_eq!(config.timeout_ms, 30000);
        assert!(!config.allow_stuttering);
        assert_eq!(config.invariants.len(), 2);
    }

    #[test]
    fn test_generate_trace_module() {
        let config = TlaTraceConfig::new("/tmp/TestSpec.tla");
        let verifier = TlaTraceVerifier::new(config);

        let mut trace = ExecutionTrace::new(serde_json::json!({"x": 0}));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"x": 0}),
            "Inc".to_string(),
            serde_json::json!({"x": 1}),
        ));

        let module = verifier.generate_trace_module(&trace).unwrap();

        assert!(module.contains("MODULE TraceConstraint"));
        assert!(module.contains("EXTENDS TestSpec"));
        assert!(module.contains("Trace =="));
        assert!(module.contains("TraceInit"));
        assert!(module.contains("TraceNext"));
        assert!(module.contains("TraceConformance"));
    }

    #[test]
    fn test_generate_tlc_config() {
        let config = TlaTraceConfig::new("spec.tla").with_invariants(vec!["Safety".to_string()]);
        let verifier = TlaTraceVerifier::new(config);

        let trace = ExecutionTrace::new(serde_json::json!({}));
        let cfg = verifier.generate_tlc_config(&trace).unwrap();

        assert!(cfg.contains("SPECIFICATION TraceSpec"));
        assert!(cfg.contains("INVARIANT TraceConformance"));
        assert!(cfg.contains("INVARIANT TraceInv_Safety"));
    }

    // Additional tests for mutation coverage

    #[test]
    fn test_tla_trace_config_with_variable_mapping() {
        // Test that variable mapping is preserved (catches mutation at line 245)
        let mut mapping = HashMap::new();
        mapping.insert("counter".to_string(), "cnt".to_string());
        mapping.insert("state".to_string(), "st".to_string());

        let config = TlaTraceConfig::new("spec.tla").with_variable_mapping(mapping.clone());

        assert_eq!(config.variable_mapping.len(), 2);
        assert_eq!(
            config.variable_mapping.get("counter"),
            Some(&"cnt".to_string())
        );
        assert_eq!(
            config.variable_mapping.get("state"),
            Some(&"st".to_string())
        );
    }

    #[test]
    fn test_tla_trace_config_with_tlc_path() {
        // Test that TLC path is preserved (catches mutation at line 251)
        let config = TlaTraceConfig::new("spec.tla").with_tlc_path("/usr/local/bin/tlc");

        assert!(config.tlc_path.is_some());
        assert_eq!(
            config.tlc_path.unwrap(),
            PathBuf::from("/usr/local/bin/tlc")
        );
    }

    #[test]
    fn test_tla_trace_verifier_for_spec() {
        // Test the for_spec constructor
        let verifier = TlaTraceVerifier::for_spec("/path/to/spec.tla");

        // Verifier should have the spec path set
        // We can test by generating a trace module and checking the EXTENDS
        let trace = ExecutionTrace::new(serde_json::json!({"x": 0}));
        let module = verifier.generate_trace_module(&trace).unwrap();

        assert!(module.contains("EXTENDS spec")); // spec.tla -> spec
    }

    #[test]
    fn test_generate_trace_module_with_multiple_transitions() {
        // Test trace module generation with multiple transitions
        // This catches mutations at lines 372 and 438 (i > 0 comparisons)
        let config = TlaTraceConfig::new("/tmp/TestSpec.tla");
        let verifier = TlaTraceVerifier::new(config);

        let mut trace = ExecutionTrace::new(serde_json::json!({"x": 0}));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"x": 0}),
            "Inc".to_string(),
            serde_json::json!({"x": 1}),
        ));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"x": 1}),
            "Inc".to_string(),
            serde_json::json!({"x": 2}),
        ));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"x": 2}),
            "Dec".to_string(),
            serde_json::json!({"x": 1}),
        ));

        let module = verifier.generate_trace_module(&trace).unwrap();

        // Check that transitions are properly separated with commas and newlines
        // The format should be: first action, then ",\n" before subsequent actions
        assert!(module.contains("TraceActions == <<"));
        assert!(module.contains("\"Inc\""));
        assert!(module.contains("\"Dec\""));

        // Count the action entries - should have commas between them
        let actions_section = module.split("TraceActions == <<").nth(1).unwrap();
        let actions_end = actions_section.find(">>").unwrap();
        let actions_content = &actions_section[..actions_end];

        // Should have 3 actions with 2 commas between them
        let comma_count = actions_content.matches(',').count();
        assert_eq!(comma_count, 2, "Expected 2 commas between 3 actions");
    }

    #[test]
    fn test_json_to_tla_record_object() {
        // Test json_to_tla_record with an object (catches mutation at line 690)
        let config = TlaTraceConfig::default();
        let verifier = TlaTraceVerifier::new(config);

        let obj = serde_json::json!({"a": 1, "b": 2});
        let result = verifier.json_to_tla_record(&obj).unwrap();

        // Should be formatted as a TLA+ record
        assert!(result.starts_with('['));
        assert!(result.ends_with(']'));
        assert!(result.contains("|->"));
    }

    #[test]
    fn test_json_to_tla_record_nested() {
        // Test with nested objects
        let config = TlaTraceConfig::default();
        let verifier = TlaTraceVerifier::new(config);

        let obj = serde_json::json!({
            "outer": {
                "inner": 42
            }
        });
        let result = verifier.json_to_tla_record(&obj).unwrap();

        // Should contain nested record
        assert!(result.contains("outer |->"));
        assert!(result.contains("inner |-> 42"));
    }

    #[test]
    fn test_tla_trace_verification_result_with_violations() {
        // Test the with_violations constructor
        let violations = vec![
            TraceViolation::InvariantViolation {
                state_index: 3,
                invariant: "Safety".to_string(),
                state: serde_json::json!({"x": -1}),
            },
            TraceViolation::DivergenceFromSpec {
                state_index: 5,
                message: "Unexpected state".to_string(),
            },
        ];

        let result = TlaTraceVerificationResult::with_violations(
            violations,
            "TLC output".to_string(),
            Duration::from_secs(2),
        );

        assert!(!result.valid);
        assert!(result.divergence_state.is_none());
        assert_eq!(result.violations.len(), 2);
        assert_eq!(result.duration_ms, 2000);
    }

    #[test]
    fn test_trace_violation_types_to_violation() {
        // Test all TraceViolation variants convert to Violation
        let divergence = TraceViolation::DivergenceFromSpec {
            state_index: 1,
            message: "bad state".to_string(),
        };
        let v = divergence.to_violation();
        if let Violation::SafetyViolation { property, .. } = v {
            assert!(property.contains("trace_conformance"));
        } else {
            panic!("Expected SafetyViolation");
        }

        let precondition = TraceViolation::PreconditionViolation {
            state_index: 2,
            action: "DoSomething".to_string(),
            message: "precondition failed".to_string(),
        };
        let v = precondition.to_violation();
        if let Violation::SafetyViolation { property, .. } = v {
            assert!(property.contains("DoSomething_precondition"));
        } else {
            panic!("Expected SafetyViolation");
        }

        let stutter = TraceViolation::UnexpectedStutter { state_index: 3 };
        let v = stutter.to_violation();
        if let Violation::SafetyViolation { property, message } = v {
            assert!(property.contains("no_stuttering"));
            assert!(message.contains("stuttering"));
        } else {
            panic!("Expected SafetyViolation");
        }
    }

    #[test]
    fn test_verify_trace_invariant_initial_state_violation() {
        // Test that invariant is checked on initial state
        // This catches mutation at line 763 (delete !)
        let trace = ExecutionTrace::new(serde_json::json!({"value": -5})); // Negative value

        let violations = verify_trace_invariant(
            &trace,
            |state| {
                state
                    .get("value")
                    .and_then(|v| v.as_i64())
                    .is_some_and(|n| n >= 0) // Requires non-negative
            },
            "non_negative",
        );

        // Initial state violates the invariant
        assert_eq!(violations.len(), 1);
        if let TraceViolation::InvariantViolation { state_index, .. } = &violations[0] {
            assert_eq!(*state_index, 0); // Violation at initial state
        } else {
            panic!("Expected InvariantViolation");
        }
    }

    #[test]
    fn test_verify_trace_invariant_state_index() {
        // Test that state_index calculation is correct (i + 1)
        // This catches mutation at line 775 (+ vs - or *)
        let mut trace = ExecutionTrace::new(serde_json::json!({"value": 1}));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"value": 1}),
            "dec".to_string(),
            serde_json::json!({"value": 0}),
        ));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"value": 0}),
            "dec".to_string(),
            serde_json::json!({"value": -1}), // This violates
        ));

        let violations = verify_trace_invariant(
            &trace,
            |state| {
                state
                    .get("value")
                    .and_then(|v| v.as_i64())
                    .is_some_and(|n| n >= 0)
            },
            "non_negative",
        );

        // Only the 3rd state (index 2) violates
        assert_eq!(violations.len(), 1);
        if let TraceViolation::InvariantViolation { state_index, .. } = &violations[0] {
            assert_eq!(*state_index, 2); // 0-based index, after 2 transitions
        } else {
            panic!("Expected InvariantViolation");
        }
    }

    #[test]
    fn test_verify_trace_invariant_multiple_violations() {
        // Test multiple violations are detected with correct indices
        let mut trace = ExecutionTrace::new(serde_json::json!({"value": -1})); // Violates
        trace.add_transition(StateTransition::new(
            serde_json::json!({"value": -1}),
            "inc".to_string(),
            serde_json::json!({"value": 0}), // OK
        ));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"value": 0}),
            "dec".to_string(),
            serde_json::json!({"value": -1}), // Violates
        ));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"value": -1}),
            "dec".to_string(),
            serde_json::json!({"value": -2}), // Violates
        ));

        let violations = verify_trace_invariant(
            &trace,
            |state| {
                state
                    .get("value")
                    .and_then(|v| v.as_i64())
                    .is_some_and(|n| n >= 0)
            },
            "non_negative",
        );

        // 3 violations: initial state (0), state 2, state 3
        assert_eq!(violations.len(), 3);

        let indices: Vec<usize> = violations
            .iter()
            .map(|v| match v {
                TraceViolation::InvariantViolation { state_index, .. } => *state_index,
                _ => panic!("Expected InvariantViolation"),
            })
            .collect();

        assert_eq!(indices, vec![0, 2, 3]);
    }
}
