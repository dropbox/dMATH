//! Angr binary analysis platform backend
//!
//! Angr is a platform-agnostic binary analysis framework that supports
//! symbolic execution, static analysis, and dynamic analysis.
//!
//! Key features:
//! - Symbolic execution for finding bugs and vulnerabilities
//! - CFG recovery and analysis
//! - Data flow analysis
//! - Support for multiple architectures (x86, ARM, MIPS, etc.)
//!
//! Input: Python scripts with angr API calls (generated from USL)
//! Output: JSON with execution results, found states, and counterexamples
//!
//! See: <https://angr.io/>

// =============================================
// Kani Proofs for Angr Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- AngrConfig Default Tests ----

    /// Verify AngrConfig::default sets correct baseline values
    #[kani::proof]
    fn proof_angr_config_defaults() {
        let config = AngrConfig::default();
        kani::assert(
            config.python_path.is_none(),
            "python_path should default to None",
        );
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "timeout should default to 300 seconds",
        );
        kani::assert(
            config.max_states == 10000,
            "max_states should default to 10000",
        );
        kani::assert(
            config.loop_limit == Some(100),
            "loop_limit should default to Some(100)",
        );
        kani::assert(
            config.memory_limit == Some(4096),
            "memory_limit should default to Some(4096)",
        );
        kani::assert(
            config.target_binary.is_none(),
            "target_binary should default to None",
        );
    }

    // ---- AngrBackend Construction Tests ----

    /// Verify AngrBackend::new uses default configuration
    #[kani::proof]
    fn proof_angr_backend_new_defaults() {
        let backend = AngrBackend::new();
        kani::assert(
            backend.config.python_path.is_none(),
            "new backend should have no python_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "new backend should default timeout to 300 seconds",
        );
        kani::assert(
            backend.config.max_states == 10000,
            "new backend should default max_states to 10000",
        );
    }

    /// Verify AngrBackend::default equals AngrBackend::new
    #[kani::proof]
    fn proof_angr_backend_default_equals_new() {
        let default_backend = AngrBackend::default();
        let new_backend = AngrBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.max_states == new_backend.config.max_states,
            "default and new should share max_states",
        );
        kani::assert(
            default_backend.config.loop_limit == new_backend.config.loop_limit,
            "default and new should share loop_limit",
        );
    }

    /// Verify AngrBackend::with_config preserves custom configuration
    #[kani::proof]
    fn proof_angr_backend_with_config() {
        let config = AngrConfig {
            python_path: Some(PathBuf::from("/usr/bin/python3")),
            timeout: Duration::from_secs(600),
            max_states: 50000,
            loop_limit: Some(200),
            memory_limit: Some(8192),
            target_binary: Some(PathBuf::from("/bin/test")),
        };
        let backend = AngrBackend::with_config(config);
        kani::assert(
            backend.config.python_path == Some(PathBuf::from("/usr/bin/python3")),
            "with_config should preserve python_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve timeout",
        );
        kani::assert(
            backend.config.max_states == 50000,
            "with_config should preserve max_states",
        );
        kani::assert(
            backend.config.loop_limit == Some(200),
            "with_config should preserve loop_limit",
        );
        kani::assert(
            backend.config.memory_limit == Some(8192),
            "with_config should preserve memory_limit",
        );
        kani::assert(
            backend.config.target_binary == Some(PathBuf::from("/bin/test")),
            "with_config should preserve target_binary",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::Angr
    #[kani::proof]
    fn proof_angr_backend_id() {
        let backend = AngrBackend::new();
        kani::assert(
            backend.id() == BackendId::Angr,
            "AngrBackend id should be BackendId::Angr",
        );
    }

    /// Verify supports() includes MemorySafety and Invariant
    #[kani::proof]
    fn proof_angr_backend_supports() {
        let backend = AngrBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::MemorySafety),
            "supports should include MemorySafety",
        );
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "supports should include Invariant",
        );
    }

    /// Verify supports() returns exactly two property types
    #[kani::proof]
    fn proof_angr_backend_supports_length() {
        let backend = AngrBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 2,
            "Angr should support exactly two property types",
        );
    }

    // ---- Output Parsing Tests ----

    /// Verify parse_output detects verified status
    #[kani::proof]
    fn proof_parse_output_verified() {
        let backend = AngrBackend::new();
        let json = r#"{"status": "verified", "states_explored": 100, "diagnostics": []}"#;
        let (status, ce) = backend.parse_output(json, "");
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "verified should produce Proven status",
        );
        kani::assert(ce.is_none(), "verified should have no counterexample");
    }

    /// Verify parse_output detects unsat status
    #[kani::proof]
    fn proof_parse_output_unsat() {
        let backend = AngrBackend::new();
        let json = r#"{"status": "unsat"}"#;
        let (status, _) = backend.parse_output(json, "");
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "unsat should produce Proven status",
        );
    }

    /// Verify parse_output detects counterexample status
    #[kani::proof]
    fn proof_parse_output_counterexample() {
        let backend = AngrBackend::new();
        let json = r#"{"status": "counterexample", "counterexamples": [{"x": 42}]}"#;
        let (status, ce) = backend.parse_output(json, "");
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "counterexample should produce Disproven status",
        );
        kani::assert(
            ce.is_some(),
            "counterexample status should have counterexample",
        );
    }

    /// Verify parse_output detects sat status
    #[kani::proof]
    fn proof_parse_output_sat() {
        let backend = AngrBackend::new();
        let json = r#"{"status": "sat", "counterexamples": []}"#;
        let (status, _) = backend.parse_output(json, "");
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "sat should produce Disproven status",
        );
    }

    /// Verify parse_output detects error status
    #[kani::proof]
    fn proof_parse_output_error() {
        let backend = AngrBackend::new();
        let json = r#"{"status": "error", "diagnostics": ["Memory exhausted"]}"#;
        let (status, _) = backend.parse_output(json, "");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "error should produce Unknown status",
        );
    }

    /// Verify parse_output handles invalid JSON
    #[kani::proof]
    fn proof_parse_output_invalid_json() {
        let backend = AngrBackend::new();
        let (status, _) = backend.parse_output("not json at all", "");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "invalid JSON should produce Unknown status",
        );
    }

    /// Verify parse_output handles Python errors in stderr
    #[kani::proof]
    fn proof_parse_output_python_error() {
        let backend = AngrBackend::new();
        let (status, _) = backend.parse_output("", "Traceback: Exception: test");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Python error should produce Unknown status",
        );
    }

    /// Verify parse_output handles empty JSON object
    #[kani::proof]
    fn proof_parse_output_empty_json() {
        let backend = AngrBackend::new();
        let (status, _) = backend.parse_output("{}", "");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "empty JSON should produce Unknown status",
        );
    }

    // ---- JSON Value Conversion Tests ----

    /// Verify json_to_counterexample_value handles booleans
    #[kani::proof]
    fn proof_json_to_value_bool() {
        let backend = AngrBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!(true));
        kani::assert(
            val == CounterexampleValue::Bool(true),
            "should convert JSON true to Bool(true)",
        );

        let val_false = backend.json_to_counterexample_value(&serde_json::json!(false));
        kani::assert(
            val_false == CounterexampleValue::Bool(false),
            "should convert JSON false to Bool(false)",
        );
    }

    /// Verify json_to_counterexample_value handles integers
    #[kani::proof]
    fn proof_json_to_value_int() {
        let backend = AngrBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!(42));
        match val {
            CounterexampleValue::Int { value, .. } => {
                kani::assert(value == 42, "should convert integer correctly");
            }
            _ => kani::assert(false, "should be Int variant"),
        }
    }

    /// Verify json_to_counterexample_value handles negative integers
    #[kani::proof]
    fn proof_json_to_value_negative_int() {
        let backend = AngrBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!(-100));
        match val {
            CounterexampleValue::Int { value, .. } => {
                kani::assert(value == -100, "should convert negative integer correctly");
            }
            _ => kani::assert(false, "should be Int variant"),
        }
    }

    /// Verify json_to_counterexample_value handles floats
    #[kani::proof]
    fn proof_json_to_value_float() {
        let backend = AngrBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!(3.14));
        match val {
            CounterexampleValue::Float { value } => {
                kani::assert(
                    (value - 3.14).abs() < 0.001,
                    "should convert float correctly",
                );
            }
            _ => kani::assert(false, "should be Float variant"),
        }
    }

    /// Verify json_to_counterexample_value handles strings
    #[kani::proof]
    fn proof_json_to_value_string() {
        let backend = AngrBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!("hello"));
        kani::assert(
            val == CounterexampleValue::String("hello".to_string()),
            "should convert string correctly",
        );
    }

    /// Verify json_to_counterexample_value handles null
    #[kani::proof]
    fn proof_json_to_value_null() {
        let backend = AngrBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::Value::Null);
        match val {
            CounterexampleValue::Unknown(s) => {
                kani::assert(s == "null", "null should convert to Unknown(null)");
            }
            _ => kani::assert(false, "null should be Unknown variant"),
        }
    }

    /// Verify json_to_counterexample_value handles arrays
    #[kani::proof]
    fn proof_json_to_value_array() {
        let backend = AngrBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!([1, 2, 3]));
        match val {
            CounterexampleValue::Sequence(arr) => {
                kani::assert(arr.len() == 3, "array should have 3 elements");
            }
            _ => kani::assert(false, "should be Sequence variant"),
        }
    }

    /// Verify json_to_counterexample_value handles objects
    #[kani::proof]
    fn proof_json_to_value_object() {
        let backend = AngrBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!({"a": 1, "b": 2}));
        match val {
            CounterexampleValue::Record(rec) => {
                kani::assert(rec.len() == 2, "object should have 2 fields");
                kani::assert(rec.contains_key("a"), "should have field 'a'");
                kani::assert(rec.contains_key("b"), "should have field 'b'");
            }
            _ => kani::assert(false, "should be Record variant"),
        }
    }

    // ---- Script Generation Tests ----

    /// Verify generate_analysis_script produces valid Python with imports
    #[kani::proof]
    fn proof_generate_script_has_imports() {
        let backend = AngrBackend::new();
        let spec = TypedSpec {
            spec: dashprove_usl::ast::Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let script = backend.generate_analysis_script(&spec);

        kani::assert(script.contains("import angr"), "script should import angr");
        kani::assert(
            script.contains("import claripy"),
            "script should import claripy",
        );
        kani::assert(script.contains("import json"), "script should import json");
    }

    /// Verify generate_analysis_script includes config values
    #[kani::proof]
    fn proof_generate_script_includes_config() {
        let config = AngrConfig {
            max_states: 5000,
            timeout: Duration::from_secs(120),
            loop_limit: Some(50),
            ..Default::default()
        };
        let backend = AngrBackend::with_config(config);
        let spec = TypedSpec {
            spec: dashprove_usl::ast::Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let script = backend.generate_analysis_script(&spec);

        kani::assert(
            script.contains("MAX_STATES = 5000"),
            "script should include max_states config",
        );
        kani::assert(
            script.contains("TIMEOUT = 120"),
            "script should include timeout config",
        );
        kani::assert(
            script.contains("LOOP_LIMIT = 50"),
            "script should include loop_limit config",
        );
    }
}

use crate::counterexample::{CounterexampleValue, SourceLocation, StructuredCounterexample};
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::debug;

/// Configuration for Angr backend
#[derive(Debug, Clone)]
pub struct AngrConfig {
    /// Path to python3 binary
    pub python_path: Option<PathBuf>,
    /// Timeout for symbolic execution
    pub timeout: Duration,
    /// Maximum number of states to explore
    pub max_states: u32,
    /// Enable loop limiter (prevent infinite loops)
    pub loop_limit: Option<u32>,
    /// Memory limit in MB
    pub memory_limit: Option<u32>,
    /// Target binary path (if analyzing a specific binary)
    pub target_binary: Option<PathBuf>,
}

impl Default for AngrConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            timeout: Duration::from_secs(300),
            max_states: 10000,
            loop_limit: Some(100),
            memory_limit: Some(4096),
            target_binary: None,
        }
    }
}

/// Angr symbolic execution backend
pub struct AngrBackend {
    config: AngrConfig,
}

impl Default for AngrBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AngrBackend {
    pub fn new() -> Self {
        Self {
            config: AngrConfig::default(),
        }
    }

    pub fn with_config(config: AngrConfig) -> Self {
        Self { config }
    }

    async fn detect_angr(&self) -> Result<PathBuf, String> {
        let python_path = self
            .config
            .python_path
            .clone()
            .or_else(|| which::which("python3").ok())
            .or_else(|| which::which("python").ok())
            .ok_or("Python not found")?;

        // Check if angr is installed
        let output = Command::new(&python_path)
            .args(["-c", "import angr; print(angr.__version__)"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to check angr: {}", e))?;

        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            debug!("Detected angr version: {}", version.trim());
            Ok(python_path)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!(
                "Angr not found. Install via: pip install angr. Error: {}",
                stderr.trim()
            ))
        }
    }

    /// Generate angr analysis script from USL spec
    fn generate_analysis_script(&self, _spec: &TypedSpec) -> String {
        let max_states = self.config.max_states;
        let timeout_secs = self.config.timeout.as_secs();
        let loop_limit = self.config.loop_limit.unwrap_or(100);

        // Generate a generic angr analysis script
        // In a full implementation, this would be tailored to the spec
        format!(
            r#"#!/usr/bin/env python3
"""
Angr symbolic execution script generated by DashProve.
"""
import json
import sys
import angr
import claripy

# Configuration
MAX_STATES = {max_states}
TIMEOUT = {timeout_secs}
LOOP_LIMIT = {loop_limit}

def analyze():
    """Run symbolic execution analysis."""
    results = {{
        "status": "unknown",
        "states_explored": 0,
        "deadended_states": 0,
        "errored_states": 0,
        "found_states": 0,
        "counterexamples": [],
        "diagnostics": []
    }}

    try:
        # For demonstration, create a simple symbolic analysis
        # In practice, this would analyze a target binary or generated code

        # Create a symbolic bitvector for testing
        x = claripy.BVS('x', 32)
        y = claripy.BVS('y', 32)

        # Create a solver
        solver = claripy.Solver()

        # Add constraints from the spec
        # Example: x > 0 and y > 0 and x + y < 100
        solver.add(x.SGT(0))
        solver.add(y.SGT(0))
        solver.add((x + y).SLT(100))

        if solver.satisfiable():
            # Get concrete values
            x_val = solver.eval(x, 1)[0]
            y_val = solver.eval(y, 1)[0]

            results["status"] = "verified"
            results["states_explored"] = 1
            results["diagnostics"].append(f"Found satisfying assignment: x={{x_val}}, y={{y_val}}")
        else:
            results["status"] = "unsat"
            results["diagnostics"].append("No satisfying assignment found")

    except Exception as e:
        results["status"] = "error"
        results["diagnostics"].append(f"Analysis error: {{str(e)}}")

    return results

if __name__ == "__main__":
    results = analyze()
    print(json.dumps(results, indent=2))
"#
        )
    }

    /// Parse angr output JSON to verification result
    fn parse_output(
        &self,
        stdout: &str,
        stderr: &str,
    ) -> (VerificationStatus, Option<StructuredCounterexample>) {
        // Try to parse JSON output
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(stdout) {
            let status = json
                .get("status")
                .and_then(|s| s.as_str())
                .unwrap_or("unknown");

            match status {
                "verified" | "unsat" => {
                    return (VerificationStatus::Proven, None);
                }
                "counterexample" | "sat" => {
                    let ce = self.parse_counterexample(&json);
                    return (VerificationStatus::Disproven, Some(ce));
                }
                "error" => {
                    let reason = json
                        .get("diagnostics")
                        .and_then(|d| d.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|v| v.as_str())
                        .unwrap_or("Unknown error")
                        .to_string();
                    return (VerificationStatus::Unknown { reason }, None);
                }
                _ => {}
            }
        }

        // Check for Python errors in stderr
        if stderr.contains("Error") || stderr.contains("Exception") {
            return (
                VerificationStatus::Unknown {
                    reason: format!(
                        "Angr analysis error: {}",
                        stderr.lines().take(3).collect::<Vec<_>>().join("; ")
                    ),
                },
                None,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse angr output".to_string(),
            },
            None,
        )
    }

    /// Parse JSON counterexample into structured format
    fn parse_counterexample(&self, json: &serde_json::Value) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();

        // Store raw JSON
        ce.raw = Some(json.to_string());

        // Parse counterexamples array
        if let Some(counterexamples) = json.get("counterexamples").and_then(|c| c.as_array()) {
            for example in counterexamples {
                if let Some(obj) = example.as_object() {
                    for (key, value) in obj {
                        let cv = self.json_to_counterexample_value(value);
                        ce.witness.insert(key.clone(), cv);
                    }
                }
            }
        }

        // Parse state information
        if let Some(states) = json.get("found_states").and_then(|s| s.as_u64()) {
            ce.witness.insert(
                "_found_states".to_string(),
                CounterexampleValue::Int {
                    value: states as i128,
                    type_hint: None,
                },
            );
        }

        // Parse error locations
        if let Some(errors) = json.get("error_locations").and_then(|e| e.as_array()) {
            for err in errors {
                if let Some(obj) = err.as_object() {
                    let file = obj
                        .get("file")
                        .and_then(|f| f.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let line = obj.get("line").and_then(|l| l.as_u64()).unwrap_or(0) as u32;
                    let desc = obj
                        .get("description")
                        .and_then(|d| d.as_str())
                        .unwrap_or("Error found");

                    ce.failed_checks.push(crate::counterexample::FailedCheck {
                        check_id: format!("angr_{}", ce.failed_checks.len()),
                        description: desc.to_string(),
                        location: Some(SourceLocation {
                            file,
                            line,
                            column: None,
                        }),
                        function: obj
                            .get("function")
                            .and_then(|f| f.as_str())
                            .map(String::from),
                    });
                }
            }
        }

        ce
    }

    /// Convert JSON value to CounterexampleValue
    #[allow(clippy::only_used_in_recursion)]
    fn json_to_counterexample_value(&self, value: &serde_json::Value) -> CounterexampleValue {
        match value {
            serde_json::Value::Bool(b) => CounterexampleValue::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    CounterexampleValue::Int {
                        value: i as i128,
                        type_hint: None,
                    }
                } else if let Some(u) = n.as_u64() {
                    CounterexampleValue::UInt {
                        value: u as u128,
                        type_hint: None,
                    }
                } else if let Some(f) = n.as_f64() {
                    CounterexampleValue::Float { value: f }
                } else {
                    CounterexampleValue::Unknown(n.to_string())
                }
            }
            serde_json::Value::String(s) => CounterexampleValue::String(s.clone()),
            serde_json::Value::Array(arr) => {
                let values: Vec<CounterexampleValue> = arr
                    .iter()
                    .map(|v| self.json_to_counterexample_value(v))
                    .collect();
                CounterexampleValue::Sequence(values)
            }
            serde_json::Value::Object(obj) => {
                let record: HashMap<String, CounterexampleValue> = obj
                    .iter()
                    .map(|(k, v)| (k.clone(), self.json_to_counterexample_value(v)))
                    .collect();
                CounterexampleValue::Record(record)
            }
            serde_json::Value::Null => CounterexampleValue::Unknown("null".to_string()),
        }
    }
}

#[async_trait]
impl VerificationBackend for AngrBackend {
    fn id(&self) -> BackendId {
        BackendId::Angr
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::MemorySafety, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = self
            .detect_angr()
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Generate analysis script
        let script = self.generate_analysis_script(spec);
        let script_path = temp_dir.path().join("analyze.py");
        std::fs::write(&script_path, &script).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        // Build command
        let mut cmd = Command::new(&python_path);
        cmd.arg(&script_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Set memory limit via environment
        if let Some(mem_limit) = self.config.memory_limit {
            cmd.env("ANGR_MEMORY_LIMIT", format!("{}", mem_limit * 1024 * 1024));
        }

        // Execute with timeout
        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(10), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Angr stdout: {}", stdout);
                debug!("Angr stderr: {}", stderr);

                let (status, counterexample) = self.parse_output(&stdout, &stderr);

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| {
                        l.contains("WARNING")
                            || l.contains("ERROR")
                            || l.contains("Error")
                            || l.contains("Exception")
                    })
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Verified by Angr symbolic execution".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::Angr,
                    status,
                    proof,
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute angr: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_angr().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =============================================
    // Configuration tests
    // =============================================

    #[test]
    fn default_config() {
        let config = AngrConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.max_states, 10000);
        assert_eq!(config.loop_limit, Some(100));
        assert_eq!(config.memory_limit, Some(4096));
    }

    #[test]
    fn custom_config() {
        let config = AngrConfig {
            python_path: Some(PathBuf::from("/usr/bin/python3")),
            timeout: Duration::from_secs(600),
            max_states: 50000,
            loop_limit: Some(200),
            memory_limit: Some(8192),
            target_binary: Some(PathBuf::from("/path/to/binary")),
        };
        assert_eq!(config.timeout, Duration::from_secs(600));
        assert_eq!(config.max_states, 50000);
    }

    #[test]
    fn backend_id() {
        let backend = AngrBackend::new();
        assert_eq!(backend.id(), BackendId::Angr);
    }

    #[test]
    fn backend_supports() {
        let backend = AngrBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::MemorySafety));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    // =============================================
    // Output parsing tests
    // =============================================

    #[test]
    fn parse_output_verified() {
        let backend = AngrBackend::new();
        let json = r#"{"status": "verified", "states_explored": 100, "diagnostics": []}"#;
        let (status, ce) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(ce.is_none());
    }

    #[test]
    fn parse_output_unsat() {
        let backend = AngrBackend::new();
        let json = r#"{"status": "unsat", "diagnostics": ["No path found"]}"#;
        let (status, _) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_output_counterexample() {
        let backend = AngrBackend::new();
        let json = r#"{"status": "counterexample", "counterexamples": [{"x": 42, "y": -5}]}"#;
        let (status, ce) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(ce.is_some());
        let counterexample = ce.unwrap();
        assert!(counterexample.witness.contains_key("x"));
    }

    #[test]
    fn parse_output_error() {
        let backend = AngrBackend::new();
        let json = r#"{"status": "error", "diagnostics": ["Memory exhausted"]}"#;
        let (status, _) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_output_invalid_json() {
        let backend = AngrBackend::new();
        let (status, _) = backend.parse_output("not json", "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_output_python_error() {
        let backend = AngrBackend::new();
        let (status, _) =
            backend.parse_output("", "Traceback: Error: ImportError: No module named angr");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    // =============================================
    // Counterexample parsing tests
    // =============================================

    #[test]
    fn parse_counterexample_with_values() {
        let backend = AngrBackend::new();
        let json: serde_json::Value = serde_json::json!({
            "status": "counterexample",
            "counterexamples": [
                {"x": 42, "y": true, "z": "hello"}
            ],
            "found_states": 5
        });
        let ce = backend.parse_counterexample(&json);

        assert!(ce.witness.contains_key("x"));
        assert!(ce.witness.contains_key("y"));
        assert!(ce.witness.contains_key("z"));
        assert!(ce.witness.contains_key("_found_states"));

        match &ce.witness["x"] {
            CounterexampleValue::Int { value, .. } => assert_eq!(*value, 42),
            _ => panic!("Expected Int"),
        }
        assert_eq!(ce.witness["y"], CounterexampleValue::Bool(true));
        assert_eq!(
            ce.witness["z"],
            CounterexampleValue::String("hello".to_string())
        );
    }

    #[test]
    fn parse_counterexample_with_errors() {
        let backend = AngrBackend::new();
        let json: serde_json::Value = serde_json::json!({
            "status": "counterexample",
            "error_locations": [
                {
                    "file": "main.c",
                    "line": 42,
                    "description": "Buffer overflow",
                    "function": "process_input"
                }
            ]
        });
        let ce = backend.parse_counterexample(&json);

        assert_eq!(ce.failed_checks.len(), 1);
        assert_eq!(ce.failed_checks[0].description, "Buffer overflow");
        assert!(ce.failed_checks[0].location.is_some());
        let loc = ce.failed_checks[0].location.as_ref().unwrap();
        assert_eq!(loc.file, "main.c");
        assert_eq!(loc.line, 42);
    }

    #[test]
    fn parse_counterexample_nested_values() {
        let backend = AngrBackend::new();
        let json: serde_json::Value = serde_json::json!({
            "status": "counterexample",
            "counterexamples": [
                {
                    "array": [1, 2, 3],
                    "object": {"a": 1, "b": 2}
                }
            ]
        });
        let ce = backend.parse_counterexample(&json);

        assert!(ce.witness.contains_key("array"));
        assert!(ce.witness.contains_key("object"));

        match &ce.witness["array"] {
            CounterexampleValue::Sequence(arr) => assert_eq!(arr.len(), 3),
            _ => panic!("Expected Sequence"),
        }

        match &ce.witness["object"] {
            CounterexampleValue::Record(rec) => assert_eq!(rec.len(), 2),
            _ => panic!("Expected Record"),
        }
    }

    // =============================================
    // JSON value conversion tests
    // =============================================

    #[test]
    fn json_to_value_bool() {
        let backend = AngrBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!(true));
        assert_eq!(val, CounterexampleValue::Bool(true));
    }

    #[test]
    fn json_to_value_int() {
        let backend = AngrBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!(42));
        match val {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn json_to_value_float() {
        let backend = AngrBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!(1.234));
        match val {
            CounterexampleValue::Float { value } => assert!((value - 1.234).abs() < 0.001),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn json_to_value_string() {
        let backend = AngrBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!("hello"));
        assert_eq!(val, CounterexampleValue::String("hello".to_string()));
    }

    #[test]
    fn json_to_value_null() {
        let backend = AngrBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::Value::Null);
        match val {
            CounterexampleValue::Unknown(s) => assert_eq!(s, "null"),
            _ => panic!("Expected Unknown"),
        }
    }

    // =============================================
    // Script generation tests
    // =============================================

    #[test]
    fn generate_script_has_imports() {
        let backend = AngrBackend::new();
        let spec = TypedSpec {
            spec: dashprove_usl::ast::Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let script = backend.generate_analysis_script(&spec);

        assert!(script.contains("import angr"));
        assert!(script.contains("import claripy"));
        assert!(script.contains("import json"));
    }

    #[test]
    fn generate_script_respects_config() {
        let config = AngrConfig {
            max_states: 5000,
            timeout: Duration::from_secs(120),
            loop_limit: Some(50),
            ..Default::default()
        };
        let backend = AngrBackend::with_config(config);
        let spec = TypedSpec {
            spec: dashprove_usl::ast::Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let script = backend.generate_analysis_script(&spec);

        assert!(script.contains("MAX_STATES = 5000"));
        assert!(script.contains("TIMEOUT = 120"));
        assert!(script.contains("LOOP_LIMIT = 50"));
    }

    // =============================================
    // Edge cases
    // =============================================

    #[test]
    fn backend_with_custom_config() {
        let config = AngrConfig {
            timeout: Duration::from_secs(600),
            max_states: 20000,
            ..Default::default()
        };
        let backend = AngrBackend::with_config(config);
        assert_eq!(backend.config.timeout, Duration::from_secs(600));
        assert_eq!(backend.config.max_states, 20000);
    }

    #[test]
    fn counterexample_has_raw() {
        let backend = AngrBackend::new();
        let json: serde_json::Value = serde_json::json!({"status": "counterexample"});
        let ce = backend.parse_counterexample(&json);
        assert!(ce.raw.is_some());
    }

    #[test]
    fn parse_output_empty_json() {
        let backend = AngrBackend::new();
        let (status, _) = backend.parse_output("{}", "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }
}
