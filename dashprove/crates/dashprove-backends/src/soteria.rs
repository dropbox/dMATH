//! Soteria backend for Rust symbolic execution
//!
//! Soteria is a symbolic execution engine for compiled Rust with Tree Borrows
//! support and path-pruning heuristics. It operates on lifted MIR/LLVM bitcode.
//!
//! Key features:
//! - Symbolic execution of Rust binaries
//! - Tree Borrows memory model for aliasing
//! - Path pruning heuristics for scalability
//! - Counterexample generation
//!
//! Input: Rust MIR, LLVM bitcode, or binary
//! Output: Counterexample traces or verification result
//!
//! See: <https://arxiv.org/abs/2511.08729>

// =============================================
// Kani Proofs for Soteria Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- SoteriaConfig Default Tests ----

    /// Verify SoteriaConfig::default sets correct baseline values
    #[kani::proof]
    fn proof_soteria_config_defaults() {
        let config = SoteriaConfig::default();
        kani::assert(
            config.soteria_path.is_none(),
            "soteria_path should default to None",
        );
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "timeout should default to 300 seconds",
        );
        kani::assert(
            config.target_binary.is_none(),
            "target_binary should default to None",
        );
        kani::assert(
            config.memory_model == SoteriaMemoryModel::TreeBorrows,
            "memory_model should default to TreeBorrows",
        );
        kani::assert(config.max_depth == 100, "max_depth should default to 100");
    }

    // ---- SoteriaBackend Construction Tests ----

    /// Verify SoteriaBackend::new uses default configuration
    #[kani::proof]
    fn proof_soteria_backend_new_defaults() {
        let backend = SoteriaBackend::new();
        kani::assert(
            backend.config.soteria_path.is_none(),
            "new backend should have no soteria_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "new backend should default timeout to 300 seconds",
        );
    }

    /// Verify SoteriaBackend::default equals SoteriaBackend::new
    #[kani::proof]
    fn proof_soteria_backend_default_equals_new() {
        let default_backend = SoteriaBackend::default();
        let new_backend = SoteriaBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.memory_model == new_backend.config.memory_model,
            "default and new should share memory_model",
        );
    }

    /// Verify SoteriaBackend::with_config preserves custom configuration
    #[kani::proof]
    fn proof_soteria_backend_with_config() {
        let config = SoteriaConfig {
            soteria_path: Some(PathBuf::from("/usr/bin/soteria")),
            timeout: Duration::from_secs(600),
            target_binary: Some(PathBuf::from("/path/to/binary.bc")),
            memory_model: SoteriaMemoryModel::StackedBorrows,
            max_depth: 200,
            solver_timeout: Duration::from_secs(60),
            enable_path_merging: true,
            extra_args: vec!["--verbose".to_string()],
        };
        let backend = SoteriaBackend::with_config(config);
        kani::assert(
            backend.config.soteria_path == Some(PathBuf::from("/usr/bin/soteria")),
            "with_config should preserve soteria_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve timeout",
        );
        kani::assert(
            backend.config.memory_model == SoteriaMemoryModel::StackedBorrows,
            "with_config should preserve memory_model",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::Soteria
    #[kani::proof]
    fn proof_soteria_backend_id() {
        let backend = SoteriaBackend::new();
        kani::assert(
            backend.id() == BackendId::Soteria,
            "SoteriaBackend id should be BackendId::Soteria",
        );
    }

    /// Verify supports() includes MemorySafety
    #[kani::proof]
    fn proof_soteria_backend_supports() {
        let backend = SoteriaBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::MemorySafety),
            "supports should include MemorySafety",
        );
        kani::assert(
            supported.contains(&PropertyType::Contract),
            "supports should include Contract",
        );
    }

    // ---- Output Parsing Tests ----

    /// Verify parse_output detects verified status
    #[kani::proof]
    fn proof_parse_output_verified() {
        let backend = SoteriaBackend::new();
        let json = r#"{"status": "safe", "paths_explored": 42}"#;
        let (status, ce) = backend.parse_output(json, "");
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "safe should produce Proven status",
        );
        kani::assert(ce.is_none(), "safe should have no counterexample");
    }

    /// Verify parse_output detects counterexample status
    #[kani::proof]
    fn proof_parse_output_counterexample() {
        let backend = SoteriaBackend::new();
        let json = r#"{"status": "counterexample", "violations": [{"type": "buffer_overflow"}]}"#;
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

    /// Verify parse_output handles solver timeout
    #[kani::proof]
    fn proof_parse_output_solver_timeout() {
        let backend = SoteriaBackend::new();
        let (status, _) = backend.parse_output("", "solver timeout");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "solver timeout should produce Unknown status",
        );
    }
}

use crate::counterexample::{
    CounterexampleValue, FailedCheck, SourceLocation, StructuredCounterexample,
};
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

/// Memory model for Soteria symbolic execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SoteriaMemoryModel {
    /// Tree Borrows model (modern Rust aliasing)
    #[default]
    TreeBorrows,
    /// Stacked Borrows model (legacy aliasing)
    StackedBorrows,
    /// No aliasing checks (faster but less precise)
    NoAliasing,
}

/// Configuration for Soteria backend
#[derive(Debug, Clone)]
pub struct SoteriaConfig {
    /// Path to soteria binary
    pub soteria_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Target binary/bitcode to analyze
    pub target_binary: Option<PathBuf>,
    /// Memory model to use
    pub memory_model: SoteriaMemoryModel,
    /// Maximum exploration depth
    pub max_depth: usize,
    /// Solver timeout per query
    pub solver_timeout: Duration,
    /// Enable path merging optimization
    pub enable_path_merging: bool,
    /// Additional command-line arguments
    pub extra_args: Vec<String>,
}

impl Default for SoteriaConfig {
    fn default() -> Self {
        Self {
            soteria_path: None,
            timeout: Duration::from_secs(300),
            target_binary: None,
            memory_model: SoteriaMemoryModel::default(),
            max_depth: 100,
            solver_timeout: Duration::from_secs(30),
            enable_path_merging: false,
            extra_args: vec![],
        }
    }
}

/// Soteria Rust symbolic execution backend
pub struct SoteriaBackend {
    config: SoteriaConfig,
}

impl Default for SoteriaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SoteriaBackend {
    pub fn new() -> Self {
        Self {
            config: SoteriaConfig::default(),
        }
    }

    pub fn with_config(config: SoteriaConfig) -> Self {
        Self { config }
    }

    async fn detect_soteria(&self) -> Result<PathBuf, String> {
        let soteria_path = self
            .config
            .soteria_path
            .clone()
            .or_else(|| which::which("soteria").ok())
            .ok_or("Soteria not found. See: https://arxiv.org/abs/2511.08729")?;

        // Check if Soteria is working
        let output = Command::new(&soteria_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to check Soteria: {}", e))?;

        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            debug!("Detected Soteria version: {}", version.trim());
            Ok(soteria_path)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!(
                "Soteria not functioning properly. Error: {}",
                stderr.trim()
            ))
        }
    }

    /// Build Soteria analysis command arguments
    fn build_analysis_args(&self, binary_path: &std::path::Path) -> Vec<String> {
        let mut args = vec![
            "verify".to_string(),
            binary_path.to_string_lossy().to_string(),
        ];

        // Memory model
        match self.config.memory_model {
            SoteriaMemoryModel::TreeBorrows => {
                args.push("--memory-model".to_string());
                args.push("tree-borrows".to_string());
            }
            SoteriaMemoryModel::StackedBorrows => {
                args.push("--memory-model".to_string());
                args.push("stacked-borrows".to_string());
            }
            SoteriaMemoryModel::NoAliasing => {
                args.push("--memory-model".to_string());
                args.push("none".to_string());
            }
        }

        // Max depth
        args.push("--max-depth".to_string());
        args.push(self.config.max_depth.to_string());

        // Solver timeout
        args.push("--solver-timeout".to_string());
        args.push(self.config.solver_timeout.as_secs().to_string());

        // Path merging
        if self.config.enable_path_merging {
            args.push("--path-merging".to_string());
        }

        // Output format
        args.push("--output-format".to_string());
        args.push("json".to_string());

        // Extra args
        for arg in &self.config.extra_args {
            args.push(arg.clone());
        }

        args
    }

    /// Parse Soteria output JSON to verification result
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
                "safe" | "verified" | "ok" => {
                    return (VerificationStatus::Proven, None);
                }
                "counterexample" | "violation" | "unsafe" | "bug" => {
                    let ce = self.parse_counterexample(&json);
                    return (VerificationStatus::Disproven, Some(ce));
                }
                "timeout" => {
                    return (
                        VerificationStatus::Unknown {
                            reason: "Soteria exploration timed out".to_string(),
                        },
                        None,
                    );
                }
                "error" => {
                    let reason = json
                        .get("message")
                        .or_else(|| json.get("error"))
                        .and_then(|d| d.as_str())
                        .unwrap_or("Unknown error")
                        .to_string();
                    return (VerificationStatus::Unknown { reason }, None);
                }
                _ => {}
            }

            // Check for violations array
            if let Some(violations) = json.get("violations").and_then(|v| v.as_array()) {
                if !violations.is_empty() {
                    let ce = self.parse_counterexample(&json);
                    return (VerificationStatus::Disproven, Some(ce));
                }
            }

            // Check for paths_explored with no violations
            if json.get("paths_explored").is_some() && json.get("violations").is_none() {
                return (VerificationStatus::Proven, None);
            }
        }

        // Check for Soteria-specific errors in stderr
        if stderr.contains("solver timeout") || stderr.contains("SMT timeout") {
            return (
                VerificationStatus::Unknown {
                    reason: "SMT solver timeout. Try reducing search depth or increasing solver timeout.".to_string(),
                },
                None,
            );
        }

        if stderr.contains("Unsupported MIR construct") {
            return (
                VerificationStatus::Unknown {
                    reason: "Soteria encountered unsupported MIR construct. Try compiling with older rustc.".to_string(),
                },
                None,
            );
        }

        if stderr.contains("memory exhausted") || stderr.contains("out of memory") {
            return (
                VerificationStatus::Unknown {
                    reason:
                        "Soteria ran out of memory. Try enabling path merging or reducing depth."
                            .to_string(),
                },
                None,
            );
        }

        if stderr.contains("Error") || stderr.contains("error") || stderr.contains("failed") {
            return (
                VerificationStatus::Unknown {
                    reason: format!(
                        "Soteria error: {}",
                        stderr.lines().take(3).collect::<Vec<_>>().join("; ")
                    ),
                },
                None,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse Soteria output".to_string(),
            },
            None,
        )
    }

    /// Parse JSON counterexample into structured format
    fn parse_counterexample(&self, json: &serde_json::Value) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();

        // Store raw JSON
        ce.raw = Some(json.to_string());

        // Parse violations array
        if let Some(violations) = json.get("violations").and_then(|v| v.as_array()) {
            for violation in violations {
                if let Some(obj) = violation.as_object() {
                    let vtype = obj
                        .get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("unknown");
                    let desc = obj
                        .get("description")
                        .or_else(|| obj.get("message"))
                        .and_then(|d| d.as_str())
                        .unwrap_or(vtype);

                    let location = obj.get("location").map(|loc| {
                        let file = loc
                            .get("file")
                            .and_then(|f| f.as_str())
                            .unwrap_or("unknown");
                        let line = loc.get("line").and_then(|l| l.as_u64()).unwrap_or(0) as u32;
                        let column = loc.get("column").and_then(|c| c.as_u64()).map(|c| c as u32);
                        SourceLocation {
                            file: file.to_string(),
                            line,
                            column,
                        }
                    });

                    ce.failed_checks.push(FailedCheck {
                        check_id: format!("soteria_{}", ce.failed_checks.len()),
                        description: desc.to_string(),
                        location,
                        function: obj
                            .get("function")
                            .and_then(|f| f.as_str())
                            .map(String::from),
                    });

                    // Add witness values
                    if let Some(inputs) = obj.get("inputs").and_then(|i| i.as_object()) {
                        for (var, val) in inputs {
                            ce.witness
                                .insert(var.clone(), self.json_to_counterexample_value(val));
                        }
                    }
                }
            }
        }

        // Parse execution trace
        if let Some(trace) = json.get("trace").and_then(|t| t.as_array()) {
            let trace_str: Vec<String> = trace
                .iter()
                .filter_map(|step| step.get("pc").and_then(|pc| pc.as_str()).map(String::from))
                .collect();
            if !trace_str.is_empty() {
                ce.witness.insert(
                    "_trace".to_string(),
                    CounterexampleValue::String(trace_str.join(" -> ")),
                );
            }
        }

        // Parse path condition
        if let Some(pc) = json.get("path_condition").and_then(|p| p.as_str()) {
            ce.witness.insert(
                "_path_condition".to_string(),
                CounterexampleValue::String(pc.to_string()),
            );
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
impl VerificationBackend for SoteriaBackend {
    fn id(&self) -> BackendId {
        BackendId::Soteria
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::MemorySafety,
            PropertyType::Contract,
            PropertyType::UndefinedBehavior,
        ]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let soteria_path = self
            .detect_soteria()
            .await
            .map_err(BackendError::Unavailable)?;

        // Soteria requires a target binary/bitcode
        let target_binary = self.config.target_binary.as_ref().ok_or_else(|| {
            BackendError::VerificationFailed(
                "Soteria requires a target binary or LLVM bitcode path".to_string(),
            )
        })?;

        if !target_binary.exists() {
            return Err(BackendError::VerificationFailed(format!(
                "Target binary not found: {}",
                target_binary.display()
            )));
        }

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Build analysis command
        let args = self.build_analysis_args(target_binary);
        let output_file = temp_dir.path().join("output.json");

        // Build command
        let mut cmd = Command::new(&soteria_path);
        cmd.args(&args)
            .arg("--output")
            .arg(&output_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Execute with timeout
        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(10), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = if output_file.exists() {
                    std::fs::read_to_string(&output_file).unwrap_or_default()
                } else {
                    String::from_utf8_lossy(&output.stdout).to_string()
                };
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Soteria stdout: {}", stdout);
                debug!("Soteria stderr: {}", stderr);

                let (status, counterexample) = self.parse_output(&stdout, &stderr);

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| {
                        l.contains("Warning")
                            || l.contains("Error")
                            || l.contains("error")
                            || l.contains("warning")
                    })
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Verified by Soteria symbolic execution".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::Soteria,
                    status,
                    proof,
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute Soteria: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_soteria().await {
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
        let config = SoteriaConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(config.soteria_path.is_none());
        assert!(config.target_binary.is_none());
        assert_eq!(config.memory_model, SoteriaMemoryModel::TreeBorrows);
        assert_eq!(config.max_depth, 100);
        assert!(!config.enable_path_merging);
    }

    #[test]
    fn custom_config() {
        let config = SoteriaConfig {
            soteria_path: Some(PathBuf::from("/usr/bin/soteria")),
            timeout: Duration::from_secs(600),
            target_binary: Some(PathBuf::from("/path/to/binary.bc")),
            memory_model: SoteriaMemoryModel::StackedBorrows,
            max_depth: 200,
            solver_timeout: Duration::from_secs(60),
            enable_path_merging: true,
            extra_args: vec!["--verbose".to_string()],
        };
        assert_eq!(config.timeout, Duration::from_secs(600));
        assert_eq!(config.memory_model, SoteriaMemoryModel::StackedBorrows);
        assert_eq!(config.max_depth, 200);
    }

    #[test]
    fn backend_id() {
        let backend = SoteriaBackend::new();
        assert_eq!(backend.id(), BackendId::Soteria);
    }

    #[test]
    fn backend_supports() {
        let backend = SoteriaBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::MemorySafety));
        assert!(supported.contains(&PropertyType::Contract));
        assert!(supported.contains(&PropertyType::UndefinedBehavior));
    }

    // =============================================
    // Output parsing tests
    // =============================================

    #[test]
    fn parse_output_safe() {
        let backend = SoteriaBackend::new();
        let json = r#"{"status": "safe", "paths_explored": 42}"#;
        let (status, ce) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(ce.is_none());
    }

    #[test]
    fn parse_output_verified() {
        let backend = SoteriaBackend::new();
        let json = r#"{"status": "verified"}"#;
        let (status, _) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_output_counterexample() {
        let backend = SoteriaBackend::new();
        let json = r#"{"status": "counterexample", "violations": [{"type": "buffer_overflow", "description": "out of bounds access"}]}"#;
        let (status, ce) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(ce.is_some());
        let counterexample = ce.unwrap();
        assert_eq!(counterexample.failed_checks.len(), 1);
    }

    #[test]
    fn parse_output_violations_without_status() {
        let backend = SoteriaBackend::new();
        let json = r#"{"violations": [{"type": "use_after_free"}]}"#;
        let (status, ce) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(ce.is_some());
    }

    #[test]
    fn parse_output_timeout() {
        let backend = SoteriaBackend::new();
        let json = r#"{"status": "timeout"}"#;
        let (status, _) = backend.parse_output(json, "");
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.contains("timed out"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    #[test]
    fn parse_output_error() {
        let backend = SoteriaBackend::new();
        let json = r#"{"status": "error", "message": "Analysis failed"}"#;
        let (status, _) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_output_invalid_json() {
        let backend = SoteriaBackend::new();
        let (status, _) = backend.parse_output("not json", "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_output_solver_timeout() {
        let backend = SoteriaBackend::new();
        let (status, _) = backend.parse_output("", "solver timeout");
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.contains("SMT solver timeout"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    #[test]
    fn parse_output_unsupported_mir() {
        let backend = SoteriaBackend::new();
        let (status, _) = backend.parse_output("", "Unsupported MIR construct");
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.contains("unsupported MIR"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    #[test]
    fn parse_output_oom() {
        let backend = SoteriaBackend::new();
        let (status, _) = backend.parse_output("", "memory exhausted");
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.contains("out of memory"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    // =============================================
    // Counterexample parsing tests
    // =============================================

    #[test]
    fn parse_counterexample_with_location() {
        let backend = SoteriaBackend::new();
        let json: serde_json::Value = serde_json::json!({
            "status": "counterexample",
            "violations": [{
                "type": "buffer_overflow",
                "description": "out of bounds",
                "location": {
                    "file": "src/main.rs",
                    "line": 42,
                    "column": 10
                },
                "function": "process_data",
                "inputs": {
                    "index": 100,
                    "len": 50
                }
            }]
        });
        let ce = backend.parse_counterexample(&json);

        assert_eq!(ce.failed_checks.len(), 1);
        assert_eq!(
            ce.failed_checks[0].function,
            Some("process_data".to_string())
        );
        assert!(ce.witness.contains_key("index"));
        assert!(ce.witness.contains_key("len"));
    }

    #[test]
    fn parse_counterexample_with_trace() {
        let backend = SoteriaBackend::new();
        let json: serde_json::Value = serde_json::json!({
            "status": "counterexample",
            "trace": [
                {"pc": "0x1000"},
                {"pc": "0x1010"},
                {"pc": "0x1020"}
            ],
            "violations": []
        });
        let ce = backend.parse_counterexample(&json);

        assert!(ce.witness.contains_key("_trace"));
        match &ce.witness["_trace"] {
            CounterexampleValue::String(s) => {
                assert!(s.contains("0x1000"));
                assert!(s.contains("0x1020"));
            }
            _ => panic!("Expected String"),
        }
    }

    #[test]
    fn parse_counterexample_with_path_condition() {
        let backend = SoteriaBackend::new();
        let json: serde_json::Value = serde_json::json!({
            "status": "counterexample",
            "path_condition": "x > 0 && y < 100",
            "violations": []
        });
        let ce = backend.parse_counterexample(&json);

        assert!(ce.witness.contains_key("_path_condition"));
    }

    #[test]
    fn counterexample_has_raw() {
        let backend = SoteriaBackend::new();
        let json: serde_json::Value = serde_json::json!({"status": "counterexample"});
        let ce = backend.parse_counterexample(&json);
        assert!(ce.raw.is_some());
    }

    // =============================================
    // JSON value conversion tests
    // =============================================

    #[test]
    fn json_to_value_bool() {
        let backend = SoteriaBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!(true));
        assert_eq!(val, CounterexampleValue::Bool(true));
    }

    #[test]
    fn json_to_value_int() {
        let backend = SoteriaBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!(42));
        match val {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn json_to_value_string() {
        let backend = SoteriaBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!("hello"));
        assert_eq!(val, CounterexampleValue::String("hello".to_string()));
    }

    #[test]
    fn json_to_value_null() {
        let backend = SoteriaBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::Value::Null);
        match val {
            CounterexampleValue::Unknown(s) => assert_eq!(s, "null"),
            _ => panic!("Expected Unknown"),
        }
    }

    // =============================================
    // Analysis args tests
    // =============================================

    #[test]
    fn build_args_default() {
        let backend = SoteriaBackend::new();
        let args = backend.build_analysis_args(std::path::Path::new("/path/to/binary.bc"));
        assert!(args.contains(&"verify".to_string()));
        assert!(args.contains(&"--memory-model".to_string()));
        assert!(args.contains(&"tree-borrows".to_string()));
        assert!(args.contains(&"--max-depth".to_string()));
        assert!(args.contains(&"100".to_string()));
    }

    #[test]
    fn build_args_stacked_borrows() {
        let config = SoteriaConfig {
            memory_model: SoteriaMemoryModel::StackedBorrows,
            ..Default::default()
        };
        let backend = SoteriaBackend::with_config(config);
        let args = backend.build_analysis_args(std::path::Path::new("/path/to/binary.bc"));
        assert!(args.contains(&"stacked-borrows".to_string()));
    }

    #[test]
    fn build_args_no_aliasing() {
        let config = SoteriaConfig {
            memory_model: SoteriaMemoryModel::NoAliasing,
            ..Default::default()
        };
        let backend = SoteriaBackend::with_config(config);
        let args = backend.build_analysis_args(std::path::Path::new("/path/to/binary.bc"));
        assert!(args.contains(&"none".to_string()));
    }

    #[test]
    fn build_args_with_path_merging() {
        let config = SoteriaConfig {
            enable_path_merging: true,
            ..Default::default()
        };
        let backend = SoteriaBackend::with_config(config);
        let args = backend.build_analysis_args(std::path::Path::new("/path/to/binary.bc"));
        assert!(args.contains(&"--path-merging".to_string()));
    }

    #[test]
    fn build_args_with_custom_depth() {
        let config = SoteriaConfig {
            max_depth: 500,
            ..Default::default()
        };
        let backend = SoteriaBackend::with_config(config);
        let args = backend.build_analysis_args(std::path::Path::new("/path/to/binary.bc"));
        assert!(args.contains(&"500".to_string()));
    }

    #[test]
    fn build_args_with_extra_args() {
        let config = SoteriaConfig {
            extra_args: vec!["--verbose".to_string(), "--debug".to_string()],
            ..Default::default()
        };
        let backend = SoteriaBackend::with_config(config);
        let args = backend.build_analysis_args(std::path::Path::new("/path/to/binary.bc"));
        assert!(args.contains(&"--verbose".to_string()));
        assert!(args.contains(&"--debug".to_string()));
    }

    // =============================================
    // Edge cases
    // =============================================

    #[test]
    fn backend_with_custom_config() {
        let config = SoteriaConfig {
            timeout: Duration::from_secs(600),
            memory_model: SoteriaMemoryModel::StackedBorrows,
            ..Default::default()
        };
        let backend = SoteriaBackend::with_config(config);
        assert_eq!(backend.config.timeout, Duration::from_secs(600));
        assert_eq!(
            backend.config.memory_model,
            SoteriaMemoryModel::StackedBorrows
        );
    }

    #[test]
    fn default_equals_new() {
        let default_backend = SoteriaBackend::default();
        let new_backend = SoteriaBackend::new();
        assert_eq!(default_backend.config.timeout, new_backend.config.timeout);
        assert_eq!(
            default_backend.config.memory_model,
            new_backend.config.memory_model
        );
    }

    #[test]
    fn parse_output_paths_explored_no_violations() {
        let backend = SoteriaBackend::new();
        let json = r#"{"paths_explored": 100}"#;
        let (status, _) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_output_empty_json() {
        let backend = SoteriaBackend::new();
        let (status, _) = backend.parse_output("{}", "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn memory_models() {
        assert_eq!(
            SoteriaMemoryModel::default(),
            SoteriaMemoryModel::TreeBorrows
        );
        assert_ne!(
            SoteriaMemoryModel::StackedBorrows,
            SoteriaMemoryModel::TreeBorrows
        );
        assert_ne!(
            SoteriaMemoryModel::NoAliasing,
            SoteriaMemoryModel::StackedBorrows
        );
    }
}
