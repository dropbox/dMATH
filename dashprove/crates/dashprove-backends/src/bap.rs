//! BAP (Binary Analysis Platform) backend
//!
//! BAP is an OCaml-based binary analysis toolkit that lifts machine code to
//! BIL (Binary Intermediate Language) for static, symbolic, and semantic analysis.
//!
//! Key features:
//! - Binary lifting to architecture-independent BIL IR
//! - CFG recovery and control flow analysis
//! - Dataflow and taint analysis
//! - Symbolic execution via Primus
//! - Multi-architecture support (x86, ARM, MIPS, etc.)
//!
//! Input: Binary executables (ELF, Mach-O, PE)
//! Output: JSON with analysis results, BIL IR, CFG data
//!
//! See: <https://binaryanalysisplatform.github.io/>

// =============================================
// Kani Proofs for BAP Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- BapConfig Default Tests ----

    /// Verify BapConfig::default sets correct baseline values
    #[kani::proof]
    fn proof_bap_config_defaults() {
        let config = BapConfig::default();
        kani::assert(config.bap_path.is_none(), "bap_path should default to None");
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "timeout should default to 300 seconds",
        );
        kani::assert(
            config.target_binary.is_none(),
            "target_binary should default to None",
        );
        kani::assert(
            config.arch.is_none(),
            "arch should default to None (auto-detect)",
        );
        kani::assert(
            config.analysis_mode == BapAnalysisMode::Lift,
            "analysis_mode should default to Lift",
        );
    }

    // ---- BapBackend Construction Tests ----

    /// Verify BapBackend::new uses default configuration
    #[kani::proof]
    fn proof_bap_backend_new_defaults() {
        let backend = BapBackend::new();
        kani::assert(
            backend.config.bap_path.is_none(),
            "new backend should have no bap_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "new backend should default timeout to 300 seconds",
        );
    }

    /// Verify BapBackend::default equals BapBackend::new
    #[kani::proof]
    fn proof_bap_backend_default_equals_new() {
        let default_backend = BapBackend::default();
        let new_backend = BapBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.analysis_mode == new_backend.config.analysis_mode,
            "default and new should share analysis_mode",
        );
    }

    /// Verify BapBackend::with_config preserves custom configuration
    #[kani::proof]
    fn proof_bap_backend_with_config() {
        let config = BapConfig {
            bap_path: Some(PathBuf::from("/usr/bin/bap")),
            timeout: Duration::from_secs(600),
            target_binary: Some(PathBuf::from("/bin/test")),
            arch: Some("x86_64".to_string()),
            analysis_mode: BapAnalysisMode::Primus,
            extra_passes: vec!["callgraph".to_string()],
        };
        let backend = BapBackend::with_config(config);
        kani::assert(
            backend.config.bap_path == Some(PathBuf::from("/usr/bin/bap")),
            "with_config should preserve bap_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve timeout",
        );
        kani::assert(
            backend.config.analysis_mode == BapAnalysisMode::Primus,
            "with_config should preserve analysis_mode",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::Bap
    #[kani::proof]
    fn proof_bap_backend_id() {
        let backend = BapBackend::new();
        kani::assert(
            backend.id() == BackendId::Bap,
            "BapBackend id should be BackendId::Bap",
        );
    }

    /// Verify supports() includes MemorySafety and Invariant
    #[kani::proof]
    fn proof_bap_backend_supports() {
        let backend = BapBackend::new();
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

    // ---- Output Parsing Tests ----

    /// Verify parse_output detects verified status
    #[kani::proof]
    fn proof_parse_output_verified() {
        let backend = BapBackend::new();
        let json = r#"{"status": "verified", "checks_passed": 5}"#;
        let (status, ce) = backend.parse_output(json, "");
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "verified should produce Proven status",
        );
        kani::assert(ce.is_none(), "verified should have no counterexample");
    }

    /// Verify parse_output detects counterexample status
    #[kani::proof]
    fn proof_parse_output_counterexample() {
        let backend = BapBackend::new();
        let json = r#"{"status": "counterexample", "violations": [{"address": "0x1000"}]}"#;
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

    /// Verify parse_output handles invalid JSON
    #[kani::proof]
    fn proof_parse_output_invalid_json() {
        let backend = BapBackend::new();
        let (status, _) = backend.parse_output("not json at all", "");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "invalid JSON should produce Unknown status",
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

/// Analysis mode for BAP
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BapAnalysisMode {
    /// Lift binary to BIL and analyze statically
    #[default]
    Lift,
    /// Use Primus for symbolic execution
    Primus,
    /// Taint analysis
    Taint,
    /// Control flow graph extraction
    Cfg,
}

/// Configuration for BAP backend
#[derive(Debug, Clone)]
pub struct BapConfig {
    /// Path to bap binary
    pub bap_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Target binary to analyze
    pub target_binary: Option<PathBuf>,
    /// Target architecture (auto-detect if None)
    pub arch: Option<String>,
    /// Analysis mode
    pub analysis_mode: BapAnalysisMode,
    /// Additional BAP passes to run
    pub extra_passes: Vec<String>,
}

impl Default for BapConfig {
    fn default() -> Self {
        Self {
            bap_path: None,
            timeout: Duration::from_secs(300),
            target_binary: None,
            arch: None,
            analysis_mode: BapAnalysisMode::default(),
            extra_passes: vec![],
        }
    }
}

/// BAP binary analysis platform backend
pub struct BapBackend {
    config: BapConfig,
}

impl Default for BapBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl BapBackend {
    pub fn new() -> Self {
        Self {
            config: BapConfig::default(),
        }
    }

    pub fn with_config(config: BapConfig) -> Self {
        Self { config }
    }

    async fn detect_bap(&self) -> Result<PathBuf, String> {
        let bap_path = self
            .config
            .bap_path
            .clone()
            .or_else(|| which::which("bap").ok())
            .ok_or("BAP not found. Install via: opam install bap")?;

        // Check if BAP is working
        let output = Command::new(&bap_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to check BAP: {}", e))?;

        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            debug!("Detected BAP version: {}", version.trim());
            Ok(bap_path)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!(
                "BAP not functioning properly. Error: {}",
                stderr.trim()
            ))
        }
    }

    /// Generate BAP analysis command based on mode
    fn build_analysis_args(&self, binary_path: &std::path::Path) -> Vec<String> {
        let mut args = vec![binary_path.to_string_lossy().to_string()];

        // Add architecture if specified
        if let Some(ref arch) = self.config.arch {
            args.push("--arch".to_string());
            args.push(arch.clone());
        }

        // Add mode-specific passes
        match self.config.analysis_mode {
            BapAnalysisMode::Lift => {
                args.push("--pass=dump".to_string());
                args.push("--dump=bil".to_string());
                args.push("--dump-format=json".to_string());
            }
            BapAnalysisMode::Primus => {
                args.push("--primus-run".to_string());
                args.push("--primus-limit=1000".to_string());
                args.push("--primus-output=json".to_string());
            }
            BapAnalysisMode::Taint => {
                args.push("--pass=taint".to_string());
                args.push("--taint-output=json".to_string());
            }
            BapAnalysisMode::Cfg => {
                args.push("--pass=callgraph".to_string());
                args.push("--dump=callgraph".to_string());
                args.push("--dump-format=json".to_string());
            }
        }

        // Add extra passes
        for pass in &self.config.extra_passes {
            args.push(format!("--pass={}", pass));
        }

        args
    }

    /// Parse BAP output JSON to verification result
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
                "verified" | "safe" | "ok" => {
                    return (VerificationStatus::Proven, None);
                }
                "counterexample" | "violation" | "unsafe" => {
                    let ce = self.parse_counterexample(&json);
                    return (VerificationStatus::Disproven, Some(ce));
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

            // Check for checks_passed without violations
            if json.get("checks_passed").is_some() && json.get("violations").is_none() {
                return (VerificationStatus::Proven, None);
            }
        }

        // Check for BAP errors in stderr
        if stderr.contains("Error") || stderr.contains("Fatal") || stderr.contains("failed") {
            return (
                VerificationStatus::Unknown {
                    reason: format!(
                        "BAP analysis error: {}",
                        stderr.lines().take(3).collect::<Vec<_>>().join("; ")
                    ),
                },
                None,
            );
        }

        // Check for architecture error
        if stderr.contains("Failed to identify architecture") {
            return (
                VerificationStatus::Unknown {
                    reason: "BAP could not detect target architecture. Specify --arch explicitly."
                        .to_string(),
                },
                None,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse BAP output".to_string(),
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
                    let address = obj
                        .get("address")
                        .and_then(|a| a.as_str())
                        .unwrap_or("unknown");
                    let desc = obj
                        .get("description")
                        .or_else(|| obj.get("type"))
                        .and_then(|d| d.as_str())
                        .unwrap_or("Violation found");

                    ce.failed_checks.push(FailedCheck {
                        check_id: format!("bap_{}", ce.failed_checks.len()),
                        description: desc.to_string(),
                        location: Some(SourceLocation {
                            file: address.to_string(),
                            line: 0,
                            column: None,
                        }),
                        function: obj
                            .get("function")
                            .and_then(|f| f.as_str())
                            .map(String::from),
                    });

                    // Add witness values
                    if let Some(regs) = obj.get("registers").and_then(|r| r.as_object()) {
                        for (reg, val) in regs {
                            ce.witness
                                .insert(reg.clone(), self.json_to_counterexample_value(val));
                        }
                    }
                }
            }
        }

        // Parse trace information
        if let Some(trace) = json.get("trace").and_then(|t| t.as_array()) {
            let trace_str: Vec<String> = trace
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            if !trace_str.is_empty() {
                ce.witness.insert(
                    "_trace".to_string(),
                    CounterexampleValue::String(trace_str.join(" -> ")),
                );
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
impl VerificationBackend for BapBackend {
    fn id(&self) -> BackendId {
        BackendId::Bap
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::MemorySafety,
            PropertyType::Invariant,
            PropertyType::UndefinedBehavior,
        ]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let bap_path = self.detect_bap().await.map_err(BackendError::Unavailable)?;

        // BAP requires a target binary
        let target_binary = self.config.target_binary.as_ref().ok_or_else(|| {
            BackendError::VerificationFailed("BAP requires a target binary path".to_string())
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
        let mut cmd = Command::new(&bap_path);
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

                debug!("BAP stdout: {}", stdout);
                debug!("BAP stderr: {}", stderr);

                let (status, counterexample) = self.parse_output(&stdout, &stderr);

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| {
                        l.contains("Warning")
                            || l.contains("Error")
                            || l.contains("error")
                            || l.contains("failed")
                    })
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Verified by BAP binary analysis".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::Bap,
                    status,
                    proof,
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute BAP: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_bap().await {
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
        let config = BapConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(config.bap_path.is_none());
        assert!(config.target_binary.is_none());
        assert!(config.arch.is_none());
        assert_eq!(config.analysis_mode, BapAnalysisMode::Lift);
    }

    #[test]
    fn custom_config() {
        let config = BapConfig {
            bap_path: Some(PathBuf::from("/usr/bin/bap")),
            timeout: Duration::from_secs(600),
            target_binary: Some(PathBuf::from("/path/to/binary")),
            arch: Some("x86_64".to_string()),
            analysis_mode: BapAnalysisMode::Primus,
            extra_passes: vec!["callgraph".to_string()],
        };
        assert_eq!(config.timeout, Duration::from_secs(600));
        assert_eq!(config.analysis_mode, BapAnalysisMode::Primus);
    }

    #[test]
    fn backend_id() {
        let backend = BapBackend::new();
        assert_eq!(backend.id(), BackendId::Bap);
    }

    #[test]
    fn backend_supports() {
        let backend = BapBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::MemorySafety));
        assert!(supported.contains(&PropertyType::Invariant));
        assert!(supported.contains(&PropertyType::UndefinedBehavior));
    }

    // =============================================
    // Output parsing tests
    // =============================================

    #[test]
    fn parse_output_verified() {
        let backend = BapBackend::new();
        let json = r#"{"status": "verified", "checks_passed": 5}"#;
        let (status, ce) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(ce.is_none());
    }

    #[test]
    fn parse_output_safe() {
        let backend = BapBackend::new();
        let json = r#"{"status": "safe", "message": "No violations found"}"#;
        let (status, _) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_output_counterexample() {
        let backend = BapBackend::new();
        let json = r#"{"status": "counterexample", "violations": [{"address": "0x1000", "type": "buffer overflow"}]}"#;
        let (status, ce) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(ce.is_some());
        let counterexample = ce.unwrap();
        assert_eq!(counterexample.failed_checks.len(), 1);
    }

    #[test]
    fn parse_output_violations_without_status() {
        let backend = BapBackend::new();
        let json = r#"{"violations": [{"address": "0x2000"}]}"#;
        let (status, ce) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(ce.is_some());
    }

    #[test]
    fn parse_output_error() {
        let backend = BapBackend::new();
        let json = r#"{"status": "error", "message": "Analysis failed"}"#;
        let (status, _) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_output_invalid_json() {
        let backend = BapBackend::new();
        let (status, _) = backend.parse_output("not json", "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_output_arch_error() {
        let backend = BapBackend::new();
        let (status, _) = backend.parse_output("", "Failed to identify architecture");
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.contains("architecture"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    // =============================================
    // Counterexample parsing tests
    // =============================================

    #[test]
    fn parse_counterexample_with_violations() {
        let backend = BapBackend::new();
        let json: serde_json::Value = serde_json::json!({
            "status": "counterexample",
            "violations": [
                {
                    "address": "0x401000",
                    "type": "buffer overflow",
                    "function": "main",
                    "registers": {
                        "rax": 0,
                        "rbx": 42
                    }
                }
            ]
        });
        let ce = backend.parse_counterexample(&json);

        assert_eq!(ce.failed_checks.len(), 1);
        assert!(ce.failed_checks[0].description.contains("buffer overflow"));
        assert!(ce.witness.contains_key("rax"));
        assert!(ce.witness.contains_key("rbx"));
    }

    #[test]
    fn parse_counterexample_with_trace() {
        let backend = BapBackend::new();
        let json: serde_json::Value = serde_json::json!({
            "status": "counterexample",
            "trace": ["0x1000", "0x1010", "0x1020"],
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
    fn counterexample_has_raw() {
        let backend = BapBackend::new();
        let json: serde_json::Value = serde_json::json!({"status": "counterexample"});
        let ce = backend.parse_counterexample(&json);
        assert!(ce.raw.is_some());
    }

    // =============================================
    // JSON value conversion tests
    // =============================================

    #[test]
    fn json_to_value_bool() {
        let backend = BapBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!(true));
        assert_eq!(val, CounterexampleValue::Bool(true));
    }

    #[test]
    fn json_to_value_int() {
        let backend = BapBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!(42));
        match val {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn json_to_value_string() {
        let backend = BapBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!("hello"));
        assert_eq!(val, CounterexampleValue::String("hello".to_string()));
    }

    #[test]
    fn json_to_value_null() {
        let backend = BapBackend::new();
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
    fn build_args_lift_mode() {
        let backend = BapBackend::new();
        let args = backend.build_analysis_args(std::path::Path::new("/bin/ls"));
        assert!(args.contains(&"--pass=dump".to_string()));
        assert!(args.contains(&"--dump=bil".to_string()));
    }

    #[test]
    fn build_args_primus_mode() {
        let config = BapConfig {
            analysis_mode: BapAnalysisMode::Primus,
            ..Default::default()
        };
        let backend = BapBackend::with_config(config);
        let args = backend.build_analysis_args(std::path::Path::new("/bin/ls"));
        assert!(args.contains(&"--primus-run".to_string()));
    }

    #[test]
    fn build_args_with_arch() {
        let config = BapConfig {
            arch: Some("arm".to_string()),
            ..Default::default()
        };
        let backend = BapBackend::with_config(config);
        let args = backend.build_analysis_args(std::path::Path::new("/bin/ls"));
        assert!(args.contains(&"--arch".to_string()));
        assert!(args.contains(&"arm".to_string()));
    }

    #[test]
    fn build_args_with_extra_passes() {
        let config = BapConfig {
            extra_passes: vec!["callgraph".to_string(), "cfg".to_string()],
            ..Default::default()
        };
        let backend = BapBackend::with_config(config);
        let args = backend.build_analysis_args(std::path::Path::new("/bin/ls"));
        assert!(args.contains(&"--pass=callgraph".to_string()));
        assert!(args.contains(&"--pass=cfg".to_string()));
    }

    // =============================================
    // Edge cases
    // =============================================

    #[test]
    fn backend_with_custom_config() {
        let config = BapConfig {
            timeout: Duration::from_secs(600),
            analysis_mode: BapAnalysisMode::Taint,
            ..Default::default()
        };
        let backend = BapBackend::with_config(config);
        assert_eq!(backend.config.timeout, Duration::from_secs(600));
        assert_eq!(backend.config.analysis_mode, BapAnalysisMode::Taint);
    }

    #[test]
    fn parse_output_checks_passed_no_violations() {
        let backend = BapBackend::new();
        let json = r#"{"checks_passed": 10}"#;
        let (status, _) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_output_empty_json() {
        let backend = BapBackend::new();
        let (status, _) = backend.parse_output("{}", "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }
}
