//! Ghidra (NSA) reverse engineering backend
//!
//! Ghidra is an open-source reverse engineering suite providing decompilation,
//! p-code IR analysis, and headless automation for binary analysis.
//!
//! Key features:
//! - Multi-architecture support (x86, ARM, MIPS, RISC-V, etc.)
//! - High-quality decompilation to C-like pseudocode
//! - P-code intermediate representation for semantic analysis
//! - Headless batch processing for CI/CD integration
//! - Java/Python scripting for custom analysis
//!
//! Input: Binary executables (ELF, Mach-O, PE, firmware)
//! Output: JSON with analysis results, p-code, decompiled code
//!
//! See: <https://ghidra-sre.org/>

// =============================================
// Kani Proofs for Ghidra Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- GhidraConfig Default Tests ----

    /// Verify GhidraConfig::default sets correct baseline values
    #[kani::proof]
    fn proof_ghidra_config_defaults() {
        let config = GhidraConfig::default();
        kani::assert(
            config.ghidra_path.is_none(),
            "ghidra_path should default to None",
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
            config.processor.is_none(),
            "processor should default to None (auto-detect)",
        );
        kani::assert(
            config.analysis_mode == GhidraAnalysisMode::Decompile,
            "analysis_mode should default to Decompile",
        );
    }

    // ---- GhidraBackend Construction Tests ----

    /// Verify GhidraBackend::new uses default configuration
    #[kani::proof]
    fn proof_ghidra_backend_new_defaults() {
        let backend = GhidraBackend::new();
        kani::assert(
            backend.config.ghidra_path.is_none(),
            "new backend should have no ghidra_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "new backend should default timeout to 300 seconds",
        );
    }

    /// Verify GhidraBackend::default equals GhidraBackend::new
    #[kani::proof]
    fn proof_ghidra_backend_default_equals_new() {
        let default_backend = GhidraBackend::default();
        let new_backend = GhidraBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.analysis_mode == new_backend.config.analysis_mode,
            "default and new should share analysis_mode",
        );
    }

    /// Verify GhidraBackend::with_config preserves custom configuration
    #[kani::proof]
    fn proof_ghidra_backend_with_config() {
        let config = GhidraConfig {
            ghidra_path: Some(PathBuf::from("/opt/ghidra")),
            timeout: Duration::from_secs(600),
            target_binary: Some(PathBuf::from("/bin/test")),
            processor: Some("x86:LE:64:default".to_string()),
            analysis_mode: GhidraAnalysisMode::Pcode,
            scripts: vec!["FindCrypto.java".to_string()],
            jvm_max_heap: Some("4G".to_string()),
        };
        let backend = GhidraBackend::with_config(config);
        kani::assert(
            backend.config.ghidra_path == Some(PathBuf::from("/opt/ghidra")),
            "with_config should preserve ghidra_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve timeout",
        );
        kani::assert(
            backend.config.analysis_mode == GhidraAnalysisMode::Pcode,
            "with_config should preserve analysis_mode",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::Ghidra
    #[kani::proof]
    fn proof_ghidra_backend_id() {
        let backend = GhidraBackend::new();
        kani::assert(
            backend.id() == BackendId::Ghidra,
            "GhidraBackend id should be BackendId::Ghidra",
        );
    }

    /// Verify supports() includes MemorySafety and Invariant
    #[kani::proof]
    fn proof_ghidra_backend_supports() {
        let backend = GhidraBackend::new();
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
        let backend = GhidraBackend::new();
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
        let backend = GhidraBackend::new();
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
        let backend = GhidraBackend::new();
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

/// Analysis mode for Ghidra
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GhidraAnalysisMode {
    /// Decompile to C-like pseudocode
    #[default]
    Decompile,
    /// Extract p-code intermediate representation
    Pcode,
    /// Generate call graph
    CallGraph,
    /// Vulnerability scanning
    VulnScan,
    /// Data flow analysis
    DataFlow,
}

/// Configuration for Ghidra backend
#[derive(Debug, Clone)]
pub struct GhidraConfig {
    /// Path to Ghidra installation directory
    pub ghidra_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Target binary to analyze
    pub target_binary: Option<PathBuf>,
    /// Processor specification (e.g., "x86:LE:64:default")
    pub processor: Option<String>,
    /// Analysis mode
    pub analysis_mode: GhidraAnalysisMode,
    /// Additional Ghidra scripts to run
    pub scripts: Vec<String>,
    /// JVM max heap size (e.g., "4G")
    pub jvm_max_heap: Option<String>,
}

impl Default for GhidraConfig {
    fn default() -> Self {
        Self {
            ghidra_path: None,
            timeout: Duration::from_secs(300),
            target_binary: None,
            processor: None,
            analysis_mode: GhidraAnalysisMode::default(),
            scripts: vec![],
            jvm_max_heap: None,
        }
    }
}

/// Ghidra reverse engineering backend
pub struct GhidraBackend {
    config: GhidraConfig,
}

impl Default for GhidraBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GhidraBackend {
    pub fn new() -> Self {
        Self {
            config: GhidraConfig::default(),
        }
    }

    pub fn with_config(config: GhidraConfig) -> Self {
        Self { config }
    }

    /// Detect Ghidra installation and analyzeHeadless script
    async fn detect_ghidra(&self) -> Result<PathBuf, String> {
        // Check explicit path first
        if let Some(ref ghidra_path) = self.config.ghidra_path {
            let headless = ghidra_path.join("support").join("analyzeHeadless");
            if headless.exists() {
                return Ok(headless);
            }
            // Try common variations
            let headless_sh = ghidra_path.join("support").join("analyzeHeadless.sh");
            if headless_sh.exists() {
                return Ok(headless_sh);
            }
            return Err(format!(
                "Ghidra installation not found at: {}",
                ghidra_path.display()
            ));
        }

        // Try common installation locations
        let common_paths = [
            "/opt/ghidra/support/analyzeHeadless",
            "/usr/local/ghidra/support/analyzeHeadless",
            "/Applications/ghidra/support/analyzeHeadless",
        ];

        for path in common_paths {
            let p = PathBuf::from(path);
            if p.exists() {
                return Ok(p);
            }
        }

        // Try GHIDRA_INSTALL_DIR env var
        if let Ok(ghidra_home) = std::env::var("GHIDRA_INSTALL_DIR") {
            let headless = PathBuf::from(&ghidra_home)
                .join("support")
                .join("analyzeHeadless");
            if headless.exists() {
                return Ok(headless);
            }
        }

        // Try which
        if let Ok(path) = which::which("analyzeHeadless") {
            return Ok(path);
        }

        Err(
            "Ghidra not found. Install from https://ghidra-sre.org/ or set GHIDRA_INSTALL_DIR"
                .to_string(),
        )
    }

    /// Build command line arguments for headless analysis
    fn build_analysis_args(
        &self,
        project_dir: &std::path::Path,
        binary_path: &std::path::Path,
        output_file: &std::path::Path,
    ) -> Vec<String> {
        let mut args = vec![
            project_dir.to_string_lossy().to_string(),
            "DashProveProject".to_string(),
            "-import".to_string(),
            binary_path.to_string_lossy().to_string(),
            "-overwrite".to_string(),
            "-analysisTimeoutPerFile".to_string(),
            self.config.timeout.as_secs().to_string(),
        ];

        // Add processor if specified
        if let Some(ref processor) = self.config.processor {
            args.push("-processor".to_string());
            args.push(processor.clone());
        }

        // Add mode-specific options
        match self.config.analysis_mode {
            GhidraAnalysisMode::Decompile => {
                args.push("-postScript".to_string());
                args.push("DecompileAllFunctions.java".to_string());
            }
            GhidraAnalysisMode::Pcode => {
                args.push("-postScript".to_string());
                args.push("ExportPcode.java".to_string());
            }
            GhidraAnalysisMode::CallGraph => {
                args.push("-postScript".to_string());
                args.push("ExportCallGraph.java".to_string());
            }
            GhidraAnalysisMode::VulnScan => {
                args.push("-postScript".to_string());
                args.push("FindVulnerabilities.java".to_string());
            }
            GhidraAnalysisMode::DataFlow => {
                args.push("-postScript".to_string());
                args.push("DataFlowAnalysis.java".to_string());
            }
        }

        // Add custom scripts
        for script in &self.config.scripts {
            args.push("-postScript".to_string());
            args.push(script.clone());
        }

        // Add JSON output script
        args.push("-postScript".to_string());
        args.push("ExportToJSON.java".to_string());
        args.push(output_file.to_string_lossy().to_string());

        args
    }

    /// Parse Ghidra output JSON to verification result
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
                "verified" | "safe" | "ok" | "clean" => {
                    return (VerificationStatus::Proven, None);
                }
                "counterexample" | "violation" | "unsafe" | "vulnerable" => {
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

            // Check for violations/vulnerabilities array
            if let Some(violations) = json
                .get("violations")
                .or_else(|| json.get("vulnerabilities"))
                .and_then(|v| v.as_array())
            {
                if !violations.is_empty() {
                    let ce = self.parse_counterexample(&json);
                    return (VerificationStatus::Disproven, Some(ce));
                }
            }

            // Check for checks_passed without violations
            if json.get("checks_passed").is_some()
                && json.get("violations").is_none()
                && json.get("vulnerabilities").is_none()
            {
                return (VerificationStatus::Proven, None);
            }

            // Check for analysis complete without issues
            if json.get("functions_analyzed").is_some() && json.get("issues").is_none() {
                return (VerificationStatus::Proven, None);
            }
        }

        // Check for Ghidra errors in stderr
        if stderr.contains("ERROR") || stderr.contains("Exception") || stderr.contains("SEVERE") {
            return (
                VerificationStatus::Unknown {
                    reason: format!(
                        "Ghidra analysis error: {}",
                        stderr.lines().take(3).collect::<Vec<_>>().join("; ")
                    ),
                },
                None,
            );
        }

        // Check for OOM error
        if stderr.contains("OutOfMemoryError") {
            return (
                VerificationStatus::Unknown {
                    reason:
                        "Ghidra ran out of memory. Try increasing JVM heap with jvm_max_heap config."
                            .to_string(),
                },
                None,
            );
        }

        // Check for processor error
        if stderr.contains("Unsupported processor") || stderr.contains("Unknown processor") {
            return (
                VerificationStatus::Unknown {
                    reason: "Ghidra could not detect processor. Specify processor explicitly."
                        .to_string(),
                },
                None,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse Ghidra output".to_string(),
            },
            None,
        )
    }

    /// Parse JSON counterexample into structured format
    fn parse_counterexample(&self, json: &serde_json::Value) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();

        // Store raw JSON
        ce.raw = Some(json.to_string());

        // Parse violations/vulnerabilities array
        let violations = json
            .get("violations")
            .or_else(|| json.get("vulnerabilities"))
            .and_then(|v| v.as_array());

        if let Some(violations) = violations {
            for violation in violations {
                if let Some(obj) = violation.as_object() {
                    let address = obj
                        .get("address")
                        .and_then(|a| a.as_str())
                        .unwrap_or("unknown");
                    let desc = obj
                        .get("description")
                        .or_else(|| obj.get("type"))
                        .or_else(|| obj.get("category"))
                        .and_then(|d| d.as_str())
                        .unwrap_or("Vulnerability found");

                    ce.failed_checks.push(FailedCheck {
                        check_id: format!("ghidra_{}", ce.failed_checks.len()),
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

                    // Add p-code or decompiled info if present
                    if let Some(pcode) = obj.get("pcode").and_then(|p| p.as_str()) {
                        ce.witness.insert(
                            format!("pcode_{}", ce.failed_checks.len() - 1),
                            CounterexampleValue::String(pcode.to_string()),
                        );
                    }

                    if let Some(decompiled) = obj.get("decompiled").and_then(|d| d.as_str()) {
                        ce.witness.insert(
                            format!("decompiled_{}", ce.failed_checks.len() - 1),
                            CounterexampleValue::String(decompiled.to_string()),
                        );
                    }
                }
            }
        }

        // Parse call trace if present
        if let Some(trace) = json.get("call_trace").and_then(|t| t.as_array()) {
            let trace_str: Vec<String> = trace
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            if !trace_str.is_empty() {
                ce.witness.insert(
                    "_call_trace".to_string(),
                    CounterexampleValue::String(trace_str.join(" -> ")),
                );
            }
        }

        // Parse data flow info if present
        if let Some(dataflow) = json.get("data_flow").and_then(|d| d.as_object()) {
            for (key, value) in dataflow {
                ce.witness.insert(
                    format!("df_{}", key),
                    self.json_to_counterexample_value(value),
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
impl VerificationBackend for GhidraBackend {
    fn id(&self) -> BackendId {
        BackendId::Ghidra
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

        let ghidra_path = self
            .detect_ghidra()
            .await
            .map_err(BackendError::Unavailable)?;

        // Ghidra requires a target binary
        let target_binary = self.config.target_binary.as_ref().ok_or_else(|| {
            BackendError::VerificationFailed("Ghidra requires a target binary path".to_string())
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

        let project_dir = temp_dir.path().join("project");
        std::fs::create_dir_all(&project_dir).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create project dir: {}", e))
        })?;

        let output_file = temp_dir.path().join("output.json");

        // Build analysis command
        let args = self.build_analysis_args(&project_dir, target_binary, &output_file);

        // Build command with optional JVM settings
        let mut cmd = Command::new(&ghidra_path);
        cmd.args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Set JVM max heap if specified
        if let Some(ref max_heap) = self.config.jvm_max_heap {
            cmd.env("_JAVA_OPTIONS", format!("-Xmx{}", max_heap));
        }

        debug!("Running Ghidra: {:?} {:?}", ghidra_path, args);

        // Execute with timeout
        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(30), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = if output_file.exists() {
                    std::fs::read_to_string(&output_file).unwrap_or_default()
                } else {
                    String::from_utf8_lossy(&output.stdout).to_string()
                };
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Ghidra stdout: {}", stdout);
                debug!("Ghidra stderr: {}", stderr);

                let (status, counterexample) = self.parse_output(&stdout, &stderr);

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| {
                        l.contains("WARN")
                            || l.contains("ERROR")
                            || l.contains("Exception")
                            || l.contains("SEVERE")
                    })
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Verified by Ghidra binary analysis".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::Ghidra,
                    status,
                    proof,
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute Ghidra: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_ghidra().await {
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
        let config = GhidraConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(config.ghidra_path.is_none());
        assert!(config.target_binary.is_none());
        assert!(config.processor.is_none());
        assert_eq!(config.analysis_mode, GhidraAnalysisMode::Decompile);
    }

    #[test]
    fn custom_config() {
        let config = GhidraConfig {
            ghidra_path: Some(PathBuf::from("/opt/ghidra")),
            timeout: Duration::from_secs(600),
            target_binary: Some(PathBuf::from("/path/to/binary")),
            processor: Some("x86:LE:64:default".to_string()),
            analysis_mode: GhidraAnalysisMode::Pcode,
            scripts: vec!["FindCrypto.java".to_string()],
            jvm_max_heap: Some("4G".to_string()),
        };
        assert_eq!(config.timeout, Duration::from_secs(600));
        assert_eq!(config.analysis_mode, GhidraAnalysisMode::Pcode);
        assert_eq!(config.jvm_max_heap, Some("4G".to_string()));
    }

    #[test]
    fn backend_id() {
        let backend = GhidraBackend::new();
        assert_eq!(backend.id(), BackendId::Ghidra);
    }

    #[test]
    fn backend_supports() {
        let backend = GhidraBackend::new();
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
        let backend = GhidraBackend::new();
        let json = r#"{"status": "verified", "checks_passed": 5}"#;
        let (status, ce) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(ce.is_none());
    }

    #[test]
    fn parse_output_safe() {
        let backend = GhidraBackend::new();
        let json = r#"{"status": "safe", "message": "No vulnerabilities found"}"#;
        let (status, _) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_output_clean() {
        let backend = GhidraBackend::new();
        let json = r#"{"status": "clean", "functions_analyzed": 42}"#;
        let (status, _) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_output_counterexample() {
        let backend = GhidraBackend::new();
        let json = r#"{"status": "counterexample", "violations": [{"address": "0x1000", "type": "buffer overflow"}]}"#;
        let (status, ce) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(ce.is_some());
        let counterexample = ce.unwrap();
        assert_eq!(counterexample.failed_checks.len(), 1);
    }

    #[test]
    fn parse_output_vulnerable() {
        let backend = GhidraBackend::new();
        let json = r#"{"status": "vulnerable", "vulnerabilities": [{"address": "0x2000", "category": "use-after-free"}]}"#;
        let (status, ce) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(ce.is_some());
    }

    #[test]
    fn parse_output_violations_without_status() {
        let backend = GhidraBackend::new();
        let json = r#"{"violations": [{"address": "0x2000"}]}"#;
        let (status, ce) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(ce.is_some());
    }

    #[test]
    fn parse_output_error() {
        let backend = GhidraBackend::new();
        let json = r#"{"status": "error", "message": "Analysis failed"}"#;
        let (status, _) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_output_invalid_json() {
        let backend = GhidraBackend::new();
        let (status, _) = backend.parse_output("not json", "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_output_oom_error() {
        let backend = GhidraBackend::new();
        let (status, _) = backend.parse_output("", "java.lang.OutOfMemoryError: Java heap space");
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.contains("memory"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    #[test]
    fn parse_output_processor_error() {
        let backend = GhidraBackend::new();
        let (status, _) = backend.parse_output("", "Unsupported processor specification");
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.contains("processor"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    #[test]
    fn parse_output_functions_analyzed() {
        let backend = GhidraBackend::new();
        let json = r#"{"functions_analyzed": 150}"#;
        let (status, _) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Proven));
    }

    // =============================================
    // Counterexample parsing tests
    // =============================================

    #[test]
    fn parse_counterexample_with_violations() {
        let backend = GhidraBackend::new();
        let json: serde_json::Value = serde_json::json!({
            "status": "counterexample",
            "violations": [
                {
                    "address": "0x401000",
                    "type": "buffer overflow",
                    "function": "main",
                    "pcode": "COPY RAX, RBX",
                    "decompiled": "buf[i] = data;"
                }
            ]
        });
        let ce = backend.parse_counterexample(&json);

        assert_eq!(ce.failed_checks.len(), 1);
        assert!(ce.failed_checks[0].description.contains("buffer overflow"));
        assert!(ce.witness.contains_key("pcode_0"));
        assert!(ce.witness.contains_key("decompiled_0"));
    }

    #[test]
    fn parse_counterexample_with_call_trace() {
        let backend = GhidraBackend::new();
        let json: serde_json::Value = serde_json::json!({
            "status": "counterexample",
            "call_trace": ["main", "process_input", "vulnerable_func"],
            "violations": []
        });
        let ce = backend.parse_counterexample(&json);

        assert!(ce.witness.contains_key("_call_trace"));
        match &ce.witness["_call_trace"] {
            CounterexampleValue::String(s) => {
                assert!(s.contains("main"));
                assert!(s.contains("vulnerable_func"));
            }
            _ => panic!("Expected String"),
        }
    }

    #[test]
    fn parse_counterexample_with_dataflow() {
        let backend = GhidraBackend::new();
        let json: serde_json::Value = serde_json::json!({
            "status": "counterexample",
            "data_flow": {
                "tainted_source": "user_input",
                "sink": "system_call"
            },
            "violations": []
        });
        let ce = backend.parse_counterexample(&json);

        assert!(ce.witness.contains_key("df_tainted_source"));
        assert!(ce.witness.contains_key("df_sink"));
    }

    #[test]
    fn counterexample_has_raw() {
        let backend = GhidraBackend::new();
        let json: serde_json::Value = serde_json::json!({"status": "counterexample"});
        let ce = backend.parse_counterexample(&json);
        assert!(ce.raw.is_some());
    }

    // =============================================
    // JSON value conversion tests
    // =============================================

    #[test]
    fn json_to_value_bool() {
        let backend = GhidraBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!(true));
        assert_eq!(val, CounterexampleValue::Bool(true));
    }

    #[test]
    fn json_to_value_int() {
        let backend = GhidraBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!(42));
        match val {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn json_to_value_string() {
        let backend = GhidraBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!("hello"));
        assert_eq!(val, CounterexampleValue::String("hello".to_string()));
    }

    #[test]
    fn json_to_value_null() {
        let backend = GhidraBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::Value::Null);
        match val {
            CounterexampleValue::Unknown(s) => assert_eq!(s, "null"),
            _ => panic!("Expected Unknown"),
        }
    }

    #[test]
    fn json_to_value_array() {
        let backend = GhidraBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!([1, 2, 3]));
        match val {
            CounterexampleValue::Sequence(arr) => assert_eq!(arr.len(), 3),
            _ => panic!("Expected Sequence"),
        }
    }

    #[test]
    fn json_to_value_object() {
        let backend = GhidraBackend::new();
        let val = backend.json_to_counterexample_value(&serde_json::json!({"a": 1, "b": 2}));
        match val {
            CounterexampleValue::Record(map) => {
                assert_eq!(map.len(), 2);
                assert!(map.contains_key("a"));
            }
            _ => panic!("Expected Record"),
        }
    }

    // =============================================
    // Analysis args tests
    // =============================================

    #[test]
    fn build_args_decompile_mode() {
        let backend = GhidraBackend::new();
        let args = backend.build_analysis_args(
            std::path::Path::new("/tmp/project"),
            std::path::Path::new("/bin/ls"),
            std::path::Path::new("/tmp/output.json"),
        );
        assert!(args.contains(&"-import".to_string()));
        assert!(args.contains(&"-postScript".to_string()));
        assert!(args.contains(&"DecompileAllFunctions.java".to_string()));
    }

    #[test]
    fn build_args_pcode_mode() {
        let config = GhidraConfig {
            analysis_mode: GhidraAnalysisMode::Pcode,
            ..Default::default()
        };
        let backend = GhidraBackend::with_config(config);
        let args = backend.build_analysis_args(
            std::path::Path::new("/tmp/project"),
            std::path::Path::new("/bin/ls"),
            std::path::Path::new("/tmp/output.json"),
        );
        assert!(args.contains(&"ExportPcode.java".to_string()));
    }

    #[test]
    fn build_args_with_processor() {
        let config = GhidraConfig {
            processor: Some("ARM:LE:32:Cortex".to_string()),
            ..Default::default()
        };
        let backend = GhidraBackend::with_config(config);
        let args = backend.build_analysis_args(
            std::path::Path::new("/tmp/project"),
            std::path::Path::new("/bin/ls"),
            std::path::Path::new("/tmp/output.json"),
        );
        assert!(args.contains(&"-processor".to_string()));
        assert!(args.contains(&"ARM:LE:32:Cortex".to_string()));
    }

    #[test]
    fn build_args_with_custom_scripts() {
        let config = GhidraConfig {
            scripts: vec![
                "FindCrypto.java".to_string(),
                "CheckBounds.java".to_string(),
            ],
            ..Default::default()
        };
        let backend = GhidraBackend::with_config(config);
        let args = backend.build_analysis_args(
            std::path::Path::new("/tmp/project"),
            std::path::Path::new("/bin/ls"),
            std::path::Path::new("/tmp/output.json"),
        );
        assert!(args.contains(&"FindCrypto.java".to_string()));
        assert!(args.contains(&"CheckBounds.java".to_string()));
    }

    // =============================================
    // Edge cases
    // =============================================

    #[test]
    fn backend_with_custom_config() {
        let config = GhidraConfig {
            timeout: Duration::from_secs(600),
            analysis_mode: GhidraAnalysisMode::VulnScan,
            ..Default::default()
        };
        let backend = GhidraBackend::with_config(config);
        assert_eq!(backend.config.timeout, Duration::from_secs(600));
        assert_eq!(backend.config.analysis_mode, GhidraAnalysisMode::VulnScan);
    }

    #[test]
    fn parse_output_checks_passed_no_violations() {
        let backend = GhidraBackend::new();
        let json = r#"{"checks_passed": 10}"#;
        let (status, _) = backend.parse_output(json, "");
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_output_empty_json() {
        let backend = GhidraBackend::new();
        let (status, _) = backend.parse_output("{}", "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn all_analysis_modes() {
        // Verify all analysis modes are distinct
        let modes = [
            GhidraAnalysisMode::Decompile,
            GhidraAnalysisMode::Pcode,
            GhidraAnalysisMode::CallGraph,
            GhidraAnalysisMode::VulnScan,
            GhidraAnalysisMode::DataFlow,
        ];
        for (i, mode1) in modes.iter().enumerate() {
            for (j, mode2) in modes.iter().enumerate() {
                if i != j {
                    assert_ne!(mode1, mode2);
                }
            }
        }
    }
}
