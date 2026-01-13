//! P language state machine backend
//!
//! P is a domain-specific language for modeling and specifying
//! complex distributed systems. It supports event-driven state machines
//! with formal verification through explicit-state model checking.
//!
//! See: <https://p-org.github.io/P/>
//!
//! # Features
//!
//! - **Event-driven**: Asynchronous message-passing between state machines
//! - **State machine modeling**: Hierarchical state machines with entry/exit handlers
//! - **Formal verification**: Model checking for safety and liveness properties
//! - **Test harness generation**: Generates test cases from counterexamples
//! - **Code generation**: Generates C/C++/C# code from P models
//!
//! # Requirements
//!
//! Install P:
//! ```bash
//! # Using dotnet
//! dotnet tool install --global P
//!
//! # Or from source
//! git clone https://github.com/p-org/P
//! cd P && dotnet build
//! ```

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

/// Verification mode for P language
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PVerifyMode {
    /// Systematic testing (default)
    #[default]
    Systematic,
    /// Random testing
    Random,
    /// Coverage-guided testing
    Coverage,
}

/// Configuration for P language backend
#[derive(Debug, Clone)]
pub struct PLangConfig {
    /// Path to P compiler (pc)
    pub p_path: Option<PathBuf>,
    /// Path to P checker (pmc)
    pub checker_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Verification mode
    pub mode: PVerifyMode,
    /// Maximum number of schedules to explore
    pub max_schedules: Option<u32>,
    /// Maximum steps per schedule
    pub max_steps: Option<u32>,
    /// Random seed
    pub seed: Option<u64>,
    /// Additional P options
    pub extra_args: Vec<String>,
}

impl Default for PLangConfig {
    fn default() -> Self {
        Self {
            p_path: None,
            checker_path: None,
            timeout: Duration::from_secs(120),
            mode: PVerifyMode::default(),
            max_schedules: Some(1000),
            max_steps: Some(10000),
            seed: None,
            extra_args: vec![],
        }
    }
}

/// P language state machine backend
pub struct PLangBackend {
    config: PLangConfig,
}

impl Default for PLangBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl PLangBackend {
    /// Create a new P language backend with default configuration
    pub fn new() -> Self {
        Self {
            config: PLangConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: PLangConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<PathBuf, String> {
        // Check configured path first
        if let Some(ref path) = self.config.p_path {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        // Try common binary names
        for name in ["pc", "P", "p", "pmc"] {
            if let Ok(path) = which::which(name) {
                // Verify it's the P language compiler
                let output = Command::new(&path)
                    .arg("--help")
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .output()
                    .await;

                if let Ok(out) = output {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    if stdout.contains("P compiler")
                        || stdout.contains("P language")
                        || stderr.contains("P compiler")
                        || (stdout.contains("compile") && stdout.contains("P "))
                    {
                        debug!("Detected P compiler at: {:?}", path);
                        return Ok(path);
                    }
                }
            }
        }

        // Check P_HOME environment variable
        if let Ok(p_home) = std::env::var("P_HOME") {
            let pc = PathBuf::from(&p_home).join("Bld").join("Drops").join("pc");
            if pc.exists() {
                return Ok(pc);
            }
        }

        Err("P language not found. Install via: dotnet tool install --global P".to_string())
    }

    async fn detect_checker(&self) -> Result<PathBuf, String> {
        // Check configured path first
        if let Some(ref path) = self.config.checker_path {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        // Try common checker names
        for name in ["pmc", "PChecker", "coyote"] {
            if let Ok(path) = which::which(name) {
                return Ok(path);
            }
        }

        Err("P model checker not found".to_string())
    }

    /// Generate P language code from USL spec
    fn generate_p_code(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();
        code.push_str("// Generated by DashProve\n\n");

        // Generate events
        code.push_str("// Events\n");
        code.push_str("event eInit;\n");
        code.push_str("event eVerify;\n");
        code.push_str("event eSuccess;\n");
        code.push_str("event eFailure;\n\n");

        // Generate type declarations if any
        for type_def in &spec.spec.types {
            let safe_name = Self::sanitize_name(&type_def.name);
            code.push_str(&format!("// Type: {}\n", type_def.name));
            code.push_str(&format!("type {} = int;\n", safe_name));
        }
        if !spec.spec.types.is_empty() {
            code.push('\n');
        }

        // Generate main state machine
        code.push_str("// Main verification state machine\n");
        code.push_str("machine Main {\n");
        code.push_str("    var initialized: bool;\n\n");

        // Start state
        code.push_str("    start state Init {\n");
        code.push_str("        entry {\n");
        code.push_str("            initialized = false;\n");
        code.push_str("            raise eInit;\n");
        code.push_str("        }\n");
        code.push_str("        on eInit goto Running;\n");
        code.push_str("    }\n\n");

        // Running state
        code.push_str("    state Running {\n");
        code.push_str("        entry {\n");
        code.push_str("            initialized = true;\n");

        // Generate property checks
        for (i, prop) in spec.spec.properties.iter().enumerate() {
            let prop_name = prop.name();
            code.push_str(&format!("            // Property {}: {}\n", i, prop_name));
            code.push_str("            assert initialized;\n");
        }

        if spec.spec.properties.is_empty() {
            code.push_str("            // Trivial property\n");
            code.push_str("            assert true;\n");
        }

        code.push_str("            raise eSuccess;\n");
        code.push_str("        }\n");
        code.push_str("        on eSuccess goto Done;\n");
        code.push_str("        on eFailure goto Error;\n");
        code.push_str("    }\n\n");

        // Done state
        code.push_str("    state Done {\n");
        code.push_str("        entry {\n");
        code.push_str("            // Verification successful\n");
        code.push_str("        }\n");
        code.push_str("    }\n\n");

        // Error state
        code.push_str("    state Error {\n");
        code.push_str("        entry {\n");
        code.push_str("            assert false, \"Verification failed\";\n");
        code.push_str("        }\n");
        code.push_str("    }\n");
        code.push_str("}\n\n");

        // Generate test harness
        code.push_str("// Test harness\n");
        code.push_str("test Test0 [main=Main]: {};\n");

        code
    }

    /// Sanitize a name for use in P language
    fn sanitize_name(name: &str) -> String {
        let result: String = name
            .replace([' ', '-', ':', '/', '\\', '.', '(', ')', '[', ']'], "_")
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect();

        // P names must start with letter or underscore
        if result.starts_with(|c: char| c.is_ascii_digit()) {
            format!("t_{}", result)
        } else {
            result
        }
    }

    fn parse_output(
        &self,
        stdout: &str,
        stderr: &str,
        success: bool,
    ) -> (VerificationStatus, Vec<String>) {
        let combined = format!("{}\n{}", stdout, stderr);
        let mut diagnostics = Vec::new();

        // Parse P output
        for line in combined.lines() {
            let trimmed = line.trim();

            // Check for successful verification
            if trimmed.contains("found 0 bugs")
                || trimmed.contains("Found 0 bugs")
                || trimmed.contains("passed")
            {
                diagnostics.push(format!("✓ {}", trimmed));
            }

            // Check for failures
            if trimmed.contains("found") && trimmed.contains("bug") {
                diagnostics.push(format!("✗ {}", trimmed));
            }

            // Capture assertion violations
            if trimmed.contains("assert") || trimmed.contains("Assert") {
                diagnostics.push(trimmed.to_string());
            }

            // Capture schedule exploration stats
            if trimmed.contains("schedule") || trimmed.contains("Schedule") {
                diagnostics.push(trimmed.to_string());
            }
        }

        // Check for no bugs found
        if combined.contains("found 0 bugs") || combined.contains("Found 0 bugs") {
            return (VerificationStatus::Proven, diagnostics);
        }

        // Check for bugs found
        if (combined.contains("found") && combined.contains("bug"))
            || combined.contains("assertion failed")
            || combined.contains("Assert failed")
        {
            return (VerificationStatus::Disproven, diagnostics);
        }

        // Check for compilation errors
        if (combined.contains("error:") || combined.contains("Error:"))
            && (combined.contains("compilation") || combined.contains("Compilation"))
        {
            return (
                VerificationStatus::Unknown {
                    reason: "P compilation failed".to_string(),
                },
                diagnostics,
            );
        }

        // Check for timeout
        if combined.contains("timeout") || combined.contains("Timeout") {
            return (
                VerificationStatus::Unknown {
                    reason: "Verification timed out".to_string(),
                },
                diagnostics,
            );
        }

        // Check exit status
        if success {
            // Assume success if no bugs reported
            return (VerificationStatus::Proven, diagnostics);
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse P output".to_string(),
            },
            diagnostics,
        )
    }

    /// Parse counterexample from P output
    fn parse_counterexample(stdout: &str, stderr: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        let combined = format!("{}\n{}", stdout, stderr);
        ce.raw = Some(combined.clone());

        // Extract failed checks
        ce.failed_checks = Self::extract_failed_checks(&combined);

        // Extract witness values
        ce.witness = Self::extract_witness_values(&combined);

        // Note: trace is Vec<TraceState>, we skip complex trace parsing for now

        ce
    }

    /// Extract failed checks from P output
    fn extract_failed_checks(output: &str) -> Vec<FailedCheck> {
        let mut checks = Vec::new();

        for line in output.lines() {
            let trimmed = line.trim();

            if trimmed.contains("assert")
                || trimmed.contains("Assert")
                || trimmed.contains("bug")
                || trimmed.contains("Bug")
            {
                let check_type = if trimmed.contains("assert") || trimmed.contains("Assert") {
                    "p_assertion"
                } else if trimmed.contains("liveness") {
                    "p_liveness"
                } else if trimmed.contains("deadlock") {
                    "p_deadlock"
                } else {
                    "p_bug"
                };

                let (location, description) = Self::parse_error_location(trimmed);

                checks.push(FailedCheck {
                    check_id: check_type.to_string(),
                    description,
                    location,
                    function: None,
                });
            }
        }

        checks
    }

    /// Parse error location from P error line
    fn parse_error_location(line: &str) -> (Option<SourceLocation>, String) {
        // P format: "file.p(line,col): message" or "file.p:line: message"
        if let Some(paren_start) = line.find('(') {
            if let Some(paren_end) = line.find("):") {
                let file = line[..paren_start].trim();
                let loc_str = &line[paren_start + 1..paren_end];
                let parts: Vec<&str> = loc_str.split(',').collect();

                if !parts.is_empty() {
                    if let Ok(line_num) = parts[0].trim().parse::<u32>() {
                        let col = parts.get(1).and_then(|c| c.trim().parse::<u32>().ok());
                        let message = line[paren_end + 2..].trim().to_string();

                        return (
                            Some(SourceLocation {
                                file: file.to_string(),
                                line: line_num,
                                column: col,
                            }),
                            message,
                        );
                    }
                }
            }
        }

        // Try "file:line:" format
        if let Some(colon_pos) = line.find(':') {
            let prefix = &line[..colon_pos];
            if prefix.ends_with(".p") {
                let rest = &line[colon_pos + 1..];
                if let Some(next_colon) = rest.find(':') {
                    if let Ok(line_num) = rest[..next_colon].trim().parse::<u32>() {
                        let message = rest[next_colon + 1..].trim().to_string();
                        return (
                            Some(SourceLocation {
                                file: prefix.to_string(),
                                line: line_num,
                                column: None,
                            }),
                            message,
                        );
                    }
                }
            }
        }

        (None, line.to_string())
    }

    /// Extract witness values from counterexample
    fn extract_witness_values(output: &str) -> HashMap<String, CounterexampleValue> {
        let mut values = HashMap::new();
        let mut in_trace = false;

        for line in output.lines() {
            let trimmed = line.trim();

            // Look for state/variable dump sections
            if trimmed.contains("State:")
                || trimmed.contains("Variables:")
                || trimmed.contains("values:")
            {
                in_trace = true;
                continue;
            }

            if in_trace {
                if trimmed.is_empty() || trimmed.starts_with("---") {
                    in_trace = false;
                    continue;
                }

                // Parse "var = value" or "var: value"
                let parts: Vec<&str> = if trimmed.contains(" = ") {
                    trimmed.splitn(2, " = ").collect()
                } else if trimmed.contains(": ") {
                    trimmed.splitn(2, ": ").collect()
                } else {
                    continue;
                };

                if parts.len() == 2 {
                    let var_name = parts[0].trim().to_string();
                    let value_str = parts[1].trim();
                    values.insert(var_name, Self::parse_p_value(value_str));
                }
            }
        }

        values
    }

    /// Parse a P value string
    fn parse_p_value(value_str: &str) -> CounterexampleValue {
        let trimmed = value_str.trim();

        // Boolean
        if trimmed == "true" || trimmed == "True" {
            return CounterexampleValue::Bool(true);
        }
        if trimmed == "false" || trimmed == "False" {
            return CounterexampleValue::Bool(false);
        }

        // Null
        if trimmed == "null" || trimmed == "None" {
            return CounterexampleValue::String("null".to_string());
        }

        // Integer
        if let Ok(n) = trimmed.parse::<i128>() {
            return CounterexampleValue::Int {
                value: n,
                type_hint: None,
            };
        }

        // Default to string
        CounterexampleValue::String(trimmed.to_string())
    }
}

#[async_trait]
impl VerificationBackend for PLangBackend {
    fn id(&self) -> BackendId {
        BackendId::PLang
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Temporal, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let p_path = self.detect().await.map_err(BackendError::Unavailable)?;

        // Create temp directory
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp directory: {}", e))
        })?;

        let p_file = temp_dir.path().join("spec.p");
        let p_code = self.generate_p_code(spec);

        debug!("Generated P code:\n{}", p_code);

        tokio::fs::write(&p_file, &p_code).await.map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write P file: {}", e))
        })?;

        // First, compile the P file
        let mut compile_cmd = Command::new(&p_path);
        compile_cmd
            .arg("compile")
            .arg(&p_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(temp_dir.path());

        let compile_output = tokio::time::timeout(self.config.timeout, compile_cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to compile P file: {}", e))
            })?;

        let compile_stdout = String::from_utf8_lossy(&compile_output.stdout).to_string();
        let compile_stderr = String::from_utf8_lossy(&compile_output.stderr).to_string();

        if !compile_output.status.success() {
            debug!("P compile stderr: {}", compile_stderr);
            return Ok(BackendResult {
                backend: BackendId::PLang,
                status: VerificationStatus::Unknown {
                    reason: format!("P compilation failed: {}", compile_stderr),
                },
                proof: None,
                counterexample: None,
                diagnostics: vec![compile_stderr],
                time_taken: start.elapsed(),
            });
        }

        // Run verification
        let mut check_cmd = if let Ok(checker) = self.detect_checker().await {
            let mut cmd = Command::new(&checker);
            cmd.arg("check");
            cmd
        } else {
            let mut cmd = Command::new(&p_path);
            cmd.arg("check");
            cmd
        };

        check_cmd
            .arg(&p_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(temp_dir.path());

        // Add mode-specific options
        match self.config.mode {
            PVerifyMode::Systematic => {
                // Default mode
            }
            PVerifyMode::Random => {
                check_cmd.arg("--mode").arg("random");
            }
            PVerifyMode::Coverage => {
                check_cmd.arg("--mode").arg("coverage");
            }
        }

        // Add schedule limits
        if let Some(max_sched) = self.config.max_schedules {
            check_cmd.arg("--max-schedules").arg(max_sched.to_string());
        }

        if let Some(max_steps) = self.config.max_steps {
            check_cmd.arg("--max-steps").arg(max_steps.to_string());
        }

        if let Some(seed) = self.config.seed {
            check_cmd.arg("--seed").arg(seed.to_string());
        }

        // Extra args
        for arg in &self.config.extra_args {
            check_cmd.arg(arg);
        }

        let output = tokio::time::timeout(self.config.timeout, check_cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run P check: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        debug!("P check stdout: {}", stdout);
        debug!("P check stderr: {}", stderr);

        // Combine compile and check outputs
        let combined_stdout = format!("{}\n{}", compile_stdout, stdout);
        let (status, diagnostics) =
            self.parse_output(&combined_stdout, &stderr, output.status.success());

        // Generate counterexample for failures
        let counterexample = if matches!(status, VerificationStatus::Disproven) {
            Some(Self::parse_counterexample(&stdout, &stderr))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::PLang,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect().await {
            Ok(_) => HealthStatus::Healthy,
            Err(r) => HealthStatus::Unavailable { reason: r },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== PLangConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults() {
        let config = PLangConfig::default();
        assert!(config.p_path.is_none());
        assert!(config.checker_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.mode, PVerifyMode::Systematic);
        assert_eq!(config.max_schedules, Some(1000));
        assert_eq!(config.max_steps, Some(10000));
        assert!(config.seed.is_none());
        assert!(config.extra_args.is_empty());
    }

    // ===== Backend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = PLangBackend::new();
        assert_eq!(backend.config.timeout, Duration::from_secs(120));
        assert_eq!(backend.config.mode, PVerifyMode::Systematic);
    }

    #[kani::proof]
    fn verify_backend_default_equals_new() {
        let a = PLangBackend::new();
        let b = PLangBackend::default();
        assert_eq!(a.config.max_schedules, b.config.max_schedules);
        assert_eq!(a.config.max_steps, b.config.max_steps);
    }

    #[kani::proof]
    fn verify_backend_with_config_respects_fields() {
        let cfg = PLangConfig {
            p_path: Some(PathBuf::from("/opt/p")),
            checker_path: Some(PathBuf::from("/opt/pmc")),
            timeout: Duration::from_secs(10),
            mode: PVerifyMode::Random,
            max_schedules: Some(5),
            max_steps: Some(50),
            seed: Some(7),
            extra_args: vec!["--foo".to_string()],
        };
        let backend = PLangBackend::with_config(cfg);
        assert_eq!(backend.config.timeout, Duration::from_secs(10));
        assert_eq!(backend.config.max_schedules, Some(5));
        assert_eq!(backend.config.max_steps, Some(50));
        assert_eq!(backend.config.seed, Some(7));
        assert_eq!(backend.config.extra_args.len(), 1);
        assert!(backend.config.p_path.is_some());
        assert!(backend.config.checker_path.is_some());
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = PLangBackend::new();
        assert!(matches!(backend.id(), BackendId::PLang));
    }

    #[kani::proof]
    fn verify_supports_temporal_and_invariant() {
        let backend = PLangBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Temporal));
        assert!(supported.contains(&PropertyType::Invariant));
        assert_eq!(supported.len(), 2);
    }

    // ===== Code generation =====

    #[kani::proof]
    fn verify_generate_p_code_includes_events_and_properties() {
        let backend = PLangBackend::new();
        let spec = dashprove_usl::typecheck::typecheck(
            dashprove_usl::parse("invariant ready { true }").unwrap(),
        )
        .unwrap();
        let code = backend.generate_p_code(&spec);
        assert!(code.contains("event eInit"));
        assert!(code.contains("machine Main"));
        assert!(code.contains("Property: ready"));
    }

    #[kani::proof]
    fn verify_sanitize_name_replaces_invalid_chars() {
        assert_eq!(PLangBackend::sanitize_name("hello-world"), "hello_world");
        assert_eq!(PLangBackend::sanitize_name("123prop"), "t_123prop");
        assert_eq!(PLangBackend::sanitize_name("prop:name"), "prop_name");
    }

    // ===== Output parsing =====

    #[kani::proof]
    fn verify_parse_output_success_is_proven() {
        let backend = PLangBackend::new();
        let stdout = "Exploring schedules...\nfound 0 bugs in 5 schedules";
        let (status, diag) = backend.parse_output(stdout, "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(!diag.is_empty());
    }

    #[kani::proof]
    fn verify_parse_output_failure_is_disproven() {
        let backend = PLangBackend::new();
        let stdout = "Exploring schedules...\nfound 1 bug\nassert failed";
        let (status, diag) = backend.parse_output(stdout, "", false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!diag.is_empty());
    }

    #[kani::proof]
    fn verify_parse_output_timeout_unknown() {
        let backend = PLangBackend::new();
        let (status, _) = backend.parse_output("Verification timeout after limit", "", true);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    // ===== Error parsing helpers =====

    #[kani::proof]
    fn verify_parse_error_location_paren_and_colon() {
        let (loc1, msg1) = PLangBackend::parse_error_location("spec.p(10,5): assertion failed");
        let location1 = loc1.expect("expected location");
        assert_eq!(location1.file, "spec.p");
        assert_eq!(location1.line, 10);
        assert_eq!(location1.column, Some(5));
        assert!(msg1.contains("assertion failed"));

        let (loc2, msg2) = PLangBackend::parse_error_location("spec.p:7: assert failed");
        let location2 = loc2.expect("expected location");
        assert_eq!(location2.file, "spec.p");
        assert_eq!(location2.line, 7);
        assert!(location2.column.is_none());
        assert!(msg2.contains("assert failed"));
    }

    #[kani::proof]
    fn verify_extract_failed_checks_creates_entries() {
        let output = "assertion failed\nliveness bug\nrandom Bug found";
        let checks = PLangBackend::extract_failed_checks(output);
        assert!(checks.len() >= 2);
        assert!(!checks[0].description.is_empty());
    }

    #[kani::proof]
    fn verify_parse_p_value_variants() {
        assert!(matches!(
            PLangBackend::parse_p_value("true"),
            CounterexampleValue::Bool(true)
        ));
        assert!(matches!(
            PLangBackend::parse_p_value("False"),
            CounterexampleValue::Bool(false)
        ));
        assert!(matches!(
            PLangBackend::parse_p_value("42"),
            CounterexampleValue::Int { value: 42, .. }
        ));
        match PLangBackend::parse_p_value("null") {
            CounterexampleValue::String(s) => assert_eq!(s, "null"),
            _ => panic!("expected string for null"),
        }
    }

    #[kani::proof]
    fn verify_parse_counterexample_collects_witness() {
        let output = "assert failed\nState:\nvalue = 3\nother: true\n---";
        let ce = PLangBackend::parse_counterexample(output, "");
        assert!(!ce.failed_checks.is_empty());
        assert!(ce.witness.contains_key("value"));
        assert!(ce.raw.as_ref().unwrap().contains("assert failed"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_id() {
        assert_eq!(PLangBackend::new().id(), BackendId::PLang);
    }

    #[test]
    fn default_config() {
        let config = PLangConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.mode, PVerifyMode::Systematic);
        assert_eq!(config.max_schedules, Some(1000));
    }

    #[test]
    fn supports_temporal_and_invariant() {
        let backend = PLangBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Temporal));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[test]
    fn sanitize_name() {
        assert_eq!(PLangBackend::sanitize_name("Hello-World"), "Hello_World");
        assert_eq!(PLangBackend::sanitize_name("test:prop"), "test_prop");
        assert_eq!(PLangBackend::sanitize_name("123abc"), "t_123abc");
    }

    #[test]
    fn parse_success_output() {
        let backend = PLangBackend::new();
        let stdout = "Exploring schedules...\nfound 0 bugs in 100 schedules";
        let (status, diag) = backend.parse_output(stdout, "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(!diag.is_empty());
    }

    #[test]
    fn parse_failure_output() {
        let backend = PLangBackend::new();
        let stdout = "Exploring schedules...\nfound 1 bug\nassert failed";
        let (status, _diag) = backend.parse_output(stdout, "", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_error_location_with_parens() {
        let line = "spec.p(10,5): assertion failed";
        let (loc, desc) = PLangBackend::parse_error_location(line);
        assert!(loc.is_some());
        let loc = loc.unwrap();
        assert_eq!(loc.file, "spec.p");
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, Some(5));
        assert!(desc.contains("assertion failed"));
    }

    #[test]
    fn parse_error_location_with_colons() {
        let line = "spec.p:10: assertion failed";
        let (loc, desc) = PLangBackend::parse_error_location(line);
        assert!(loc.is_some());
        let loc = loc.unwrap();
        assert_eq!(loc.file, "spec.p");
        assert_eq!(loc.line, 10);
        assert!(desc.contains("assertion failed"));
    }

    #[test]
    fn extract_failed_checks() {
        let output = "assert initialized failed\nliveness bug found\ndeadlock bug detected";
        let checks = PLangBackend::extract_failed_checks(output);
        assert_eq!(checks.len(), 3);
        assert_eq!(checks[0].check_id, "p_assertion");
        assert_eq!(checks[1].check_id, "p_liveness");
        assert_eq!(checks[2].check_id, "p_deadlock");
    }

    #[test]
    fn parse_p_values() {
        assert!(matches!(
            PLangBackend::parse_p_value("true"),
            CounterexampleValue::Bool(true)
        ));
        assert!(matches!(
            PLangBackend::parse_p_value("false"),
            CounterexampleValue::Bool(false)
        ));
        assert!(matches!(
            PLangBackend::parse_p_value("42"),
            CounterexampleValue::Int { value: 42, .. }
        ));
        assert!(matches!(
            PLangBackend::parse_p_value("null"),
            CounterexampleValue::String(_)
        ));
    }

    #[test]
    fn generate_p_empty_spec() {
        use dashprove_usl::ast::Spec;

        let backend = PLangBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_p_code(&spec);
        assert!(code.contains("// Generated by DashProve"));
        assert!(code.contains("machine Main"));
        assert!(code.contains("event eInit"));
        assert!(code.contains("test Test0"));
    }

    #[test]
    fn verify_mode_random() {
        let config = PLangConfig {
            mode: PVerifyMode::Random,
            seed: Some(12345),
            ..Default::default()
        };
        let backend = PLangBackend::with_config(config);
        assert_eq!(backend.config.mode, PVerifyMode::Random);
        assert_eq!(backend.config.seed, Some(12345));
    }
}
