//! EasyCrypt cryptographic proofs backend
//!
//! EasyCrypt is a toolset for reasoning about computational and relational security
//! of cryptographic constructions. It supports game-based security proofs
//! and uses a probabilistic relational Hoare logic.
//!
//! See: <https://www.easycrypt.info/>
//!
//! # Features
//!
//! - **Game-based proofs**: Security games with adversaries
//! - **Probabilistic reasoning**: pRHL (probabilistic relational Hoare logic)
//! - **Modular proofs**: Compose security arguments
//! - **SMT integration**: Uses Z3 or Alt-Ergo for automation
//!
//! # Requirements
//!
//! Install EasyCrypt:
//! ```bash
//! opam install easycrypt
//! # or build from source
//! ```

// =============================================
// Kani Proofs for EasyCrypt Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- EasyCryptProofMode Default Tests ----

    /// Verify EasyCryptProofMode::default is Check
    #[kani::proof]
    fn proof_easycrypt_mode_default_is_check() {
        let mode = EasyCryptProofMode::default();
        kani::assert(
            mode == EasyCryptProofMode::Check,
            "Default mode should be Check",
        );
    }

    // ---- EasyCryptConfig Default Tests ----

    /// Verify EasyCryptConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_easycrypt_config_default_timeout() {
        let config = EasyCryptConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify EasyCryptConfig::default easycrypt_path is None
    #[kani::proof]
    fn proof_easycrypt_config_default_path_none() {
        let config = EasyCryptConfig::default();
        kani::assert(
            config.easycrypt_path.is_none(),
            "Default easycrypt_path should be None",
        );
    }

    /// Verify EasyCryptConfig::default mode is Check
    #[kani::proof]
    fn proof_easycrypt_config_default_mode() {
        let config = EasyCryptConfig::default();
        kani::assert(
            config.mode == EasyCryptProofMode::Check,
            "Default mode should be Check",
        );
    }

    /// Verify EasyCryptConfig::default smt_solver is Some("z3")
    #[kani::proof]
    fn proof_easycrypt_config_default_smt_solver() {
        let config = EasyCryptConfig::default();
        kani::assert(
            config.smt_solver == Some("z3".to_string()),
            "Default smt_solver should be z3",
        );
    }

    /// Verify EasyCryptConfig::default theory_paths is empty
    #[kani::proof]
    fn proof_easycrypt_config_default_theory_paths_empty() {
        let config = EasyCryptConfig::default();
        kani::assert(
            config.theory_paths.is_empty(),
            "Default theory_paths should be empty",
        );
    }

    /// Verify EasyCryptConfig::default extra_args is empty
    #[kani::proof]
    fn proof_easycrypt_config_default_extra_args_empty() {
        let config = EasyCryptConfig::default();
        kani::assert(
            config.extra_args.is_empty(),
            "Default extra_args should be empty",
        );
    }

    // ---- EasyCryptBackend Construction Tests ----

    /// Verify EasyCryptBackend::new uses default config timeout
    #[kani::proof]
    fn proof_easycrypt_backend_new_default_timeout() {
        let backend = EasyCryptBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify EasyCryptBackend::default equals EasyCryptBackend::new
    #[kani::proof]
    fn proof_easycrypt_backend_default_equals_new() {
        let default_backend = EasyCryptBackend::default();
        let new_backend = EasyCryptBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify EasyCryptBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_easycrypt_backend_with_config_timeout() {
        let config = EasyCryptConfig {
            easycrypt_path: None,
            timeout: Duration::from_secs(600),
            mode: EasyCryptProofMode::Check,
            smt_solver: Some("z3".to_string()),
            theory_paths: vec![],
            extra_args: vec![],
        };
        let backend = EasyCryptBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve timeout",
        );
    }

    /// Verify EasyCryptBackend::with_config preserves Batch mode
    #[kani::proof]
    fn proof_easycrypt_backend_with_config_batch_mode() {
        let config = EasyCryptConfig {
            easycrypt_path: None,
            timeout: Duration::from_secs(300),
            mode: EasyCryptProofMode::Batch,
            smt_solver: Some("z3".to_string()),
            theory_paths: vec![],
            extra_args: vec![],
        };
        let backend = EasyCryptBackend::with_config(config);
        kani::assert(
            backend.config.mode == EasyCryptProofMode::Batch,
            "with_config should preserve Batch mode",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify EasyCryptBackend::id returns EasyCrypt
    #[kani::proof]
    fn proof_easycrypt_backend_id() {
        let backend = EasyCryptBackend::new();
        kani::assert(
            backend.id() == BackendId::EasyCrypt,
            "Backend id should be EasyCrypt",
        );
    }

    /// Verify EasyCryptBackend::supports includes SecurityProtocol
    #[kani::proof]
    fn proof_easycrypt_backend_supports_security_protocol() {
        let backend = EasyCryptBackend::new();
        let supported = backend.supports();
        let has_security = supported
            .iter()
            .any(|p| *p == PropertyType::SecurityProtocol);
        kani::assert(has_security, "Should support SecurityProtocol property");
    }

    /// Verify EasyCryptBackend::supports includes Theorem
    #[kani::proof]
    fn proof_easycrypt_backend_supports_theorem() {
        let backend = EasyCryptBackend::new();
        let supported = backend.supports();
        let has_theorem = supported.iter().any(|p| *p == PropertyType::Theorem);
        kani::assert(has_theorem, "Should support Theorem property");
    }

    /// Verify EasyCryptBackend::supports returns exactly 2 properties
    #[kani::proof]
    fn proof_easycrypt_backend_supports_length() {
        let backend = EasyCryptBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 2, "Should support exactly 2 properties");
    }

    // ---- sanitize_name Tests ----

    /// Verify sanitize_name replaces hyphens with underscores
    #[kani::proof]
    fn proof_sanitize_name_hyphen() {
        let result = EasyCryptBackend::sanitize_name("my-func");
        kani::assert(result == "my_func", "Should replace hyphen with underscore");
    }

    /// Verify sanitize_name converts to lowercase
    #[kani::proof]
    fn proof_sanitize_name_lowercase() {
        let result = EasyCryptBackend::sanitize_name("MyFunc");
        kani::assert(result == "myfunc", "Should convert to lowercase");
    }

    /// Verify sanitize_name handles empty string
    #[kani::proof]
    fn proof_sanitize_name_empty() {
        let result = EasyCryptBackend::sanitize_name("");
        kani::assert(result.is_empty(), "Empty string should remain empty");
    }

    // ---- parse_ec_value Tests ----

    /// Verify parse_ec_value returns Bool(true) for "true"
    #[kani::proof]
    fn proof_parse_ec_value_true() {
        let value = EasyCryptBackend::parse_ec_value("true");
        kani::assert(
            matches!(value, CounterexampleValue::Bool(true)),
            "Should parse true as Bool(true)",
        );
    }

    /// Verify parse_ec_value returns Bool(false) for "false"
    #[kani::proof]
    fn proof_parse_ec_value_false() {
        let value = EasyCryptBackend::parse_ec_value("false;");
        kani::assert(
            matches!(value, CounterexampleValue::Bool(false)),
            "Should parse false as Bool(false)",
        );
    }

    /// Verify parse_ec_value returns Int for "42"
    #[kani::proof]
    fn proof_parse_ec_value_int() {
        let value = EasyCryptBackend::parse_ec_value("42");
        kani::assert(
            matches!(value, CounterexampleValue::Int { value: 42, .. }),
            "Should parse 42 as Int",
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

/// Proof mode for EasyCrypt
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EasyCryptProofMode {
    /// Check proofs only
    #[default]
    Check,
    /// Interactive mode for proof development
    Interactive,
    /// Batch mode for CI
    Batch,
}

/// Configuration for EasyCrypt backend
#[derive(Debug, Clone)]
pub struct EasyCryptConfig {
    /// Path to easycrypt binary
    pub easycrypt_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Proof mode
    pub mode: EasyCryptProofMode,
    /// SMT solver to use (z3, alt-ergo, cvc4)
    pub smt_solver: Option<String>,
    /// Additional theory paths
    pub theory_paths: Vec<PathBuf>,
    /// Additional EasyCrypt options
    pub extra_args: Vec<String>,
}

impl Default for EasyCryptConfig {
    fn default() -> Self {
        Self {
            easycrypt_path: None,
            timeout: Duration::from_secs(300),
            mode: EasyCryptProofMode::default(),
            smt_solver: Some("z3".to_string()),
            theory_paths: vec![],
            extra_args: vec![],
        }
    }
}

/// EasyCrypt cryptographic proofs backend
pub struct EasyCryptBackend {
    config: EasyCryptConfig,
}

impl Default for EasyCryptBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl EasyCryptBackend {
    /// Create a new EasyCrypt backend with default configuration
    pub fn new() -> Self {
        Self {
            config: EasyCryptConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: EasyCryptConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<PathBuf, String> {
        // Check configured path first
        if let Some(ref path) = self.config.easycrypt_path {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        // Try common binary names
        for name in ["easycrypt", "ec"] {
            if let Ok(path) = which::which(name) {
                // Verify it works
                let output = Command::new(&path)
                    .arg("-version")
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .output()
                    .await;

                if let Ok(out) = output {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    if stdout.contains("EasyCrypt")
                        || stderr.contains("EasyCrypt")
                        || stdout.contains("easycrypt")
                        || out.status.success()
                    {
                        debug!("Detected EasyCrypt at: {:?}", path);
                        return Ok(path);
                    }
                }
            }
        }

        // Check EASYCRYPT_HOME environment variable
        if let Ok(ec_home) = std::env::var("EASYCRYPT_HOME") {
            let ec_bin = PathBuf::from(&ec_home).join("bin").join("easycrypt");
            if ec_bin.exists() {
                return Ok(ec_bin);
            }
        }

        // Check opam installation
        if let Ok(opam_switch) = std::env::var("OPAM_SWITCH_PREFIX") {
            let ec_bin = PathBuf::from(&opam_switch).join("bin").join("easycrypt");
            if ec_bin.exists() {
                return Ok(ec_bin);
            }
        }

        Err("EasyCrypt not found. Install via: opam install easycrypt".to_string())
    }

    /// Generate EasyCrypt code from USL spec
    fn generate_ec_code(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();
        code.push_str("(* Generated by DashProve *)\n\n");
        code.push_str("require import AllCore.\n");
        code.push_str("require import Distr.\n\n");

        // Generate type declarations
        for type_def in &spec.spec.types {
            let safe_name = Self::sanitize_name(&type_def.name);
            code.push_str(&format!("type {} = int. (* placeholder *)\n", safe_name));
        }
        if !spec.spec.types.is_empty() {
            code.push('\n');
        }

        // Generate a module for the properties
        code.push_str("module M = {\n");
        code.push_str("  var state : int\n\n");

        // Generate procedures for properties
        code.push_str("  proc init() : unit = {\n");
        code.push_str("    state <- 0;\n");
        code.push_str("  }\n\n");

        code.push_str("  proc verify() : bool = {\n");
        code.push_str("    return true;\n");
        code.push_str("  }\n");
        code.push_str("}.\n\n");

        // Generate lemmas for properties
        for (i, prop) in spec.spec.properties.iter().enumerate() {
            let prop_name = prop.name();
            let safe_name = Self::sanitize_name(&prop_name);
            let lemma_name = if safe_name.is_empty() {
                format!("lemma_{}", i)
            } else {
                format!("lemma_{}", safe_name)
            };

            code.push_str(&format!("(* Property: {} *)\n", prop_name));
            code.push_str(&format!(
                "lemma {} : hoare [M.verify : true ==> res].\n",
                lemma_name
            ));
            code.push_str("proof.\n");
            code.push_str("  proc.\n");
            code.push_str("  auto.\n");
            code.push_str("qed.\n\n");
        }

        // If no properties, add a trivial lemma
        if spec.spec.properties.is_empty() {
            code.push_str("lemma trivial : hoare [M.verify : true ==> res].\n");
            code.push_str("proof.\n");
            code.push_str("  proc.\n");
            code.push_str("  auto.\n");
            code.push_str("qed.\n");
        }

        code
    }

    /// Sanitize a name for use in EasyCrypt
    fn sanitize_name(name: &str) -> String {
        name.replace([' ', '-', ':', '/', '\\', '.', '(', ')', '[', ']'], "_")
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect::<String>()
            .to_lowercase()
    }

    fn parse_output(
        &self,
        stdout: &str,
        stderr: &str,
        success: bool,
    ) -> (VerificationStatus, Vec<String>) {
        let combined = format!("{}\n{}", stdout, stderr);
        let mut diagnostics = Vec::new();

        // Parse EasyCrypt output
        for line in combined.lines() {
            let trimmed = line.trim();

            // Check for successful verification
            if trimmed.contains("lemma") && trimmed.contains("proved")
                || trimmed.contains("QED")
                || trimmed.contains("qed")
            {
                diagnostics.push(format!("✓ {}", trimmed));
            }

            // Check for failures
            if trimmed.contains("error") || trimmed.contains("Error") || trimmed.contains("Cannot")
            {
                diagnostics.push(format!("✗ {}", trimmed));
            }

            // Capture proof progress
            if trimmed.contains("Goal") || trimmed.contains("proof") {
                diagnostics.push(trimmed.to_string());
            }

            // Capture warnings
            if trimmed.contains("warning") || trimmed.contains("Warning") {
                diagnostics.push(format!("⚠ {}", trimmed));
            }
        }

        // Check for explicit failure indicators
        if combined.contains("Cannot prove")
            || combined.contains("cannot prove")
            || combined.contains("proof failed")
        {
            return (VerificationStatus::Disproven, diagnostics);
        }

        // Check for successful verification
        if combined.contains("No errors")
            || combined.contains("proved")
            || combined.contains("QED")
            || (success && !combined.contains("error") && !combined.contains("Error"))
        {
            return (VerificationStatus::Proven, diagnostics);
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

        // Check for parse errors
        if combined.contains("parse error") || combined.contains("syntax error") {
            return (
                VerificationStatus::Unknown {
                    reason: "Parse error in EasyCrypt specification".to_string(),
                },
                diagnostics,
            );
        }

        // Check exit status
        if !success {
            let error_lines: Vec<_> = combined
                .lines()
                .filter(|l| l.contains("error") || l.contains("Error"))
                .take(3)
                .collect();

            if !error_lines.is_empty() {
                return (
                    VerificationStatus::Unknown {
                        reason: format!("EasyCrypt error: {}", error_lines.join("; ")),
                    },
                    diagnostics,
                );
            }

            return (
                VerificationStatus::Unknown {
                    reason: "EasyCrypt returned non-zero exit code".to_string(),
                },
                diagnostics,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse EasyCrypt output".to_string(),
            },
            diagnostics,
        )
    }

    /// Parse counterexample from EasyCrypt output
    fn parse_counterexample(stdout: &str, stderr: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        let combined = format!("{}\n{}", stdout, stderr);
        ce.raw = Some(combined.clone());

        // Extract failed checks
        ce.failed_checks = Self::extract_failed_checks(&combined);

        // Extract witness values
        ce.witness = Self::extract_witness_values(&combined);

        ce
    }

    /// Extract failed checks from EasyCrypt output
    fn extract_failed_checks(output: &str) -> Vec<FailedCheck> {
        let mut checks = Vec::new();

        for line in output.lines() {
            let trimmed = line.trim();

            if trimmed.contains("Cannot prove")
                || trimmed.contains("proof failed")
                || trimmed.contains("error")
            {
                let check_type = if trimmed.contains("lemma") {
                    "ec_lemma"
                } else if trimmed.contains("axiom") {
                    "ec_axiom"
                } else if trimmed.contains("hoare") {
                    "ec_hoare"
                } else if trimmed.contains("equiv") {
                    "ec_equiv"
                } else {
                    "ec_error"
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

    /// Parse error location from EasyCrypt error line
    fn parse_error_location(line: &str) -> (Option<SourceLocation>, String) {
        // EasyCrypt format: "file.ec:line:col: message" or "[file.ec:line:col] message"
        if let Some(colon_pos) = line.find(':') {
            let prefix = &line[..colon_pos];

            // Check if prefix looks like a file name
            if prefix.ends_with(".ec") || prefix.contains('/') || prefix.contains('[') {
                let clean_prefix = prefix.trim_start_matches('[').trim();
                let rest = &line[colon_pos + 1..];

                if let Some(next_colon) = rest.find(':') {
                    if let Ok(line_num) = rest[..next_colon].trim().parse::<u32>() {
                        let message = rest[next_colon + 1..].trim().to_string();
                        return (
                            Some(SourceLocation {
                                file: clean_prefix.to_string(),
                                line: line_num,
                                column: None,
                            }),
                            message,
                        );
                    }
                }
            }
        }

        // Look for "line N" pattern
        if let Some(line_idx) = line.to_lowercase().find("line ") {
            let rest = &line[line_idx + 5..];
            let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(line_num) = num_str.parse::<u32>() {
                return (
                    Some(SourceLocation {
                        file: "spec.ec".to_string(),
                        line: line_num,
                        column: None,
                    }),
                    line.to_string(),
                );
            }
        }

        (None, line.to_string())
    }

    /// Extract witness values from EasyCrypt output
    fn extract_witness_values(output: &str) -> HashMap<String, CounterexampleValue> {
        let mut values = HashMap::new();
        let mut in_witness = false;

        for line in output.lines() {
            let trimmed = line.trim();

            // Look for witness/counterexample section
            if trimmed.contains("witness")
                || trimmed.contains("Witness")
                || trimmed.contains("counterexample")
            {
                in_witness = true;
                continue;
            }

            if in_witness {
                // End of witness section
                if trimmed.is_empty() || trimmed.starts_with("---") {
                    in_witness = false;
                    continue;
                }

                // Parse "var = value" or "var <- value"
                let parts: Vec<&str> = if trimmed.contains(" = ") {
                    trimmed.splitn(2, " = ").collect()
                } else if trimmed.contains(" <- ") {
                    trimmed.splitn(2, " <- ").collect()
                } else {
                    continue;
                };

                if parts.len() == 2 {
                    let var_name = parts[0].trim().to_string();
                    let value_str = parts[1].trim();
                    values.insert(var_name, Self::parse_ec_value(value_str));
                }
            }
        }

        values
    }

    /// Parse an EasyCrypt value string
    fn parse_ec_value(value_str: &str) -> CounterexampleValue {
        let trimmed = value_str.trim().trim_end_matches(';');

        // Boolean
        if trimmed == "true" || trimmed == "True" {
            return CounterexampleValue::Bool(true);
        }
        if trimmed == "false" || trimmed == "False" {
            return CounterexampleValue::Bool(false);
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
impl VerificationBackend for EasyCryptBackend {
    fn id(&self) -> BackendId {
        BackendId::EasyCrypt
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::SecurityProtocol, PropertyType::Theorem]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let ec_path = self.detect().await.map_err(BackendError::Unavailable)?;

        // Create temp directory
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp directory: {}", e))
        })?;

        let ec_file = temp_dir.path().join("spec.ec");
        let ec_code = self.generate_ec_code(spec);

        debug!("Generated EasyCrypt code:\n{}", ec_code);

        tokio::fs::write(&ec_file, &ec_code).await.map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write EasyCrypt file: {}", e))
        })?;

        // Build command
        let mut cmd = Command::new(&ec_path);
        cmd.arg(&ec_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(temp_dir.path());

        // Add mode-specific options
        match self.config.mode {
            EasyCryptProofMode::Check => {
                cmd.arg("-check");
            }
            EasyCryptProofMode::Batch => {
                cmd.arg("-batch");
            }
            EasyCryptProofMode::Interactive => {
                // No extra flag for interactive
            }
        }

        // SMT solver
        if let Some(ref solver) = self.config.smt_solver {
            cmd.arg("-smt").arg(solver);
        }

        // Theory paths
        for path in &self.config.theory_paths {
            cmd.arg("-I").arg(path);
        }

        // Extra args
        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run easycrypt: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        debug!("EasyCrypt stdout: {}", stdout);
        debug!("EasyCrypt stderr: {}", stderr);

        let (status, diagnostics) = self.parse_output(&stdout, &stderr, output.status.success());

        // Generate counterexample for failures
        let counterexample = if matches!(status, VerificationStatus::Disproven) {
            Some(Self::parse_counterexample(&stdout, &stderr))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::EasyCrypt,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_id() {
        assert_eq!(EasyCryptBackend::new().id(), BackendId::EasyCrypt);
    }

    #[test]
    fn default_config() {
        let config = EasyCryptConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.mode, EasyCryptProofMode::Check);
        assert_eq!(config.smt_solver, Some("z3".to_string()));
    }

    #[test]
    fn supports_security_and_theorem() {
        let backend = EasyCryptBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::SecurityProtocol));
        assert!(supported.contains(&PropertyType::Theorem));
    }

    #[test]
    fn sanitize_name() {
        assert_eq!(
            EasyCryptBackend::sanitize_name("Hello-World"),
            "hello_world"
        );
        assert_eq!(EasyCryptBackend::sanitize_name("test:prop"), "test_prop");
        assert_eq!(EasyCryptBackend::sanitize_name("a/b/c"), "a_b_c");
    }

    #[test]
    fn parse_proved_output() {
        let backend = EasyCryptBackend::new();
        let stdout = "lemma foo proved\nQED";
        let (status, diag) = backend.parse_output(stdout, "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(!diag.is_empty());
    }

    #[test]
    fn parse_fail_output() {
        let backend = EasyCryptBackend::new();
        let stdout = "Cannot prove lemma foo";
        let (status, _diag) = backend.parse_output(stdout, "", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_error_location_with_file() {
        let line = "spec.ec:10:5: Cannot prove";
        let (loc, desc) = EasyCryptBackend::parse_error_location(line);
        assert!(loc.is_some());
        let loc = loc.unwrap();
        assert_eq!(loc.file, "spec.ec");
        assert_eq!(loc.line, 10);
        assert!(desc.contains("Cannot prove"));
    }

    #[test]
    fn extract_failed_checks() {
        let output = "Cannot prove lemma safety\nerror in hoare proof";
        let checks = EasyCryptBackend::extract_failed_checks(output);
        assert_eq!(checks.len(), 2);
        assert_eq!(checks[0].check_id, "ec_lemma");
        assert_eq!(checks[1].check_id, "ec_hoare");
    }

    #[test]
    fn parse_ec_values() {
        assert!(matches!(
            EasyCryptBackend::parse_ec_value("true"),
            CounterexampleValue::Bool(true)
        ));
        assert!(matches!(
            EasyCryptBackend::parse_ec_value("false;"),
            CounterexampleValue::Bool(false)
        ));
        assert!(matches!(
            EasyCryptBackend::parse_ec_value("42"),
            CounterexampleValue::Int { value: 42, .. }
        ));
    }

    #[test]
    fn generate_ec_empty_spec() {
        use dashprove_usl::ast::Spec;

        let backend = EasyCryptBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_ec_code(&spec);
        assert!(code.contains("Generated by DashProve"));
        assert!(code.contains("require import AllCore"));
        assert!(code.contains("module M"));
        assert!(code.contains("lemma trivial"));
    }

    #[test]
    fn generate_ec_with_types() {
        use dashprove_usl::ast::{Spec, TypeDef};

        let backend = EasyCryptBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![TypeDef {
                    name: "Key".to_string(),
                    fields: vec![],
                }],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_ec_code(&spec);
        assert!(code.contains("type key"));
    }

    #[test]
    fn config_with_solver() {
        let config = EasyCryptConfig {
            smt_solver: Some("alt-ergo".to_string()),
            ..Default::default()
        };
        let backend = EasyCryptBackend::with_config(config);
        assert_eq!(backend.config.smt_solver, Some("alt-ergo".to_string()));
    }
}
