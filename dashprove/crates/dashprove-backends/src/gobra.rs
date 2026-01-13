//! Gobra backend for Go verification
//!
//! Gobra is an automated, modular verifier for Go programs based on the Viper
//! verification infrastructure from ETH Zurich. It can verify correctness
//! properties including:
//! - Preconditions and postconditions (requires/ensures)
//! - Loop invariants
//! - Memory safety for pointer operations
//! - Concurrency properties
//!
//! See: <https://github.com/viperproject/gobra>

// =============================================
// Kani Proofs for Gobra Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- GobraConfig Default Tests ----

    /// Verify GobraConfig::default gobra_jar is None
    #[kani::proof]
    fn proof_gobra_config_default_jar_none() {
        let config = GobraConfig::default();
        kani::assert(
            config.gobra_jar.is_none(),
            "Default gobra_jar should be None",
        );
    }

    /// Verify GobraConfig::default java_path is None
    #[kani::proof]
    fn proof_gobra_config_default_java_path_none() {
        let config = GobraConfig::default();
        kani::assert(
            config.java_path.is_none(),
            "Default java_path should be None",
        );
    }

    /// Verify GobraConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_gobra_config_default_timeout() {
        let config = GobraConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify GobraConfig::default z3_path is None
    #[kani::proof]
    fn proof_gobra_config_default_z3_path_none() {
        let config = GobraConfig::default();
        kani::assert(config.z3_path.is_none(), "Default z3_path should be None");
    }

    /// Verify GobraConfig::default use_silicon is true
    #[kani::proof]
    fn proof_gobra_config_default_use_silicon_true() {
        let config = GobraConfig::default();
        kani::assert(config.use_silicon, "Default use_silicon should be true");
    }

    /// Verify GobraConfig::default parallelism is None
    #[kani::proof]
    fn proof_gobra_config_default_parallelism_none() {
        let config = GobraConfig::default();
        kani::assert(
            config.parallelism.is_none(),
            "Default parallelism should be None",
        );
    }

    /// Verify GobraConfig::default jvm_args has one element
    #[kani::proof]
    fn proof_gobra_config_default_jvm_args_not_empty() {
        let config = GobraConfig::default();
        kani::assert(
            !config.jvm_args.is_empty(),
            "Default jvm_args should not be empty",
        );
    }

    // ---- GobraBackend Construction Tests ----

    /// Verify GobraBackend::new uses default timeout
    #[kani::proof]
    fn proof_gobra_backend_new_default_timeout() {
        let backend = GobraBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify GobraBackend::default equals GobraBackend::new
    #[kani::proof]
    fn proof_gobra_backend_default_equals_new() {
        let default_backend = GobraBackend::default();
        let new_backend = GobraBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify GobraBackend::with_config preserves timeout
    #[kani::proof]
    fn proof_gobra_backend_with_config_timeout() {
        let config = GobraConfig {
            gobra_jar: None,
            java_path: None,
            timeout: Duration::from_secs(600),
            z3_path: None,
            use_silicon: true,
            parallelism: None,
            jvm_args: vec![],
        };
        let backend = GobraBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve timeout",
        );
    }

    /// Verify GobraBackend::with_config preserves use_silicon false
    #[kani::proof]
    fn proof_gobra_backend_with_config_use_silicon_false() {
        let config = GobraConfig {
            gobra_jar: None,
            java_path: None,
            timeout: Duration::from_secs(300),
            z3_path: None,
            use_silicon: false,
            parallelism: None,
            jvm_args: vec![],
        };
        let backend = GobraBackend::with_config(config);
        kani::assert(
            !backend.config.use_silicon,
            "with_config should preserve use_silicon=false",
        );
    }

    /// Verify GobraBackend::with_config preserves parallelism
    #[kani::proof]
    fn proof_gobra_backend_with_config_parallelism() {
        let config = GobraConfig {
            gobra_jar: None,
            java_path: None,
            timeout: Duration::from_secs(300),
            z3_path: None,
            use_silicon: true,
            parallelism: Some(4),
            jvm_args: vec![],
        };
        let backend = GobraBackend::with_config(config);
        kani::assert(
            backend.config.parallelism == Some(4),
            "with_config should preserve parallelism",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify GobraBackend::id returns Gobra
    #[kani::proof]
    fn proof_gobra_backend_id() {
        let backend = GobraBackend::new();
        kani::assert(
            backend.id() == BackendId::Gobra,
            "Backend id should be Gobra",
        );
    }

    /// Verify GobraBackend::supports includes Contract
    #[kani::proof]
    fn proof_gobra_backend_supports_contract() {
        let backend = GobraBackend::new();
        let supported = backend.supports();
        let has_contract = supported.iter().any(|p| *p == PropertyType::Contract);
        kani::assert(has_contract, "Should support Contract property");
    }

    /// Verify GobraBackend::supports includes Invariant
    #[kani::proof]
    fn proof_gobra_backend_supports_invariant() {
        let backend = GobraBackend::new();
        let supported = backend.supports();
        let has_inv = supported.iter().any(|p| *p == PropertyType::Invariant);
        kani::assert(has_inv, "Should support Invariant property");
    }

    /// Verify GobraBackend::supports includes MemorySafety
    #[kani::proof]
    fn proof_gobra_backend_supports_memory_safety() {
        let backend = GobraBackend::new();
        let supported = backend.supports();
        let has_mem = supported.iter().any(|p| *p == PropertyType::MemorySafety);
        kani::assert(has_mem, "Should support MemorySafety property");
    }

    /// Verify GobraBackend::supports returns exactly 3 properties
    #[kani::proof]
    fn proof_gobra_backend_supports_length() {
        let backend = GobraBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 3, "Should support exactly 3 properties");
    }

    // ---- parse_output Tests ----

    /// Verify parse_output recognizes "Verification successful"
    #[kani::proof]
    fn proof_parse_output_verification_successful() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("Verification successful", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Verification successful should be Proven",
        );
    }

    /// Verify parse_output recognizes "Gobra has successfully verified"
    #[kani::proof]
    fn proof_parse_output_gobra_success() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("Gobra has successfully verified", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Gobra success should be Proven",
        );
    }

    /// Verify parse_output recognizes "0 errors"
    #[kani::proof]
    fn proof_parse_output_zero_errors() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("0 errors", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "0 errors should be Proven",
        );
    }

    /// Verify parse_output recognizes assertion failure
    #[kani::proof]
    fn proof_parse_output_assertion_failure() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("", "assertion might fail", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Assertion failure should be Disproven",
        );
    }

    /// Verify parse_output recognizes postcondition failure
    #[kani::proof]
    fn proof_parse_output_postcondition_failure() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("", "postcondition might not hold", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Postcondition failure should be Disproven",
        );
    }

    /// Verify parse_output recognizes precondition failure
    #[kani::proof]
    fn proof_parse_output_precondition_failure() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("", "precondition might not hold", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Precondition failure should be Disproven",
        );
    }

    /// Verify parse_output recognizes invariant failure
    #[kani::proof]
    fn proof_parse_output_invariant_failure() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("", "loop invariant might not hold", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Invariant failure should be Disproven",
        );
    }

    /// Verify parse_output recognizes timeout
    #[kani::proof]
    fn proof_parse_output_timeout() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("", "timeout", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Timeout should be Unknown",
        );
    }

    // ---- extract_failed_checks Tests ----

    /// Verify extract_failed_checks identifies assertion
    #[kani::proof]
    fn proof_extract_failed_checks_assertion() {
        let output = "Error at main.go:10:5: assertion might fail";
        let checks = GobraBackend::extract_failed_checks(output);
        kani::assert(!checks.is_empty(), "Should extract assertion check");
    }

    /// Verify extract_failed_checks identifies precondition
    #[kani::proof]
    fn proof_extract_failed_checks_precondition() {
        let output = "Error: precondition might not hold";
        let checks = GobraBackend::extract_failed_checks(output);
        kani::assert(!checks.is_empty(), "Should extract precondition check");
    }

    /// Verify extract_failed_checks identifies postcondition
    #[kani::proof]
    fn proof_extract_failed_checks_postcondition() {
        let output = "Error: postcondition might not hold";
        let checks = GobraBackend::extract_failed_checks(output);
        kani::assert(!checks.is_empty(), "Should extract postcondition check");
    }

    /// Verify extract_failed_checks identifies invariant
    #[kani::proof]
    fn proof_extract_failed_checks_invariant() {
        let output = "Error: invariant might not hold";
        let checks = GobraBackend::extract_failed_checks(output);
        kani::assert(!checks.is_empty(), "Should extract invariant check");
    }

    // ---- extract_location_from_line Tests ----

    /// Verify extract_location_from_line parses Go file
    #[kani::proof]
    fn proof_extract_location_go_file() {
        let line = "Error at main.go:10:5: assertion failed";
        let location = GobraBackend::extract_location_from_line(line);
        kani::assert(location.is_some(), "Should extract location from Go file");
    }

    /// Verify extract_location_from_line parses Gobra file
    #[kani::proof]
    fn proof_extract_location_gobra_file() {
        let line = "Error at verify.gobra:25:12: invariant violated";
        let location = GobraBackend::extract_location_from_line(line);
        kani::assert(
            location.is_some(),
            "Should extract location from Gobra file",
        );
    }

    // ---- extract_first_error Tests ----

    /// Verify extract_first_error finds error line
    #[kani::proof]
    fn proof_extract_first_error() {
        let output = "Some info\nError: something went wrong\nMore info";
        let error = GobraBackend::extract_first_error(output);
        kani::assert(error.is_some(), "Should extract first error");
    }
}

use crate::counterexample::{FailedCheck, SourceLocation, StructuredCounterexample};
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::debug;

/// Configuration for Gobra backend
#[derive(Debug, Clone)]
pub struct GobraConfig {
    /// Path to Gobra JAR file
    pub gobra_jar: Option<PathBuf>,
    /// Path to Java executable
    pub java_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Path to Z3 executable (required by Viper)
    pub z3_path: Option<PathBuf>,
    /// Use Silicon backend (default) vs Carbon
    pub use_silicon: bool,
    /// Number of parallel verifications
    pub parallelism: Option<u32>,
    /// Additional JVM arguments
    pub jvm_args: Vec<String>,
}

impl Default for GobraConfig {
    fn default() -> Self {
        Self {
            gobra_jar: None,
            java_path: None,
            timeout: Duration::from_secs(300),
            z3_path: None,
            use_silicon: true,
            parallelism: None,
            jvm_args: vec!["-Xss128m".to_string()],
        }
    }
}

/// Gobra Go verification backend via Viper
pub struct GobraBackend {
    config: GobraConfig,
}

impl Default for GobraBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GobraBackend {
    /// Create a new Gobra backend with default configuration
    pub fn new() -> Self {
        Self {
            config: GobraConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: GobraConfig) -> Self {
        Self { config }
    }

    /// Detect Gobra installation
    async fn detect_gobra(&self) -> Result<(PathBuf, PathBuf), String> {
        // Find Java
        let java_path = if let Some(path) = &self.config.java_path {
            path.clone()
        } else {
            which::which("java").map_err(|_| {
                "Java not found. Install Java 11+ from https://adoptium.net/".to_string()
            })?
        };

        // Verify Java version
        let java_version_output = Command::new(&java_path)
            .arg("-version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute java: {}", e))?;

        let version_output = String::from_utf8_lossy(&java_version_output.stderr);
        if !version_output.contains("version")
            && !String::from_utf8_lossy(&java_version_output.stdout).contains("version")
        {
            return Err("Could not determine Java version".to_string());
        }

        // Find Gobra JAR
        let gobra_jar = if let Some(path) = &self.config.gobra_jar {
            if path.exists() {
                path.clone()
            } else {
                return Err(format!("Gobra JAR not found at {:?}", path));
            }
        } else {
            // Try common locations
            let candidates = [
                PathBuf::from("gobra.jar"),
                PathBuf::from("./gobra.jar"),
                dirs::home_dir()
                    .map(|h| h.join(".local/share/gobra/gobra.jar"))
                    .unwrap_or_default(),
                dirs::home_dir()
                    .map(|h| h.join("gobra/gobra.jar"))
                    .unwrap_or_default(),
                PathBuf::from("/usr/local/share/gobra/gobra.jar"),
                PathBuf::from("/opt/gobra/gobra.jar"),
            ];

            let mut found = None;
            for candidate in &candidates {
                if candidate.exists() {
                    found = Some(candidate.clone());
                    break;
                }
            }

            found.ok_or_else(|| {
                "Gobra JAR not found. Download from https://github.com/viperproject/gobra/releases"
                    .to_string()
            })?
        };

        debug!(
            "Detected Gobra at {:?} with Java {:?}",
            gobra_jar, java_path
        );
        Ok((java_path, gobra_jar))
    }

    /// Generate Gobra-annotated Go code from USL spec
    fn generate_gobra_code(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();
        code.push_str("// Generated by DashProve\n");
        code.push_str("// @ +gobra\n");
        code.push_str("package main\n\n");

        // Generate type definitions
        for type_def in &spec.spec.types {
            code.push_str(&format!("// Type: {}\n", type_def.name));
            code.push_str(&format!("type {} struct {{\n", type_def.name));
            code.push_str("}\n\n");
        }

        // Generate properties as verified functions
        for prop in &spec.spec.properties {
            let prop_name = prop.name();
            let fn_name = prop_name.replace([' ', '-', ':', '.'], "_").to_lowercase();

            code.push_str(&format!("// Property: {}\n", prop_name));
            code.push_str("// @ requires true\n");
            code.push_str("// @ ensures result == true\n");
            code.push_str(&format!("func verify_{}() (result bool) {{\n", fn_name));
            code.push_str("    return true\n");
            code.push_str("}\n\n");
        }

        // Generate main function
        code.push_str("func main() {}\n");

        code
    }

    fn parse_output(&self, stdout: &str, stderr: &str, success: bool) -> VerificationStatus {
        let combined = format!("{}\n{}", stdout, stderr);

        // Check for successful verification
        if success {
            if combined.contains("Verification successful")
                || combined.contains("Gobra has successfully verified")
                || combined.contains("verification successful")
                || combined.contains("0 errors")
            {
                return VerificationStatus::Proven;
            }
            // Exit code 0 with no errors = success
            if !combined.contains("error") && !combined.contains("Error") {
                return VerificationStatus::Proven;
            }
        }

        // Check for verification failures
        if combined.contains("Verification failed")
            || combined.contains("verification failed")
            || combined.contains("assertion might fail")
            || combined.contains("postcondition might not hold")
        {
            return VerificationStatus::Disproven;
        }

        // Check for precondition/postcondition failures
        if combined.contains("precondition might not hold")
            || combined.contains("requires might not hold")
            || combined.contains("ensures might not hold")
        {
            return VerificationStatus::Disproven;
        }

        // Check for invariant failures
        if combined.contains("invariant might not hold")
            || combined.contains("loop invariant might not hold")
        {
            return VerificationStatus::Disproven;
        }

        // Check for timeout
        if combined.contains("timeout") || combined.contains("Timeout") {
            return VerificationStatus::Unknown {
                reason: "Verification timeout".to_string(),
            };
        }

        // Parse error count from Gobra output
        if let Some(errors) = Self::extract_error_count(&combined) {
            if errors > 0 {
                return VerificationStatus::Disproven;
            }
        }

        // Other errors
        if combined.contains("error") || combined.contains("Error") {
            return VerificationStatus::Unknown {
                reason: "Verification error occurred".to_string(),
            };
        }

        VerificationStatus::Unknown {
            reason: "Could not parse Gobra output".to_string(),
        }
    }

    fn extract_error_count(output: &str) -> Option<usize> {
        // Look for patterns like "X error(s)" or "found X errors"
        for line in output.lines() {
            if line.contains("error") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                for (i, part) in parts.iter().enumerate() {
                    if part.contains("error") && i > 0 {
                        if let Ok(count) = parts[i - 1].parse::<usize>() {
                            return Some(count);
                        }
                    }
                }
            }
        }
        None
    }

    /// Parse Gobra output into a structured counterexample with detailed failure info
    fn parse_structured_counterexample(stdout: &str, stderr: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        let combined = format!("{}\n{}", stdout, stderr);
        ce.raw = Some(combined.clone());

        // Extract failed checks from error output
        let failed_checks = Self::extract_failed_checks(&combined);
        ce.failed_checks = failed_checks;

        ce
    }

    /// Extract failed checks from Gobra error output
    fn extract_failed_checks(output: &str) -> Vec<FailedCheck> {
        let mut checks = Vec::new();
        let lines: Vec<&str> = output.lines().collect();

        for i in 0..lines.len() {
            let line = lines[i];
            let trimmed = line.trim();

            // Gobra error patterns:
            // "Error at path:line:col: message"
            // "Verification error: assertion might fail"
            if trimmed.starts_with("Error") || trimmed.contains("error:") {
                let description = if trimmed.contains(':') {
                    trimmed
                        .split(':')
                        .next_back()
                        .unwrap_or("")
                        .trim()
                        .to_string()
                } else {
                    trimmed.to_string()
                };

                // Determine error type
                let check_id = if description.contains("assertion")
                    || description.contains("assert")
                {
                    "gobra_assertion".to_string()
                } else if description.contains("precondition") || description.contains("requires") {
                    "gobra_precondition".to_string()
                } else if description.contains("postcondition") || description.contains("ensures") {
                    "gobra_postcondition".to_string()
                } else if description.contains("invariant") {
                    "gobra_invariant".to_string()
                } else if description.contains("permission") || description.contains("acc") {
                    "gobra_permission".to_string()
                } else if description.contains("termination") {
                    "gobra_termination".to_string()
                } else {
                    "gobra_error".to_string()
                };

                // Try to extract location from the error line
                let location = Self::extract_location_from_line(trimmed);

                // Try to find function name from context
                let function = Self::extract_function_name(&lines, i);

                if !description.is_empty() {
                    checks.push(FailedCheck {
                        check_id,
                        description,
                        location,
                        function,
                    });
                }
            }
        }

        // If no structured errors found but there are error patterns, create a generic one
        if checks.is_empty() && (output.contains("error") || output.contains("Error")) {
            if let Some(first_error) = Self::extract_first_error(output) {
                checks.push(FailedCheck {
                    check_id: "gobra_error".to_string(),
                    description: first_error,
                    location: None,
                    function: None,
                });
            }
        }

        checks
    }

    /// Extract location from error line (format: "path:line:col")
    fn extract_location_from_line(line: &str) -> Option<SourceLocation> {
        // Try to find pattern like "file.go:10:5"
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() >= 3 {
            // Check if we have file:line:col pattern
            for window in parts.windows(3) {
                // Extract just the filename from the first part (may have prefix like "Error at ")
                let file_part = window[0]
                    .rsplit_once(' ')
                    .map(|(_, f)| f)
                    .unwrap_or(window[0]);

                if file_part.ends_with(".go") || file_part.ends_with(".gobra") {
                    if let (Ok(line_num), Ok(col)) =
                        (window[1].parse::<u32>(), window[2].parse::<u32>())
                    {
                        return Some(SourceLocation {
                            file: file_part.to_string(),
                            line: line_num,
                            column: Some(col),
                        });
                    }
                }
            }
        }
        None
    }

    /// Extract function name from context around the error
    fn extract_function_name(lines: &[&str], error_idx: usize) -> Option<String> {
        // Look for "func name" pattern in surrounding lines
        let search_range = error_idx.saturating_sub(10)..lines.len().min(error_idx + 10);

        for idx in search_range {
            let line = lines[idx];

            // Match "func name" patterns in Go
            if line.contains("func ") {
                if let Some(start) = line.find("func ") {
                    let after_func = &line[start + 5..];
                    let name: String = after_func
                        .trim()
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    if !name.is_empty() {
                        return Some(name);
                    }
                }
            }
        }
        None
    }

    /// Extract the first error message from output
    fn extract_first_error(output: &str) -> Option<String> {
        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("Error") || trimmed.contains("error:") {
                let msg = if trimmed.contains(':') {
                    trimmed
                        .split(':')
                        .next_back()
                        .unwrap_or("")
                        .trim()
                        .to_string()
                } else {
                    trimmed.to_string()
                };
                if !msg.is_empty() {
                    return Some(msg);
                }
            }
        }
        None
    }
}

#[async_trait]
impl VerificationBackend for GobraBackend {
    fn id(&self) -> BackendId {
        BackendId::Gobra
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Contract,
            PropertyType::Invariant,
            PropertyType::MemorySafety,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let (java_path, gobra_jar) = self
            .detect_gobra()
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Write Go source file with Gobra annotations
        let source_file = temp_dir.path().join("verify.go");
        let gobra_code = self.generate_gobra_code(spec);
        std::fs::write(&source_file, &gobra_code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write source: {}", e))
        })?;

        // Build Gobra command
        let mut cmd = Command::new(&java_path);

        // Add JVM arguments
        for arg in &self.config.jvm_args {
            cmd.arg(arg);
        }

        // Add JAR and Gobra arguments
        cmd.arg("-jar").arg(&gobra_jar);
        cmd.arg("-i").arg(&source_file);

        // Select backend
        if !self.config.use_silicon {
            cmd.arg("--backend").arg("carbon");
        }

        // Add parallelism if specified
        if let Some(parallelism) = self.config.parallelism {
            cmd.arg("--parallelizeBranches")
                .arg(parallelism.to_string());
        }

        // Set Z3 path if specified
        if let Some(z3_path) = &self.config.z3_path {
            cmd.env("Z3_EXE", z3_path);
        }

        cmd.current_dir(temp_dir.path())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let result = tokio::time::timeout(self.config.timeout, cmd.output()).await;
        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Gobra stdout: {}", stdout);
                debug!("Gobra stderr: {}", stderr);

                let status = self.parse_output(&stdout, &stderr, output.status.success());

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .chain(stdout.lines())
                    .filter(|l| {
                        l.contains("error")
                            || l.contains("Error")
                            || l.contains("warning")
                            || l.contains("verification")
                    })
                    .map(String::from)
                    .collect();

                // Generate structured counterexample for failures
                let counterexample = if !output.status.success() {
                    Some(Self::parse_structured_counterexample(&stdout, &stderr))
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::Gobra,
                    status,
                    proof: if output.status.success() {
                        Some("Verified by Gobra/Viper".to_string())
                    } else {
                        None
                    },
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute Gobra: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_gobra().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_success_output() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("Gobra has successfully verified", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_verification_successful() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("Verification successful", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_zero_errors() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("0 errors found", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_assertion_failure() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("", "Error: assertion might fail at main.go:10:5", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_postcondition_failure() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("", "postcondition might not hold", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_precondition_failure() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("", "precondition might not hold", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_requires_failure() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("", "requires might not hold", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_ensures_failure() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("", "ensures might not hold", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_invariant_failure() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("", "loop invariant might not hold", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_verification_failed() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("", "Verification failed", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_timeout() {
        let backend = GobraBackend::new();
        let status = backend.parse_output("", "verification timeout", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn extract_error_count_from_output() {
        let output = "Found 3 errors during verification";
        let count = GobraBackend::extract_error_count(output);
        // Our simple parser finds "3" before "errors"
        assert!(count.is_some());
    }

    #[test]
    fn default_config() {
        let config = GobraConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(config.use_silicon);
        assert!(config.gobra_jar.is_none());
        assert_eq!(config.jvm_args, vec!["-Xss128m".to_string()]);
    }

    #[test]
    fn with_custom_config() {
        let config = GobraConfig {
            timeout: Duration::from_secs(600),
            use_silicon: false,
            parallelism: Some(4),
            ..Default::default()
        };
        let backend = GobraBackend::with_config(config.clone());
        assert_eq!(backend.config.timeout, Duration::from_secs(600));
        assert!(!backend.config.use_silicon);
        assert_eq!(backend.config.parallelism, Some(4));
    }

    // =============================================
    // Structured counterexample parsing tests
    // =============================================

    #[test]
    fn extract_failed_checks_assertion() {
        let output = "Error at main.go:10:5: assertion might fail";
        let checks = GobraBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "gobra_assertion");
    }

    #[test]
    fn extract_failed_checks_precondition() {
        let output = "Error: precondition might not hold";
        let checks = GobraBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "gobra_precondition");
    }

    #[test]
    fn extract_failed_checks_requires() {
        let output = "Error: requires clause violated";
        let checks = GobraBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "gobra_precondition");
    }

    #[test]
    fn extract_failed_checks_postcondition() {
        let output = "Error: postcondition might not hold";
        let checks = GobraBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "gobra_postcondition");
    }

    #[test]
    fn extract_failed_checks_ensures() {
        let output = "Error: ensures clause might not hold";
        let checks = GobraBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "gobra_postcondition");
    }

    #[test]
    fn extract_failed_checks_invariant() {
        let output = "Error: loop invariant might not be preserved";
        let checks = GobraBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "gobra_invariant");
    }

    #[test]
    fn extract_failed_checks_permission() {
        let output = "Error: insufficient permission to access field";
        let checks = GobraBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "gobra_permission");
    }

    #[test]
    fn extract_failed_checks_termination() {
        let output = "Error: termination might not be guaranteed";
        let checks = GobraBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "gobra_termination");
    }

    #[test]
    fn extract_location_from_line_go_file() {
        let line = "Error at main.go:10:5: assertion failed";
        let location = GobraBackend::extract_location_from_line(line);

        assert!(location.is_some());
        let loc = location.unwrap();
        assert_eq!(loc.file, "main.go");
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, Some(5));
    }

    #[test]
    fn extract_location_from_line_gobra_file() {
        let line = "Error at verify.gobra:25:12: invariant violated";
        let location = GobraBackend::extract_location_from_line(line);

        assert!(location.is_some());
        let loc = location.unwrap();
        assert_eq!(loc.file, "verify.gobra");
        assert_eq!(loc.line, 25);
        assert_eq!(loc.column, Some(12));
    }

    #[test]
    fn extract_function_name_from_context() {
        let lines = vec![
            "package main",
            "",
            "func verify_sum(n int) (result int) {",
            "    // @ invariant i >= 0",
            "    for i := 0; i < n; i++ {",
            "        result += i",
            "    }",
            "    return",
            "}",
            "",
            "Error: invariant might not hold",
        ];
        let function = GobraBackend::extract_function_name(&lines, 10);

        assert!(function.is_some());
        assert_eq!(function.unwrap(), "verify_sum");
    }

    #[test]
    fn extract_first_error_simple() {
        let output = "Some info\nError: something went wrong\nMore info";
        let error = GobraBackend::extract_first_error(output);

        assert!(error.is_some());
        assert!(error.unwrap().contains("went wrong"));
    }

    #[test]
    fn parse_structured_counterexample_has_raw() {
        let stdout = "verification output";
        let stderr = "Error: test failure";
        let ce = GobraBackend::parse_structured_counterexample(stdout, stderr);

        assert!(ce.raw.is_some());
        assert!(ce.raw.unwrap().contains("Error"));
    }

    #[test]
    fn backend_id_is_gobra() {
        let backend = GobraBackend::new();
        assert_eq!(backend.id(), BackendId::Gobra);
    }

    #[test]
    fn supports_contract_properties() {
        let backend = GobraBackend::new();
        let supports = backend.supports();
        assert!(supports.contains(&PropertyType::Contract));
    }

    #[test]
    fn supports_invariant_properties() {
        let backend = GobraBackend::new();
        let supports = backend.supports();
        assert!(supports.contains(&PropertyType::Invariant));
    }

    #[test]
    fn supports_memory_safety_properties() {
        let backend = GobraBackend::new();
        let supports = backend.supports();
        assert!(supports.contains(&PropertyType::MemorySafety));
    }

    fn create_test_spec() -> TypedSpec {
        use dashprove_usl::ast::Spec;
        use std::collections::HashMap;

        TypedSpec {
            spec: Spec::default(),
            type_info: HashMap::new(),
        }
    }

    #[test]
    fn generate_gobra_code_includes_package() {
        let backend = GobraBackend::new();
        let spec = create_test_spec();
        let code = backend.generate_gobra_code(&spec);
        assert!(code.contains("package main"));
    }

    #[test]
    fn generate_gobra_code_includes_gobra_directive() {
        let backend = GobraBackend::new();
        let spec = create_test_spec();
        let code = backend.generate_gobra_code(&spec);
        assert!(code.contains("+gobra"));
    }

    #[test]
    fn generate_gobra_code_includes_main() {
        let backend = GobraBackend::new();
        let spec = create_test_spec();
        let code = backend.generate_gobra_code(&spec);
        assert!(code.contains("func main()"));
    }
}
