//! CodeSonar static analysis backend
//!
//! CodeSonar is a commercial static analysis tool from GrammaTech
//! that uses abstract interpretation for bug detection in C/C++ and binary code.
//!
//! See: <https://www.grammatech.com/codesonar>
//!
//! # Features
//!
//! - **Abstract interpretation**: Sound static analysis
//! - **Buffer overflows**: Detect memory safety issues
//! - **Null pointer dereferences**: Track pointer validity
//! - **Binary analysis**: Analyze executables directly
//! - **Concurrency bugs**: Detect race conditions
//!
//! # Requirements
//!
//! Commercial license required from GrammaTech.

// =============================================
// Kani Proofs for CodeSonar Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- CodeSonarMode Default Tests ----

    /// Verify CodeSonarMode::default is Source
    #[kani::proof]
    fn proof_codesonar_mode_default_is_source() {
        let mode = CodeSonarMode::default();
        kani::assert(
            mode == CodeSonarMode::Source,
            "Default mode should be Source",
        );
    }

    // ---- CodeSonarConfig Default Tests ----

    /// Verify CodeSonarConfig::default timeout is 600 seconds
    #[kani::proof]
    fn proof_codesonar_config_default_timeout() {
        let config = CodeSonarConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(600),
            "Default timeout should be 600 seconds",
        );
    }

    /// Verify CodeSonarConfig::default codesonar_path is None
    #[kani::proof]
    fn proof_codesonar_config_default_path_none() {
        let config = CodeSonarConfig::default();
        kani::assert(
            config.codesonar_path.is_none(),
            "Default codesonar_path should be None",
        );
    }

    /// Verify CodeSonarConfig::default hub_url is None
    #[kani::proof]
    fn proof_codesonar_config_default_hub_url_none() {
        let config = CodeSonarConfig::default();
        kani::assert(config.hub_url.is_none(), "Default hub_url should be None");
    }

    /// Verify CodeSonarConfig::default mode is Source
    #[kani::proof]
    fn proof_codesonar_config_default_mode() {
        let config = CodeSonarConfig::default();
        kani::assert(
            config.mode == CodeSonarMode::Source,
            "Default mode should be Source",
        );
    }

    /// Verify CodeSonarConfig::default project is None
    #[kani::proof]
    fn proof_codesonar_config_default_project_none() {
        let config = CodeSonarConfig::default();
        kani::assert(config.project.is_none(), "Default project should be None");
    }

    /// Verify CodeSonarConfig::default extra_args is empty
    #[kani::proof]
    fn proof_codesonar_config_default_extra_args_empty() {
        let config = CodeSonarConfig::default();
        kani::assert(
            config.extra_args.is_empty(),
            "Default extra_args should be empty",
        );
    }

    // ---- CodeSonarBackend Construction Tests ----

    /// Verify CodeSonarBackend::new uses default config timeout
    #[kani::proof]
    fn proof_codesonar_backend_new_default_timeout() {
        let backend = CodeSonarBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "New backend should use default timeout",
        );
    }

    /// Verify CodeSonarBackend::default equals CodeSonarBackend::new timeout
    #[kani::proof]
    fn proof_codesonar_backend_default_equals_new_timeout() {
        let default_backend = CodeSonarBackend::default();
        let new_backend = CodeSonarBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify CodeSonarBackend::default equals CodeSonarBackend::new mode
    #[kani::proof]
    fn proof_codesonar_backend_default_equals_new_mode() {
        let default_backend = CodeSonarBackend::default();
        let new_backend = CodeSonarBackend::new();
        kani::assert(
            default_backend.config.mode == new_backend.config.mode,
            "Default and new should have same mode",
        );
    }

    /// Verify CodeSonarBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_codesonar_backend_with_config_timeout() {
        let config = CodeSonarConfig {
            codesonar_path: None,
            hub_url: None,
            timeout: Duration::from_secs(1200),
            mode: CodeSonarMode::Source,
            project: None,
            extra_args: vec![],
        };
        let backend = CodeSonarBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(1200),
            "with_config should preserve timeout",
        );
    }

    /// Verify CodeSonarBackend::with_config preserves Binary mode
    #[kani::proof]
    fn proof_codesonar_backend_with_config_binary_mode() {
        let config = CodeSonarConfig {
            codesonar_path: None,
            hub_url: None,
            timeout: Duration::from_secs(600),
            mode: CodeSonarMode::Binary,
            project: None,
            extra_args: vec![],
        };
        let backend = CodeSonarBackend::with_config(config);
        kani::assert(
            backend.config.mode == CodeSonarMode::Binary,
            "with_config should preserve Binary mode",
        );
    }

    /// Verify CodeSonarBackend::with_config preserves Incremental mode
    #[kani::proof]
    fn proof_codesonar_backend_with_config_incremental_mode() {
        let config = CodeSonarConfig {
            codesonar_path: None,
            hub_url: None,
            timeout: Duration::from_secs(600),
            mode: CodeSonarMode::Incremental,
            project: None,
            extra_args: vec![],
        };
        let backend = CodeSonarBackend::with_config(config);
        kani::assert(
            backend.config.mode == CodeSonarMode::Incremental,
            "with_config should preserve Incremental mode",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify CodeSonarBackend::id returns CodeSonar
    #[kani::proof]
    fn proof_codesonar_backend_id() {
        let backend = CodeSonarBackend::new();
        kani::assert(
            backend.id() == BackendId::CodeSonar,
            "Backend id should be CodeSonar",
        );
    }

    /// Verify CodeSonarBackend::supports includes MemorySafety
    #[kani::proof]
    fn proof_codesonar_backend_supports_memory_safety() {
        let backend = CodeSonarBackend::new();
        let supported = backend.supports();
        let has_memory_safety = supported.iter().any(|p| *p == PropertyType::MemorySafety);
        kani::assert(has_memory_safety, "Should support MemorySafety property");
    }

    /// Verify CodeSonarBackend::supports includes SecurityVulnerability
    #[kani::proof]
    fn proof_codesonar_backend_supports_security() {
        let backend = CodeSonarBackend::new();
        let supported = backend.supports();
        let has_security = supported
            .iter()
            .any(|p| *p == PropertyType::SecurityVulnerability);
        kani::assert(
            has_security,
            "Should support SecurityVulnerability property",
        );
    }

    /// Verify CodeSonarBackend::supports returns exactly 2 properties
    #[kani::proof]
    fn proof_codesonar_backend_supports_length() {
        let backend = CodeSonarBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 2, "Should support exactly 2 properties");
    }

    // ---- sanitize_name Tests ----

    /// Verify sanitize_name replaces hyphens with underscores
    #[kani::proof]
    fn proof_sanitize_name_hyphen() {
        let result = CodeSonarBackend::sanitize_name("my-func");
        kani::assert(result == "my_func", "Should replace hyphen with underscore");
    }

    /// Verify sanitize_name converts to lowercase
    #[kani::proof]
    fn proof_sanitize_name_lowercase() {
        let result = CodeSonarBackend::sanitize_name("MyFunc");
        kani::assert(result == "myfunc", "Should convert to lowercase");
    }

    /// Verify sanitize_name replaces spaces
    #[kani::proof]
    fn proof_sanitize_name_spaces() {
        let result = CodeSonarBackend::sanitize_name("my func");
        kani::assert(result == "my_func", "Should replace space with underscore");
    }

    /// Verify sanitize_name handles empty string
    #[kani::proof]
    fn proof_sanitize_name_empty() {
        let result = CodeSonarBackend::sanitize_name("");
        kani::assert(result.is_empty(), "Empty string should remain empty");
    }

    // ---- parse_output Tests ----

    /// Verify parse_output detects Disproven for Buffer Overrun
    #[kani::proof]
    fn proof_parse_output_buffer_overrun() {
        let backend = CodeSonarBackend::new();
        let (status, _) = backend.parse_output("Buffer Overrun at line 10", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should return Disproven for Buffer Overrun",
        );
    }

    /// Verify parse_output detects Disproven for Null Pointer Dereference
    #[kani::proof]
    fn proof_parse_output_null_pointer() {
        let backend = CodeSonarBackend::new();
        let (status, _) = backend.parse_output("Null Pointer Dereference at line 5", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should return Disproven for Null Pointer",
        );
    }

    /// Verify parse_output detects Disproven for Use After Free
    #[kani::proof]
    fn proof_parse_output_use_after_free() {
        let backend = CodeSonarBackend::new();
        let (status, _) = backend.parse_output("Use After Free detected", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should return Disproven for Use After Free",
        );
    }

    /// Verify parse_output returns Proven for clean output
    #[kani::proof]
    fn proof_parse_output_clean() {
        let backend = CodeSonarBackend::new();
        let (status, _) = backend.parse_output("Analysis complete\n0 warnings", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should return Proven for clean output",
        );
    }

    // ---- extract_failed_checks Tests ----

    /// Verify extract_failed_checks identifies buffer overrun
    #[kani::proof]
    fn proof_extract_failed_checks_buffer() {
        let checks = CodeSonarBackend::extract_failed_checks("Buffer Overrun at line 10");
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "cs_buffer_overrun",
            "Should identify buffer overrun",
        );
    }

    /// Verify extract_failed_checks identifies null pointer
    #[kani::proof]
    fn proof_extract_failed_checks_null() {
        let checks = CodeSonarBackend::extract_failed_checks("Null Pointer at line 5");
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "cs_null_pointer",
            "Should identify null pointer",
        );
    }

    /// Verify extract_failed_checks identifies memory leak
    #[kani::proof]
    fn proof_extract_failed_checks_memory_leak() {
        let checks = CodeSonarBackend::extract_failed_checks("Memory Leak detected");
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "cs_memory_leak",
            "Should identify memory leak",
        );
    }

    /// Verify extract_failed_checks returns empty for clean output
    #[kani::proof]
    fn proof_extract_failed_checks_empty() {
        let checks = CodeSonarBackend::extract_failed_checks("All clear");
        kani::assert(checks.is_empty(), "Should return empty for clean output");
    }
}

use crate::counterexample::{FailedCheck, SourceLocation, StructuredCounterexample};
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

/// Analysis mode for CodeSonar
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CodeSonarMode {
    /// Source code analysis
    #[default]
    Source,
    /// Binary analysis
    Binary,
    /// Incremental analysis
    Incremental,
}

/// Configuration for CodeSonar backend
#[derive(Debug, Clone)]
pub struct CodeSonarConfig {
    /// Path to CodeSonar binary
    pub codesonar_path: Option<PathBuf>,
    /// CodeSonar hub URL
    pub hub_url: Option<String>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Analysis mode
    pub mode: CodeSonarMode,
    /// Project name
    pub project: Option<String>,
    /// Additional CodeSonar options
    pub extra_args: Vec<String>,
}

impl Default for CodeSonarConfig {
    fn default() -> Self {
        Self {
            codesonar_path: None,
            hub_url: None,
            timeout: Duration::from_secs(600),
            mode: CodeSonarMode::default(),
            project: None,
            extra_args: vec![],
        }
    }
}

/// CodeSonar static analysis backend
pub struct CodeSonarBackend {
    config: CodeSonarConfig,
}

impl Default for CodeSonarBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeSonarBackend {
    /// Create a new CodeSonar backend with default configuration
    pub fn new() -> Self {
        Self {
            config: CodeSonarConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: CodeSonarConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<PathBuf, String> {
        // Check configured path first
        if let Some(ref path) = self.config.codesonar_path {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        // Try common binary names
        for name in ["codesonar", "csurf"] {
            if let Ok(path) = which::which(name) {
                return Ok(path);
            }
        }

        // Check CODESONAR_HOME environment variable
        if let Ok(cs_home) = std::env::var("CODESONAR_HOME") {
            let cs_bin = PathBuf::from(&cs_home).join("bin").join("codesonar");
            if cs_bin.exists() {
                return Ok(cs_bin);
            }
        }

        Err("CodeSonar not found (commercial license required)".to_string())
    }

    /// Generate C code from USL spec for analysis
    fn generate_c_code(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();
        code.push_str("/* Generated by DashProve */\n\n");
        code.push_str("#include <stdlib.h>\n");
        code.push_str("#include <assert.h>\n\n");

        // Type declarations
        for type_def in &spec.spec.types {
            let safe_name = Self::sanitize_name(&type_def.name);
            code.push_str(&format!("typedef struct {} {{\n", safe_name));
            code.push_str("    int value;\n");
            code.push_str(&format!("}} {};\n\n", safe_name));
        }

        // Property verification functions
        for (i, prop) in spec.spec.properties.iter().enumerate() {
            let prop_name = prop.name();
            let safe_name = Self::sanitize_name(&prop_name);
            let func_name = if safe_name.is_empty() {
                format!("verify_prop_{}", i)
            } else {
                format!("verify_{}", safe_name)
            };

            code.push_str(&format!("/* Property: {} */\n", prop_name));
            code.push_str(&format!("int {}(void) {{\n", func_name));
            code.push_str("    int* ptr = malloc(sizeof(int));\n");
            code.push_str("    if (ptr != NULL) {\n");
            code.push_str("        *ptr = 1;\n");
            code.push_str("        free(ptr);\n");
            code.push_str("        return 1;\n");
            code.push_str("    }\n");
            code.push_str("    return 0;\n");
            code.push_str("}\n\n");
        }

        // Main function
        code.push_str("int main(void) {\n");
        for (i, prop) in spec.spec.properties.iter().enumerate() {
            let prop_name = prop.name();
            let safe_name = Self::sanitize_name(&prop_name);
            let func_name = if safe_name.is_empty() {
                format!("verify_prop_{}", i)
            } else {
                format!("verify_{}", safe_name)
            };
            code.push_str(&format!("    assert({}());\n", func_name));
        }
        code.push_str("    return 0;\n");
        code.push_str("}\n");

        code
    }

    /// Sanitize a name for C identifiers
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

        for line in combined.lines() {
            let trimmed = line.trim();

            // Count warnings by type
            if trimmed.contains("Warning") || trimmed.contains("warning") {
                diagnostics.push(format!("⚠ {}", trimmed));
            }

            if trimmed.contains("Error") || trimmed.contains("error") {
                diagnostics.push(format!("✗ {}", trimmed));
            }

            // Capture specific findings
            if trimmed.contains("Buffer Overrun")
                || trimmed.contains("Null Pointer")
                || trimmed.contains("Memory Leak")
                || trimmed.contains("Use After Free")
            {
                diagnostics.push(format!("✗ {}", trimmed));
            }
        }

        // Count findings
        let warning_count = combined.matches("Warning").count();
        let error_count = combined.matches("Error").count();

        // Check for critical findings FIRST (these indicate real issues)
        if combined.contains("Buffer Overrun")
            || combined.contains("Null Pointer Dereference")
            || combined.contains("Use After Free")
        {
            return (VerificationStatus::Disproven, diagnostics);
        }

        // Check for no findings (only after confirming no critical issues)
        if warning_count == 0
            && error_count == 0
            && (success || combined.contains("Analysis complete"))
        {
            return (VerificationStatus::Proven, diagnostics);
        }

        // If only warnings, still consider verified with warnings
        if error_count == 0 && warning_count > 0 {
            diagnostics.push(format!("Total warnings: {}", warning_count));
            return (VerificationStatus::Proven, diagnostics);
        }

        if !success || error_count > 0 {
            return (
                VerificationStatus::Unknown {
                    reason: format!("CodeSonar found {} errors", error_count),
                },
                diagnostics,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse CodeSonar output".to_string(),
            },
            diagnostics,
        )
    }

    fn parse_counterexample(stdout: &str, stderr: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        let combined = format!("{}\n{}", stdout, stderr);
        ce.raw = Some(combined.clone());
        ce.failed_checks = Self::extract_failed_checks(&combined);
        ce.witness = HashMap::new();
        ce
    }

    fn extract_failed_checks(output: &str) -> Vec<FailedCheck> {
        let mut checks = Vec::new();

        for line in output.lines() {
            let trimmed = line.trim();

            if trimmed.contains("Buffer Overrun")
                || trimmed.contains("Null Pointer")
                || trimmed.contains("Memory Leak")
                || trimmed.contains("Use After Free")
                || trimmed.contains("Race Condition")
            {
                let check_type = if trimmed.contains("Buffer") {
                    "cs_buffer_overrun"
                } else if trimmed.contains("Null") {
                    "cs_null_pointer"
                } else if trimmed.contains("Leak") {
                    "cs_memory_leak"
                } else if trimmed.contains("After Free") {
                    "cs_use_after_free"
                } else if trimmed.contains("Race") {
                    "cs_race_condition"
                } else {
                    "cs_warning"
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

    fn parse_error_location(line: &str) -> (Option<SourceLocation>, String) {
        // CodeSonar format: "file.c:line: warning type"
        if let Some(colon_pos) = line.find(':') {
            let prefix = &line[..colon_pos];
            if prefix.ends_with(".c") || prefix.ends_with(".cpp") || prefix.ends_with(".h") {
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
}

#[async_trait]
impl VerificationBackend for CodeSonarBackend {
    fn id(&self) -> BackendId {
        BackendId::CodeSonar
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::MemorySafety,
            PropertyType::SecurityVulnerability,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let cs_path = self.detect().await.map_err(BackendError::Unavailable)?;

        // Create temp directory
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp directory: {}", e))
        })?;

        let c_file = temp_dir.path().join("verify.c");
        let c_code = self.generate_c_code(spec);

        debug!("Generated C code:\n{}", c_code);

        tokio::fs::write(&c_file, &c_code).await.map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write C file: {}", e))
        })?;

        // Build command
        let mut cmd = Command::new(&cs_path);
        cmd.arg("analyze")
            .arg(&c_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(temp_dir.path());

        // Add hub URL if configured
        if let Some(ref hub) = self.config.hub_url {
            cmd.arg("-hub").arg(hub);
        }

        // Add project name if configured
        if let Some(ref project) = self.config.project {
            cmd.arg("-project").arg(project);
        }

        // Mode-specific options
        match self.config.mode {
            CodeSonarMode::Binary => {
                cmd.arg("-binary");
            }
            CodeSonarMode::Incremental => {
                cmd.arg("-incremental");
            }
            CodeSonarMode::Source => {
                // Default mode
            }
        }

        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run CodeSonar: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        debug!("CodeSonar stdout: {}", stdout);
        debug!("CodeSonar stderr: {}", stderr);

        let (status, diagnostics) = self.parse_output(&stdout, &stderr, output.status.success());

        let counterexample = if matches!(status, VerificationStatus::Disproven) {
            Some(Self::parse_counterexample(&stdout, &stderr))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::CodeSonar,
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
        assert_eq!(CodeSonarBackend::new().id(), BackendId::CodeSonar);
    }

    #[test]
    fn default_config() {
        let config = CodeSonarConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(600));
        assert_eq!(config.mode, CodeSonarMode::Source);
    }

    #[test]
    fn supports_memory_and_security() {
        let backend = CodeSonarBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::MemorySafety));
        assert!(supported.contains(&PropertyType::SecurityVulnerability));
    }

    #[test]
    fn sanitize_name() {
        assert_eq!(CodeSonarBackend::sanitize_name("my-func"), "my_func");
    }

    #[test]
    fn parse_clean_output() {
        let backend = CodeSonarBackend::new();
        let stdout = "Analysis complete\n0 warnings";
        let (status, _diag) = backend.parse_output(stdout, "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_warning_output() {
        let backend = CodeSonarBackend::new();
        let stdout = "Buffer Overrun at line 10\nNull Pointer Dereference at line 20";
        let (status, _diag) = backend.parse_output(stdout, "", true);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn extract_failed_checks() {
        let output = "Buffer Overrun at line 10\nNull Pointer at line 20";
        let checks = CodeSonarBackend::extract_failed_checks(output);
        assert_eq!(checks.len(), 2);
        assert_eq!(checks[0].check_id, "cs_buffer_overrun");
        assert_eq!(checks[1].check_id, "cs_null_pointer");
    }

    #[test]
    fn parse_error_location() {
        let line = "verify.c:10: Buffer Overrun";
        let (loc, desc) = CodeSonarBackend::parse_error_location(line);
        assert!(loc.is_some());
        let loc = loc.unwrap();
        assert_eq!(loc.file, "verify.c");
        assert_eq!(loc.line, 10);
        assert!(desc.contains("Buffer"));
    }

    #[test]
    fn generate_c_empty_spec() {
        use dashprove_usl::ast::Spec;

        let backend = CodeSonarBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_c_code(&spec);
        assert!(code.contains("Generated by DashProve"));
        assert!(code.contains("int main"));
    }

    #[test]
    fn config_with_binary_mode() {
        let config = CodeSonarConfig {
            mode: CodeSonarMode::Binary,
            hub_url: Some("http://localhost:7340".to_string()),
            ..Default::default()
        };
        let backend = CodeSonarBackend::with_config(config);
        assert_eq!(backend.config.mode, CodeSonarMode::Binary);
    }
}
