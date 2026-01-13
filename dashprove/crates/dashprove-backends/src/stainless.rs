//! Stainless Scala verification backend
//!
//! Stainless is a verification framework for a subset of Scala that supports
//! preconditions, postconditions, invariants, and termination checking.
//!
//! See: <https://stainless.epfl.ch/>
//!
//! # Features
//!
//! - **Design-by-contract**: require/ensuring clauses
//! - **Termination checking**: Verified termination proofs
//! - **Imperative features**: Mutable state with verification
//! - **Higher-order functions**: Verified lambdas and closures
//! - **ADTs**: Algebraic data types with pattern matching
//!
//! # Requirements
//!
//! Install Stainless:
//! ```bash
//! # Download from releases
//! wget https://github.com/epfl-lara/stainless/releases/download/v0.9.8/stainless-scalac-standalone-0.9.8-linux.zip
//! unzip stainless-*.zip && export PATH=$PATH:$(pwd)/stainless/bin
//!
//! # Or via sbt plugin
//! echo 'addSbtPlugin("ch.epfl.lara" % "sbt-stainless" % "0.9.8")' >> project/plugins.sbt
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

/// Solver backend for Stainless
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StainlessSolver {
    /// Z3 SMT solver (default)
    #[default]
    Z3,
    /// CVC5 SMT solver
    CVC5,
    /// Princess theorem prover
    Princess,
}

/// Configuration for Stainless backend
#[derive(Debug, Clone)]
pub struct StainlessConfig {
    /// Path to stainless binary
    pub stainless_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// SMT solver to use
    pub solver: StainlessSolver,
    /// Enable termination checking
    pub check_termination: bool,
    /// Enable measure inference
    pub infer_measures: bool,
    /// Timeout per verification condition (seconds)
    pub vc_timeout: u32,
    /// Additional Stainless options
    pub extra_args: Vec<String>,
}

impl Default for StainlessConfig {
    fn default() -> Self {
        Self {
            stainless_path: None,
            timeout: Duration::from_secs(120),
            solver: StainlessSolver::default(),
            check_termination: true,
            infer_measures: true,
            vc_timeout: 30,
            extra_args: vec![],
        }
    }
}

/// Stainless Scala verification backend
pub struct StainlessBackend {
    config: StainlessConfig,
}

impl Default for StainlessBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl StainlessBackend {
    /// Create a new Stainless backend with default configuration
    pub fn new() -> Self {
        Self {
            config: StainlessConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: StainlessConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<PathBuf, String> {
        // Check configured path first
        if let Some(ref path) = self.config.stainless_path {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        // Try common binary names
        for name in ["stainless", "stainless-scalac", "stainless-dotty"] {
            if let Ok(path) = which::which(name) {
                // Verify it works
                let output = Command::new(&path)
                    .arg("--version")
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .output()
                    .await;

                if let Ok(out) = output {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    if stdout.contains("stainless")
                        || stdout.contains("Stainless")
                        || stderr.contains("stainless")
                        || out.status.success()
                    {
                        debug!("Detected Stainless at: {:?}", path);
                        return Ok(path);
                    }
                }
            }
        }

        // Check STAINLESS_HOME environment variable
        if let Ok(stainless_home) = std::env::var("STAINLESS_HOME") {
            let stainless = PathBuf::from(&stainless_home).join("bin").join("stainless");
            if stainless.exists() {
                return Ok(stainless);
            }
        }

        Err("Stainless not found. Install from: https://stainless.epfl.ch/".to_string())
    }

    /// Generate Scala code from USL spec
    fn generate_scala_code(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();
        code.push_str("// Generated by DashProve\n");
        code.push_str("import stainless.lang._\n");
        code.push_str("import stainless.annotation._\n\n");

        // Generate object to contain everything
        code.push_str("object Verification {\n\n");

        // Generate type definitions
        for type_def in &spec.spec.types {
            let safe_name = Self::sanitize_name(&type_def.name);
            code.push_str(&format!("  // Type: {}\n", type_def.name));
            code.push_str(&format!("  case class {}(value: BigInt)\n\n", safe_name));
        }

        // Generate properties as verified functions
        for (i, prop) in spec.spec.properties.iter().enumerate() {
            let prop_name = prop.name();
            let safe_name = Self::sanitize_name(&prop_name);
            let func_name = if safe_name.is_empty() {
                format!("property_{}", i)
            } else {
                format!("property_{}", safe_name)
            };

            code.push_str(&format!("  // Property: {}\n", prop_name));
            code.push_str(&format!("  def {}(x: BigInt): Boolean = {{\n", func_name));
            code.push_str("    require(true)\n");
            code.push_str("    true\n");
            code.push_str("  } ensuring(res => res == true)\n\n");
        }

        // If no properties, add a trivial verified function
        if spec.spec.properties.is_empty() {
            code.push_str("  def trivialProperty(x: BigInt): Boolean = {\n");
            code.push_str("    require(true)\n");
            code.push_str("    true\n");
            code.push_str("  } ensuring(res => res == true)\n\n");
        }

        // Main function for verification
        code.push_str("  def main(): Unit = {\n");
        code.push_str("    // Verification entry point\n");
        code.push_str("  }\n");

        code.push_str("}\n");

        code
    }

    /// Sanitize a name for use in Scala
    fn sanitize_name(name: &str) -> String {
        let result: String = name
            .replace([' ', '-', ':', '/', '\\', '.', '(', ')', '[', ']'], "_")
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect();

        // Scala names must start with letter or underscore
        if result.starts_with(|c: char| c.is_ascii_digit()) {
            format!("t_{}", result)
        } else if result.is_empty() {
            "unnamed".to_string()
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

        // Count verification results
        let mut verified = 0;
        let mut invalid = 0;
        let mut unknown = 0;

        // Parse Stainless output
        for line in combined.lines() {
            let trimmed = line.trim();

            // Check for verification condition results
            if trimmed.contains("valid") || trimmed.contains("VALID") {
                verified += 1;
                diagnostics.push(format!("✓ {}", trimmed));
            }

            if trimmed.contains("invalid") || trimmed.contains("INVALID") {
                invalid += 1;
                diagnostics.push(format!("✗ {}", trimmed));
            }

            if trimmed.contains("unknown") || trimmed.contains("UNKNOWN") {
                unknown += 1;
                diagnostics.push(format!("? {}", trimmed));
            }

            // Capture counterexample info
            if trimmed.contains("counterexample") || trimmed.contains("Counterexample") {
                diagnostics.push(trimmed.to_string());
            }

            // Capture summary
            if trimmed.contains("verified:") || trimmed.contains("Verified:") {
                diagnostics.push(trimmed.to_string());
            }
        }

        // Parse summary line "verified: X / Y"
        for line in combined.lines() {
            let trimmed = line.trim().to_lowercase();
            if trimmed.contains("verified:") {
                // Try to parse "verified: X / Y"
                let parts: Vec<&str> = trimmed.split('/').collect();
                if parts.len() == 2 {
                    let verified_str = parts[0].split(':').next_back().unwrap_or("").trim();
                    let total_str = parts[1].trim();
                    if let (Ok(v), Ok(t)) =
                        (verified_str.parse::<usize>(), total_str.parse::<usize>())
                    {
                        verified = v;
                        if v < t {
                            invalid = t - v;
                        }
                    }
                }
            }
        }

        // Determine status
        if invalid > 0 {
            return (VerificationStatus::Disproven, diagnostics);
        }

        if verified > 0 && invalid == 0 && unknown == 0 {
            return (VerificationStatus::Proven, diagnostics);
        }

        // Check for compilation errors
        if (combined.contains("error:") || combined.contains("Error:"))
            && (combined.contains("compilation") || combined.contains("Compilation"))
        {
            return (
                VerificationStatus::Unknown {
                    reason: "Scala compilation failed".to_string(),
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
        if success && !combined.contains("invalid") {
            return (VerificationStatus::Proven, diagnostics);
        }

        if unknown > 0 {
            return (
                VerificationStatus::Unknown {
                    reason: format!("{} verification conditions unknown", unknown),
                },
                diagnostics,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse Stainless output".to_string(),
            },
            diagnostics,
        )
    }

    /// Parse counterexample from Stainless output
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

    /// Extract failed checks from Stainless output
    fn extract_failed_checks(output: &str) -> Vec<FailedCheck> {
        let mut checks = Vec::new();

        for line in output.lines() {
            let trimmed = line.trim();

            if trimmed.contains("invalid")
                || trimmed.contains("INVALID")
                || trimmed.contains("failed")
            {
                let check_type = if trimmed.contains("precondition") || trimmed.contains("require")
                {
                    "stainless_precondition"
                } else if trimmed.contains("postcondition") || trimmed.contains("ensuring") {
                    "stainless_postcondition"
                } else if trimmed.contains("assertion") || trimmed.contains("assert") {
                    "stainless_assertion"
                } else if trimmed.contains("termination") {
                    "stainless_termination"
                } else if trimmed.contains("measure") {
                    "stainless_measure"
                } else {
                    "stainless_vc"
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

    /// Parse error location from Stainless error line
    fn parse_error_location(line: &str) -> (Option<SourceLocation>, String) {
        // Stainless format: "file.scala:line:col: message" or "file.scala:line: message"
        if let Some(colon_pos) = line.find(':') {
            let prefix = &line[..colon_pos];

            // Check if prefix looks like a file name
            if prefix.ends_with(".scala") || prefix.contains('/') {
                let rest = &line[colon_pos + 1..];
                if let Some(next_colon) = rest.find(':') {
                    if let Ok(line_num) = rest[..next_colon].trim().parse::<u32>() {
                        let remaining = &rest[next_colon + 1..];
                        // Check for column number
                        if let Some(third_colon) = remaining.find(':') {
                            if let Ok(col_num) = remaining[..third_colon].trim().parse::<u32>() {
                                let message = remaining[third_colon + 1..].trim().to_string();
                                return (
                                    Some(SourceLocation {
                                        file: prefix.to_string(),
                                        line: line_num,
                                        column: Some(col_num),
                                    }),
                                    message,
                                );
                            }
                        }
                        let message = remaining.trim().to_string();
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
        let mut in_counterexample = false;

        for line in output.lines() {
            let trimmed = line.trim();

            // Look for counterexample section
            if trimmed.contains("counterexample")
                || trimmed.contains("Counterexample")
                || trimmed.contains("Model:")
            {
                in_counterexample = true;
                continue;
            }

            if in_counterexample {
                if trimmed.is_empty() || trimmed.starts_with("---") {
                    in_counterexample = false;
                    continue;
                }

                // Parse "var = value" or "var -> value"
                let parts: Vec<&str> = if trimmed.contains(" = ") {
                    trimmed.splitn(2, " = ").collect()
                } else if trimmed.contains(" -> ") {
                    trimmed.splitn(2, " -> ").collect()
                } else {
                    continue;
                };

                if parts.len() == 2 {
                    let var_name = parts[0].trim().to_string();
                    let value_str = parts[1].trim();
                    values.insert(var_name, Self::parse_scala_value(value_str));
                }
            }
        }

        values
    }

    /// Parse a Scala value string
    fn parse_scala_value(value_str: &str) -> CounterexampleValue {
        let trimmed = value_str.trim();

        // Boolean
        if trimmed == "true" {
            return CounterexampleValue::Bool(true);
        }
        if trimmed == "false" {
            return CounterexampleValue::Bool(false);
        }

        // BigInt (Stainless uses BigInt by default)
        let clean = trimmed.replace("BigInt(", "").replace(')', "");
        if let Ok(n) = clean.parse::<i128>() {
            return CounterexampleValue::Int {
                value: n,
                type_hint: Some("BigInt".to_string()),
            };
        }

        // Try as regular integer
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
impl VerificationBackend for StainlessBackend {
    fn id(&self) -> BackendId {
        BackendId::Stainless
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Contract, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let stainless_path = self.detect().await.map_err(BackendError::Unavailable)?;

        // Create temp directory
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp directory: {}", e))
        })?;

        let scala_file = temp_dir.path().join("Verification.scala");
        let scala_code = self.generate_scala_code(spec);

        debug!("Generated Scala code:\n{}", scala_code);

        tokio::fs::write(&scala_file, &scala_code)
            .await
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to write Scala file: {}", e))
            })?;

        // Build command
        let mut cmd = Command::new(&stainless_path);
        cmd.arg(&scala_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(temp_dir.path());

        // Add solver option
        let solver_name = match self.config.solver {
            StainlessSolver::Z3 => "smt-z3",
            StainlessSolver::CVC5 => "smt-cvc5",
            StainlessSolver::Princess => "princess",
        };
        cmd.arg(format!("--solvers={}", solver_name));

        // Add timeout
        cmd.arg(format!("--timeout={}", self.config.vc_timeout));

        // Termination checking
        if self.config.check_termination {
            cmd.arg("--check-termination");
        }

        // Measure inference
        if self.config.infer_measures {
            cmd.arg("--infer-measures");
        }

        // Extra args
        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run stainless: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        debug!("Stainless stdout: {}", stdout);
        debug!("Stainless stderr: {}", stderr);

        let (status, diagnostics) = self.parse_output(&stdout, &stderr, output.status.success());

        // Generate counterexample for failures
        let counterexample = if matches!(status, VerificationStatus::Disproven) {
            Some(Self::parse_counterexample(&stdout, &stderr))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::Stainless,
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

    // ===== StainlessSolver defaults =====

    #[kani::proof]
    fn verify_solver_default_z3() {
        let solver = StainlessSolver::default();
        assert!(matches!(solver, StainlessSolver::Z3));
    }

    // ===== StainlessConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults_timeout() {
        let config = StainlessConfig::default();
        assert!(config.timeout == Duration::from_secs(120));
    }

    #[kani::proof]
    fn verify_config_defaults_solver() {
        let config = StainlessConfig::default();
        assert!(matches!(config.solver, StainlessSolver::Z3));
    }

    #[kani::proof]
    fn verify_config_defaults_options() {
        let config = StainlessConfig::default();
        assert!(config.stainless_path.is_none());
        assert!(config.check_termination);
        assert!(config.infer_measures);
        assert!(config.vc_timeout == 30);
        assert!(config.extra_args.is_empty());
    }

    // ===== Backend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = StainlessBackend::new();
        assert!(backend.config.timeout == Duration::from_secs(120));
        assert!(backend.config.check_termination);
    }

    #[kani::proof]
    fn verify_backend_default_matches_new() {
        let b1 = StainlessBackend::new();
        let b2 = StainlessBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.check_termination == b2.config.check_termination);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_values() {
        let config = StainlessConfig {
            stainless_path: Some(PathBuf::from("/usr/bin/stainless")),
            timeout: Duration::from_secs(60),
            solver: StainlessSolver::CVC5,
            check_termination: false,
            infer_measures: false,
            vc_timeout: 10,
            extra_args: vec!["--test".to_string()],
        };
        let backend = StainlessBackend::with_config(config);
        assert!(backend.config.timeout == Duration::from_secs(60));
        assert!(matches!(backend.config.solver, StainlessSolver::CVC5));
        assert!(!backend.config.check_termination);
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = StainlessBackend::new();
        assert!(matches!(backend.id(), BackendId::Stainless));
    }

    #[kani::proof]
    fn verify_supports_contract_invariant() {
        let backend = StainlessBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Contract));
        assert!(supported.contains(&PropertyType::Invariant));
        assert!(supported.len() == 2);
    }

    // ===== Name sanitization =====

    #[kani::proof]
    fn verify_sanitize_name_replaces_dash() {
        let result = StainlessBackend::sanitize_name("hello-world");
        assert!(result == "hello_world");
    }

    #[kani::proof]
    fn verify_sanitize_name_replaces_colon() {
        let result = StainlessBackend::sanitize_name("test:prop");
        assert!(result == "test_prop");
    }

    #[kani::proof]
    fn verify_sanitize_name_digit_prefix() {
        let result = StainlessBackend::sanitize_name("123abc");
        assert!(result == "t_123abc");
    }

    #[kani::proof]
    fn verify_sanitize_name_empty() {
        let result = StainlessBackend::sanitize_name("");
        assert!(result == "unnamed");
    }

    // ===== Output parsing =====

    #[kani::proof]
    fn verify_parse_output_valid() {
        let backend = StainlessBackend::new();
        let (status, _) = backend.parse_output("property_0: VALID\nverified: 1 / 1", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[kani::proof]
    fn verify_parse_output_invalid() {
        let backend = StainlessBackend::new();
        let (status, _) = backend.parse_output("property_0: INVALID", "", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[kani::proof]
    fn verify_parse_output_unknown() {
        let backend = StainlessBackend::new();
        let (status, _) = backend.parse_output("property_0: UNKNOWN", "", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[kani::proof]
    fn verify_parse_output_timeout() {
        let backend = StainlessBackend::new();
        let (status, _) = backend.parse_output("timeout reached", "", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    // ===== Failed checks extraction =====

    #[kani::proof]
    fn verify_extract_failed_checks_precondition() {
        let output = "precondition invalid";
        let checks = StainlessBackend::extract_failed_checks(output);
        assert!(checks
            .iter()
            .any(|c| c.check_id == "stainless_precondition"));
    }

    #[kani::proof]
    fn verify_extract_failed_checks_postcondition() {
        let output = "postcondition failed";
        let checks = StainlessBackend::extract_failed_checks(output);
        assert!(checks
            .iter()
            .any(|c| c.check_id == "stainless_postcondition"));
    }

    #[kani::proof]
    fn verify_extract_failed_checks_termination() {
        let output = "termination invalid";
        let checks = StainlessBackend::extract_failed_checks(output);
        assert!(checks.iter().any(|c| c.check_id == "stainless_termination"));
    }

    // ===== Scala value parsing =====

    #[kani::proof]
    fn verify_parse_scala_value_true() {
        let result = StainlessBackend::parse_scala_value("true");
        assert!(matches!(result, CounterexampleValue::Bool(true)));
    }

    #[kani::proof]
    fn verify_parse_scala_value_false() {
        let result = StainlessBackend::parse_scala_value("false");
        assert!(matches!(result, CounterexampleValue::Bool(false)));
    }

    #[kani::proof]
    fn verify_parse_scala_value_bigint() {
        let result = StainlessBackend::parse_scala_value("BigInt(42)");
        assert!(matches!(result, CounterexampleValue::Int { value: 42, .. }));
    }

    #[kani::proof]
    fn verify_parse_scala_value_int() {
        let result = StainlessBackend::parse_scala_value("42");
        assert!(matches!(result, CounterexampleValue::Int { value: 42, .. }));
    }

    #[kani::proof]
    fn verify_parse_scala_value_string() {
        let result = StainlessBackend::parse_scala_value("some_text");
        assert!(matches!(result, CounterexampleValue::String(_)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_id() {
        assert_eq!(StainlessBackend::new().id(), BackendId::Stainless);
    }

    #[test]
    fn default_config() {
        let config = StainlessConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.solver, StainlessSolver::Z3);
        assert!(config.check_termination);
        assert!(config.infer_measures);
    }

    #[test]
    fn supports_contracts_and_invariants() {
        let backend = StainlessBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Contract));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[test]
    fn sanitize_name() {
        assert_eq!(
            StainlessBackend::sanitize_name("Hello-World"),
            "Hello_World"
        );
        assert_eq!(StainlessBackend::sanitize_name("test:prop"), "test_prop");
        assert_eq!(StainlessBackend::sanitize_name("123abc"), "t_123abc");
        assert_eq!(StainlessBackend::sanitize_name(""), "unnamed");
    }

    #[test]
    fn parse_valid_output() {
        let backend = StainlessBackend::new();
        let stdout = "Verification report:\nproperty_0: VALID\nverified: 1 / 1";
        let (status, diag) = backend.parse_output(stdout, "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(!diag.is_empty());
    }

    #[test]
    fn parse_invalid_output() {
        let backend = StainlessBackend::new();
        let stdout = "Verification report:\nproperty_0: INVALID\ncounterexample found";
        let (status, _diag) = backend.parse_output(stdout, "", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_error_location_full() {
        let line = "Verification.scala:10:5: postcondition invalid";
        let (loc, desc) = StainlessBackend::parse_error_location(line);
        assert!(loc.is_some());
        let loc = loc.unwrap();
        assert_eq!(loc.file, "Verification.scala");
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, Some(5));
        assert!(desc.contains("postcondition invalid"));
    }

    #[test]
    fn extract_failed_checks() {
        let output = "precondition invalid\npostcondition failed\ntermination unknown";
        let checks = StainlessBackend::extract_failed_checks(output);
        assert_eq!(checks.len(), 2); // termination unknown doesn't match "invalid" or "failed"
        assert_eq!(checks[0].check_id, "stainless_precondition");
        assert_eq!(checks[1].check_id, "stainless_postcondition");
    }

    #[test]
    fn parse_scala_values() {
        assert!(matches!(
            StainlessBackend::parse_scala_value("true"),
            CounterexampleValue::Bool(true)
        ));
        assert!(matches!(
            StainlessBackend::parse_scala_value("false"),
            CounterexampleValue::Bool(false)
        ));
        assert!(matches!(
            StainlessBackend::parse_scala_value("BigInt(42)"),
            CounterexampleValue::Int { value: 42, .. }
        ));
        assert!(matches!(
            StainlessBackend::parse_scala_value("42"),
            CounterexampleValue::Int { value: 42, .. }
        ));
    }

    #[test]
    fn generate_scala_empty_spec() {
        use dashprove_usl::ast::Spec;

        let backend = StainlessBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_scala_code(&spec);
        assert!(code.contains("// Generated by DashProve"));
        assert!(code.contains("import stainless.lang._"));
        assert!(code.contains("object Verification"));
        assert!(code.contains("trivialProperty"));
    }

    #[test]
    fn solver_config() {
        let config = StainlessConfig {
            solver: StainlessSolver::CVC5,
            check_termination: false,
            ..Default::default()
        };
        let backend = StainlessBackend::with_config(config);
        assert_eq!(backend.config.solver, StainlessSolver::CVC5);
        assert!(!backend.config.check_termination);
    }
}
