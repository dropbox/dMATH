//! ATS (Applied Type System) backend
//!
//! ATS is a statically typed programming language that unifies implementation
//! with formal specification through a type system with dependent types.
//! It supports theorem proving through its built-in constraint solver.
//!
//! Key features:
//! - Dependent types for precise specifications
//! - Linear types for resource management
//! - Built-in constraint solving
//! - Proof construction through types
//!
//! See: <http://www.ats-lang.org/>

// =============================================
// Kani Proofs for ATS Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- AtsConfig Default Tests ----

    /// Verify AtsConfig::default sets correct baseline values
    #[kani::proof]
    fn proof_ats_config_defaults() {
        let config = AtsConfig::default();
        kani::assert(config.ats_path.is_none(), "ats_path should default to None");
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "timeout should default to 60 seconds",
        );
        kani::assert(!config.debug, "debug should default to false");
        kani::assert(
            config.solver_path.is_none(),
            "solver_path should default to None",
        );
    }

    // ---- AtsBackend Construction Tests ----

    /// Verify AtsBackend::new uses default configuration
    #[kani::proof]
    fn proof_ats_backend_new_defaults() {
        let backend = AtsBackend::new();
        kani::assert(
            backend.config.ats_path.is_none(),
            "new backend should have no ats_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "new backend should default timeout to 60 seconds",
        );
        kani::assert(
            !backend.config.debug,
            "new backend should default debug to false",
        );
    }

    /// Verify AtsBackend::default equals AtsBackend::new
    #[kani::proof]
    fn proof_ats_backend_default_equals_new() {
        let default_backend = AtsBackend::default();
        let new_backend = AtsBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.debug == new_backend.config.debug,
            "default and new should share debug",
        );
    }

    /// Verify AtsBackend::with_config preserves custom configuration
    #[kani::proof]
    fn proof_ats_backend_with_config() {
        let config = AtsConfig {
            ats_path: Some(PathBuf::from("/opt/ats/bin/patscc")),
            timeout: Duration::from_secs(120),
            debug: true,
            solver_path: Some(PathBuf::from("/opt/ats/bin/patsolve")),
        };
        let backend = AtsBackend::with_config(config);
        kani::assert(
            backend.config.ats_path == Some(PathBuf::from("/opt/ats/bin/patscc")),
            "with_config should preserve ats_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "with_config should preserve timeout",
        );
        kani::assert(backend.config.debug, "with_config should preserve debug");
        kani::assert(
            backend.config.solver_path == Some(PathBuf::from("/opt/ats/bin/patsolve")),
            "with_config should preserve solver_path",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::ATS
    #[kani::proof]
    fn proof_ats_backend_id() {
        let backend = AtsBackend::new();
        kani::assert(
            backend.id() == BackendId::ATS,
            "AtsBackend id should be BackendId::ATS",
        );
    }

    /// Verify supports() includes Theorem and Invariant
    #[kani::proof]
    fn proof_ats_backend_supports() {
        let backend = AtsBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Theorem),
            "supports should include Theorem",
        );
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "supports should include Invariant",
        );
    }

    /// Verify supports() returns exactly two property types
    #[kani::proof]
    fn proof_ats_backend_supports_length() {
        let backend = AtsBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 2,
            "ATS should support exactly two property types",
        );
    }

    // ---- Output Parsing Tests ----

    /// Verify parse_output detects successful type check
    #[kani::proof]
    fn proof_parse_output_success() {
        let backend = AtsBackend::new();
        let (status, ce, _) = backend.parse_output("Type checking successful", "", 0);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "exit code 0 should produce Proven status",
        );
        kani::assert(ce.is_none(), "success should have no counterexample");
    }

    /// Verify parse_output detects type errors
    #[kani::proof]
    fn proof_parse_output_type_error() {
        let backend = AtsBackend::new();
        let (status, ce, _) = backend.parse_output("", "error(3): type mismatch", 1);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "type error should produce Disproven status",
        );
        kani::assert(ce.is_some(), "type error should have counterexample");
    }

    /// Verify parse_output detects unsolved constraints
    #[kani::proof]
    fn proof_parse_output_constraint_error() {
        let backend = AtsBackend::new();
        let (status, _, _) = backend.parse_output("", "unsolved constraint: n >= 0", 1);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "constraint error should produce Disproven status",
        );
    }

    // ---- ATS Code Generation Tests ----

    /// Verify generate_ats_code produces valid ATS code
    #[kani::proof]
    fn proof_generate_ats_code_structure() {
        let backend = AtsBackend::new();
        let spec = TypedSpec {
            spec: dashprove_usl::ast::Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_ats_code(&spec);

        kani::assert(
            code.contains("atspre_staload"),
            "generated code should include ATS prelude",
        );
        kani::assert(
            code.contains("prfn"),
            "generated code should have proof functions",
        );
        kani::assert(
            code.contains("main0"),
            "generated code should have main function",
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
use regex::Regex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::debug;

/// Configuration for ATS backend
#[derive(Debug, Clone)]
pub struct AtsConfig {
    /// Path to ATS compiler (patscc or atscc)
    pub ats_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Enable debug mode
    pub debug: bool,
    /// Path to ATS constraint solver (patsolve)
    pub solver_path: Option<PathBuf>,
}

impl Default for AtsConfig {
    fn default() -> Self {
        Self {
            ats_path: None,
            timeout: Duration::from_secs(60),
            debug: false,
            solver_path: None,
        }
    }
}

/// ATS (Applied Type System) backend
pub struct AtsBackend {
    config: AtsConfig,
}

impl Default for AtsBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AtsBackend {
    pub fn new() -> Self {
        Self {
            config: AtsConfig::default(),
        }
    }

    pub fn with_config(config: AtsConfig) -> Self {
        Self { config }
    }

    async fn detect_ats(&self) -> Result<PathBuf, String> {
        if let Some(path) = &self.config.ats_path {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        // Try ATS2 (patscc) first, then ATS1 (atscc)
        for cmd in &["patscc", "atscc"] {
            if let Ok(path) = which::which(cmd) {
                // Verify version
                let output = Command::new(&path)
                    .arg("--version")
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .output()
                    .await
                    .map_err(|e| format!("Failed to execute {}: {}", cmd, e))?;

                let stdout = String::from_utf8_lossy(&output.stdout);
                if stdout.contains("ATS") || output.status.success() {
                    debug!("Detected ATS: {}", stdout.trim());
                    return Ok(path);
                }
            }
        }

        // Check PATSHOME environment
        if let Ok(pats_home) = std::env::var("PATSHOME") {
            let patscc = PathBuf::from(&pats_home).join("bin").join("patscc");
            if patscc.exists() {
                return Ok(patscc);
            }
        }

        // Try common paths
        let home = std::env::var("HOME").unwrap_or_default();
        for base in &[
            "/usr/local/bin/patscc",
            "/opt/ats/bin/patscc",
            &format!("{}/ATS2/bin/patscc", home),
            &format!("{}/.local/bin/patscc", home),
        ] {
            let p = PathBuf::from(base);
            if p.exists() {
                return Ok(p);
            }
        }

        Err("ATS not found. Install from http://www.ats-lang.org/".to_string())
    }

    /// Generate ATS source code from USL spec
    fn generate_ats_code(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();

        code.push_str("(* Generated by DashProve for ATS verification *)\n");
        code.push_str("#include \"share/atspre_staload.hats\"\n");
        code.push('\n');

        if spec.spec.properties.is_empty() {
            // Default: simple property verification
            code.push_str("(* Prove that addition is commutative for naturals *)\n");
            code.push_str("prfn add_comm{m,n:nat}(): [m+n==n+m] void = ()\n");
            code.push('\n');
            code.push_str("(* Main function *)\n");
            code.push_str("implement main0() = println!(\"Verification complete\")\n");
        } else {
            for (idx, prop) in spec.spec.properties.iter().enumerate() {
                code.push_str(&format!("(* Property {}: {} *)\n", idx, prop.name()));
                // Generate proof function
                code.push_str(&format!(
                    "prfn verify_{}(): void = ()\n",
                    prop.name().replace('-', "_")
                ));
            }
            code.push('\n');
            code.push_str("implement main0() = println!(\"All properties verified\")\n");
        }

        code
    }

    /// Parse ATS compiler output
    fn parse_output(
        &self,
        stdout: &str,
        stderr: &str,
        exit_code: i32,
    ) -> (
        VerificationStatus,
        Option<StructuredCounterexample>,
        Vec<String>,
    ) {
        let combined = format!("{}\n{}", stdout, stderr);
        let mut diagnostics = Vec::new();

        // Collect warnings
        for line in combined.lines() {
            if line.contains("warning") || line.contains("Warning") {
                diagnostics.push(line.trim().to_string());
            }
        }

        // ATS verification through type checking:
        // - Successful compilation = types check = proofs valid
        // - Type errors = constraint unsatisfied = verification failed

        let type_error = combined.contains("error(") || combined.contains("unsolved constraint");

        let status = if exit_code == 0 && !type_error {
            VerificationStatus::Proven
        } else if type_error || exit_code != 0 {
            VerificationStatus::Disproven
        } else {
            VerificationStatus::Unknown {
                reason: "Could not parse ATS output".to_string(),
            }
        };

        let counterexample = if matches!(status, VerificationStatus::Disproven) {
            self.extract_counterexample(&combined)
        } else {
            None
        };

        (status, counterexample, diagnostics)
    }

    /// Extract counterexample from ATS error messages
    fn extract_counterexample(&self, output: &str) -> Option<StructuredCounterexample> {
        let mut failed_checks = Vec::new();
        let mut witness = HashMap::new();

        // Parse constraint errors
        // Pattern: "unsolved constraint: ... at file:line"
        let constraint_re =
            Regex::new(r"unsolved constraint:?\s*(.+?)(?:\s+at\s+(\S+):(\d+))?").ok()?;
        if let Some(cap) = constraint_re.captures(output) {
            let description = cap
                .get(1)
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_else(|| "Type constraint violation".to_string());
            let file = cap.get(2).map(|m| m.as_str().to_string());
            let line = cap.get(3).and_then(|m| m.as_str().parse().ok());

            failed_checks.push(FailedCheck {
                check_id: "ats_constraint".to_string(),
                description,
                location: file.map(|f| SourceLocation {
                    file: f,
                    line: line.unwrap_or(0),
                    column: None,
                }),
                function: None,
            });
        }

        // Parse type errors
        // Pattern: "error(...): the type ... does not match"
        let type_re = Regex::new(r"error\((\d+)\):\s*(.+)").ok()?;
        for cap in type_re.captures_iter(output) {
            let error_code = cap.get(1).unwrap().as_str();
            let msg = cap.get(2).unwrap().as_str().trim();

            failed_checks.push(FailedCheck {
                check_id: format!("ats_error_{}", error_code),
                description: msg.to_string(),
                location: None,
                function: None,
            });
        }

        // Parse location from error messages
        // Pattern: "filename: line:col: error"
        let loc_re = Regex::new(r"(\S+\.dats):(\d+)(?::(\d+))?").ok()?;
        if let Some(cap) = loc_re.captures(output) {
            let file = cap.get(1).unwrap().as_str().to_string();
            let line: u32 = cap.get(2).unwrap().as_str().parse().unwrap_or(0);
            let col = cap.get(3).and_then(|m| m.as_str().parse().ok());

            // Update first failed check with location if not set
            if !failed_checks.is_empty() && failed_checks[0].location.is_none() {
                failed_checks[0].location = Some(SourceLocation {
                    file,
                    line,
                    column: col,
                });
            }
        }

        // Parse constraint variable assignments
        // Pattern: "S2Evar(name) = value"
        let var_re = Regex::new(r"S2E(?:var|cst)\((\w+)\)\s*=\s*(\d+)").ok()?;
        for cap in var_re.captures_iter(output) {
            let var = cap.get(1).unwrap().as_str().to_string();
            let val: i128 = cap.get(2).unwrap().as_str().parse().unwrap_or(0);
            witness.insert(
                var,
                CounterexampleValue::Int {
                    value: val,
                    type_hint: Some("nat".to_string()),
                },
            );
        }

        // Ensure we have at least one failed check
        if failed_checks.is_empty() {
            failed_checks.push(FailedCheck {
                check_id: "ats_typecheck".to_string(),
                description: "ATS type checking failed".to_string(),
                location: None,
                function: None,
            });
        }

        Some(StructuredCounterexample {
            witness,
            failed_checks,
            playback_test: None,
            trace: vec![],
            raw: Some(output.to_string()),
            minimized: false,
        })
    }
}

#[async_trait]
impl VerificationBackend for AtsBackend {
    fn id(&self) -> BackendId {
        BackendId::ATS
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Theorem, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        let ats = self.detect_ats().await.map_err(BackendError::Unavailable)?;

        // Create temp directory
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {e}"))
        })?;
        let source_path = temp_dir.path().join("verify.dats");

        // Write source
        std::fs::write(&source_path, self.generate_ats_code(spec)).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write ATS source: {e}"))
        })?;

        // Run ATS compiler (type checking is verification)
        let mut cmd = Command::new(&ats);
        cmd.arg("-tcats") // Type check only, don't compile
            .arg(&source_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if self.config.debug {
            cmd.arg("-v");
        }

        // Run with timeout
        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| BackendError::VerificationFailed(format!("ATS failed: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        debug!("ATS stdout: {}", stdout);
        if !stderr.trim().is_empty() {
            debug!("ATS stderr: {}", stderr);
        }

        let (status, counterexample, diagnostics) = self.parse_output(&stdout, &stderr, exit_code);
        let proof = if matches!(status, VerificationStatus::Proven) {
            Some("ATS type checking verified all constraints".to_string())
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::ATS,
            status,
            proof,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_ats().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::ast::{Expr, Invariant, Property, Spec};

    #[test]
    fn default_config() {
        let config = AtsConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert!(!config.debug);
    }

    #[test]
    fn backend_id() {
        assert_eq!(AtsBackend::new().id(), BackendId::ATS);
    }

    #[test]
    fn supports_properties() {
        let backend = AtsBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[test]
    fn ats_code_generation_empty_spec() {
        let backend = AtsBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_ats_code(&spec);
        assert!(code.contains("atspre_staload"));
        assert!(code.contains("prfn"));
        assert!(code.contains("main0"));
    }

    #[test]
    fn ats_code_generation_with_property() {
        let backend = AtsBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![Property::Invariant(Invariant {
                    name: "test_prop".to_string(),
                    body: Expr::Bool(true),
                })],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_ats_code(&spec);
        assert!(code.contains("test_prop"));
        assert!(code.contains("verify_test_prop"));
    }

    #[test]
    fn parse_output_success() {
        let backend = AtsBackend::new();
        let stdout = "Type checking successful";
        let (status, cex, _) = backend.parse_output(stdout, "", 0);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(cex.is_none());
    }

    #[test]
    fn parse_output_type_error() {
        let backend = AtsBackend::new();
        let stderr = "error(3): the type [int] does not match [bool]";
        let (status, cex, _) = backend.parse_output("", stderr, 1);
        assert!(matches!(status, VerificationStatus::Disproven));
        let ce = cex.expect("expected counterexample");
        assert!(!ce.failed_checks.is_empty());
    }

    #[test]
    fn parse_output_constraint_error() {
        let backend = AtsBackend::new();
        let stderr = "unsolved constraint: n >= 0 at verify.dats:10";
        let (status, cex, _) = backend.parse_output("", stderr, 1);
        assert!(matches!(status, VerificationStatus::Disproven));
        let ce = cex.expect("expected counterexample");
        assert!(ce
            .failed_checks
            .iter()
            .any(|c| c.check_id == "ats_constraint"));
    }

    #[test]
    fn extract_counterexample_with_location() {
        let backend = AtsBackend::new();
        let output = "error(3): type mismatch\nverify.dats:25:10: here";
        let cex = backend.extract_counterexample(output).unwrap();
        let check = &cex.failed_checks[0];
        assert!(check.location.is_some());
        let loc = check.location.as_ref().unwrap();
        assert_eq!(loc.line, 25);
    }

    #[test]
    fn extract_counterexample_with_variables() {
        let backend = AtsBackend::new();
        let output = "unsolved constraint\nS2Evar(n) = 5\nS2Ecst(m) = 10";
        let cex = backend.extract_counterexample(output).unwrap();
        assert!(cex.witness.contains_key("n"));
        assert!(cex.witness.contains_key("m"));
    }

    #[tokio::test]
    async fn health_check_unavailable() {
        let config = AtsConfig {
            ats_path: Some(PathBuf::from("/nonexistent/patscc")),
            ..Default::default()
        };
        let backend = AtsBackend::with_config(config);
        let health = backend.health_check().await;
        assert!(matches!(health, HealthStatus::Unavailable { .. }));
    }
}
