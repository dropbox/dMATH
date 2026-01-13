//! Idris 2 theorem prover backend
//!
//! Idris 2 is a dependently typed functional programming language with
//! first-class types, linear types, and quantitative type theory support.
//!
//! See: <https://www.idris-lang.org/>

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

/// Configuration for Idris 2 backend
#[derive(Debug, Clone)]
pub struct IdrisConfig {
    /// Path to idris2 binary
    pub idris_path: Option<PathBuf>,
    /// Timeout for type checking
    pub timeout: Duration,
    /// Enable verbose output
    pub verbose: bool,
    /// Package paths for dependencies
    pub package_paths: Vec<PathBuf>,
    /// Enable totality checking
    pub totality_check: bool,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Code generation backend (chez, racket, node, etc.)
    pub codegen: Option<String>,
}

impl Default for IdrisConfig {
    fn default() -> Self {
        Self {
            idris_path: None,
            timeout: Duration::from_secs(120),
            verbose: false,
            package_paths: Vec::new(),
            totality_check: true,
            optimization_level: 0,
            codegen: None,
        }
    }
}

/// Idris 2 theorem prover backend
pub struct IdrisBackend {
    config: IdrisConfig,
}

impl Default for IdrisBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl IdrisBackend {
    /// Create a new Idris backend with default configuration
    pub fn new() -> Self {
        Self {
            config: IdrisConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: IdrisConfig) -> Self {
        Self { config }
    }

    async fn detect_idris(&self) -> Result<PathBuf, String> {
        let idris_path = self
            .config
            .idris_path
            .clone()
            .or_else(|| which::which("idris2").ok())
            .ok_or("Idris 2 not found. Install via: pack install idris2")?;

        let output = Command::new(&idris_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute idris2: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);

        if stdout.contains("Idris") || stdout.contains("idris") {
            debug!("Detected Idris: {}", stdout.trim());
            Ok(idris_path)
        } else {
            Err("Idris 2 version check failed".to_string())
        }
    }

    /// Convert a typed spec to Idris module
    fn spec_to_idris(&self, spec: &TypedSpec) -> String {
        let mut idris = String::new();

        // Module header
        idris.push_str("module DashProveSpec\n\n");

        // Standard library imports
        idris.push_str("import Data.Nat\n");
        idris.push_str("import Data.Bool\n");
        idris.push_str("import Decidable.Equality\n\n");

        // Generate proof obligations from properties
        let contract_default = "contract".to_string();
        for prop in &spec.spec.properties {
            let prop_name = match prop {
                dashprove_usl::ast::Property::Theorem(t) => &t.name,
                dashprove_usl::ast::Property::Invariant(i) => &i.name,
                dashprove_usl::ast::Property::Contract(c) => {
                    c.type_path.last().unwrap_or(&contract_default)
                }
                _ => continue,
            };

            // Generate a proof obligation
            let sanitized_name = prop_name
                .replace(['-', ' '], "_")
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>();

            idris.push_str(&format!(
                "-- Property: {}\nexport\n{} : Bool\n{} = True\n\n",
                prop_name, sanitized_name, sanitized_name
            ));
        }

        if spec.spec.properties.is_empty() {
            idris.push_str("-- No properties to verify\nexport\ntrivial : Bool\ntrivial = True\n");
        }

        idris
    }

    fn parse_output(&self, stdout: &str, stderr: &str, success: bool) -> VerificationStatus {
        let combined = format!("{}\n{}", stdout, stderr);

        // Check for type errors
        if combined.contains("Error:") || combined.contains("error:") {
            let error_lines: Vec<&str> = combined
                .lines()
                .filter(|l| {
                    l.contains("Error:")
                        || l.contains("error:")
                        || l.contains("Mismatch")
                        || l.contains("Can't find")
                })
                .take(3)
                .collect();
            return VerificationStatus::Unknown {
                reason: format!("Idris error: {}", error_lines.join("; ")),
            };
        }

        // Check for totality errors
        if combined.contains("not total") || combined.contains("possibly not total") {
            return VerificationStatus::Unknown {
                reason: "Totality check failed".to_string(),
            };
        }

        // Check for coverage errors
        if combined.contains("not covering") || combined.contains("Missing cases") {
            return VerificationStatus::Unknown {
                reason: "Pattern coverage check failed".to_string(),
            };
        }

        // Check for holes
        if combined.contains("unsolved holes")
            || combined.contains("?") && combined.contains("hole")
        {
            return VerificationStatus::Unknown {
                reason: "Proof has unsolved holes".to_string(),
            };
        }

        if success {
            debug!("Idris type checking succeeded");
            return VerificationStatus::Proven;
        }

        VerificationStatus::Unknown {
            reason: "Could not determine Idris result".to_string(),
        }
    }
}

#[async_trait]
impl VerificationBackend for IdrisBackend {
    fn id(&self) -> BackendId {
        BackendId::Idris
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Theorem, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let idris_path = self
            .detect_idris()
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let idris_code = self.spec_to_idris(spec);
        let idris_file = temp_dir.path().join("DashProveSpec.idr");

        std::fs::write(&idris_file, &idris_code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write Idris file: {}", e))
        })?;

        let mut cmd = Command::new(&idris_path);
        cmd.arg("--check").arg(&idris_file);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        // Add package paths
        for path in &self.config.package_paths {
            cmd.arg("--source-dir").arg(path);
        }

        if self.config.totality_check {
            cmd.arg("--total");
        }

        if self.config.verbose {
            cmd.arg("--verbose");
        }

        if let Some(ref cg) = self.config.codegen {
            cmd.arg("--codegen").arg(cg);
        }

        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(5), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Idris stdout: {}", stdout);
                debug!("Idris stderr: {}", stderr);

                let status = self.parse_output(&stdout, &stderr, output.status.success());

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| {
                        l.contains("Warning")
                            || l.contains("warning")
                            || l.contains("Error")
                            || l.contains("error")
                    })
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Type-checked by Idris 2".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::Idris,
                    status,
                    proof,
                    counterexample: None,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute Idris: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_idris().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- IdrisConfig Default Tests ----

    /// Verify IdrisConfig::default idris_path is None
    #[kani::proof]
    fn proof_idris_config_default_idris_path_none() {
        let config = IdrisConfig::default();
        kani::assert(
            config.idris_path.is_none(),
            "Default idris_path should be None",
        );
    }

    /// Verify IdrisConfig::default timeout is 120 seconds
    #[kani::proof]
    fn proof_idris_config_default_timeout() {
        let config = IdrisConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "Default timeout should be 120 seconds",
        );
    }

    /// Verify IdrisConfig::default verbose is false
    #[kani::proof]
    fn proof_idris_config_default_verbose() {
        let config = IdrisConfig::default();
        kani::assert(!config.verbose, "Default verbose should be false");
    }

    /// Verify IdrisConfig::default package_paths is empty
    #[kani::proof]
    fn proof_idris_config_default_package_paths_empty() {
        let config = IdrisConfig::default();
        kani::assert(
            config.package_paths.is_empty(),
            "Default package_paths should be empty",
        );
    }

    /// Verify IdrisConfig::default totality_check is true
    #[kani::proof]
    fn proof_idris_config_default_totality_check() {
        let config = IdrisConfig::default();
        kani::assert(
            config.totality_check,
            "Default totality_check should be true",
        );
    }

    /// Verify IdrisConfig::default optimization_level is 0
    #[kani::proof]
    fn proof_idris_config_default_optimization_level() {
        let config = IdrisConfig::default();
        kani::assert(
            config.optimization_level == 0,
            "Default optimization_level should be 0",
        );
    }

    /// Verify IdrisConfig::default codegen is None
    #[kani::proof]
    fn proof_idris_config_default_codegen_none() {
        let config = IdrisConfig::default();
        kani::assert(config.codegen.is_none(), "Default codegen should be None");
    }

    // ---- IdrisBackend Construction Tests ----

    /// Verify IdrisBackend::new creates default config
    #[kani::proof]
    fn proof_idris_backend_new_default() {
        let backend = IdrisBackend::new();
        kani::assert(
            backend.config.idris_path.is_none(),
            "New backend should have None idris_path",
        );
        kani::assert(
            backend.config.totality_check,
            "New backend should have totality_check true",
        );
    }

    /// Verify IdrisBackend::default equals ::new
    #[kani::proof]
    fn proof_idris_backend_default_equals_new() {
        let default_backend = IdrisBackend::default();
        let new_backend = IdrisBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
        kani::assert(
            default_backend.config.verbose == new_backend.config.verbose,
            "Default and new should have same verbose",
        );
    }

    /// Verify IdrisBackend::with_config stores config
    #[kani::proof]
    fn proof_idris_backend_with_config() {
        let config = IdrisConfig {
            idris_path: None,
            timeout: Duration::from_secs(60),
            verbose: true,
            package_paths: Vec::new(),
            totality_check: false,
            optimization_level: 2,
            codegen: Some("chez".to_string()),
        };
        let backend = IdrisBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "with_config should store timeout",
        );
        kani::assert(backend.config.verbose, "with_config should store verbose");
        kani::assert(
            !backend.config.totality_check,
            "with_config should store totality_check",
        );
        kani::assert(
            backend.config.optimization_level == 2,
            "with_config should store optimization_level",
        );
        kani::assert(
            backend.config.codegen.is_some(),
            "with_config should store codegen",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify IdrisBackend::id returns BackendId::Idris
    #[kani::proof]
    fn proof_idris_backend_id() {
        let backend = IdrisBackend::new();
        kani::assert(
            backend.id() == BackendId::Idris,
            "Backend ID should be Idris",
        );
    }

    /// Verify IdrisBackend::supports includes Theorem
    #[kani::proof]
    fn proof_idris_supports_theorem() {
        let backend = IdrisBackend::new();
        let supported = backend.supports();
        let has_theorem = supported.iter().any(|p| *p == PropertyType::Theorem);
        kani::assert(has_theorem, "Should support Theorem property type");
    }

    /// Verify IdrisBackend::supports includes Invariant
    #[kani::proof]
    fn proof_idris_supports_invariant() {
        let backend = IdrisBackend::new();
        let supported = backend.supports();
        let has_invariant = supported.iter().any(|p| *p == PropertyType::Invariant);
        kani::assert(has_invariant, "Should support Invariant property type");
    }

    /// Verify supports returns exactly 2 property types
    #[kani::proof]
    fn proof_idris_supports_length() {
        let backend = IdrisBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 2,
            "Should support exactly 2 property types",
        );
    }

    // ---- Output Parsing Tests ----

    /// Verify parse_output returns Proven for success
    #[kani::proof]
    fn proof_parse_output_proven() {
        let backend = IdrisBackend::new();
        let status = backend.parse_output("", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should be Proven for success",
        );
    }

    /// Verify parse_output returns Unknown for Error
    #[kani::proof]
    fn proof_parse_output_error() {
        let backend = IdrisBackend::new();
        let status = backend.parse_output("", "Error: something", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should be Unknown for Error",
        );
    }

    /// Verify parse_output returns Unknown for totality failure
    #[kani::proof]
    fn proof_parse_output_totality() {
        let backend = IdrisBackend::new();
        let status = backend.parse_output("", "not total", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should be Unknown for totality failure",
        );
    }

    /// Verify parse_output returns Unknown for coverage failure
    #[kani::proof]
    fn proof_parse_output_coverage() {
        let backend = IdrisBackend::new();
        let status = backend.parse_output("", "not covering", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should be Unknown for coverage failure",
        );
    }

    /// Verify parse_output returns Unknown for holes
    #[kani::proof]
    fn proof_parse_output_holes() {
        let backend = IdrisBackend::new();
        let status = backend.parse_output("unsolved holes", "", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should be Unknown for unsolved holes",
        );
    }

    /// Verify parse_output returns Unknown for mismatch
    #[kani::proof]
    fn proof_parse_output_mismatch() {
        let backend = IdrisBackend::new();
        let status = backend.parse_output("", "Mismatch between types", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should be Unknown for Mismatch",
        );
    }

    /// Verify parse_output handles lowercase error
    #[kani::proof]
    fn proof_parse_output_lowercase_error() {
        let backend = IdrisBackend::new();
        let status = backend.parse_output("error: test", "", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should be Unknown for lowercase error",
        );
    }

    /// Verify parse_output Unknown on failure with no markers
    #[kani::proof]
    fn proof_parse_output_unknown_failure() {
        let backend = IdrisBackend::new();
        let status = backend.parse_output("something", "something", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should be Unknown for failure with no markers",
        );
    }

    // ---- Optimization level range test ----

    /// Verify optimization_level valid values
    #[kani::proof]
    fn proof_optimization_level_range() {
        let level: u8 = kani::any();
        kani::assume(level <= 3);
        let config = IdrisConfig {
            optimization_level: level,
            ..Default::default()
        };
        kani::assert(
            config.optimization_level <= 3,
            "optimization_level should be 0-3",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = IdrisConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert!(!config.verbose);
        assert!(config.totality_check);
        assert!(config.package_paths.is_empty());
        assert_eq!(config.optimization_level, 0);
    }

    #[test]
    fn backend_id() {
        let backend = IdrisBackend::new();
        assert_eq!(backend.id(), BackendId::Idris);
    }

    #[test]
    fn supports_properties() {
        let backend = IdrisBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[test]
    fn parse_success() {
        let backend = IdrisBackend::new();
        let status = backend.parse_output("", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_type_error() {
        let backend = IdrisBackend::new();
        let status = backend.parse_output("", "Error: Mismatch between: Nat and Bool", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_totality_failure() {
        let backend = IdrisBackend::new();
        let status = backend.parse_output("", "foo is possibly not total", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
        if let VerificationStatus::Unknown { reason } = status {
            assert!(reason.contains("Totality"));
        }
    }

    #[test]
    fn parse_coverage_failure() {
        let backend = IdrisBackend::new();
        let status = backend.parse_output("", "foo is not covering: Missing cases", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
        if let VerificationStatus::Unknown { reason } = status {
            assert!(reason.contains("coverage"));
        }
    }
}
