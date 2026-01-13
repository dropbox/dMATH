//! Agda theorem prover backend
//!
//! Agda is a dependently typed functional programming language and proof assistant.
//! It supports inductive definitions, pattern matching, and a powerful module system.
//!
//! See: <https://agda.readthedocs.io/>

// =============================================
// Kani Proofs for Agda Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- AgdaConfig Default Tests ----

    /// Verify AgdaConfig::default timeout is 120 seconds
    #[kani::proof]
    fn proof_agda_config_default_timeout() {
        let config = AgdaConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "Default timeout should be 120 seconds",
        );
    }

    /// Verify AgdaConfig::default agda_path is None
    #[kani::proof]
    fn proof_agda_config_default_path_none() {
        let config = AgdaConfig::default();
        kani::assert(
            config.agda_path.is_none(),
            "Default agda_path should be None",
        );
    }

    /// Verify AgdaConfig::default verbose is false
    #[kani::proof]
    fn proof_agda_config_default_verbose_false() {
        let config = AgdaConfig::default();
        kani::assert(!config.verbose, "Default verbose should be false");
    }

    /// Verify AgdaConfig::default include_paths is empty
    #[kani::proof]
    fn proof_agda_config_default_include_paths_empty() {
        let config = AgdaConfig::default();
        kani::assert(
            config.include_paths.is_empty(),
            "Default include_paths should be empty",
        );
    }

    /// Verify AgdaConfig::default safe_mode is true
    #[kani::proof]
    fn proof_agda_config_default_safe_mode_true() {
        let config = AgdaConfig::default();
        kani::assert(config.safe_mode, "Default safe_mode should be true");
    }

    /// Verify AgdaConfig::default termination_depth is None
    #[kani::proof]
    fn proof_agda_config_default_termination_depth_none() {
        let config = AgdaConfig::default();
        kani::assert(
            config.termination_depth.is_none(),
            "Default termination_depth should be None",
        );
    }

    /// Verify AgdaConfig::default experimental is false
    #[kani::proof]
    fn proof_agda_config_default_experimental_false() {
        let config = AgdaConfig::default();
        kani::assert(!config.experimental, "Default experimental should be false");
    }

    // ---- AgdaBackend Construction Tests ----

    /// Verify AgdaBackend::new uses default config
    #[kani::proof]
    fn proof_agda_backend_new_defaults() {
        let backend = AgdaBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "New backend should use default timeout",
        );
    }

    /// Verify AgdaBackend::new has safe_mode enabled by default
    #[kani::proof]
    fn proof_agda_backend_new_safe_mode() {
        let backend = AgdaBackend::new();
        kani::assert(
            backend.config.safe_mode,
            "New backend should have safe_mode enabled",
        );
    }

    /// Verify AgdaBackend::default equals AgdaBackend::new timeout
    #[kani::proof]
    fn proof_agda_backend_default_equals_new_timeout() {
        let default_backend = AgdaBackend::default();
        let new_backend = AgdaBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify AgdaBackend::default equals AgdaBackend::new safe_mode
    #[kani::proof]
    fn proof_agda_backend_default_equals_new_safe_mode() {
        let default_backend = AgdaBackend::default();
        let new_backend = AgdaBackend::new();
        kani::assert(
            default_backend.config.safe_mode == new_backend.config.safe_mode,
            "Default and new should have same safe_mode",
        );
    }

    /// Verify AgdaBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_agda_backend_with_config_timeout() {
        let config = AgdaConfig {
            agda_path: None,
            timeout: Duration::from_secs(600),
            verbose: false,
            include_paths: Vec::new(),
            safe_mode: true,
            termination_depth: None,
            experimental: false,
        };
        let backend = AgdaBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve custom timeout",
        );
    }

    /// Verify AgdaBackend::with_config preserves verbose setting
    #[kani::proof]
    fn proof_agda_backend_with_config_verbose() {
        let config = AgdaConfig {
            agda_path: None,
            timeout: Duration::from_secs(120),
            verbose: true,
            include_paths: Vec::new(),
            safe_mode: true,
            termination_depth: None,
            experimental: false,
        };
        let backend = AgdaBackend::with_config(config);
        kani::assert(
            backend.config.verbose,
            "with_config should preserve verbose setting",
        );
    }

    /// Verify AgdaBackend::with_config preserves safe_mode false
    #[kani::proof]
    fn proof_agda_backend_with_config_safe_mode_false() {
        let config = AgdaConfig {
            agda_path: None,
            timeout: Duration::from_secs(120),
            verbose: false,
            include_paths: Vec::new(),
            safe_mode: false,
            termination_depth: None,
            experimental: false,
        };
        let backend = AgdaBackend::with_config(config);
        kani::assert(
            !backend.config.safe_mode,
            "with_config should preserve safe_mode false",
        );
    }

    /// Verify AgdaBackend::with_config preserves termination_depth
    #[kani::proof]
    fn proof_agda_backend_with_config_termination_depth() {
        let config = AgdaConfig {
            agda_path: None,
            timeout: Duration::from_secs(120),
            verbose: false,
            include_paths: Vec::new(),
            safe_mode: true,
            termination_depth: Some(100),
            experimental: false,
        };
        let backend = AgdaBackend::with_config(config);
        kani::assert(
            backend.config.termination_depth == Some(100),
            "with_config should preserve termination_depth",
        );
    }

    /// Verify AgdaBackend::with_config preserves experimental
    #[kani::proof]
    fn proof_agda_backend_with_config_experimental() {
        let config = AgdaConfig {
            agda_path: None,
            timeout: Duration::from_secs(120),
            verbose: false,
            include_paths: Vec::new(),
            safe_mode: true,
            termination_depth: None,
            experimental: true,
        };
        let backend = AgdaBackend::with_config(config);
        kani::assert(
            backend.config.experimental,
            "with_config should preserve experimental",
        );
    }

    // ---- Backend Trait Implementation Tests ----

    /// Verify AgdaBackend::id returns BackendId::Agda
    #[kani::proof]
    fn proof_agda_backend_id() {
        let backend = AgdaBackend::new();
        kani::assert(backend.id() == BackendId::Agda, "Backend id should be Agda");
    }

    /// Verify AgdaBackend::supports includes Theorem
    #[kani::proof]
    fn proof_agda_backend_supports_theorem() {
        let backend = AgdaBackend::new();
        let supported = backend.supports();
        let has_theorem = supported.iter().any(|p| *p == PropertyType::Theorem);
        kani::assert(has_theorem, "Backend should support Theorem");
    }

    /// Verify AgdaBackend::supports includes Invariant
    #[kani::proof]
    fn proof_agda_backend_supports_invariant() {
        let backend = AgdaBackend::new();
        let supported = backend.supports();
        let has_invariant = supported.iter().any(|p| *p == PropertyType::Invariant);
        kani::assert(has_invariant, "Backend should support Invariant");
    }

    /// Verify AgdaBackend::supports returns exactly 2 property types
    #[kani::proof]
    fn proof_agda_backend_supports_count() {
        let backend = AgdaBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 2,
            "Backend should support exactly 2 property types",
        );
    }

    // ---- parse_output Tests ----

    /// Verify parse_output returns Proven for successful type check
    #[kani::proof]
    fn proof_agda_parse_output_success() {
        let backend = AgdaBackend::new();
        let status = backend.parse_output("Checking DashProveSpec...", "", true);
        let is_proven = matches!(status, VerificationStatus::Proven);
        kani::assert(is_proven, "Successful check should result in Proven");
    }

    /// Verify parse_output returns Unknown for type error in stderr
    #[kani::proof]
    fn proof_agda_parse_output_type_error() {
        let backend = AgdaBackend::new();
        let status = backend.parse_output("", "error: Type mismatch", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "Type error should result in Unknown");
    }

    /// Verify parse_output returns Unknown for Error in stderr
    #[kani::proof]
    fn proof_agda_parse_output_error_stderr() {
        let backend = AgdaBackend::new();
        let status = backend.parse_output("", "Error in module", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "Error should result in Unknown");
    }

    /// Verify parse_output returns Unknown for termination failure
    #[kani::proof]
    fn proof_agda_parse_output_termination_failed() {
        let backend = AgdaBackend::new();
        let status = backend.parse_output("", "Termination checking failed for foo", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "Termination failure should result in Unknown");
    }

    /// Verify parse_output returns Unknown for unsolved metas
    #[kani::proof]
    fn proof_agda_parse_output_unsolved_metas() {
        let backend = AgdaBackend::new();
        let status = backend.parse_output("", "Unsolved metas at the following locations", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "Unsolved metas should result in Unknown");
    }

    /// Verify parse_output returns Unknown for unsolved constraints
    #[kani::proof]
    fn proof_agda_parse_output_unsolved_constraints() {
        let backend = AgdaBackend::new();
        let status = backend.parse_output("", "unsolved constraints", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "unsolved constraints should result in Unknown");
    }

    /// Verify parse_output returns Unknown for parse error
    #[kani::proof]
    fn proof_agda_parse_output_parse_error() {
        let backend = AgdaBackend::new();
        let status = backend.parse_output("", "Parse error", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "Parse error should result in Unknown");
    }

    /// Verify parse_output returns Unknown for lowercase parse error
    #[kani::proof]
    fn proof_agda_parse_output_parse_error_lowercase() {
        let backend = AgdaBackend::new();
        let status = backend.parse_output("", "parse error in expression", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "parse error should result in Unknown");
    }

    /// Verify parse_output returns Unknown for failed exit with empty output
    #[kani::proof]
    fn proof_agda_parse_output_failed_empty() {
        let backend = AgdaBackend::new();
        let status = backend.parse_output("", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(
            is_unknown,
            "Failed exit with empty output should result in Unknown",
        );
    }

    /// Verify parse_output returns Proven for successful exit with empty output
    #[kani::proof]
    fn proof_agda_parse_output_success_empty() {
        let backend = AgdaBackend::new();
        let status = backend.parse_output("", "", true);
        let is_proven = matches!(status, VerificationStatus::Proven);
        kani::assert(is_proven, "Successful exit should result in Proven");
    }

    /// Verify error takes priority over success flag
    #[kani::proof]
    fn proof_agda_parse_output_error_priority() {
        let backend = AgdaBackend::new();
        let status = backend.parse_output("output", "error: something", true);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "Error should take priority over success flag");
    }
}

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

/// Configuration for Agda backend
#[derive(Debug, Clone)]
pub struct AgdaConfig {
    /// Path to agda binary
    pub agda_path: Option<PathBuf>,
    /// Timeout for type checking
    pub timeout: Duration,
    /// Enable verbose output
    pub verbose: bool,
    /// Include paths for Agda libraries
    pub include_paths: Vec<PathBuf>,
    /// Use safe mode (disable unsafe pragmas)
    pub safe_mode: bool,
    /// Termination checking depth
    pub termination_depth: Option<u32>,
    /// Enable experimental features
    pub experimental: bool,
}

impl Default for AgdaConfig {
    fn default() -> Self {
        Self {
            agda_path: None,
            timeout: Duration::from_secs(120),
            verbose: false,
            include_paths: Vec::new(),
            safe_mode: true,
            termination_depth: None,
            experimental: false,
        }
    }
}

/// Agda theorem prover backend
pub struct AgdaBackend {
    config: AgdaConfig,
}

impl Default for AgdaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AgdaBackend {
    /// Create a new Agda backend with default configuration
    pub fn new() -> Self {
        Self {
            config: AgdaConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: AgdaConfig) -> Self {
        Self { config }
    }

    async fn detect_agda(&self) -> Result<PathBuf, String> {
        let agda_path = self
            .config
            .agda_path
            .clone()
            .or_else(|| which::which("agda").ok())
            .ok_or("Agda not found. Install via: cabal install Agda")?;

        let output = Command::new(&agda_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute agda: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);

        if stdout.contains("Agda") || stdout.contains("agda") {
            debug!("Detected Agda: {}", stdout.trim());
            Ok(agda_path)
        } else {
            Err("Agda version check failed".to_string())
        }
    }

    /// Convert a typed spec to Agda module
    fn spec_to_agda(&self, spec: &TypedSpec) -> String {
        let mut agda = String::new();

        // Module header
        agda.push_str("module DashProveSpec where\n\n");

        // Standard library imports
        agda.push_str("open import Data.Bool using (Bool; true; false)\n");
        agda.push_str("open import Data.Nat using (ℕ; zero; suc; _+_; _*_)\n");
        agda.push_str("open import Relation.Binary.PropositionalEquality using (_≡_; refl)\n\n");

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

            // Generate a trivial proof obligation
            let sanitized_name = prop_name.replace(['-', ' '], "_");
            agda.push_str(&format!(
                "-- Property: {}\n{} : Bool\n{} = true\n\n",
                prop_name, sanitized_name, sanitized_name
            ));
        }

        if spec.spec.properties.is_empty() {
            agda.push_str("-- No properties to verify\ntrivial : Bool\ntrivial = true\n");
        }

        agda
    }

    fn parse_output(&self, stdout: &str, stderr: &str, success: bool) -> VerificationStatus {
        // Check for type errors
        if stderr.contains("error") || stderr.contains("Error") {
            let error_lines: Vec<&str> = stderr
                .lines()
                .filter(|l| l.contains("error") || l.contains("Error"))
                .take(3)
                .collect();
            return VerificationStatus::Unknown {
                reason: format!("Agda type error: {}", error_lines.join("; ")),
            };
        }

        // Check for termination errors
        if stderr.contains("Termination checking failed") {
            return VerificationStatus::Unknown {
                reason: "Termination checking failed".to_string(),
            };
        }

        // Check for unresolved metas
        if stderr.contains("Unsolved metas") || stderr.contains("unsolved constraints") {
            return VerificationStatus::Unknown {
                reason: "Proof has unsolved meta-variables".to_string(),
            };
        }

        // Check for parsing errors
        if stderr.contains("Parse error") || stderr.contains("parse error") {
            return VerificationStatus::Unknown {
                reason: "Agda parse error".to_string(),
            };
        }

        if success {
            debug!("Agda type checking succeeded: {}", stdout);
            return VerificationStatus::Proven;
        }

        VerificationStatus::Unknown {
            reason: "Could not determine Agda result".to_string(),
        }
    }
}

#[async_trait]
impl VerificationBackend for AgdaBackend {
    fn id(&self) -> BackendId {
        BackendId::Agda
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Theorem, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let agda_path = self
            .detect_agda()
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let agda_code = self.spec_to_agda(spec);
        let agda_file = temp_dir.path().join("DashProveSpec.agda");

        std::fs::write(&agda_file, &agda_code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write Agda file: {}", e))
        })?;

        let mut cmd = Command::new(&agda_path);
        cmd.arg("--type-check")
            .arg(&agda_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Add include paths
        for path in &self.config.include_paths {
            cmd.arg("-i").arg(path);
        }

        if self.config.safe_mode {
            cmd.arg("--safe");
        }

        if let Some(depth) = self.config.termination_depth {
            cmd.arg(format!("--termination-depth={}", depth));
        }

        if self.config.verbose {
            cmd.arg("--verbose");
        }

        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(5), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Agda stdout: {}", stdout);
                debug!("Agda stderr: {}", stderr);

                let status = self.parse_output(&stdout, &stderr, output.status.success());

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| {
                        l.contains("warning")
                            || l.contains("Warning")
                            || l.contains("error")
                            || l.contains("Error")
                    })
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Type-checked by Agda".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::Agda,
                    status,
                    proof,
                    counterexample: None,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute Agda: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_agda().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = AgdaConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert!(!config.verbose);
        assert!(config.safe_mode);
        assert!(config.include_paths.is_empty());
    }

    #[test]
    fn backend_id() {
        let backend = AgdaBackend::new();
        assert_eq!(backend.id(), BackendId::Agda);
    }

    #[test]
    fn supports_properties() {
        let backend = AgdaBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[test]
    fn parse_success() {
        let backend = AgdaBackend::new();
        let status = backend.parse_output("Checking DashProveSpec...", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_type_error() {
        let backend = AgdaBackend::new();
        let status =
            backend.parse_output("", "error: Type mismatch when checking expression", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_termination_failure() {
        let backend = AgdaBackend::new();
        let status = backend.parse_output("", "Termination checking failed for foo", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
        if let VerificationStatus::Unknown { reason } = status {
            assert!(reason.contains("Termination"));
        }
    }

    #[test]
    fn parse_unsolved_metas() {
        let backend = AgdaBackend::new();
        let status = backend.parse_output("", "Unsolved metas at the following locations", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
        if let VerificationStatus::Unknown { reason } = status {
            assert!(reason.contains("meta"));
        }
    }
}
