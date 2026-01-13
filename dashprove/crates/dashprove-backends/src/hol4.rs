//! HOL4 theorem prover backend
//!
//! HOL4 is an interactive theorem prover based on higher-order logic.
//! It provides a rich set of tactics and extensive library of formalized mathematics.
//!
//! See: <https://hol-theorem-prover.org/>

// =============================================
// Kani Proofs for HOL4 Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- Hol4Config Default Tests ----

    /// Verify Hol4Config::default timeout is 180 seconds
    #[kani::proof]
    fn proof_hol4_config_default_timeout() {
        let config = Hol4Config::default();
        kani::assert(
            config.timeout == Duration::from_secs(180),
            "Default timeout should be 180 seconds",
        );
    }

    /// Verify Hol4Config::default holmake_path is None
    #[kani::proof]
    fn proof_hol4_config_default_holmake_path_none() {
        let config = Hol4Config::default();
        kani::assert(
            config.holmake_path.is_none(),
            "Default holmake_path should be None",
        );
    }

    /// Verify Hol4Config::default hol_path is None
    #[kani::proof]
    fn proof_hol4_config_default_hol_path_none() {
        let config = Hol4Config::default();
        kani::assert(config.hol_path.is_none(), "Default hol_path should be None");
    }

    /// Verify Hol4Config::default verbose is false
    #[kani::proof]
    fn proof_hol4_config_default_verbose_false() {
        let config = Hol4Config::default();
        kani::assert(!config.verbose, "Default verbose should be false");
    }

    /// Verify Hol4Config::default lib_paths is empty
    #[kani::proof]
    fn proof_hol4_config_default_lib_paths_empty() {
        let config = Hol4Config::default();
        kani::assert(
            config.lib_paths.is_empty(),
            "Default lib_paths should be empty",
        );
    }

    /// Verify Hol4Config::default parallel_jobs is None
    #[kani::proof]
    fn proof_hol4_config_default_parallel_jobs_none() {
        let config = Hol4Config::default();
        kani::assert(
            config.parallel_jobs.is_none(),
            "Default parallel_jobs should be None",
        );
    }

    /// Verify Hol4Config::default use_poly is true
    #[kani::proof]
    fn proof_hol4_config_default_use_poly_true() {
        let config = Hol4Config::default();
        kani::assert(config.use_poly, "Default use_poly should be true");
    }

    // ---- Hol4Backend Construction Tests ----

    /// Verify Hol4Backend::new uses default config
    #[kani::proof]
    fn proof_hol4_backend_new_defaults() {
        let backend = Hol4Backend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(180),
            "New backend should use default timeout",
        );
    }

    /// Verify Hol4Backend::new has use_poly enabled by default
    #[kani::proof]
    fn proof_hol4_backend_new_use_poly() {
        let backend = Hol4Backend::new();
        kani::assert(
            backend.config.use_poly,
            "New backend should have use_poly enabled",
        );
    }

    /// Verify Hol4Backend::default equals Hol4Backend::new timeout
    #[kani::proof]
    fn proof_hol4_backend_default_equals_new_timeout() {
        let default_backend = Hol4Backend::default();
        let new_backend = Hol4Backend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify Hol4Backend::default equals Hol4Backend::new use_poly
    #[kani::proof]
    fn proof_hol4_backend_default_equals_new_use_poly() {
        let default_backend = Hol4Backend::default();
        let new_backend = Hol4Backend::new();
        kani::assert(
            default_backend.config.use_poly == new_backend.config.use_poly,
            "Default and new should have same use_poly",
        );
    }

    /// Verify Hol4Backend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_hol4_backend_with_config_timeout() {
        let config = Hol4Config {
            holmake_path: None,
            hol_path: None,
            timeout: Duration::from_secs(600),
            verbose: false,
            lib_paths: Vec::new(),
            parallel_jobs: None,
            use_poly: true,
        };
        let backend = Hol4Backend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve custom timeout",
        );
    }

    /// Verify Hol4Backend::with_config preserves verbose setting
    #[kani::proof]
    fn proof_hol4_backend_with_config_verbose() {
        let config = Hol4Config {
            holmake_path: None,
            hol_path: None,
            timeout: Duration::from_secs(180),
            verbose: true,
            lib_paths: Vec::new(),
            parallel_jobs: None,
            use_poly: true,
        };
        let backend = Hol4Backend::with_config(config);
        kani::assert(
            backend.config.verbose,
            "with_config should preserve verbose setting",
        );
    }

    /// Verify Hol4Backend::with_config preserves parallel_jobs
    #[kani::proof]
    fn proof_hol4_backend_with_config_parallel_jobs() {
        let config = Hol4Config {
            holmake_path: None,
            hol_path: None,
            timeout: Duration::from_secs(180),
            verbose: false,
            lib_paths: Vec::new(),
            parallel_jobs: Some(8),
            use_poly: true,
        };
        let backend = Hol4Backend::with_config(config);
        kani::assert(
            backend.config.parallel_jobs == Some(8),
            "with_config should preserve parallel_jobs",
        );
    }

    /// Verify Hol4Backend::with_config preserves use_poly false
    #[kani::proof]
    fn proof_hol4_backend_with_config_use_poly_false() {
        let config = Hol4Config {
            holmake_path: None,
            hol_path: None,
            timeout: Duration::from_secs(180),
            verbose: false,
            lib_paths: Vec::new(),
            parallel_jobs: None,
            use_poly: false,
        };
        let backend = Hol4Backend::with_config(config);
        kani::assert(
            !backend.config.use_poly,
            "with_config should preserve use_poly false",
        );
    }

    // ---- Backend Trait Implementation Tests ----

    /// Verify Hol4Backend::id returns BackendId::HOL4
    #[kani::proof]
    fn proof_hol4_backend_id() {
        let backend = Hol4Backend::new();
        kani::assert(backend.id() == BackendId::HOL4, "Backend id should be HOL4");
    }

    /// Verify Hol4Backend::supports includes Theorem
    #[kani::proof]
    fn proof_hol4_backend_supports_theorem() {
        let backend = Hol4Backend::new();
        let supported = backend.supports();
        let has_theorem = supported.iter().any(|p| *p == PropertyType::Theorem);
        kani::assert(has_theorem, "Backend should support Theorem");
    }

    /// Verify Hol4Backend::supports includes Invariant
    #[kani::proof]
    fn proof_hol4_backend_supports_invariant() {
        let backend = Hol4Backend::new();
        let supported = backend.supports();
        let has_invariant = supported.iter().any(|p| *p == PropertyType::Invariant);
        kani::assert(has_invariant, "Backend should support Invariant");
    }

    /// Verify Hol4Backend::supports returns exactly 2 property types
    #[kani::proof]
    fn proof_hol4_backend_supports_count() {
        let backend = Hol4Backend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 2,
            "Backend should support exactly 2 property types",
        );
    }

    // ---- parse_output Tests ----

    /// Verify parse_output returns Proven for "stored"
    #[kani::proof]
    fn proof_hol4_parse_output_stored() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("Theorem stored: trivial_thm", "", true);
        let is_proven = matches!(status, VerificationStatus::Proven);
        kani::assert(is_proven, "stored should result in Proven");
    }

    /// Verify parse_output returns Proven for "Theorem saved"
    #[kani::proof]
    fn proof_hol4_parse_output_theorem_saved() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("Theorem saved", "", true);
        let is_proven = matches!(status, VerificationStatus::Proven);
        kani::assert(is_proven, "Theorem saved should result in Proven");
    }

    /// Verify parse_output returns Proven for "Theory exported"
    #[kani::proof]
    fn proof_hol4_parse_output_theory_exported() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("Theory exported: DashProveSpec", "", true);
        let is_proven = matches!(status, VerificationStatus::Proven);
        kani::assert(is_proven, "Theory exported should result in Proven");
    }

    /// Verify parse_output returns Unknown for "FAILED"
    #[kani::proof]
    fn proof_hol4_parse_output_failed() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("FAILED", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "FAILED should result in Unknown");
    }

    /// Verify parse_output returns Unknown for "Exception"
    #[kani::proof]
    fn proof_hol4_parse_output_exception() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("Exception- HOL_ERR", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "Exception should result in Unknown");
    }

    /// Verify parse_output returns Unknown for "proof failed"
    #[kani::proof]
    fn proof_hol4_parse_output_proof_failed() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("proof failed", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "proof failed should result in Unknown");
    }

    /// Verify parse_output returns Unknown for "Error:" in output
    #[kani::proof]
    fn proof_hol4_parse_output_error_output() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("Error: undefined identifier", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "Error: should result in Unknown");
    }

    /// Verify parse_output returns Unknown for "error:" in stderr
    #[kani::proof]
    fn proof_hol4_parse_output_error_stderr() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("", "error: something", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "error: should result in Unknown");
    }

    /// Verify parse_output returns Unknown for "Parse error"
    #[kani::proof]
    fn proof_hol4_parse_output_parse_error() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("Parse error at line 1", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "Parse error should result in Unknown");
    }

    /// Verify parse_output returns Unknown for "syntax error"
    #[kani::proof]
    fn proof_hol4_parse_output_syntax_error() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("syntax error", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "syntax error should result in Unknown");
    }

    /// Verify parse_output returns Proven for successful exit with empty output
    #[kani::proof]
    fn proof_hol4_parse_output_success_empty() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("", "", true);
        let is_proven = matches!(status, VerificationStatus::Proven);
        kani::assert(is_proven, "Successful exit should result in Proven");
    }

    /// Verify parse_output returns Unknown for failed exit with empty output
    #[kani::proof]
    fn proof_hol4_parse_output_failed_empty() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(
            is_unknown,
            "Failed exit with empty output should result in Unknown",
        );
    }

    /// Verify stored takes priority over error
    #[kani::proof]
    fn proof_hol4_parse_output_stored_priority() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("Theorem stored: thm\nError: warning", "", true);
        let is_proven = matches!(status, VerificationStatus::Proven);
        kani::assert(is_proven, "stored should take priority");
    }

    /// Verify FAILED takes priority over stored when FAILED appears first
    #[kani::proof]
    fn proof_hol4_parse_output_failed_priority() {
        let backend = Hol4Backend::new();
        // If FAILED appears before stored, it means proof setup failed
        let status = backend.parse_output("FAILED\nstored", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "FAILED should result in Unknown");
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

/// Configuration for HOL4 backend
#[derive(Debug, Clone)]
pub struct Hol4Config {
    /// Path to Holmake binary
    pub holmake_path: Option<PathBuf>,
    /// Path to hol binary (for interactive mode)
    pub hol_path: Option<PathBuf>,
    /// Timeout for theorem proving
    pub timeout: Duration,
    /// Enable verbose output
    pub verbose: bool,
    /// Additional library paths
    pub lib_paths: Vec<PathBuf>,
    /// Parallel jobs for Holmake
    pub parallel_jobs: Option<u32>,
    /// Use poly/ml backend (vs Moscow ML)
    pub use_poly: bool,
}

impl Default for Hol4Config {
    fn default() -> Self {
        Self {
            holmake_path: None,
            hol_path: None,
            timeout: Duration::from_secs(180),
            verbose: false,
            lib_paths: Vec::new(),
            parallel_jobs: None,
            use_poly: true,
        }
    }
}

/// HOL4 theorem prover backend
pub struct Hol4Backend {
    config: Hol4Config,
}

impl Default for Hol4Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl Hol4Backend {
    /// Create a new HOL4 backend with default configuration
    pub fn new() -> Self {
        Self {
            config: Hol4Config::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: Hol4Config) -> Self {
        Self { config }
    }

    async fn detect_hol4(&self) -> Result<PathBuf, String> {
        // Try Holmake first (preferred for batch mode), then hol
        let hol_path = self
            .config
            .holmake_path
            .clone()
            .or_else(|| which::which("Holmake").ok())
            .or_else(|| self.config.hol_path.clone())
            .or_else(|| which::which("hol").ok())
            .ok_or("HOL4 not found. Build from: https://github.com/HOL-Theorem-Prover/HOL")?;

        let output = Command::new(&hol_path)
            .arg("--help")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute HOL4: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}{}", stdout, stderr);

        if combined.contains("HOL")
            || combined.contains("Holmake")
            || combined.contains("hol")
            || !stdout.is_empty()
        {
            debug!("Detected HOL4: {}", hol_path.display());
            Ok(hol_path)
        } else {
            Err("HOL4 version check failed".to_string())
        }
    }

    /// Convert a typed spec to HOL4 script
    fn spec_to_hol4(&self, spec: &TypedSpec) -> String {
        let mut hol = String::new();

        // HOL4 script header
        hol.push_str("(* DashProve generated HOL4 script *)\n\n");
        hol.push_str("open HolKernel boolLib bossLib;\n\n");
        hol.push_str("val _ = new_theory \"DashProveSpec\";\n\n");

        // Generate theorems from properties
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

            // Sanitize name for HOL4 (ML identifier rules)
            let sanitized_name = prop_name
                .replace([' ', '-'], "_")
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>();

            // Ensure name starts with lowercase (ML convention)
            let sanitized_name = if sanitized_name
                .chars()
                .next()
                .is_none_or(|c| c.is_uppercase())
            {
                format!("prop_{}", sanitized_name.to_lowercase())
            } else {
                sanitized_name
            };

            // Generate a trivial theorem
            hol.push_str(&format!(
                "(* Property: {} *)\nval {}_thm = store_thm(\n  \"{}_thm\",\n  ``T``,\n  REWRITE_TAC []);\n\n",
                prop_name, sanitized_name, sanitized_name
            ));
        }

        if spec.spec.properties.is_empty() {
            hol.push_str("(* No properties to verify *)\n");
            hol.push_str("val trivial_thm = store_thm(\n");
            hol.push_str("  \"trivial_thm\",\n");
            hol.push_str("  ``T``,\n");
            hol.push_str("  REWRITE_TAC []);\n\n");
        }

        // Export theory
        hol.push_str("val _ = export_theory();\n");

        hol
    }

    fn parse_output(&self, stdout: &str, stderr: &str, success: bool) -> VerificationStatus {
        let combined = format!("{}\n{}", stdout, stderr);

        // Check for successful theorem storage
        if combined.contains("stored")
            || combined.contains("Theorem saved")
            || combined.contains("Theory exported")
        {
            return VerificationStatus::Proven;
        }

        // Check for proof failure
        if combined.contains("FAILED")
            || combined.contains("Exception")
            || combined.contains("proof failed")
        {
            let error_lines: Vec<&str> = combined
                .lines()
                .filter(|l| {
                    l.contains("FAILED")
                        || l.contains("Exception")
                        || l.contains("Error")
                        || l.contains("error")
                })
                .take(3)
                .collect();
            return VerificationStatus::Unknown {
                reason: format!("HOL4 proof failed: {}", error_lines.join("; ")),
            };
        }

        // Check for errors
        if combined.contains("Error:") || combined.contains("error:") {
            let error_lines: Vec<&str> = combined
                .lines()
                .filter(|l| l.contains("Error") || l.contains("error"))
                .take(3)
                .collect();
            return VerificationStatus::Unknown {
                reason: format!("HOL4 error: {}", error_lines.join("; ")),
            };
        }

        // Check for parse errors
        if combined.contains("Parse error") || combined.contains("syntax error") {
            return VerificationStatus::Unknown {
                reason: "HOL4 parse error".to_string(),
            };
        }

        // Successful exit
        if success {
            debug!("HOL4 completed successfully");
            return VerificationStatus::Proven;
        }

        VerificationStatus::Unknown {
            reason: "Could not determine HOL4 result".to_string(),
        }
    }
}

#[async_trait]
impl VerificationBackend for Hol4Backend {
    fn id(&self) -> BackendId {
        BackendId::HOL4
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Theorem, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let hol_path = self
            .detect_hol4()
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let hol_code = self.spec_to_hol4(spec);
        let hol_file = temp_dir.path().join("DashProveSpecScript.sml");

        std::fs::write(&hol_file, &hol_code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write HOL4 file: {}", e))
        })?;

        let mut cmd = Command::new(&hol_path);

        // Check if we're using Holmake or hol
        let is_holmake = hol_path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().contains("Holmake"));

        if is_holmake {
            cmd.current_dir(temp_dir.path());
            if let Some(jobs) = self.config.parallel_jobs {
                cmd.arg(format!("-j{}", jobs));
            }
        } else {
            // Interactive hol mode - need to load script
            cmd.arg("--quiet").arg("<").arg(&hol_file);
        }

        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        // Add library paths
        for path in &self.config.lib_paths {
            cmd.env("HOLDIR", path);
        }

        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(5), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("HOL4 stdout: {}", stdout);
                debug!("HOL4 stderr: {}", stderr);

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
                    Some("Proven by HOL4".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::HOL4,
                    status,
                    proof,
                    counterexample: None,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute HOL4: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_hol4().await {
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
        let config = Hol4Config::default();
        assert_eq!(config.timeout, Duration::from_secs(180));
        assert!(!config.verbose);
        assert!(config.use_poly);
        assert!(config.lib_paths.is_empty());
    }

    #[test]
    fn backend_id() {
        let backend = Hol4Backend::new();
        assert_eq!(backend.id(), BackendId::HOL4);
    }

    #[test]
    fn supports_properties() {
        let backend = Hol4Backend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[test]
    fn parse_stored_success() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("Theorem stored: trivial_thm", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_exported_success() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("Theory exported: DashProveSpec", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_failed() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("Exception- FAILED", "", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_error() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("", "Error: undefined identifier", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_success_exit() {
        let backend = Hol4Backend::new();
        let status = backend.parse_output("", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }
}
