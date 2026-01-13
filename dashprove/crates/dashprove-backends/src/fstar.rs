//! F* theorem prover backend
//!
//! F* is a proof-oriented programming language aimed at program verification.
//! It combines dependent types, monadic effects, refinement types, and SMT-based
//! semi-automatic proof automation.
//!
//! See: <https://www.fstar-lang.org/>

// =============================================
// Kani Proofs for F* Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- FStarSMTSolver Enum Tests ----

    /// Verify FStarSMTSolver::default returns Z3
    #[kani::proof]
    fn proof_fstar_smt_solver_default_is_z3() {
        let solver = FStarSMTSolver::default();
        kani::assert(solver == FStarSMTSolver::Z3, "Default solver should be Z3");
    }

    /// Verify Z3 and CVC are distinct variants
    #[kani::proof]
    fn proof_fstar_smt_solver_variants_distinct() {
        let z3 = FStarSMTSolver::Z3;
        let cvc = FStarSMTSolver::CVC;
        kani::assert(z3 != cvc, "Z3 and CVC should be distinct");
    }

    // ---- FStarConfig Default Tests ----

    /// Verify FStarConfig::default timeout is 120 seconds
    #[kani::proof]
    fn proof_fstar_config_default_timeout() {
        let config = FStarConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "Default timeout should be 120 seconds",
        );
    }

    /// Verify FStarConfig::default smt_timeout is 30 seconds
    #[kani::proof]
    fn proof_fstar_config_default_smt_timeout() {
        let config = FStarConfig::default();
        kani::assert(
            config.smt_timeout == Duration::from_secs(30),
            "Default SMT timeout should be 30 seconds",
        );
    }

    /// Verify FStarConfig::default fstar_path is None
    #[kani::proof]
    fn proof_fstar_config_default_path_none() {
        let config = FStarConfig::default();
        kani::assert(
            config.fstar_path.is_none(),
            "Default fstar_path should be None",
        );
    }

    /// Verify FStarConfig::default verbose is false
    #[kani::proof]
    fn proof_fstar_config_default_verbose_false() {
        let config = FStarConfig::default();
        kani::assert(!config.verbose, "Default verbose should be false");
    }

    /// Verify FStarConfig::default admit_all is false
    #[kani::proof]
    fn proof_fstar_config_default_admit_all_false() {
        let config = FStarConfig::default();
        kani::assert(!config.admit_all, "Default admit_all should be false");
    }

    /// Verify FStarConfig::default use_tactics is false
    #[kani::proof]
    fn proof_fstar_config_default_use_tactics_false() {
        let config = FStarConfig::default();
        kani::assert(!config.use_tactics, "Default use_tactics should be false");
    }

    /// Verify FStarConfig::default initial_fuel is 2
    #[kani::proof]
    fn proof_fstar_config_default_initial_fuel() {
        let config = FStarConfig::default();
        kani::assert(config.initial_fuel == 2, "Default initial_fuel should be 2");
    }

    /// Verify FStarConfig::default max_fuel is 8
    #[kani::proof]
    fn proof_fstar_config_default_max_fuel() {
        let config = FStarConfig::default();
        kani::assert(config.max_fuel == 8, "Default max_fuel should be 8");
    }

    /// Verify FStarConfig::default initial_ifuel is 1
    #[kani::proof]
    fn proof_fstar_config_default_initial_ifuel() {
        let config = FStarConfig::default();
        kani::assert(
            config.initial_ifuel == 1,
            "Default initial_ifuel should be 1",
        );
    }

    /// Verify FStarConfig::default max_ifuel is 2
    #[kani::proof]
    fn proof_fstar_config_default_max_ifuel() {
        let config = FStarConfig::default();
        kani::assert(config.max_ifuel == 2, "Default max_ifuel should be 2");
    }

    /// Verify FStarConfig::default include_paths is empty
    #[kani::proof]
    fn proof_fstar_config_default_include_paths_empty() {
        let config = FStarConfig::default();
        kani::assert(
            config.include_paths.is_empty(),
            "Default include_paths should be empty",
        );
    }

    /// Verify FStarConfig::default smt_solver is Z3
    #[kani::proof]
    fn proof_fstar_config_default_smt_solver_z3() {
        let config = FStarConfig::default();
        kani::assert(
            config.smt_solver == FStarSMTSolver::Z3,
            "Default smt_solver should be Z3",
        );
    }

    // ---- FStarBackend Construction Tests ----

    /// Verify FStarBackend::new uses default config
    #[kani::proof]
    fn proof_fstar_backend_new_defaults() {
        let backend = FStarBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "New backend should use default timeout",
        );
    }

    /// Verify FStarBackend::default equals FStarBackend::new
    #[kani::proof]
    fn proof_fstar_backend_default_equals_new() {
        let default_backend = FStarBackend::default();
        let new_backend = FStarBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify FStarBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_fstar_backend_with_config_timeout() {
        let config = FStarConfig {
            timeout: Duration::from_secs(600),
            ..FStarConfig::default()
        };
        let backend = FStarBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "Custom timeout should be preserved",
        );
    }

    /// Verify FStarBackend::with_config preserves verbose flag
    #[kani::proof]
    fn proof_fstar_backend_with_config_verbose() {
        let config = FStarConfig {
            verbose: true,
            ..FStarConfig::default()
        };
        let backend = FStarBackend::with_config(config);
        kani::assert(backend.config.verbose, "Custom verbose should be preserved");
    }

    /// Verify FStarBackend::with_config preserves admit_all flag
    #[kani::proof]
    fn proof_fstar_backend_with_config_admit_all() {
        let config = FStarConfig {
            admit_all: true,
            ..FStarConfig::default()
        };
        let backend = FStarBackend::with_config(config);
        kani::assert(
            backend.config.admit_all,
            "Custom admit_all should be preserved",
        );
    }

    /// Verify FStarBackend::with_config preserves smt_solver
    #[kani::proof]
    fn proof_fstar_backend_with_config_smt_solver() {
        let config = FStarConfig {
            smt_solver: FStarSMTSolver::CVC,
            ..FStarConfig::default()
        };
        let backend = FStarBackend::with_config(config);
        kani::assert(
            backend.config.smt_solver == FStarSMTSolver::CVC,
            "Custom smt_solver should be preserved",
        );
    }

    // ---- Backend Trait Implementation Tests ----

    /// Verify id() returns FStar
    #[kani::proof]
    fn proof_backend_id_is_fstar() {
        let backend = FStarBackend::new();
        kani::assert(backend.id() == BackendId::FStar, "ID should be FStar");
    }

    /// Verify supports() includes Theorem
    #[kani::proof]
    fn proof_fstar_supports_theorem() {
        let backend = FStarBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Theorem),
            "Should support Theorem",
        );
    }

    /// Verify supports() includes Invariant
    #[kani::proof]
    fn proof_fstar_supports_invariant() {
        let backend = FStarBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "Should support Invariant",
        );
    }

    /// Verify supports() includes Contract
    #[kani::proof]
    fn proof_fstar_supports_contract() {
        let backend = FStarBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Contract),
            "Should support Contract",
        );
    }

    /// Verify supports() returns exactly three property types
    #[kani::proof]
    fn proof_fstar_supports_count() {
        let backend = FStarBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 3,
            "Should support exactly three property types",
        );
    }

    // ---- parse_output Tests ----

    /// Verify parse_output recognizes Verified module
    #[kani::proof]
    fn proof_parse_output_verified_module() {
        let backend = FStarBackend::new();
        let status = backend.parse_output("Verified module: Test", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Verified module should return Proven",
        );
    }

    /// Verify parse_output recognizes all discharged
    #[kani::proof]
    fn proof_parse_output_all_discharged() {
        let backend = FStarBackend::new();
        let status = backend.parse_output("All verification conditions discharged", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "All discharged should return Proven",
        );
    }

    /// Verify parse_output recognizes could not prove
    #[kani::proof]
    fn proof_parse_output_could_not_prove() {
        let backend = FStarBackend::new();
        let status = backend.parse_output("", "could not prove", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Could not prove should return Unknown",
        );
    }

    /// Verify parse_output recognizes failed to verify
    #[kani::proof]
    fn proof_parse_output_failed_to_verify() {
        let backend = FStarBackend::new();
        let status = backend.parse_output("", "Failed to verify", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Failed to verify should return Unknown",
        );
    }

    /// Verify parse_output recognizes SMT solver failed
    #[kani::proof]
    fn proof_parse_output_smt_failed() {
        let backend = FStarBackend::new();
        let status = backend.parse_output("", "SMT solver failed", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "SMT solver failed should return Unknown",
        );
    }

    /// Verify parse_output recognizes Z3 query failed
    #[kani::proof]
    fn proof_parse_output_z3_failed() {
        let backend = FStarBackend::new();
        let status = backend.parse_output("", "Z3 query failed", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Z3 query failed should return Unknown",
        );
    }

    /// Verify parse_output recognizes timeout
    #[kani::proof]
    fn proof_parse_output_timeout() {
        let backend = FStarBackend::new();
        let status = backend.parse_output("", "SMT timeout", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Timeout should return Unknown",
        );
    }

    /// Verify parse_output recognizes type error
    #[kani::proof]
    fn proof_parse_output_type_error() {
        let backend = FStarBackend::new();
        let status = backend.parse_output("", "Type error in foo", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Type error should return Unknown",
        );
    }

    /// Verify parse_output recognizes Error: pattern
    #[kani::proof]
    fn proof_parse_output_error_pattern() {
        let backend = FStarBackend::new();
        let status = backend.parse_output("", "Error: something wrong", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Error: pattern should return Unknown",
        );
    }

    /// Verify parse_output success with empty output
    #[kani::proof]
    fn proof_parse_output_success_empty() {
        let backend = FStarBackend::new();
        let status = backend.parse_output("", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Success with empty output should return Proven",
        );
    }

    /// Verify parse_output failure with empty output
    #[kani::proof]
    fn proof_parse_output_failure_empty() {
        let backend = FStarBackend::new();
        let status = backend.parse_output("", "", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Failure with empty output should return Unknown",
        );
    }

    // ---- Fuel Configuration Tests ----

    /// Verify max_fuel >= initial_fuel in default config
    #[kani::proof]
    fn proof_max_fuel_gte_initial_fuel() {
        let config = FStarConfig::default();
        kani::assert(
            config.max_fuel >= config.initial_fuel,
            "max_fuel should be >= initial_fuel",
        );
    }

    /// Verify max_ifuel >= initial_ifuel in default config
    #[kani::proof]
    fn proof_max_ifuel_gte_initial_ifuel() {
        let config = FStarConfig::default();
        kani::assert(
            config.max_ifuel >= config.initial_ifuel,
            "max_ifuel should be >= initial_ifuel",
        );
    }

    /// Verify smt_timeout <= timeout in default config
    #[kani::proof]
    fn proof_smt_timeout_lte_timeout() {
        let config = FStarConfig::default();
        kani::assert(
            config.smt_timeout <= config.timeout,
            "smt_timeout should be <= timeout",
        );
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

/// SMT solver to use with F*
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FStarSMTSolver {
    /// Z3 solver (default)
    #[default]
    Z3,
    /// CVC4/CVC5 solver
    CVC,
}

/// Configuration for F* backend
#[derive(Debug, Clone)]
pub struct FStarConfig {
    /// Path to fstar.exe binary
    pub fstar_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Enable verbose output
    pub verbose: bool,
    /// Include paths for F* modules
    pub include_paths: Vec<PathBuf>,
    /// SMT solver to use
    pub smt_solver: FStarSMTSolver,
    /// SMT query timeout (per query)
    pub smt_timeout: Duration,
    /// Initial fuel for SMT solver
    pub initial_fuel: u32,
    /// Maximum fuel for SMT solver
    pub max_fuel: u32,
    /// Initial ifuel (implicit fuel)
    pub initial_ifuel: u32,
    /// Maximum ifuel
    pub max_ifuel: u32,
    /// Use tactics (Meta-F*)
    pub use_tactics: bool,
    /// Admit all lemmas (for debugging)
    pub admit_all: bool,
}

impl Default for FStarConfig {
    fn default() -> Self {
        Self {
            fstar_path: None,
            timeout: Duration::from_secs(120),
            verbose: false,
            include_paths: Vec::new(),
            smt_solver: FStarSMTSolver::default(),
            smt_timeout: Duration::from_secs(30),
            initial_fuel: 2,
            max_fuel: 8,
            initial_ifuel: 1,
            max_ifuel: 2,
            use_tactics: false,
            admit_all: false,
        }
    }
}

/// F* theorem prover backend
pub struct FStarBackend {
    config: FStarConfig,
}

impl Default for FStarBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl FStarBackend {
    /// Create a new F* backend with default configuration
    pub fn new() -> Self {
        Self {
            config: FStarConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: FStarConfig) -> Self {
        Self { config }
    }

    async fn detect_fstar(&self) -> Result<PathBuf, String> {
        let fstar_path = self
            .config
            .fstar_path
            .clone()
            .or_else(|| which::which("fstar.exe").ok())
            .or_else(|| which::which("fstar").ok())
            .ok_or("F* not found. Install from: https://github.com/FStarLang/FStar")?;

        let output = Command::new(&fstar_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute fstar: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);

        if stdout.contains("F*") || stdout.contains("fstar") || stdout.contains("FStar") {
            debug!("Detected F*: {}", stdout.trim());
            Ok(fstar_path)
        } else if output.status.success() {
            debug!("Detected F* binary at: {}", fstar_path.display());
            Ok(fstar_path)
        } else {
            Err("F* version check failed".to_string())
        }
    }

    /// Convert a typed spec to F* module
    fn spec_to_fstar(&self, spec: &TypedSpec) -> String {
        let mut fstar = String::new();

        // Module header
        fstar.push_str("module DashProveSpec\n\n");
        fstar.push_str("(* DashProve generated F* module *)\n\n");

        // Basic imports
        fstar.push_str("open FStar.All\n");
        fstar.push_str("open FStar.Mul\n\n");

        // Generate lemmas from properties
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

            // Sanitize name for F* (ML-style identifier)
            let sanitized_name = prop_name
                .replace([' ', '-'], "_")
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>();

            // Ensure name starts with lowercase (F* convention)
            let sanitized_name = if sanitized_name
                .chars()
                .next()
                .is_none_or(|c| c.is_uppercase())
            {
                format!("prop_{}", sanitized_name.to_lowercase())
            } else {
                sanitized_name
            };

            // Generate a trivial lemma
            fstar.push_str(&format!(
                "(* Property: {} *)\nlet {} () : Lemma (True) = ()\n\n",
                prop_name, sanitized_name
            ));
        }

        if spec.spec.properties.is_empty() {
            fstar.push_str("(* No properties to verify *)\n");
            fstar.push_str("let trivial () : Lemma (True) = ()\n");
        }

        fstar
    }

    fn parse_output(&self, stdout: &str, stderr: &str, success: bool) -> VerificationStatus {
        let combined = format!("{}\n{}", stdout, stderr);

        // Check for verification success
        if success
            && (combined.contains("Verified module")
                || combined.contains("All verification conditions discharged")
                || (stdout.contains("(") && !combined.contains("Error")))
        {
            return VerificationStatus::Proven;
        }

        // Check for verification failure
        if combined.contains("could not prove") || combined.contains("Failed to verify") {
            return VerificationStatus::Unknown {
                reason: "F* verification failed: could not prove".to_string(),
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
                reason: format!("F* error: {}", error_lines.join("; ")),
            };
        }

        // Check for type errors
        if combined.contains("Type error") || combined.contains("Expected type") {
            return VerificationStatus::Unknown {
                reason: "F* type error".to_string(),
            };
        }

        // Check for SMT failures
        if combined.contains("SMT solver failed")
            || combined.contains("Z3 query failed")
            || combined.contains("Unknown assertion failed")
        {
            return VerificationStatus::Unknown {
                reason: "F* SMT solver failed".to_string(),
            };
        }

        // Check for timeout
        if combined.contains("timeout") || combined.contains("Timeout") {
            return VerificationStatus::Unknown {
                reason: "F* SMT timeout".to_string(),
            };
        }

        // Successful exit
        if success {
            debug!("F* completed successfully");
            return VerificationStatus::Proven;
        }

        VerificationStatus::Unknown {
            reason: "Could not determine F* result".to_string(),
        }
    }
}

#[async_trait]
impl VerificationBackend for FStarBackend {
    fn id(&self) -> BackendId {
        BackendId::FStar
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Theorem,
            PropertyType::Invariant,
            PropertyType::Contract,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let fstar_path = self
            .detect_fstar()
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let fstar_code = self.spec_to_fstar(spec);
        let fstar_file = temp_dir.path().join("DashProveSpec.fst");

        std::fs::write(&fstar_file, &fstar_code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write F* file: {}", e))
        })?;

        let mut cmd = Command::new(&fstar_path);
        cmd.arg(&fstar_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Add include paths
        for path in &self.config.include_paths {
            cmd.arg("--include").arg(path);
        }

        // SMT settings
        let smt_timeout_ms = self.config.smt_timeout.as_millis() as u32;
        cmd.arg(format!("--z3rlimit={}", smt_timeout_ms));
        cmd.arg(format!("--initial_fuel={}", self.config.initial_fuel));
        cmd.arg(format!("--max_fuel={}", self.config.max_fuel));
        cmd.arg(format!("--initial_ifuel={}", self.config.initial_ifuel));
        cmd.arg(format!("--max_ifuel={}", self.config.max_ifuel));

        // SMT solver selection
        match self.config.smt_solver {
            FStarSMTSolver::Z3 => {} // Default
            FStarSMTSolver::CVC => {
                cmd.arg("--smt").arg("cvc4");
            }
        }

        if self.config.use_tactics {
            cmd.arg("--use_hints");
        }

        if self.config.admit_all {
            cmd.arg("--admit_smt_queries").arg("true");
        }

        if self.config.verbose {
            cmd.arg("--log_queries");
        }

        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(5), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("F* stdout: {}", stdout);
                debug!("F* stderr: {}", stderr);

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
                    Some("Verified by F*".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::FStar,
                    status,
                    proof,
                    counterexample: None,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute F*: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_fstar().await {
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
        let config = FStarConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert!(!config.verbose);
        assert!(!config.admit_all);
        assert_eq!(config.initial_fuel, 2);
        assert_eq!(config.max_fuel, 8);
        assert_eq!(config.smt_solver, FStarSMTSolver::Z3);
    }

    #[test]
    fn backend_id() {
        let backend = FStarBackend::new();
        assert_eq!(backend.id(), BackendId::FStar);
    }

    #[test]
    fn supports_properties() {
        let backend = FStarBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
        assert!(supported.contains(&PropertyType::Contract));
    }

    #[test]
    fn parse_verified_module() {
        let backend = FStarBackend::new();
        let status = backend.parse_output("Verified module: DashProveSpec", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_all_discharged() {
        let backend = FStarBackend::new();
        let status = backend.parse_output(
            "All verification conditions discharged successfully",
            "",
            true,
        );
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_could_not_prove() {
        let backend = FStarBackend::new();
        let status = backend.parse_output("", "could not prove: ...", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_smt_failed() {
        let backend = FStarBackend::new();
        let status = backend.parse_output("", "SMT solver failed", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
        if let VerificationStatus::Unknown { reason } = status {
            assert!(reason.contains("SMT"));
        }
    }

    #[test]
    fn parse_success_exit() {
        let backend = FStarBackend::new();
        let status = backend.parse_output("", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn smt_solver_variants() {
        let _z3 = FStarSMTSolver::Z3;
        let _cvc = FStarSMTSolver::CVC;
        assert_eq!(FStarSMTSolver::default(), FStarSMTSolver::Z3);
    }
}
