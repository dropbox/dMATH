//! RustBelt backend for Coq-based Rust semantic verification
//!
//! RustBelt is a formal semantic model for Rust developed at MPI-SWS that provides
//! machine-checked proofs (in Coq with the Iris framework) of Rust's type system
//! soundness, including for unsafe code.
//!
//! Key features:
//! - Semantic soundness proofs for Rust's type system
//! - Verification of unsafe code abstractions
//! - Based on Iris separation logic in Coq
//! - Foundational (proves the type system itself, not individual programs)
//!
//! Note: RustBelt is primarily a research framework for proving properties about
//! Rust's semantics, not a practical verification tool for application code.
//!
//! See: <https://plv.mpi-sws.org/rustbelt/>

use crate::traits::*;
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use regex::Regex;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::{debug, info};

// =============================================
// Kani Proofs for RustBelt Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- RustBeltConfig Default Tests ----

    /// Verify RustBeltConfig::default coq_path is None
    #[kani::proof]
    fn proof_rustbelt_config_default_coq_path_none() {
        let config = RustBeltConfig::default();
        kani::assert(config.coq_path.is_none(), "Default coq_path should be None");
    }

    /// Verify RustBeltConfig::default lambda_rust_path is None
    #[kani::proof]
    fn proof_rustbelt_config_default_lambda_rust_path_none() {
        let config = RustBeltConfig::default();
        kani::assert(
            config.lambda_rust_path.is_none(),
            "Default lambda_rust_path should be None",
        );
    }

    /// Verify RustBeltConfig::default timeout is 600 seconds
    #[kani::proof]
    fn proof_rustbelt_config_default_timeout() {
        let config = RustBeltConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(600),
            "Default timeout should be 600 seconds",
        );
    }

    /// Verify RustBeltConfig::default parallel_jobs is None
    #[kani::proof]
    fn proof_rustbelt_config_default_parallel_jobs_none() {
        let config = RustBeltConfig::default();
        kani::assert(
            config.parallel_jobs.is_none(),
            "Default parallel_jobs should be None",
        );
    }

    /// Verify RustBeltConfig::default verbose is false
    #[kani::proof]
    fn proof_rustbelt_config_default_verbose_false() {
        let config = RustBeltConfig::default();
        kani::assert(!config.verbose, "Default verbose should be false");
    }

    // ---- RustBeltBackend Construction Tests ----

    /// Verify RustBeltBackend::new uses default timeout
    #[kani::proof]
    fn proof_rustbelt_backend_new_default_timeout() {
        let backend = RustBeltBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "New backend should use default timeout",
        );
    }

    /// Verify RustBeltBackend::default equals RustBeltBackend::new
    #[kani::proof]
    fn proof_rustbelt_backend_default_equals_new() {
        let b1 = RustBeltBackend::new();
        let b2 = RustBeltBackend::default();
        kani::assert(
            b1.config.timeout == b2.config.timeout,
            "default and new should have same timeout",
        );
    }

    /// Verify RustBeltBackend::with_config preserves timeout
    #[kani::proof]
    fn proof_rustbelt_backend_with_config_preserves_timeout() {
        let config = RustBeltConfig {
            timeout: Duration::from_secs(1200),
            ..RustBeltConfig::default()
        };
        let backend = RustBeltBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(1200),
            "with_config should preserve custom timeout",
        );
    }

    // ---- ProofResult Tests ----

    /// Verify ProofResult::Qed is_success returns true
    #[kani::proof]
    fn proof_result_qed_is_success() {
        let result = ProofResult::Qed;
        kani::assert(result.is_success(), "Qed should return is_success true");
    }

    /// Verify ProofResult::Error is_success returns false
    #[kani::proof]
    fn proof_result_error_is_not_success() {
        let result = ProofResult::Error;
        kani::assert(!result.is_success(), "Error should return is_success false");
    }

    /// Verify ProofResult::Timeout is_success returns false
    #[kani::proof]
    fn proof_result_timeout_is_not_success() {
        let result = ProofResult::Timeout;
        kani::assert(
            !result.is_success(),
            "Timeout should return is_success false",
        );
    }
}

/// Result of a Coq proof compilation/check
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofResult {
    /// Proof completed successfully (Qed)
    Qed,
    /// Proof compilation failed with errors
    Error,
    /// Proof check timed out
    Timeout,
}

impl ProofResult {
    /// Check if the proof succeeded
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, ProofResult::Qed)
    }
}

/// Configuration for RustBelt backend
#[derive(Debug, Clone)]
pub struct RustBeltConfig {
    /// Path to coqc compiler (if not in PATH)
    pub coq_path: Option<PathBuf>,
    /// Path to lambda-rust/RustBelt development
    pub lambda_rust_path: Option<PathBuf>,
    /// Timeout for proof checking
    pub timeout: Duration,
    /// Number of parallel jobs for make (-j)
    pub parallel_jobs: Option<usize>,
    /// Enable verbose output
    pub verbose: bool,
    /// Additional Coq arguments
    pub coq_args: Vec<String>,
}

impl Default for RustBeltConfig {
    fn default() -> Self {
        Self {
            coq_path: None,
            lambda_rust_path: None,
            timeout: Duration::from_secs(600), // Coq proofs can be slow
            parallel_jobs: None,
            verbose: false,
            coq_args: Vec::new(),
        }
    }
}

/// RustBelt backend for Coq-based Rust semantic verification
#[derive(Debug, Clone)]
pub struct RustBeltBackend {
    /// Configuration
    pub config: RustBeltConfig,
}

impl Default for RustBeltBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RustBeltBackend {
    /// Create a new RustBelt backend with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: RustBeltConfig::default(),
        }
    }

    /// Create a new RustBelt backend with custom configuration
    #[must_use]
    pub fn with_config(config: RustBeltConfig) -> Self {
        Self { config }
    }

    /// Get the path to the coqc compiler
    fn get_coqc_path(&self) -> PathBuf {
        self.config
            .coq_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("coqc"))
    }

    /// Check if Coq is available
    pub async fn check_available(&self) -> Result<bool, BackendError> {
        let coqc_path = self.get_coqc_path();
        let output = Command::new(&coqc_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;

        match output {
            Ok(o) => Ok(o.status.success()),
            Err(_) => Ok(false),
        }
    }

    /// Parse Coq output to determine proof result
    fn parse_output(&self, stdout: &str, stderr: &str, timed_out: bool) -> ProofResult {
        if timed_out {
            return ProofResult::Timeout;
        }

        // Check for successful compilation (Qed, Defined, or no errors)
        if stderr.is_empty() || stdout.contains("Qed.") || stdout.contains("Defined.") {
            // Also check for error indicators
            if stderr.contains("Error:") || stderr.contains("Anomaly:") {
                return ProofResult::Error;
            }
            return ProofResult::Qed;
        }

        ProofResult::Error
    }

    /// Extract Iris tactics from output
    fn extract_tactics_used(&self, output: &str) -> Vec<String> {
        let mut tactics = Vec::new();
        let iris_tactics = [
            "iDestruct",
            "iSplitL",
            "iSplitR",
            "iNext",
            "iLob",
            "iIntros",
            "iApply",
            "iExists",
            "iModIntro",
            "iFrame",
        ];

        for tactic in &iris_tactics {
            if output.contains(tactic) {
                tactics.push(tactic.to_string());
            }
        }

        tactics
    }

    /// Extract error messages from Coq output
    fn extract_errors(&self, stderr: &str) -> Vec<String> {
        let mut errors = Vec::new();
        let error_regex = Regex::new(r"Error:\s*(.+)").unwrap();

        for cap in error_regex.captures_iter(stderr) {
            if let Some(msg) = cap.get(1) {
                errors.push(msg.as_str().trim().to_string());
            }
        }

        errors
    }

    /// Compile and check a Coq proof file
    pub async fn check_proof(&self, coq_file: &PathBuf) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        let coqc_path = self.get_coqc_path();

        let mut cmd = Command::new(&coqc_path);
        cmd.arg(coq_file);

        // Add include paths for Iris and lambda-rust if available
        if let Some(lambda_rust) = &self.config.lambda_rust_path {
            cmd.arg("-R").arg(lambda_rust).arg("lambda_rust");
        }

        for arg in &self.config.coq_args {
            cmd.arg(arg);
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        debug!("Running coqc: {:?}", cmd);

        let output_result = tokio::time::timeout(self.config.timeout, cmd.output()).await;

        let elapsed = start.elapsed();

        let (stdout, stderr, timed_out) = match output_result {
            Ok(Ok(output)) => (
                String::from_utf8_lossy(&output.stdout).to_string(),
                String::from_utf8_lossy(&output.stderr).to_string(),
                false,
            ),
            Ok(Err(e)) => {
                return Err(BackendError::VerificationFailed(format!(
                    "Failed to run coqc: {}",
                    e
                )));
            }
            Err(_) => ("".to_string(), "Timeout".to_string(), true),
        };

        let proof_result = self.parse_output(&stdout, &stderr, timed_out);
        let tactics = self.extract_tactics_used(&stdout);
        let errors = self.extract_errors(&stderr);

        let status = match proof_result {
            ProofResult::Qed => VerificationStatus::Proven,
            ProofResult::Error => VerificationStatus::Unknown {
                reason: "Coq proof failed".to_string(),
            },
            ProofResult::Timeout => VerificationStatus::Unknown {
                reason: "Coq proof verification timed out".to_string(),
            },
        };

        let mut diagnostics = Vec::new();
        if !tactics.is_empty() {
            diagnostics.push(format!("Iris tactics used: {}", tactics.join(", ")));
        }
        for error in &errors {
            diagnostics.push(format!("Error: {}", error));
        }
        if timed_out {
            diagnostics.push("Proof verification timed out".to_string());
        }

        info!(
            "RustBelt/Coq verification completed in {:?}: {:?}",
            elapsed, status
        );

        Ok(BackendResult {
            backend: BackendId::RustBelt,
            status,
            proof: if proof_result.is_success() {
                Some("Coq proof verified (Qed)".to_string())
            } else {
                None
            },
            counterexample: None,
            diagnostics,
            time_taken: elapsed,
        })
    }

    /// Check a Coq proof from source string
    pub async fn check_proof_source(&self, source: &str) -> Result<BackendResult, BackendError> {
        // Create temp directory and file
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let source_path = temp_dir.path().join("proof.v");
        std::fs::write(&source_path, source).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write source: {}", e))
        })?;

        self.check_proof(&source_path).await
    }
}

#[async_trait]
impl VerificationBackend for RustBeltBackend {
    fn id(&self) -> BackendId {
        BackendId::RustBelt
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Theorem,
            PropertyType::Contract,
            PropertyType::MemorySafety,
        ]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        // For now, return a placeholder - actual USL compilation would go here
        Ok(BackendResult {
            backend: BackendId::RustBelt,
            status: VerificationStatus::Unknown {
                reason: "USL to RustBelt/Coq compilation not yet implemented".to_string(),
            },
            proof: None,
            counterexample: None,
            diagnostics: vec!["RustBelt backend requires direct Coq proof verification".to_string()],
            time_taken: Duration::from_secs(0),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.check_available().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "Coq not found".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("RustBelt/Coq health check failed: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rustbelt_config_default() {
        let config = RustBeltConfig::default();
        assert!(config.coq_path.is_none());
        assert!(config.lambda_rust_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(600));
        assert!(config.parallel_jobs.is_none());
        assert!(!config.verbose);
        assert!(config.coq_args.is_empty());
    }

    #[test]
    fn test_rustbelt_backend_new() {
        let backend = RustBeltBackend::new();
        assert_eq!(backend.config.timeout, Duration::from_secs(600));
    }

    #[test]
    fn test_rustbelt_backend_with_config() {
        let config = RustBeltConfig {
            timeout: Duration::from_secs(1200),
            verbose: true,
            ..RustBeltConfig::default()
        };
        let backend = RustBeltBackend::with_config(config);
        assert_eq!(backend.config.timeout, Duration::from_secs(1200));
        assert!(backend.config.verbose);
    }

    #[test]
    fn test_proof_result_is_success() {
        assert!(ProofResult::Qed.is_success());
        assert!(!ProofResult::Error.is_success());
        assert!(!ProofResult::Timeout.is_success());
    }

    #[test]
    fn test_parse_output_qed() {
        let backend = RustBeltBackend::new();
        let result = backend.parse_output("Proof. ... Qed.", "", false);
        assert_eq!(result, ProofResult::Qed);
    }

    #[test]
    fn test_parse_output_error() {
        let backend = RustBeltBackend::new();
        let result = backend.parse_output("", "Error: Tactic failure.", false);
        assert_eq!(result, ProofResult::Error);
    }

    #[test]
    fn test_parse_output_timeout() {
        let backend = RustBeltBackend::new();
        let result = backend.parse_output("", "", true);
        assert_eq!(result, ProofResult::Timeout);
    }

    #[test]
    fn test_extract_tactics() {
        let backend = RustBeltBackend::new();
        let output = "iDestruct \"H\" as \"[H1 H2]\". iSplitL \"H1\". iApply \"H2\".";
        let tactics = backend.extract_tactics_used(output);
        assert!(tactics.contains(&"iDestruct".to_string()));
        assert!(tactics.contains(&"iSplitL".to_string()));
        assert!(tactics.contains(&"iApply".to_string()));
    }

    #[test]
    fn test_extract_errors() {
        let backend = RustBeltBackend::new();
        let stderr = "Error: Tactic failure.\nError: Cannot unify.";
        let errors = backend.extract_errors(stderr);
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_backend_id() {
        let backend = RustBeltBackend::new();
        assert_eq!(backend.id(), BackendId::RustBelt);
    }

    #[test]
    fn test_supports() {
        let backend = RustBeltBackend::new();
        let types = backend.supports();
        assert!(types.contains(&PropertyType::Theorem));
        assert!(types.contains(&PropertyType::Contract));
        assert!(types.contains(&PropertyType::MemorySafety));
    }
}
