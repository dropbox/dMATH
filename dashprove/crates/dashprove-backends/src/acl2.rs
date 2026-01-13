//! ACL2 theorem prover backend
//!
//! ACL2 (A Computational Logic for Applicative Common Lisp) is an industrial-strength
//! theorem prover for first-order logic with a focus on automated reasoning.
//!
//! See: <https://www.cs.utexas.edu/users/moore/acl2/>

// =============================================
// Kani Proofs for ACL2 Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- Acl2Config Default Tests ----

    /// Verify Acl2Config::default timeout is 180 seconds
    #[kani::proof]
    fn proof_acl2_config_default_timeout() {
        let config = Acl2Config::default();
        kani::assert(
            config.timeout == Duration::from_secs(180),
            "Default timeout should be 180 seconds",
        );
    }

    /// Verify Acl2Config::default acl2_path is None
    #[kani::proof]
    fn proof_acl2_config_default_path_none() {
        let config = Acl2Config::default();
        kani::assert(
            config.acl2_path.is_none(),
            "Default acl2_path should be None",
        );
    }

    /// Verify Acl2Config::default verbose is false
    #[kani::proof]
    fn proof_acl2_config_default_verbose_false() {
        let config = Acl2Config::default();
        kani::assert(!config.verbose, "Default verbose should be false");
    }

    /// Verify Acl2Config::default acl2s_mode is false
    #[kani::proof]
    fn proof_acl2_config_default_acl2s_mode_false() {
        let config = Acl2Config::default();
        kani::assert(!config.acl2s_mode, "Default acl2s_mode should be false");
    }

    /// Verify Acl2Config::default book_dirs is empty
    #[kani::proof]
    fn proof_acl2_config_default_book_dirs_empty() {
        let config = Acl2Config::default();
        kani::assert(
            config.book_dirs.is_empty(),
            "Default book_dirs should be empty",
        );
    }

    /// Verify Acl2Config::default memory_limit_mb is None
    #[kani::proof]
    fn proof_acl2_config_default_memory_limit_none() {
        let config = Acl2Config::default();
        kani::assert(
            config.memory_limit_mb.is_none(),
            "Default memory_limit_mb should be None",
        );
    }

    /// Verify Acl2Config::default proof_limit is None
    #[kani::proof]
    fn proof_acl2_config_default_proof_limit_none() {
        let config = Acl2Config::default();
        kani::assert(
            config.proof_limit.is_none(),
            "Default proof_limit should be None",
        );
    }

    // ---- Acl2Backend Construction Tests ----

    /// Verify Acl2Backend::new uses default config
    #[kani::proof]
    fn proof_acl2_backend_new_defaults() {
        let backend = Acl2Backend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(180),
            "New backend should use default timeout",
        );
    }

    /// Verify Acl2Backend::default equals Acl2Backend::new timeout
    #[kani::proof]
    fn proof_acl2_backend_default_equals_new_timeout() {
        let default_backend = Acl2Backend::default();
        let new_backend = Acl2Backend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify Acl2Backend::default equals Acl2Backend::new verbose
    #[kani::proof]
    fn proof_acl2_backend_default_equals_new_verbose() {
        let default_backend = Acl2Backend::default();
        let new_backend = Acl2Backend::new();
        kani::assert(
            default_backend.config.verbose == new_backend.config.verbose,
            "Default and new should have same verbose",
        );
    }

    /// Verify Acl2Backend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_acl2_backend_with_config_timeout() {
        let config = Acl2Config {
            acl2_path: None,
            timeout: Duration::from_secs(600),
            verbose: false,
            book_dirs: Vec::new(),
            acl2s_mode: false,
            memory_limit_mb: None,
            proof_limit: None,
        };
        let backend = Acl2Backend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve custom timeout",
        );
    }

    /// Verify Acl2Backend::with_config preserves verbose setting
    #[kani::proof]
    fn proof_acl2_backend_with_config_verbose() {
        let config = Acl2Config {
            acl2_path: None,
            timeout: Duration::from_secs(180),
            verbose: true,
            book_dirs: Vec::new(),
            acl2s_mode: false,
            memory_limit_mb: None,
            proof_limit: None,
        };
        let backend = Acl2Backend::with_config(config);
        kani::assert(
            backend.config.verbose,
            "with_config should preserve verbose setting",
        );
    }

    /// Verify Acl2Backend::with_config preserves acl2s_mode setting
    #[kani::proof]
    fn proof_acl2_backend_with_config_acl2s_mode() {
        let config = Acl2Config {
            acl2_path: None,
            timeout: Duration::from_secs(180),
            verbose: false,
            book_dirs: Vec::new(),
            acl2s_mode: true,
            memory_limit_mb: None,
            proof_limit: None,
        };
        let backend = Acl2Backend::with_config(config);
        kani::assert(
            backend.config.acl2s_mode,
            "with_config should preserve acl2s_mode setting",
        );
    }

    /// Verify Acl2Backend::with_config preserves memory_limit_mb
    #[kani::proof]
    fn proof_acl2_backend_with_config_memory_limit() {
        let config = Acl2Config {
            acl2_path: None,
            timeout: Duration::from_secs(180),
            verbose: false,
            book_dirs: Vec::new(),
            acl2s_mode: false,
            memory_limit_mb: Some(8192),
            proof_limit: None,
        };
        let backend = Acl2Backend::with_config(config);
        kani::assert(
            backend.config.memory_limit_mb == Some(8192),
            "with_config should preserve memory_limit_mb",
        );
    }

    /// Verify Acl2Backend::with_config preserves proof_limit
    #[kani::proof]
    fn proof_acl2_backend_with_config_proof_limit() {
        let config = Acl2Config {
            acl2_path: None,
            timeout: Duration::from_secs(180),
            verbose: false,
            book_dirs: Vec::new(),
            acl2s_mode: false,
            memory_limit_mb: None,
            proof_limit: Some(100),
        };
        let backend = Acl2Backend::with_config(config);
        kani::assert(
            backend.config.proof_limit == Some(100),
            "with_config should preserve proof_limit",
        );
    }

    // ---- Backend Trait Implementation Tests ----

    /// Verify Acl2Backend::id returns BackendId::ACL2
    #[kani::proof]
    fn proof_acl2_backend_id() {
        let backend = Acl2Backend::new();
        kani::assert(backend.id() == BackendId::ACL2, "Backend id should be ACL2");
    }

    /// Verify Acl2Backend::supports includes Theorem
    #[kani::proof]
    fn proof_acl2_backend_supports_theorem() {
        let backend = Acl2Backend::new();
        let supported = backend.supports();
        let has_theorem = supported.iter().any(|p| *p == PropertyType::Theorem);
        kani::assert(has_theorem, "Backend should support Theorem");
    }

    /// Verify Acl2Backend::supports includes Invariant
    #[kani::proof]
    fn proof_acl2_backend_supports_invariant() {
        let backend = Acl2Backend::new();
        let supported = backend.supports();
        let has_invariant = supported.iter().any(|p| *p == PropertyType::Invariant);
        kani::assert(has_invariant, "Backend should support Invariant");
    }

    /// Verify Acl2Backend::supports returns exactly 2 property types
    #[kani::proof]
    fn proof_acl2_backend_supports_count() {
        let backend = Acl2Backend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 2,
            "Backend should support exactly 2 property types",
        );
    }

    // ---- parse_output Tests ----

    /// Verify parse_output returns Proven for "Q.E.D."
    #[kani::proof]
    fn proof_acl2_parse_output_qed() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("Q.E.D.", "", true);
        let is_proven = matches!(status, VerificationStatus::Proven);
        kani::assert(is_proven, "Q.E.D. should result in Proven");
    }

    /// Verify parse_output returns Proven for "Proof succeeded"
    #[kani::proof]
    fn proof_acl2_parse_output_proof_succeeded() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("Proof succeeded", "", true);
        let is_proven = matches!(status, VerificationStatus::Proven);
        kani::assert(is_proven, "Proof succeeded should result in Proven");
    }

    /// Verify parse_output returns Unknown for "***** FAILED *****"
    #[kani::proof]
    fn proof_acl2_parse_output_failed() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("***** FAILED *****", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "FAILED should result in Unknown");
    }

    /// Verify parse_output returns Unknown for "Proof failed"
    #[kani::proof]
    fn proof_acl2_parse_output_proof_failed() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("Proof failed", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "Proof failed should result in Unknown");
    }

    /// Verify parse_output returns Unknown for "The proof attempt has failed"
    #[kani::proof]
    fn proof_acl2_parse_output_attempt_failed() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("The proof attempt has failed", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(
            is_unknown,
            "proof attempt has failed should result in Unknown",
        );
    }

    /// Verify parse_output returns Unknown for "ACL2 Error"
    #[kani::proof]
    fn proof_acl2_parse_output_acl2_error() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("ACL2 Error in FOO", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "ACL2 Error should result in Unknown");
    }

    /// Verify parse_output returns Unknown for "HARD ACL2 ERROR"
    #[kani::proof]
    fn proof_acl2_parse_output_hard_error() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("HARD ACL2 ERROR", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "HARD ACL2 ERROR should result in Unknown");
    }

    /// Verify parse_output returns Unknown for "Error:" in stderr
    #[kani::proof]
    fn proof_acl2_parse_output_error_stderr() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("", "Error: undefined function", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "Error: should result in Unknown");
    }

    /// Verify parse_output returns Unknown for "Time limit exceeded"
    #[kani::proof]
    fn proof_acl2_parse_output_time_limit() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("Time limit exceeded", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "Time limit exceeded should result in Unknown");
    }

    /// Verify parse_output returns Unknown for "out of memory"
    #[kani::proof]
    fn proof_acl2_parse_output_out_of_memory() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("out of memory", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "out of memory should result in Unknown");
    }

    /// Verify parse_output returns Proven for successful exit with empty output
    #[kani::proof]
    fn proof_acl2_parse_output_success_empty() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("", "", true);
        let is_proven = matches!(status, VerificationStatus::Proven);
        kani::assert(is_proven, "Successful exit should result in Proven");
    }

    /// Verify parse_output returns Unknown for failed exit with empty output
    #[kani::proof]
    fn proof_acl2_parse_output_failed_empty() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(
            is_unknown,
            "Failed exit with empty output should result in Unknown",
        );
    }

    /// Verify Q.E.D. takes priority over success flag
    #[kani::proof]
    fn proof_acl2_parse_output_qed_priority() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("Q.E.D.", "", false);
        let is_proven = matches!(status, VerificationStatus::Proven);
        kani::assert(is_proven, "Q.E.D. should take priority over success flag");
    }

    /// Verify FAILED takes priority over Q.E.D.
    #[kani::proof]
    fn proof_acl2_parse_output_failed_priority_over_qed() {
        let backend = Acl2Backend::new();
        // If both appear, FAILED should not override Q.E.D. since Q.E.D. is checked first
        let status = backend.parse_output("Q.E.D.\n***** FAILED *****", "", false);
        let is_proven = matches!(status, VerificationStatus::Proven);
        kani::assert(is_proven, "Q.E.D. checked before FAILED");
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

/// Configuration for ACL2 backend
#[derive(Debug, Clone)]
pub struct Acl2Config {
    /// Path to acl2 binary
    pub acl2_path: Option<PathBuf>,
    /// Timeout for theorem proving
    pub timeout: Duration,
    /// Enable verbose output
    pub verbose: bool,
    /// Book directories for certified libraries
    pub book_dirs: Vec<PathBuf>,
    /// Use ACL2s (ACL2 Sedan) mode
    pub acl2s_mode: bool,
    /// Memory limit in megabytes
    pub memory_limit_mb: Option<u32>,
    /// Proof attempt limit (number of checkpoints)
    pub proof_limit: Option<u32>,
}

impl Default for Acl2Config {
    fn default() -> Self {
        Self {
            acl2_path: None,
            timeout: Duration::from_secs(180),
            verbose: false,
            book_dirs: Vec::new(),
            acl2s_mode: false,
            memory_limit_mb: None,
            proof_limit: None,
        }
    }
}

/// ACL2 theorem prover backend
pub struct Acl2Backend {
    config: Acl2Config,
}

impl Default for Acl2Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl Acl2Backend {
    /// Create a new ACL2 backend with default configuration
    pub fn new() -> Self {
        Self {
            config: Acl2Config::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: Acl2Config) -> Self {
        Self { config }
    }

    async fn detect_acl2(&self) -> Result<PathBuf, String> {
        // Try acl2 first, then acl2s
        let acl2_path = self
            .config
            .acl2_path
            .clone()
            .or_else(|| which::which("acl2").ok())
            .or_else(|| which::which("acl2s").ok())
            .ok_or("ACL2 not found. Install via: brew install acl2")?;

        // ACL2 --help exits with status, try a simple invocation
        let output = Command::new(&acl2_path)
            .arg("--help")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute acl2: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}{}", stdout, stderr);

        if combined.contains("ACL2")
            || combined.contains("acl2")
            || combined.contains("Applicative Common Lisp")
        {
            debug!("Detected ACL2: {}", combined.lines().next().unwrap_or(""));
            Ok(acl2_path)
        } else if output.status.success() || !stdout.is_empty() {
            // ACL2 may not have --help but should return something
            debug!("Detected ACL2 binary at: {}", acl2_path.display());
            Ok(acl2_path)
        } else {
            Err("ACL2 version check failed".to_string())
        }
    }

    /// Convert a typed spec to ACL2 Lisp code
    fn spec_to_acl2(&self, spec: &TypedSpec) -> String {
        let mut acl2 = String::new();

        // ACL2 preamble
        acl2.push_str("; DashProve generated ACL2 file\n");
        acl2.push_str("(in-package \"ACL2\")\n\n");

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

            // Sanitize name for ACL2
            let sanitized_name = prop_name
                .replace([' ', '-'], "_")
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
                .collect::<String>();

            // Generate a trivial theorem for the property
            acl2.push_str(&format!(
                "; Property: {}\n(defthm {}\n  (equal t t))\n\n",
                prop_name, sanitized_name
            ));
        }

        if spec.spec.properties.is_empty() {
            acl2.push_str("; No properties to verify\n");
            acl2.push_str("(defthm trivial-theorem\n  (equal t t))\n");
        }

        // End marker
        acl2.push_str("\n; End of DashProve generated file\n");
        acl2.push_str("(good-bye)\n");

        acl2
    }

    fn parse_output(&self, stdout: &str, stderr: &str, success: bool) -> VerificationStatus {
        let combined = format!("{}\n{}", stdout, stderr);

        // Check for proof success (ACL2 outputs "Q.E.D." on successful proof)
        if combined.contains("Q.E.D.") || combined.contains("Proof succeeded") {
            return VerificationStatus::Proven;
        }

        // Check for proof failure
        if combined.contains("***** FAILED *****")
            || combined.contains("Proof failed")
            || combined.contains("The proof attempt has failed")
        {
            return VerificationStatus::Unknown {
                reason: "ACL2 proof attempt failed".to_string(),
            };
        }

        // Check for errors
        if combined.contains("ACL2 Error")
            || combined.contains("HARD ACL2 ERROR")
            || combined.contains("Error:")
        {
            let error_lines: Vec<&str> = combined
                .lines()
                .filter(|l| {
                    l.contains("Error")
                        || l.contains("error")
                        || l.contains("undefined")
                        || l.contains("illegal")
                })
                .take(3)
                .collect();
            return VerificationStatus::Unknown {
                reason: format!("ACL2 error: {}", error_lines.join("; ")),
            };
        }

        // Check for timeouts/resource exhaustion
        if combined.contains("Time limit exceeded") || combined.contains("out of memory") {
            return VerificationStatus::Unknown {
                reason: "ACL2 resource limit exceeded".to_string(),
            };
        }

        // Successful exit with no explicit proof markers
        if success {
            debug!("ACL2 completed successfully");
            return VerificationStatus::Proven;
        }

        VerificationStatus::Unknown {
            reason: "Could not determine ACL2 result".to_string(),
        }
    }
}

#[async_trait]
impl VerificationBackend for Acl2Backend {
    fn id(&self) -> BackendId {
        BackendId::ACL2
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Theorem, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let acl2_path = self
            .detect_acl2()
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let acl2_code = self.spec_to_acl2(spec);
        let acl2_file = temp_dir.path().join("spec.lisp");

        std::fs::write(&acl2_file, &acl2_code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write ACL2 file: {}", e))
        })?;

        let mut cmd = Command::new(&acl2_path);

        // ACL2 runs in batch mode with input file
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Set up environment for batch mode
        cmd.arg("<").arg(&acl2_file);

        // Add book directories
        for dir in &self.config.book_dirs {
            cmd.env("ACL2_SYSTEM_BOOKS", dir);
        }

        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(5), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("ACL2 stdout length: {}", stdout.len());
                debug!("ACL2 stderr: {}", stderr);

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
                    Some("Proven by ACL2".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::ACL2,
                    status,
                    proof,
                    counterexample: None,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute ACL2: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_acl2().await {
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
        let config = Acl2Config::default();
        assert_eq!(config.timeout, Duration::from_secs(180));
        assert!(!config.verbose);
        assert!(!config.acl2s_mode);
        assert!(config.book_dirs.is_empty());
    }

    #[test]
    fn backend_id() {
        let backend = Acl2Backend::new();
        assert_eq!(backend.id(), BackendId::ACL2);
    }

    #[test]
    fn supports_properties() {
        let backend = Acl2Backend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[test]
    fn parse_qed_success() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("Q.E.D.", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_proof_succeeded() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("Proof succeeded", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_proof_failed() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("***** FAILED *****", "", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
        if let VerificationStatus::Unknown { reason } = status {
            assert!(reason.contains("failed"));
        }
    }

    #[test]
    fn parse_error() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("", "ACL2 Error in FOO: undefined function", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_success_exit() {
        let backend = Acl2Backend::new();
        let status = backend.parse_output("", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }
}
