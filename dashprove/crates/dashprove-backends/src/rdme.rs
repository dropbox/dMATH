//! cargo-rdme README from rustdoc backend
//!
//! Rdme generates README files from rustdoc documentation,
//! ensuring README stays in sync with crate documentation.
//!
//! See: <https://github.com/orium/cargo-rdme>

use crate::counterexample::{FailedCheck, StructuredCounterexample};
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::process::Command;

/// Configuration for rdme README generation backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdmeConfig {
    pub timeout: Duration,
    pub check_only: bool,
    pub workspace: bool,
}

impl Default for RdmeConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            check_only: true,
            workspace: false,
        }
    }
}

impl RdmeConfig {
    pub fn with_check_only(mut self, enabled: bool) -> Self {
        self.check_only = enabled;
        self
    }

    pub fn with_workspace(mut self, enabled: bool) -> Self {
        self.workspace = enabled;
        self
    }
}

/// README sync statistics from rdme
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RdmeStats {
    pub files_checked: u64,
    pub files_in_sync: u64,
    pub files_out_of_sync: u64,
    pub out_of_sync_files: Vec<String>,
}

/// cargo-rdme README generation backend
pub struct RdmeBackend {
    config: RdmeConfig,
}

impl Default for RdmeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RdmeBackend {
    pub fn new() -> Self {
        Self {
            config: RdmeConfig::default(),
        }
    }

    pub fn with_config(config: RdmeConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<PathBuf, String> {
        which::which("cargo-rdme")
            .map_err(|_| "cargo-rdme not found. Install via cargo install cargo-rdme".to_string())
    }

    /// Check README synchronization with docs
    pub async fn check_readme(
        &self,
        crate_path: &std::path::Path,
    ) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        self.detect().await.map_err(BackendError::Unavailable)?;

        let mut cmd = Command::new("cargo");
        cmd.current_dir(crate_path);
        cmd.arg("rdme");

        if self.config.check_only {
            cmd.arg("--check");
        }

        if self.config.workspace {
            cmd.arg("--workspace");
        }

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run cargo rdme: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let stats = self.parse_output(&stdout, &stderr, output.status.success());
        let (status, counterexample) = self.evaluate_results(&stats);

        let mut diagnostics = Vec::new();
        diagnostics.push(format!(
            "README files: {} checked, {} in sync, {} out of sync",
            stats.files_checked, stats.files_in_sync, stats.files_out_of_sync
        ));

        Ok(BackendResult {
            backend: BackendId::Rdme,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    fn parse_output(&self, stdout: &str, stderr: &str, success: bool) -> RdmeStats {
        let mut stats = RdmeStats::default();
        let combined = format!("{}\n{}", stdout, stderr);

        // Check for sync status
        for line in combined.lines() {
            let line_lower = line.to_lowercase();

            // Success messages
            if line_lower.contains("readme is up-to-date") || line_lower.contains("up to date") {
                stats.files_checked += 1;
                stats.files_in_sync += 1;
            }

            // Out of sync messages
            if line_lower.contains("out of date")
                || line_lower.contains("not up-to-date")
                || line_lower.contains("differs from")
            {
                stats.files_checked += 1;
                stats.files_out_of_sync += 1;
                // Try to extract file name
                if line.contains("README") {
                    stats.out_of_sync_files.push(line.trim().to_string());
                }
            }
        }

        // If no explicit info but command failed, assume out of sync
        if stats.files_checked == 0 {
            stats.files_checked = 1;
            if success {
                stats.files_in_sync = 1;
            } else {
                stats.files_out_of_sync = 1;
                stats.out_of_sync_files.push("README.md".to_string());
            }
        }

        stats
    }

    fn evaluate_results(
        &self,
        stats: &RdmeStats,
    ) -> (VerificationStatus, Option<StructuredCounterexample>) {
        if stats.files_out_of_sync > 0 {
            let failed_checks: Vec<FailedCheck> = stats
                .out_of_sync_files
                .iter()
                .map(|file| FailedCheck {
                    check_id: format!("readme_out_of_sync:{}", file),
                    description: format!("README '{}' is out of sync with docs", file),
                    location: None,
                    function: None,
                })
                .collect();

            let failed_checks = if failed_checks.is_empty() {
                vec![FailedCheck {
                    check_id: "readme_out_of_sync".to_string(),
                    description: format!("{} README file(s) out of sync", stats.files_out_of_sync),
                    location: None,
                    function: None,
                }]
            } else {
                failed_checks
            };

            return (
                VerificationStatus::Disproven,
                Some(StructuredCounterexample {
                    witness: HashMap::new(),
                    failed_checks,
                    playback_test: Some("Run `cargo rdme` to update README".to_string()),
                    trace: vec![],
                    raw: Some(format!(
                        "{} README file(s) out of sync with documentation",
                        stats.files_out_of_sync
                    )),
                    minimized: false,
                }),
            );
        }

        if stats.files_checked == 0 {
            return (
                VerificationStatus::Unknown {
                    reason: "No README files checked".to_string(),
                },
                None,
            );
        }

        (VerificationStatus::Proven, None)
    }
}

#[async_trait]
impl VerificationBackend for RdmeBackend {
    fn id(&self) -> BackendId {
        BackendId::Rdme
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Lint]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        self.detect().await.map_err(BackendError::Unavailable)?;

        let mut diagnostics = Vec::new();
        diagnostics.push("rdme ready for README synchronization".to_string());
        if self.config.check_only {
            diagnostics.push("Check-only mode (no modifications)".to_string());
        }

        Ok(BackendResult {
            backend: BackendId::Rdme,
            status: VerificationStatus::Proven,
            proof: None,
            counterexample: None,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== RdmeConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults_timeout() {
        let config = RdmeConfig::default();
        assert!(config.timeout == Duration::from_secs(60));
    }

    #[kani::proof]
    fn verify_config_defaults_check_only() {
        let config = RdmeConfig::default();
        assert!(config.check_only);
    }

    #[kani::proof]
    fn verify_config_defaults_workspace() {
        let config = RdmeConfig::default();
        assert!(!config.workspace);
    }

    // ===== RdmeConfig builders =====

    #[kani::proof]
    fn verify_config_with_check_only() {
        let config = RdmeConfig::default().with_check_only(false);
        assert!(!config.check_only);
    }

    #[kani::proof]
    fn verify_config_with_workspace() {
        let config = RdmeConfig::default().with_workspace(true);
        assert!(config.workspace);
    }

    // ===== Backend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = RdmeBackend::new();
        assert!(backend.config.timeout == Duration::from_secs(60));
        assert!(backend.config.check_only);
    }

    #[kani::proof]
    fn verify_backend_default_matches_new() {
        let b1 = RdmeBackend::new();
        let b2 = RdmeBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.check_only == b2.config.check_only);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_values() {
        let config = RdmeConfig {
            timeout: Duration::from_secs(120),
            check_only: false,
            workspace: true,
        };
        let backend = RdmeBackend::with_config(config);
        assert!(backend.config.timeout == Duration::from_secs(120));
        assert!(!backend.config.check_only);
        assert!(backend.config.workspace);
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = RdmeBackend::new();
        assert!(matches!(backend.id(), BackendId::Rdme));
    }

    #[kani::proof]
    fn verify_supports_lint() {
        let backend = RdmeBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Lint));
        assert!(supported.len() == 1);
    }

    // ===== Output parsing =====

    #[kani::proof]
    fn verify_parse_output_in_sync() {
        let backend = RdmeBackend::new();
        let stdout = "README is up-to-date\n";
        let stats = backend.parse_output(stdout, "", true);
        assert!(stats.files_checked == 1);
        assert!(stats.files_in_sync == 1);
        assert!(stats.files_out_of_sync == 0);
    }

    #[kani::proof]
    fn verify_parse_output_up_to_date() {
        let backend = RdmeBackend::new();
        let stdout = "up to date";
        let stats = backend.parse_output(stdout, "", true);
        assert!(stats.files_in_sync >= 1);
    }

    #[kani::proof]
    fn verify_parse_output_out_of_sync() {
        let backend = RdmeBackend::new();
        let stderr = "README.md is out of date\n";
        let stats = backend.parse_output("", stderr, false);
        assert!(stats.files_out_of_sync >= 1);
    }

    #[kani::proof]
    fn verify_parse_output_not_up_to_date() {
        let backend = RdmeBackend::new();
        let stderr = "README is not up-to-date";
        let stats = backend.parse_output("", stderr, false);
        assert!(stats.files_out_of_sync >= 1);
    }

    #[kani::proof]
    fn verify_parse_output_differs_from() {
        let backend = RdmeBackend::new();
        let stderr = "README differs from documentation";
        let stats = backend.parse_output("", stderr, false);
        assert!(stats.files_out_of_sync >= 1);
    }

    #[kani::proof]
    fn verify_parse_output_default_success() {
        let backend = RdmeBackend::new();
        let stats = backend.parse_output("", "", true);
        assert!(stats.files_checked == 1);
        assert!(stats.files_in_sync == 1);
    }

    #[kani::proof]
    fn verify_parse_output_default_failure() {
        let backend = RdmeBackend::new();
        let stats = backend.parse_output("", "", false);
        assert!(stats.files_checked == 1);
        assert!(stats.files_out_of_sync == 1);
    }

    // ===== Result evaluation =====

    #[kani::proof]
    fn verify_evaluate_results_pass() {
        let backend = RdmeBackend::new();
        let stats = RdmeStats {
            files_checked: 1,
            files_in_sync: 1,
            files_out_of_sync: 0,
            out_of_sync_files: vec![],
        };
        let (status, cex) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(cex.is_none());
    }

    #[kani::proof]
    fn verify_evaluate_results_fail() {
        let backend = RdmeBackend::new();
        let stats = RdmeStats {
            files_checked: 1,
            files_in_sync: 0,
            files_out_of_sync: 1,
            out_of_sync_files: vec!["README.md".to_string()],
        };
        let (status, cex) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(cex.is_some());
    }

    #[kani::proof]
    fn verify_evaluate_results_no_files() {
        let backend = RdmeBackend::new();
        let stats = RdmeStats {
            files_checked: 0,
            files_in_sync: 0,
            files_out_of_sync: 0,
            out_of_sync_files: vec![],
        };
        let (status, _) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[kani::proof]
    fn verify_evaluate_results_multiple_out_of_sync() {
        let backend = RdmeBackend::new();
        let stats = RdmeStats {
            files_checked: 3,
            files_in_sync: 1,
            files_out_of_sync: 2,
            out_of_sync_files: vec!["README.md".to_string(), "OTHER.md".to_string()],
        };
        let (status, cex) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
        let cex = cex.unwrap();
        assert!(cex.failed_checks.len() == 2);
    }

    #[kani::proof]
    fn verify_evaluate_results_out_of_sync_no_files_listed() {
        let backend = RdmeBackend::new();
        let stats = RdmeStats {
            files_checked: 1,
            files_in_sync: 0,
            files_out_of_sync: 1,
            out_of_sync_files: vec![],
        };
        let (status, cex) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
        let cex = cex.unwrap();
        // Should create a default failed check
        assert!(cex.failed_checks.len() == 1);
        assert!(cex.failed_checks[0].check_id == "readme_out_of_sync");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_id() {
        assert_eq!(RdmeBackend::new().id(), BackendId::Rdme);
    }

    #[test]
    fn test_config_default() {
        let config = RdmeConfig::default();
        assert!(config.check_only);
        assert!(!config.workspace);
    }

    #[test]
    fn test_config_builder() {
        let config = RdmeConfig::default()
            .with_check_only(false)
            .with_workspace(true);

        assert!(!config.check_only);
        assert!(config.workspace);
    }

    #[test]
    fn test_parse_output_in_sync() {
        let backend = RdmeBackend::new();
        let stdout = "README is up-to-date\n";
        let stats = backend.parse_output(stdout, "", true);

        assert_eq!(stats.files_checked, 1);
        assert_eq!(stats.files_in_sync, 1);
        assert_eq!(stats.files_out_of_sync, 0);
    }

    #[test]
    fn test_parse_output_out_of_sync() {
        let backend = RdmeBackend::new();
        let stderr = "README.md is out of date\n";
        let stats = backend.parse_output("", stderr, false);

        assert_eq!(stats.files_out_of_sync, 1);
    }

    #[test]
    fn test_evaluate_results_pass() {
        let backend = RdmeBackend::new();
        let stats = RdmeStats {
            files_checked: 1,
            files_in_sync: 1,
            files_out_of_sync: 0,
            out_of_sync_files: vec![],
        };

        let (status, _) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn test_evaluate_results_fail() {
        let backend = RdmeBackend::new();
        let stats = RdmeStats {
            files_checked: 1,
            files_in_sync: 0,
            files_out_of_sync: 1,
            out_of_sync_files: vec!["README.md".to_string()],
        };

        let (status, cex) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(cex.is_some());
    }

    #[tokio::test]
    async fn test_health_check() {
        let backend = RdmeBackend::new();
        let status = backend.health_check().await;
        match status {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(reason.contains("cargo-rdme"));
            }
            _ => panic!("Unexpected status"),
        }
    }

    #[tokio::test]
    async fn test_verify_returns_result_or_unavailable() {
        use dashprove_usl::{parse, typecheck};

        let backend = RdmeBackend::new();
        let spec = parse("theorem test { true }").expect("parse");
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;
        match result {
            Ok(r) => {
                assert_eq!(r.backend, BackendId::Rdme);
            }
            Err(BackendError::Unavailable(_)) => {}
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}
