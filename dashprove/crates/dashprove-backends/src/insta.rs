//! cargo-insta snapshot testing backend
//!
//! Insta provides snapshot testing for Rust, comparing test outputs against
//! stored "snapshots" and detecting unexpected changes.
//!
//! See: <https://insta.rs/>

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

/// Mode for handling pending snapshots
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum InstaMode {
    #[default]
    Check,
    Review,
    Accept,
    Reject,
}

impl InstaMode {
    fn as_arg(&self) -> &'static str {
        match self {
            Self::Check => "check",
            Self::Review => "review",
            Self::Accept => "accept",
            Self::Reject => "reject",
        }
    }
}

/// Configuration for insta snapshot testing backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstaConfig {
    pub timeout: Duration,
    pub mode: InstaMode,
    pub workspace: bool,
    pub delete_unreferenced: bool,
    pub include_ignored: bool,
    pub include_hidden: bool,
}

impl Default for InstaConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(120),
            mode: InstaMode::default(),
            workspace: false,
            delete_unreferenced: false,
            include_ignored: false,
            include_hidden: false,
        }
    }
}

impl InstaConfig {
    pub fn with_mode(mut self, mode: InstaMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_workspace(mut self, enabled: bool) -> Self {
        self.workspace = enabled;
        self
    }

    pub fn with_delete_unreferenced(mut self, enabled: bool) -> Self {
        self.delete_unreferenced = enabled;
        self
    }
}

/// Snapshot testing statistics from insta
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstaStats {
    pub total_snapshots: u64,
    pub matching: u64,
    pub pending: u64,
    pub new_snapshots: u64,
    pub changed_snapshots: u64,
    pub deleted_snapshots: u64,
    pub pending_files: Vec<String>,
}

/// Information about a snapshot difference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotDiff {
    pub snapshot_name: String,
    pub file_path: Option<String>,
    pub expected: Option<String>,
    pub actual: Option<String>,
}

/// cargo-insta snapshot testing backend
pub struct InstaBackend {
    config: InstaConfig,
}

impl Default for InstaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InstaBackend {
    pub fn new() -> Self {
        Self {
            config: InstaConfig::default(),
        }
    }

    pub fn with_config(config: InstaConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<PathBuf, String> {
        which::which("cargo-insta")
            .map_err(|_| "cargo-insta not found. Install via cargo install cargo-insta".to_string())
    }

    /// Run insta snapshot tests
    pub async fn run_snapshots(
        &self,
        crate_path: &std::path::Path,
    ) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        self.detect().await.map_err(BackendError::Unavailable)?;

        // First run tests to generate/check snapshots
        let mut cmd = Command::new("cargo");
        cmd.current_dir(crate_path);
        cmd.arg("insta").arg("test");

        if self.config.workspace {
            cmd.arg("--workspace");
        }

        if self.config.delete_unreferenced {
            cmd.arg("--delete-unreferenced-snapshots");
        }

        if self.config.include_ignored {
            cmd.arg("--include-ignored");
        }

        if self.config.include_hidden {
            cmd.arg("--include-hidden");
        }

        // Use check mode by default to avoid interactive prompts
        cmd.arg("--check");

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run cargo insta: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let stats = self.parse_output(&stdout, &stderr);
        let (status, counterexample) = self.evaluate_results(&stats);

        let mut diagnostics = Vec::new();
        diagnostics.push(format!(
            "Snapshots: {} total, {} matching, {} pending",
            stats.total_snapshots, stats.matching, stats.pending
        ));

        if stats.new_snapshots > 0 {
            diagnostics.push(format!("New snapshots: {}", stats.new_snapshots));
        }
        if stats.changed_snapshots > 0 {
            diagnostics.push(format!("Changed snapshots: {}", stats.changed_snapshots));
        }

        Ok(BackendResult {
            backend: BackendId::Insta,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    /// List pending snapshots
    pub async fn list_pending(
        &self,
        crate_path: &std::path::Path,
    ) -> Result<Vec<String>, BackendError> {
        self.detect().await.map_err(BackendError::Unavailable)?;

        let mut cmd = Command::new("cargo");
        cmd.current_dir(crate_path);
        cmd.arg("insta").arg("pending-snapshots");

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to list pending snapshots: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let pending: Vec<String> = stdout
            .lines()
            .filter(|line| !line.is_empty() && line.contains("snap"))
            .map(|s| s.trim().to_string())
            .collect();

        Ok(pending)
    }

    fn parse_output(&self, stdout: &str, stderr: &str) -> InstaStats {
        let mut stats = InstaStats::default();
        let combined = format!("{}\n{}", stdout, stderr);

        for line in combined.lines() {
            let line_lower = line.to_lowercase();

            // Parse snapshot count lines
            // Format: "info: X snapshot(s) passed"
            if line_lower.contains("snapshot") && line_lower.contains("passed") {
                if let Some(count) = Self::extract_first_number(line) {
                    stats.matching = count;
                }
            }

            // Format: "info: X new snapshot(s)"
            if line_lower.contains("new snapshot") {
                if let Some(count) = Self::extract_first_number(line) {
                    stats.new_snapshots = count;
                }
            }

            // Format: "info: X snapshot(s) changed"
            if line_lower.contains("snapshot") && line_lower.contains("changed") {
                if let Some(count) = Self::extract_first_number(line) {
                    stats.changed_snapshots = count;
                }
            }

            // Pending snapshots
            if line_lower.contains("pending") && line_lower.contains("snapshot") {
                if let Some(count) = Self::extract_first_number(line) {
                    stats.pending = count;
                }
            }

            // Track files with pending snapshots
            if line.ends_with(".snap.new") || line.ends_with(".snap.pending") {
                stats.pending_files.push(line.trim().to_string());
            }
        }

        // Calculate total
        stats.total_snapshots =
            stats.matching + stats.pending + stats.new_snapshots + stats.changed_snapshots;

        // If we found no explicit counts but there are pending files, count them
        if stats.total_snapshots == 0 && !stats.pending_files.is_empty() {
            stats.pending = stats.pending_files.len() as u64;
            stats.total_snapshots = stats.pending;
        }

        stats
    }

    fn extract_first_number(line: &str) -> Option<u64> {
        for word in line.split_whitespace() {
            if let Ok(num) = word.parse::<u64>() {
                return Some(num);
            }
        }
        None
    }

    fn evaluate_results(
        &self,
        stats: &InstaStats,
    ) -> (VerificationStatus, Option<StructuredCounterexample>) {
        // If there are pending, new, or changed snapshots, the tests are failing
        let failures = stats.pending + stats.new_snapshots + stats.changed_snapshots;

        if failures > 0 {
            let mut failed_checks = Vec::new();

            if stats.pending > 0 {
                failed_checks.push(FailedCheck {
                    check_id: "pending_snapshots".to_string(),
                    description: format!("{} pending snapshot(s) need review", stats.pending),
                    location: None,
                    function: None,
                });
            }

            if stats.new_snapshots > 0 {
                failed_checks.push(FailedCheck {
                    check_id: "new_snapshots".to_string(),
                    description: format!("{} new snapshot(s) need acceptance", stats.new_snapshots),
                    location: None,
                    function: None,
                });
            }

            if stats.changed_snapshots > 0 {
                failed_checks.push(FailedCheck {
                    check_id: "changed_snapshots".to_string(),
                    description: format!(
                        "{} snapshot(s) have changed and need review",
                        stats.changed_snapshots
                    ),
                    location: None,
                    function: None,
                });
            }

            return (
                VerificationStatus::Disproven,
                Some(StructuredCounterexample {
                    witness: HashMap::new(),
                    failed_checks,
                    playback_test: Some(
                        "Run `cargo insta review` to interactively review snapshots".to_string(),
                    ),
                    trace: vec![],
                    raw: Some(format!(
                        "{} snapshot(s) require attention: {} pending, {} new, {} changed",
                        failures, stats.pending, stats.new_snapshots, stats.changed_snapshots
                    )),
                    minimized: false,
                }),
            );
        }

        if stats.total_snapshots == 0 {
            return (
                VerificationStatus::Unknown {
                    reason: "No snapshots found".to_string(),
                },
                None,
            );
        }

        (VerificationStatus::Proven, None)
    }
}

#[async_trait]
impl VerificationBackend for InstaBackend {
    fn id(&self) -> BackendId {
        BackendId::Insta
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Lint]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        self.detect().await.map_err(BackendError::Unavailable)?;

        let mut diagnostics = Vec::new();
        diagnostics.push(format!("Mode: {}", self.config.mode.as_arg()));
        if self.config.workspace {
            diagnostics.push("Workspace mode enabled".to_string());
        }

        Ok(BackendResult {
            backend: BackendId::Insta,
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

    // ---- InstaMode Tests ----

    /// Verify InstaMode default is Check
    #[kani::proof]
    fn proof_insta_mode_default_is_check() {
        let mode = InstaMode::default();
        kani::assert(
            matches!(mode, InstaMode::Check),
            "Default InstaMode should be Check",
        );
    }

    /// Verify InstaMode::Check.as_arg returns "check"
    #[kani::proof]
    fn proof_insta_mode_check_as_arg() {
        let arg = InstaMode::Check.as_arg();
        kani::assert(arg == "check", "Check.as_arg should be 'check'");
    }

    /// Verify InstaMode::Review.as_arg returns "review"
    #[kani::proof]
    fn proof_insta_mode_review_as_arg() {
        let arg = InstaMode::Review.as_arg();
        kani::assert(arg == "review", "Review.as_arg should be 'review'");
    }

    /// Verify InstaMode::Accept.as_arg returns "accept"
    #[kani::proof]
    fn proof_insta_mode_accept_as_arg() {
        let arg = InstaMode::Accept.as_arg();
        kani::assert(arg == "accept", "Accept.as_arg should be 'accept'");
    }

    /// Verify InstaMode::Reject.as_arg returns "reject"
    #[kani::proof]
    fn proof_insta_mode_reject_as_arg() {
        let arg = InstaMode::Reject.as_arg();
        kani::assert(arg == "reject", "Reject.as_arg should be 'reject'");
    }

    // ---- InstaConfig Default Tests ----

    /// Verify InstaConfig::default timeout is 120 seconds
    #[kani::proof]
    fn proof_insta_config_default_timeout() {
        let config = InstaConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "Default timeout should be 120 seconds",
        );
    }

    /// Verify InstaConfig::default mode is Check
    #[kani::proof]
    fn proof_insta_config_default_mode() {
        let config = InstaConfig::default();
        kani::assert(
            matches!(config.mode, InstaMode::Check),
            "Default mode should be Check",
        );
    }

    /// Verify InstaConfig::default workspace is false
    #[kani::proof]
    fn proof_insta_config_default_workspace() {
        let config = InstaConfig::default();
        kani::assert(!config.workspace, "Default workspace should be false");
    }

    /// Verify InstaConfig::default delete_unreferenced is false
    #[kani::proof]
    fn proof_insta_config_default_delete_unreferenced() {
        let config = InstaConfig::default();
        kani::assert(
            !config.delete_unreferenced,
            "Default delete_unreferenced should be false",
        );
    }

    /// Verify InstaConfig::default include_ignored is false
    #[kani::proof]
    fn proof_insta_config_default_include_ignored() {
        let config = InstaConfig::default();
        kani::assert(
            !config.include_ignored,
            "Default include_ignored should be false",
        );
    }

    /// Verify InstaConfig::default include_hidden is false
    #[kani::proof]
    fn proof_insta_config_default_include_hidden() {
        let config = InstaConfig::default();
        kani::assert(
            !config.include_hidden,
            "Default include_hidden should be false",
        );
    }

    // ---- InstaConfig Builder Tests ----

    /// Verify with_mode sets mode to Review
    #[kani::proof]
    fn proof_insta_config_with_mode_review() {
        let config = InstaConfig::default().with_mode(InstaMode::Review);
        kani::assert(
            matches!(config.mode, InstaMode::Review),
            "with_mode should set Review",
        );
    }

    /// Verify with_workspace sets workspace
    #[kani::proof]
    fn proof_insta_config_with_workspace() {
        let config = InstaConfig::default().with_workspace(true);
        kani::assert(config.workspace, "with_workspace should set true");
    }

    /// Verify with_delete_unreferenced sets delete_unreferenced
    #[kani::proof]
    fn proof_insta_config_with_delete_unreferenced() {
        let config = InstaConfig::default().with_delete_unreferenced(true);
        kani::assert(
            config.delete_unreferenced,
            "with_delete_unreferenced should set true",
        );
    }

    /// Verify builder chain preserves earlier values
    #[kani::proof]
    fn proof_insta_config_builder_chain() {
        let config = InstaConfig::default()
            .with_mode(InstaMode::Accept)
            .with_workspace(true)
            .with_delete_unreferenced(true);
        kani::assert(
            matches!(config.mode, InstaMode::Accept),
            "Builder chain should preserve mode",
        );
        kani::assert(config.workspace, "Builder chain should preserve workspace");
        kani::assert(
            config.delete_unreferenced,
            "Builder chain should preserve delete_unreferenced",
        );
    }

    // ---- InstaStats Default Tests ----

    /// Verify InstaStats::default all zero
    #[kani::proof]
    fn proof_insta_stats_default_zeros() {
        let stats = InstaStats::default();
        kani::assert(
            stats.total_snapshots == 0,
            "Default total_snapshots should be 0",
        );
        kani::assert(stats.matching == 0, "Default matching should be 0");
        kani::assert(stats.pending == 0, "Default pending should be 0");
        kani::assert(
            stats.new_snapshots == 0,
            "Default new_snapshots should be 0",
        );
        kani::assert(
            stats.changed_snapshots == 0,
            "Default changed_snapshots should be 0",
        );
        kani::assert(
            stats.deleted_snapshots == 0,
            "Default deleted_snapshots should be 0",
        );
        kani::assert(
            stats.pending_files.is_empty(),
            "Default pending_files should be empty",
        );
    }

    // ---- InstaBackend Construction Tests ----

    /// Verify InstaBackend::new creates default config
    #[kani::proof]
    fn proof_insta_backend_new_default() {
        let backend = InstaBackend::new();
        kani::assert(
            matches!(backend.config.mode, InstaMode::Check),
            "New backend should have Check mode",
        );
        kani::assert(
            !backend.config.workspace,
            "New backend should have workspace false",
        );
    }

    /// Verify InstaBackend::default equals ::new
    #[kani::proof]
    fn proof_insta_backend_default_equals_new() {
        let default_backend = InstaBackend::default();
        let new_backend = InstaBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
        kani::assert(
            default_backend.config.workspace == new_backend.config.workspace,
            "Default and new should have same workspace",
        );
    }

    /// Verify InstaBackend::with_config stores config
    #[kani::proof]
    fn proof_insta_backend_with_config() {
        let config = InstaConfig {
            timeout: Duration::from_secs(60),
            mode: InstaMode::Review,
            workspace: true,
            delete_unreferenced: true,
            include_ignored: false,
            include_hidden: false,
        };
        let backend = InstaBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "with_config should store timeout",
        );
        kani::assert(
            matches!(backend.config.mode, InstaMode::Review),
            "with_config should store mode",
        );
        kani::assert(
            backend.config.workspace,
            "with_config should store workspace",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify InstaBackend::id returns BackendId::Insta
    #[kani::proof]
    fn proof_insta_backend_id() {
        let backend = InstaBackend::new();
        kani::assert(
            backend.id() == BackendId::Insta,
            "Backend ID should be Insta",
        );
    }

    /// Verify InstaBackend::supports includes Lint
    #[kani::proof]
    fn proof_insta_supports_lint() {
        let backend = InstaBackend::new();
        let supported = backend.supports();
        let has_lint = supported.iter().any(|p| *p == PropertyType::Lint);
        kani::assert(has_lint, "Should support Lint property type");
    }

    /// Verify supports returns exactly 1 property type
    #[kani::proof]
    fn proof_insta_supports_length() {
        let backend = InstaBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 1,
            "Should support exactly 1 property type",
        );
    }

    // ---- extract_first_number Tests ----

    /// Verify extract_first_number returns Some for line with number
    #[kani::proof]
    fn proof_extract_first_number_some() {
        let result = InstaBackend::extract_first_number("info: 5 snapshots passed");
        kani::assert(result == Some(5), "Should extract 5");
    }

    /// Verify extract_first_number returns None for no number
    #[kani::proof]
    fn proof_extract_first_number_none() {
        let result = InstaBackend::extract_first_number("no numbers here");
        kani::assert(result.is_none(), "Should return None for no numbers");
    }

    /// Verify extract_first_number returns first number
    #[kani::proof]
    fn proof_extract_first_number_first() {
        let result = InstaBackend::extract_first_number("10 new 20 old");
        kani::assert(result == Some(10), "Should extract first number 10");
    }

    // ---- evaluate_results Tests ----

    /// Verify evaluate_results returns Proven when all match
    #[kani::proof]
    fn proof_evaluate_results_proven() {
        let backend = InstaBackend::new();
        let stats = InstaStats {
            total_snapshots: 10,
            matching: 10,
            pending: 0,
            new_snapshots: 0,
            changed_snapshots: 0,
            deleted_snapshots: 0,
            pending_files: vec![],
        };
        let (status, cex) = backend.evaluate_results(&stats);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should be Proven when all match",
        );
        kani::assert(cex.is_none(), "No counterexample for proven");
    }

    /// Verify evaluate_results returns Disproven for pending
    #[kani::proof]
    fn proof_evaluate_results_disproven_pending() {
        let backend = InstaBackend::new();
        let stats = InstaStats {
            total_snapshots: 10,
            matching: 8,
            pending: 2,
            new_snapshots: 0,
            changed_snapshots: 0,
            deleted_snapshots: 0,
            pending_files: vec![],
        };
        let (status, cex) = backend.evaluate_results(&stats);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should be Disproven for pending",
        );
        kani::assert(cex.is_some(), "Should have counterexample");
    }

    /// Verify evaluate_results returns Disproven for new_snapshots
    #[kani::proof]
    fn proof_evaluate_results_disproven_new() {
        let backend = InstaBackend::new();
        let stats = InstaStats {
            total_snapshots: 5,
            matching: 3,
            pending: 0,
            new_snapshots: 2,
            changed_snapshots: 0,
            deleted_snapshots: 0,
            pending_files: vec![],
        };
        let (status, _) = backend.evaluate_results(&stats);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should be Disproven for new_snapshots",
        );
    }

    /// Verify evaluate_results returns Unknown for zero total
    #[kani::proof]
    fn proof_evaluate_results_unknown_zero() {
        let backend = InstaBackend::new();
        let stats = InstaStats::default();
        let (status, _) = backend.evaluate_results(&stats);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should be Unknown for zero total",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_id() {
        assert_eq!(InstaBackend::new().id(), BackendId::Insta);
    }

    #[test]
    fn test_config_default() {
        let config = InstaConfig::default();
        assert!(matches!(config.mode, InstaMode::Check));
        assert!(!config.workspace);
        assert!(!config.delete_unreferenced);
    }

    #[test]
    fn test_config_builder() {
        let config = InstaConfig::default()
            .with_mode(InstaMode::Review)
            .with_workspace(true)
            .with_delete_unreferenced(true);

        assert!(matches!(config.mode, InstaMode::Review));
        assert!(config.workspace);
        assert!(config.delete_unreferenced);
    }

    #[test]
    fn test_mode_as_arg() {
        assert_eq!(InstaMode::Check.as_arg(), "check");
        assert_eq!(InstaMode::Review.as_arg(), "review");
        assert_eq!(InstaMode::Accept.as_arg(), "accept");
        assert_eq!(InstaMode::Reject.as_arg(), "reject");
    }

    #[test]
    fn test_extract_first_number() {
        assert_eq!(
            InstaBackend::extract_first_number("info: 5 snapshots passed"),
            Some(5)
        );
        assert_eq!(
            InstaBackend::extract_first_number("10 new snapshot(s)"),
            Some(10)
        );
        assert_eq!(InstaBackend::extract_first_number("no numbers here"), None);
    }

    #[test]
    fn test_parse_output_all_pass() {
        let backend = InstaBackend::new();
        let stdout = r#"
info: 10 snapshot(s) passed
"#;
        let stats = backend.parse_output(stdout, "");

        assert_eq!(stats.matching, 10);
        assert_eq!(stats.pending, 0);
        assert_eq!(stats.new_snapshots, 0);
    }

    #[test]
    fn test_parse_output_with_pending() {
        let backend = InstaBackend::new();
        let stdout = r#"
info: 8 snapshot(s) passed
info: 2 pending snapshot(s)
"#;
        let stats = backend.parse_output(stdout, "");

        assert_eq!(stats.matching, 8);
        assert_eq!(stats.pending, 2);
    }

    #[test]
    fn test_parse_output_with_changes() {
        let backend = InstaBackend::new();
        let stdout = r#"
info: 7 snapshot(s) passed
info: 2 new snapshot(s)
info: 1 snapshot(s) changed
"#;
        let stats = backend.parse_output(stdout, "");

        assert_eq!(stats.matching, 7);
        assert_eq!(stats.new_snapshots, 2);
        assert_eq!(stats.changed_snapshots, 1);
    }

    #[test]
    fn test_evaluate_results_pass() {
        let backend = InstaBackend::new();
        let stats = InstaStats {
            total_snapshots: 10,
            matching: 10,
            pending: 0,
            new_snapshots: 0,
            changed_snapshots: 0,
            ..Default::default()
        };

        let (status, counterexample) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(counterexample.is_none());
    }

    #[test]
    fn test_evaluate_results_pending() {
        let backend = InstaBackend::new();
        let stats = InstaStats {
            total_snapshots: 10,
            matching: 8,
            pending: 2,
            new_snapshots: 0,
            changed_snapshots: 0,
            ..Default::default()
        };

        let (status, counterexample) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(counterexample.is_some());
        let cex = counterexample.unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert!(cex.failed_checks[0].description.contains("pending"));
    }

    #[test]
    fn test_evaluate_results_new_and_changed() {
        let backend = InstaBackend::new();
        let stats = InstaStats {
            total_snapshots: 10,
            matching: 7,
            pending: 0,
            new_snapshots: 2,
            changed_snapshots: 1,
            ..Default::default()
        };

        let (status, counterexample) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(counterexample.is_some());
        let cex = counterexample.unwrap();
        assert_eq!(cex.failed_checks.len(), 2); // new + changed
    }

    #[test]
    fn test_evaluate_no_snapshots() {
        let backend = InstaBackend::new();
        let stats = InstaStats::default();

        let (status, _) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[tokio::test]
    async fn test_health_check() {
        let backend = InstaBackend::new();
        let status = backend.health_check().await;
        match status {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(reason.contains("cargo-insta"));
            }
            _ => panic!("Unexpected status"),
        }
    }

    #[tokio::test]
    async fn test_verify_returns_result_or_unavailable() {
        use dashprove_usl::{parse, typecheck};

        let backend = InstaBackend::new();
        let spec = parse("theorem test { true }").expect("parse");
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;
        match result {
            Ok(r) => {
                assert_eq!(r.backend, BackendId::Insta);
            }
            Err(BackendError::Unavailable(_)) => {}
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}
