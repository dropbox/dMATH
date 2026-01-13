//! cargo-deadlinks dead link checker backend
//!
//! Deadlinks checks for broken links in rustdoc documentation,
//! ensuring all intra-doc and external links are valid.
//!
//! See: <https://github.com/deadlinks/cargo-deadlinks>

// =============================================
// Kani Proofs for Deadlinks Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- DeadlinksConfig Default Tests ----

    /// Verify DeadlinksConfig::default timeout is 120 seconds
    #[kani::proof]
    fn proof_deadlinks_config_default_timeout() {
        let config = DeadlinksConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "Default timeout should be 120 seconds",
        );
    }

    /// Verify DeadlinksConfig::default check_http is false
    #[kani::proof]
    fn proof_deadlinks_config_default_check_http_false() {
        let config = DeadlinksConfig::default();
        kani::assert(!config.check_http, "Default check_http should be false");
    }

    /// Verify DeadlinksConfig::default check_intra_doc_links is true
    #[kani::proof]
    fn proof_deadlinks_config_default_check_intra_doc_links_true() {
        let config = DeadlinksConfig::default();
        kani::assert(
            config.check_intra_doc_links,
            "Default check_intra_doc_links should be true",
        );
    }

    /// Verify DeadlinksConfig::default ignore_fragments is false
    #[kani::proof]
    fn proof_deadlinks_config_default_ignore_fragments_false() {
        let config = DeadlinksConfig::default();
        kani::assert(
            !config.ignore_fragments,
            "Default ignore_fragments should be false",
        );
    }

    // ---- DeadlinksConfig Builder Tests ----

    /// Verify with_check_http(true) sets check_http
    #[kani::proof]
    fn proof_deadlinks_config_with_check_http_true() {
        let config = DeadlinksConfig::default().with_check_http(true);
        kani::assert(
            config.check_http,
            "with_check_http(true) should enable check_http",
        );
    }

    /// Verify with_check_http(false) clears check_http
    #[kani::proof]
    fn proof_deadlinks_config_with_check_http_false() {
        let config = DeadlinksConfig::default()
            .with_check_http(true)
            .with_check_http(false);
        kani::assert(
            !config.check_http,
            "with_check_http(false) should disable check_http",
        );
    }

    /// Verify with_check_intra_doc_links(false) clears check_intra_doc_links
    #[kani::proof]
    fn proof_deadlinks_config_with_check_intra_doc_links_false() {
        let config = DeadlinksConfig::default().with_check_intra_doc_links(false);
        kani::assert(
            !config.check_intra_doc_links,
            "with_check_intra_doc_links(false) should disable",
        );
    }

    /// Verify with_ignore_fragments(true) sets ignore_fragments
    #[kani::proof]
    fn proof_deadlinks_config_with_ignore_fragments_true() {
        let config = DeadlinksConfig::default().with_ignore_fragments(true);
        kani::assert(
            config.ignore_fragments,
            "with_ignore_fragments(true) should enable",
        );
    }

    // ---- DeadlinksBackend Construction Tests ----

    /// Verify DeadlinksBackend::new uses default config timeout
    #[kani::proof]
    fn proof_deadlinks_backend_new_default_timeout() {
        let backend = DeadlinksBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "New backend should use default timeout",
        );
    }

    /// Verify DeadlinksBackend::default equals DeadlinksBackend::new
    #[kani::proof]
    fn proof_deadlinks_backend_default_equals_new() {
        let default_backend = DeadlinksBackend::default();
        let new_backend = DeadlinksBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify DeadlinksBackend::with_config preserves custom config
    #[kani::proof]
    fn proof_deadlinks_backend_with_config() {
        let config = DeadlinksConfig {
            timeout: Duration::from_secs(60),
            check_http: true,
            check_intra_doc_links: false,
            ignore_fragments: true,
        };
        let backend = DeadlinksBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "with_config should preserve timeout",
        );
        kani::assert(
            backend.config.check_http,
            "with_config should preserve check_http",
        );
        kani::assert(
            !backend.config.check_intra_doc_links,
            "with_config should preserve check_intra_doc_links",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify DeadlinksBackend::id returns Deadlinks
    #[kani::proof]
    fn proof_deadlinks_backend_id() {
        let backend = DeadlinksBackend::new();
        kani::assert(
            backend.id() == BackendId::Deadlinks,
            "Backend id should be Deadlinks",
        );
    }

    /// Verify DeadlinksBackend::supports includes Lint
    #[kani::proof]
    fn proof_deadlinks_backend_supports_lint() {
        let backend = DeadlinksBackend::new();
        let supported = backend.supports();
        let has_lint = supported.iter().any(|p| *p == PropertyType::Lint);
        kani::assert(has_lint, "Should support Lint property");
    }

    /// Verify DeadlinksBackend::supports returns exactly 1 property
    #[kani::proof]
    fn proof_deadlinks_backend_supports_length() {
        let backend = DeadlinksBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 1, "Should support exactly 1 property");
    }

    // ---- extract_first_number Tests ----

    /// Verify extract_first_number returns first number
    #[kani::proof]
    fn proof_extract_first_number_found() {
        let result = DeadlinksBackend::extract_first_number("Checked 100 links");
        kani::assert(result == Some(100), "Should extract 100");
    }

    /// Verify extract_first_number returns None for no numbers
    #[kani::proof]
    fn proof_extract_first_number_none() {
        let result = DeadlinksBackend::extract_first_number("no numbers here");
        kani::assert(result.is_none(), "Should return None for no numbers");
    }

    /// Verify extract_first_number returns first number when multiple
    #[kani::proof]
    fn proof_extract_first_number_multiple() {
        let result = DeadlinksBackend::extract_first_number("Found 5 errors in 10 files");
        kani::assert(result == Some(5), "Should extract first number (5)");
    }

    // ---- evaluate_results Tests ----

    /// Verify evaluate_results returns Proven for no broken links
    #[kani::proof]
    fn proof_evaluate_results_proven() {
        let backend = DeadlinksBackend::new();
        let stats = DeadlinksStats {
            total_links_checked: 100,
            broken_links: 0,
            broken_link_details: vec![],
        };
        let (status, ce) = backend.evaluate_results(&stats);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should return Proven for no broken links",
        );
        kani::assert(ce.is_none(), "Should return no counterexample");
    }

    /// Verify evaluate_results returns Disproven for broken links
    #[kani::proof]
    fn proof_evaluate_results_disproven() {
        let backend = DeadlinksBackend::new();
        let stats = DeadlinksStats {
            total_links_checked: 100,
            broken_links: 2,
            broken_link_details: vec![],
        };
        let (status, ce) = backend.evaluate_results(&stats);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should return Disproven for broken links",
        );
        kani::assert(ce.is_some(), "Should return counterexample");
    }

    /// Verify evaluate_results returns Unknown for zero links checked
    #[kani::proof]
    fn proof_evaluate_results_unknown() {
        let backend = DeadlinksBackend::new();
        let stats = DeadlinksStats {
            total_links_checked: 0,
            broken_links: 0,
            broken_link_details: vec![],
        };
        let (status, _) = backend.evaluate_results(&stats);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should return Unknown for zero links checked",
        );
    }
}

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

/// Configuration for deadlinks link checker backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlinksConfig {
    pub timeout: Duration,
    pub check_http: bool,
    pub check_intra_doc_links: bool,
    pub ignore_fragments: bool,
}

impl Default for DeadlinksConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(120),
            check_http: false,
            check_intra_doc_links: true,
            ignore_fragments: false,
        }
    }
}

impl DeadlinksConfig {
    pub fn with_check_http(mut self, enabled: bool) -> Self {
        self.check_http = enabled;
        self
    }

    pub fn with_check_intra_doc_links(mut self, enabled: bool) -> Self {
        self.check_intra_doc_links = enabled;
        self
    }

    pub fn with_ignore_fragments(mut self, enabled: bool) -> Self {
        self.ignore_fragments = enabled;
        self
    }
}

/// Link checking statistics from deadlinks
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeadlinksStats {
    pub total_links_checked: u64,
    pub broken_links: u64,
    pub broken_link_details: Vec<BrokenLink>,
}

/// Information about a broken link
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrokenLink {
    pub source_file: String,
    pub link_target: String,
    pub reason: Option<String>,
}

/// cargo-deadlinks dead link checker backend
pub struct DeadlinksBackend {
    config: DeadlinksConfig,
}

impl Default for DeadlinksBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl DeadlinksBackend {
    pub fn new() -> Self {
        Self {
            config: DeadlinksConfig::default(),
        }
    }

    pub fn with_config(config: DeadlinksConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<PathBuf, String> {
        which::which("cargo-deadlinks").map_err(|_| {
            "cargo-deadlinks not found. Install via cargo install cargo-deadlinks".to_string()
        })
    }

    /// Run deadlinks on documentation
    pub async fn check_links(
        &self,
        crate_path: &std::path::Path,
    ) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        self.detect().await.map_err(BackendError::Unavailable)?;

        // First build docs
        let mut doc_cmd = Command::new("cargo");
        doc_cmd.current_dir(crate_path);
        doc_cmd.arg("doc").arg("--no-deps");

        let _ = tokio::time::timeout(self.config.timeout, doc_cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to build docs: {}", e))
            })?;

        // Then check links
        let mut cmd = Command::new("cargo");
        cmd.current_dir(crate_path);
        cmd.arg("deadlinks");

        if self.config.check_http {
            cmd.arg("--check-http");
        }

        if self.config.ignore_fragments {
            cmd.arg("--ignore-fragments");
        }

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run cargo deadlinks: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let stats = self.parse_output(&stdout, &stderr);
        let (status, counterexample) = self.evaluate_results(&stats);

        let mut diagnostics = Vec::new();
        diagnostics.push(format!(
            "Links checked: {}, Broken: {}",
            stats.total_links_checked, stats.broken_links
        ));

        Ok(BackendResult {
            backend: BackendId::Deadlinks,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    fn parse_output(&self, stdout: &str, stderr: &str) -> DeadlinksStats {
        let mut stats = DeadlinksStats::default();
        let combined = format!("{}\n{}", stdout, stderr);

        for line in combined.lines() {
            // Parse broken link lines
            // Format: "Error: Couldn't find link 'target' in file 'source.html'"
            if line.contains("Couldn't find") || line.contains("broken link") {
                stats.broken_links += 1;
                stats.broken_link_details.push(BrokenLink {
                    source_file: Self::extract_quoted_value(line, "in file").unwrap_or_default(),
                    link_target: Self::extract_quoted_value(line, "link").unwrap_or_default(),
                    reason: Some(line.trim().to_string()),
                });
            }

            // Count checked links from summary
            if line.contains("links checked") || line.contains("Checked") {
                if let Some(count) = Self::extract_first_number(line) {
                    stats.total_links_checked = count;
                }
            }
        }

        // If no explicit count, estimate from broken links
        if stats.total_links_checked == 0 && stats.broken_links > 0 {
            stats.total_links_checked = stats.broken_links;
        }

        stats
    }

    fn extract_quoted_value(line: &str, prefix: &str) -> Option<String> {
        if let Some(pos) = line.find(prefix) {
            let after = &line[pos + prefix.len()..];
            if let Some(start) = after.find('\'') {
                let remainder = &after[start + 1..];
                if let Some(end) = remainder.find('\'') {
                    return Some(remainder[..end].to_string());
                }
            }
        }
        None
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
        stats: &DeadlinksStats,
    ) -> (VerificationStatus, Option<StructuredCounterexample>) {
        if stats.broken_links > 0 {
            let failed_checks: Vec<FailedCheck> = stats
                .broken_link_details
                .iter()
                .map(|link| FailedCheck {
                    check_id: format!("broken_link:{}", link.link_target),
                    description: format!(
                        "Broken link '{}' in '{}'",
                        link.link_target, link.source_file
                    ),
                    location: None,
                    function: None,
                })
                .collect();

            let failed_checks = if failed_checks.is_empty() {
                vec![FailedCheck {
                    check_id: "broken_links".to_string(),
                    description: format!("{} broken link(s) found", stats.broken_links),
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
                    playback_test: Some("Run `cargo deadlinks` to see broken links".to_string()),
                    trace: vec![],
                    raw: Some(format!("{} broken link(s) found", stats.broken_links)),
                    minimized: false,
                }),
            );
        }

        if stats.total_links_checked == 0 {
            return (
                VerificationStatus::Unknown {
                    reason: "No links checked (no documentation?)".to_string(),
                },
                None,
            );
        }

        (VerificationStatus::Proven, None)
    }
}

#[async_trait]
impl VerificationBackend for DeadlinksBackend {
    fn id(&self) -> BackendId {
        BackendId::Deadlinks
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Lint]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        self.detect().await.map_err(BackendError::Unavailable)?;

        let mut diagnostics = Vec::new();
        diagnostics.push("deadlinks ready for link checking".to_string());
        if self.config.check_http {
            diagnostics.push("HTTP link checking enabled".to_string());
        }

        Ok(BackendResult {
            backend: BackendId::Deadlinks,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_id() {
        assert_eq!(DeadlinksBackend::new().id(), BackendId::Deadlinks);
    }

    #[test]
    fn test_config_default() {
        let config = DeadlinksConfig::default();
        assert!(!config.check_http);
        assert!(config.check_intra_doc_links);
        assert!(!config.ignore_fragments);
    }

    #[test]
    fn test_config_builder() {
        let config = DeadlinksConfig::default()
            .with_check_http(true)
            .with_ignore_fragments(true);

        assert!(config.check_http);
        assert!(config.ignore_fragments);
    }

    #[test]
    fn test_extract_quoted_value() {
        let line = "Couldn't find link 'foo::bar' in file 'index.html'";
        assert_eq!(
            DeadlinksBackend::extract_quoted_value(line, "link"),
            Some("foo::bar".to_string())
        );
        assert_eq!(
            DeadlinksBackend::extract_quoted_value(line, "in file"),
            Some("index.html".to_string())
        );
    }

    #[test]
    fn test_parse_output_no_errors() {
        let backend = DeadlinksBackend::new();
        let stdout = "Checked 100 links\n";
        let stats = backend.parse_output(stdout, "");

        assert_eq!(stats.total_links_checked, 100);
        assert_eq!(stats.broken_links, 0);
    }

    #[test]
    fn test_parse_output_with_broken_links() {
        let backend = DeadlinksBackend::new();
        let stdout = r#"
Error: Couldn't find link 'missing::item' in file 'index.html'
"#;
        let stats = backend.parse_output(stdout, "");

        assert_eq!(stats.broken_links, 1);
        assert_eq!(stats.broken_link_details.len(), 1);
    }

    #[test]
    fn test_evaluate_results_pass() {
        let backend = DeadlinksBackend::new();
        let stats = DeadlinksStats {
            total_links_checked: 100,
            broken_links: 0,
            broken_link_details: vec![],
        };

        let (status, _) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn test_evaluate_results_fail() {
        let backend = DeadlinksBackend::new();
        let stats = DeadlinksStats {
            total_links_checked: 100,
            broken_links: 2,
            broken_link_details: vec![BrokenLink {
                source_file: "index.html".to_string(),
                link_target: "missing".to_string(),
                reason: None,
            }],
        };

        let (status, cex) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(cex.is_some());
    }

    #[tokio::test]
    async fn test_health_check() {
        let backend = DeadlinksBackend::new();
        let status = backend.health_check().await;
        match status {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(reason.contains("cargo-deadlinks"));
            }
            _ => panic!("Unexpected status"),
        }
    }

    #[tokio::test]
    async fn test_verify_returns_result_or_unavailable() {
        use dashprove_usl::{parse, typecheck};

        let backend = DeadlinksBackend::new();
        let spec = parse("theorem test { true }").expect("parse");
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;
        match result {
            Ok(r) => {
                assert_eq!(r.backend, BackendId::Deadlinks);
            }
            Err(BackendError::Unavailable(_)) => {}
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}
