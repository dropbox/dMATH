//! Deny backend for dependency policy enforcement
//!
//! This backend runs cargo-deny to enforce dependency policies including
//! license compliance, duplicate dependencies, and security advisories.
//! It wraps the `dashprove-static` crate's Deny support.

// =============================================
// Kani Proofs for Deny Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- DenyCheck as_str Tests ----

    /// Verify DenyCheck::Licenses as_str returns "licenses"
    #[kani::proof]
    fn proof_deny_check_licenses_as_str() {
        let check = DenyCheck::Licenses;
        kani::assert(
            check.as_str() == "licenses",
            "Licenses should return licenses",
        );
    }

    /// Verify DenyCheck::Advisories as_str returns "advisories"
    #[kani::proof]
    fn proof_deny_check_advisories_as_str() {
        let check = DenyCheck::Advisories;
        kani::assert(
            check.as_str() == "advisories",
            "Advisories should return advisories",
        );
    }

    /// Verify DenyCheck::Bans as_str returns "bans"
    #[kani::proof]
    fn proof_deny_check_bans_as_str() {
        let check = DenyCheck::Bans;
        kani::assert(check.as_str() == "bans", "Bans should return bans");
    }

    /// Verify DenyCheck::Sources as_str returns "sources"
    #[kani::proof]
    fn proof_deny_check_sources_as_str() {
        let check = DenyCheck::Sources;
        kani::assert(check.as_str() == "sources", "Sources should return sources");
    }

    /// Verify DenyCheck::All as_str returns "all"
    #[kani::proof]
    fn proof_deny_check_all_as_str() {
        let check = DenyCheck::All;
        kani::assert(check.as_str() == "all", "All should return all");
    }

    // ---- DenyConfig Default Tests ----

    /// Verify DenyConfig::default crate_path is None
    #[kani::proof]
    fn proof_deny_config_default_crate_path_none() {
        let config = DenyConfig::default();
        kani::assert(
            config.crate_path.is_none(),
            "Default crate_path should be None",
        );
    }

    /// Verify DenyConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_deny_config_default_timeout() {
        let config = DenyConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify DenyConfig::default checks contains All
    #[kani::proof]
    fn proof_deny_config_default_checks() {
        let config = DenyConfig::default();
        kani::assert(
            config.checks.len() == 1,
            "Default checks should have 1 element",
        );
        kani::assert(
            config.checks[0] == DenyCheck::All,
            "Default checks should contain All",
        );
    }

    /// Verify DenyConfig::default config_path is None
    #[kani::proof]
    fn proof_deny_config_default_config_path_none() {
        let config = DenyConfig::default();
        kani::assert(
            config.config_path.is_none(),
            "Default config_path should be None",
        );
    }

    // ---- DenyConfig Builder Tests ----

    /// Verify with_crate_path preserves path
    #[kani::proof]
    fn proof_deny_config_with_crate_path() {
        let config = DenyConfig::default().with_crate_path(PathBuf::from("/test/path"));
        kani::assert(
            config.crate_path == Some(PathBuf::from("/test/path")),
            "with_crate_path should set path",
        );
    }

    /// Verify with_timeout preserves timeout
    #[kani::proof]
    fn proof_deny_config_with_timeout() {
        let config = DenyConfig::default().with_timeout(Duration::from_secs(600));
        kani::assert(
            config.timeout == Duration::from_secs(600),
            "with_timeout should set timeout",
        );
    }

    /// Verify with_checks preserves checks
    #[kani::proof]
    fn proof_deny_config_with_checks() {
        let config =
            DenyConfig::default().with_checks(vec![DenyCheck::Licenses, DenyCheck::Advisories]);
        kani::assert(config.checks.len() == 2, "with_checks should set 2 checks");
        kani::assert(
            config.checks[0] == DenyCheck::Licenses,
            "First check should be Licenses",
        );
    }

    /// Verify with_config_path preserves path
    #[kani::proof]
    fn proof_deny_config_with_config_path() {
        let config = DenyConfig::default().with_config_path(PathBuf::from("/test/deny.toml"));
        kani::assert(
            config.config_path == Some(PathBuf::from("/test/deny.toml")),
            "with_config_path should set path",
        );
    }

    // ---- DenyBackend Construction Tests ----

    /// Verify DenyBackend::new uses default config timeout
    #[kani::proof]
    fn proof_deny_backend_new_default_timeout() {
        let backend = DenyBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify DenyBackend::default equals DenyBackend::new
    #[kani::proof]
    fn proof_deny_backend_default_equals_new() {
        let default_backend = DenyBackend::default();
        let new_backend = DenyBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify DenyBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_deny_backend_with_config_timeout() {
        let config = DenyConfig {
            crate_path: None,
            timeout: Duration::from_secs(600),
            checks: vec![DenyCheck::All],
            config_path: None,
        };
        let backend = DenyBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve timeout",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify DenyBackend::id returns Deny
    #[kani::proof]
    fn proof_deny_backend_id() {
        let backend = DenyBackend::new();
        kani::assert(backend.id() == BackendId::Deny, "Backend id should be Deny");
    }

    /// Verify DenyBackend::supports includes DependencyPolicy
    #[kani::proof]
    fn proof_deny_backend_supports_dependency_policy() {
        let backend = DenyBackend::new();
        let supported = backend.supports();
        let has_policy = supported
            .iter()
            .any(|p| *p == PropertyType::DependencyPolicy);
        kani::assert(has_policy, "Should support DependencyPolicy property");
    }

    /// Verify DenyBackend::supports returns exactly 1 property
    #[kani::proof]
    fn proof_deny_backend_supports_length() {
        let backend = DenyBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 1, "Should support exactly 1 property");
    }
}

use crate::counterexample::StructuredCounterexample;
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_static::{AnalysisConfig, AnalysisTool, Severity, StaticAnalysisBackend};
use dashprove_usl::typecheck::TypedSpec;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Deny check categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenyCheck {
    /// License compliance
    Licenses,
    /// Security advisories
    Advisories,
    /// Banned crates
    Bans,
    /// Duplicate dependencies
    Sources,
    /// All checks
    All,
}

impl DenyCheck {
    /// Get string representation of check type
    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Licenses => "licenses",
            Self::Advisories => "advisories",
            Self::Bans => "bans",
            Self::Sources => "sources",
            Self::All => "all",
        }
    }
}

/// Configuration for Deny backend
#[derive(Debug, Clone)]
pub struct DenyConfig {
    /// Path to the crate to analyze
    pub crate_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Which checks to run
    pub checks: Vec<DenyCheck>,
    /// Path to deny.toml config file
    pub config_path: Option<PathBuf>,
}

impl Default for DenyConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            timeout: Duration::from_secs(300),
            checks: vec![DenyCheck::All],
            config_path: None,
        }
    }
}

impl DenyConfig {
    /// Set the crate path to analyze
    pub fn with_crate_path(mut self, path: PathBuf) -> Self {
        self.crate_path = Some(path);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set specific checks to run
    pub fn with_checks(mut self, checks: Vec<DenyCheck>) -> Self {
        self.checks = checks;
        self
    }

    /// Set path to deny.toml config
    pub fn with_config_path(mut self, path: PathBuf) -> Self {
        self.config_path = Some(path);
        self
    }
}

/// Deny verification backend for dependency policy enforcement
///
/// cargo-deny enforces policies on your dependencies:
/// - **licenses**: Ensure dependencies use allowed licenses
/// - **advisories**: Check for security vulnerabilities
/// - **bans**: Block specific crates or versions
/// - **sources**: Control where dependencies can come from
///
/// # Requirements
///
/// Install cargo-deny:
/// ```bash
/// cargo install cargo-deny
/// ```
///
/// Create a deny.toml configuration file in your crate root.
pub struct DenyBackend {
    config: DenyConfig,
}

impl DenyBackend {
    /// Create a new Deny backend with default configuration
    pub fn new() -> Self {
        Self {
            config: DenyConfig::default(),
        }
    }

    /// Create a new Deny backend with custom configuration
    pub fn with_config(config: DenyConfig) -> Self {
        Self { config }
    }

    /// Run Deny on a crate path
    pub async fn analyze_crate(&self, crate_path: &Path) -> Result<BackendResult, BackendError> {
        // Check for deny.toml
        let config_path = self
            .config
            .config_path
            .clone()
            .unwrap_or_else(|| crate_path.join("deny.toml"));

        if !config_path.exists() {
            return Err(BackendError::CompilationFailed(format!(
                "deny.toml not found at {:?}. Create a deny.toml configuration file.",
                config_path
            )));
        }

        let analysis_config = AnalysisConfig {
            warnings_as_errors: true,
            all_features: false,
            features: Vec::new(),
            targets: Vec::new(),
            timeout: self.config.timeout,
        };

        let backend = StaticAnalysisBackend::new(AnalysisTool::Deny, analysis_config);

        // Check if tool is installed
        match backend.check_installed().await {
            Ok(true) => {}
            Ok(false) => {
                return Err(BackendError::Unavailable(
                    "cargo-deny not installed. Run: cargo install cargo-deny".to_string(),
                ));
            }
            Err(e) => {
                return Err(BackendError::Unavailable(format!(
                    "Failed to check cargo-deny installation: {}",
                    e
                )));
            }
        }

        // Run analysis
        let result = backend
            .run_on_crate(crate_path)
            .await
            .map_err(|e| BackendError::VerificationFailed(e.to_string()))?;

        // Count policy violations by severity
        let error_count = result
            .findings
            .iter()
            .filter(|f| f.severity == Severity::Error)
            .count();
        let warning_count = result
            .findings
            .iter()
            .filter(|f| f.severity == Severity::Warning)
            .count();

        // Determine status
        let status = if error_count > 0 {
            VerificationStatus::Disproven
        } else if warning_count > 0 {
            VerificationStatus::Partial {
                verified_percentage: 100.0 - (warning_count as f64 * 5.0).min(50.0),
            }
        } else {
            VerificationStatus::Proven
        };

        // Build diagnostic messages
        let mut diagnostics = Vec::new();

        // Summary
        let summary = if result.passed && error_count == 0 {
            "Deny: All dependency policies satisfied".to_string()
        } else {
            format!(
                "Deny: {} policy violations, {} warnings found",
                error_count, warning_count
            )
        };
        diagnostics.push(summary);

        for finding in &result.findings {
            let severity_str = match finding.severity {
                Severity::Error => "DENIED",
                Severity::Warning => "WARNING",
                Severity::Info => "INFO",
                Severity::Help => "HELP",
            };

            diagnostics.push(format!(
                "[{}] {}: {}",
                severity_str, finding.code, finding.message
            ));
        }

        // Build counterexample if violations found
        let counterexample = if error_count > 0 || warning_count > 0 {
            Some(StructuredCounterexample::from_raw(result.raw_output))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::Deny,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for DenyBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for DenyBackend {
    fn id(&self) -> BackendId {
        BackendId::Deny
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::DependencyPolicy]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Deny backend requires crate_path pointing to a Rust crate with deny.toml"
                    .to_string(),
            )
        })?;

        self.analyze_crate(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        let backend = StaticAnalysisBackend::new(AnalysisTool::Deny, AnalysisConfig::default());

        match backend.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "cargo-deny not installed".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Check failed: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deny_config_default() {
        let config = DenyConfig::default();
        assert!(config.crate_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.checks, vec![DenyCheck::All]);
        assert!(config.config_path.is_none());
    }

    #[test]
    fn test_deny_config_builder() {
        let config = DenyConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_timeout(Duration::from_secs(60))
            .with_checks(vec![DenyCheck::Licenses, DenyCheck::Advisories])
            .with_config_path(PathBuf::from("/test/deny.toml"));

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.checks.len(), 2);
        assert_eq!(config.config_path, Some(PathBuf::from("/test/deny.toml")));
    }

    #[test]
    fn test_deny_backend_id() {
        let backend = DenyBackend::new();
        assert_eq!(backend.id(), BackendId::Deny);
    }

    #[test]
    fn test_deny_supports_dependency_policy() {
        let backend = DenyBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::DependencyPolicy));
    }

    #[test]
    fn test_deny_check_as_str() {
        assert_eq!(DenyCheck::Licenses.as_str(), "licenses");
        assert_eq!(DenyCheck::Advisories.as_str(), "advisories");
        assert_eq!(DenyCheck::Bans.as_str(), "bans");
        assert_eq!(DenyCheck::Sources.as_str(), "sources");
        assert_eq!(DenyCheck::All.as_str(), "all");
    }

    #[tokio::test]
    async fn test_deny_health_check() {
        let backend = DenyBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(
                    reason.contains("deny") || reason.contains("Deny") || reason.contains("Check")
                );
            }
            HealthStatus::Degraded { .. } => {}
        }
    }

    #[tokio::test]
    async fn test_deny_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = DenyBackend::new();
        let spec = parse("theorem test { true }").expect("parse");
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;
        assert!(result.is_err());
        if let Err(BackendError::Unavailable(msg)) = result {
            assert!(msg.contains("crate_path"));
        } else {
            panic!("Expected Unavailable error");
        }
    }
}
