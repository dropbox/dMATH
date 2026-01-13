//! Audit backend for security vulnerability scanning
//!
//! This backend runs cargo-audit to scan for security vulnerabilities in
//! dependencies. It wraps the `dashprove-static` crate's Audit support.

// =============================================
// Kani Proofs for Audit Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- AuditConfig Default Tests ----

    /// Verify AuditConfig::default sets correct baseline values
    #[kani::proof]
    fn proof_audit_config_defaults() {
        let config = AuditConfig::default();
        kani::assert(
            config.crate_path.is_none(),
            "crate_path should default to None",
        );
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "timeout should default to 300 seconds",
        );
        kani::assert(
            config.ignore_advisories.is_empty(),
            "ignore_advisories should default to empty",
        );
        kani::assert(
            !config.warnings_as_errors,
            "warnings_as_errors should default to false",
        );
        kani::assert(
            config.update_database,
            "update_database should default to true",
        );
    }

    // ---- AuditConfig Builder Tests ----

    /// Verify with_crate_path updates crate_path
    #[kani::proof]
    fn proof_audit_config_with_crate_path() {
        let config = AuditConfig::default().with_crate_path(PathBuf::from("/project"));
        kani::assert(
            config.crate_path == Some(PathBuf::from("/project")),
            "with_crate_path should set crate_path",
        );
    }

    /// Verify with_timeout updates timeout
    #[kani::proof]
    fn proof_audit_config_with_timeout() {
        let config = AuditConfig::default().with_timeout(Duration::from_secs(60));
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "with_timeout should set timeout",
        );
    }

    /// Verify with_ignored_advisories updates ignore_advisories
    #[kani::proof]
    fn proof_audit_config_with_ignored_advisories() {
        let config =
            AuditConfig::default().with_ignored_advisories(vec!["RUSTSEC-2020-0001".to_string()]);
        kani::assert(
            config.ignore_advisories.len() == 1,
            "with_ignored_advisories should set advisories",
        );
        kani::assert(
            config.ignore_advisories[0] == "RUSTSEC-2020-0001",
            "advisory should match",
        );
    }

    /// Verify with_warnings_as_errors updates warnings_as_errors
    #[kani::proof]
    fn proof_audit_config_with_warnings_as_errors() {
        let config = AuditConfig::default().with_warnings_as_errors();
        kani::assert(
            config.warnings_as_errors,
            "with_warnings_as_errors should set to true",
        );
    }

    /// Verify without_database_update updates update_database
    #[kani::proof]
    fn proof_audit_config_without_database_update() {
        let config = AuditConfig::default().without_database_update();
        kani::assert(
            !config.update_database,
            "without_database_update should set to false",
        );
    }

    /// Verify builder chaining works correctly
    #[kani::proof]
    fn proof_audit_config_builder_chain() {
        let config = AuditConfig::default()
            .with_crate_path(PathBuf::from("/crate"))
            .with_timeout(Duration::from_secs(120))
            .with_warnings_as_errors()
            .without_database_update();

        kani::assert(
            config.crate_path == Some(PathBuf::from("/crate")),
            "crate_path should be set after chain",
        );
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "timeout should be set after chain",
        );
        kani::assert(
            config.warnings_as_errors,
            "warnings_as_errors should be true after chain",
        );
        kani::assert(
            !config.update_database,
            "update_database should be false after chain",
        );
    }

    // ---- AuditBackend Construction Tests ----

    /// Verify AuditBackend::new uses default configuration
    #[kani::proof]
    fn proof_audit_backend_new_defaults() {
        let backend = AuditBackend::new();
        kani::assert(
            backend.config.crate_path.is_none(),
            "new backend should have no crate_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "new backend should default timeout to 300 seconds",
        );
        kani::assert(
            !backend.config.warnings_as_errors,
            "new backend should default warnings_as_errors to false",
        );
    }

    /// Verify AuditBackend::default equals AuditBackend::new
    #[kani::proof]
    fn proof_audit_backend_default_equals_new() {
        let default_backend = AuditBackend::default();
        let new_backend = AuditBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.warnings_as_errors == new_backend.config.warnings_as_errors,
            "default and new should share warnings_as_errors",
        );
    }

    /// Verify AuditBackend::with_config preserves custom configuration
    #[kani::proof]
    fn proof_audit_backend_with_config() {
        let config = AuditConfig {
            crate_path: Some(PathBuf::from("/work/project")),
            timeout: Duration::from_secs(60),
            ignore_advisories: vec!["RUSTSEC-2021-0001".to_string()],
            warnings_as_errors: true,
            update_database: false,
        };
        let backend = AuditBackend::with_config(config);
        kani::assert(
            backend.config.crate_path == Some(PathBuf::from("/work/project")),
            "with_config should preserve crate_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "with_config should preserve timeout",
        );
        kani::assert(
            backend.config.ignore_advisories.len() == 1,
            "with_config should preserve ignore_advisories",
        );
        kani::assert(
            backend.config.warnings_as_errors,
            "with_config should preserve warnings_as_errors",
        );
        kani::assert(
            !backend.config.update_database,
            "with_config should preserve update_database",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::Audit
    #[kani::proof]
    fn proof_audit_backend_id() {
        let backend = AuditBackend::new();
        kani::assert(
            backend.id() == BackendId::Audit,
            "AuditBackend id should be BackendId::Audit",
        );
    }

    /// Verify supports() includes SecurityVulnerability
    #[kani::proof]
    fn proof_audit_backend_supports() {
        let backend = AuditBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::SecurityVulnerability),
            "supports should include SecurityVulnerability",
        );
    }

    /// Verify supports() returns exactly one property type
    #[kani::proof]
    fn proof_audit_backend_supports_length() {
        let backend = AuditBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 1,
            "Audit should support exactly one property type",
        );
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

/// Configuration for Audit backend
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Path to the crate to analyze
    pub crate_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Ignore specific advisories
    pub ignore_advisories: Vec<String>,
    /// Treat warnings as errors
    pub warnings_as_errors: bool,
    /// Database update before scan
    pub update_database: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            timeout: Duration::from_secs(300),
            ignore_advisories: Vec::new(),
            warnings_as_errors: false,
            update_database: true,
        }
    }
}

impl AuditConfig {
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

    /// Ignore specific advisories
    pub fn with_ignored_advisories(mut self, advisories: Vec<String>) -> Self {
        self.ignore_advisories = advisories;
        self
    }

    /// Treat warnings as errors
    pub fn with_warnings_as_errors(mut self) -> Self {
        self.warnings_as_errors = true;
        self
    }

    /// Disable database update
    pub fn without_database_update(mut self) -> Self {
        self.update_database = false;
        self
    }
}

/// Audit verification backend for security vulnerability scanning
///
/// cargo-audit scans Cargo.lock for dependencies with known security vulnerabilities
/// from the RustSec Advisory Database:
/// - CVE vulnerabilities
/// - RUSTSEC advisories
/// - Yanked crate warnings
///
/// # Requirements
///
/// Install cargo-audit:
/// ```bash
/// cargo install cargo-audit
/// ```
pub struct AuditBackend {
    config: AuditConfig,
}

impl AuditBackend {
    /// Create a new Audit backend with default configuration
    pub fn new() -> Self {
        Self {
            config: AuditConfig::default(),
        }
    }

    /// Create a new Audit backend with custom configuration
    pub fn with_config(config: AuditConfig) -> Self {
        Self { config }
    }

    /// Run Audit on a crate path
    pub async fn analyze_crate(&self, crate_path: &Path) -> Result<BackendResult, BackendError> {
        let analysis_config = AnalysisConfig {
            warnings_as_errors: self.config.warnings_as_errors,
            all_features: false,
            features: Vec::new(),
            targets: Vec::new(),
            timeout: self.config.timeout,
        };

        let backend = StaticAnalysisBackend::new(AnalysisTool::Audit, analysis_config);

        // Check if tool is installed
        match backend.check_installed().await {
            Ok(true) => {}
            Ok(false) => {
                return Err(BackendError::Unavailable(
                    "cargo-audit not installed. Run: cargo install cargo-audit".to_string(),
                ));
            }
            Err(e) => {
                return Err(BackendError::Unavailable(format!(
                    "Failed to check cargo-audit installation: {}",
                    e
                )));
            }
        }

        // Run analysis
        let result = backend
            .run_on_crate(crate_path)
            .await
            .map_err(|e| BackendError::VerificationFailed(e.to_string()))?;

        // Count vulnerabilities by severity
        let critical_count = result
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
        let status = if critical_count > 0 || (warning_count > 0 && self.config.warnings_as_errors)
        {
            VerificationStatus::Disproven
        } else if warning_count > 0 {
            VerificationStatus::Partial {
                verified_percentage: 100.0 - (warning_count as f64 * 10.0).min(100.0),
            }
        } else {
            VerificationStatus::Proven
        };

        // Build diagnostic messages
        let mut diagnostics = Vec::new();

        // Summary
        let summary = if result.passed && critical_count == 0 {
            "Audit: No known vulnerabilities found".to_string()
        } else {
            format!(
                "Audit: {} critical, {} warning vulnerabilities found",
                critical_count, warning_count
            )
        };
        diagnostics.push(summary);

        for finding in &result.findings {
            let severity_str = match finding.severity {
                Severity::Error => "CRITICAL",
                Severity::Warning => "WARNING",
                Severity::Info => "INFO",
                Severity::Help => "HELP",
            };

            diagnostics.push(format!(
                "[{}] {}: {}",
                severity_str, finding.code, finding.message
            ));
        }

        // Build counterexample if vulnerabilities found
        let counterexample = if critical_count > 0 || warning_count > 0 {
            Some(StructuredCounterexample::from_raw(result.raw_output))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::Audit,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for AuditBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for AuditBackend {
    fn id(&self) -> BackendId {
        BackendId::Audit
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::SecurityVulnerability]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Audit backend requires crate_path pointing to a Rust crate".to_string(),
            )
        })?;

        self.analyze_crate(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        let backend = StaticAnalysisBackend::new(AnalysisTool::Audit, AnalysisConfig::default());

        match backend.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "cargo-audit not installed".to_string(),
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
    fn test_audit_config_default() {
        let config = AuditConfig::default();
        assert!(config.crate_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(config.ignore_advisories.is_empty());
        assert!(!config.warnings_as_errors);
        assert!(config.update_database);
    }

    #[test]
    fn test_audit_config_builder() {
        let config = AuditConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_timeout(Duration::from_secs(60))
            .with_ignored_advisories(vec!["RUSTSEC-2020-0001".to_string()])
            .with_warnings_as_errors()
            .without_database_update();

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.ignore_advisories.len(), 1);
        assert!(config.warnings_as_errors);
        assert!(!config.update_database);
    }

    #[test]
    fn test_audit_backend_id() {
        let backend = AuditBackend::new();
        assert_eq!(backend.id(), BackendId::Audit);
    }

    #[test]
    fn test_audit_supports_security_vulnerability() {
        let backend = AuditBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::SecurityVulnerability));
    }

    #[tokio::test]
    async fn test_audit_health_check() {
        let backend = AuditBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(
                    reason.contains("audit")
                        || reason.contains("Audit")
                        || reason.contains("Check")
                );
            }
            HealthStatus::Degraded { .. } => {}
        }
    }

    #[tokio::test]
    async fn test_audit_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = AuditBackend::new();
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
