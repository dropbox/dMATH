//! Vet backend for supply chain auditing
//!
//! This backend runs cargo-vet to audit the supply chain of dependencies,
//! ensuring third-party code has been reviewed. It wraps the `dashprove-static`
//! crate's Vet support.

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

/// Vet audit criteria
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditCriteria {
    /// Code has been reviewed for correctness
    Safe,
    /// Code has been reviewed and is safe to run
    SafeToRun,
    /// Code has been reviewed and safe to deploy
    SafeToDeploy,
}

impl AuditCriteria {
    /// Get string representation of audit criteria
    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::SafeToRun => "safe-to-run",
            Self::SafeToDeploy => "safe-to-deploy",
        }
    }
}

/// Configuration for Vet backend
#[derive(Debug, Clone)]
pub struct VetConfig {
    /// Path to the crate to analyze
    pub crate_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Required audit criteria
    pub criteria: AuditCriteria,
    /// Trust audits from specific organizations
    pub trusted_orgs: Vec<String>,
}

impl Default for VetConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            timeout: Duration::from_secs(300),
            criteria: AuditCriteria::SafeToRun,
            trusted_orgs: Vec::new(),
        }
    }
}

impl VetConfig {
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

    /// Set required audit criteria
    pub fn with_criteria(mut self, criteria: AuditCriteria) -> Self {
        self.criteria = criteria;
        self
    }

    /// Add trusted organizations
    pub fn with_trusted_orgs(mut self, orgs: Vec<String>) -> Self {
        self.trusted_orgs = orgs;
        self
    }
}

/// Vet verification backend for supply chain auditing
///
/// cargo-vet helps ensure third-party Rust code has been audited:
/// - Track which dependencies have been reviewed
/// - Import audit results from trusted organizations
/// - Suggest which dependencies need review
///
/// # Requirements
///
/// Install cargo-vet:
/// ```bash
/// cargo install cargo-vet
/// ```
///
/// Initialize vet in your crate:
/// ```bash
/// cargo vet init
/// ```
pub struct VetBackend {
    config: VetConfig,
}

impl VetBackend {
    /// Create a new Vet backend with default configuration
    pub fn new() -> Self {
        Self {
            config: VetConfig::default(),
        }
    }

    /// Create a new Vet backend with custom configuration
    pub fn with_config(config: VetConfig) -> Self {
        Self { config }
    }

    /// Run Vet on a crate path
    pub async fn analyze_crate(&self, crate_path: &Path) -> Result<BackendResult, BackendError> {
        let analysis_config = AnalysisConfig {
            warnings_as_errors: false,
            all_features: false,
            features: Vec::new(),
            targets: Vec::new(),
            timeout: self.config.timeout,
        };

        let backend = StaticAnalysisBackend::new(AnalysisTool::Vet, analysis_config);

        // Check if tool is installed
        match backend.check_installed().await {
            Ok(true) => {}
            Ok(false) => {
                return Err(BackendError::Unavailable(
                    "cargo-vet not installed. Run: cargo install cargo-vet".to_string(),
                ));
            }
            Err(e) => {
                return Err(BackendError::Unavailable(format!(
                    "Failed to check cargo-vet installation: {}",
                    e
                )));
            }
        }

        // Run analysis
        let result = backend
            .run_on_crate(crate_path)
            .await
            .map_err(|e| BackendError::VerificationFailed(e.to_string()))?;

        // Count unvetted dependencies
        let unvetted_count = result
            .findings
            .iter()
            .filter(|f| {
                f.severity == Severity::Warning
                    && (f.message.contains("unvetted") || f.message.contains("not audited"))
            })
            .count();

        let error_count = result
            .findings
            .iter()
            .filter(|f| f.severity == Severity::Error)
            .count();

        // Determine status
        let total_deps = result.findings.len().max(1);
        let vetted_deps = total_deps.saturating_sub(unvetted_count);

        let status = if error_count > 0 {
            VerificationStatus::Disproven
        } else if unvetted_count == 0 {
            VerificationStatus::Proven
        } else {
            VerificationStatus::Partial {
                verified_percentage: (vetted_deps as f64 / total_deps as f64) * 100.0,
            }
        };

        // Build diagnostic messages
        let mut diagnostics = Vec::new();

        // Summary
        let summary = if unvetted_count == 0 && error_count == 0 {
            "Vet: All dependencies have been audited".to_string()
        } else {
            format!(
                "Vet: {} unvetted dependencies ({:.1}% audited)",
                unvetted_count,
                (vetted_deps as f64 / total_deps as f64) * 100.0
            )
        };
        diagnostics.push(summary);

        for finding in &result.findings {
            let severity_str = match finding.severity {
                Severity::Error => "ERROR",
                Severity::Warning => "UNVETTED",
                Severity::Info => "AUDITED",
                Severity::Help => "HELP",
            };

            diagnostics.push(format!(
                "[{}] {}: {}",
                severity_str, finding.code, finding.message
            ));
        }

        // Build counterexample if unvetted deps found
        let counterexample = if unvetted_count > 0 {
            Some(StructuredCounterexample::from_raw(result.raw_output))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::Vet,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for VetBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for VetBackend {
    fn id(&self) -> BackendId {
        BackendId::Vet
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::SupplyChain]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Vet backend requires crate_path pointing to a Rust crate".to_string(),
            )
        })?;

        self.analyze_crate(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        let backend = StaticAnalysisBackend::new(AnalysisTool::Vet, AnalysisConfig::default());

        match backend.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "cargo-vet not installed".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Check failed: {}", e),
            },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== AuditCriteria =====

    #[kani::proof]
    fn verify_audit_criteria_safe_as_str() {
        let criteria = AuditCriteria::Safe;
        assert!(criteria.as_str() == "safe");
    }

    #[kani::proof]
    fn verify_audit_criteria_safe_to_run_as_str() {
        let criteria = AuditCriteria::SafeToRun;
        assert!(criteria.as_str() == "safe-to-run");
    }

    #[kani::proof]
    fn verify_audit_criteria_safe_to_deploy_as_str() {
        let criteria = AuditCriteria::SafeToDeploy;
        assert!(criteria.as_str() == "safe-to-deploy");
    }

    // ===== VetConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults_timeout() {
        let config = VetConfig::default();
        assert!(config.timeout == Duration::from_secs(300));
    }

    #[kani::proof]
    fn verify_config_defaults_crate_path() {
        let config = VetConfig::default();
        assert!(config.crate_path.is_none());
    }

    #[kani::proof]
    fn verify_config_defaults_criteria() {
        let config = VetConfig::default();
        assert!(config.criteria == AuditCriteria::SafeToRun);
    }

    #[kani::proof]
    fn verify_config_defaults_trusted_orgs() {
        let config = VetConfig::default();
        assert!(config.trusted_orgs.is_empty());
    }

    // ===== VetConfig builders =====

    #[kani::proof]
    fn verify_config_with_crate_path() {
        let config = VetConfig::default().with_crate_path(PathBuf::from("/test/crate"));
        assert!(config.crate_path.is_some());
    }

    #[kani::proof]
    fn verify_config_with_timeout() {
        let config = VetConfig::default().with_timeout(Duration::from_secs(60));
        assert!(config.timeout == Duration::from_secs(60));
    }

    #[kani::proof]
    fn verify_config_with_criteria() {
        let config = VetConfig::default().with_criteria(AuditCriteria::SafeToDeploy);
        assert!(config.criteria == AuditCriteria::SafeToDeploy);
    }

    #[kani::proof]
    fn verify_config_with_trusted_orgs() {
        let config = VetConfig::default().with_trusted_orgs(vec!["mozilla".to_string()]);
        assert!(!config.trusted_orgs.is_empty());
    }

    // ===== VetBackend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = VetBackend::new();
        assert!(backend.config.timeout == Duration::from_secs(300));
        assert!(backend.config.criteria == AuditCriteria::SafeToRun);
    }

    #[kani::proof]
    fn verify_backend_default_matches_new() {
        let b1 = VetBackend::new();
        let b2 = VetBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.criteria == b2.config.criteria);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_values() {
        let config = VetConfig {
            crate_path: Some(PathBuf::from("/test")),
            timeout: Duration::from_secs(60),
            criteria: AuditCriteria::Safe,
            trusted_orgs: vec![],
        };
        let backend = VetBackend::with_config(config);
        assert!(backend.config.timeout == Duration::from_secs(60));
        assert!(backend.config.criteria == AuditCriteria::Safe);
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = VetBackend::new();
        assert!(matches!(backend.id(), BackendId::Vet));
    }

    #[kani::proof]
    fn verify_supports_supply_chain() {
        let backend = VetBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::SupplyChain));
        assert!(supported.len() == 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vet_config_default() {
        let config = VetConfig::default();
        assert!(config.crate_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.criteria, AuditCriteria::SafeToRun);
        assert!(config.trusted_orgs.is_empty());
    }

    #[test]
    fn test_vet_config_builder() {
        let config = VetConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_timeout(Duration::from_secs(60))
            .with_criteria(AuditCriteria::SafeToDeploy)
            .with_trusted_orgs(vec!["mozilla".to_string(), "rust-lang".to_string()]);

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.criteria, AuditCriteria::SafeToDeploy);
        assert_eq!(config.trusted_orgs.len(), 2);
    }

    #[test]
    fn test_vet_backend_id() {
        let backend = VetBackend::new();
        assert_eq!(backend.id(), BackendId::Vet);
    }

    #[test]
    fn test_vet_supports_supply_chain() {
        let backend = VetBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::SupplyChain));
    }

    #[test]
    fn test_audit_criteria_as_str() {
        assert_eq!(AuditCriteria::Safe.as_str(), "safe");
        assert_eq!(AuditCriteria::SafeToRun.as_str(), "safe-to-run");
        assert_eq!(AuditCriteria::SafeToDeploy.as_str(), "safe-to-deploy");
    }

    #[tokio::test]
    async fn test_vet_health_check() {
        let backend = VetBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(
                    reason.contains("vet") || reason.contains("Vet") || reason.contains("Check")
                );
            }
            HealthStatus::Degraded { .. } => {}
        }
    }

    #[tokio::test]
    async fn test_vet_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = VetBackend::new();
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
