//! Rudra backend for memory safety bug detection in unsafe Rust
//!
//! This backend runs Rudra to find memory safety bugs in unsafe Rust code.
//! Rudra specializes in detecting issues like:
//! - Higher-order safety invariant violations
//! - Send/Sync trait misuse
//! - Panic safety issues

use crate::counterexample::StructuredCounterexample;
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::process::Command;

/// Configuration for Rudra backend
#[derive(Debug, Clone)]
pub struct RudraConfig {
    /// Path to the crate to analyze
    pub crate_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Enable all Rudra analyses
    pub all_analyses: bool,
    /// Enable Send/Sync analysis
    pub check_send_sync: bool,
    /// Enable panic safety analysis
    pub check_panic_safety: bool,
    /// Enable higher-order invariant analysis
    pub check_higher_order: bool,
}

impl Default for RudraConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            timeout: Duration::from_secs(300),
            all_analyses: true,
            check_send_sync: true,
            check_panic_safety: true,
            check_higher_order: true,
        }
    }
}

impl RudraConfig {
    /// Set the crate path
    pub fn with_crate_path(mut self, path: PathBuf) -> Self {
        self.crate_path = Some(path);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable all analyses
    pub fn with_all_analyses(mut self) -> Self {
        self.all_analyses = true;
        self
    }

    /// Enable/disable Send/Sync analysis
    pub fn with_send_sync_checking(mut self, check: bool) -> Self {
        self.check_send_sync = check;
        self
    }

    /// Enable/disable panic safety analysis
    pub fn with_panic_safety_checking(mut self, check: bool) -> Self {
        self.check_panic_safety = check;
        self
    }

    /// Enable/disable higher-order invariant analysis
    pub fn with_higher_order_checking(mut self, check: bool) -> Self {
        self.check_higher_order = check;
        self
    }
}

/// Rudra verification backend for unsafe Rust analysis
///
/// Rudra is a static analyzer that finds memory safety bugs in unsafe Rust code.
/// It specializes in detecting:
/// - Higher-order invariant violations (e.g., incorrect unsafe block usage)
/// - Send/Sync trait misuse (data race vulnerabilities)
/// - Panic safety issues (double-free, use-after-free on panic)
///
/// # Requirements
///
/// Install Rudra:
/// ```bash
/// cargo install rudra
/// ```
pub struct RudraBackend {
    config: RudraConfig,
}

impl RudraBackend {
    /// Create a new Rudra backend with default configuration
    pub fn new() -> Self {
        Self {
            config: RudraConfig::default(),
        }
    }

    /// Create a new Rudra backend with custom configuration
    pub fn with_config(config: RudraConfig) -> Self {
        Self { config }
    }

    /// Run Rudra on a crate
    pub async fn analyze_crate(&self, crate_path: &Path) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Build Rudra command
        let args = vec!["rudra"];

        // Rudra uses RUDRA_FLAGS environment variable for configuration
        let flags = self.build_rudra_flags();

        // Run Rudra
        let output = Command::new("cargo")
            .args(&args)
            .current_dir(crate_path)
            .env("RUDRA_FLAGS", &flags)
            .output()
            .await
            .map_err(|e| BackendError::VerificationFailed(format!("Failed to run Rudra: {}", e)))?;

        let duration = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}\n{}", stdout, stderr);

        // Parse Rudra output
        let (status, findings) = self.parse_rudra_output(&combined, output.status.success());

        // Build diagnostics
        let mut diagnostics = Vec::new();

        // Summary
        let summary = match &status {
            VerificationStatus::Proven => "Rudra: No unsafe issues found".to_string(),
            VerificationStatus::Disproven => {
                format!("Rudra: {} potential unsafe issues found", findings.len())
            }
            VerificationStatus::Partial {
                verified_percentage,
            } => {
                format!("Rudra: Analysis {:.1}% complete", verified_percentage)
            }
            VerificationStatus::Unknown { reason } => format!("Rudra: {}", reason),
        };
        diagnostics.push(summary);
        diagnostics.extend(findings.clone());

        // Build counterexample if issues found
        let counterexample = if !findings.is_empty() {
            Some(StructuredCounterexample::from_raw(combined.clone()))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::Rudra,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: duration,
        })
    }

    fn build_rudra_flags(&self) -> String {
        let mut flags = Vec::new();

        if self.config.check_send_sync {
            flags.push("-Zrudra-send-sync");
        }
        if self.config.check_panic_safety {
            flags.push("-Zrudra-panic-safety");
        }
        if self.config.check_higher_order {
            flags.push("-Zrudra-higher-order");
        }

        flags.join(" ")
    }

    fn parse_rudra_output(&self, output: &str, success: bool) -> (VerificationStatus, Vec<String>) {
        let mut findings = Vec::new();

        // Look for Rudra-specific warning patterns
        for line in output.lines() {
            // Rudra issues typically include:
            // - [UnsafeDataflow]
            // - [SendSyncVariance]
            // - [PanicSafety]
            // - [HigherOrderInvariant]
            if line.contains("[UnsafeDataflow]")
                || line.contains("[SendSyncVariance]")
                || line.contains("[PanicSafety]")
                || line.contains("[HigherOrderInvariant]")
                || line.contains("unsafe block")
                || line.contains("Send/Sync")
                || line.contains("memory safety")
            {
                findings.push(line.trim().to_string());
            }
            // Also capture warning/error lines
            if (line.contains("warning:") || line.contains("error:"))
                && (line.contains("unsafe")
                    || line.contains("safety")
                    || line.contains("Send")
                    || line.contains("Sync"))
            {
                findings.push(line.trim().to_string());
            }
        }

        let status = if findings.is_empty() && success {
            VerificationStatus::Proven
        } else if !findings.is_empty() {
            VerificationStatus::Disproven
        } else if !success {
            VerificationStatus::Unknown {
                reason: "Rudra analysis failed".to_string(),
            }
        } else {
            VerificationStatus::Proven
        };

        (status, findings)
    }

    /// Check if Rudra is installed
    pub async fn check_installed(&self) -> Result<bool, BackendError> {
        let output = Command::new("cargo")
            .args(["rudra", "--version"])
            .output()
            .await;

        match output {
            Ok(out) => Ok(out.status.success()),
            Err(_) => Ok(false),
        }
    }
}

impl Default for RudraBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for RudraBackend {
    fn id(&self) -> BackendId {
        BackendId::Rudra
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::MemorySafety, PropertyType::DataRace]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Rudra backend requires crate_path pointing to a Rust crate".to_string(),
            )
        })?;

        self.analyze_crate(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        match self.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "Rudra not installed. Install with: cargo install rudra".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Failed to check Rudra: {}", e),
            },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== RudraConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults_timeout() {
        let config = RudraConfig::default();
        assert!(config.timeout == Duration::from_secs(300));
    }

    #[kani::proof]
    fn verify_config_defaults_options() {
        let config = RudraConfig::default();
        assert!(config.crate_path.is_none());
        assert!(config.all_analyses);
        assert!(config.check_send_sync);
        assert!(config.check_panic_safety);
        assert!(config.check_higher_order);
    }

    // ===== RudraConfig builders =====

    #[kani::proof]
    fn verify_config_with_crate_path() {
        let config = RudraConfig::default().with_crate_path(PathBuf::from("/test"));
        assert!(config.crate_path.is_some());
    }

    #[kani::proof]
    fn verify_config_with_timeout() {
        let config = RudraConfig::default().with_timeout(Duration::from_secs(60));
        assert!(config.timeout == Duration::from_secs(60));
    }

    #[kani::proof]
    fn verify_config_with_send_sync_checking() {
        let config = RudraConfig::default().with_send_sync_checking(false);
        assert!(!config.check_send_sync);
    }

    #[kani::proof]
    fn verify_config_with_panic_safety_checking() {
        let config = RudraConfig::default().with_panic_safety_checking(false);
        assert!(!config.check_panic_safety);
    }

    #[kani::proof]
    fn verify_config_with_higher_order_checking() {
        let config = RudraConfig::default().with_higher_order_checking(false);
        assert!(!config.check_higher_order);
    }

    #[kani::proof]
    fn verify_config_with_all_analyses() {
        let mut config = RudraConfig::default();
        config.all_analyses = false;
        config = config.with_all_analyses();
        assert!(config.all_analyses);
    }

    // ===== Backend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = RudraBackend::new();
        assert!(backend.config.timeout == Duration::from_secs(300));
        assert!(backend.config.all_analyses);
    }

    #[kani::proof]
    fn verify_backend_default_matches_new() {
        let b1 = RudraBackend::new();
        let b2 = RudraBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.check_send_sync == b2.config.check_send_sync);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_values() {
        let config = RudraConfig {
            crate_path: Some(PathBuf::from("/test")),
            timeout: Duration::from_secs(60),
            all_analyses: false,
            check_send_sync: false,
            check_panic_safety: false,
            check_higher_order: false,
        };
        let backend = RudraBackend::with_config(config);
        assert!(backend.config.timeout == Duration::from_secs(60));
        assert!(!backend.config.all_analyses);
        assert!(!backend.config.check_send_sync);
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = RudraBackend::new();
        assert!(matches!(backend.id(), BackendId::Rudra));
    }

    #[kani::proof]
    fn verify_supports_memory_safety_and_data_race() {
        let backend = RudraBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::MemorySafety));
        assert!(supported.contains(&PropertyType::DataRace));
        assert!(supported.len() == 2);
    }

    // ===== Rudra flags =====

    #[kani::proof]
    fn verify_build_rudra_flags_all() {
        let backend = RudraBackend::new();
        let flags = backend.build_rudra_flags();
        assert!(flags.contains("-Zrudra-send-sync"));
        assert!(flags.contains("-Zrudra-panic-safety"));
        assert!(flags.contains("-Zrudra-higher-order"));
    }

    #[kani::proof]
    fn verify_build_rudra_flags_empty() {
        let config = RudraConfig {
            check_send_sync: false,
            check_panic_safety: false,
            check_higher_order: false,
            ..Default::default()
        };
        let backend = RudraBackend::with_config(config);
        let flags = backend.build_rudra_flags();
        assert!(flags.is_empty());
    }

    // ===== Output parsing =====

    #[kani::proof]
    fn verify_parse_rudra_output_clean() {
        let backend = RudraBackend::new();
        let (status, findings) = backend.parse_rudra_output("Analysis complete", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(findings.is_empty());
    }

    #[kani::proof]
    fn verify_parse_rudra_output_unsafe_dataflow() {
        let backend = RudraBackend::new();
        let (status, findings) = backend.parse_rudra_output("[UnsafeDataflow] issue", false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
    }

    #[kani::proof]
    fn verify_parse_rudra_output_send_sync_variance() {
        let backend = RudraBackend::new();
        let (status, findings) = backend.parse_rudra_output("[SendSyncVariance] issue", false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
    }

    #[kani::proof]
    fn verify_parse_rudra_output_panic_safety() {
        let backend = RudraBackend::new();
        let (status, findings) = backend.parse_rudra_output("[PanicSafety] issue", false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
    }

    #[kani::proof]
    fn verify_parse_rudra_output_higher_order() {
        let backend = RudraBackend::new();
        let (status, findings) = backend.parse_rudra_output("[HigherOrderInvariant] issue", false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
    }

    #[kani::proof]
    fn verify_parse_rudra_output_failed_no_issues() {
        let backend = RudraBackend::new();
        let (status, _) = backend.parse_rudra_output("some error", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rudra_config_default() {
        let config = RudraConfig::default();
        assert!(config.crate_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(config.all_analyses);
        assert!(config.check_send_sync);
        assert!(config.check_panic_safety);
        assert!(config.check_higher_order);
    }

    #[test]
    fn test_rudra_config_builder() {
        let config = RudraConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_timeout(Duration::from_secs(120))
            .with_send_sync_checking(false)
            .with_panic_safety_checking(false)
            .with_higher_order_checking(false);

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert!(!config.check_send_sync);
        assert!(!config.check_panic_safety);
        assert!(!config.check_higher_order);
    }

    #[test]
    fn test_rudra_backend_id() {
        let backend = RudraBackend::new();
        assert_eq!(backend.id(), BackendId::Rudra);
    }

    #[test]
    fn test_rudra_supports_memory_safety() {
        let backend = RudraBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::MemorySafety));
        assert!(supported.contains(&PropertyType::DataRace));
    }

    #[test]
    fn test_rudra_parse_output_clean() {
        let backend = RudraBackend::new();
        let (status, findings) = backend.parse_rudra_output("Analysis complete", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(findings.is_empty());
    }

    #[test]
    fn test_rudra_parse_output_with_issues() {
        let backend = RudraBackend::new();
        let output =
            "[UnsafeDataflow] Potential issue in foo\n[SendSyncVariance] Incorrect Send impl";
        let (status, findings) = backend.parse_rudra_output(output, false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert_eq!(findings.len(), 2);
    }

    #[tokio::test]
    async fn test_rudra_health_check() {
        let backend = RudraBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(reason.contains("Rudra") || reason.contains("rudra"));
            }
            HealthStatus::Degraded { .. } => {}
        }
    }

    #[tokio::test]
    async fn test_rudra_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = RudraBackend::new();
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
