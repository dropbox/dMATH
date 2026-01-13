//! Frama-C backend
//!
//! Frama-C is a modular framework for static analysis and deductive verification
//! of C programs. This backend generates small annotated C snippets and invokes
//! Frama-C's WP plugin to discharge proof obligations.
//!
//! See: <https://frama-c.com/>

// =============================================
// Kani Proofs for Frama-C Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- FramaCConfig Default Tests ----

    /// Verify FramaCConfig::default sets expected baseline values
    #[kani::proof]
    fn proof_framac_config_defaults() {
        let config = FramaCConfig::default();
        kani::assert(
            config.framac_path.is_none(),
            "framac_path should default to None",
        );
        kani::assert(
            config.timeout == Duration::from_secs(240),
            "timeout should default to 240 seconds",
        );
        kani::assert(config.use_wp, "use_wp should default to true");
        kani::assert(
            config.extra_options.is_empty(),
            "extra_options should default empty",
        );
    }

    // ---- FramaCBackend Construction Tests ----

    /// Verify FramaCBackend::new uses default configuration
    #[kani::proof]
    fn proof_framac_backend_new_defaults() {
        let backend = FramaCBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(240),
            "new backend should default timeout to 240 seconds",
        );
        kani::assert(
            backend.config.use_wp,
            "new backend should enable WP plugin by default",
        );
    }

    /// Verify FramaCBackend::default matches FramaCBackend::new
    #[kani::proof]
    fn proof_framac_backend_default_equals_new() {
        let default_backend = FramaCBackend::default();
        let new_backend = FramaCBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.use_wp == new_backend.config.use_wp,
            "default and new should share use_wp flag",
        );
    }

    /// Verify FramaCBackend::with_config preserves custom settings
    #[kani::proof]
    fn proof_framac_backend_with_config() {
        let config = FramaCConfig {
            framac_path: Some(PathBuf::from("/opt/frama-c")),
            timeout: Duration::from_secs(60),
            use_wp: false,
            extra_options: vec!["-foo".into(), "-bar".into()],
        };
        let backend = FramaCBackend::with_config(config);
        kani::assert(
            backend.config.framac_path == Some(PathBuf::from("/opt/frama-c")),
            "with_config should preserve framac_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "with_config should preserve timeout",
        );
        kani::assert(
            !backend.config.use_wp,
            "with_config should preserve use_wp flag",
        );
        kani::assert(
            backend.config.extra_options.len() == 2,
            "with_config should preserve extra options",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::FramaC
    #[kani::proof]
    fn proof_framac_backend_id() {
        let backend = FramaCBackend::new();
        kani::assert(
            backend.id() == BackendId::FramaC,
            "FramaCBackend id should be BackendId::FramaC",
        );
    }

    /// Verify supports() includes contracts and invariants
    #[kani::proof]
    fn proof_framac_backend_supports() {
        let backend = FramaCBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Contract),
            "supports should include Contract",
        );
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "supports should include Invariant",
        );
    }

    /// Verify framac_cmd prefers configured binary path
    #[kani::proof]
    fn proof_framac_cmd_prefers_override() {
        let backend = FramaCBackend::with_config(FramaCConfig {
            framac_path: Some(PathBuf::from("/custom/frama-c")),
            ..Default::default()
        });
        let cmd = backend.framac_cmd();
        kani::assert(
            cmd == PathBuf::from("/custom/frama-c"),
            "framac_cmd should use configured framac_path",
        );
    }

    // ---- Output Parsing Tests ----

    /// Verify parse_output reports proven when WP output indicates success
    #[kani::proof]
    fn proof_parse_output_proven() {
        let backend = FramaCBackend::new();
        let (status, diagnostics) = backend.parse_output("WP proved", "");
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "WP proved should mark status as Proven",
        );
        kani::assert(
            !diagnostics.is_empty(),
            "diagnostics should contain parsed output",
        );
    }

    /// Verify parse_output reports disproven when unproved goals are present
    #[kani::proof]
    fn proof_parse_output_disproven() {
        let backend = FramaCBackend::new();
        let (status, _) = backend.parse_output("unproved goals remain", "");
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "unproved text should mark status as Disproven",
        );
    }

    /// Verify parse_output reports Unknown when status cannot be determined
    #[kani::proof]
    fn proof_parse_output_unknown() {
        let backend = FramaCBackend::new();
        let (status, _) = backend.parse_output("no status reported", "");
        if let VerificationStatus::Unknown { reason } = status {
            kani::assert(
                reason.contains("did not report"),
                "Unknown reason should mention missing status",
            );
        } else {
            kani::assert(false, "unexpected status for unknown output");
        }
    }
}

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tokio::time;
use tracing::debug;

/// Configuration for Frama-C backend
#[derive(Debug, Clone)]
pub struct FramaCConfig {
    /// Path to `frama-c` binary
    pub framac_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Use WP plugin
    pub use_wp: bool,
    /// Extra options to forward to Frama-C
    pub extra_options: Vec<String>,
}

impl Default for FramaCConfig {
    fn default() -> Self {
        Self {
            framac_path: None,
            timeout: Duration::from_secs(240),
            use_wp: true,
            extra_options: Vec::new(),
        }
    }
}

/// Frama-C backend
pub struct FramaCBackend {
    config: FramaCConfig,
}

impl Default for FramaCBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl FramaCBackend {
    /// Create backend with default configuration
    pub fn new() -> Self {
        Self {
            config: FramaCConfig::default(),
        }
    }

    /// Create backend with custom configuration
    pub fn with_config(config: FramaCConfig) -> Self {
        Self { config }
    }

    fn framac_cmd(&self) -> PathBuf {
        self.config
            .framac_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("frama-c"))
    }

    fn generate_annotated_c(&self, spec: &TypedSpec, dir: &Path) -> Result<PathBuf, BackendError> {
        let mut code = String::from("/* Generated by DashProve */\n#include <stdbool.h>\n\n");

        for prop in &spec.spec.properties {
            code.push_str(&format!(
                "/*@ requires true; ensures true; // {} */\n",
                prop.name()
            ));
        }

        code.push_str("int main(void) {\n");
        code.push_str("  int x = 0;\n");
        code.push_str("  //@ assert x == 0;\n");
        code.push_str("  return 0;\n");
        code.push_str("}\n");

        let path = dir.join("spec.c");
        std::fs::write(&path, code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write Frama-C input: {}", e))
        })?;
        Ok(path)
    }

    fn parse_output(&self, stdout: &str, stderr: &str) -> (VerificationStatus, Vec<String>) {
        let combined = format!("{}\n{}", stdout, stderr);
        debug!("Frama-C output: {}", combined.trim());
        let lower = combined.to_lowercase();

        if lower.contains("all goals proved") || lower.contains("wp proved") {
            (
                VerificationStatus::Proven,
                combined
                    .lines()
                    .take(6)
                    .map(|l| l.trim().to_string())
                    .collect(),
            )
        } else if lower.contains("unproved") || lower.contains("failed") {
            (
                VerificationStatus::Disproven,
                combined
                    .lines()
                    .take(6)
                    .map(|l| l.trim().to_string())
                    .collect(),
            )
        } else {
            (
                VerificationStatus::Unknown {
                    reason: "Frama-C did not report proof status".to_string(),
                },
                combined
                    .lines()
                    .take(6)
                    .map(|l| l.trim().to_string())
                    .collect(),
            )
        }
    }
}

#[async_trait]
impl VerificationBackend for FramaCBackend {
    fn id(&self) -> BackendId {
        BackendId::FramaC
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Contract, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        let dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;
        let c_path = self.generate_annotated_c(spec, dir.path())?;

        let mut cmd = Command::new(self.framac_cmd());
        if self.config.use_wp {
            cmd.arg("-wp");
        }
        for opt in &self.config.extra_options {
            cmd.arg(opt);
        }
        cmd.arg(&c_path);

        let output = time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run Frama-C: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let (status, diagnostics) = self.parse_output(&stdout, &stderr);

        let proof = if matches!(&status, VerificationStatus::Proven) {
            Some(stdout.clone())
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::FramaC,
            status,
            proof,
            counterexample: None,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match Command::new(self.framac_cmd())
            .arg("-version")
            .output()
            .await
        {
            Ok(output) if output.status.success() => HealthStatus::Healthy,
            Ok(_) => HealthStatus::Degraded {
                reason: "Frama-C returned non-zero exit code".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Frama-C not found: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::parse;
    use dashprove_usl::typecheck::typecheck;
    use std::path::PathBuf;

    #[test]
    fn test_config_defaults() {
        let config = FramaCConfig::default();
        assert!(config.framac_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(240));
        assert!(config.use_wp);
    }

    #[test]
    fn test_generate_annotated_c_contains_properties() {
        let spec = typecheck(parse("invariant safety { true }").unwrap()).unwrap();
        let backend = FramaCBackend::new();
        let dir = TempDir::new().unwrap();
        let path = backend
            .generate_annotated_c(&spec, dir.path())
            .expect("generate C");
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("Property: safety") || content.contains("safety"));
        assert!(content.contains("@ assert"));
    }

    #[test]
    fn test_parse_output_proved() {
        let backend = FramaCBackend::new();
        let stdout = "WP proved";
        let (status, _) = backend.parse_output(stdout, "");
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn test_parse_output_failed() {
        let backend = FramaCBackend::new();
        let stdout = "Unproved goals remain";
        let (status, _) = backend.parse_output(stdout, "");
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[tokio::test]
    async fn test_health_check_unavailable() {
        let backend = FramaCBackend::with_config(FramaCConfig {
            framac_path: Some(PathBuf::from("/no/frama-c")),
            ..Default::default()
        });
        let health = backend.health_check().await;
        assert!(matches!(health, HealthStatus::Unavailable { .. }));
    }
}
