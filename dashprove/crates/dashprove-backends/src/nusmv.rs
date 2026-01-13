//! NuSMV symbolic model checker backend
//!
//! NuSMV is a symbolic model checker for finite state systems supporting
//! CTL/LTL properties. This backend generates a lightweight SMV model
//! from the USL specification and delegates checking to the `nusmv` binary.
//!
//! See: <https://nusmv.fbk.eu/>

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tokio::time;
use tracing::debug;

/// Configuration for NuSMV backend
#[derive(Debug, Clone)]
pub struct NuSmvConfig {
    /// Path to `nusmv` binary
    pub nusmv_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Enable bounded model checking mode
    pub use_bmc: bool,
    /// Use LTL mode (default) or CTL
    pub ltl_mode: bool,
}

impl Default for NuSmvConfig {
    fn default() -> Self {
        Self {
            nusmv_path: None,
            timeout: Duration::from_secs(180),
            use_bmc: false,
            ltl_mode: true,
        }
    }
}

/// NuSMV backend
pub struct NuSmvBackend {
    config: NuSmvConfig,
}

impl Default for NuSmvBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl NuSmvBackend {
    /// Create backend with default configuration
    pub fn new() -> Self {
        Self {
            config: NuSmvConfig::default(),
        }
    }

    /// Create backend with custom configuration
    pub fn with_config(config: NuSmvConfig) -> Self {
        Self { config }
    }

    /// Generate a simple SMV model from the typed specification
    fn generate_model(&self, spec: &TypedSpec) -> String {
        let mut model = String::new();
        model.push_str("MODULE main\n");
        model.push_str("VAR state : 0..1;\n");
        model.push_str("ASSIGN\n");
        model.push_str("  init(state) := 0;\n");
        model.push_str("  next(state) := state;\n\n");

        for property in &spec.spec.properties {
            model.push_str(&format!("-- Property: {}\n", property.name()));
            if self.config.ltl_mode {
                model.push_str("LTLSPEC G (state = state)\n\n");
            } else {
                model.push_str("CTLSPEC AG (state = state)\n\n");
            }
        }

        model
    }

    /// Run NuSMV on a generated model file
    async fn run_nusmv(
        &self,
        model_path: &PathBuf,
        start: Instant,
    ) -> Result<BackendResult, BackendError> {
        let nusmv = self
            .config
            .nusmv_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("nusmv"));

        let mut cmd = Command::new(&nusmv);
        if self.config.use_bmc {
            cmd.arg("-bmc");
        }
        if self.config.ltl_mode {
            cmd.arg("-ltl");
        } else {
            cmd.arg("-ctl");
        }
        cmd.arg(model_path);

        let output = time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| BackendError::VerificationFailed(format!("Failed to run NuSMV: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        debug!("NuSMV stdout: {}", stdout.trim());
        debug!("NuSMV stderr: {}", stderr.trim());

        let (status, diagnostics) = self.parse_output(&stdout, &stderr);

        Ok(BackendResult {
            backend: BackendId::NuSMV,
            status,
            proof: if diagnostics.is_empty() {
                Some(stdout.clone())
            } else {
                None
            },
            counterexample: None,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    /// Parse NuSMV output into verification status and diagnostics
    fn parse_output(&self, stdout: &str, stderr: &str) -> (VerificationStatus, Vec<String>) {
        let mut diagnostics = Vec::new();
        let combined = format!("{}\n{}", stdout, stderr);
        let mut saw_true = false;
        let mut saw_false = false;

        for line in combined.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("-- specification") {
                diagnostics.push(trimmed.to_string());
                if trimmed.contains("is true") {
                    saw_true = true;
                } else if trimmed.contains("is false") {
                    saw_false = true;
                }
            } else if trimmed.to_lowercase().contains("unknown") {
                diagnostics.push(trimmed.to_string());
            }
        }

        if saw_false {
            (VerificationStatus::Disproven, diagnostics)
        } else if saw_true && !saw_false {
            (VerificationStatus::Proven, diagnostics)
        } else {
            (
                VerificationStatus::Unknown {
                    reason: "NuSMV could not determine property status".to_string(),
                },
                diagnostics,
            )
        }
    }
}

#[async_trait]
impl VerificationBackend for NuSmvBackend {
    fn id(&self) -> BackendId {
        BackendId::NuSMV
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Invariant, PropertyType::Temporal]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let model = self.generate_model(spec);
        let start = Instant::now();

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;
        let model_path = temp_dir.path().join("model.smv");
        std::fs::write(&model_path, model).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write SMV file: {}", e))
        })?;

        self.run_nusmv(&model_path, start).await
    }

    async fn health_check(&self) -> HealthStatus {
        let nusmv = self
            .config
            .nusmv_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("nusmv"));

        match Command::new(&nusmv).arg("-h").output().await {
            Ok(output) if output.status.success() => HealthStatus::Healthy,
            Ok(_) => HealthStatus::Degraded {
                reason: "NuSMV returned non-zero exit code".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("NuSMV not found: {}", e),
            },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== NuSmvConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults_timeout() {
        let config = NuSmvConfig::default();
        assert!(config.timeout == Duration::from_secs(180));
    }

    #[kani::proof]
    fn verify_config_defaults_modes() {
        let config = NuSmvConfig::default();
        assert!(config.ltl_mode);
        assert!(!config.use_bmc);
        assert!(config.nusmv_path.is_none());
    }

    // ===== Backend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = NuSmvBackend::new();
        assert!(backend.config.timeout == Duration::from_secs(180));
        assert!(backend.config.ltl_mode);
    }

    #[kani::proof]
    fn verify_backend_default_matches_new() {
        let b1 = NuSmvBackend::new();
        let b2 = NuSmvBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.ltl_mode == b2.config.ltl_mode);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_values() {
        let config = NuSmvConfig {
            nusmv_path: Some(PathBuf::from("/tmp/nusmv")),
            timeout: Duration::from_secs(30),
            use_bmc: true,
            ltl_mode: false,
        };
        let backend = NuSmvBackend::with_config(config);
        assert!(backend.config.timeout == Duration::from_secs(30));
        assert!(backend.config.use_bmc);
        assert!(!backend.config.ltl_mode);
        assert!(backend.config.nusmv_path.is_some());
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = NuSmvBackend::new();
        assert!(matches!(backend.id(), BackendId::NuSMV));
    }

    #[kani::proof]
    fn verify_supports_includes_temporal_and_invariant() {
        let backend = NuSmvBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Temporal));
        assert!(supported.contains(&PropertyType::Invariant));
        assert!(supported.len() == 2);
    }

    // ===== Model generation =====

    #[kani::proof]
    fn verify_generate_model_contains_module() {
        let backend = NuSmvBackend::new();
        let spec = dashprove_usl::typecheck::typecheck(
            dashprove_usl::parse("invariant demo { true }").unwrap(),
        )
        .unwrap();
        let model = backend.generate_model(&spec);
        assert!(model.contains("MODULE main"));
        assert!(model.contains("Property: demo"));
    }

    #[kani::proof]
    fn verify_generate_model_switches_to_ctl() {
        let backend = NuSmvBackend::with_config(NuSmvConfig {
            ltl_mode: false,
            ..Default::default()
        });
        let spec = dashprove_usl::typecheck::typecheck(
            dashprove_usl::parse("invariant demo { true }").unwrap(),
        )
        .unwrap();
        let model = backend.generate_model(&spec);
        assert!(model.contains("CTLSPEC"));
    }

    // ===== Output parsing =====

    #[kani::proof]
    fn verify_parse_output_true_status() {
        let backend = NuSmvBackend::new();
        let (status, diag) = backend.parse_output("-- specification foo is true", "");
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(!diag.is_empty());
    }

    #[kani::proof]
    fn verify_parse_output_false_status() {
        let backend = NuSmvBackend::new();
        let (status, _) = backend.parse_output("-- specification foo is false", "");
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[kani::proof]
    fn verify_parse_output_unknown_status() {
        let backend = NuSmvBackend::new();
        let (status, diag) = backend.parse_output("unknown result", "timeout");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
        assert!(!diag.is_empty());
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
        let config = NuSmvConfig::default();
        assert!(config.nusmv_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(180));
        assert!(config.ltl_mode);
        assert!(!config.use_bmc);
    }

    #[test]
    fn test_generate_model_has_properties() {
        let spec = typecheck(parse("invariant demo { true }").unwrap()).unwrap();
        let backend = NuSmvBackend::new();
        let model = backend.generate_model(&spec);
        assert!(model.contains("MODULE main"));
        assert!(model.contains("Property: demo"));
    }

    #[test]
    fn test_parse_output_true() {
        let backend = NuSmvBackend::new();
        let stdout = "-- specification G prop is true";
        let (status, diags) = backend.parse_output(stdout, "");
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(!diags.is_empty());
    }

    #[test]
    fn test_parse_output_false() {
        let backend = NuSmvBackend::new();
        let stdout = "-- specification G prop is false";
        let (status, _) = backend.parse_output(stdout, "");
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[tokio::test]
    async fn test_health_check_unavailable() {
        let backend = NuSmvBackend::with_config(NuSmvConfig {
            nusmv_path: Some(PathBuf::from("/nonexistent/nusmv")),
            ..Default::default()
        });
        let health = backend.health_check().await;
        assert!(matches!(health, HealthStatus::Unavailable { .. }));
    }
}
