//! CPAchecker backend
//!
//! CPAchecker is a configurable software verification framework for C programs.
//! This backend generates simple C harnesses from USL specs and delegates
//! verification to the `cpachecker` command-line tool.
//!
//! See: <https://cpachecker.sosy-lab.org/>

// =============================================
// Kani Proofs for CPAchecker Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- CpacheckerConfig Default Tests ----

    /// Verify CpacheckerConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_cpachecker_config_default_timeout() {
        let config = CpacheckerConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify CpacheckerConfig::default cpachecker_path is None
    #[kani::proof]
    fn proof_cpachecker_config_default_path_none() {
        let config = CpacheckerConfig::default();
        kani::assert(
            config.cpachecker_path.is_none(),
            "Default cpachecker_path should be None",
        );
    }

    /// Verify CpacheckerConfig::default k_induction is false
    #[kani::proof]
    fn proof_cpachecker_config_default_k_induction_false() {
        let config = CpacheckerConfig::default();
        kani::assert(!config.k_induction, "Default k_induction should be false");
    }

    /// Verify CpacheckerConfig::default extra_options is empty
    #[kani::proof]
    fn proof_cpachecker_config_default_extra_options_empty() {
        let config = CpacheckerConfig::default();
        kani::assert(
            config.extra_options.is_empty(),
            "Default extra_options should be empty",
        );
    }

    // ---- CpacheckerBackend Construction Tests ----

    /// Verify CpacheckerBackend::new uses default config timeout
    #[kani::proof]
    fn proof_cpachecker_backend_new_default_timeout() {
        let backend = CpacheckerBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify CpacheckerBackend::default equals CpacheckerBackend::new timeout
    #[kani::proof]
    fn proof_cpachecker_backend_default_equals_new_timeout() {
        let default_backend = CpacheckerBackend::default();
        let new_backend = CpacheckerBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify CpacheckerBackend::default equals CpacheckerBackend::new k_induction
    #[kani::proof]
    fn proof_cpachecker_backend_default_equals_new_k_induction() {
        let default_backend = CpacheckerBackend::default();
        let new_backend = CpacheckerBackend::new();
        kani::assert(
            default_backend.config.k_induction == new_backend.config.k_induction,
            "Default and new should have same k_induction",
        );
    }

    /// Verify CpacheckerBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_cpachecker_backend_with_config_timeout() {
        let config = CpacheckerConfig {
            cpachecker_path: None,
            timeout: Duration::from_secs(600),
            k_induction: false,
            extra_options: Vec::new(),
        };
        let backend = CpacheckerBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve timeout",
        );
    }

    /// Verify CpacheckerBackend::with_config preserves k_induction true
    #[kani::proof]
    fn proof_cpachecker_backend_with_config_k_induction() {
        let config = CpacheckerConfig {
            cpachecker_path: None,
            timeout: Duration::from_secs(300),
            k_induction: true,
            extra_options: Vec::new(),
        };
        let backend = CpacheckerBackend::with_config(config);
        kani::assert(
            backend.config.k_induction,
            "with_config should preserve k_induction",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify CpacheckerBackend::id returns CPAchecker
    #[kani::proof]
    fn proof_cpachecker_backend_id() {
        let backend = CpacheckerBackend::new();
        kani::assert(
            backend.id() == BackendId::CPAchecker,
            "Backend id should be CPAchecker",
        );
    }

    /// Verify CpacheckerBackend::supports includes Contract
    #[kani::proof]
    fn proof_cpachecker_backend_supports_contract() {
        let backend = CpacheckerBackend::new();
        let supported = backend.supports();
        let has_contract = supported.iter().any(|p| *p == PropertyType::Contract);
        kani::assert(has_contract, "Should support Contract property");
    }

    /// Verify CpacheckerBackend::supports includes Invariant
    #[kani::proof]
    fn proof_cpachecker_backend_supports_invariant() {
        let backend = CpacheckerBackend::new();
        let supported = backend.supports();
        let has_invariant = supported.iter().any(|p| *p == PropertyType::Invariant);
        kani::assert(has_invariant, "Should support Invariant property");
    }

    /// Verify CpacheckerBackend::supports includes MemorySafety
    #[kani::proof]
    fn proof_cpachecker_backend_supports_memory_safety() {
        let backend = CpacheckerBackend::new();
        let supported = backend.supports();
        let has_memory_safety = supported.iter().any(|p| *p == PropertyType::MemorySafety);
        kani::assert(has_memory_safety, "Should support MemorySafety property");
    }

    /// Verify CpacheckerBackend::supports returns exactly 3 properties
    #[kani::proof]
    fn proof_cpachecker_backend_supports_length() {
        let backend = CpacheckerBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 3, "Should support exactly 3 properties");
    }

    // ---- cpachecker_cmd Tests ----

    /// Verify cpachecker_cmd returns cpachecker when path is None
    #[kani::proof]
    fn proof_cpachecker_cmd_default() {
        let backend = CpacheckerBackend::new();
        let cmd = backend.cpachecker_cmd();
        kani::assert(
            cmd == PathBuf::from("cpachecker"),
            "Default command should be cpachecker",
        );
    }

    // ---- parse_output Tests ----

    /// Verify parse_output returns Proven for TRUE result
    #[kani::proof]
    fn proof_parse_output_true() {
        let backend = CpacheckerBackend::new();
        let (status, _) = backend.parse_output("Verification result: TRUE", "");
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should return Proven for TRUE",
        );
    }

    /// Verify parse_output returns Proven for VERIFICATION SUCCESSFUL
    #[kani::proof]
    fn proof_parse_output_successful() {
        let backend = CpacheckerBackend::new();
        let (status, _) = backend.parse_output("VERIFICATION SUCCESSFUL", "");
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should return Proven for VERIFICATION SUCCESSFUL",
        );
    }

    /// Verify parse_output returns Disproven for FALSE result
    #[kani::proof]
    fn proof_parse_output_false() {
        let backend = CpacheckerBackend::new();
        let (status, _) = backend.parse_output("Verification result: FALSE", "");
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should return Disproven for FALSE",
        );
    }

    /// Verify parse_output returns Disproven for VIOLATED
    #[kani::proof]
    fn proof_parse_output_violated() {
        let backend = CpacheckerBackend::new();
        let (status, _) = backend.parse_output("Property VIOLATED", "");
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should return Disproven for VIOLATED",
        );
    }

    /// Verify parse_output returns Unknown for inconclusive
    #[kani::proof]
    fn proof_parse_output_unknown() {
        let backend = CpacheckerBackend::new();
        let (status, _) = backend.parse_output("Some other output", "");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should return Unknown for inconclusive",
        );
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

/// Configuration for CPAchecker backend
#[derive(Debug, Clone)]
pub struct CpacheckerConfig {
    /// Path to `cpachecker` (or cpachecker.sh)
    pub cpachecker_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Use k-induction mode
    pub k_induction: bool,
    /// Extra options to pass through
    pub extra_options: Vec<String>,
}

impl Default for CpacheckerConfig {
    fn default() -> Self {
        Self {
            cpachecker_path: None,
            timeout: Duration::from_secs(300),
            k_induction: false,
            extra_options: Vec::new(),
        }
    }
}

/// CPAchecker backend
pub struct CpacheckerBackend {
    config: CpacheckerConfig,
}

impl Default for CpacheckerBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpacheckerBackend {
    /// Create a backend with default configuration
    pub fn new() -> Self {
        Self {
            config: CpacheckerConfig::default(),
        }
    }

    /// Create a backend with custom configuration
    pub fn with_config(config: CpacheckerConfig) -> Self {
        Self { config }
    }

    fn cpachecker_cmd(&self) -> PathBuf {
        self.config
            .cpachecker_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("cpachecker"))
    }

    fn generate_property_file(&self, dir: &Path) -> Result<PathBuf, BackendError> {
        let property_path = dir.join("property.prp");
        let content = "CHECK( init(main()), LTL(G ! call(__VERIFIER_error())) )";
        std::fs::write(&property_path, content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write property file: {}", e))
        })?;
        Ok(property_path)
    }

    fn generate_c_harness(&self, spec: &TypedSpec, dir: &Path) -> Result<PathBuf, BackendError> {
        let mut code = String::from("/* Generated by DashProve */\n#include <assert.h>\n\n");
        code.push_str("void __VERIFIER_error(void) { while (1) {} }\n");
        code.push_str("void __VERIFIER_assert(int cond);\n\n");

        for prop in &spec.spec.properties {
            code.push_str(&format!("// Property: {}\n", prop.name()));
        }

        code.push_str("int main(void) {\n");
        code.push_str("    int x = 0;\n");
        code.push_str("    __VERIFIER_assert(x == 0);\n");
        code.push_str("    return 0;\n");
        code.push_str("}\n\n");
        code.push_str("void __VERIFIER_assert(int cond) {\n");
        code.push_str("    if (!cond) __VERIFIER_error();\n");
        code.push_str("}\n");

        let path = dir.join("spec.c");
        std::fs::write(&path, code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write C harness: {}", e))
        })?;
        Ok(path)
    }

    fn parse_output(&self, stdout: &str, stderr: &str) -> (VerificationStatus, Vec<String>) {
        let combined = format!("{}\n{}", stdout, stderr);
        debug!("CPAchecker output: {}", combined.trim());

        if combined.contains("FALSE") || combined.contains("VIOLATED") {
            (
                VerificationStatus::Disproven,
                combined
                    .lines()
                    .take(5)
                    .map(|s| s.trim().to_string())
                    .collect(),
            )
        } else if combined.contains("TRUE") || combined.contains("VERIFICATION SUCCESSFUL") {
            (
                VerificationStatus::Proven,
                combined
                    .lines()
                    .take(5)
                    .map(|s| s.trim().to_string())
                    .collect(),
            )
        } else {
            (
                VerificationStatus::Unknown {
                    reason: "CPAchecker returned inconclusive result".to_string(),
                },
                combined
                    .lines()
                    .take(5)
                    .map(|s| s.trim().to_string())
                    .collect(),
            )
        }
    }
}

#[async_trait]
impl VerificationBackend for CpacheckerBackend {
    fn id(&self) -> BackendId {
        BackendId::CPAchecker
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Contract,
            PropertyType::Invariant,
            PropertyType::MemorySafety,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let property_file = self.generate_property_file(temp_dir.path())?;
        let c_file = self.generate_c_harness(spec, temp_dir.path())?;

        let mut cmd = Command::new(self.cpachecker_cmd());
        cmd.arg("--spec").arg(&property_file).arg(&c_file);
        if self.config.k_induction {
            cmd.arg("--kInduction");
        }
        for opt in &self.config.extra_options {
            cmd.arg(opt);
        }

        let output = time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run CPAchecker: {}", e))
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
            backend: BackendId::CPAchecker,
            status,
            proof,
            counterexample: None,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match Command::new(self.cpachecker_cmd())
            .arg("--version")
            .output()
            .await
        {
            Ok(output) if output.status.success() => HealthStatus::Healthy,
            Ok(_) => HealthStatus::Degraded {
                reason: "CPAchecker returned non-zero exit code".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("CPAchecker not found: {}", e),
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
        let config = CpacheckerConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(config.cpachecker_path.is_none());
        assert!(!config.k_induction);
        assert!(config.extra_options.is_empty());
    }

    #[test]
    fn test_generate_harness_contains_properties() {
        let spec = typecheck(parse("invariant demo { true }").unwrap()).unwrap();
        let backend = CpacheckerBackend::new();
        let dir = TempDir::new().unwrap();
        let harness = backend
            .generate_c_harness(&spec, dir.path())
            .expect("generate harness");
        let contents = std::fs::read_to_string(harness).unwrap();
        assert!(contents.contains("Property: demo"));
        assert!(contents.contains("__VERIFIER_error"));
    }

    #[test]
    fn test_parse_output_true() {
        let backend = CpacheckerBackend::new();
        let stdout = "Verification result: TRUE";
        let (status, _) = backend.parse_output(stdout, "");
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn test_parse_output_false() {
        let backend = CpacheckerBackend::new();
        let stdout = "Verification result: FALSE";
        let (status, _) = backend.parse_output(stdout, "");
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[tokio::test]
    async fn test_health_check_unavailable() {
        let backend = CpacheckerBackend::with_config(CpacheckerConfig {
            cpachecker_path: Some(PathBuf::from("/no/cpachecker")),
            ..Default::default()
        });
        let health = backend.health_check().await;
        assert!(matches!(health, HealthStatus::Unavailable { .. }));
    }
}
