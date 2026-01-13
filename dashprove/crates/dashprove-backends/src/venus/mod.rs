//! Venus DNN verification backend
//!
//! Venus is a complete DNN verifier using:
//! - Mixed Integer Linear Programming (MILP)
//! - Branch and bound for scalability
//! - Multiple LP solver backends (Gurobi, GLPK, CBC)
//!
//! # Installation
//!
//! ```bash
//! pip install venus-verification
//! ```

// =============================================
// Kani Proofs for Venus Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use std::time::Duration;

    // ---- SolverBackend Tests ----

    /// Verify SolverBackend::default is Gurobi
    #[kani::proof]
    fn proof_solver_backend_default_is_gurobi() {
        let solver = SolverBackend::default();
        kani::assert(
            matches!(solver, SolverBackend::Gurobi),
            "Default solver should be Gurobi",
        );
    }

    /// Verify SolverBackend::Gurobi as_str
    #[kani::proof]
    fn proof_solver_backend_gurobi_str() {
        let solver = SolverBackend::Gurobi;
        kani::assert(solver.as_str() == "gurobi", "Gurobi should be gurobi");
    }

    /// Verify SolverBackend::GLPK as_str
    #[kani::proof]
    fn proof_solver_backend_glpk_str() {
        let solver = SolverBackend::GLPK;
        kani::assert(solver.as_str() == "glpk", "GLPK should be glpk");
    }

    /// Verify SolverBackend::CBC as_str
    #[kani::proof]
    fn proof_solver_backend_cbc_str() {
        let solver = SolverBackend::CBC;
        kani::assert(solver.as_str() == "cbc", "CBC should be cbc");
    }

    // ---- VenusConfig Default Tests ----

    /// Verify VenusConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_venus_config_default_timeout() {
        let config = VenusConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify VenusConfig::default python_path is None
    #[kani::proof]
    fn proof_venus_config_default_python_path_none() {
        let config = VenusConfig::default();
        kani::assert(
            config.python_path.is_none(),
            "Default python_path should be None",
        );
    }

    /// Verify VenusConfig::default solver is Gurobi
    #[kani::proof]
    fn proof_venus_config_default_solver() {
        let config = VenusConfig::default();
        kani::assert(
            matches!(config.solver, SolverBackend::Gurobi),
            "Default solver should be Gurobi",
        );
    }

    /// Verify VenusConfig::default epsilon is 0.01
    #[kani::proof]
    fn proof_venus_config_default_epsilon() {
        let config = VenusConfig::default();
        kani::assert(config.epsilon == 0.01, "Default epsilon should be 0.01");
    }

    /// Verify VenusConfig::default num_workers is 1
    #[kani::proof]
    fn proof_venus_config_default_num_workers() {
        let config = VenusConfig::default();
        kani::assert(config.num_workers == 1, "Default num_workers should be 1");
    }

    /// Verify VenusConfig::default model_path is None
    #[kani::proof]
    fn proof_venus_config_default_model_path_none() {
        let config = VenusConfig::default();
        kani::assert(
            config.model_path.is_none(),
            "Default model_path should be None",
        );
    }

    /// Verify VenusConfig::default use_bnb is true
    #[kani::proof]
    fn proof_venus_config_default_use_bnb() {
        let config = VenusConfig::default();
        kani::assert(config.use_bnb, "Default use_bnb should be true");
    }

    /// Verify VenusConfig::with_glpk sets solver to GLPK
    #[kani::proof]
    fn proof_venus_config_with_glpk() {
        let config = VenusConfig::with_glpk();
        kani::assert(
            matches!(config.solver, SolverBackend::GLPK),
            "with_glpk should set solver to GLPK",
        );
    }

    // ---- VenusBackend Construction Tests ----

    /// Verify VenusBackend::new uses default config
    #[kani::proof]
    fn proof_venus_backend_new_defaults() {
        let backend = VenusBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify VenusBackend::default equals VenusBackend::new
    #[kani::proof]
    fn proof_venus_backend_default_equals_new() {
        let default_backend = VenusBackend::default();
        let new_backend = VenusBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify VenusBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_venus_backend_with_config_timeout() {
        let config = VenusConfig {
            python_path: None,
            solver: SolverBackend::Gurobi,
            epsilon: 0.01,
            num_workers: 1,
            timeout: Duration::from_secs(600),
            model_path: None,
            use_bnb: true,
        };
        let backend = VenusBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "Custom timeout should be preserved",
        );
    }

    /// Verify VenusBackend::with_config preserves custom epsilon
    #[kani::proof]
    fn proof_venus_backend_with_config_epsilon() {
        let config = VenusConfig {
            python_path: None,
            solver: SolverBackend::Gurobi,
            epsilon: 0.05,
            num_workers: 1,
            timeout: Duration::from_secs(300),
            model_path: None,
            use_bnb: true,
        };
        let backend = VenusBackend::with_config(config);
        kani::assert(
            backend.config.epsilon == 0.05,
            "Custom epsilon should be preserved",
        );
    }

    /// Verify VenusBackend::with_config preserves solver
    #[kani::proof]
    fn proof_venus_backend_with_config_solver() {
        let config = VenusConfig {
            python_path: None,
            solver: SolverBackend::CBC,
            epsilon: 0.01,
            num_workers: 1,
            timeout: Duration::from_secs(300),
            model_path: None,
            use_bnb: true,
        };
        let backend = VenusBackend::with_config(config);
        kani::assert(
            matches!(backend.config.solver, SolverBackend::CBC),
            "Custom solver should be preserved",
        );
    }

    /// Verify VenusBackend::with_config preserves num_workers
    #[kani::proof]
    fn proof_venus_backend_with_config_num_workers() {
        let config = VenusConfig {
            python_path: None,
            solver: SolverBackend::Gurobi,
            epsilon: 0.01,
            num_workers: 4,
            timeout: Duration::from_secs(300),
            model_path: None,
            use_bnb: true,
        };
        let backend = VenusBackend::with_config(config);
        kani::assert(
            backend.config.num_workers == 4,
            "Custom num_workers should be preserved",
        );
    }

    /// Verify VenusBackend::with_config preserves use_bnb
    #[kani::proof]
    fn proof_venus_backend_with_config_use_bnb() {
        let config = VenusConfig {
            python_path: None,
            solver: SolverBackend::Gurobi,
            epsilon: 0.01,
            num_workers: 1,
            timeout: Duration::from_secs(300),
            model_path: None,
            use_bnb: false,
        };
        let backend = VenusBackend::with_config(config);
        kani::assert(
            !backend.config.use_bnb,
            "Custom use_bnb should be preserved",
        );
    }

    // ---- Backend Trait Implementation Tests ----

    /// Verify id() returns Venus
    #[kani::proof]
    fn proof_backend_id_is_venus() {
        let backend = VenusBackend::new();
        kani::assert(backend.id() == BackendId::Venus, "ID should be Venus");
    }

    /// Verify supports() includes NeuralRobustness
    #[kani::proof]
    fn proof_venus_supports_neural_robustness() {
        let backend = VenusBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::NeuralRobustness),
            "Should support NeuralRobustness",
        );
    }

    /// Verify supports() includes NeuralReachability
    #[kani::proof]
    fn proof_venus_supports_neural_reachability() {
        let backend = VenusBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::NeuralReachability),
            "Should support NeuralReachability",
        );
    }

    /// Verify supports() returns exactly two property types
    #[kani::proof]
    fn proof_venus_supports_count() {
        let backend = VenusBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 2,
            "Should support exactly two property types",
        );
    }

    // ---- Script Parsing Tests ----

    /// Verify parse_venus_output detects VERIFIED status
    #[kani::proof]
    fn proof_parse_venus_verified() {
        let (status, _) = script::parse_venus_output("VENUS_STATUS: VERIFIED", "");
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "VERIFIED should be Proven",
        );
    }

    /// Verify parse_venus_output detects PARTIALLY_VERIFIED status
    #[kani::proof]
    fn proof_parse_venus_partial() {
        let (status, _) = script::parse_venus_output("VENUS_STATUS: PARTIALLY_VERIFIED", "");
        kani::assert(
            matches!(status, VerificationStatus::Partial { .. }),
            "PARTIALLY_VERIFIED should be Partial",
        );
    }

    /// Verify parse_venus_output detects NOT_VERIFIED status
    #[kani::proof]
    fn proof_parse_venus_not_verified() {
        let (status, _) = script::parse_venus_output("VENUS_STATUS: NOT_VERIFIED", "");
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "NOT_VERIFIED should be Disproven",
        );
    }

    /// Verify parse_venus_output detects VENUS_ERROR
    #[kani::proof]
    fn proof_parse_venus_error() {
        let (status, _) = script::parse_venus_output("VENUS_ERROR: Something failed", "");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "VENUS_ERROR should be Unknown",
        );
    }

    /// Verify parse_venus_output returns Unknown for empty
    #[kani::proof]
    fn proof_parse_venus_empty() {
        let (status, _) = script::parse_venus_output("", "");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Empty output should be Unknown",
        );
    }
}

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{SolverBackend, VenusConfig};

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::process::Stdio;
use std::time::Instant;
use tempfile::TempDir;
use tokio::process::Command;
use tracing::debug;

/// Venus DNN verifier
pub struct VenusBackend {
    config: VenusConfig,
}

impl Default for VenusBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl VenusBackend {
    pub fn new() -> Self {
        Self {
            config: VenusConfig::default(),
        }
    }

    pub fn with_config(config: VenusConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for VenusBackend {
    fn id(&self) -> BackendId {
        BackendId::Venus
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::NeuralRobustness,
            PropertyType::NeuralReachability,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_venus(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_venus_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("venus_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated Venus script:\n{}", script_content);

        let mut cmd = Command::new(&python_path);
        cmd.arg(&script_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let result = tokio::time::timeout(self.config.timeout, cmd.output()).await;
        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Venus stdout: {}", stdout);
                debug!("Venus stderr: {}", stderr);

                let (status, counterexample) = script::parse_venus_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::Venus,
                    status,
                    proof: None,
                    counterexample,
                    diagnostics: if stderr.is_empty() {
                        vec![]
                    } else {
                        vec![stderr]
                    },
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_venus(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
