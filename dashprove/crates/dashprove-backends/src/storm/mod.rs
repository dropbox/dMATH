//! Storm backend for probabilistic model checking
//!
//! Storm is a modern probabilistic model checker supporting DTMCs, CTMCs, MDPs,
//! and other stochastic models. It can verify PCTL, CSL, and reward properties.
//!
//! See: <https://www.stormchecker.org/>
//!
//! # USL to PRISM Compilation
//!
//! This backend compiles USL probabilistic properties to PRISM models:
//!
//! - `probability(condition) >= bound` → PCTL property `P>=bound[F condition]`
//! - Types with numeric ranges become state variables
//! - Probabilistic transitions extracted from property patterns
//! - `eventually(P)` → `F P` (finally)
//! - `always(P)` → `G P` (globally)
//! - `response_time < bound` → bounded reachability

mod config;
mod detection;
mod parsing;
mod pctl;
mod prism;
mod util;

#[cfg(test)]
mod tests;

pub use config::StormConfig;

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

/// Storm probabilistic model checking backend
pub struct StormBackend {
    config: StormConfig,
}

impl Default for StormBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl StormBackend {
    /// Create a new Storm backend with default configuration
    pub fn new() -> Self {
        Self {
            config: StormConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: StormConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for StormBackend {
    fn id(&self) -> BackendId {
        BackendId::Storm
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Probabilistic]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let storm_path = detection::detect_storm(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let prism_code = prism::generate_prism(spec);
        let model_path = temp_dir.path().join("model.pm");
        std::fs::write(&model_path, &prism_code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write model: {}", e))
        })?;

        // Generate PCTL property from spec
        let pctl_property = pctl::generate_pctl_property(spec);

        let mut cmd = Command::new(&storm_path);
        cmd.arg("--prism")
            .arg(&model_path)
            .arg("--prop")
            .arg(&pctl_property)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let result = tokio::time::timeout(self.config.timeout, cmd.output()).await;
        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Storm stdout: {}", stdout);
                let status = parsing::parse_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::Storm,
                    status,
                    proof: None,
                    counterexample: None,
                    diagnostics: vec![],
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute Storm: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_storm(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
