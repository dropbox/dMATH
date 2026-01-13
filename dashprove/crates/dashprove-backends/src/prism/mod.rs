//! PRISM backend for probabilistic model checking
//!
//! PRISM is a probabilistic symbolic model checker supporting DTMCs, CTMCs, MDPs,
//! PTAs, and POMDPs. It uses symbolic data structures (BDDs/MTBDDs) for efficient
//! state space representation and can verify PCTL, CSL, LTL, and reward properties.
//!
//! See: <https://www.prismmodelchecker.org/>
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
//! - `until(P, Q)` → `P U Q`
//! - `next(P)` → `X P`

mod config;
mod detection;
mod model;
mod parsing;
mod pctl;
mod util;

#[cfg(test)]
mod tests;

pub use config::{PrismConfig, PrismEngine};

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

/// PRISM probabilistic model checking backend
pub struct PrismBackend {
    config: PrismConfig,
}

impl Default for PrismBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl PrismBackend {
    /// Create a new PRISM backend with default configuration
    pub fn new() -> Self {
        Self {
            config: PrismConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: PrismConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for PrismBackend {
    fn id(&self) -> BackendId {
        BackendId::Prism
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Probabilistic]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let prism_path = detection::detect_prism(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Generate model file
        let model_code = model::generate_prism_model(spec);
        let model_path = temp_dir.path().join("model.pm");
        std::fs::write(&model_path, &model_code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write model: {}", e))
        })?;

        // Generate PCTL property from spec
        let pctl_property = pctl::generate_pctl_property(spec);

        let mut cmd = Command::new(&prism_path);
        cmd.arg(&model_path)
            .arg("-pctl")
            .arg(&pctl_property)
            .arg(self.config.engine.as_arg())
            .arg("-epsilon")
            .arg(self.config.precision.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(max_iters) = self.config.max_iters {
            cmd.arg("-maxiters").arg(max_iters.to_string());
        }

        let result = tokio::time::timeout(self.config.timeout, cmd.output()).await;
        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("PRISM stdout: {}", stdout);
                let status = parsing::parse_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::Prism,
                    status,
                    proof: None,
                    counterexample: None,
                    diagnostics: vec![],
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute PRISM: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_prism(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
