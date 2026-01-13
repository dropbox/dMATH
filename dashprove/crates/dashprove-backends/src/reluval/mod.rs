//! ReluVal neural network verification backend
//!
//! ReluVal is a neural network verifier using:
//! - Interval arithmetic optimized for ReLU networks
//! - Symbolic propagation with efficient refinement
//! - Fast C implementation
//!
//! See: <https://github.com/tcwangshiqi-columbia/ReluVal>
//!
//! # Installation
//!
//! ```bash
//! git clone https://github.com/tcwangshiqi-columbia/ReluVal
//! cd ReluVal && make
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{RefinementMode, ReluValConfig};

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

/// ReluVal neural network verifier
pub struct ReluValBackend {
    config: ReluValConfig,
}

impl Default for ReluValBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ReluValBackend {
    pub fn new() -> Self {
        Self {
            config: ReluValConfig::default(),
        }
    }

    pub fn with_config(config: ReluValConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for ReluValBackend {
    fn id(&self) -> BackendId {
        BackendId::ReluVal
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::NeuralRobustness,
            PropertyType::NeuralReachability,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_reluval(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_reluval_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("reluval_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated ReluVal script:\n{}", script_content);

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

                debug!("ReluVal stdout: {}", stdout);
                debug!("ReluVal stderr: {}", stderr);

                let (status, counterexample) = script::parse_reluval_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::ReluVal,
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
        match detection::detect_reluval(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
