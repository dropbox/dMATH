//! MNBaB neural network verification backend
//!
//! MNBaB (Multi-Neuron Branch and Bound) is a neural network verifier using:
//! - Multi-neuron relaxation for tighter bounds
//! - Branch-and-bound search with various branching strategies
//! - GPU acceleration support
//!
//! See: <https://github.com/eth-sri/mn-bab>
//!
//! # Installation
//!
//! ```bash
//! git clone https://github.com/eth-sri/mn-bab
//! cd mn-bab && pip install -e .
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{BranchingStrategy, MNBaBConfig};

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

/// MNBaB neural network verifier
pub struct MNBaBBackend {
    config: MNBaBConfig,
}

impl Default for MNBaBBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MNBaBBackend {
    pub fn new() -> Self {
        Self {
            config: MNBaBConfig::default(),
        }
    }

    pub fn with_config(config: MNBaBConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for MNBaBBackend {
    fn id(&self) -> BackendId {
        BackendId::MNBaB
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::NeuralRobustness,
            PropertyType::NeuralReachability,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_mnbab(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_mnbab_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("mnbab_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated MNBaB script:\n{}", script_content);

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

                debug!("MNBaB stdout: {}", stdout);
                debug!("MNBaB stderr: {}", stderr);

                let (status, counterexample) = script::parse_mnbab_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::MNBaB,
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
        match detection::detect_mnbab(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
