//! Neurify neural network verification backend
//!
//! Neurify is a neural network verifier using:
//! - Symbolic interval propagation
//! - Gradient-based splitting for refinement
//! - Efficient C implementation
//!
//! See: <https://github.com/tcwangshiqi-columbia/Neurify>
//!
//! # Installation
//!
//! ```bash
//! git clone https://github.com/tcwangshiqi-columbia/Neurify
//! cd Neurify && make
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{NeurifyConfig, SplitMethod};

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

/// Neurify neural network verifier
pub struct NeurifyBackend {
    config: NeurifyConfig,
}

impl Default for NeurifyBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl NeurifyBackend {
    pub fn new() -> Self {
        Self {
            config: NeurifyConfig::default(),
        }
    }

    pub fn with_config(config: NeurifyConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for NeurifyBackend {
    fn id(&self) -> BackendId {
        BackendId::Neurify
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::NeuralRobustness,
            PropertyType::NeuralReachability,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_neurify(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_neurify_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("neurify_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated Neurify script:\n{}", script_content);

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

                debug!("Neurify stdout: {}", stdout);
                debug!("Neurify stderr: {}", stderr);

                let (status, counterexample) = script::parse_neurify_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::Neurify,
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
        match detection::detect_neurify(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
