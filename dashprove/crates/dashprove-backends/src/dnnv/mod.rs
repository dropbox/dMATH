//! DNNV (Deep Neural Network Verification) framework backend
//!
//! DNNV is a unified framework for DNN verification that:
//! - Provides a common interface to multiple verifiers
//! - Uses DNNP property specification language
//! - Supports ONNX model format
//! - Integrates Planet, Marabou, ERAN, and more
//!
//! See: <https://github.com/dlshriver/dnnv>
//!
//! # Installation
//!
//! ```bash
//! pip install dnnv
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{DnnvConfig, VerifierBackend};

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

/// DNNV verification framework backend
pub struct DnnvBackend {
    config: DnnvConfig,
}

impl Default for DnnvBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl DnnvBackend {
    pub fn new() -> Self {
        Self {
            config: DnnvConfig::default(),
        }
    }

    pub fn with_config(config: DnnvConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for DnnvBackend {
    fn id(&self) -> BackendId {
        BackendId::DNNV
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::NeuralRobustness,
            PropertyType::NeuralReachability,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_dnnv(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_dnnv_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("dnnv_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated DNNV script:\n{}", script_content);

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

                debug!("DNNV stdout: {}", stdout);
                debug!("DNNV stderr: {}", stderr);

                let (status, counterexample) = script::parse_dnnv_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::DNNV,
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
        match detection::detect_dnnv(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
