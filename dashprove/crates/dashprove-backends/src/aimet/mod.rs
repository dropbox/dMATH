//! AIMET (AI Model Efficiency Toolkit) backend
//!
//! AIMET is Qualcomm's model efficiency toolkit providing:
//! - Post-training quantization
//! - Quantization-aware training
//! - Cross-layer equalization
//! - AdaRound (adaptive rounding)
//! - Spatial SVD and channel pruning
//!
//! See: <https://github.com/quic/aimet>
//!
//! # Installation
//!
//! AIMET requires installation from the Qualcomm repository.
//! See: <https://github.com/quic/aimet#installation>

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{AimetBitWidth, AimetCompressionMode, AimetConfig, QuantScheme, RoundingMode};

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

/// AIMET compression backend
pub struct AimetBackend {
    config: AimetConfig,
}

impl Default for AimetBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AimetBackend {
    pub fn new() -> Self {
        Self {
            config: AimetConfig::default(),
        }
    }

    pub fn with_config(config: AimetConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for AimetBackend {
    fn id(&self) -> BackendId {
        BackendId::AIMET
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::ModelCompression]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_aimet(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_aimet_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("aimet_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated AIMET script:\n{}", script_content);

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

                debug!("AIMET stdout: {}", stdout);
                debug!("AIMET stderr: {}", stderr);

                let (status, counterexample) = script::parse_aimet_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::AIMET,
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
        match detection::detect_aimet(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
