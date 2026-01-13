//! Intel Neural Compressor backend
//!
//! Intel Neural Compressor provides model compression and quantization
//! capabilities for efficient inference:
//! - Post-training quantization (static and dynamic)
//! - Quantization-aware training
//! - Model pruning and knowledge distillation
//! - Mixed-precision optimization
//!
//! See: <https://github.com/intel/neural-compressor>
//!
//! # Installation
//!
//! ```bash
//! pip install neural-compressor
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{
    CalibrationMethod, NeuralCompressorConfig, QuantDataType, QuantizationApproach, TuningStrategy,
};

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

/// Intel Neural Compressor backend
pub struct NeuralCompressorBackend {
    config: NeuralCompressorConfig,
}

impl Default for NeuralCompressorBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralCompressorBackend {
    pub fn new() -> Self {
        Self {
            config: NeuralCompressorConfig::default(),
        }
    }

    pub fn with_config(config: NeuralCompressorConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for NeuralCompressorBackend {
    fn id(&self) -> BackendId {
        BackendId::NeuralCompressor
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::ModelCompression]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_neural_compressor(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_nc_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("nc_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated Neural Compressor script:\n{}", script_content);

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

                debug!("Neural Compressor stdout: {}", stdout);
                debug!("Neural Compressor stderr: {}", stderr);

                let (status, counterexample) = script::parse_nc_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::NeuralCompressor,
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
        match detection::detect_neural_compressor(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
