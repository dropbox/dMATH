//! Brevitas quantization-aware training backend
//!
//! Brevitas is a PyTorch library for neural network quantization with
//! a focus on quantization-aware training (QAT):
//! - Flexible bit-width for weights and activations (1-8 bits)
//! - Support for binary and ternary neural networks
//! - Per-tensor, per-channel, and per-group quantization
//! - Export to FINN for FPGA deployment
//! - Export to ONNX/QONNX
//!
//! See: <https://github.com/Xilinx/brevitas>
//!
//! # Installation
//!
//! ```bash
//! pip install brevitas
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{
    ActivationBitWidth, BrevitasConfig, ExportFormat, QuantMethod, ScalingMode, WeightBitWidth,
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

/// Brevitas quantization backend
pub struct BrevitasBackend {
    config: BrevitasConfig,
}

impl Default for BrevitasBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl BrevitasBackend {
    pub fn new() -> Self {
        Self {
            config: BrevitasConfig::default(),
        }
    }

    pub fn with_config(config: BrevitasConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for BrevitasBackend {
    fn id(&self) -> BackendId {
        BackendId::Brevitas
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::ModelCompression]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_brevitas(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_brevitas_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("brevitas_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated Brevitas script:\n{}", script_content);

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

                debug!("Brevitas stdout: {}", stdout);
                debug!("Brevitas stderr: {}", stderr);

                let (status, counterexample) = script::parse_brevitas_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::Brevitas,
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
        match detection::detect_brevitas(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
