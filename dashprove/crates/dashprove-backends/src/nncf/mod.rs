//! NNCF (Neural Network Compression Framework) backend
//!
//! NNCF provides a suite of compression algorithms for neural networks:
//! - Quantization (INT8, INT4, mixed precision)
//! - Magnitude-based pruning and sparsity
//! - Filter pruning for structured sparsity
//! - Knowledge distillation
//!
//! See: <https://github.com/openvinotoolkit/nncf>
//!
//! # Installation
//!
//! ```bash
//! pip install nncf
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{BitWidth, CompressionMode, NNCFConfig, PruningSchedule, QuantizationMode};

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

/// NNCF compression backend
pub struct NNCFBackend {
    config: NNCFConfig,
}

impl Default for NNCFBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl NNCFBackend {
    pub fn new() -> Self {
        Self {
            config: NNCFConfig::default(),
        }
    }

    pub fn with_config(config: NNCFConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for NNCFBackend {
    fn id(&self) -> BackendId {
        BackendId::NNCF
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::ModelCompression]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_nncf(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_nncf_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("nncf_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated NNCF script:\n{}", script_content);

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

                debug!("NNCF stdout: {}", stdout);
                debug!("NNCF stderr: {}", stderr);

                let (status, counterexample) = script::parse_nncf_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::NNCF,
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
        match detection::detect_nncf(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
