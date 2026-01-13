//! ONNX Runtime model optimization backend
//!
//! ONNX Runtime provides cross-platform, high-performance ML inference with:
//! - Multiple execution providers (CPU, CUDA, TensorRT, OpenVINO, etc.)
//! - Graph optimizations (constant folding, node fusion, etc.)
//! - Memory optimization patterns
//!
//! See: <https://onnxruntime.ai/>
//!
//! # Installation
//!
//! ```bash
//! pip install onnxruntime          # CPU only
//! pip install onnxruntime-gpu      # With CUDA support
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{ExecutionProvider, GraphOptimizationLevel, OnnxRuntimeConfig};

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

/// ONNX Runtime optimization backend
pub struct OnnxRuntimeBackend {
    config: OnnxRuntimeConfig,
}

impl Default for OnnxRuntimeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl OnnxRuntimeBackend {
    pub fn new() -> Self {
        Self {
            config: OnnxRuntimeConfig::default(),
        }
    }

    pub fn with_config(config: OnnxRuntimeConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for OnnxRuntimeBackend {
    fn id(&self) -> BackendId {
        BackendId::ONNXRuntime
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::ModelOptimization]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_onnxruntime(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_onnxruntime_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("onnxruntime_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated ONNX Runtime script:\n{}", script_content);

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

                debug!("ONNX Runtime stdout: {}", stdout);
                debug!("ONNX Runtime stderr: {}", stderr);

                let (status, counterexample) = script::parse_onnxruntime_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::ONNXRuntime,
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
        match detection::detect_onnxruntime(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
