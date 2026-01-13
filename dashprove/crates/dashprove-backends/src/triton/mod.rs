//! Triton GPU programming backend
//!
//! Triton is a language and compiler for writing highly efficient GPU code.
//! It provides Python-like syntax with automatic optimization for NVIDIA and AMD GPUs.
//!
//! Key features:
//! - Automatic memory coalescing and shared memory management
//! - Automatic kernel fusion and optimization
//! - Support for matrix operations and custom kernels
//!
//! See: <https://triton-lang.org/>
//!
//! # Installation
//!
//! ```bash
//! pip install triton
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{CompilationMode, OptimizationLevel, TritonConfig, TritonTarget};

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

/// Triton optimization backend
pub struct TritonBackend {
    config: TritonConfig,
}

impl Default for TritonBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TritonBackend {
    pub fn new() -> Self {
        Self {
            config: TritonConfig::default(),
        }
    }

    pub fn with_config(config: TritonConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for TritonBackend {
    fn id(&self) -> BackendId {
        BackendId::Triton
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::ModelOptimization]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_triton(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_triton_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("triton_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated Triton script:\n{}", script_content);

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

                debug!("Triton stdout: {}", stdout);
                debug!("Triton stderr: {}", stderr);

                let (status, counterexample) = script::parse_triton_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::Triton,
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
        match detection::detect_triton(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
