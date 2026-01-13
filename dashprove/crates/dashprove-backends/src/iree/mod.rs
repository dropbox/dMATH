//! IREE ML compiler backend
//!
//! IREE (Intermediate Representation Execution Environment) is a compiler
//! and runtime for ML models with support for multiple backends:
//! - LLVM CPU (reference implementation)
//! - Vulkan SPIR-V (cross-platform GPU)
//! - CUDA (NVIDIA GPU)
//! - Metal (Apple GPU)
//! - WebGPU (browser deployment)
//!
//! See: <https://iree.dev/>
//!
//! # Installation
//!
//! ```bash
//! pip install iree-compiler iree-runtime
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{ExecutionMode, IREEConfig, IREETarget, InputFormat};

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

/// IREE optimization backend
pub struct IREEBackend {
    config: IREEConfig,
}

impl Default for IREEBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl IREEBackend {
    pub fn new() -> Self {
        Self {
            config: IREEConfig::default(),
        }
    }

    pub fn with_config(config: IREEConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for IREEBackend {
    fn id(&self) -> BackendId {
        BackendId::IREE
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::ModelOptimization]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_iree(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_iree_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("iree_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated IREE script:\n{}", script_content);

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

                debug!("IREE stdout: {}", stdout);
                debug!("IREE stderr: {}", stderr);

                let (status, counterexample) = script::parse_iree_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::IREE,
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
        match detection::detect_iree(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
