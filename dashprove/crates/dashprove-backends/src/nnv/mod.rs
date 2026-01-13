//! NNV (Neural Network Verification) backend
//!
//! NNV is a comprehensive neural network verification library that supports:
//! - Star set reachability analysis
//! - Zonotope abstraction
//! - Interval arithmetic
//! - Multiple neural network architectures (feedforward, CNN, RNN)
//!
//! See: <https://github.com/stanleybak/nnenum>
//!
//! # USL to NNV Compilation
//!
//! This backend compiles USL neural properties to NNV Python scripts:
//!
//! - Robustness properties: `|x - x0| <= epsilon` → input set bounds
//! - Model references: `.onnx` paths → network loading
//! - Verification method selection based on property characteristics
//!
//! # Installation
//!
//! ```bash
//! pip install nnv
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{NnvConfig, VerificationMethod};

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

/// NNV neural network verification backend
pub struct NnvBackend {
    config: NnvConfig,
}

impl Default for NnvBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl NnvBackend {
    /// Create a new NNV backend with default configuration
    pub fn new() -> Self {
        Self {
            config: NnvConfig::default(),
        }
    }

    /// Create a new backend with custom configuration
    pub fn with_config(config: NnvConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for NnvBackend {
    fn id(&self) -> BackendId {
        BackendId::NNV
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::NeuralRobustness,
            PropertyType::NeuralReachability,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Detect NNV installation
        let python_path = detection::detect_nnv(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        // Create temp directory for the script
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Generate the NNV verification script
        let script_content = script::generate_nnv_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("nnv_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write NNV script: {}", e))
        })?;

        debug!("Generated NNV script:\n{}", script_content);

        // Run the NNV verification script
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

                debug!("NNV stdout: {}", stdout);
                debug!("NNV stderr: {}", stderr);

                let (status, counterexample) = script::parse_nnv_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::NNV,
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
                "Failed to execute NNV script: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_nnv(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
