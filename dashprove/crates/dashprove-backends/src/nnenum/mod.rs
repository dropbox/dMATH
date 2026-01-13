//! Nnenum (neural network enumeration) backend
//!
//! Nnenum is an enumeration-based neural network verifier that:
//! - Exactly enumerates all linear regions of ReLU networks
//! - Provides complete verification guarantees
//! - Supports parallel verification
//!
//! See: <https://github.com/stanleybak/nnenum>
//!
//! # USL to Nnenum Compilation
//!
//! This backend compiles USL neural properties to nnenum Python scripts:
//!
//! - Robustness properties: `|x - x0| <= epsilon` → L-inf input bounds
//! - Model references: `.onnx` paths → network loading
//! - Output constraints for classification preservation
//!
//! # Installation
//!
//! ```bash
//! pip install nnenum
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{EnumerationStrategy, NnenumConfig};

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

/// Nnenum enumeration-based neural network verifier
pub struct NnenumBackend {
    config: NnenumConfig,
}

impl Default for NnenumBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl NnenumBackend {
    /// Create a new nnenum backend with default configuration
    pub fn new() -> Self {
        Self {
            config: NnenumConfig::default(),
        }
    }

    /// Create a new backend with custom configuration
    pub fn with_config(config: NnenumConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for NnenumBackend {
    fn id(&self) -> BackendId {
        BackendId::Nnenum
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::NeuralRobustness,
            PropertyType::NeuralReachability,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_nnenum(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_nnenum_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("nnenum_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write nnenum script: {}", e))
        })?;

        debug!("Generated nnenum script:\n{}", script_content);

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

                debug!("nnenum stdout: {}", stdout);
                debug!("nnenum stderr: {}", stderr);

                let (status, counterexample) = script::parse_nnenum_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::Nnenum,
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
                "Failed to execute nnenum script: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_nnenum(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
