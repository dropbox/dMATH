//! Foolbox backend for adversarial robustness evaluation
//!
//! Foolbox is a Python library for creating adversarial examples that fool
//! neural networks. It supports many attack methods and provides a clean API
//! for robustness evaluation.
//!
//! See: <https://github.com/bethgelab/foolbox>
//!
//! # Installation
//!
//! ```bash
//! pip install foolbox
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::FoolboxConfig;

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

/// Foolbox adversarial evaluation backend
pub struct FoolboxBackend {
    config: FoolboxConfig,
}

impl Default for FoolboxBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl FoolboxBackend {
    /// Create a new Foolbox backend with default configuration
    pub fn new() -> Self {
        Self {
            config: FoolboxConfig::default(),
        }
    }

    /// Create a new backend with custom configuration
    pub fn with_config(config: FoolboxConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for FoolboxBackend {
    fn id(&self) -> BackendId {
        BackendId::Foolbox
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::AdversarialRobustness]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Detect Foolbox installation
        let python_path = detection::detect_foolbox(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        // Create temp directory for the script
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Generate the Foolbox evaluation script
        let script_content = script::generate_foolbox_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("foolbox_eval.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write Foolbox script: {}", e))
        })?;

        debug!("Generated Foolbox script:\n{}", script_content);

        // Run the Foolbox evaluation script
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

                debug!("Foolbox stdout: {}", stdout);
                debug!("Foolbox stderr: {}", stderr);

                let (status, counterexample) = script::parse_foolbox_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::Foolbox,
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
                "Failed to execute Foolbox script: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_foolbox(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
