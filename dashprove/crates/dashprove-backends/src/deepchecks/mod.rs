//! Deepchecks backend
//!
//! Deepchecks provides ML validation suites:
//! - Data integrity validation
//! - Train-test validation for data drift
//! - Model evaluation with automated checks
//!
//! See: <https://deepchecks.com/>
//!
//! # Installation
//!
//! ```bash
//! pip install deepchecks
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{DeepchecksConfig, SeverityThreshold, SuiteType, TaskType};

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

/// Deepchecks ML validation backend
pub struct DeepchecksBackend {
    config: DeepchecksConfig,
}

impl Default for DeepchecksBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl DeepchecksBackend {
    pub fn new() -> Self {
        Self {
            config: DeepchecksConfig::default(),
        }
    }

    pub fn with_config(config: DeepchecksConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for DeepchecksBackend {
    fn id(&self) -> BackendId {
        BackendId::Deepchecks
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::DataQuality]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_deepchecks(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_deepchecks_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("deepchecks_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated Deepchecks script:\n{}", script_content);

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

                debug!("Deepchecks stdout: {}", stdout);
                debug!("Deepchecks stderr: {}", stderr);

                let (status, counterexample) = script::parse_deepchecks_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::Deepchecks,
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
        match detection::detect_deepchecks(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
