//! Auto-LiRPA neural network bound propagation backend
//!
//! Auto-LiRPA computes certified output bounds using:
//! - Interval Bound Propagation (IBP)
//! - Linear relaxation (CROWN)
//! - Optimized alpha-CROWN
//!
//! See: <https://github.com/Verified-Intelligence/auto_LiRPA>
//!
//! # Installation
//!
//! ```bash
//! pip install auto_lirpa
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{AutoLirpaConfig, BoundMethod};

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

/// Auto-LiRPA bound propagation backend
pub struct AutoLirpaBackend {
    config: AutoLirpaConfig,
}

impl Default for AutoLirpaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoLirpaBackend {
    pub fn new() -> Self {
        Self {
            config: AutoLirpaConfig::default(),
        }
    }

    pub fn with_config(config: AutoLirpaConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for AutoLirpaBackend {
    fn id(&self) -> BackendId {
        BackendId::AutoLiRPA
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::NeuralRobustness,
            PropertyType::NeuralReachability,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_autolirpa(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_autolirpa_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("autolirpa_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated Auto-LiRPA script:\n{}", script_content);

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

                debug!("Auto-LiRPA stdout: {}", stdout);
                debug!("Auto-LiRPA stderr: {}", stderr);

                let (status, counterexample) = script::parse_autolirpa_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::AutoLiRPA,
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
        match detection::detect_autolirpa(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
