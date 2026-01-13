//! AIF360 backend
//!
//! IBM AI Fairness 360 provides comprehensive bias detection:
//! - Fairness metrics (70+ metrics)
//! - Bias mitigation algorithms
//! - Pre/in/post-processing techniques
//!
//! See: <https://aif360.mybluemix.net/>
//!
//! # Installation
//!
//! ```bash
//! pip install aif360
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{AIF360Config, AIF360MitigationAlgorithm, BiasMetric};

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

/// AIF360 bias assessment backend
pub struct AIF360Backend {
    config: AIF360Config,
}

impl Default for AIF360Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl AIF360Backend {
    pub fn new() -> Self {
        Self {
            config: AIF360Config::default(),
        }
    }

    pub fn with_config(config: AIF360Config) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for AIF360Backend {
    fn id(&self) -> BackendId {
        BackendId::AIF360
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Fairness]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_aif360(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_aif360_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("aif360_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated AIF360 script:\n{}", script_content);

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

                debug!("AIF360 stdout: {}", stdout);
                debug!("AIF360 stderr: {}", stderr);

                let (status, counterexample) = script::parse_aif360_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::AIF360,
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
        match detection::detect_aif360(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
