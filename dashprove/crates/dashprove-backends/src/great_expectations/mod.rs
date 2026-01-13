//! Great Expectations backend
//!
//! Great Expectations provides data validation and documentation:
//! - Expectation suites for data quality rules
//! - Data profiling and schema inference
//! - Validation result documentation
//!
//! See: <https://greatexpectations.io/>
//!
//! # Installation
//!
//! ```bash
//! pip install great_expectations
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{DataSourceType, GreatExpectationsConfig, ResultFormat, ValidationLevel};

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

/// Great Expectations data validation backend
pub struct GreatExpectationsBackend {
    config: GreatExpectationsConfig,
}

impl Default for GreatExpectationsBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GreatExpectationsBackend {
    pub fn new() -> Self {
        Self {
            config: GreatExpectationsConfig::default(),
        }
    }

    pub fn with_config(config: GreatExpectationsConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for GreatExpectationsBackend {
    fn id(&self) -> BackendId {
        BackendId::GreatExpectations
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::DataQuality]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_great_expectations(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_great_expectations_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("gx_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated Great Expectations script:\n{}", script_content);

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

                debug!("Great Expectations stdout: {}", stdout);
                debug!("Great Expectations stderr: {}", stderr);

                let (status, counterexample) =
                    script::parse_great_expectations_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::GreatExpectations,
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
        match detection::detect_great_expectations(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
