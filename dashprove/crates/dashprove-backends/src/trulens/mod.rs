//! TruLens LLM evaluation backend
//!
//! Provides comprehensive LLM evaluation using TruLens framework.
//! Supports feedback functions for relevance, groundedness, and coherence.
//!
//! Installation:
//! ```bash
//! pip install trulens-eval
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{FeedbackProvider, FeedbackType, TruLensConfig};

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

/// TruLens backend implementation
pub struct TruLensBackend {
    config: TruLensConfig,
}

impl Default for TruLensBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TruLensBackend {
    pub fn new() -> Self {
        Self {
            config: TruLensConfig::default(),
        }
    }

    pub fn with_config(config: TruLensConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for TruLensBackend {
    fn id(&self) -> BackendId {
        BackendId::TruLens
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::LLMEvaluation]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_trulens(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_trulens_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("trulens_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated TruLens script:\n{}", script_content);

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

                debug!("TruLens stdout: {}", stdout);
                debug!("TruLens stderr: {}", stderr);

                let (status, counterexample) = script::parse_trulens_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::TruLens,
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
        match detection::detect_trulens(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
