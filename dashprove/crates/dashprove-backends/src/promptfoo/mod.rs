//! Promptfoo prompt evaluation backend
//!
//! Provides LLM prompt testing and evaluation using the promptfoo framework.
//! Supports various assertion types including contains, equals, regex, and JSON.
//!
//! Installation:
//! ```bash
//! npm install -g promptfoo
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{AssertionType, OutputFormat, PromptfooConfig};

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

/// Promptfoo backend implementation
pub struct PromptfooBackend {
    config: PromptfooConfig,
}

impl Default for PromptfooBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptfooBackend {
    pub fn new() -> Self {
        Self {
            config: PromptfooConfig::default(),
        }
    }

    pub fn with_config(config: PromptfooConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for PromptfooBackend {
    fn id(&self) -> BackendId {
        BackendId::Promptfoo
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::LLMEvaluation]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let node_path = detection::detect_promptfoo(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_promptfoo_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("promptfoo_verify.js");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated Promptfoo script:\n{}", script_content);

        let mut cmd = Command::new(&node_path);
        cmd.arg(&script_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let result = tokio::time::timeout(self.config.timeout, cmd.output()).await;
        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Promptfoo stdout: {}", stdout);
                debug!("Promptfoo stderr: {}", stderr);

                let (status, counterexample) = script::parse_promptfoo_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::Promptfoo,
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
        match detection::detect_promptfoo(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
