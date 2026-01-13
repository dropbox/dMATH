//! DeepEval backend for LLM testing
//!
//! This backend integrates with DeepEval to provide:
//! - Answer relevancy evaluation
//! - Faithfulness testing
//! - Hallucination detection
//! - Toxicity and bias detection

mod config;
mod detection;
pub mod script;
#[cfg(test)]
mod tests;

pub use config::{DeepEvalConfig, DeepEvalMetric, TestCaseType};

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::process::Stdio;
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// DeepEval verification backend
pub struct DeepEvalBackend {
    config: DeepEvalConfig,
}

impl DeepEvalBackend {
    /// Create a new DeepEval backend with default configuration
    pub fn new() -> Self {
        Self {
            config: DeepEvalConfig::default(),
        }
    }

    /// Create a new DeepEval backend with custom configuration
    pub fn with_config(config: DeepEvalConfig) -> Self {
        Self { config }
    }
}

impl Default for DeepEvalBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for DeepEvalBackend {
    fn id(&self) -> BackendId {
        BackendId::DeepEval
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::LLMEvaluation,
            PropertyType::HallucinationDetection,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Check if DeepEval is available
        let python_path = match &self.config.python_path {
            Some(p) => p.clone(),
            None => detection::detect_deepeval().ok_or_else(|| {
                BackendError::Unavailable(
                    "DeepEval not found. Install with: pip install deepeval".to_string(),
                )
            })?,
        };

        // Generate verification script
        let script = script::generate_deepeval_script(spec, &self.config)?;

        // Run the script
        let mut child = Command::new(&python_path)
            .arg("-c")
            .arg(&script)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to spawn Python: {}", e))
            })?;

        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.shutdown().await;
        }

        let output = tokio::time::timeout(self.config.timeout, child.wait_with_output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| BackendError::VerificationFailed(format!("Process error: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let (status, counterexample) = script::parse_deepeval_output(&stdout, &stderr);

        Ok(BackendResult {
            backend: BackendId::DeepEval,
            status,
            proof: None,
            counterexample,
            diagnostics: if stderr.is_empty() {
                vec![]
            } else {
                vec![stderr.to_string()]
            },
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_deepeval() {
            Some(_) => {
                if detection::get_deepeval_version().is_some() {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Degraded {
                        reason: "DeepEval found but version check failed".to_string(),
                    }
                }
            }
            None => HealthStatus::Unavailable {
                reason: "DeepEval not installed. Install with: pip install deepeval".to_string(),
            },
        }
    }
}
