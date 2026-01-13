//! Ragas backend for RAG evaluation
//!
//! This backend integrates with Ragas to provide:
//! - Faithfulness evaluation
//! - Answer relevancy metrics
//! - Context precision/recall
//! - Full pipeline evaluation

mod config;
mod detection;
pub mod script;
#[cfg(test)]
mod tests;

pub use config::{EvaluationMode, RagasConfig, RagasMetric};

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::process::Stdio;
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// Ragas verification backend
pub struct RagasBackend {
    config: RagasConfig,
}

impl RagasBackend {
    /// Create a new Ragas backend with default configuration
    pub fn new() -> Self {
        Self {
            config: RagasConfig::default(),
        }
    }

    /// Create a new Ragas backend with custom configuration
    pub fn with_config(config: RagasConfig) -> Self {
        Self { config }
    }
}

impl Default for RagasBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for RagasBackend {
    fn id(&self) -> BackendId {
        BackendId::Ragas
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::LLMEvaluation]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Check if Ragas is available
        let python_path = match &self.config.python_path {
            Some(p) => p.clone(),
            None => detection::detect_ragas().ok_or_else(|| {
                BackendError::Unavailable(
                    "Ragas not found. Install with: pip install ragas".to_string(),
                )
            })?,
        };

        // Generate verification script
        let script = script::generate_ragas_script(spec, &self.config)?;

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

        let (status, counterexample) = script::parse_ragas_output(&stdout, &stderr);

        Ok(BackendResult {
            backend: BackendId::Ragas,
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
        match detection::detect_ragas() {
            Some(_) => {
                if detection::get_ragas_version().is_some() {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Degraded {
                        reason: "Ragas found but version check failed".to_string(),
                    }
                }
            }
            None => HealthStatus::Unavailable {
                reason: "Ragas not installed. Install with: pip install ragas".to_string(),
            },
        }
    }
}
