//! FactScore backend for factual accuracy evaluation
//!
//! This backend integrates with FactScore to provide:
//! - Factual precision scoring
//! - Sentence-level fact extraction
//! - Claim-level verification
//! - Entity and triple verification

mod config;
mod detection;
pub mod script;
#[cfg(test)]
mod tests;

pub use config::{ExtractionMethod, FactScoreConfig, KnowledgeSource};

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::process::Stdio;
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// FactScore verification backend
pub struct FactScoreBackend {
    config: FactScoreConfig,
}

impl FactScoreBackend {
    /// Create a new FactScore backend with default configuration
    pub fn new() -> Self {
        Self {
            config: FactScoreConfig::default(),
        }
    }

    /// Create a new FactScore backend with custom configuration
    pub fn with_config(config: FactScoreConfig) -> Self {
        Self { config }
    }
}

impl Default for FactScoreBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for FactScoreBackend {
    fn id(&self) -> BackendId {
        BackendId::FactScore
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::HallucinationDetection]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Check if FactScore is available
        let python_path = match &self.config.python_path {
            Some(p) => p.clone(),
            None => detection::detect_factscore().ok_or_else(|| {
                BackendError::Unavailable(
                    "FactScore not found. Install with: pip install factscore".to_string(),
                )
            })?,
        };

        // Generate verification script
        let script = script::generate_factscore_script(spec, &self.config)?;

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

        let (status, counterexample) = script::parse_factscore_output(&stdout, &stderr);

        Ok(BackendResult {
            backend: BackendId::FactScore,
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
        match detection::detect_factscore() {
            Some(_) => {
                if detection::get_factscore_version().is_some() {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Degraded {
                        reason: "FactScore found but version check failed".to_string(),
                    }
                }
            }
            None => HealthStatus::Unavailable {
                reason: "FactScore not installed. Install with: pip install factscore".to_string(),
            },
        }
    }
}
