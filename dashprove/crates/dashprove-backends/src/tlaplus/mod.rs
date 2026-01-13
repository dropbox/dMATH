//! TLA+ backend implementation
//!
//! This backend executes TLA+ specifications using the TLC model checker.
//! TLC can be invoked either via the `tlc` command or via Java with the tla2tools.jar.

mod config;
mod detection;
mod execution;
mod parsing;
mod spec;
mod trace;
mod util;
mod values;

#[cfg(test)]
mod tests;

pub use config::TlaPlusConfig;
pub use detection::TlcDetection;

use crate::traits::*;
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use tempfile::TempDir;
use tracing::{info, warn};

/// TLA+ verification backend using TLC model checker
pub struct TlaPlusBackend {
    config: TlaPlusConfig,
}

impl TlaPlusBackend {
    /// Create a new TLA+ backend with default configuration
    pub fn new() -> Self {
        Self {
            config: TlaPlusConfig::default(),
        }
    }

    /// Create a new TLA+ backend with custom configuration
    pub fn with_config(config: TlaPlusConfig) -> Self {
        Self { config }
    }
}

impl Default for TlaPlusBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for TlaPlusBackend {
    fn id(&self) -> BackendId {
        BackendId::TlaPlus
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Temporal, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        // Create a mutable copy for detection caching
        let detection = detection::detect_tlc(&self.config).await;

        if let TlcDetection::NotFound(reason) = &detection {
            return Err(BackendError::Unavailable(reason.clone()));
        }

        // Create temp directory for spec files
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::CompilationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Write spec files
        let (tla_path, cfg_path) = spec::write_spec(spec, temp_dir.path()).await?;

        // Run TLC
        let output = execution::run_tlc(&self.config, &detection, &tla_path, &cfg_path).await?;

        // Parse results
        let result = parsing::parse_output(&output);

        Ok(result)
    }

    async fn health_check(&self) -> HealthStatus {
        let detection = detection::detect_tlc(&self.config).await;
        match detection {
            TlcDetection::Standalone(path) => {
                info!("TLC available at {:?}", path);
                HealthStatus::Healthy
            }
            TlcDetection::Jar {
                java_path,
                jar_path,
            } => {
                info!(
                    "TLC available via JAR at {:?} (Java: {:?})",
                    jar_path, java_path
                );
                HealthStatus::Healthy
            }
            TlcDetection::NotFound(reason) => {
                warn!("TLC not available: {}", reason);
                HealthStatus::Unavailable { reason }
            }
        }
    }
}
