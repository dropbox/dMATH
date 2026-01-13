//! Apalache backend implementation
//!
//! This backend executes TLA+ specifications using the Apalache symbolic model checker.
//! Unlike TLC which uses explicit-state model checking, Apalache uses symbolic
//! model checking via SMT solvers (Z3), enabling unbounded verification.
//!
//! Key differences from TLC:
//! - Symbolic: Uses SMT solver instead of explicit state enumeration
//! - Type-safe: Requires type annotations for all variables
//! - Unbounded: Can verify properties for arbitrary parameter sizes
//! - Slower: More expensive per state but explores infinite state spaces

mod config;
mod detection;
mod execution;
mod parsing;
mod spec;

#[cfg(test)]
mod tests;

pub use config::{ApalacheConfig, ApalacheMode};
pub use detection::ApalacheDetection;

use crate::traits::*;
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use tempfile::TempDir;
use tracing::{info, warn};

/// Apalache verification backend using symbolic model checking
pub struct ApalacheBackend {
    config: ApalacheConfig,
}

impl ApalacheBackend {
    /// Create a new Apalache backend with default configuration
    pub fn new() -> Self {
        Self {
            config: ApalacheConfig::default(),
        }
    }

    /// Create a new Apalache backend with custom configuration
    pub fn with_config(config: ApalacheConfig) -> Self {
        Self { config }
    }
}

impl Default for ApalacheBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for ApalacheBackend {
    fn id(&self) -> BackendId {
        BackendId::Apalache
    }

    fn supports(&self) -> Vec<PropertyType> {
        // Apalache supports the same property types as TLC
        // but is better suited for unbounded verification
        vec![PropertyType::Temporal, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        // Detect Apalache installation
        let detection = detection::detect_apalache(&self.config).await;

        if let ApalacheDetection::NotFound(reason) = &detection {
            return Err(BackendError::Unavailable(reason.clone()));
        }

        // Create temp directory for spec files
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::CompilationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Write spec files with type annotations
        let (tla_path, cfg_path) = spec::write_spec(spec, temp_dir.path()).await?;

        // Run Apalache
        let output =
            execution::run_apalache(&self.config, &detection, &tla_path, Some(&cfg_path)).await?;

        // Parse results
        let result = parsing::parse_output(&output);

        Ok(result)
    }

    async fn health_check(&self) -> HealthStatus {
        let detection = detection::detect_apalache(&self.config).await;
        match detection {
            ApalacheDetection::Standalone(path) => {
                info!("Apalache available at {:?}", path);
                HealthStatus::Healthy
            }
            ApalacheDetection::Jar {
                java_path,
                jar_path,
            } => {
                info!(
                    "Apalache available via JAR at {:?} (Java: {:?})",
                    jar_path, java_path
                );
                HealthStatus::Healthy
            }
            ApalacheDetection::NotFound(reason) => {
                warn!("Apalache not available: {}", reason);
                HealthStatus::Unavailable { reason }
            }
        }
    }
}
