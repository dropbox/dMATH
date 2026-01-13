//! VeriNet neural network verification backend
//!
//! VeriNet is a complete verifier using:
//! - Symbolic interval propagation
//! - Input/ReLU splitting for refinement
//! - GPU acceleration support
//!
//! See: <https://github.com/vas-group-imperial/VeriNet>
//!
//! # Installation
//!
//! ```bash
//! git clone https://github.com/vas-group-imperial/VeriNet
//! cd VeriNet && pip install -e .
//! ```

// =============================================
// Kani Proofs for VeriNet Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use std::time::Duration;

    // ---- SplittingStrategy Tests ----

    /// Verify SplittingStrategy::default is Input
    #[kani::proof]
    fn proof_splitting_strategy_default_is_input() {
        let strategy = SplittingStrategy::default();
        kani::assert(
            matches!(strategy, SplittingStrategy::Input),
            "Default strategy should be Input",
        );
    }

    /// Verify SplittingStrategy::Input as_str
    #[kani::proof]
    fn proof_splitting_strategy_input_str() {
        let strategy = SplittingStrategy::Input;
        kani::assert(strategy.as_str() == "input", "Input should be input");
    }

    /// Verify SplittingStrategy::ReLU as_str
    #[kani::proof]
    fn proof_splitting_strategy_relu_str() {
        let strategy = SplittingStrategy::ReLU;
        kani::assert(strategy.as_str() == "relu", "ReLU should be relu");
    }

    /// Verify SplittingStrategy::Adaptive as_str
    #[kani::proof]
    fn proof_splitting_strategy_adaptive_str() {
        let strategy = SplittingStrategy::Adaptive;
        kani::assert(
            strategy.as_str() == "adaptive",
            "Adaptive should be adaptive",
        );
    }

    // ---- VeriNetConfig Default Tests ----

    /// Verify VeriNetConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_verinet_config_default_timeout() {
        let config = VeriNetConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify VeriNetConfig::default python_path is None
    #[kani::proof]
    fn proof_verinet_config_default_python_path_none() {
        let config = VeriNetConfig::default();
        kani::assert(
            config.python_path.is_none(),
            "Default python_path should be None",
        );
    }

    /// Verify VeriNetConfig::default strategy is Input
    #[kani::proof]
    fn proof_verinet_config_default_strategy() {
        let config = VeriNetConfig::default();
        kani::assert(
            matches!(config.strategy, SplittingStrategy::Input),
            "Default strategy should be Input",
        );
    }

    /// Verify VeriNetConfig::default epsilon is 0.01
    #[kani::proof]
    fn proof_verinet_config_default_epsilon() {
        let config = VeriNetConfig::default();
        kani::assert(config.epsilon == 0.01, "Default epsilon should be 0.01");
    }

    /// Verify VeriNetConfig::default max_depth is 15
    #[kani::proof]
    fn proof_verinet_config_default_max_depth() {
        let config = VeriNetConfig::default();
        kani::assert(config.max_depth == 15, "Default max_depth should be 15");
    }

    /// Verify VeriNetConfig::default model_path is None
    #[kani::proof]
    fn proof_verinet_config_default_model_path_none() {
        let config = VeriNetConfig::default();
        kani::assert(
            config.model_path.is_none(),
            "Default model_path should be None",
        );
    }

    /// Verify VeriNetConfig::default use_gpu is false
    #[kani::proof]
    fn proof_verinet_config_default_use_gpu() {
        let config = VeriNetConfig::default();
        kani::assert(!config.use_gpu, "Default use_gpu should be false");
    }

    /// Verify VeriNetConfig::complete uses ReLU strategy
    #[kani::proof]
    fn proof_verinet_config_complete_strategy() {
        let config = VeriNetConfig::complete();
        kani::assert(
            matches!(config.strategy, SplittingStrategy::ReLU),
            "Complete config should use ReLU strategy",
        );
    }

    /// Verify VeriNetConfig::complete has max_depth 20
    #[kani::proof]
    fn proof_verinet_config_complete_max_depth() {
        let config = VeriNetConfig::complete();
        kani::assert(
            config.max_depth == 20,
            "Complete config should have max_depth 20",
        );
    }

    /// Verify VeriNetConfig::fast uses Input strategy
    #[kani::proof]
    fn proof_verinet_config_fast_strategy() {
        let config = VeriNetConfig::fast();
        kani::assert(
            matches!(config.strategy, SplittingStrategy::Input),
            "Fast config should use Input strategy",
        );
    }

    /// Verify VeriNetConfig::fast has max_depth 10
    #[kani::proof]
    fn proof_verinet_config_fast_max_depth() {
        let config = VeriNetConfig::fast();
        kani::assert(
            config.max_depth == 10,
            "Fast config should have max_depth 10",
        );
    }

    // ---- VeriNetBackend Construction Tests ----

    /// Verify VeriNetBackend::new uses default config
    #[kani::proof]
    fn proof_verinet_backend_new_defaults() {
        let backend = VeriNetBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify VeriNetBackend::default equals VeriNetBackend::new
    #[kani::proof]
    fn proof_verinet_backend_default_equals_new() {
        let default_backend = VeriNetBackend::default();
        let new_backend = VeriNetBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify VeriNetBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_verinet_backend_with_config_timeout() {
        let config = VeriNetConfig {
            python_path: None,
            strategy: SplittingStrategy::Input,
            epsilon: 0.01,
            max_depth: 15,
            timeout: Duration::from_secs(600),
            model_path: None,
            use_gpu: false,
        };
        let backend = VeriNetBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "Custom timeout should be preserved",
        );
    }

    /// Verify VeriNetBackend::with_config preserves custom epsilon
    #[kani::proof]
    fn proof_verinet_backend_with_config_epsilon() {
        let config = VeriNetConfig {
            python_path: None,
            strategy: SplittingStrategy::Input,
            epsilon: 0.05,
            max_depth: 15,
            timeout: Duration::from_secs(300),
            model_path: None,
            use_gpu: false,
        };
        let backend = VeriNetBackend::with_config(config);
        kani::assert(
            backend.config.epsilon == 0.05,
            "Custom epsilon should be preserved",
        );
    }

    /// Verify VeriNetBackend::with_config preserves strategy
    #[kani::proof]
    fn proof_verinet_backend_with_config_strategy() {
        let config = VeriNetConfig {
            python_path: None,
            strategy: SplittingStrategy::Adaptive,
            epsilon: 0.01,
            max_depth: 15,
            timeout: Duration::from_secs(300),
            model_path: None,
            use_gpu: false,
        };
        let backend = VeriNetBackend::with_config(config);
        kani::assert(
            matches!(backend.config.strategy, SplittingStrategy::Adaptive),
            "Custom strategy should be preserved",
        );
    }

    /// Verify VeriNetBackend::with_config preserves max_depth
    #[kani::proof]
    fn proof_verinet_backend_with_config_max_depth() {
        let config = VeriNetConfig {
            python_path: None,
            strategy: SplittingStrategy::Input,
            epsilon: 0.01,
            max_depth: 25,
            timeout: Duration::from_secs(300),
            model_path: None,
            use_gpu: false,
        };
        let backend = VeriNetBackend::with_config(config);
        kani::assert(
            backend.config.max_depth == 25,
            "Custom max_depth should be preserved",
        );
    }

    /// Verify VeriNetBackend::with_config preserves use_gpu
    #[kani::proof]
    fn proof_verinet_backend_with_config_use_gpu() {
        let config = VeriNetConfig {
            python_path: None,
            strategy: SplittingStrategy::Input,
            epsilon: 0.01,
            max_depth: 15,
            timeout: Duration::from_secs(300),
            model_path: None,
            use_gpu: true,
        };
        let backend = VeriNetBackend::with_config(config);
        kani::assert(backend.config.use_gpu, "Custom use_gpu should be preserved");
    }

    // ---- Backend Trait Implementation Tests ----

    /// Verify id() returns VeriNet
    #[kani::proof]
    fn proof_backend_id_is_verinet() {
        let backend = VeriNetBackend::new();
        kani::assert(backend.id() == BackendId::VeriNet, "ID should be VeriNet");
    }

    /// Verify supports() includes NeuralRobustness
    #[kani::proof]
    fn proof_verinet_supports_neural_robustness() {
        let backend = VeriNetBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::NeuralRobustness),
            "Should support NeuralRobustness",
        );
    }

    /// Verify supports() includes NeuralReachability
    #[kani::proof]
    fn proof_verinet_supports_neural_reachability() {
        let backend = VeriNetBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::NeuralReachability),
            "Should support NeuralReachability",
        );
    }

    /// Verify supports() returns exactly two property types
    #[kani::proof]
    fn proof_verinet_supports_count() {
        let backend = VeriNetBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 2,
            "Should support exactly two property types",
        );
    }

    // ---- Script Parsing Tests ----

    /// Verify parse_verinet_output detects VERIFIED status
    #[kani::proof]
    fn proof_parse_verinet_verified() {
        let (status, _) = script::parse_verinet_output("VERINET_STATUS: VERIFIED", "");
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "VERIFIED should be Proven",
        );
    }

    /// Verify parse_verinet_output detects PARTIALLY_VERIFIED status
    #[kani::proof]
    fn proof_parse_verinet_partial() {
        let (status, _) = script::parse_verinet_output("VERINET_STATUS: PARTIALLY_VERIFIED", "");
        kani::assert(
            matches!(status, VerificationStatus::Partial { .. }),
            "PARTIALLY_VERIFIED should be Partial",
        );
    }

    /// Verify parse_verinet_output detects NOT_VERIFIED status
    #[kani::proof]
    fn proof_parse_verinet_not_verified() {
        let (status, _) = script::parse_verinet_output("VERINET_STATUS: NOT_VERIFIED", "");
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "NOT_VERIFIED should be Disproven",
        );
    }

    /// Verify parse_verinet_output detects VERINET_ERROR
    #[kani::proof]
    fn proof_parse_verinet_error() {
        let (status, _) = script::parse_verinet_output("VERINET_ERROR: Something failed", "");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "VERINET_ERROR should be Unknown",
        );
    }

    /// Verify parse_verinet_output returns Unknown for empty
    #[kani::proof]
    fn proof_parse_verinet_empty() {
        let (status, _) = script::parse_verinet_output("", "");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Empty output should be Unknown",
        );
    }
}

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{SplittingStrategy, VeriNetConfig};

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

/// VeriNet neural network verifier
pub struct VeriNetBackend {
    config: VeriNetConfig,
}

impl Default for VeriNetBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl VeriNetBackend {
    pub fn new() -> Self {
        Self {
            config: VeriNetConfig::default(),
        }
    }

    pub fn with_config(config: VeriNetConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for VeriNetBackend {
    fn id(&self) -> BackendId {
        BackendId::VeriNet
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::NeuralRobustness,
            PropertyType::NeuralReachability,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_verinet(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_verinet_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("verinet_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated VeriNet script:\n{}", script_content);

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

                debug!("VeriNet stdout: {}", stdout);
                debug!("VeriNet stderr: {}", stderr);

                let (status, counterexample) = script::parse_verinet_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::VeriNet,
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
        match detection::detect_verinet(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
