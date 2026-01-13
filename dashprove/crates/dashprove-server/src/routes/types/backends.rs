use dashprove_backends::PropertyType;
use serde::{Deserialize, Serialize};

pub use dashprove::backend_ids::{
    backend_metric_label, default_backends, BackendIdParam, SUPPORTED_BACKENDS,
};

// ============ Backends Listing Types ============

/// Response listing available backends
#[derive(Debug, Serialize, Deserialize)]
pub struct BackendsResponse {
    /// List of available backends
    pub backends: Vec<BackendInfo>,
}

/// Information about a single backend
#[derive(Debug, Serialize, Deserialize)]
pub struct BackendInfo {
    /// Backend identifier
    pub id: BackendIdParam,
    /// Display name
    pub name: String,
    /// Property types this backend supports
    pub supports: Vec<PropertyTypeResponse>,
    /// Current health status
    pub health: HealthStatusResponse,
}

/// Property type for response
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropertyTypeResponse {
    /// Mathematical theorem
    Theorem,
    /// Temporal logic property
    Temporal,
    /// Pre/post contract
    Contract,
    /// State invariant
    Invariant,
    /// Refinement relation
    Refinement,
    /// Neural network robustness
    NeuralRobustness,
    /// Neural network reachability
    NeuralReachability,
    /// Probabilistic property
    Probabilistic,
    /// Security protocol property
    SecurityProtocol,
    /// Platform API constraint (external API state machines)
    PlatformApi,
}

impl From<PropertyType> for PropertyTypeResponse {
    fn from(value: PropertyType) -> Self {
        match value {
            PropertyType::Theorem => PropertyTypeResponse::Theorem,
            PropertyType::Temporal => PropertyTypeResponse::Temporal,
            PropertyType::Contract => PropertyTypeResponse::Contract,
            PropertyType::Invariant => PropertyTypeResponse::Invariant,
            PropertyType::Refinement => PropertyTypeResponse::Refinement,
            PropertyType::NeuralRobustness => PropertyTypeResponse::NeuralRobustness,
            PropertyType::NeuralReachability => PropertyTypeResponse::NeuralReachability,
            PropertyType::Probabilistic => PropertyTypeResponse::Probabilistic,
            PropertyType::SecurityProtocol => PropertyTypeResponse::SecurityProtocol,
            PropertyType::PlatformApi => PropertyTypeResponse::PlatformApi,
            // New property types (Phase 12) - map to reasonable defaults
            PropertyType::AdversarialRobustness => PropertyTypeResponse::NeuralRobustness,
            PropertyType::MemorySafety
            | PropertyType::UndefinedBehavior
            | PropertyType::DataRace
            | PropertyType::MemoryLeak => PropertyTypeResponse::Contract,
            PropertyType::Fuzzing | PropertyType::PropertyBased | PropertyType::MutationTesting => {
                PropertyTypeResponse::Contract
            }
            PropertyType::Lint
            | PropertyType::ApiCompatibility
            | PropertyType::SecurityVulnerability => PropertyTypeResponse::Contract,
            PropertyType::DependencyPolicy
            | PropertyType::SupplyChain
            | PropertyType::UnsafeAudit => PropertyTypeResponse::SecurityProtocol,
            PropertyType::ModelOptimization | PropertyType::ModelCompression => {
                PropertyTypeResponse::NeuralRobustness
            }
            PropertyType::DataQuality | PropertyType::Fairness | PropertyType::Interpretability => {
                PropertyTypeResponse::Contract
            }
            PropertyType::LLMGuardrails
            | PropertyType::LLMEvaluation
            | PropertyType::HallucinationDetection => PropertyTypeResponse::Contract,
        }
    }
}

/// Health status for response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum HealthStatusResponse {
    /// Backend is working normally
    Healthy,
    /// Backend is partially available
    Degraded {
        /// Description of the degradation
        reason: String,
    },
    /// Backend is not available
    Unavailable {
        /// Description of why unavailable
        reason: String,
    },
}
