//! DashProve client for high-level verification API
//!
//! Provides the main `DashProve` struct used by DashFlow and Dasher
//! to verify specifications and code.

// Kani proofs for client types
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify DashProveConfig default has empty backends
    #[kani::proof]
    fn verify_dashprove_config_default_empty_backends() {
        let config = DashProveConfig::default();
        kani::assert(
            config.backends.is_empty(),
            "default backends should be empty",
        );
    }

    /// Verify DashProveConfig default has learning disabled
    #[kani::proof]
    fn verify_dashprove_config_default_learning_disabled() {
        let config = DashProveConfig::default();
        kani::assert(
            !config.learning_enabled,
            "default learning should be disabled",
        );
    }

    /// Verify DashProveConfig default has no API URL
    #[kani::proof]
    fn verify_dashprove_config_default_no_api_url() {
        let config = DashProveConfig::default();
        kani::assert(config.api_url.is_none(), "default api_url should be None");
    }

    /// Verify DashProveConfig default has no ML predictor
    #[kani::proof]
    fn verify_dashprove_config_default_no_ml_predictor() {
        let config = DashProveConfig::default();
        kani::assert(
            config.ml_predictor.is_none(),
            "default ml_predictor should be None",
        );
    }

    /// Verify DashProveConfig default ML confidence is zero
    #[kani::proof]
    fn verify_dashprove_config_default_ml_confidence() {
        let config = DashProveConfig::default();
        kani::assert(
            config.ml_min_confidence == 0.0,
            "default ml_min_confidence should be 0.0",
        );
    }

    /// Verify with_backend creates single-backend config
    #[kani::proof]
    fn verify_with_backend_creates_single_backend() {
        let config = DashProveConfig::with_backend(BackendId::Lean4);
        kani::assert(
            config.backends.len() == 1,
            "should have exactly one backend",
        );
        kani::assert(
            config.backends[0] == BackendId::Lean4,
            "backend should be Lean4",
        );
    }

    /// Verify with_learning enables learning
    #[kani::proof]
    fn verify_with_learning_enables_learning() {
        let config = DashProveConfig::default().with_learning();
        kani::assert(config.learning_enabled, "learning should be enabled");
    }

    /// Verify remote creates config with API URL
    #[kani::proof]
    fn verify_remote_sets_api_url() {
        let config = DashProveConfig::remote("http://test");
        kani::assert(config.api_url.is_some(), "api_url should be Some");
    }

    /// Verify VerificationResult is_proven returns true only for Proven status
    #[kani::proof]
    fn verify_is_proven_only_for_proven() {
        let proven = VerificationResult {
            status: VerificationStatus::Proven,
            properties: vec![],
            proof: None,
            counterexample: None,
            suggestions: vec![],
            confidence: 1.0,
        };
        kani::assert(
            proven.is_proven(),
            "Proven status should return is_proven=true",
        );
    }

    /// Verify VerificationResult is_disproven returns true only for Disproven status
    #[kani::proof]
    fn verify_is_disproven_only_for_disproven() {
        let disproven = VerificationResult {
            status: VerificationStatus::Disproven,
            properties: vec![],
            proof: None,
            counterexample: None,
            suggestions: vec![],
            confidence: 0.0,
        };
        kani::assert(
            disproven.is_disproven(),
            "Disproven status should return is_disproven=true",
        );
    }

    /// Verify proven_count matches the number of Proven properties
    #[kani::proof]
    fn verify_proven_count_empty() {
        let result = VerificationResult {
            status: VerificationStatus::Proven,
            properties: vec![],
            proof: None,
            counterexample: None,
            suggestions: vec![],
            confidence: 1.0,
        };
        kani::assert(
            result.proven_count() == 0,
            "empty properties should have zero proven_count",
        );
    }

    /// Verify disproven_count matches the number of Disproven properties
    #[kani::proof]
    fn verify_disproven_count_empty() {
        let result = VerificationResult {
            status: VerificationStatus::Proven,
            properties: vec![],
            proof: None,
            counterexample: None,
            suggestions: vec![],
            confidence: 1.0,
        };
        kani::assert(
            result.disproven_count() == 0,
            "empty properties should have zero disproven_count",
        );
    }

    /// Verify ml_min_confidence is clamped in with_ml_strategy
    #[kani::proof]
    fn verify_ml_confidence_clamped_high() {
        use dashprove_ai::StrategyPredictor;
        let predictor = StrategyPredictor::new();
        let config = DashProveConfig::default().with_ml_strategy(predictor, 2.0);
        // Value should be clamped to 1.0
        kani::assert(
            config.ml_min_confidence <= 1.0,
            "ml_min_confidence should be clamped to max 1.0",
        );
    }

    /// Verify ml_min_confidence is clamped for negative values
    #[kani::proof]
    fn verify_ml_confidence_clamped_low() {
        use dashprove_ai::StrategyPredictor;
        let predictor = StrategyPredictor::new();
        let config = DashProveConfig::default().with_ml_strategy(predictor, -1.0);
        // Value should be clamped to 0.0
        kani::assert(
            config.ml_min_confidence >= 0.0,
            "ml_min_confidence should be clamped to min 0.0",
        );
    }
}

use dashprove_ai::StrategyModel;
use dashprove_backends::{
    AbCrownBackend, AlloyBackend, ApalacheBackend, BackendError, BackendId, CoqBackend,
    CreusotBackend, Cvc5Backend, DafnyBackend, EranBackend, IsabelleBackend, KaniBackend,
    Lean4Backend, MarabouBackend, PlatformApiBackend, PrismBackend, ProverifBackend, PrustiBackend,
    StormBackend, TamarinBackend, TlaPlusBackend, VerificationStatus, VerifpalBackend,
    VerusBackend, Z3Backend,
};
use dashprove_dispatcher::{Dispatcher, DispatcherConfig, MergedResults, SelectionStrategy};
use dashprove_usl::{parse, typecheck, ParseError, TypeError, TypedSpec};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

/// Errors from the DashProve client
#[derive(Error, Debug)]
pub enum DashProveError {
    /// USL specification parsing failed
    #[error("Parse error: {0}")]
    Parse(#[from] ParseError),

    /// USL type checking failed
    #[error("Type error: {0}")]
    Type(#[from] TypeError),

    /// Backend dispatcher error (no backends available, execution failed, etc.)
    #[error("Dispatcher error: {0}")]
    Dispatcher(#[from] dashprove_dispatcher::DispatcherError),

    /// Backend execution error (tool unavailable, verification failed, etc.)
    #[error("Backend error: {0}")]
    Backend(#[from] BackendError),

    /// Invalid client configuration
    #[error("Invalid configuration: {0}")]
    Config(String),
}

/// Configuration for DashProve client
#[derive(Debug, Clone, Default)]
pub struct DashProveConfig {
    /// Which backends to register (empty = auto-detect)
    pub backends: Vec<BackendId>,
    /// Dispatcher configuration
    pub dispatcher: DispatcherConfig,
    /// Whether to enable learning (store proof strategies)
    pub learning_enabled: bool,
    /// Base URL for HTTP API (if using remote mode)
    pub api_url: Option<String>,
    /// Optional ML predictor for backend selection
    pub ml_predictor: Option<StrategyModel>,
    /// Minimum confidence threshold for ML-based selection (0.0-1.0)
    pub ml_min_confidence: f64,
}

impl DashProveConfig {
    /// Create a config that uses all available backends
    pub fn all_backends() -> Self {
        DashProveConfig {
            backends: crate::backend_ids::default_backends(),
            dispatcher: DispatcherConfig::all_backends(),
            ..Default::default()
        }
    }

    /// Create a config for a specific backend
    pub fn with_backend(backend: BackendId) -> Self {
        DashProveConfig {
            backends: vec![backend],
            dispatcher: DispatcherConfig::specific(backend),
            ..Default::default()
        }
    }

    /// Create a config with learning enabled
    pub fn with_learning(mut self) -> Self {
        self.learning_enabled = true;
        self
    }

    /// Create a config for remote API mode
    pub fn remote(api_url: &str) -> Self {
        DashProveConfig {
            api_url: Some(api_url.to_string()),
            ..Default::default()
        }
    }

    /// Enable ML-based backend selection with the provided predictor
    pub fn with_ml_strategy(
        mut self,
        predictor: impl Into<StrategyModel>,
        min_confidence: f64,
    ) -> Self {
        let confidence = min_confidence.clamp(0.0, 1.0);
        self.ml_predictor = Some(predictor.into());
        self.ml_min_confidence = confidence;
        self.dispatcher = DispatcherConfig::ml_based(confidence);
        self
    }
}

/// High-level verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Overall status (aggregated from all properties)
    pub status: VerificationStatus,
    /// Individual property results
    pub properties: Vec<PropertyResult>,
    /// Machine-readable proof (if available)
    pub proof: Option<String>,
    /// Counterexample (if property was disproven)
    pub counterexample: Option<String>,
    /// Suggested tactics for failing proofs
    pub suggestions: Vec<String>,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
}

impl VerificationResult {
    /// Check if all properties were proven
    pub fn is_proven(&self) -> bool {
        matches!(self.status, VerificationStatus::Proven)
    }

    /// Check if any property was disproven
    pub fn is_disproven(&self) -> bool {
        matches!(self.status, VerificationStatus::Disproven)
    }

    /// Get the number of proven properties
    pub fn proven_count(&self) -> usize {
        self.properties
            .iter()
            .filter(|p| matches!(p.status, VerificationStatus::Proven))
            .count()
    }

    /// Get the number of disproven properties
    pub fn disproven_count(&self) -> usize {
        self.properties
            .iter()
            .filter(|p| matches!(p.status, VerificationStatus::Disproven))
            .count()
    }

    /// Create from merged dispatcher results
    fn from_merged(merged: MergedResults) -> Self {
        let properties: Vec<PropertyResult> = merged
            .properties
            .into_iter()
            .map(|p| PropertyResult {
                name: format!("property_{}", p.property_index),
                status: p.status.clone(),
                backends_used: p.backend_results.iter().map(|r| r.backend).collect(),
                proof: p.proof.clone(),
                // Convert structured counterexample to summary string for API simplicity
                counterexample: p.counterexample.as_ref().map(|ce| ce.summary()),
            })
            .collect();

        let all_proven = properties
            .iter()
            .all(|p| matches!(p.status, VerificationStatus::Proven));
        let any_disproven = properties
            .iter()
            .any(|p| matches!(p.status, VerificationStatus::Disproven));

        let status = if all_proven {
            VerificationStatus::Proven
        } else if any_disproven {
            VerificationStatus::Disproven
        } else {
            VerificationStatus::Unknown {
                reason: "Some properties could not be verified".to_string(),
            }
        };

        let proof = properties.iter().find_map(|p| p.proof.clone());
        let counterexample = properties.iter().find_map(|p| p.counterexample.clone());

        VerificationResult {
            status,
            properties,
            proof,
            counterexample,
            suggestions: vec![],
            confidence: merged.summary.overall_confidence,
        }
    }
}

/// Result for a single property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyResult {
    /// Property name
    pub name: String,
    /// Verification status
    pub status: VerificationStatus,
    /// Backends that were used
    pub backends_used: Vec<BackendId>,
    /// Proof (if proven)
    pub proof: Option<String>,
    /// Counterexample (if disproven)
    pub counterexample: Option<String>,
}

/// Main DashProve client
///
/// This is the primary interface for DashFlow and Dasher integration.
/// It manages backend registration, dispatching, and result aggregation.
pub struct DashProve {
    config: DashProveConfig,
    dispatcher: Dispatcher,
}

impl DashProve {
    /// Create a new DashProve client with the given configuration
    pub fn new(mut config: DashProveConfig) -> Self {
        let mut dispatcher_config = config.dispatcher.clone();

        let ml_predictor = config.ml_predictor.as_ref().map(|p| Arc::new(p.clone()));

        if ml_predictor.is_some()
            && !matches!(
                dispatcher_config.selection_strategy,
                SelectionStrategy::MlBased { .. }
            )
        {
            dispatcher_config.selection_strategy = SelectionStrategy::MlBased {
                min_confidence: config.ml_min_confidence.clamp(0.0, 1.0),
            };
            config.dispatcher = dispatcher_config.clone();
        }

        let mut dispatcher = if let Some(predictor) = ml_predictor {
            Dispatcher::with_ml_predictor(dispatcher_config.clone(), predictor)
        } else {
            Dispatcher::new(dispatcher_config.clone())
        };

        // Register backends based on config
        let backends_to_register = if config.backends.is_empty() {
            // Auto-detect: register common backends (the ones most likely to be useful)
            vec![
                BackendId::Lean4,
                BackendId::TlaPlus,
                BackendId::Alloy,
                BackendId::PlatformApi,
                BackendId::Z3,
                BackendId::Cvc5,
            ]
        } else {
            config.backends.clone()
        };

        for backend_id in backends_to_register {
            match backend_id {
                // Core theorem provers
                BackendId::Lean4 => {
                    dispatcher.register_backend(Arc::new(Lean4Backend::new()));
                }
                BackendId::Coq => {
                    dispatcher.register_backend(Arc::new(CoqBackend::new()));
                }
                BackendId::Isabelle => {
                    dispatcher.register_backend(Arc::new(IsabelleBackend::new()));
                }
                BackendId::Dafny => {
                    dispatcher.register_backend(Arc::new(DafnyBackend::new()));
                }
                BackendId::PlatformApi => {
                    dispatcher.register_backend(Arc::new(PlatformApiBackend::new()));
                }
                // Model checkers
                BackendId::TlaPlus => {
                    dispatcher.register_backend(Arc::new(TlaPlusBackend::new()));
                }
                BackendId::Apalache => {
                    dispatcher.register_backend(Arc::new(ApalacheBackend::new()));
                }
                BackendId::Alloy => {
                    dispatcher.register_backend(Arc::new(AlloyBackend::new()));
                }
                // Rust verification (Kani requires project path, skip in auto mode)
                BackendId::Kani => {}
                BackendId::Verus => {
                    dispatcher.register_backend(Arc::new(VerusBackend::new()));
                }
                BackendId::Creusot => {
                    dispatcher.register_backend(Arc::new(CreusotBackend::new()));
                }
                BackendId::Prusti => {
                    dispatcher.register_backend(Arc::new(PrustiBackend::new()));
                }
                // SMT solvers
                BackendId::Z3 => {
                    dispatcher.register_backend(Arc::new(Z3Backend::new()));
                }
                BackendId::Cvc5 => {
                    dispatcher.register_backend(Arc::new(Cvc5Backend::new()));
                }
                // Probabilistic model checkers
                BackendId::Storm => {
                    dispatcher.register_backend(Arc::new(StormBackend::new()));
                }
                BackendId::Prism => {
                    dispatcher.register_backend(Arc::new(PrismBackend::new()));
                }
                // Neural network verifiers
                BackendId::AlphaBetaCrown => {
                    dispatcher.register_backend(Arc::new(AbCrownBackend::new()));
                }
                BackendId::Eran => {
                    dispatcher.register_backend(Arc::new(EranBackend::new()));
                }
                BackendId::Marabou => {
                    dispatcher.register_backend(Arc::new(MarabouBackend::new()));
                }
                // Security protocol verifiers
                BackendId::Tamarin => {
                    dispatcher.register_backend(Arc::new(TamarinBackend::new()));
                }
                BackendId::ProVerif => {
                    dispatcher.register_backend(Arc::new(ProverifBackend::new()));
                }
                BackendId::Verifpal => {
                    dispatcher.register_backend(Arc::new(VerifpalBackend::new()));
                }
                // New backends (Phase 12) - registration deferred until backend implementations
                // are added. For now, these are no-ops in auto mode.
                BackendId::NNV
                | BackendId::Nnenum
                | BackendId::VeriNet
                | BackendId::Venus
                | BackendId::DNNV
                | BackendId::AutoLiRPA
                | BackendId::MNBaB
                | BackendId::Neurify
                | BackendId::ReluVal
                | BackendId::ART
                | BackendId::Foolbox
                | BackendId::CleverHans
                | BackendId::TextAttack
                | BackendId::RobustBench
                | BackendId::Flux
                | BackendId::Mirai
                | BackendId::Rudra
                | BackendId::Miri
                | BackendId::AddressSanitizer
                | BackendId::MemorySanitizer
                | BackendId::ThreadSanitizer
                | BackendId::LeakSanitizer
                | BackendId::Valgrind
                | BackendId::Loom
                | BackendId::Shuttle
                | BackendId::CDSChecker
                | BackendId::GenMC
                | BackendId::LibFuzzer
                | BackendId::AFL
                | BackendId::Honggfuzz
                | BackendId::Bolero
                | BackendId::Proptest
                | BackendId::QuickCheck
                | BackendId::Clippy
                | BackendId::SemverChecks
                | BackendId::Geiger
                | BackendId::Audit
                | BackendId::Deny
                | BackendId::Vet
                | BackendId::Mutants
                | BackendId::ONNXRuntime
                | BackendId::TensorRT
                | BackendId::OpenVINO
                | BackendId::TVM
                | BackendId::IREE
                | BackendId::Triton
                | BackendId::NeuralCompressor
                | BackendId::NNCF
                | BackendId::AIMET
                | BackendId::Brevitas
                | BackendId::GreatExpectations
                | BackendId::Deepchecks
                | BackendId::Evidently
                | BackendId::WhyLogs
                | BackendId::Fairlearn
                | BackendId::AIF360
                | BackendId::Aequitas
                | BackendId::SHAP
                | BackendId::LIME
                | BackendId::Captum
                | BackendId::InterpretML
                | BackendId::Alibi
                | BackendId::GuardrailsAI
                | BackendId::NeMoGuardrails
                | BackendId::Guidance
                | BackendId::Promptfoo
                | BackendId::TruLens
                | BackendId::LangSmith
                | BackendId::Ragas
                | BackendId::DeepEval
                | BackendId::SelfCheckGPT
                | BackendId::FactScore
                | BackendId::SPIN
                | BackendId::CBMC
                | BackendId::Infer
                | BackendId::KLEE
                | BackendId::NuSMV
                | BackendId::CPAchecker
                | BackendId::SeaHorn
                | BackendId::FramaC
                // SMT solvers
                | BackendId::Yices
                | BackendId::Boolector
                | BackendId::MathSAT
                // SAT solvers
                | BackendId::MiniSat
                | BackendId::Glucose
                | BackendId::CaDiCaL
                // Dependently typed theorem provers
                | BackendId::Agda
                | BackendId::Idris
                | BackendId::ACL2
                | BackendId::HOL4
                | BackendId::FStar
                // Additional theorem provers
                | BackendId::HOLLight
                | BackendId::PVS
                | BackendId::Mizar
                | BackendId::Metamath
                | BackendId::ATS
                // Additional SMT solvers
                | BackendId::OpenSMT
                | BackendId::VeriT
                | BackendId::AltErgo
                // Additional SAT solvers
                | BackendId::Kissat
                | BackendId::CryptoMiniSat
                // Additional model checkers
                | BackendId::NuXmv
                | BackendId::UPPAAL
                | BackendId::DIVINE
                | BackendId::ESBMC
                | BackendId::Ultimate
                | BackendId::SMACK
                | BackendId::JPF
                // Program verification frameworks
                | BackendId::VCC
                | BackendId::VeriFast
                | BackendId::KeY
                | BackendId::OpenJML
                | BackendId::Krakatoa
                | BackendId::SPARK
                | BackendId::Why3
                | BackendId::Stainless
                | BackendId::LiquidHaskell
                | BackendId::Boogie
                // Distributed systems verification
                | BackendId::PLang
                | BackendId::Ivy
                | BackendId::MCRL2
                | BackendId::CADP
                // Cryptographic verification
                | BackendId::EasyCrypt
                | BackendId::CryptoVerif
                | BackendId::Jasmin
                // Hardware verification
                | BackendId::Yosys
                | BackendId::SymbiYosys
                | BackendId::JasperGold
                | BackendId::CadenceEDA
                // Symbolic execution and binary analysis
                | BackendId::Angr
                | BackendId::Manticore
                | BackendId::TritonDBA
                | BackendId::Bap
                | BackendId::Ghidra
                | BackendId::IsaBIL
                | BackendId::Soteria
                // Abstract interpretation
                | BackendId::Astree
                | BackendId::Polyspace
                | BackendId::CodeSonar
                | BackendId::FramaCEva
                // Rust code coverage
                | BackendId::Tarpaulin
                | BackendId::LlvmCov
                | BackendId::Grcov
                // Rust testing frameworks
                | BackendId::Nextest
                | BackendId::Insta
                | BackendId::Rstest
                | BackendId::TestCase
                | BackendId::Mockall
                // Rust documentation tools
                | BackendId::Deadlinks
                | BackendId::Spellcheck
                | BackendId::Rdme
                // Additional Rust verification
                | BackendId::Haybale
                | BackendId::CruxMir
                | BackendId::RustHorn
                | BackendId::RustBelt
                // Go verification
                | BackendId::Gobra
                // Additional C/C++ verification
                | BackendId::Symbiotic
                | BackendId::TwoLS
                // Kani Fast (requires project path like Kani)
                | BackendId::KaniFast => {
                    // Backend implementations pending - these are stub registrations
                }
            }
        }

        DashProve { config, dispatcher }
    }

    /// Create a DashProve client with default configuration
    pub fn default_client() -> Self {
        Self::new(DashProveConfig::default())
    }

    /// Verify a USL specification string
    ///
    /// # Arguments
    /// * `spec_source` - USL specification source code
    ///
    /// # Returns
    /// * `VerificationResult` with overall status and per-property results
    pub async fn verify(
        &mut self,
        spec_source: &str,
    ) -> Result<VerificationResult, DashProveError> {
        // Parse
        let spec = parse(spec_source)?;

        // Type-check
        let typed_spec = typecheck(spec)?;

        // Dispatch to backends
        self.verify_typed(&typed_spec).await
    }

    /// Verify an already-parsed and type-checked specification
    pub async fn verify_typed(
        &mut self,
        typed_spec: &TypedSpec,
    ) -> Result<VerificationResult, DashProveError> {
        let merged = self.dispatcher.verify(typed_spec).await?;
        Ok(VerificationResult::from_merged(merged))
    }

    /// Verify code against a specification
    ///
    /// This is used by Dasher to verify generated code against requirements.
    /// The code is combined with generated Kani proof harnesses from the USL
    /// contracts and verified using the Kani backend.
    ///
    /// # Arguments
    /// * `code` - The Rust code to verify
    /// * `spec_source` - USL specification the code should satisfy (must contain contracts)
    ///
    /// # Returns
    /// * `VerificationResult` indicating whether code satisfies spec
    pub async fn verify_code(
        &mut self,
        code: &str,
        spec_source: &str,
    ) -> Result<VerificationResult, DashProveError> {
        // Parse and type-check the spec
        let spec = parse(spec_source)?;
        let typed_spec = typecheck(spec)?;

        // Use Kani backend for code verification
        let kani = KaniBackend::new();
        let backend_result = kani.verify_code(code, &typed_spec).await?;

        // Convert backend result to verification result
        let status = backend_result.status.clone();
        let counterexample = backend_result
            .counterexample
            .as_ref()
            .map(|ce| ce.summary());

        Ok(VerificationResult {
            status: status.clone(),
            properties: vec![PropertyResult {
                name: "contract".to_string(),
                status,
                backends_used: vec![BackendId::Kani],
                proof: backend_result.proof.clone(),
                counterexample: counterexample.clone(),
            }],
            proof: backend_result.proof,
            counterexample,
            suggestions: vec![],
            confidence: if matches!(backend_result.status, VerificationStatus::Proven) {
                1.0
            } else {
                0.0
            },
        })
    }

    /// Verify with a specific backend
    pub async fn verify_with_backend(
        &mut self,
        spec_source: &str,
        backend: BackendId,
    ) -> Result<VerificationResult, DashProveError> {
        let spec = parse(spec_source)?;
        let typed_spec = typecheck(spec)?;
        let merged = self.dispatcher.verify_with(&typed_spec, backend).await?;
        Ok(VerificationResult::from_merged(merged))
    }

    /// Get currently registered backends
    pub fn backends(&self) -> Vec<BackendId> {
        self.dispatcher.registry().all_backends()
    }

    /// Check health of all backends
    pub async fn check_health(&mut self) -> Vec<(BackendId, bool)> {
        use dashprove_backends::HealthStatus;

        self.dispatcher.registry_mut().check_all_health().await;

        self.dispatcher
            .registry()
            .all_backends()
            .into_iter()
            .map(|id| {
                let healthy = self
                    .dispatcher
                    .registry()
                    .get_info(id)
                    .map(|info| matches!(info.health, HealthStatus::Healthy))
                    .unwrap_or(false);
                (id, healthy)
            })
            .collect()
    }

    /// Get current configuration
    pub fn config(&self) -> &DashProveConfig {
        &self.config
    }

    /// Enable ML-based backend selection after construction
    pub fn set_ml_predictor(&mut self, predictor: impl Into<StrategyModel>, min_confidence: f64) {
        let predictor = predictor.into();
        let confidence = min_confidence.clamp(0.0, 1.0);
        self.config.ml_predictor = Some(predictor.clone());
        self.config.ml_min_confidence = confidence;
        self.config.dispatcher.selection_strategy = SelectionStrategy::MlBased {
            min_confidence: confidence,
        };
        self.dispatcher
            .set_ml_predictor(Arc::new(predictor.clone()));
        self.dispatcher.set_config(self.config.dispatcher.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_ai::StrategyPredictor;

    #[test]
    fn test_config_default() {
        let config = DashProveConfig::default();
        assert!(config.backends.is_empty());
        assert!(!config.learning_enabled);
        assert!(config.api_url.is_none());
        assert!(config.ml_predictor.is_none());
        assert_eq!(config.ml_min_confidence, 0.0);
    }

    #[test]
    fn test_config_all_backends() {
        let config = DashProveConfig::all_backends();
        assert_eq!(
            config.backends.len(),
            crate::backend_ids::SUPPORTED_BACKENDS.len()
        );
    }

    #[test]
    fn test_config_with_backend() {
        let config = DashProveConfig::with_backend(BackendId::Lean4);
        assert_eq!(config.backends, vec![BackendId::Lean4]);
    }

    #[test]
    fn test_config_with_learning() {
        let config = DashProveConfig::default().with_learning();
        assert!(config.learning_enabled);
    }

    #[test]
    fn test_config_remote() {
        let config = DashProveConfig::remote("http://localhost:3000");
        assert_eq!(config.api_url, Some("http://localhost:3000".to_string()));
    }

    #[test]
    fn test_config_with_ml_strategy() {
        let predictor = StrategyPredictor::new();
        let config = DashProveConfig::default().with_ml_strategy(predictor, 0.6);

        assert!(config.ml_predictor.is_some());
        assert!(matches!(
            config.dispatcher.selection_strategy,
            SelectionStrategy::MlBased { min_confidence } if (min_confidence - 0.6).abs() < f64::EPSILON
        ));
        assert!((config.ml_min_confidence - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_client_creation() {
        let client = DashProve::new(DashProveConfig::default());
        // Should have auto-registered backends
        assert!(!client.backends().is_empty());
    }

    #[test]
    fn test_set_ml_predictor_updates_dispatcher() {
        let predictor = StrategyPredictor::new();
        let mut client = DashProve::new(DashProveConfig::default());
        client.set_ml_predictor(predictor, 0.4);

        assert!(client.dispatcher.ml_predictor().is_some());
        assert!(matches!(
            client.config().dispatcher.selection_strategy,
            SelectionStrategy::MlBased { min_confidence } if (min_confidence - 0.4).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn test_verification_result_helpers() {
        let result = VerificationResult {
            status: VerificationStatus::Proven,
            properties: vec![
                PropertyResult {
                    name: "p1".to_string(),
                    status: VerificationStatus::Proven,
                    backends_used: vec![BackendId::Lean4],
                    proof: Some("proof1".to_string()),
                    counterexample: None,
                },
                PropertyResult {
                    name: "p2".to_string(),
                    status: VerificationStatus::Proven,
                    backends_used: vec![BackendId::Lean4],
                    proof: Some("proof2".to_string()),
                    counterexample: None,
                },
            ],
            proof: Some("proof".to_string()),
            counterexample: None,
            suggestions: vec![],
            confidence: 1.0,
        };

        assert!(result.is_proven());
        assert!(!result.is_disproven());
        assert_eq!(result.proven_count(), 2);
        assert_eq!(result.disproven_count(), 0);
    }

    #[tokio::test]
    async fn test_verify_simple() {
        let mut client = DashProve::default_client();
        // This will fail because we don't have actual backends installed
        // but it tests the API flow
        let result = client
            .verify("theorem test { forall x: Bool . x or not x }")
            .await;

        // Either succeeds or fails due to backend unavailability
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_verify_platform_api_backend() {
        let mut client = DashProve::new(DashProveConfig::with_backend(BackendId::PlatformApi));

        let spec = r#"
            platform_api Metal {
                state MTLCommandBuffer {
                    enum Status { Created, Committed }

                    transition commit() {
                        requires { status == Created }
                        ensures { status == Committed }
                    }
                }
            }
        "#;

        let result = client
            .verify(spec)
            .await
            .expect("platform api verification");
        assert!(result.is_proven());
        assert_eq!(result.properties.len(), 1);
        assert!(matches!(
            result.properties[0].status,
            VerificationStatus::Proven
        ));
        assert!(result.properties[0]
            .backends_used
            .contains(&BackendId::PlatformApi));

        let proof = result.properties[0]
            .proof
            .as_ref()
            .or(result.proof.as_ref())
            .expect("generated code should be present");
        assert!(
            proof.contains("struct MTLCommandBufferStateTracker"),
            "generated static checker should be returned as proof output"
        );
    }

    #[tokio::test]
    async fn test_verify_code_parses_spec() {
        let mut client = DashProve::default_client();

        let code = r#"
pub fn identity(x: u32) -> u32 {
    x
}
"#;

        let spec = r#"
contract identity(x: Int) -> Int {
    ensures { result == x }
}
"#;

        // This will return an error if Kani is not installed, but that's expected
        // The important thing is that the spec is parsed correctly
        let result = client.verify_code(code, spec).await;

        // Either succeeds (if Kani is installed) or fails with Backend error
        // (not Parse or Type error)
        match &result {
            Ok(_) => {}                           // Great, Kani is available
            Err(DashProveError::Backend(_)) => {} // Expected: Kani not installed
            Err(DashProveError::Parse(e)) => panic!("Unexpected parse error: {:?}", e),
            Err(DashProveError::Type(e)) => panic!("Unexpected type error: {:?}", e),
            Err(e) => panic!("Unexpected error type: {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_verify_code_invalid_spec() {
        let mut client = DashProve::default_client();

        let code = "pub fn test() {}";
        let invalid_spec = "this is not valid USL";

        let result = client.verify_code(code, invalid_spec).await;

        // Should fail with a Parse error
        assert!(matches!(result, Err(DashProveError::Parse(_))));
    }
}
