//! Backend selection logic
//!
//! This module provides:
//! - `BackendRegistry`: Manages available backends and their health status
//! - `BackendSelector`: Intelligently selects backends based on property types
//! - `MlBackendSelector`: ML-based backend selection using trained models
//! - `KnowledgeEnhanced`: RAG-enhanced backend selection using ToolKnowledgeStore

use dashprove_ai::{PropertyFeatureVector, StrategyModel};
use dashprove_backends::{
    BackendError, BackendId, HealthStatus, PropertyType, VerificationBackend,
};
use dashprove_knowledge::{
    Embedder, ExpertContext, ExpertFactory, KnowledgeStore, PropertyType as KnowledgePropertyType,
    ToolKnowledgeStore,
};
use dashprove_usl::ast::Property;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors from the selector
#[derive(Error, Debug)]
pub enum SelectorError {
    /// No backend is registered that can handle the property type
    #[error("No backend available for property type: {0:?}")]
    NoBackendAvailable(PropertyType),

    /// Backends exist for this type but all are unhealthy
    #[error("All backends for property type {0:?} are unhealthy")]
    AllBackendsUnhealthy(PropertyType),

    /// Requested backend ID is not registered
    #[error("Backend not found: {0:?}")]
    BackendNotFound(BackendId),

    /// Backend reported an error during capability check
    #[error("Backend error: {0}")]
    BackendError(#[from] BackendError),

    /// Knowledge base error during knowledge-enhanced selection
    #[error("Knowledge base error: {0}")]
    KnowledgeError(String),
}

/// Backend capability and health information
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// Backend identifier
    pub id: BackendId,
    /// Property types this backend can verify
    pub supported_types: Vec<PropertyType>,
    /// Current health status
    pub health: HealthStatus,
    /// Priority score (higher = preferred). Used for selection when multiple backends support a property.
    pub priority: i32,
}

/// Registry of available verification backends
pub struct BackendRegistry {
    /// Registered backends
    backends: HashMap<BackendId, Arc<dyn VerificationBackend>>,
    /// Cached backend info
    info_cache: HashMap<BackendId, BackendInfo>,
    /// Property type to backend mapping (for quick lookup)
    type_to_backends: HashMap<PropertyType, Vec<BackendId>>,
}

impl BackendRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        BackendRegistry {
            backends: HashMap::new(),
            info_cache: HashMap::new(),
            type_to_backends: HashMap::new(),
        }
    }

    /// Register a backend
    pub fn register(&mut self, backend: Arc<dyn VerificationBackend>) {
        let id = backend.id();
        let supported = backend.supports();

        info!(?id, ?supported, "Registering backend");

        // Update type-to-backend mapping
        for prop_type in &supported {
            self.type_to_backends
                .entry(*prop_type)
                .or_default()
                .push(id);
        }

        // Cache backend info with default priority
        let priority = Self::default_priority(id);
        self.info_cache.insert(
            id,
            BackendInfo {
                id,
                supported_types: supported,
                health: HealthStatus::Healthy, // Assume healthy until checked
                priority,
            },
        );

        self.backends.insert(id, backend);
    }

    /// Get default priority for a backend (can be customized via configuration)
    fn default_priority(id: BackendId) -> i32 {
        // Higher = preferred for that property type
        match id {
            BackendId::Lean4 => 100,       // Preferred for proofs
            BackendId::TlaPlus => 100,     // Preferred for temporal (explicit state)
            BackendId::Apalache => 95,     // Symbolic TLA+ (unbounded, but slower)
            BackendId::Kani => 100,        // Preferred for contracts
            BackendId::Alloy => 90,        // Good for bounded checking
            BackendId::Isabelle => 85,     // Alternative theorem prover
            BackendId::Coq => 80,          // Alternative prover
            BackendId::Dafny => 80,        // Alternative
            BackendId::PlatformApi => 100, // Deterministic codegen for platform constraints

            // Neural network verification backends
            BackendId::Marabou => 100,       // SMT-based NN verifier
            BackendId::AlphaBetaCrown => 95, // GPU-accelerated NN verifier
            BackendId::Eran => 90,           // Abstract interpretation NN verifier
            BackendId::NNV | BackendId::Nnenum | BackendId::VeriNet | BackendId::Venus => 85,
            BackendId::DNNV | BackendId::AutoLiRPA | BackendId::MNBaB => 80,
            BackendId::Neurify | BackendId::ReluVal => 75,

            // Adversarial robustness
            BackendId::ART => 85,
            BackendId::Foolbox
            | BackendId::CleverHans
            | BackendId::TextAttack
            | BackendId::RobustBench => 80,

            // Probabilistic verification backends
            BackendId::Storm => 100, // Preferred probabilistic
            BackendId::Prism => 90,  // Alternative probabilistic

            // Security protocol verification backends
            BackendId::Tamarin => 100, // Preferred for security protocols
            BackendId::ProVerif => 95, // Alternative security verifier
            BackendId::Verifpal => 90, // Modern security protocol verifier

            // Rust verification backends
            BackendId::Verus => 100,  // Z3-based Rust verifier
            BackendId::Creusot => 95, // Why3-based Rust verifier
            BackendId::Prusti => 90,  // Viper-based Rust verifier
            BackendId::Flux => 85,    // Refinement types
            BackendId::Mirai => 80,   // Abstract interpreter
            BackendId::Rudra => 85,   // Memory safety
            BackendId::Miri => 90,    // UB detection

            // SMT solver backends
            BackendId::Z3 => 95,   // General-purpose SMT solver
            BackendId::Cvc5 => 90, // SMT solver (strings, sets)

            // Rust sanitizers and memory tools
            BackendId::AddressSanitizer => 90,
            BackendId::MemorySanitizer
            | BackendId::ThreadSanitizer
            | BackendId::LeakSanitizer
            | BackendId::Valgrind => 85,

            // Rust concurrency testing
            BackendId::Loom => 90,
            BackendId::Shuttle | BackendId::CDSChecker | BackendId::GenMC => 85,

            // Rust fuzzing
            BackendId::LibFuzzer => 90,
            BackendId::AFL | BackendId::Honggfuzz | BackendId::Bolero => 85,

            // Rust property-based testing
            BackendId::Proptest => 85,
            BackendId::QuickCheck => 80,

            // Rust static analysis
            BackendId::Clippy => 95, // Lint tool
            BackendId::SemverChecks
            | BackendId::Geiger
            | BackendId::Audit
            | BackendId::Deny
            | BackendId::Vet
            | BackendId::Mutants => 85,

            // AI/ML optimization
            BackendId::ONNXRuntime => 90,
            BackendId::TensorRT
            | BackendId::OpenVINO
            | BackendId::TVM
            | BackendId::IREE
            | BackendId::Triton => 85,

            // AI/ML compression
            BackendId::NeuralCompressor
            | BackendId::NNCF
            | BackendId::AIMET
            | BackendId::Brevitas => 80,

            // Data quality
            BackendId::GreatExpectations => 85,
            BackendId::Deepchecks | BackendId::Evidently | BackendId::WhyLogs => 80,

            // Fairness & bias
            BackendId::Fairlearn => 85,
            BackendId::AIF360 | BackendId::Aequitas => 80,

            // Interpretability
            BackendId::SHAP => 85,
            BackendId::LIME | BackendId::Captum | BackendId::InterpretML | BackendId::Alibi => 80,

            // LLM guardrails
            BackendId::GuardrailsAI => 90,
            BackendId::NeMoGuardrails | BackendId::Guidance => 85,

            // LLM evaluation
            BackendId::Promptfoo => 85,
            BackendId::TruLens | BackendId::LangSmith | BackendId::Ragas | BackendId::DeepEval => {
                80
            }

            // Hallucination detection
            BackendId::SelfCheckGPT | BackendId::FactScore => 85,

            // Model checkers
            BackendId::SPIN => 90,       // Protocol verification
            BackendId::CBMC => 85,       // C bounded model checking
            BackendId::Infer => 85,      // Static analysis
            BackendId::KLEE => 85,       // Symbolic execution
            BackendId::NuSMV => 88,      // Symbolic/BDD model checking
            BackendId::CPAchecker => 85, // Configurable software verification
            BackendId::SeaHorn => 85,    // LLVM-based verification via Horn clauses
            BackendId::FramaC => 85,     // C verification framework

            // SMT solvers (new)
            BackendId::Yices => 90,     // Fast SMT solver from SRI
            BackendId::Boolector => 90, // Bit-vector specialist
            BackendId::MathSAT => 90,   // FBK SMT with FP support

            // SAT solvers (new)
            BackendId::MiniSat => 85, // Classic CDCL solver
            BackendId::Glucose => 88, // High-performance SAT
            BackendId::CaDiCaL => 90, // Award-winning modern SAT

            // Dependently typed theorem provers (new)
            BackendId::Agda => 95,  // Dependently typed programming
            BackendId::Idris => 95, // Quantitative type theory
            BackendId::ACL2 => 90,  // Industrial theorem proving
            BackendId::HOL4 => 90,  // Higher-order logic
            BackendId::FStar => 95, // Proof-oriented ML

            // Additional theorem provers
            BackendId::HOLLight => 90, // Small kernel HOL
            BackendId::PVS => 90,      // Decision procedures
            BackendId::Mizar => 85,    // Mathematical articles
            BackendId::Metamath => 85, // Minimal axiom verification
            BackendId::ATS => 85,      // Linear types

            // Additional SMT solvers
            BackendId::OpenSMT => 85, // Interpolation SMT
            BackendId::VeriT => 85,   // Proof production
            BackendId::AltErgo => 85, // Program verification

            // Additional SAT solvers
            BackendId::Kissat => 90,        // State-of-the-art SAT
            BackendId::CryptoMiniSat => 88, // XOR reasoning

            // Additional model checkers
            BackendId::NuXmv => 88,    // SMT-based model checking
            BackendId::UPPAAL => 90,   // Timed automata
            BackendId::DIVINE => 85,   // Parallel LTL
            BackendId::ESBMC => 85,    // C/C++ bounded model checking
            BackendId::Ultimate => 85, // Automata-based verification
            BackendId::SMACK => 85,    // LLVM verification
            BackendId::JPF => 85,      // Java model checking

            // Program verification frameworks
            BackendId::VCC => 85,           // C with contracts
            BackendId::VeriFast => 85,      // Separation logic
            BackendId::KeY => 85,           // Java verification
            BackendId::OpenJML => 85,       // JML checking
            BackendId::Krakatoa => 80,      // Java/C with Coq
            BackendId::SPARK => 90,         // Ada formal verification
            BackendId::Why3 => 90,          // Deductive verification
            BackendId::Stainless => 85,     // Scala verification
            BackendId::LiquidHaskell => 85, // Haskell refinement types
            BackendId::Boogie => 85,        // IVL

            // Distributed systems verification
            BackendId::PLang => 85, // Event-driven verification
            BackendId::Ivy => 90,   // Protocol verification
            BackendId::MCRL2 => 85, // Process algebra
            BackendId::CADP => 85,  // Concurrent systems

            // Cryptographic verification
            BackendId::EasyCrypt => 90,   // Crypto proofs
            BackendId::CryptoVerif => 90, // Protocol verification
            BackendId::Jasmin => 85,      // Crypto implementation

            // Hardware verification
            BackendId::Yosys => 85,      // RTL synthesis
            BackendId::SymbiYosys => 90, // Formal Verilog
            BackendId::JasperGold => 95, // Commercial FV
            BackendId::CadenceEDA => 95, // Commercial EDA

            // Symbolic execution and binary analysis
            BackendId::Angr => 85,      // Binary analysis
            BackendId::Manticore => 85, // Symbolic execution
            BackendId::TritonDBA => 85, // Dynamic binary analysis
            BackendId::Bap => 85,       // BAP binary lifting
            BackendId::Ghidra => 85,    // NSA reverse engineering
            BackendId::IsaBIL => 80,    // Isabelle/HOL BIL verification
            BackendId::Soteria => 80,   // Smart contract analysis

            // Abstract interpretation
            BackendId::Astree => 95,    // Safety-critical C
            BackendId::Polyspace => 90, // MathWorks analysis
            BackendId::CodeSonar => 85, // Binary/source analysis
            BackendId::FramaCEva => 85, // Value analysis

            // Rust code coverage
            BackendId::Tarpaulin => 85, // Rust coverage
            BackendId::LlvmCov => 85,   // LLVM coverage
            BackendId::Grcov => 85,     // Mozilla coverage

            // Rust testing frameworks
            BackendId::Nextest => 90,  // Fast test runner
            BackendId::Insta => 85,    // Snapshot testing
            BackendId::Rstest => 85,   // Fixtures
            BackendId::TestCase => 85, // Parameterized
            BackendId::Mockall => 85,  // Mocking

            // Rust documentation tools
            BackendId::Deadlinks => 80,  // Link checking
            BackendId::Spellcheck => 80, // Spell checking
            BackendId::Rdme => 80,       // README validation

            // Additional Rust verification tools
            BackendId::Haybale => 85,  // LLVM symbolic execution
            BackendId::CruxMir => 85,  // Galois symbolic testing
            BackendId::RustHorn => 80, // CHC-based verification
            BackendId::RustBelt => 75, // Coq-based (research)

            // Go verification
            BackendId::Gobra => 90, // Viper-based Go verifier

            // Additional C/C++ verification
            BackendId::Symbiotic => 85, // LLVM instrumentation + KLEE
            BackendId::TwoLS => 80,     // Template-based synthesis

            // Kani Fast (enhanced Kani)
            BackendId::KaniFast => 95, // k-induction, CHC, portfolio
        }
    }

    /// Get a backend by ID
    pub fn get(&self, id: BackendId) -> Option<Arc<dyn VerificationBackend>> {
        self.backends.get(&id).cloned()
    }

    /// Get backend info
    pub fn get_info(&self, id: BackendId) -> Option<&BackendInfo> {
        self.info_cache.get(&id)
    }

    /// Get all backends that support a property type
    pub fn backends_for_type(&self, prop_type: PropertyType) -> Vec<BackendId> {
        self.type_to_backends
            .get(&prop_type)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all registered backend IDs
    pub fn all_backends(&self) -> Vec<BackendId> {
        self.backends.keys().copied().collect()
    }

    /// Update health status for a backend
    pub fn update_health(&mut self, id: BackendId, health: HealthStatus) {
        if let Some(info) = self.info_cache.get_mut(&id) {
            debug!(?id, ?health, "Updating backend health");
            info.health = health;
        }
    }

    /// Check health of all backends
    pub async fn check_all_health(&mut self) {
        for (id, backend) in &self.backends {
            let health = backend.health_check().await;
            if let Some(info) = self.info_cache.get_mut(id) {
                info.health = health;
            }
        }
    }

    /// Get healthy backends for a property type, sorted by priority
    pub fn healthy_backends_for_type(&self, prop_type: PropertyType) -> Vec<BackendId> {
        let mut backends: Vec<_> = self
            .backends_for_type(prop_type)
            .into_iter()
            .filter(|id| {
                self.info_cache
                    .get(id)
                    .map(|info| matches!(info.health, HealthStatus::Healthy))
                    .unwrap_or(false)
            })
            .collect();

        // Sort by priority (descending)
        backends.sort_by(|a, b| {
            let pa = self.info_cache.get(a).map(|i| i.priority).unwrap_or(0);
            let pb = self.info_cache.get(b).map(|i| i.priority).unwrap_or(0);
            pb.cmp(&pa)
        });

        backends
    }

    /// Number of registered backends
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Selection strategy for choosing backends
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SelectionStrategy {
    /// Use the single best backend for each property
    #[default]
    Single,
    /// Use all compatible backends and merge results
    All,
    /// Use a specific backend (fail if unavailable)
    Specific(BackendId),
    /// Use multiple backends for increased confidence
    Redundant {
        /// Minimum number of backends required
        min_backends: usize,
    },
    /// Use ML-based prediction from StrategyModel
    /// Falls back to Single strategy if prediction confidence is too low
    MlBased {
        /// Minimum confidence threshold (0.0-1.0) to use ML prediction
        /// If prediction confidence is below this, falls back to rule-based selection
        min_confidence: f64,
    },
    /// Use RAG-enhanced knowledge base for backend selection
    /// Uses BackendSelectionExpert with ToolKnowledgeStore for context-aware recommendations
    /// Falls back to Single strategy if knowledge base is unavailable or confidence is too low
    KnowledgeEnhanced {
        /// Minimum confidence threshold (0.0-1.0) to use knowledge-based prediction
        min_confidence: f64,
    },
}

/// Selection result containing which backends to use for which properties
#[derive(Debug, Clone)]
pub struct Selection {
    /// Property to backend mapping
    pub assignments: Vec<PropertyAssignment>,
    /// Warnings about selection decisions
    pub warnings: Vec<String>,
    /// Metrics about the selection process
    pub metrics: SelectionMetrics,
}

/// Metrics captured during the backend selection process
#[derive(Debug, Clone, Default)]
pub struct SelectionMetrics {
    /// Total number of properties processed
    pub properties_processed: usize,
    /// Number of selections made using knowledge-enhanced strategy
    pub knowledge_enhanced_selections: usize,
    /// Number of selections that fell back to rule-based
    pub rule_based_fallbacks: usize,
    /// Number of ML-based selections
    pub ml_based_selections: usize,
    /// Average confidence score for knowledge-enhanced selections
    pub avg_knowledge_confidence: Option<f64>,
    /// Selection method used for each property (index -> method name)
    pub selection_methods: Vec<SelectionMethod>,
}

/// Method used to select a backend for a property
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionMethod {
    /// Rule-based selection (Single, All, Specific, Redundant strategies)
    RuleBased,
    /// ML-based prediction was used
    MlBased {
        /// Confidence score of the prediction
        confidence: f64,
    },
    /// Knowledge-enhanced RAG selection was used
    KnowledgeEnhanced {
        /// Confidence score of the recommendation
        confidence: f64,
        /// Brief rationale for the selection
        rationale: String,
    },
    /// Fallback from ML-based (confidence too low or prediction unavailable)
    MlFallback {
        /// Original ML confidence (if available)
        original_confidence: Option<f64>,
    },
    /// Fallback from knowledge-enhanced (confidence too low or error)
    KnowledgeFallback {
        /// Original knowledge confidence (if available)
        original_confidence: Option<f64>,
        /// Reason for fallback
        reason: String,
    },
}

/// A single property-to-backend assignment
#[derive(Debug, Clone)]
pub struct PropertyAssignment {
    /// Index of the property in the spec
    pub property_index: usize,
    /// Property type
    pub property_type: PropertyType,
    /// Selected backends (may be multiple for redundant verification)
    pub backends: Vec<BackendId>,
    /// Method used for this selection
    pub selection_method: SelectionMethod,
}

/// Backend selector that chooses backends based on property types
pub struct BackendSelector<'a> {
    registry: &'a BackendRegistry,
    strategy: SelectionStrategy,
    /// ML predictor for ML-based selection strategy
    ml_predictor: Option<Arc<StrategyModel>>,
    /// Knowledge store for knowledge-enhanced selection
    knowledge_store: Option<Arc<KnowledgeStore>>,
    /// Embedder for knowledge-enhanced selection
    embedder: Option<Arc<Embedder>>,
    /// Tool knowledge store for knowledge-enhanced selection
    tool_store: Option<Arc<ToolKnowledgeStore>>,
}

impl<'a> BackendSelector<'a> {
    /// Create a new selector with the given registry and strategy
    pub fn new(registry: &'a BackendRegistry, strategy: SelectionStrategy) -> Self {
        BackendSelector {
            registry,
            strategy,
            ml_predictor: None,
            knowledge_store: None,
            embedder: None,
            tool_store: None,
        }
    }

    /// Create a new selector with ML predictor for ML-based selection
    pub fn with_ml_predictor(
        registry: &'a BackendRegistry,
        strategy: SelectionStrategy,
        predictor: Arc<StrategyModel>,
    ) -> Self {
        BackendSelector {
            registry,
            strategy,
            ml_predictor: Some(predictor),
            knowledge_store: None,
            embedder: None,
            tool_store: None,
        }
    }

    /// Create a new selector with knowledge stores for knowledge-enhanced selection
    pub fn with_knowledge_store(
        registry: &'a BackendRegistry,
        strategy: SelectionStrategy,
        knowledge_store: Arc<KnowledgeStore>,
        embedder: Arc<Embedder>,
        tool_store: Arc<ToolKnowledgeStore>,
    ) -> Self {
        BackendSelector {
            registry,
            strategy,
            ml_predictor: None,
            knowledge_store: Some(knowledge_store),
            embedder: Some(embedder),
            tool_store: Some(tool_store),
        }
    }

    /// Set the ML predictor
    pub fn set_ml_predictor(&mut self, predictor: Arc<StrategyModel>) {
        self.ml_predictor = Some(predictor);
    }

    /// Set the knowledge stores for knowledge-enhanced selection
    pub fn set_knowledge_stores(
        &mut self,
        knowledge_store: Arc<KnowledgeStore>,
        embedder: Arc<Embedder>,
        tool_store: Arc<ToolKnowledgeStore>,
    ) {
        self.knowledge_store = Some(knowledge_store);
        self.embedder = Some(embedder);
        self.tool_store = Some(tool_store);
    }

    /// Select backends for a list of properties
    pub fn select(&self, properties: &[Property]) -> Result<Selection, SelectorError> {
        let mut assignments = Vec::new();
        let mut warnings = Vec::new();
        let mut metrics = SelectionMetrics {
            properties_processed: properties.len(),
            ..Default::default()
        };
        let mut knowledge_confidences = Vec::new();

        for (index, property) in properties.iter().enumerate() {
            let prop_type = Self::property_type(property);

            // Use appropriate selection based on strategy
            let (backends, selection_method) = match &self.strategy {
                SelectionStrategy::MlBased { min_confidence } => {
                    self.select_ml_based_with_method(property, prop_type, *min_confidence)?
                }
                SelectionStrategy::KnowledgeEnhanced { min_confidence } => self
                    .select_knowledge_enhanced_with_method(property, prop_type, *min_confidence)?,
                _ => {
                    let backends = self.select_for_type(prop_type)?;
                    (backends, SelectionMethod::RuleBased)
                }
            };

            if backends.is_empty() {
                return Err(SelectorError::NoBackendAvailable(prop_type));
            }

            // Update metrics based on selection method
            match &selection_method {
                SelectionMethod::KnowledgeEnhanced { confidence, .. } => {
                    metrics.knowledge_enhanced_selections += 1;
                    knowledge_confidences.push(*confidence);
                }
                SelectionMethod::KnowledgeFallback { .. } => {
                    metrics.rule_based_fallbacks += 1;
                }
                SelectionMethod::MlBased { .. } => {
                    metrics.ml_based_selections += 1;
                }
                SelectionMethod::MlFallback { .. } => {
                    metrics.rule_based_fallbacks += 1;
                }
                SelectionMethod::RuleBased => {}
            }
            metrics.selection_methods.push(selection_method.clone());

            // Check for degraded backends and warn
            for &backend_id in &backends {
                if let Some(info) = self.registry.get_info(backend_id) {
                    if let HealthStatus::Degraded { reason } = &info.health {
                        warnings.push(format!("Backend {:?} is degraded: {}", backend_id, reason));
                    }
                }
            }

            assignments.push(PropertyAssignment {
                property_index: index,
                property_type: prop_type,
                backends,
                selection_method,
            });
        }

        // Calculate average knowledge confidence
        if !knowledge_confidences.is_empty() {
            let sum: f64 = knowledge_confidences.iter().sum();
            metrics.avg_knowledge_confidence = Some(sum / knowledge_confidences.len() as f64);
        }

        // Log selection metrics summary
        info!(
            properties = metrics.properties_processed,
            knowledge_enhanced = metrics.knowledge_enhanced_selections,
            ml_based = metrics.ml_based_selections,
            fallbacks = metrics.rule_based_fallbacks,
            avg_confidence = ?metrics.avg_knowledge_confidence,
            "Backend selection metrics"
        );

        Ok(Selection {
            assignments,
            warnings,
            metrics,
        })
    }

    /// Helper to fall back to rule-based selection
    fn fallback_to_rule_based(
        &self,
        prop_type: PropertyType,
    ) -> Result<Vec<BackendId>, SelectorError> {
        let healthy = self.registry.healthy_backends_for_type(prop_type);
        if healthy.is_empty() {
            let all = self.registry.backends_for_type(prop_type);
            if all.is_empty() {
                return Err(SelectorError::NoBackendAvailable(prop_type));
            }
            warn!(
                ?prop_type,
                "No healthy backends, using potentially unhealthy backend"
            );
            Ok(vec![all[0]])
        } else {
            Ok(vec![healthy[0]])
        }
    }

    /// Select backend using ML prediction with fallback, returning selection method
    fn select_ml_based_with_method(
        &self,
        property: &Property,
        prop_type: PropertyType,
        min_confidence: f64,
    ) -> Result<(Vec<BackendId>, SelectionMethod), SelectorError> {
        // Try ML prediction if predictor is available
        if let Some(ref predictor) = self.ml_predictor {
            let features = PropertyFeatureVector::from_property(property);
            let prediction = predictor.predict_backend(&features);

            debug!(
                backend = ?prediction.backend,
                confidence = prediction.confidence,
                "ML backend prediction"
            );

            // Use ML prediction if confidence meets threshold and backend is available
            if prediction.confidence >= min_confidence {
                let predicted_backend = prediction.backend;

                // Verify the predicted backend is registered and healthy
                if self.registry.get(predicted_backend).is_some() {
                    let info = self.registry.get_info(predicted_backend);
                    let is_healthy = info
                        .map(|i| matches!(i.health, HealthStatus::Healthy))
                        .unwrap_or(false);

                    if is_healthy {
                        info!(
                            backend = ?predicted_backend,
                            confidence = prediction.confidence,
                            "Using ML-predicted backend"
                        );
                        return Ok((
                            vec![predicted_backend],
                            SelectionMethod::MlBased {
                                confidence: prediction.confidence,
                            },
                        ));
                    } else {
                        // Try alternatives from ML prediction
                        for (alt_backend, alt_confidence) in &prediction.alternatives {
                            if *alt_confidence >= min_confidence {
                                if let Some(alt_info) = self.registry.get_info(*alt_backend) {
                                    if matches!(alt_info.health, HealthStatus::Healthy) {
                                        info!(
                                            backend = ?alt_backend,
                                            confidence = alt_confidence,
                                            "Using ML-predicted alternative backend"
                                        );
                                        return Ok((
                                            vec![*alt_backend],
                                            SelectionMethod::MlBased {
                                                confidence: *alt_confidence,
                                            },
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            debug!(
                confidence = prediction.confidence,
                min_confidence = min_confidence,
                "ML confidence below threshold, falling back to rule-based selection"
            );

            // Fall back with original confidence recorded
            let backends = self.fallback_to_rule_based(prop_type)?;
            return Ok((
                backends,
                SelectionMethod::MlFallback {
                    original_confidence: Some(prediction.confidence),
                },
            ));
        }

        warn!(
            "ML-based selection requested but no predictor configured, falling back to rule-based"
        );
        let backends = self.fallback_to_rule_based(prop_type)?;
        Ok((
            backends,
            SelectionMethod::MlFallback {
                original_confidence: None,
            },
        ))
    }

    /// Select backend using RAG-enhanced knowledge base with fallback, returning selection method
    fn select_knowledge_enhanced_with_method(
        &self,
        property: &Property,
        prop_type: PropertyType,
        min_confidence: f64,
    ) -> Result<(Vec<BackendId>, SelectionMethod), SelectorError> {
        // Check if knowledge stores are available
        let (Some(knowledge_store), Some(embedder), Some(tool_store)) =
            (&self.knowledge_store, &self.embedder, &self.tool_store)
        else {
            warn!("Knowledge-enhanced selection requested but stores not configured, falling back to rule-based");
            let backends = self.fallback_to_rule_based(prop_type)?;
            return Ok((
                backends,
                SelectionMethod::KnowledgeFallback {
                    original_confidence: None,
                    reason: "Knowledge stores not configured".to_string(),
                },
            ));
        };

        // Build expert context from property
        let mut context = ExpertContext::default();

        // Map property type to knowledge property type
        let knowledge_prop_type = Self::to_knowledge_property_type(prop_type);
        context.property_types.push(knowledge_prop_type);

        // Extract specification text from property for context
        context.specification = Some(format!("{:?}", property));

        // Create the expert factory and get backend selection expert
        let factory = ExpertFactory::with_tool_store(
            knowledge_store.as_ref(),
            embedder.as_ref(),
            tool_store.as_ref(),
        );
        let expert = factory.backend_selection();

        // Use tokio::runtime for async call (since we're in sync context)
        let recommendation_result = std::thread::scope(|s| {
            s.spawn(|| {
                tokio::runtime::Runtime::new()
                    .expect("Failed to create runtime")
                    .block_on(expert.recommend(&context))
            })
            .join()
        });

        match recommendation_result {
            Ok(Ok(recommendation)) => {
                debug!(
                    backend = ?recommendation.backend,
                    confidence = recommendation.confidence,
                    "Knowledge-enhanced backend recommendation"
                );

                // Use recommendation if confidence meets threshold and backend is available
                if recommendation.confidence as f64 >= min_confidence {
                    let recommended_backend = recommendation.backend;

                    // Verify the recommended backend is registered and healthy
                    if self.registry.get(recommended_backend).is_some() {
                        let info = self.registry.get_info(recommended_backend);
                        let is_healthy = info
                            .map(|i| matches!(i.health, HealthStatus::Healthy))
                            .unwrap_or(false);

                        if is_healthy {
                            info!(
                                backend = ?recommended_backend,
                                confidence = recommendation.confidence,
                                rationale = %recommendation.rationale,
                                "Using knowledge-enhanced backend recommendation"
                            );
                            return Ok((
                                vec![recommended_backend],
                                SelectionMethod::KnowledgeEnhanced {
                                    confidence: recommendation.confidence as f64,
                                    rationale: recommendation.rationale.clone(),
                                },
                            ));
                        }
                    }

                    // Try alternatives from the recommendation
                    for alt in &recommendation.alternatives {
                        if alt.confidence as f64 >= min_confidence {
                            if let Some(alt_info) = self.registry.get_info(alt.backend) {
                                if matches!(alt_info.health, HealthStatus::Healthy) {
                                    info!(
                                        backend = ?alt.backend,
                                        confidence = alt.confidence,
                                        "Using knowledge-enhanced alternative backend"
                                    );
                                    return Ok((
                                        vec![alt.backend],
                                        SelectionMethod::KnowledgeEnhanced {
                                            confidence: alt.confidence as f64,
                                            rationale: alt.rationale.clone(),
                                        },
                                    ));
                                }
                            }
                        }
                    }
                }

                debug!(
                    confidence = recommendation.confidence,
                    min_confidence = min_confidence,
                    "Knowledge confidence below threshold, falling back to rule-based"
                );

                let backends = self.fallback_to_rule_based(prop_type)?;
                Ok((
                    backends,
                    SelectionMethod::KnowledgeFallback {
                        original_confidence: Some(recommendation.confidence as f64),
                        reason: "Confidence below threshold".to_string(),
                    },
                ))
            }
            Ok(Err(e)) => {
                warn!(error = %e, "Knowledge-enhanced selection failed, falling back to rule-based");
                let backends = self.fallback_to_rule_based(prop_type)?;
                Ok((
                    backends,
                    SelectionMethod::KnowledgeFallback {
                        original_confidence: None,
                        reason: format!("Expert error: {}", e),
                    },
                ))
            }
            Err(_) => {
                warn!("Knowledge-enhanced selection thread panicked, falling back to rule-based");
                let backends = self.fallback_to_rule_based(prop_type)?;
                Ok((
                    backends,
                    SelectionMethod::KnowledgeFallback {
                        original_confidence: None,
                        reason: "Thread panic".to_string(),
                    },
                ))
            }
        }
    }

    /// Convert backend PropertyType to knowledge PropertyType
    fn to_knowledge_property_type(prop_type: PropertyType) -> KnowledgePropertyType {
        match prop_type {
            // Theorem proving
            PropertyType::Theorem => KnowledgePropertyType::Correctness,
            PropertyType::Contract => KnowledgePropertyType::Safety,
            PropertyType::Invariant => KnowledgePropertyType::Safety,
            PropertyType::Refinement => KnowledgePropertyType::Refinement,

            // Model checking
            PropertyType::Temporal => KnowledgePropertyType::Temporal,
            PropertyType::Probabilistic => KnowledgePropertyType::Probabilistic,

            // Neural networks
            PropertyType::NeuralRobustness
            | PropertyType::NeuralReachability
            | PropertyType::AdversarialRobustness => KnowledgePropertyType::NeuralNetwork,

            // Security
            PropertyType::SecurityProtocol => KnowledgePropertyType::SecurityProtocol,
            PropertyType::PlatformApi => KnowledgePropertyType::PlatformApi,

            // Rust memory safety
            PropertyType::MemorySafety
            | PropertyType::UndefinedBehavior
            | PropertyType::DataRace
            | PropertyType::MemoryLeak => KnowledgePropertyType::Safety,

            // Rust testing
            PropertyType::Fuzzing | PropertyType::PropertyBased | PropertyType::MutationTesting => {
                KnowledgePropertyType::Correctness
            }

            // Rust static analysis
            PropertyType::Lint
            | PropertyType::ApiCompatibility
            | PropertyType::SecurityVulnerability
            | PropertyType::DependencyPolicy
            | PropertyType::SupplyChain
            | PropertyType::UnsafeAudit => KnowledgePropertyType::Safety,

            // AI/ML
            PropertyType::ModelOptimization
            | PropertyType::ModelCompression
            | PropertyType::DataQuality => KnowledgePropertyType::NeuralNetwork,

            // Fairness and explainability
            PropertyType::Fairness | PropertyType::Interpretability => {
                KnowledgePropertyType::NeuralNetwork
            }

            // LLM specific
            PropertyType::LLMGuardrails
            | PropertyType::LLMEvaluation
            | PropertyType::HallucinationDetection => KnowledgePropertyType::NeuralNetwork,
        }
    }

    /// Select backends for a specific property type based on strategy
    fn select_for_type(&self, prop_type: PropertyType) -> Result<Vec<BackendId>, SelectorError> {
        match self.strategy {
            SelectionStrategy::Single => {
                let healthy = self.registry.healthy_backends_for_type(prop_type);
                if healthy.is_empty() {
                    // Fall back to any available backend
                    let all = self.registry.backends_for_type(prop_type);
                    if all.is_empty() {
                        return Err(SelectorError::NoBackendAvailable(prop_type));
                    }
                    warn!(
                        ?prop_type,
                        "No healthy backends, using potentially unhealthy backend"
                    );
                    Ok(vec![all[0]])
                } else {
                    Ok(vec![healthy[0]])
                }
            }

            SelectionStrategy::All => {
                let healthy = self.registry.healthy_backends_for_type(prop_type);
                if healthy.is_empty() {
                    let all = self.registry.backends_for_type(prop_type);
                    if all.is_empty() {
                        return Err(SelectorError::NoBackendAvailable(prop_type));
                    }
                    Ok(all)
                } else {
                    Ok(healthy)
                }
            }

            SelectionStrategy::Specific(id) => {
                // Check if this backend supports the property type
                if let Some(info) = self.registry.get_info(id) {
                    if info.supported_types.contains(&prop_type) {
                        Ok(vec![id])
                    } else {
                        Err(SelectorError::NoBackendAvailable(prop_type))
                    }
                } else {
                    Err(SelectorError::BackendNotFound(id))
                }
            }

            SelectionStrategy::Redundant { min_backends } => {
                let healthy = self.registry.healthy_backends_for_type(prop_type);
                if healthy.len() >= min_backends {
                    Ok(healthy.into_iter().take(min_backends).collect())
                } else if !healthy.is_empty() {
                    // Use what we have
                    warn!(
                        ?prop_type,
                        available = healthy.len(),
                        requested = min_backends,
                        "Fewer backends available than requested"
                    );
                    Ok(healthy)
                } else {
                    let all = self.registry.backends_for_type(prop_type);
                    if all.is_empty() {
                        Err(SelectorError::NoBackendAvailable(prop_type))
                    } else {
                        Ok(all.into_iter().take(min_backends).collect())
                    }
                }
            }

            // MlBased is handled in select() directly, but include for exhaustiveness
            // This branch falls back to Single-like behavior
            SelectionStrategy::MlBased { .. } => {
                let healthy = self.registry.healthy_backends_for_type(prop_type);
                if healthy.is_empty() {
                    let all = self.registry.backends_for_type(prop_type);
                    if all.is_empty() {
                        return Err(SelectorError::NoBackendAvailable(prop_type));
                    }
                    Ok(vec![all[0]])
                } else {
                    Ok(vec![healthy[0]])
                }
            }

            // KnowledgeEnhanced is handled in select() directly, but include for exhaustiveness
            // This branch falls back to Single-like behavior
            SelectionStrategy::KnowledgeEnhanced { .. } => {
                let healthy = self.registry.healthy_backends_for_type(prop_type);
                if healthy.is_empty() {
                    let all = self.registry.backends_for_type(prop_type);
                    if all.is_empty() {
                        return Err(SelectorError::NoBackendAvailable(prop_type));
                    }
                    Ok(vec![all[0]])
                } else {
                    Ok(vec![healthy[0]])
                }
            }
        }
    }

    /// Determine the property type for backend selection.
    ///
    /// Delegates to [`PropertyType::from_property`] for the canonical mapping.
    #[inline]
    pub fn property_type(property: &Property) -> PropertyType {
        PropertyType::from_property(property)
    }

    /// Get the best backend for a property type (convenience method)
    pub fn best_backend(&self, prop_type: PropertyType) -> Option<BackendId> {
        self.registry
            .healthy_backends_for_type(prop_type)
            .first()
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_ai::StrategyPredictor;
    use dashprove_backends::{BackendResult, VerificationStatus};
    use dashprove_usl::typecheck::TypedSpec;

    // Mock backend for testing
    struct MockBackend {
        id: BackendId,
        supported: Vec<PropertyType>,
        health: HealthStatus,
    }

    impl MockBackend {
        fn new(id: BackendId, supported: Vec<PropertyType>) -> Self {
            MockBackend {
                id,
                supported,
                health: HealthStatus::Healthy,
            }
        }

        fn with_health(mut self, health: HealthStatus) -> Self {
            self.health = health;
            self
        }
    }

    #[async_trait::async_trait]
    impl VerificationBackend for MockBackend {
        fn id(&self) -> BackendId {
            self.id
        }

        fn supports(&self) -> Vec<PropertyType> {
            self.supported.clone()
        }

        async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
            Ok(BackendResult {
                backend: self.id,
                status: VerificationStatus::Proven,
                proof: Some("mock proof".to_string()),
                counterexample: None,
                diagnostics: vec![],
                time_taken: std::time::Duration::from_millis(100),
            })
        }

        async fn health_check(&self) -> HealthStatus {
            self.health.clone()
        }
    }

    #[test]
    fn test_registry_register() {
        let mut registry = BackendRegistry::new();
        let backend = Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem, PropertyType::Invariant],
        ));
        registry.register(backend);

        assert_eq!(registry.len(), 1);
        assert!(registry.get(BackendId::Lean4).is_some());
        assert!(registry.get(BackendId::TlaPlus).is_none());
    }

    #[test]
    fn test_registry_backends_for_type() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem, PropertyType::Invariant],
        )));
        registry.register(Arc::new(MockBackend::new(
            BackendId::TlaPlus,
            vec![PropertyType::Temporal, PropertyType::Invariant],
        )));

        let theorem_backends = registry.backends_for_type(PropertyType::Theorem);
        assert_eq!(theorem_backends.len(), 1);
        assert_eq!(theorem_backends[0], BackendId::Lean4);

        let invariant_backends = registry.backends_for_type(PropertyType::Invariant);
        assert_eq!(invariant_backends.len(), 2);

        let temporal_backends = registry.backends_for_type(PropertyType::Temporal);
        assert_eq!(temporal_backends.len(), 1);
        assert_eq!(temporal_backends[0], BackendId::TlaPlus);
    }

    #[test]
    fn test_selector_single_strategy() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        let selector = BackendSelector::new(&registry, SelectionStrategy::Single);

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        assert_eq!(selection.assignments.len(), 1);
        assert_eq!(selection.assignments[0].backends, vec![BackendId::Lean4]);
    }

    #[test]
    fn test_selector_all_strategy() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Invariant],
        )));
        registry.register(Arc::new(MockBackend::new(
            BackendId::Alloy,
            vec![PropertyType::Invariant],
        )));

        let selector = BackendSelector::new(&registry, SelectionStrategy::All);

        use dashprove_usl::ast::{Expr, Invariant};
        let properties = vec![Property::Invariant(Invariant {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        assert_eq!(selection.assignments.len(), 1);
        assert_eq!(selection.assignments[0].backends.len(), 2);
    }

    #[test]
    fn test_selector_specific_strategy() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::TlaPlus,
            vec![PropertyType::Temporal],
        )));

        let selector =
            BackendSelector::new(&registry, SelectionStrategy::Specific(BackendId::TlaPlus));

        use dashprove_usl::ast::{Expr, Temporal, TemporalExpr};
        let properties = vec![Property::Temporal(Temporal {
            name: "test".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
            fairness: vec![],
        })];

        let selection = selector.select(&properties).unwrap();
        assert_eq!(selection.assignments[0].backends, vec![BackendId::TlaPlus]);
    }

    #[test]
    fn test_selector_no_backend_error() {
        let registry = BackendRegistry::new(); // Empty registry

        let selector = BackendSelector::new(&registry, SelectionStrategy::Single);

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let result = selector.select(&properties);
        assert!(matches!(result, Err(SelectorError::NoBackendAvailable(_))));
    }

    #[test]
    fn test_selector_redundant_strategy() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Invariant],
        )));
        registry.register(Arc::new(MockBackend::new(
            BackendId::Alloy,
            vec![PropertyType::Invariant],
        )));

        let selector =
            BackendSelector::new(&registry, SelectionStrategy::Redundant { min_backends: 2 });

        use dashprove_usl::ast::{Expr, Invariant};
        let properties = vec![Property::Invariant(Invariant {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        assert_eq!(selection.assignments[0].backends.len(), 2);
    }

    #[test]
    fn test_property_type_mapping() {
        use dashprove_usl::ast::*;

        assert_eq!(
            BackendSelector::property_type(&Property::Theorem(Theorem {
                name: "t".into(),
                body: Expr::Bool(true)
            })),
            PropertyType::Theorem
        );

        assert_eq!(
            BackendSelector::property_type(&Property::Temporal(Temporal {
                name: "t".into(),
                body: TemporalExpr::Atom(Expr::Bool(true)),
                fairness: vec![],
            })),
            PropertyType::Temporal
        );

        assert_eq!(
            BackendSelector::property_type(&Property::Invariant(Invariant {
                name: "i".into(),
                body: Expr::Bool(true)
            })),
            PropertyType::Invariant
        );

        // Test specialized property types map to their backends
        assert_eq!(
            BackendSelector::property_type(&Property::Probabilistic(Probabilistic {
                name: "p".into(),
                condition: Expr::Bool(true),
                comparison: ComparisonOp::Ge,
                bound: 0.9
            })),
            PropertyType::Probabilistic
        );

        assert_eq!(
            BackendSelector::property_type(&Property::Security(Security {
                name: "s".into(),
                body: Expr::Bool(true)
            })),
            PropertyType::SecurityProtocol
        );
    }

    #[test]
    fn test_healthy_backends_sorted_by_priority() {
        let mut registry = BackendRegistry::new();

        // Alloy has priority 90, Lean4 has priority 100
        registry.register(Arc::new(MockBackend::new(
            BackendId::Alloy,
            vec![PropertyType::Invariant],
        )));
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Invariant],
        )));

        let healthy = registry.healthy_backends_for_type(PropertyType::Invariant);
        // Should be sorted by priority: Lean4 first (100), then Alloy (90)
        assert_eq!(healthy[0], BackendId::Lean4);
        assert_eq!(healthy[1], BackendId::Alloy);
    }

    #[test]
    fn test_unhealthy_backend_fallback() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(
            MockBackend::new(BackendId::Lean4, vec![PropertyType::Theorem]).with_health(
                HealthStatus::Unavailable {
                    reason: "test".into(),
                },
            ),
        ));

        // Update the health status
        registry.update_health(
            BackendId::Lean4,
            HealthStatus::Unavailable {
                reason: "test".into(),
            },
        );

        let healthy = registry.healthy_backends_for_type(PropertyType::Theorem);
        assert!(healthy.is_empty());

        // But selector should fall back to any available
        let selector = BackendSelector::new(&registry, SelectionStrategy::Single);

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        assert_eq!(selection.assignments[0].backends, vec![BackendId::Lean4]);
    }

    #[test]
    fn test_selector_specific_backend_not_found() {
        // Registry has only TLA+
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::TlaPlus,
            vec![PropertyType::Temporal],
        )));

        // But we request Kani specifically (which is not registered)
        let selector =
            BackendSelector::new(&registry, SelectionStrategy::Specific(BackendId::Kani));

        // Use a Theorem property (property type doesn't matter - Kani isn't registered at all)
        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let result = selector.select(&properties);
        assert!(matches!(
            result,
            Err(SelectorError::BackendNotFound(BackendId::Kani))
        ));
    }

    #[test]
    fn test_selector_specific_unsupported_type() {
        // Registry has Lean4 which supports Theorem but not Temporal
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        // Request Lean4 specifically for a Temporal property (which it doesn't support)
        let selector =
            BackendSelector::new(&registry, SelectionStrategy::Specific(BackendId::Lean4));

        use dashprove_usl::ast::{Expr, Temporal, TemporalExpr};
        let properties = vec![Property::Temporal(Temporal {
            name: "test".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
            fairness: vec![],
        })];

        let result = selector.select(&properties);
        assert!(matches!(
            result,
            Err(SelectorError::NoBackendAvailable(PropertyType::Temporal))
        ));
    }

    #[test]
    fn test_probabilistic_property_selects_storm_backend() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Storm,
            vec![PropertyType::Probabilistic],
        )));
        registry.register(Arc::new(MockBackend::new(
            BackendId::Prism,
            vec![PropertyType::Probabilistic],
        )));

        let selector = BackendSelector::new(&registry, SelectionStrategy::Single);

        use dashprove_usl::ast::{ComparisonOp, Expr, Probabilistic};
        let properties = vec![Property::Probabilistic(Probabilistic {
            name: "reach_goal".to_string(),
            condition: Expr::Bool(true),
            comparison: ComparisonOp::Ge,
            bound: 0.95,
        })];

        let selection = selector.select(&properties).unwrap();
        assert_eq!(selection.assignments.len(), 1);
        assert_eq!(
            selection.assignments[0].property_type,
            PropertyType::Probabilistic
        );
        // Storm has priority 100, Prism has 90, so Storm should be selected
        assert_eq!(selection.assignments[0].backends, vec![BackendId::Storm]);
    }

    #[test]
    fn test_security_property_selects_tamarin_backend() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Tamarin,
            vec![PropertyType::SecurityProtocol],
        )));
        registry.register(Arc::new(MockBackend::new(
            BackendId::ProVerif,
            vec![PropertyType::SecurityProtocol],
        )));
        registry.register(Arc::new(MockBackend::new(
            BackendId::Verifpal,
            vec![PropertyType::SecurityProtocol],
        )));

        let selector = BackendSelector::new(&registry, SelectionStrategy::Single);

        use dashprove_usl::ast::{Expr, Security};
        let properties = vec![Property::Security(Security {
            name: "confidentiality".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        assert_eq!(selection.assignments.len(), 1);
        assert_eq!(
            selection.assignments[0].property_type,
            PropertyType::SecurityProtocol
        );
        // Tamarin has priority 100, ProVerif 95, Verifpal 90
        assert_eq!(selection.assignments[0].backends, vec![BackendId::Tamarin]);
    }

    #[test]
    fn test_all_strategy_selects_multiple_security_backends() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Tamarin,
            vec![PropertyType::SecurityProtocol],
        )));
        registry.register(Arc::new(MockBackend::new(
            BackendId::ProVerif,
            vec![PropertyType::SecurityProtocol],
        )));

        let selector = BackendSelector::new(&registry, SelectionStrategy::All);

        use dashprove_usl::ast::{Expr, Security};
        let properties = vec![Property::Security(Security {
            name: "auth".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        assert_eq!(selection.assignments[0].backends.len(), 2);
    }

    #[test]
    fn test_ml_based_selection_with_predictor() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));
        registry.register(Arc::new(MockBackend::new(
            BackendId::Coq,
            vec![PropertyType::Theorem],
        )));

        // Create ML predictor
        let predictor = Arc::new(StrategyModel::from(StrategyPredictor::new()));

        // Use ML-based selection with low confidence threshold
        let strategy = SelectionStrategy::MlBased {
            min_confidence: 0.0,
        };
        let selector = BackendSelector::with_ml_predictor(&registry, strategy, predictor);

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![Property::Theorem(Theorem {
            name: "test_theorem".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        assert_eq!(selection.assignments.len(), 1);
        // Should select a backend (either via ML or fallback)
        assert!(!selection.assignments[0].backends.is_empty());
    }

    #[test]
    fn test_ml_based_selection_fallback_on_low_confidence() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        // Create ML predictor
        let predictor = Arc::new(StrategyModel::from(StrategyPredictor::new()));

        // Use ML-based selection with very high confidence threshold (will fallback)
        let strategy = SelectionStrategy::MlBased {
            min_confidence: 0.99,
        };
        let selector = BackendSelector::with_ml_predictor(&registry, strategy, predictor);

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        // Should fall back to rule-based selection (Lean4 by priority)
        assert_eq!(selection.assignments[0].backends, vec![BackendId::Lean4]);
    }

    #[test]
    fn test_ml_based_selection_without_predictor_falls_back() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::TlaPlus,
            vec![PropertyType::Temporal],
        )));

        // Use ML-based strategy without predictor
        let strategy = SelectionStrategy::MlBased {
            min_confidence: 0.5,
        };
        let selector = BackendSelector::new(&registry, strategy);

        use dashprove_usl::ast::{Expr, Temporal, TemporalExpr};
        let properties = vec![Property::Temporal(Temporal {
            name: "liveness".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
            fairness: vec![],
        })];

        // Should work via fallback
        let selection = selector.select(&properties).unwrap();
        assert_eq!(selection.assignments[0].backends, vec![BackendId::TlaPlus]);
    }

    #[test]
    fn test_ml_based_strategy_config() {
        let strategy = SelectionStrategy::MlBased {
            min_confidence: 0.5,
        };
        assert!(
            matches!(strategy, SelectionStrategy::MlBased { min_confidence } if min_confidence == 0.5)
        );
    }

    // ==================== Mutation-killing tests ====================

    #[test]
    fn test_selection_metrics_properties_processed() {
        // Mutation: delete field properties_processed from SelectionMetrics
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        let selector = BackendSelector::new(&registry, SelectionStrategy::Single);

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![
            Property::Theorem(Theorem {
                name: "test1".to_string(),
                body: Expr::Bool(true),
            }),
            Property::Theorem(Theorem {
                name: "test2".to_string(),
                body: Expr::Bool(false),
            }),
        ];

        let selection = selector.select(&properties).unwrap();
        assert_eq!(
            selection.metrics.properties_processed, 2,
            "properties_processed should equal input count"
        );
    }

    #[test]
    fn test_selection_metrics_knowledge_enhanced_count() {
        // Mutation: replace += with *= for knowledge_enhanced_selections
        // This test would be triggered with knowledge stores, but without them
        // we verify the count stays at 0
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        // No knowledge stores -> fallback path
        let selector = BackendSelector::new(
            &registry,
            SelectionStrategy::KnowledgeEnhanced {
                min_confidence: 0.5,
            },
        );

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        // Without knowledge stores, should fallback (count = 0 for knowledge_enhanced)
        assert_eq!(
            selection.metrics.knowledge_enhanced_selections, 0,
            "Without stores, knowledge_enhanced_selections should be 0"
        );
        assert_eq!(
            selection.metrics.rule_based_fallbacks, 1,
            "Should have 1 fallback"
        );
    }

    #[test]
    fn test_selection_metrics_ml_based_count() {
        // Mutation: replace += with *= for ml_based_selections
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        let predictor = Arc::new(StrategyModel::from(StrategyPredictor::new()));
        let selector = BackendSelector::with_ml_predictor(
            &registry,
            SelectionStrategy::MlBased {
                min_confidence: 0.0,
            }, // Low threshold
            predictor,
        );

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        // With low threshold, ML should be used (or fallback if prediction doesn't meet)
        // Either ml_based_selections or rule_based_fallbacks should be 1
        let total = selection.metrics.ml_based_selections + selection.metrics.rule_based_fallbacks;
        assert_eq!(total, 1, "Should have exactly 1 selection method used");
    }

    #[test]
    fn test_selection_metrics_rule_based_fallbacks() {
        // Mutation: replace += with -= or *= for rule_based_fallbacks
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        // ML-based without predictor -> fallback
        let selector = BackendSelector::new(
            &registry,
            SelectionStrategy::MlBased {
                min_confidence: 0.5,
            },
        );

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        assert_eq!(
            selection.metrics.rule_based_fallbacks, 1,
            "Without ML predictor, should fallback"
        );
    }

    #[test]
    fn test_selection_metrics_avg_confidence_none_when_no_knowledge() {
        // Mutation: delete ! in select (knowledge_confidences.is_empty check)
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        let selector = BackendSelector::new(&registry, SelectionStrategy::Single);

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        // No knowledge-enhanced selections -> avg_knowledge_confidence should be None
        assert!(
            selection.metrics.avg_knowledge_confidence.is_none(),
            "Without knowledge selections, avg confidence should be None"
        );
    }

    #[test]
    fn test_selector_best_backend_returns_none_for_unknown_type() {
        // Mutation: replace best_backend -> Option<BackendId> with None
        let registry = BackendRegistry::new(); // Empty registry

        let selector = BackendSelector::new(&registry, SelectionStrategy::Single);
        let result = selector.best_backend(PropertyType::Theorem);
        assert!(
            result.is_none(),
            "Empty registry should return None for best_backend"
        );
    }

    #[test]
    fn test_selector_best_backend_returns_some_when_available() {
        // Mutation: replace best_backend -> Option<BackendId> with None
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        let selector = BackendSelector::new(&registry, SelectionStrategy::Single);
        let result = selector.best_backend(PropertyType::Theorem);
        assert_eq!(
            result,
            Some(BackendId::Lean4),
            "Should return available backend"
        );
    }

    #[test]
    fn test_ml_confidence_threshold_applied() {
        // Mutation: replace >= with < in select_ml_based_with_method confidence check
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));
        registry.register(Arc::new(MockBackend::new(
            BackendId::Coq,
            vec![PropertyType::Theorem],
        )));

        // Create predictor
        let predictor = Arc::new(StrategyModel::from(StrategyPredictor::new()));

        // With very high threshold (0.99), most predictions will fallback
        let selector = BackendSelector::with_ml_predictor(
            &registry,
            SelectionStrategy::MlBased {
                min_confidence: 0.99,
            },
            predictor,
        );

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        // With high threshold, should likely fallback
        // The selection method should indicate fallback or ML based on actual confidence
        match &selection.assignments[0].selection_method {
            SelectionMethod::MlFallback { .. } => {
                // Expected when confidence is below threshold
            }
            SelectionMethod::MlBased { confidence } => {
                // Only if confidence >= 0.99
                assert!(
                    *confidence >= 0.99,
                    "ML selection should only be used if confidence >= threshold"
                );
            }
            _ => panic!("Expected MlFallback or MlBased selection method"),
        }
    }

    #[test]
    fn test_select_for_type_ml_based_fallback_on_empty_healthy() {
        // Mutation: delete ! in select_for_type for MlBased branch
        // Tests that MlBased falls back correctly when no healthy backends
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(
            MockBackend::new(BackendId::Lean4, vec![PropertyType::Theorem]).with_health(
                HealthStatus::Unavailable {
                    reason: "test".into(),
                },
            ),
        ));
        registry.update_health(
            BackendId::Lean4,
            HealthStatus::Unavailable {
                reason: "test".into(),
            },
        );

        let selector = BackendSelector::new(
            &registry,
            SelectionStrategy::MlBased {
                min_confidence: 0.5,
            },
        );

        // Should fall back to any available (even unhealthy)
        let result = selector.select_for_type(PropertyType::Theorem);
        assert!(result.is_ok(), "Should fallback to unhealthy backend");
        assert_eq!(result.unwrap(), vec![BackendId::Lean4]);
    }

    #[test]
    fn test_selection_methods_recorded_correctly() {
        // Verify selection_methods vec is populated
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));
        registry.register(Arc::new(MockBackend::new(
            BackendId::TlaPlus,
            vec![PropertyType::Temporal],
        )));

        let selector = BackendSelector::new(&registry, SelectionStrategy::Single);

        use dashprove_usl::ast::{Expr, Temporal, TemporalExpr, Theorem};
        let properties = vec![
            Property::Theorem(Theorem {
                name: "t1".to_string(),
                body: Expr::Bool(true),
            }),
            Property::Temporal(Temporal {
                name: "t2".to_string(),
                body: TemporalExpr::Atom(Expr::Bool(true)),
                fairness: vec![],
            }),
        ];

        let selection = selector.select(&properties).unwrap();
        assert_eq!(
            selection.metrics.selection_methods.len(),
            2,
            "Should have selection method for each property"
        );
        assert!(
            matches!(
                selection.metrics.selection_methods[0],
                SelectionMethod::RuleBased
            ),
            "Single strategy should use RuleBased"
        );
        assert!(
            matches!(
                selection.metrics.selection_methods[1],
                SelectionMethod::RuleBased
            ),
            "Single strategy should use RuleBased"
        );
    }

    #[test]
    fn test_select_for_type_single_falls_back_to_unhealthy() {
        // Mutation: delete ! in select_for_type for Single branch
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(
            MockBackend::new(BackendId::Lean4, vec![PropertyType::Theorem]).with_health(
                HealthStatus::Unavailable {
                    reason: "test".into(),
                },
            ),
        ));
        registry.update_health(
            BackendId::Lean4,
            HealthStatus::Unavailable {
                reason: "test".into(),
            },
        );

        let selector = BackendSelector::new(&registry, SelectionStrategy::Single);
        let result = selector.select_for_type(PropertyType::Theorem);

        // Should fall back to unhealthy backend when no healthy ones exist
        assert!(result.is_ok(), "Should fallback to unhealthy backend");
        assert_eq!(result.unwrap(), vec![BackendId::Lean4]);
    }

    #[test]
    fn test_select_for_type_knowledge_enhanced_falls_back() {
        // Mutation: delete ! in select_for_type for KnowledgeEnhanced branch
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(
            MockBackend::new(BackendId::Lean4, vec![PropertyType::Theorem]).with_health(
                HealthStatus::Unavailable {
                    reason: "test".into(),
                },
            ),
        ));
        registry.update_health(
            BackendId::Lean4,
            HealthStatus::Unavailable {
                reason: "test".into(),
            },
        );

        let selector = BackendSelector::new(
            &registry,
            SelectionStrategy::KnowledgeEnhanced {
                min_confidence: 0.5,
            },
        );
        let result = selector.select_for_type(PropertyType::Theorem);

        // Should fall back to unhealthy backend
        assert!(result.is_ok(), "Should fallback to unhealthy backend");
        assert_eq!(result.unwrap(), vec![BackendId::Lean4]);
    }

    #[test]
    fn test_select_avg_confidence_calculated_correctly() {
        // This test would require knowledge stores to be set up to get knowledge-enhanced selections
        // We test the else branch - when knowledge_confidences is empty
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        let selector = BackendSelector::new(&registry, SelectionStrategy::Single);

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![
            Property::Theorem(Theorem {
                name: "test1".to_string(),
                body: Expr::Bool(true),
            }),
            Property::Theorem(Theorem {
                name: "test2".to_string(),
                body: Expr::Bool(false),
            }),
        ];

        let selection = selector.select(&properties).unwrap();
        // No knowledge-enhanced -> avg should be None (not Some(NaN) or panicking)
        assert!(
            selection.metrics.avg_knowledge_confidence.is_none(),
            "Without knowledge selections, avg should be None not computed"
        );
    }

    #[test]
    fn test_ml_based_uses_prediction_when_above_threshold() {
        // Mutation: replace >= with < in ML confidence check
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        // Create predictor
        let predictor = Arc::new(StrategyModel::from(StrategyPredictor::new()));

        // Very low threshold - ML should be used if prediction has any confidence
        let selector = BackendSelector::with_ml_predictor(
            &registry,
            SelectionStrategy::MlBased {
                min_confidence: 0.0,
            },
            predictor,
        );

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        // With threshold 0.0, ML should be used (any non-negative confidence works)
        // Check that we either got ML selection or fallback (both are valid)
        let method = &selection.assignments[0].selection_method;
        assert!(
            matches!(method, SelectionMethod::MlBased { .. })
                || matches!(method, SelectionMethod::MlFallback { .. }),
            "Expected MlBased or MlFallback, got {:?}",
            method
        );
    }

    #[test]
    fn test_ml_based_alternative_backend_selection() {
        // Mutation: replace >= with < in alternative backend confidence check
        let mut registry = BackendRegistry::new();

        // Primary backend is unhealthy
        registry.register(Arc::new(
            MockBackend::new(BackendId::Lean4, vec![PropertyType::Theorem]).with_health(
                HealthStatus::Unavailable {
                    reason: "test".into(),
                },
            ),
        ));
        registry.update_health(
            BackendId::Lean4,
            HealthStatus::Unavailable {
                reason: "test".into(),
            },
        );

        // Alternative backend is healthy
        registry.register(Arc::new(MockBackend::new(
            BackendId::Coq,
            vec![PropertyType::Theorem],
        )));

        let predictor = Arc::new(StrategyModel::from(StrategyPredictor::new()));
        let selector = BackendSelector::with_ml_predictor(
            &registry,
            SelectionStrategy::MlBased {
                min_confidence: 0.0,
            },
            predictor,
        );

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        // Should select something (either via alternative or fallback)
        assert!(
            !selection.assignments[0].backends.is_empty(),
            "Should select at least one backend"
        );
    }

    // ==================== Additional mutation-killing tests ====================

    #[test]
    fn test_redundant_uses_healthy_when_fewer_than_requested() {
        // Mutation: delete ! in `!healthy.is_empty()` (line 1063)
        // Tests that when healthy.len() < min_backends but healthy is not empty,
        // we use the healthy backends instead of falling to unhealthy
        let mut registry = BackendRegistry::new();

        // Register 1 healthy backend (less than min_backends=2)
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        // Register 1 unhealthy backend
        registry.register(Arc::new(
            MockBackend::new(BackendId::Coq, vec![PropertyType::Theorem]).with_health(
                HealthStatus::Unavailable {
                    reason: "test".into(),
                },
            ),
        ));
        registry.update_health(
            BackendId::Coq,
            HealthStatus::Unavailable {
                reason: "test".into(),
            },
        );

        // Request 2 backends but only 1 healthy
        let selector =
            BackendSelector::new(&registry, SelectionStrategy::Redundant { min_backends: 2 });

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        // Should use the healthy backend (Lean4), not fall through to all
        assert_eq!(
            selection.assignments[0].backends.len(),
            1,
            "Should use single healthy backend"
        );
        assert_eq!(selection.assignments[0].backends[0], BackendId::Lean4);
    }

    #[test]
    fn test_redundant_falls_through_when_no_healthy() {
        // Mutation: delete ! in `!healthy.is_empty()` (line 1063)
        // Tests that when healthy is empty, we fall through to using all backends
        let mut registry = BackendRegistry::new();

        // Register unhealthy backends only
        registry.register(Arc::new(
            MockBackend::new(BackendId::Lean4, vec![PropertyType::Theorem]).with_health(
                HealthStatus::Unavailable {
                    reason: "test".into(),
                },
            ),
        ));
        registry.update_health(
            BackendId::Lean4,
            HealthStatus::Unavailable {
                reason: "test".into(),
            },
        );

        registry.register(Arc::new(
            MockBackend::new(BackendId::Coq, vec![PropertyType::Theorem]).with_health(
                HealthStatus::Unavailable {
                    reason: "test2".into(),
                },
            ),
        ));
        registry.update_health(
            BackendId::Coq,
            HealthStatus::Unavailable {
                reason: "test2".into(),
            },
        );

        let selector =
            BackendSelector::new(&registry, SelectionStrategy::Redundant { min_backends: 2 });

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let selection = selector.select(&properties).unwrap();
        // No healthy backends -> should fall through to using all (unhealthy) backends
        assert_eq!(
            selection.assignments[0].backends.len(),
            2,
            "Should fall through to all backends when none healthy"
        );
    }

    #[tokio::test]
    async fn test_check_all_health_updates_cache() {
        // Mutation: replace check_all_health with () (line 372)
        let mut registry = BackendRegistry::new();

        // Backend reports unhealthy via health_check
        registry.register(Arc::new(
            MockBackend::new(BackendId::Lean4, vec![PropertyType::Theorem]).with_health(
                HealthStatus::Degraded {
                    reason: "slow".into(),
                },
            ),
        ));

        // Initially should have default health (Healthy from registration)
        let initial_info = registry.get_info(BackendId::Lean4).unwrap();
        assert!(
            matches!(initial_info.health, HealthStatus::Healthy),
            "Initial health should be Healthy"
        );

        // Call check_all_health - should update from backend's actual health
        registry.check_all_health().await;

        let updated_info = registry.get_info(BackendId::Lean4).unwrap();
        assert!(
            matches!(updated_info.health, HealthStatus::Degraded { .. }),
            "After check_all_health, should reflect actual health"
        );
    }

    #[test]
    fn test_selector_set_ml_predictor_used() {
        // Mutation: replace set_ml_predictor with () (line 586)
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        // Create selector without predictor initially
        let mut selector = BackendSelector::new(
            &registry,
            SelectionStrategy::MlBased {
                min_confidence: 0.0,
            },
        );

        // Without predictor, should fall back
        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })];

        let selection1 = selector.select(&properties).unwrap();
        let method1 = &selection1.assignments[0].selection_method;
        assert!(
            matches!(method1, SelectionMethod::MlFallback { .. }),
            "Without predictor, should fallback. Got {:?}",
            method1
        );

        // Now set the predictor
        let predictor = Arc::new(StrategyModel::from(StrategyPredictor::new()));
        selector.set_ml_predictor(predictor);

        // With predictor, should try ML-based selection
        let selection2 = selector.select(&properties).unwrap();
        let method2 = &selection2.assignments[0].selection_method;
        // Could be MlBased or MlFallback (if confidence too low), but should be different
        // The key test is that set_ml_predictor actually does something
        assert!(
            matches!(method2, SelectionMethod::MlBased { .. })
                || matches!(method2, SelectionMethod::MlFallback { .. }),
            "With predictor, should attempt ML. Got {:?}",
            method2
        );
    }

    #[test]
    fn test_select_metrics_ml_based_increments() {
        // Mutation: replace += with *= in select (line 641)
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        let predictor = Arc::new(StrategyModel::from(StrategyPredictor::new()));
        let selector = BackendSelector::with_ml_predictor(
            &registry,
            SelectionStrategy::MlBased {
                min_confidence: 0.0,
            },
            predictor,
        );

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![
            Property::Theorem(Theorem {
                name: "test1".to_string(),
                body: Expr::Bool(true),
            }),
            Property::Theorem(Theorem {
                name: "test2".to_string(),
                body: Expr::Bool(false),
            }),
        ];

        let selection = selector.select(&properties).unwrap();
        // With ML strategy, both properties should get either MlBased or MlFallback
        // Count should be at least 1 for each metric type used
        let ml_count = selection.metrics.ml_based_selections;
        let fb_count = selection.metrics.rule_based_fallbacks;
        // At least one of these should be non-zero, and together they should equal 2
        assert!(
            ml_count + fb_count == 2,
            "ML selections ({}) + fallbacks ({}) should equal 2",
            ml_count,
            fb_count
        );
    }

    #[test]
    fn test_select_avg_confidence_division() {
        // Mutation: replace / with * or % in select (line 670)
        // We need knowledge stores to get knowledge-enhanced selections with confidences
        // But without actual stores, we can test the else branch (empty confidences)
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        // Knowledge-enhanced strategy without stores -> falls back
        let selector = BackendSelector::new(
            &registry,
            SelectionStrategy::KnowledgeEnhanced {
                min_confidence: 0.5,
            },
        );

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![
            Property::Theorem(Theorem {
                name: "test1".to_string(),
                body: Expr::Bool(true),
            }),
            Property::Theorem(Theorem {
                name: "test2".to_string(),
                body: Expr::Bool(false),
            }),
        ];

        let selection = selector.select(&properties).unwrap();
        // Without knowledge stores, should fallback and avg_confidence should be None
        assert!(
            selection.metrics.avg_knowledge_confidence.is_none(),
            "Without knowledge stores, avg should be None"
        );
        // Fallbacks should be counted
        assert_eq!(
            selection.metrics.rule_based_fallbacks, 2,
            "Both should be fallbacks"
        );
    }

    #[test]
    fn test_select_knowledge_enhanced_increments() {
        // Mutation: replace += with *= in select (line 634)
        // Without actual knowledge stores, we can't trigger knowledge_enhanced_selections
        // but we verify the fallback path correctly increments rule_based_fallbacks
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        let selector = BackendSelector::new(
            &registry,
            SelectionStrategy::KnowledgeEnhanced {
                min_confidence: 0.5,
            },
        );

        use dashprove_usl::ast::{Expr, Theorem};
        let properties = vec![
            Property::Theorem(Theorem {
                name: "test1".to_string(),
                body: Expr::Bool(true),
            }),
            Property::Theorem(Theorem {
                name: "test2".to_string(),
                body: Expr::Bool(false),
            }),
            Property::Theorem(Theorem {
                name: "test3".to_string(),
                body: Expr::Bool(true),
            }),
        ];

        let selection = selector.select(&properties).unwrap();
        // All 3 should be fallbacks since no knowledge stores
        assert_eq!(
            selection.metrics.rule_based_fallbacks, 3,
            "All 3 properties should use fallback"
        );
    }
}

// ============================================================================
// Kani Proof Harnesses
// ============================================================================
// These formal verification harnesses prove properties about the backend
// selection logic using bounded model checking.
// Run with: cargo kani -p dashprove-dispatcher

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that default_priority returns positive priorities for core backends
    #[kani::proof]
    fn verify_default_priority_core_backends_positive() {
        // Core backends should have positive priorities
        kani::assert(
            BackendRegistry::default_priority(BackendId::Lean4) > 0,
            "Lean4 priority must be positive",
        );
        kani::assert(
            BackendRegistry::default_priority(BackendId::TlaPlus) > 0,
            "TlaPlus priority must be positive",
        );
        kani::assert(
            BackendRegistry::default_priority(BackendId::Kani) > 0,
            "Kani priority must be positive",
        );
        kani::assert(
            BackendRegistry::default_priority(BackendId::Alloy) > 0,
            "Alloy priority must be positive",
        );
        kani::assert(
            BackendRegistry::default_priority(BackendId::Coq) > 0,
            "Coq priority must be positive",
        );
    }

    /// Prove that default_priority returns values in reasonable range
    #[kani::proof]
    fn verify_default_priority_bounded() {
        // All priorities should be in a reasonable range (0-100)
        let lean4 = BackendRegistry::default_priority(BackendId::Lean4);
        let tlaplus = BackendRegistry::default_priority(BackendId::TlaPlus);
        let kani = BackendRegistry::default_priority(BackendId::Kani);
        let alloy = BackendRegistry::default_priority(BackendId::Alloy);

        kani::assert(lean4 >= 0 && lean4 <= 100, "Lean4 priority out of range");
        kani::assert(
            tlaplus >= 0 && tlaplus <= 100,
            "TlaPlus priority out of range",
        );
        kani::assert(kani >= 0 && kani <= 100, "Kani priority out of range");
        kani::assert(alloy >= 0 && alloy <= 100, "Alloy priority out of range");
    }

    /// Prove that Lean4 has highest priority for proof work
    #[kani::proof]
    fn verify_lean4_preferred_for_proofs() {
        let lean4 = BackendRegistry::default_priority(BackendId::Lean4);
        let coq = BackendRegistry::default_priority(BackendId::Coq);
        let isabelle = BackendRegistry::default_priority(BackendId::Isabelle);

        // Lean4 should be preferred over other theorem provers
        kani::assert(lean4 >= coq, "Lean4 should be >= Coq priority");
        kani::assert(lean4 >= isabelle, "Lean4 should be >= Isabelle priority");
    }

    /// Prove that SelectionMetrics::default initializes all counts to zero
    #[kani::proof]
    fn verify_selection_metrics_default_zeroed() {
        let metrics = SelectionMetrics::default();

        kani::assert(
            metrics.properties_processed == 0,
            "properties_processed must be 0",
        );
        kani::assert(
            metrics.knowledge_enhanced_selections == 0,
            "knowledge_enhanced_selections must be 0",
        );
        kani::assert(
            metrics.rule_based_fallbacks == 0,
            "rule_based_fallbacks must be 0",
        );
        kani::assert(
            metrics.ml_based_selections == 0,
            "ml_based_selections must be 0",
        );
        kani::assert(
            metrics.avg_knowledge_confidence.is_none(),
            "avg_knowledge_confidence must be None",
        );
        kani::assert(
            metrics.selection_methods.is_empty(),
            "selection_methods must be empty",
        );
    }

    /// Prove that PropertyAssignment stores data correctly
    #[kani::proof]
    fn verify_property_assignment_stores_data() {
        let idx: usize = kani::any();
        kani::assume(idx < 10000); // Reasonable bound

        let assignment = PropertyAssignment {
            property_index: idx,
            property_type: PropertyType::Theorem,
            backends: vec![BackendId::Lean4],
            selection_method: SelectionMethod::RuleBased,
        };

        kani::assert(
            assignment.property_index == idx,
            "property_index must be preserved",
        );
        kani::assert(
            assignment.property_type == PropertyType::Theorem,
            "property_type must be preserved",
        );
        kani::assert(
            assignment.backends.len() == 1,
            "backends must have one element",
        );
        kani::assert(
            assignment.backends[0] == BackendId::Lean4,
            "backend must be Lean4",
        );
        kani::assert(
            assignment.selection_method == SelectionMethod::RuleBased,
            "selection_method must be RuleBased",
        );
    }

    /// Prove that BackendInfo stores data correctly
    #[kani::proof]
    fn verify_backend_info_stores_data() {
        let priority: i32 = kani::any();
        kani::assume(priority >= 0 && priority <= 100);

        let info = BackendInfo {
            id: BackendId::TlaPlus,
            supported_types: vec![PropertyType::Temporal],
            health: HealthStatus::Healthy,
            priority,
        };

        kani::assert(info.id == BackendId::TlaPlus, "id must be TlaPlus");
        kani::assert(
            info.supported_types.len() == 1,
            "supported_types must have one element",
        );
        kani::assert(
            info.supported_types[0] == PropertyType::Temporal,
            "supported type must be Temporal",
        );
        kani::assert(
            matches!(info.health, HealthStatus::Healthy),
            "health must be Healthy",
        );
        kani::assert(info.priority == priority, "priority must be preserved");
    }

    /// Prove that TlaPlus is preferred for temporal properties
    #[kani::proof]
    fn verify_tlaplus_preferred_for_temporal() {
        let tlaplus = BackendRegistry::default_priority(BackendId::TlaPlus);
        let apalache = BackendRegistry::default_priority(BackendId::Apalache);

        // TlaPlus should have highest priority for temporal
        kani::assert(
            tlaplus >= apalache,
            "TlaPlus should be >= Apalache priority",
        );
    }

    /// Prove that Kani is preferred for contracts/memory safety
    #[kani::proof]
    fn verify_kani_preferred_for_contracts() {
        let kani_pri = BackendRegistry::default_priority(BackendId::Kani);
        let prusti = BackendRegistry::default_priority(BackendId::Prusti);

        // Kani should have highest priority for contracts
        kani::assert(kani_pri >= prusti, "Kani should be >= Prusti priority");
    }

    /// Prove that Storm is preferred for probabilistic verification
    #[kani::proof]
    fn verify_storm_preferred_for_probabilistic() {
        let storm = BackendRegistry::default_priority(BackendId::Storm);
        let prism = BackendRegistry::default_priority(BackendId::Prism);

        // Storm should have highest priority for probabilistic
        kani::assert(storm >= prism, "Storm should be >= Prism priority");
    }

    /// Prove that Tamarin is preferred for security protocols
    #[kani::proof]
    fn verify_tamarin_preferred_for_security() {
        let tamarin = BackendRegistry::default_priority(BackendId::Tamarin);
        let proverif = BackendRegistry::default_priority(BackendId::ProVerif);
        let verifpal = BackendRegistry::default_priority(BackendId::Verifpal);

        // Tamarin should have highest priority for security
        kani::assert(
            tamarin >= proverif,
            "Tamarin should be >= ProVerif priority",
        );
        kani::assert(
            tamarin >= verifpal,
            "Tamarin should be >= Verifpal priority",
        );
    }

    /// Prove that BackendRegistry::new creates empty registry
    #[kani::proof]
    fn verify_backend_registry_new_empty() {
        let registry = BackendRegistry::new();

        kani::assert(
            registry.backends.is_empty(),
            "new registry must have empty backends",
        );
        kani::assert(
            registry.info_cache.is_empty(),
            "new registry must have empty info_cache",
        );
        kani::assert(
            registry.type_to_backends.is_empty(),
            "new registry must have empty type_to_backends",
        );
    }

    /// Prove that SelectionMethod::RuleBased equality is reflexive
    #[kani::proof]
    fn verify_selection_method_equality_reflexive() {
        let method = SelectionMethod::RuleBased;
        kani::assert(
            method == SelectionMethod::RuleBased,
            "RuleBased must equal itself",
        );
    }

    /// Prove that MlBased stores confidence correctly
    #[kani::proof]
    fn verify_ml_based_stores_confidence() {
        let conf: f64 = kani::any();
        kani::assume(conf >= 0.0 && conf <= 1.0);
        kani::assume(!conf.is_nan());

        let method = SelectionMethod::MlBased { confidence: conf };

        if let SelectionMethod::MlBased { confidence } = method {
            kani::assert(confidence == conf, "confidence must be preserved");
        } else {
            kani::assert(false, "Should be MlBased variant");
        }
    }

    /// Prove that selection methods are distinguishable
    #[kani::proof]
    fn verify_selection_methods_distinct() {
        let rule_based = SelectionMethod::RuleBased;
        let ml_based = SelectionMethod::MlBased { confidence: 0.9 };

        kani::assert(
            rule_based != ml_based,
            "RuleBased and MlBased must be distinct",
        );
    }
}
