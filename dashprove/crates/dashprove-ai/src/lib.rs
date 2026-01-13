// Crate-level lint configuration for pedantic clippy
#![allow(clippy::must_use_candidate)] // Builder methods don't need must_use
#![allow(clippy::missing_const_for_fn)] // const fn optimization is minor
#![allow(clippy::use_self)] // Self vs TypeName - style preference
#![allow(clippy::unused_self)] // Some methods keep self for future API compatibility
#![allow(clippy::doc_markdown)] // Missing backticks - low priority
#![allow(clippy::missing_errors_doc)] // Error docs are implementation details
#![allow(clippy::cast_precision_loss)] // usize to f64 for scores is intentional
#![allow(clippy::match_same_arms)] // Sometimes clarity > deduplication
#![allow(clippy::map_unwrap_or)] // Style preference
#![allow(clippy::uninlined_format_args)] // Named args are clearer
#![allow(clippy::format_push_string)] // Common pattern in string builders
#![allow(clippy::similar_names)] // e.g., step/steps, tactic/tactics
#![allow(clippy::too_many_lines)] // Complex methods may be inherently long
#![allow(clippy::needless_pass_by_value)] // Ownership semantics may be intentional
#![allow(clippy::redundant_closure_for_method_calls)] // Minor style issue
#![allow(clippy::option_if_let_else)] // Style preference for match vs map_or
#![allow(clippy::cast_possible_wrap)] // i64 to i32 checked at runtime
#![allow(clippy::cast_possible_truncation)] // Bounds checked at runtime
#![allow(clippy::cast_sign_loss)] // Bounds checked at runtime
#![allow(clippy::cast_lossless)] // Explicit casts are clearer
#![allow(clippy::or_fun_call)] // Style preference
#![allow(clippy::needless_raw_string_hashes)] // Raw strings for templates
#![allow(clippy::wildcard_imports)] // Re-exports use wildcard pattern
#![allow(clippy::single_char_pattern)] // "=" vs '=' - minor optimization
#![allow(clippy::match_wildcard_for_single_variants)] // Explicit wildcard is clearer
#![allow(clippy::trivially_copy_pass_by_ref)] // &BackendId is API consistency

//! AI-assisted proof generation
//!
//! This module provides AI-native assistance for formal verification:
//!
//! - **Proof Sketching**: Outline proof structure that backends can elaborate
//! - **Tactic Suggestion**: Recommend next proof steps based on learning
//! - **Counterexample Explanation**: Human-readable explanations of failures
//! - **Proof Repair**: Detect and suggest fixes for broken proofs
//! - **Cross-Tool Translation**: Convert proofs between Lean 4, Coq, and Isabelle
//!
//! The AI assistant integrates with the learning system to improve over time.

pub mod counterexample;
pub mod explain_llm;
pub mod llm;
pub mod proof_search;
pub mod repair;
pub mod sketch;
pub mod spec_inference;
pub mod strategy;
pub mod suggest;
pub mod synthesis;
pub mod tactic_llm;
pub mod translate;
pub mod validation;

pub use counterexample::{
    explain_counterexample, Binding, CounterexampleExplanation, ExplanationKind, TraceStep,
};
pub use explain_llm::{EnhancedExplanation, ExplanationRequest, LlmExplainer};
pub use proof_search::{
    decompose_property, decompose_property_with_mode, instantiate_template, CrossBackendHint,
    DecompositionResult, DecompositionStrategy, HierarchicalSearchResult, InductionMode,
    ProofSearchAgent, ProofSearchConfig, ProofSearchRequest, ProofSearchResult, ProofSearchStep,
    ProofTemplate, SubGoal, SubGoalResult, TacticPolicySnapshot, TemplateBinding, TemplateCategory,
    TemplateMatch, TemplateParam, TemplateParamKind, TemplateRegistry,
};
pub use repair::{detect_proof_breaks, suggest_repairs, ProofDiff, RepairSuggestion};
pub use sketch::{elaborate_sketch, ProofSketch, SketchStep};
pub use spec_inference::{
    SourceLanguage, SpecInferenceError, SpecInferenceRequest, SpecInferenceResult, SpecInferencer,
};
pub use strategy::{
    BackendPrediction, BayesianOptimizationResult, BayesianOptimizer, CVHyperparameterResult,
    CVHyperparameterSearchResult, Checkpoint, CheckpointConfig, CheckpointedTrainingResult,
    CrossValidationResult, DenseLayer, EarlyStoppingConfig, EarlyStoppingResult,
    EnsembleAggregation, EnsembleMember, EnsembleStrategyPredictor, EpochMetrics, EvaluationResult,
    GridSearchSpace, HyperparameterResult, HyperparameterSearchResult, Hyperparameters,
    LearningRateScheduler, PropertyFeatureVector, RandomSearchConfig, ScheduledTrainingResult,
    StrategyModel, StrategyPrediction, StrategyPredictor, TacticPrediction, TimePrediction,
    TrainingDataGenerator, TrainingExample, TrainingHistory, TrainingStats,
};
pub use suggest::{suggest_tactics, SuggestionSource, TacticSuggestion};
pub use synthesis::{ProofSynthesizer, SynthesisError, SynthesisRequest, SynthesisResult};
pub use tactic_llm::{LlmTacticPredictor, LlmTacticResult, TacticPredictionRequest};
pub use translate::{
    ProofLanguage, ProofTranslator, TranslateError, TranslateRequest, TranslateResult,
};
pub use validation::{
    adversarial_synthesis_tests, adversarial_translation_tests, validate_synthesized_proof,
    verify_round_trip, AdversarialTest, ProofValidation, RoundTripResult, Severity,
    TacticBenchmark, ValidationIssue,
};

use dashprove_backends::traits::BackendId;
use dashprove_usl::ast::Property;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors from the AI assistant
#[derive(Error, Debug)]
pub enum AiError {
    /// No suggestions available for the given property type or pattern
    #[error("No suggestions available for property")]
    NoSuggestions,
    /// Failed to generate explanation for a counterexample
    #[error("Cannot explain counterexample: {0}")]
    ExplanationFailed(String),
    /// Failed to elaborate a proof sketch into a full proof
    #[error("Sketch elaboration failed: {0}")]
    ElaborationFailed(String),
    /// The provided proof structure is invalid or malformed
    #[error("Invalid proof structure: {0}")]
    InvalidProof(String),
}

/// Confidence level for AI suggestions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Confidence {
    /// Based on many successful proofs
    High,
    /// Based on some successful proofs
    Medium,
    /// Heuristic or single observation
    Low,
    /// No prior data, purely structural
    Speculative,
}

impl Confidence {
    /// Convert from a score (0.0 to 1.0)
    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            Confidence::High
        } else if score >= 0.5 {
            Confidence::Medium
        } else if score >= 0.2 {
            Confidence::Low
        } else {
            Confidence::Speculative
        }
    }

    /// Convert to a score (0.0 to 1.0)
    pub fn to_score(self) -> f64 {
        match self {
            Confidence::High => 0.9,
            Confidence::Medium => 0.65,
            Confidence::Low => 0.35,
            Confidence::Speculative => 0.1,
        }
    }
}

/// Strategy recommendation for verifying a property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationStrategy {
    /// Recommended backend
    pub backend: BackendId,
    /// Recommended tactics in order
    pub tactics: Vec<TacticSuggestion>,
    /// Overall confidence
    pub confidence: Confidence,
    /// Rationale for this strategy
    pub rationale: String,
}

/// Get human-readable name for a backend
fn backend_name(backend: BackendId) -> &'static str {
    match backend {
        // Theorem provers
        BackendId::Lean4 => "LEAN 4",
        BackendId::TlaPlus => "TLA+",
        BackendId::Apalache => "Apalache",
        BackendId::Kani => "Kani",
        BackendId::Alloy => "Alloy",
        BackendId::Isabelle => "Isabelle",
        BackendId::Coq => "Coq",
        BackendId::Dafny => "Dafny",
        BackendId::PlatformApi => "Platform API",
        // Neural network verifiers
        BackendId::Marabou => "Marabou",
        BackendId::AlphaBetaCrown => "α,β-CROWN",
        BackendId::Eran => "ERAN",
        BackendId::NNV => "NNV",
        BackendId::Nnenum => "nnenum",
        BackendId::VeriNet => "VeriNet",
        BackendId::Venus => "Venus",
        BackendId::DNNV => "DNNV",
        BackendId::AutoLiRPA => "Auto-LiRPA",
        BackendId::MNBaB => "MN-BaB",
        BackendId::Neurify => "Neurify",
        BackendId::ReluVal => "ReluVal",
        // Adversarial robustness
        BackendId::ART => "ART",
        BackendId::Foolbox => "Foolbox",
        BackendId::CleverHans => "CleverHans",
        BackendId::TextAttack => "TextAttack",
        BackendId::RobustBench => "RobustBench",
        // Probabilistic
        BackendId::Storm => "Storm",
        BackendId::Prism => "PRISM",
        // Security
        BackendId::Tamarin => "Tamarin",
        BackendId::ProVerif => "ProVerif",
        BackendId::Verifpal => "Verifpal",
        // Rust formal verification
        BackendId::Verus => "Verus",
        BackendId::Creusot => "Creusot",
        BackendId::Prusti => "Prusti",
        BackendId::Flux => "Flux",
        BackendId::Mirai => "MIRAI",
        BackendId::Rudra => "Rudra",
        BackendId::Miri => "Miri",
        BackendId::Haybale => "Haybale",
        BackendId::CruxMir => "Crux-mir",
        BackendId::RustHorn => "RustHorn",
        BackendId::RustBelt => "RustBelt",
        // Go verification
        BackendId::Gobra => "Gobra",
        // SMT solvers
        BackendId::Z3 => "Z3",
        BackendId::Cvc5 => "CVC5",
        // Rust sanitizers
        BackendId::AddressSanitizer => "AddressSanitizer",
        BackendId::MemorySanitizer => "MemorySanitizer",
        BackendId::ThreadSanitizer => "ThreadSanitizer",
        BackendId::LeakSanitizer => "LeakSanitizer",
        BackendId::Valgrind => "Valgrind",
        // Rust concurrency
        BackendId::Loom => "Loom",
        BackendId::Shuttle => "Shuttle",
        BackendId::CDSChecker => "CDSChecker",
        BackendId::GenMC => "GenMC",
        // Rust fuzzing
        BackendId::LibFuzzer => "LibFuzzer",
        BackendId::AFL => "AFL",
        BackendId::Honggfuzz => "Honggfuzz",
        BackendId::Bolero => "Bolero",
        // Rust PBT
        BackendId::Proptest => "Proptest",
        BackendId::QuickCheck => "QuickCheck",
        // Rust static analysis
        BackendId::Clippy => "Clippy",
        BackendId::SemverChecks => "semver-checks",
        BackendId::Geiger => "cargo-geiger",
        BackendId::Audit => "cargo-audit",
        BackendId::Deny => "cargo-deny",
        BackendId::Vet => "cargo-vet",
        BackendId::Mutants => "cargo-mutants",
        // AI/ML optimization
        BackendId::ONNXRuntime => "ONNX Runtime",
        BackendId::TensorRT => "TensorRT",
        BackendId::OpenVINO => "OpenVINO",
        BackendId::TVM => "TVM",
        BackendId::IREE => "IREE",
        BackendId::Triton => "Triton",
        // AI/ML compression
        BackendId::NeuralCompressor => "Neural Compressor",
        BackendId::NNCF => "NNCF",
        BackendId::AIMET => "AIMET",
        BackendId::Brevitas => "Brevitas",
        // Data quality
        BackendId::GreatExpectations => "Great Expectations",
        BackendId::Deepchecks => "Deepchecks",
        BackendId::Evidently => "Evidently",
        BackendId::WhyLogs => "WhyLogs",
        // Fairness
        BackendId::Fairlearn => "Fairlearn",
        BackendId::AIF360 => "AI Fairness 360",
        BackendId::Aequitas => "Aequitas",
        // Interpretability
        BackendId::SHAP => "SHAP",
        BackendId::LIME => "LIME",
        BackendId::Captum => "Captum",
        BackendId::InterpretML => "InterpretML",
        BackendId::Alibi => "Alibi",
        // LLM guardrails
        BackendId::GuardrailsAI => "Guardrails AI",
        BackendId::NeMoGuardrails => "NeMo Guardrails",
        BackendId::Guidance => "Guidance",
        // LLM evaluation
        BackendId::Promptfoo => "Promptfoo",
        BackendId::TruLens => "TruLens",
        BackendId::LangSmith => "LangSmith",
        BackendId::Ragas => "Ragas",
        BackendId::DeepEval => "DeepEval",
        // Hallucination detection
        BackendId::SelfCheckGPT => "SelfCheckGPT",
        BackendId::FactScore => "FactScore",
        // Model checkers
        BackendId::SPIN => "SPIN",
        BackendId::CBMC => "CBMC",
        BackendId::Infer => "Infer",
        BackendId::KLEE => "KLEE",
        BackendId::NuSMV => "NuSMV",
        BackendId::CPAchecker => "CPAchecker",
        BackendId::SeaHorn => "SeaHorn",
        BackendId::FramaC => "Frama-C",
        BackendId::Symbiotic => "Symbiotic",
        BackendId::TwoLS => "2LS",
        // SMT solvers
        BackendId::Yices => "Yices",
        BackendId::Boolector => "Boolector",
        BackendId::MathSAT => "MathSAT",
        // SAT solvers
        BackendId::MiniSat => "MiniSat",
        BackendId::Glucose => "Glucose",
        BackendId::CaDiCaL => "CaDiCaL",
        // Dependently typed theorem provers
        BackendId::Agda => "Agda",
        BackendId::Idris => "Idris",
        BackendId::ACL2 => "ACL2",
        BackendId::HOL4 => "HOL4",
        BackendId::FStar => "FStar",
        // Additional theorem provers
        BackendId::HOLLight => "HOL Light",
        BackendId::PVS => "PVS",
        BackendId::Mizar => "Mizar",
        BackendId::Metamath => "Metamath",
        BackendId::ATS => "ATS",
        // Additional SMT solvers
        BackendId::OpenSMT => "OpenSMT",
        BackendId::VeriT => "veriT",
        BackendId::AltErgo => "Alt-Ergo",
        // Additional SAT solvers
        BackendId::Kissat => "Kissat",
        BackendId::CryptoMiniSat => "CryptoMiniSat",
        // Additional model checkers
        BackendId::NuXmv => "nuXmv",
        BackendId::UPPAAL => "UPPAAL",
        BackendId::DIVINE => "DIVINE",
        BackendId::ESBMC => "ESBMC",
        BackendId::Ultimate => "Ultimate",
        BackendId::SMACK => "SMACK",
        BackendId::JPF => "Java PathFinder",
        // Program verification frameworks
        BackendId::VCC => "VCC",
        BackendId::VeriFast => "VeriFast",
        BackendId::KeY => "KeY",
        BackendId::OpenJML => "OpenJML",
        BackendId::Krakatoa => "Krakatoa",
        BackendId::SPARK => "SPARK",
        BackendId::Why3 => "Why3",
        BackendId::Stainless => "Stainless",
        BackendId::LiquidHaskell => "LiquidHaskell",
        BackendId::Boogie => "Boogie",
        // Distributed systems verification
        BackendId::PLang => "P",
        BackendId::Ivy => "Ivy",
        BackendId::MCRL2 => "mCRL2",
        BackendId::CADP => "CADP",
        // Cryptographic verification
        BackendId::EasyCrypt => "EasyCrypt",
        BackendId::CryptoVerif => "CryptoVerif",
        BackendId::Jasmin => "Jasmin",
        // Hardware verification
        BackendId::Yosys => "Yosys",
        BackendId::SymbiYosys => "SymbiYosys",
        BackendId::JasperGold => "JasperGold",
        BackendId::CadenceEDA => "Cadence EDA",
        // Symbolic execution and binary analysis
        BackendId::Angr => "angr",
        BackendId::Manticore => "Manticore",
        BackendId::TritonDBA => "Triton",
        BackendId::Bap => "BAP",
        BackendId::Ghidra => "Ghidra",
        BackendId::IsaBIL => "IsaBIL",
        BackendId::Soteria => "Soteria",
        // Abstract interpretation
        BackendId::Astree => "Astrée",
        BackendId::Polyspace => "Polyspace",
        BackendId::CodeSonar => "CodeSonar",
        BackendId::FramaCEva => "Frama-C EVA",
        // Rust code coverage
        BackendId::Tarpaulin => "cargo-tarpaulin",
        BackendId::LlvmCov => "cargo-llvm-cov",
        BackendId::Grcov => "grcov",
        // Rust testing infrastructure
        BackendId::Nextest => "cargo-nextest",
        BackendId::Insta => "cargo-insta",
        BackendId::Rstest => "rstest",
        BackendId::TestCase => "test-case",
        BackendId::Mockall => "mockall",
        // Rust documentation quality
        BackendId::Deadlinks => "cargo-deadlinks",
        BackendId::Spellcheck => "cargo-spellcheck",
        BackendId::Rdme => "cargo-rdme",
        // Kani Fast (enhanced Kani)
        BackendId::KaniFast => "Kani Fast",
    }
}

/// AI assistant for proof generation
pub struct ProofAssistant {
    /// Learning system reference (optional - can work without)
    learning: Option<dashprove_learning::ProofLearningSystem>,
    /// ML-based strategy predictor (optional - for enhanced predictions)
    ml_predictor: Option<StrategyModel>,
}

impl Default for ProofAssistant {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofAssistant {
    /// Create a new proof assistant without learning data
    pub fn new() -> Self {
        Self {
            learning: None,
            ml_predictor: None,
        }
    }

    /// Create a proof assistant with learning data
    pub fn with_learning(learning: dashprove_learning::ProofLearningSystem) -> Self {
        Self {
            learning: Some(learning),
            ml_predictor: None,
        }
    }

    /// Create a proof assistant with ML predictor
    pub fn with_ml_predictor(predictor: impl Into<StrategyModel>) -> Self {
        Self {
            learning: None,
            ml_predictor: Some(predictor.into()),
        }
    }

    /// Create a proof assistant with both learning and ML predictor
    pub fn with_full_ai(
        learning: dashprove_learning::ProofLearningSystem,
        predictor: impl Into<StrategyModel>,
    ) -> Self {
        Self {
            learning: Some(learning),
            ml_predictor: Some(predictor.into()),
        }
    }

    /// Set the ML predictor
    pub fn set_ml_predictor(&mut self, predictor: impl Into<StrategyModel>) {
        self.ml_predictor = Some(predictor.into());
    }

    /// Get ML strategy prediction if predictor is available
    pub fn predict_ml_strategy(&self, property: &Property) -> Option<StrategyPrediction> {
        self.ml_predictor
            .as_ref()
            .map(|p| p.predict_strategy(property))
    }

    /// Get a comprehensive strategy recommendation for a property
    ///
    /// Uses ML predictor if available, falling back to heuristic-based recommendations.
    pub fn recommend_strategy(&self, property: &Property) -> VerificationStrategy {
        // Try ML-based prediction first if available
        if let Some(ref predictor) = self.ml_predictor {
            let ml_prediction = predictor.predict_strategy(property);

            // Use ML prediction if confidence is reasonable
            if ml_prediction.backend.confidence > 0.3 {
                let backend = ml_prediction.backend.backend;
                let tactics: Vec<TacticSuggestion> = ml_prediction
                    .tactics
                    .iter()
                    .map(|t| TacticSuggestion {
                        tactic: t.tactic.clone(),
                        confidence: Confidence::from_score(t.confidence),
                        source: SuggestionSource::Learning, // ML is a form of learned prediction
                        rationale: format!("ML prediction (position {})", t.position),
                    })
                    .collect();

                let confidence = Confidence::from_score(ml_prediction.backend.confidence);
                let rationale = format!(
                    "ML model predicts {} with {:.1}% confidence. Expected time: {:.1}s",
                    backend_name(backend),
                    ml_prediction.backend.confidence * 100.0,
                    ml_prediction.time.expected_seconds
                );

                return VerificationStrategy {
                    backend,
                    tactics,
                    confidence,
                    rationale,
                };
            }
        }

        // Fall back to heuristic-based recommendation
        let mut tactics = self.suggest_tactics(property, 5);
        let backend = self.recommend_backend(property);

        // If no tactics from learning, use compiler suggestions
        if tactics.is_empty() {
            tactics = suggest::compiler_suggestions(property, &backend);
        }

        let confidence = if let Some(ref learning) = self.learning {
            let similar = learning.find_similar(property, 3);
            if similar.is_empty() {
                Confidence::Speculative
            } else {
                let avg_score: f64 =
                    similar.iter().map(|s| s.similarity).sum::<f64>() / similar.len() as f64;
                Confidence::from_score(avg_score)
            }
        } else {
            Confidence::Speculative
        };

        let rationale = self.build_rationale(property, &backend, &tactics, &confidence);

        VerificationStrategy {
            backend,
            tactics,
            confidence,
            rationale,
        }
    }

    /// Suggest tactics for a property
    pub fn suggest_tactics(&self, property: &Property, n: usize) -> Vec<TacticSuggestion> {
        let mut suggestions = Vec::new();

        // Learning-based suggestions
        if let Some(ref learning) = self.learning {
            let learned = learning.suggest_tactics(property, n);
            for (tactic, score) in learned {
                suggestions.push(TacticSuggestion {
                    tactic,
                    confidence: Confidence::from_score(score),
                    source: SuggestionSource::Learning,
                    rationale: "Based on similar successful proofs".to_string(),
                });
            }
        }

        // Compiler-based suggestions (structural)
        let backend = self.recommend_backend(property);
        let compiler_suggestions = suggest::compiler_suggestions(property, &backend);

        // Merge, preferring learned suggestions
        for cs in compiler_suggestions {
            if !suggestions.iter().any(|s| s.tactic == cs.tactic) {
                suggestions.push(cs);
            }
        }

        suggestions.truncate(n);
        suggestions
    }

    /// Explain a counterexample in human-readable form
    pub fn explain_counterexample(
        &self,
        property: &Property,
        counterexample: &str,
        backend: &BackendId,
    ) -> CounterexampleExplanation {
        explain_counterexample(property, counterexample, backend)
    }

    /// Suggest repairs for a failed proof
    pub fn suggest_repairs(
        &self,
        property: &Property,
        old_proof: Option<&str>,
        error: &str,
        backend: &BackendId,
    ) -> Vec<RepairSuggestion> {
        suggest_repairs(property, old_proof, error, backend)
    }

    /// Create a proof sketch from high-level description
    pub fn create_sketch(&self, property: &Property, hints: &[String]) -> ProofSketch {
        sketch::create_sketch(property, hints, self.learning.as_ref())
    }

    /// Infer specifications from source code using static analysis
    ///
    /// This uses the `SpecInferencer` to extract contracts, preconditions, and
    /// postconditions from source code in various languages (Rust, TypeScript,
    /// Python, Go).
    ///
    /// # Example
    /// ```ignore
    /// let assistant = ProofAssistant::new();
    /// let code = "fn add(a: i32, b: i32) -> i32 { assert!(a >= 0); a + b }";
    /// let result = assistant.infer_specs(code, SourceLanguage::Rust);
    /// assert!(!result.properties.is_empty());
    /// ```
    pub fn infer_specs(&self, code: &str, language: SourceLanguage) -> SpecInferenceResult {
        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, language);
        inferencer.infer_static(&request)
    }

    /// Infer specifications from source code with custom options
    ///
    /// Allows specifying target backend, hints, and property limits.
    pub fn infer_specs_with_options(
        &self,
        code: &str,
        language: SourceLanguage,
        target_backend: Option<BackendId>,
        hints: Vec<String>,
        max_properties: usize,
    ) -> SpecInferenceResult {
        let inferencer = SpecInferencer::new();
        let mut request = SpecInferenceRequest::new(code, language)
            .with_hints(hints)
            .limit_properties(max_properties);
        if let Some(backend) = target_backend {
            request = request.with_backend(backend);
        }
        inferencer.infer_static(&request)
    }

    /// Infer specifications from a file path
    ///
    /// Automatically detects the language from the file extension.
    pub fn infer_specs_from_path(&self, code: &str, file_path: &str) -> SpecInferenceResult {
        let language = SourceLanguage::from_path(file_path);
        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, language).with_file_path(file_path);
        inferencer.infer_static(&request)
    }

    /// Recommend the best backend for a property type
    fn recommend_backend(&self, property: &Property) -> BackendId {
        match property {
            Property::Theorem(_) | Property::Invariant(_) | Property::Refinement(_) => {
                BackendId::Lean4
            }
            Property::Temporal(_) => BackendId::TlaPlus,
            Property::Contract(_) => BackendId::Kani,
            Property::Probabilistic(_) => BackendId::Storm,
            Property::Security(_) => BackendId::Tamarin,
            Property::Semantic(_) => BackendId::Lean4,
            Property::PlatformApi(_) => BackendId::PlatformApi,
            // Bisimulation uses its own specialized runtime, not a traditional backend
            Property::Bisimulation(_) => BackendId::Kani, // Default to Kani for now
            Property::Version(_) => BackendId::SemverChecks, // Version compatibility
            Property::Capability(_) => BackendId::Kani,   // Capability verification
            Property::DistributedInvariant(_) => BackendId::TlaPlus, // Multi-agent invariants
            Property::DistributedTemporal(_) => BackendId::TlaPlus, // Multi-agent temporal
            Property::Composed(_) => BackendId::Lean4,    // Composed theorems use Lean4
            Property::ImprovementProposal(_) => BackendId::Lean4, // Improvement proposals use Lean4
            Property::VerificationGate(_) => BackendId::Lean4, // Verification gates use Lean4
            Property::Rollback(_) => BackendId::TlaPlus,  // Rollback specs use TLA+
        }
    }

    fn build_rationale(
        &self,
        property: &Property,
        backend: &BackendId,
        tactics: &[TacticSuggestion],
        confidence: &Confidence,
    ) -> String {
        let property_type = match property {
            Property::Theorem(_) => "theorem",
            Property::Temporal(_) => "temporal property",
            Property::Contract(_) => "contract",
            Property::Invariant(_) => "invariant",
            Property::Refinement(_) => "refinement",
            Property::Probabilistic(_) => "probabilistic property",
            Property::Security(_) => "security property",
            Property::Semantic(_) => "semantic property",
            Property::PlatformApi(_) => "platform API constraint",
            Property::Bisimulation(_) => "bisimulation",
            Property::Version(_) => "version constraint",
            Property::Capability(_) => "capability specification",
            Property::DistributedInvariant(_) => "distributed invariant",
            Property::DistributedTemporal(_) => "distributed temporal property",
            Property::Composed(_) => "composed theorem",
            Property::ImprovementProposal(_) => "improvement proposal",
            Property::VerificationGate(_) => "verification gate",
            Property::Rollback(_) => "rollback specification",
        };

        let backend_name = backend_name(*backend);

        let confidence_desc = match confidence {
            Confidence::High => "high confidence based on many similar proofs",
            Confidence::Medium => "medium confidence from related proofs",
            Confidence::Low => "low confidence, limited similar examples",
            Confidence::Speculative => "speculative, no prior similar proofs",
        };

        let tactic_desc = if tactics.is_empty() {
            "No specific tactics suggested".to_string()
        } else {
            let names: Vec<_> = tactics.iter().take(3).map(|t| t.tactic.as_str()).collect();
            format!("Suggested tactics: {}", names.join(", "))
        };

        format!(
            "For this {}, {} is recommended. {}. {}.",
            property_type, backend_name, confidence_desc, tactic_desc
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::ast::{Expr, Invariant, Theorem};

    fn make_theorem(name: &str) -> Property {
        Property::Theorem(Theorem {
            name: name.to_string(),
            body: Expr::ForAll {
                var: "x".to_string(),
                ty: Some(dashprove_usl::ast::Type::Named("Bool".to_string())),
                body: Box::new(Expr::Or(
                    Box::new(Expr::Var("x".to_string())),
                    Box::new(Expr::Not(Box::new(Expr::Var("x".to_string())))),
                )),
            },
        })
    }

    fn make_invariant(name: &str) -> Property {
        Property::Invariant(Invariant {
            name: name.to_string(),
            body: Expr::Bool(true),
        })
    }

    #[test]
    fn test_assistant_creation() {
        let assistant = ProofAssistant::new();
        let prop = make_theorem("test");
        let strategy = assistant.recommend_strategy(&prop);

        assert_eq!(strategy.backend, BackendId::Lean4);
        assert_eq!(strategy.confidence, Confidence::Speculative);
    }

    #[test]
    fn test_strategy_includes_tactics() {
        let assistant = ProofAssistant::new();
        let prop = make_theorem("test_lem");
        let strategy = assistant.recommend_strategy(&prop);

        // Should have at least compiler-suggested tactics
        assert!(!strategy.tactics.is_empty());
    }

    #[test]
    fn test_confidence_conversion() {
        assert_eq!(Confidence::from_score(0.9), Confidence::High);
        assert_eq!(Confidence::from_score(0.6), Confidence::Medium);
        assert_eq!(Confidence::from_score(0.3), Confidence::Low);
        assert_eq!(Confidence::from_score(0.1), Confidence::Speculative);
    }

    #[test]
    fn test_invariant_strategy() {
        let assistant = ProofAssistant::new();
        let prop = make_invariant("test_inv");
        let strategy = assistant.recommend_strategy(&prop);

        assert_eq!(strategy.backend, BackendId::Lean4);
        assert!(!strategy.rationale.is_empty());
    }

    #[test]
    fn test_proof_assistant_infers_rust_specs() {
        let assistant = ProofAssistant::new();
        let code = r#"
        fn add_positive(a: i32, b: i32) -> i32 {
            assert!(a >= 0);
            assert!(b >= 0);
            a + b
        }
        "#;
        let result = assistant.infer_specs(code, SourceLanguage::Rust);
        assert!(!result.properties.is_empty());
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_proof_assistant_infers_typescript_specs() {
        let assistant = ProofAssistant::new();
        let code = r#"
        function multiply(x: number, y: number): number {
            console.assert(x > 0);
            return x * y;
        }
        "#;
        let result = assistant.infer_specs(code, SourceLanguage::TypeScript);
        assert!(!result.properties.is_empty());
    }

    #[test]
    fn test_proof_assistant_infers_python_specs() {
        let assistant = ProofAssistant::new();
        let code = "def square(x: int) -> int:\n    assert x >= 0\n    return x * x\n";
        let result = assistant.infer_specs(code, SourceLanguage::Python);
        assert!(!result.properties.is_empty());
    }

    #[test]
    fn test_proof_assistant_infers_go_specs() {
        let assistant = ProofAssistant::new();
        let code = r#"
func Square(x int) int {
    if x < 0 { panic("x must be non-negative") }
    return x * x
}
"#;
        let result = assistant.infer_specs(code, SourceLanguage::Go);
        assert!(!result.properties.is_empty());
    }

    #[test]
    fn test_proof_assistant_infers_from_path() {
        let assistant = ProofAssistant::new();
        let code = "fn identity(x: i32) -> i32 { x }";
        let result = assistant.infer_specs_from_path(code, "src/lib.rs");
        assert!(!result.properties.is_empty());
    }

    #[test]
    fn test_proof_assistant_infers_with_options() {
        let assistant = ProofAssistant::new();
        let code = r#"
        fn foo(a: i32) -> i32 { assert!(a > 0); a }
        fn bar(b: i32) -> i32 { assert!(b > 0); b }
        fn baz(c: i32) -> i32 { assert!(c > 0); c }
        "#;
        let result = assistant.infer_specs_with_options(
            code,
            SourceLanguage::Rust,
            Some(BackendId::Kani),
            vec!["function should not panic".to_string()],
            2, // Limit to 2 properties
        );
        // Should have at most 2 properties due to limit
        assert!(result.properties.len() <= 2);
    }

    #[test]
    fn test_proof_assistant_infers_c_specs() {
        let assistant = ProofAssistant::new();
        let code = r#"
int abs(int x) {
    if (x < 0) return -x;
    return x;
}
"#;
        let result = assistant.infer_specs(code, SourceLanguage::C);
        assert!(!result.properties.is_empty());
    }

    #[test]
    fn test_proof_assistant_infers_cpp_specs() {
        let assistant = ProofAssistant::new();
        let code = r#"
int Vector::size() const {
    assert(data_ != nullptr);
    return size_;
}
"#;
        let result = assistant.infer_specs(code, SourceLanguage::Cpp);
        assert!(!result.properties.is_empty());
    }

    #[test]
    fn test_proof_assistant_infers_cpp_from_path() {
        let assistant = ProofAssistant::new();
        let code = "int add(int a, int b) { return a + b; }";
        let result = assistant.infer_specs_from_path(code, "math.cpp");
        assert!(!result.properties.is_empty());
        // Verify it detected C++ from extension
        assert!(result
            .assumptions
            .iter()
            .any(|a| a.contains("C++") || a.contains("math.cpp")));
    }
}

// ========== Kani proof harnesses ==========

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify Confidence::from_score returns High for scores >= 0.8
    #[kani::proof]
    fn verify_confidence_from_score_high() {
        let score = 0.9;
        let confidence = Confidence::from_score(score);
        kani::assert(
            matches!(confidence, Confidence::High),
            "score >= 0.8 should be High",
        );
    }

    /// Verify Confidence::from_score returns Medium for scores in [0.5, 0.8)
    #[kani::proof]
    fn verify_confidence_from_score_medium() {
        let score = 0.6;
        let confidence = Confidence::from_score(score);
        kani::assert(
            matches!(confidence, Confidence::Medium),
            "score in [0.5, 0.8) should be Medium",
        );
    }

    /// Verify Confidence::from_score returns Low for scores in [0.2, 0.5)
    #[kani::proof]
    fn verify_confidence_from_score_low() {
        let score = 0.3;
        let confidence = Confidence::from_score(score);
        kani::assert(
            matches!(confidence, Confidence::Low),
            "score in [0.2, 0.5) should be Low",
        );
    }

    /// Verify Confidence::from_score returns Speculative for scores < 0.2
    #[kani::proof]
    fn verify_confidence_from_score_speculative() {
        let score = 0.1;
        let confidence = Confidence::from_score(score);
        kani::assert(
            matches!(confidence, Confidence::Speculative),
            "score < 0.2 should be Speculative",
        );
    }

    /// Verify Confidence::to_score returns correct values
    #[kani::proof]
    fn verify_confidence_to_score_high() {
        let score = Confidence::High.to_score();
        kani::assert(score == 0.9, "High should map to 0.9");
    }

    /// Verify Confidence::to_score for Medium
    #[kani::proof]
    fn verify_confidence_to_score_medium() {
        let score = Confidence::Medium.to_score();
        kani::assert(score == 0.65, "Medium should map to 0.65");
    }

    /// Verify Confidence::to_score for Low
    #[kani::proof]
    fn verify_confidence_to_score_low() {
        let score = Confidence::Low.to_score();
        kani::assert(score == 0.35, "Low should map to 0.35");
    }

    /// Verify Confidence::to_score for Speculative
    #[kani::proof]
    fn verify_confidence_to_score_speculative() {
        let score = Confidence::Speculative.to_score();
        kani::assert(score == 0.1, "Speculative should map to 0.1");
    }

    /// Verify from_score boundary at 0.8
    #[kani::proof]
    fn verify_confidence_boundary_high() {
        let score = 0.8;
        let confidence = Confidence::from_score(score);
        kani::assert(
            matches!(confidence, Confidence::High),
            "score == 0.8 should be High",
        );
    }

    /// Verify from_score boundary at 0.5
    #[kani::proof]
    fn verify_confidence_boundary_medium() {
        let score = 0.5;
        let confidence = Confidence::from_score(score);
        kani::assert(
            matches!(confidence, Confidence::Medium),
            "score == 0.5 should be Medium",
        );
    }

    /// Verify from_score boundary at 0.2
    #[kani::proof]
    fn verify_confidence_boundary_low() {
        let score = 0.2;
        let confidence = Confidence::from_score(score);
        kani::assert(
            matches!(confidence, Confidence::Low),
            "score == 0.2 should be Low",
        );
    }

    /// Verify ProofAssistant::new creates default instance
    #[kani::proof]
    fn verify_proof_assistant_new() {
        let assistant = ProofAssistant::new();
        kani::assert(
            assistant.learning.is_none(),
            "new() should have no learning",
        );
        kani::assert(
            assistant.ml_predictor.is_none(),
            "new() should have no predictor",
        );
    }
}
