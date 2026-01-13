//! ML-based strategy prediction for proof automation
//!
//! This module provides machine learning-based prediction of optimal verification
//! strategies. It uses a simple neural network trained on successful proofs to
//! predict:
//! - Best backend for a given property
//! - Optimal tactic sequence
//! - Expected verification time
//!
//! The model is trained on historical proof data from the learning system.
//!
//! ## Module Structure
//! - `features`: Feature extraction from USL properties
//! - `neural`: Neural network primitives (layers, activations)
//! - `ensemble`: Ensemble model support for combining multiple predictors
//! - `hyperparameters`: Hyperparameter search configuration types
//! - `checkpointing`: Model checkpointing for training
//! - `cv_hyperparameters`: Cross-validation in hyperparameter search
//! - `bayesian`: Bayesian optimization for hyperparameter search
//! - `predictions`: Prediction result types
//! - `training`: Training metrics and result types
//! - `training_data`: Training data types and generators
//! - `scheduler`: Learning rate scheduler types
//! - `checkpoint_training`: Model checkpointing during training
//! - `hyperparam_search`: Hyperparameter search algorithms (grid, random, Bayesian)
//! - `core_training`: Core training methods (train, evaluate, early stopping, cross-validation)
//! - `scheduler_training`: Learning rate scheduler-based training methods

mod bayesian;
mod checkpoint_training;
mod checkpointing;
mod core_training;
mod cv_hyperparameters;
mod ensemble;
mod features;
mod hyperparam_search;
mod hyperparameters;
mod neural;
mod predictions;
mod scheduler;
mod scheduler_training;
mod training;
mod training_data;

// Re-export from features module
pub use features::{normalize, PropertyFeatureVector, MAX_TACTICS, NUM_BACKENDS, NUM_FEATURES};

// Re-export from neural module
pub use neural::{relu_derivative, softmax, DenseLayer};

// Re-export from ensemble module
pub use ensemble::{EnsembleAggregation, EnsembleMember, EnsembleStrategyPredictor, StrategyModel};

// Re-export from hyperparameters module
pub use hyperparameters::{
    GridSearchSpace, HyperparameterResult, HyperparameterSearchResult, Hyperparameters,
    RandomSearchConfig,
};

// Re-export from checkpointing module
pub use checkpointing::{Checkpoint, CheckpointConfig, CheckpointedTrainingResult};

// Re-export from cv_hyperparameters module
pub use cv_hyperparameters::{CVHyperparameterResult, CVHyperparameterSearchResult};

// Re-export from bayesian module
pub use bayesian::{BayesianOptimizationResult, BayesianOptimizer};

// Re-export from predictions module
pub use predictions::{BackendPrediction, TacticPrediction, TimePrediction};

// Re-export from training module
pub use training::{
    CrossValidationResult, EarlyStoppingConfig, EarlyStoppingResult, EpochMetrics,
    EvaluationResult, TrainingHistory,
};

// Re-export from training_data module
pub use training_data::{
    StrategyPrediction, TrainingDataGenerator, TrainingExample, TrainingStats,
};

// Re-export from scheduler module
pub use scheduler::{LearningRateScheduler, ScheduledTrainingResult};

// Internal use
use hyperparameters::SimpleRng;

use dashprove_backends::BackendId;
use dashprove_usl::ast::Property;
use serde::{Deserialize, Serialize};

/// Neural network-based strategy predictor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyPredictor {
    /// Hidden layer 1 (features -> 64)
    pub(crate) hidden1: DenseLayer,
    /// Hidden layer 2 (64 -> 32)
    pub(crate) hidden2: DenseLayer,
    /// Backend output layer (32 -> num_backends)
    pub(crate) backend_output: DenseLayer,
    /// Tactic output layer (32 -> num_tactics * max_tactics)
    tactic_output: DenseLayer,
    /// Time prediction layer (32 -> 1)
    time_output: DenseLayer,
    /// Known tactic names for output decoding
    tactic_names: Vec<String>,
}

impl Default for StrategyPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyPredictor {
    /// Create a new strategy predictor with random weights
    pub fn new() -> Self {
        let default_tactics = vec![
            // LEAN 4 tactics
            "simp".to_string(),
            "decide".to_string(),
            "rfl".to_string(),
            "intro".to_string(),
            "apply".to_string(),
            "exact".to_string(),
            "cases".to_string(),
            "induction".to_string(),
            "omega".to_string(),
            "ring".to_string(),
            "linarith".to_string(),
            "norm_num".to_string(),
            "constructor".to_string(),
            "rcases".to_string(),
            "use".to_string(),
            "ext".to_string(),
            // Coq tactics
            "auto".to_string(),
            "intuition".to_string(),
            "rewrite".to_string(),
            "unfold".to_string(),
            // Isabelle tactics
            "blast".to_string(),
            "force".to_string(),
            "sledgehammer".to_string(),
            // TLA+ strategies
            "model_check".to_string(),
            "state_space".to_string(),
            // Kani strategies
            "symbolic_exec".to_string(),
            "bounded_check".to_string(),
            // Alloy strategies
            "sat_solve".to_string(),
            "counterexample".to_string(),
            // SMT tactics
            "smt".to_string(),
        ];

        StrategyPredictor {
            hidden1: DenseLayer::new(NUM_FEATURES, 64),
            hidden2: DenseLayer::new(64, 32),
            backend_output: DenseLayer::new(32, NUM_BACKENDS),
            tactic_output: DenseLayer::new(32, default_tactics.len() * MAX_TACTICS),
            time_output: DenseLayer::new(32, 1),
            tactic_names: default_tactics,
        }
    }

    /// Predict the best backend for a property
    pub fn predict_backend(&self, features: &PropertyFeatureVector) -> BackendPrediction {
        // Forward pass
        let h1 = self.hidden1.forward_relu(&features.features);
        let h2 = self.hidden2.forward_relu(&h1);
        let backend_probs = self.backend_output.forward_softmax(&h2);

        // Find best backend
        let (best_idx, best_prob) = backend_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let backend = idx_to_backend(best_idx);

        // Get top-3 alternatives
        let mut ranked: Vec<_> = backend_probs.iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let alternatives: Vec<(BackendId, f64)> = ranked
            .iter()
            .skip(1)
            .take(3)
            .map(|(idx, &prob)| (idx_to_backend(*idx), prob))
            .collect();

        BackendPrediction {
            backend,
            confidence: *best_prob,
            alternatives,
        }
    }

    /// Predict probability distribution over backends
    ///
    /// Returns probabilities for each backend in index order. Useful for
    /// ensembles that need full distributions rather than just the top-1
    /// prediction.
    pub fn backend_probabilities(&self, features: &PropertyFeatureVector) -> Vec<(BackendId, f64)> {
        let h1 = self.hidden1.forward_relu(&features.features);
        let h2 = self.hidden2.forward_relu(&h1);
        let backend_probs = self.backend_output.forward_softmax(&h2);

        backend_probs
            .iter()
            .enumerate()
            .map(|(idx, prob)| (idx_to_backend(idx), *prob))
            .collect()
    }

    /// Predict the best tactic sequence for a property
    pub fn predict_tactics(
        &self,
        features: &PropertyFeatureVector,
        n: usize,
    ) -> Vec<TacticPrediction> {
        let h1 = self.hidden1.forward_relu(&features.features);
        let h2 = self.hidden2.forward_relu(&h1);
        let tactic_raw = self.tactic_output.forward(&h2);

        // Reshape to (MAX_TACTICS, num_tactics) and apply softmax per position
        let num_tactics = self.tactic_names.len();
        let mut predictions = Vec::new();

        for pos in 0..n.min(MAX_TACTICS) {
            let start = pos * num_tactics;
            let end = start + num_tactics;
            if end > tactic_raw.len() {
                break;
            }
            let position_logits: Vec<f64> = tactic_raw[start..end].to_vec();
            let position_probs = softmax(&position_logits);

            // Find best tactic for this position
            let (best_idx, best_prob) = position_probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            if *best_prob > 0.1 && best_idx < self.tactic_names.len() {
                // Only include if confident enough
                predictions.push(TacticPrediction {
                    tactic: self.tactic_names[best_idx].clone(),
                    confidence: *best_prob,
                    position: pos,
                });
            }
        }

        predictions
    }

    /// Predict expected verification time in seconds
    pub fn predict_time(&self, features: &PropertyFeatureVector) -> TimePrediction {
        let h1 = self.hidden1.forward_relu(&features.features);
        let h2 = self.hidden2.forward_relu(&h1);
        let time_raw = self.time_output.forward(&h2);

        // Convert to positive time using exponential
        let predicted_log_time = time_raw.first().copied().unwrap_or(0.0);
        let predicted_time = predicted_log_time.exp().clamp(0.1, 3600.0);

        // Estimate confidence based on feature complexity
        let complexity = features.get("complexity_score").unwrap_or(0.5);
        let confidence = 1.0 - complexity * 0.5;

        TimePrediction {
            expected_seconds: predicted_time,
            confidence,
            range: (predicted_time * 0.5, predicted_time * 2.0),
        }
    }

    /// Full strategy prediction combining all outputs
    pub fn predict_strategy(&self, property: &Property) -> StrategyPrediction {
        let features = PropertyFeatureVector::from_property(property);

        let backend = self.predict_backend(&features);
        let tactics = self.predict_tactics(&features, 5);
        let time = self.predict_time(&features);

        StrategyPrediction {
            backend,
            tactics,
            time,
            features,
        }
    }

    /// Save model to file
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    /// Load model from file
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Train from corpus entries and return training statistics
    ///
    /// This is a convenience method that combines:
    /// 1. Generating training data from corpus entries
    /// 2. Training the model on that data
    /// 3. Returning statistics about what was trained
    ///
    /// # Arguments
    /// * `entries` - Iterator of (property, backend, tactics, time_seconds) tuples
    /// * `learning_rate` - Learning rate for training (typically 0.01-0.1)
    /// * `epochs` - Number of training passes (typically 10-100)
    ///
    /// # Returns
    /// Training statistics including number of examples and examples per backend
    pub fn train_from_corpus<'a, I>(
        &mut self,
        entries: I,
        learning_rate: f64,
        epochs: usize,
    ) -> TrainingStats
    where
        I: IntoIterator<Item = (&'a Property, BackendId, &'a [String], f64)>,
    {
        let generator = TrainingDataGenerator::from_corpus_entries(entries);
        let stats = generator.stats();
        let training_data = generator.get_training_data();

        if !training_data.is_empty() {
            self.train(&training_data, learning_rate, epochs);
        }

        stats
    }

    /// Train and save model to file, returning training statistics
    ///
    /// Combines training from corpus with saving the trained model.
    pub fn train_and_save<'a, I, P>(
        &mut self,
        entries: I,
        learning_rate: f64,
        epochs: usize,
        path: P,
    ) -> std::io::Result<TrainingStats>
    where
        I: IntoIterator<Item = (&'a Property, BackendId, &'a [String], f64)>,
        P: AsRef<std::path::Path>,
    {
        let stats = self.train_from_corpus(entries, learning_rate, epochs);
        self.save(path)?;
        Ok(stats)
    }
}

/// Convert backend index to BackendId
pub(crate) fn idx_to_backend(idx: usize) -> BackendId {
    match idx {
        // Core backends (0-21)
        0 => BackendId::Lean4,
        1 => BackendId::TlaPlus,
        2 => BackendId::Kani,
        3 => BackendId::Alloy,
        4 => BackendId::Isabelle,
        5 => BackendId::Coq,
        6 => BackendId::Dafny,
        7 => BackendId::Marabou,
        8 => BackendId::AlphaBetaCrown,
        9 => BackendId::Eran,
        10 => BackendId::Storm,
        11 => BackendId::Prism,
        12 => BackendId::Tamarin,
        13 => BackendId::ProVerif,
        14 => BackendId::Verifpal,
        15 => BackendId::Verus,
        16 => BackendId::Creusot,
        17 => BackendId::Prusti,
        18 => BackendId::Z3,
        19 => BackendId::Cvc5,
        20 => BackendId::PlatformApi,
        21 => BackendId::Apalache,
        // Phase 12 backends (22-97)
        22 => BackendId::NNV,
        23 => BackendId::Nnenum,
        24 => BackendId::VeriNet,
        25 => BackendId::Venus,
        26 => BackendId::DNNV,
        27 => BackendId::AutoLiRPA,
        28 => BackendId::MNBaB,
        29 => BackendId::Neurify,
        30 => BackendId::ReluVal,
        31 => BackendId::ART,
        32 => BackendId::Foolbox,
        33 => BackendId::CleverHans,
        34 => BackendId::TextAttack,
        35 => BackendId::RobustBench,
        36 => BackendId::Flux,
        37 => BackendId::Mirai,
        38 => BackendId::Rudra,
        39 => BackendId::Miri,
        40 => BackendId::AddressSanitizer,
        41 => BackendId::MemorySanitizer,
        42 => BackendId::ThreadSanitizer,
        43 => BackendId::LeakSanitizer,
        44 => BackendId::Loom,
        45 => BackendId::Shuttle,
        46 => BackendId::CDSChecker,
        47 => BackendId::GenMC,
        48 => BackendId::LibFuzzer,
        49 => BackendId::AFL,
        50 => BackendId::Honggfuzz,
        51 => BackendId::Bolero,
        52 => BackendId::Proptest,
        53 => BackendId::QuickCheck,
        54 => BackendId::Clippy,
        55 => BackendId::SemverChecks,
        56 => BackendId::Geiger,
        57 => BackendId::Audit,
        58 => BackendId::Deny,
        59 => BackendId::Vet,
        60 => BackendId::Mutants,
        61 => BackendId::ONNXRuntime,
        62 => BackendId::TensorRT,
        63 => BackendId::OpenVINO,
        64 => BackendId::TVM,
        65 => BackendId::IREE,
        66 => BackendId::Triton,
        67 => BackendId::NeuralCompressor,
        68 => BackendId::NNCF,
        69 => BackendId::AIMET,
        70 => BackendId::Brevitas,
        71 => BackendId::GreatExpectations,
        72 => BackendId::Deepchecks,
        73 => BackendId::Evidently,
        74 => BackendId::WhyLogs,
        75 => BackendId::Fairlearn,
        76 => BackendId::AIF360,
        77 => BackendId::Aequitas,
        78 => BackendId::SHAP,
        79 => BackendId::LIME,
        80 => BackendId::Captum,
        81 => BackendId::InterpretML,
        82 => BackendId::Alibi,
        83 => BackendId::GuardrailsAI,
        84 => BackendId::NeMoGuardrails,
        85 => BackendId::Guidance,
        86 => BackendId::Promptfoo,
        87 => BackendId::TruLens,
        88 => BackendId::LangSmith,
        89 => BackendId::Ragas,
        90 => BackendId::DeepEval,
        91 => BackendId::SelfCheckGPT,
        92 => BackendId::FactScore,
        93 => BackendId::Valgrind,
        // Model checkers (94-97)
        94 => BackendId::SPIN,
        95 => BackendId::CBMC,
        96 => BackendId::Infer,
        97 => BackendId::KLEE,
        // SMT solvers (98-100)
        98 => BackendId::Yices,
        99 => BackendId::Boolector,
        100 => BackendId::MathSAT,
        // SAT solvers (101-103)
        101 => BackendId::MiniSat,
        102 => BackendId::Glucose,
        103 => BackendId::CaDiCaL,
        // Dependently typed theorem provers (104-108)
        104 => BackendId::Agda,
        105 => BackendId::Idris,
        106 => BackendId::ACL2,
        107 => BackendId::HOL4,
        108 => BackendId::FStar,
        // Additional model checkers and program verifiers (109-112)
        109 => BackendId::NuSMV,
        110 => BackendId::CPAchecker,
        111 => BackendId::SeaHorn,
        112 => BackendId::FramaC,
        // Additional theorem provers (113-117)
        113 => BackendId::HOLLight,
        114 => BackendId::PVS,
        115 => BackendId::Mizar,
        116 => BackendId::Metamath,
        117 => BackendId::ATS,
        // Additional SMT solvers (118-120)
        118 => BackendId::OpenSMT,
        119 => BackendId::VeriT,
        120 => BackendId::AltErgo,
        // Additional SAT solvers (121-122)
        121 => BackendId::Kissat,
        122 => BackendId::CryptoMiniSat,
        // Additional model checkers (123-129)
        123 => BackendId::NuXmv,
        124 => BackendId::UPPAAL,
        125 => BackendId::DIVINE,
        126 => BackendId::ESBMC,
        127 => BackendId::Ultimate,
        128 => BackendId::SMACK,
        129 => BackendId::JPF,
        // Program verification frameworks (130-139)
        130 => BackendId::VCC,
        131 => BackendId::VeriFast,
        132 => BackendId::KeY,
        133 => BackendId::OpenJML,
        134 => BackendId::Krakatoa,
        135 => BackendId::SPARK,
        136 => BackendId::Why3,
        137 => BackendId::Stainless,
        138 => BackendId::LiquidHaskell,
        139 => BackendId::Boogie,
        // Distributed systems verification (140-143)
        140 => BackendId::PLang,
        141 => BackendId::Ivy,
        142 => BackendId::MCRL2,
        143 => BackendId::CADP,
        // Cryptographic verification (144-146)
        144 => BackendId::EasyCrypt,
        145 => BackendId::CryptoVerif,
        146 => BackendId::Jasmin,
        // Hardware verification (147-150)
        147 => BackendId::Yosys,
        148 => BackendId::SymbiYosys,
        149 => BackendId::JasperGold,
        150 => BackendId::CadenceEDA,
        // Symbolic execution and binary analysis (151-157)
        151 => BackendId::Angr,
        152 => BackendId::Manticore,
        153 => BackendId::TritonDBA,
        154 => BackendId::Bap,
        155 => BackendId::Ghidra,
        156 => BackendId::IsaBIL,
        157 => BackendId::Soteria,
        // Abstract interpretation (158-161)
        158 => BackendId::Astree,
        159 => BackendId::Polyspace,
        160 => BackendId::CodeSonar,
        161 => BackendId::FramaCEva,
        // Rust code coverage (162-164)
        162 => BackendId::Tarpaulin,
        163 => BackendId::LlvmCov,
        164 => BackendId::Grcov,
        // Rust testing frameworks (165-169)
        165 => BackendId::Nextest,
        166 => BackendId::Insta,
        167 => BackendId::Rstest,
        168 => BackendId::TestCase,
        169 => BackendId::Mockall,
        // Rust documentation tools (170-172)
        170 => BackendId::Deadlinks,
        171 => BackendId::Spellcheck,
        172 => BackendId::Rdme,
        // Additional Rust verification (173-176)
        173 => BackendId::Haybale,
        174 => BackendId::CruxMir,
        175 => BackendId::RustHorn,
        176 => BackendId::RustBelt,
        // Go verification (177)
        177 => BackendId::Gobra,
        // Additional C/C++ verification (178-179)
        178 => BackendId::Symbiotic,
        179 => BackendId::TwoLS,
        _ => BackendId::Lean4, // Default fallback for out of range
    }
}

/// Convert BackendId to index
pub(crate) fn backend_to_idx(backend: BackendId) -> usize {
    match backend {
        // Core backends (0-21)
        BackendId::Lean4 => 0,
        BackendId::TlaPlus => 1,
        BackendId::Kani => 2,
        BackendId::Alloy => 3,
        BackendId::Isabelle => 4,
        BackendId::Coq => 5,
        BackendId::Dafny => 6,
        BackendId::Marabou => 7,
        BackendId::AlphaBetaCrown => 8,
        BackendId::Eran => 9,
        BackendId::Storm => 10,
        BackendId::Prism => 11,
        BackendId::Tamarin => 12,
        BackendId::ProVerif => 13,
        BackendId::Verifpal => 14,
        BackendId::Verus => 15,
        BackendId::Creusot => 16,
        BackendId::Prusti => 17,
        BackendId::Z3 => 18,
        BackendId::Cvc5 => 19,
        BackendId::PlatformApi => 20,
        BackendId::Apalache => 21,
        // Phase 12 backends (22-97)
        BackendId::NNV => 22,
        BackendId::Nnenum => 23,
        BackendId::VeriNet => 24,
        BackendId::Venus => 25,
        BackendId::DNNV => 26,
        BackendId::AutoLiRPA => 27,
        BackendId::MNBaB => 28,
        BackendId::Neurify => 29,
        BackendId::ReluVal => 30,
        BackendId::ART => 31,
        BackendId::Foolbox => 32,
        BackendId::CleverHans => 33,
        BackendId::TextAttack => 34,
        BackendId::RobustBench => 35,
        BackendId::Flux => 36,
        BackendId::Mirai => 37,
        BackendId::Rudra => 38,
        BackendId::Miri => 39,
        BackendId::AddressSanitizer => 40,
        BackendId::MemorySanitizer => 41,
        BackendId::ThreadSanitizer => 42,
        BackendId::LeakSanitizer => 43,
        BackendId::Valgrind => 93,
        BackendId::Loom => 44,
        BackendId::Shuttle => 45,
        BackendId::CDSChecker => 46,
        BackendId::GenMC => 47,
        BackendId::LibFuzzer => 48,
        BackendId::AFL => 49,
        BackendId::Honggfuzz => 50,
        BackendId::Bolero => 51,
        BackendId::Proptest => 52,
        BackendId::QuickCheck => 53,
        BackendId::Clippy => 54,
        BackendId::SemverChecks => 55,
        BackendId::Geiger => 56,
        BackendId::Audit => 57,
        BackendId::Deny => 58,
        BackendId::Vet => 59,
        BackendId::Mutants => 60,
        BackendId::ONNXRuntime => 61,
        BackendId::TensorRT => 62,
        BackendId::OpenVINO => 63,
        BackendId::TVM => 64,
        BackendId::IREE => 65,
        BackendId::Triton => 66,
        BackendId::NeuralCompressor => 67,
        BackendId::NNCF => 68,
        BackendId::AIMET => 69,
        BackendId::Brevitas => 70,
        BackendId::GreatExpectations => 71,
        BackendId::Deepchecks => 72,
        BackendId::Evidently => 73,
        BackendId::WhyLogs => 74,
        BackendId::Fairlearn => 75,
        BackendId::AIF360 => 76,
        BackendId::Aequitas => 77,
        BackendId::SHAP => 78,
        BackendId::LIME => 79,
        BackendId::Captum => 80,
        BackendId::InterpretML => 81,
        BackendId::Alibi => 82,
        BackendId::GuardrailsAI => 83,
        BackendId::NeMoGuardrails => 84,
        BackendId::Guidance => 85,
        BackendId::Promptfoo => 86,
        BackendId::TruLens => 87,
        BackendId::LangSmith => 88,
        BackendId::Ragas => 89,
        BackendId::DeepEval => 90,
        BackendId::SelfCheckGPT => 91,
        BackendId::FactScore => 92,
        // Model checkers (94-97)
        BackendId::SPIN => 94,
        BackendId::CBMC => 95,
        BackendId::Infer => 96,
        BackendId::KLEE => 97,
        // SMT solvers (98-100)
        BackendId::Yices => 98,
        BackendId::Boolector => 99,
        BackendId::MathSAT => 100,
        // SAT solvers (101-103)
        BackendId::MiniSat => 101,
        BackendId::Glucose => 102,
        BackendId::CaDiCaL => 103,
        // Dependently typed theorem provers (104-108)
        BackendId::Agda => 104,
        BackendId::Idris => 105,
        BackendId::ACL2 => 106,
        BackendId::HOL4 => 107,
        BackendId::FStar => 108,
        // Additional model checkers and program verifiers (109-112)
        BackendId::NuSMV => 109,
        BackendId::CPAchecker => 110,
        BackendId::SeaHorn => 111,
        BackendId::FramaC => 112,
        // Additional theorem provers (113-117)
        BackendId::HOLLight => 113,
        BackendId::PVS => 114,
        BackendId::Mizar => 115,
        BackendId::Metamath => 116,
        BackendId::ATS => 117,
        // Additional SMT solvers (118-120)
        BackendId::OpenSMT => 118,
        BackendId::VeriT => 119,
        BackendId::AltErgo => 120,
        // Additional SAT solvers (121-122)
        BackendId::Kissat => 121,
        BackendId::CryptoMiniSat => 122,
        // Additional model checkers (123-129)
        BackendId::NuXmv => 123,
        BackendId::UPPAAL => 124,
        BackendId::DIVINE => 125,
        BackendId::ESBMC => 126,
        BackendId::Ultimate => 127,
        BackendId::SMACK => 128,
        BackendId::JPF => 129,
        // Program verification frameworks (130-139)
        BackendId::VCC => 130,
        BackendId::VeriFast => 131,
        BackendId::KeY => 132,
        BackendId::OpenJML => 133,
        BackendId::Krakatoa => 134,
        BackendId::SPARK => 135,
        BackendId::Why3 => 136,
        BackendId::Stainless => 137,
        BackendId::LiquidHaskell => 138,
        BackendId::Boogie => 139,
        // Distributed systems verification (140-143)
        BackendId::PLang => 140,
        BackendId::Ivy => 141,
        BackendId::MCRL2 => 142,
        BackendId::CADP => 143,
        // Cryptographic verification (144-146)
        BackendId::EasyCrypt => 144,
        BackendId::CryptoVerif => 145,
        BackendId::Jasmin => 146,
        // Hardware verification (147-150)
        BackendId::Yosys => 147,
        BackendId::SymbiYosys => 148,
        BackendId::JasperGold => 149,
        BackendId::CadenceEDA => 150,
        // Symbolic execution and binary analysis (151-157)
        BackendId::Angr => 151,
        BackendId::Manticore => 152,
        BackendId::TritonDBA => 153,
        BackendId::Bap => 154,
        BackendId::Ghidra => 155,
        BackendId::IsaBIL => 156,
        BackendId::Soteria => 157,
        // Abstract interpretation (158-161)
        BackendId::Astree => 158,
        BackendId::Polyspace => 159,
        BackendId::CodeSonar => 160,
        BackendId::FramaCEva => 161,
        // Rust code coverage (162-164)
        BackendId::Tarpaulin => 162,
        BackendId::LlvmCov => 163,
        BackendId::Grcov => 164,
        // Rust testing frameworks (165-169)
        BackendId::Nextest => 165,
        BackendId::Insta => 166,
        BackendId::Rstest => 167,
        BackendId::TestCase => 168,
        BackendId::Mockall => 169,
        // Rust documentation tools (170-172)
        BackendId::Deadlinks => 170,
        BackendId::Spellcheck => 171,
        BackendId::Rdme => 172,
        // Additional Rust verification (173-176)
        BackendId::Haybale => 173,
        BackendId::CruxMir => 174,
        BackendId::RustHorn => 175,
        BackendId::RustBelt => 176,
        // Go verification (177)
        BackendId::Gobra => 177,
        // Additional C/C++ verification (178-179)
        BackendId::Symbiotic => 178,
        BackendId::TwoLS => 179,
        // Kani Fast (180)
        BackendId::KaniFast => 180,
    }
}

#[cfg(test)]
mod tests;
