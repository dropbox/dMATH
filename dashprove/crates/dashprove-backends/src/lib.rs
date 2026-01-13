//! Verification backend implementations
//!
//! Each backend implements the `VerificationBackend` trait.
//! See docs/DESIGN.md for the full trait specification.
//!
//! # Backends
//!
//! ## Core Verification
//! - **TLA+**: Model checking for temporal properties and invariants via TLC
//! - **Apalache**: Symbolic model checking for TLA+ via SMT (unbounded verification)
//! - **LEAN 4**: Theorem proving for mathematical properties via lake
//! - **Kani**: Model checking for Rust contracts via cargo-kani
//! - **Alloy**: Bounded model checking via SAT solving
//! - **Isabelle**: Interactive theorem proving via Isabelle/HOL
//! - **Coq**: Proof assistant based on the Calculus of Constructions
//! - **Dafny**: Verification-aware programming with built-in proofs
//! - **Platform API**: Static checker generation for external API contracts
//!
//! ## Rust Verification
//! - **Verus**: Rust verification via Z3 SMT solver
//! - **Creusot**: Rust verification via Why3 platform
//! - **Prusti**: Rust verification via Viper infrastructure
//!
//! ## SMT Solvers
//! - **Z3**: Microsoft Research SMT solver (general-purpose)
//! - **CVC5**: Stanford/Iowa SMT solver (strings, sets, datatypes)

// Crate-level lint configuration for pedantic clippy
// These are allowed because this crate has extensive formatting/visualization code
// where inline format args in large HTML templates reduce readability
#![allow(clippy::uninlined_format_args)] // HTML templates with many named args
#![allow(clippy::format_push_string)] // Common pattern in visualization builders
#![allow(clippy::similar_names)] // e.g., trace1/trace2, state1/state2
#![allow(clippy::too_many_lines)] // Visualization methods are inherently long
#![allow(clippy::cast_precision_loss)] // usize to f64 for ratios is intentional
#![allow(clippy::doc_markdown)] // Missing backticks - low priority for visualization code
#![allow(clippy::use_self)] // Self vs TypeName - style preference, not correctness
#![allow(clippy::unused_self)] // Some methods keep self for future API compatibility
#![allow(clippy::missing_const_for_fn)] // const fn optimization is minor
#![allow(clippy::must_use_candidate)] // Visualization builders don't need must_use
#![allow(clippy::redundant_closure_for_method_calls)] // Minor style issue
#![allow(clippy::single_char_pattern)] // "." vs '.' - minor optimization
#![allow(clippy::option_if_let_else)] // Style preference for match vs map_or
#![allow(clippy::or_fun_call)] // Style preference, minor optimization
#![allow(clippy::unnecessary_wraps)] // Some wrapping is for API consistency
#![allow(clippy::map_unwrap_or)] // Style preference
#![allow(clippy::wildcard_imports)] // Re-exports use wildcard pattern
#![allow(clippy::unnecessary_debug_formatting)] // Debug formatting is intentional
#![allow(clippy::cast_possible_truncation)] // Bounds are checked at runtime
#![allow(clippy::cast_sign_loss)] // Bounds are checked at runtime
#![allow(clippy::missing_panics_doc)] // Panics are implementation details
#![allow(clippy::needless_raw_string_hashes)] // Raw strings for HTML templates
#![allow(clippy::return_self_not_must_use)] // Builder pattern doesn't need must_use
#![allow(clippy::redundant_else)] // Style preference for explicit else
#![allow(clippy::stable_sort_primitive)] // Minor optimization
#![allow(clippy::match_same_arms)] // Sometimes clarity > deduplication
#![allow(clippy::redundant_clone)] // Clarity over micro-optimization
#![allow(clippy::items_after_statements)] // Sometimes clearer to define helpers near use
#![allow(clippy::match_wildcard_for_single_variants)] // Explicit matching is clearer

// Core backends (theorem provers, model checkers)
pub mod alloy;
pub mod apalache;
pub mod coq;
pub mod counterexample;
pub mod dafny;
pub mod isabelle;
pub mod kani;
pub mod kanifast;
pub mod lean4;
pub mod platform_api;
pub mod tlaplus;
pub mod traits;
pub mod util;

// Dependently typed theorem provers
pub mod acl2;
pub mod agda;
pub mod fstar;
pub mod hol4;
pub mod idris;

// Neural network verification backends
pub mod abcrown;
pub mod autolirpa;
pub mod dnnv;
pub mod eran;
pub mod marabou;
pub mod mnbab;
pub mod neurify;
pub mod nnenum;
pub mod nnv;
pub mod reluval;
pub mod venus;
pub mod verinet;

// Adversarial robustness backends
pub mod art;
pub mod cleverhans;
pub mod foolbox;
pub mod robustbench;
pub mod textattack;

// Probabilistic verification backends
pub mod prism;
pub mod storm;

// Security protocol verification backends
pub mod proverif;
pub mod tamarin;
pub mod verifpal;

// Model checker backends
pub mod cbmc;
pub mod cpachecker;
pub mod framac;
pub mod infer;
pub mod klee;
pub mod nusmv;
pub mod seahorn;
pub mod spin;
pub mod symbiotic;
pub mod twols;

// Rust verification backends
pub mod afl;
pub mod asan;
pub mod audit;
pub mod bolero;
pub mod cdschecker;
pub mod clippy;
pub mod creusot;
pub mod cruxmir;
pub mod deny;
pub mod flux;
pub mod geiger;
pub mod genmc;
pub mod haybale;
pub mod honggfuzz;
pub mod libfuzzer;
pub mod loom;
pub mod lsan;
pub mod mirai;
pub mod miri;
pub mod msan;
pub mod mutants;
pub mod proptest;
pub mod prusti;
pub mod quickcheck;
pub mod rudra;
pub mod rustbelt;
pub mod rusthorn;
pub mod semver_checks;
pub mod shuttle;
pub mod tsan;
pub mod valgrind;
pub mod verus;
pub mod vet;

// Go verification backends
pub mod gobra;

// SMT solver backends
pub mod boolector;
pub mod cvc5;
pub mod mathsat;
pub mod yices;
pub mod z3;

// SAT solver backends
pub mod cadical;
pub mod glucose;
pub mod minisat;

// AI/ML optimization backends
pub mod iree;
pub mod onnxruntime;
pub mod openvino;
pub mod tensorrt;
pub mod triton;
pub mod tvm;

// AI/ML compression backends
pub mod aimet;
pub mod brevitas;
pub mod neural_compressor;
pub mod nncf;

// Data quality backends
pub mod deepchecks;
pub mod evidently;
pub mod great_expectations;
pub mod whylogs;

// Fairness/bias backends
pub mod aequitas;
pub mod aif360;
pub mod fairlearn;

// Interpretability backends
pub mod alibi;
pub mod captum;
pub mod interpretml;
pub mod lime;
pub mod shap;

// LLM guardrails backends
pub mod guardrails_ai;
pub mod guidance;
pub mod nemo_guardrails;

// LLM evaluation backends
pub mod deepeval;
pub mod langsmith;
pub mod promptfoo;
pub mod ragas;
pub mod trulens;

// Hallucination detection backends
pub mod factscore;
pub mod selfcheckgpt;

// Additional theorem provers
pub mod altergo;
pub mod ats;
pub mod hollight;
pub mod metamath;
pub mod mizar;
pub mod pvs;

// Additional SMT solvers
pub mod opensmt;
pub mod verit;

// Additional SAT solvers
pub mod cryptominisat;
pub mod kissat;

// Additional model checkers
pub mod divine;
pub mod esbmc;
pub mod jpf;
pub mod nuxmv;
pub mod smack;
pub mod ultimate;
pub mod uppaal;

// Program verification frameworks
pub mod boogie;
pub mod key;
pub mod krakatoa;
pub mod liquidhaskell;
pub mod openjml;
pub mod spark;
pub mod stainless;
pub mod vcc;
pub mod verifast;
pub mod why3;

// Distributed systems verification
pub mod cadp;
pub mod ivy;
pub mod mcrl2;
pub mod plang;

// Cryptographic verification
pub mod cryptoverif;
pub mod easycrypt;
pub mod jasmin;

// Hardware verification
pub mod cadence_eda;
pub mod jaspergold;
pub mod symbiyosys;
pub mod yosys;

// Symbolic execution and binary analysis
pub mod angr;
pub mod bap;
pub mod ghidra;
pub mod isabil;
pub mod manticore;
pub mod soteria;
pub mod triton_dba;

// Abstract interpretation
pub mod astree;
pub mod codesonar;
pub mod framac_eva;
pub mod polyspace;

// Rust code coverage
pub mod grcov;
pub mod llvm_cov;
pub mod tarpaulin;

// Rust testing infrastructure
pub mod insta;
pub mod mockall;
pub mod nextest;
pub mod rstest;
pub mod test_case;

// Rust documentation quality
pub mod deadlinks;
pub mod rdme;
pub mod spellcheck;

// Re-export core backends
pub use alloy::{AlloyBackend, AlloyConfig};
pub use apalache::{ApalacheBackend, ApalacheConfig, ApalacheMode};
pub use coq::{CoqBackend, CoqConfig};
pub use counterexample::*;
pub use dafny::{DafnyBackend, DafnyConfig};
pub use isabelle::{IsabelleBackend, IsabelleConfig};
pub use kani::{KaniBackend, KaniConfig};
pub use kanifast::{KaniFastBackend, KaniFastConfig};
pub use lean4::{Lean4Backend, Lean4Config};
pub use platform_api::PlatformApiBackend;
pub use tlaplus::{TlaPlusBackend, TlaPlusConfig};
pub use traits::*;

// Re-export Go verification backends
pub use gobra::{GobraBackend, GobraConfig};

// Re-export neural network verification backends
pub use abcrown::{AbCrownBackend, AbCrownConfig};
pub use autolirpa::{AutoLirpaBackend, AutoLirpaConfig, BoundMethod};
pub use dnnv::{DnnvBackend, DnnvConfig, VerifierBackend};
pub use eran::{EranBackend, EranConfig, EranDomain};
pub use marabou::{MarabouBackend, MarabouConfig};
pub use mnbab::{BranchingStrategy, MNBaBBackend, MNBaBConfig};
pub use neurify::{NeurifyBackend, NeurifyConfig, SplitMethod};
pub use nnenum::{EnumerationStrategy, NnenumBackend, NnenumConfig};
pub use nnv::{NnvBackend, NnvConfig, VerificationMethod};
pub use reluval::{RefinementMode, ReluValBackend, ReluValConfig};
pub use venus::{SolverBackend, VenusBackend, VenusConfig};
pub use verinet::{SplittingStrategy, VeriNetBackend, VeriNetConfig};

// Re-export adversarial robustness backends
pub use art::{ArtBackend, ArtConfig};
pub use cleverhans::{CleverHansBackend, CleverHansConfig};
pub use foolbox::{FoolboxBackend, FoolboxConfig};
pub use robustbench::{RobustBenchBackend, RobustBenchConfig};
pub use textattack::{TextAttackBackend, TextAttackConfig};

// Re-export probabilistic verification backends
pub use prism::{PrismBackend, PrismConfig, PrismEngine};
pub use storm::{StormBackend, StormConfig};

// Re-export security protocol verification backends
pub use proverif::{ProverifBackend, ProverifConfig};
pub use tamarin::{TamarinBackend, TamarinConfig};
pub use verifpal::{VerifpalAnalysis, VerifpalBackend, VerifpalConfig};

// Re-export Rust verification backends
pub use afl::{AflBackend, AflConfig};
pub use asan::{AsanBackend, AsanConfig};
pub use audit::{AuditBackend, AuditConfig};
pub use bolero::{BoleroBackend, BoleroConfig};
pub use cdschecker::{
    CDSCheckerBackend, CDSCheckerConfig, ExplorationStrategy as CDSExplorationStrategy,
    MemoryModelMode,
};
pub use clippy::{ClippyBackend, ClippyConfig};
pub use creusot::{CreusotBackend, CreusotConfig};
pub use cruxmir::{CruxMirBackend, CruxMirConfig, CruxMirSolver};
pub use deny::{DenyBackend, DenyCheck, DenyConfig};
pub use flux::{FluxBackend, FluxConfig, FluxMode, FluxSolver};
pub use geiger::{GeigerBackend, GeigerConfig};
pub use genmc::{DPORAlgorithm, GenMCBackend, GenMCConfig, GenMCMemoryModel};
pub use haybale::{HaybaleBackend, HaybaleConfig};
pub use honggfuzz::{HonggfuzzBackend, HonggfuzzConfig};
pub use libfuzzer::{LibFuzzerBackend, LibFuzzerConfig};
pub use loom::{LoomBackend, LoomConfig};
pub use lsan::{LsanBackend, LsanConfig};
pub use mirai::{MiraiBackend, MiraiConfig};
pub use miri::{MiriBackend, MiriBackendConfig};
pub use msan::{MsanBackend, MsanConfig};
pub use mutants::{MutantsBackend, MutantsConfig};
pub use proptest::{ProptestBackend, ProptestConfig};
pub use prusti::{PrustiBackend, PrustiConfig};
pub use quickcheck::{QuickCheckBackend, QuickCheckConfig};
pub use rudra::{RudraBackend, RudraConfig};
pub use rustbelt::{ProofResult, RustBeltBackend, RustBeltConfig};
pub use rusthorn::{ChcSolver, RustHornBackend, RustHornConfig};
pub use semver_checks::{SemverChecksBackend, SemverChecksConfig};
pub use shuttle::{SchedulingStrategy, ShuttleBackend, ShuttleConfig};
pub use tsan::{TsanBackend, TsanConfig};
pub use valgrind::{LeakCheckLevel, ValgrindBackend, ValgrindConfig, ValgrindTool};
pub use verus::{VerusBackend, VerusConfig};
pub use vet::{AuditCriteria, VetBackend, VetConfig};

// Re-export SMT solver backends
pub use boolector::{BoolectorBackend, BoolectorConfig, BoolectorEngine};
pub use cvc5::{Cvc5Backend, Cvc5Config};
pub use mathsat::{MathSatBackend, MathSatConfig, MathSatTheory};
pub use yices::{YicesBackend, YicesConfig};
pub use z3::{Z3Backend, Z3Config};

// Re-export SAT solver backends
pub use cadical::{CaDiCaLBackend, CaDiCaLConfig, CaDiCaLPreprocessing};
pub use glucose::{GlucoseBackend, GlucoseConfig, GlucoseRestartStrategy};
pub use minisat::{MiniSatBackend, MiniSatConfig};

// Re-export AI/ML optimization backends
pub use iree::{ExecutionMode, IREEBackend, IREEConfig, IREETarget, InputFormat};
pub use onnxruntime::{
    ExecutionProvider, GraphOptimizationLevel, OnnxRuntimeBackend, OnnxRuntimeConfig,
};
pub use openvino::{
    DeviceTarget, InferencePrecision, OpenVINOBackend, OpenVINOConfig, PerformanceHint,
};
pub use tensorrt::{OptimizationProfile, PrecisionMode, TensorRTBackend, TensorRTConfig};
pub use triton::{CompilationMode, OptimizationLevel, TritonBackend, TritonConfig, TritonTarget};
pub use tvm::{OptLevel, TVMBackend, TVMConfig, TVMTarget, TuningMode};

// Re-export AI/ML compression backends
pub use aimet::{
    AimetBackend, AimetBitWidth, AimetCompressionMode, AimetConfig, QuantScheme, RoundingMode,
};
pub use brevitas::{
    ActivationBitWidth, BrevitasBackend, BrevitasConfig, ExportFormat, QuantMethod, ScalingMode,
    WeightBitWidth,
};
pub use neural_compressor::{
    CalibrationMethod, NeuralCompressorBackend, NeuralCompressorConfig, QuantDataType,
    QuantizationApproach, TuningStrategy,
};
pub use nncf::{
    BitWidth, CompressionMode, NNCFBackend, NNCFConfig, PruningSchedule, QuantizationMode,
};

// Re-export data quality backends
pub use deepchecks::{DeepchecksBackend, DeepchecksConfig, SeverityThreshold, SuiteType, TaskType};
pub use evidently::{EvidentlyBackend, EvidentlyConfig, OutputFormat, ReportType, StatTestMethod};
pub use great_expectations::{
    DataSourceType, GreatExpectationsBackend, GreatExpectationsConfig, ResultFormat,
    ValidationLevel,
};
pub use whylogs::{
    ConstraintType, ProfileType, WhyLogsBackend, WhyLogsConfig, WhyLogsOutputFormat,
};

// Re-export fairness/bias backends
pub use aequitas::{AequitasBackend, AequitasConfig, AequitasMetric, ReferenceGroup};
pub use aif360::{AIF360Backend, AIF360Config, AIF360MitigationAlgorithm, BiasMetric};
pub use fairlearn::{
    FairlearnBackend, FairlearnConfig, FairnessConstraint, FairnessMetric, MitigationMethod,
};

// Re-export interpretability backends
pub use alibi::{AlibiBackend, AlibiConfig, AlibiExplainer};
pub use captum::{CaptumBackend, CaptumConfig, CaptumMethod};
pub use interpretml::{InterpretExplainer, InterpretMlBackend, InterpretMlConfig, InterpretTask};
pub use lime::{KernelWidth, LimeBackend, LimeConfig, LimeTaskType};
pub use shap::{ShapBackend, ShapConfig, ShapExplainer, ShapModelType};

// Re-export LLM guardrails backends
pub use guardrails_ai::{GuardrailType, GuardrailsAIBackend, GuardrailsAIConfig, StrictnessLevel};
pub use guidance::{GenerationMode, GuidanceBackend, GuidanceConfig, ValidationMode};
pub use nemo_guardrails::{ColangVersion, NeMoGuardrailsBackend, NeMoGuardrailsConfig, RailType};

// Re-export LLM evaluation backends
pub use deepeval::{DeepEvalBackend, DeepEvalConfig, DeepEvalMetric, TestCaseType};
pub use langsmith::{EvaluationType, LangSmithBackend, LangSmithConfig, TracingMode};
pub use promptfoo::{
    AssertionType, OutputFormat as PromptfooOutputFormat, PromptfooBackend, PromptfooConfig,
};
pub use ragas::{EvaluationMode, RagasBackend, RagasConfig, RagasMetric};
pub use trulens::{FeedbackProvider, FeedbackType, TruLensBackend, TruLensConfig};

// Re-export hallucination detection backends
pub use factscore::{ExtractionMethod, FactScoreBackend, FactScoreConfig, KnowledgeSource};
pub use selfcheckgpt::{CheckMethod, SamplingStrategy, SelfCheckGPTBackend, SelfCheckGPTConfig};

// Re-export dependently typed theorem prover backends
pub use acl2::{Acl2Backend, Acl2Config};
pub use agda::{AgdaBackend, AgdaConfig};
pub use fstar::{FStarBackend, FStarConfig, FStarSMTSolver};
pub use hol4::{Hol4Backend, Hol4Config};
pub use idris::{IdrisBackend, IdrisConfig};

// Re-export model checker backends
pub use cbmc::{CbmcBackend, CbmcConfig, CbmcMode};
pub use cpachecker::{CpacheckerBackend, CpacheckerConfig};
pub use framac::{FramaCBackend, FramaCConfig};
pub use infer::{InferAnalysis, InferBackend, InferConfig};
pub use klee::{KleeBackend, KleeConfig, KleeSearcher};
pub use nusmv::{NuSmvBackend, NuSmvConfig};
pub use seahorn::{SeaHornBackend, SeaHornConfig};
pub use spin::{SpinBackend, SpinConfig, SpinSearchMode};
pub use symbiotic::{SymbioticBackend, SymbioticConfig, SymbioticProperty};
pub use twols::{TwoLsAnalysis, TwoLsBackend, TwoLsConfig};

// Re-export additional theorem prover backends
pub use altergo::{AltErgoBackend, AltErgoConfig};
pub use ats::{AtsBackend, AtsConfig};
pub use hollight::{HolLightBackend, HolLightConfig};
pub use metamath::{MetamathBackend, MetamathConfig};
pub use mizar::{MizarBackend, MizarConfig};
pub use pvs::{PvsBackend, PvsConfig};

// Re-export additional SMT solver backends
pub use opensmt::{OpenSmtBackend, OpenSmtConfig};
pub use verit::{VeriTBackend, VeriTConfig};

// Re-export additional SAT solver backends
pub use cryptominisat::{CryptoMiniSatBackend, CryptoMiniSatConfig};
pub use kissat::{KissatBackend, KissatConfig};

// Re-export additional model checker backends
pub use divine::{DivineBackend, DivineConfig};
pub use esbmc::{EsbmcBackend, EsbmcConfig};
pub use jpf::{JpfBackend, JpfConfig};
pub use nuxmv::{NuXmvBackend, NuXmvConfig};
pub use smack::{SmackBackend, SmackConfig};
pub use ultimate::{UltimateBackend, UltimateConfig};
pub use uppaal::{UppaalBackend, UppaalConfig};

// Re-export program verification framework backends
pub use boogie::{BoogieBackend, BoogieConfig};
pub use key::{KeyBackend, KeyConfig};
pub use krakatoa::{KrakatoaBackend, KrakatoaConfig};
pub use liquidhaskell::{LiquidHaskellBackend, LiquidHaskellConfig};
pub use openjml::{OpenJmlBackend, OpenJmlConfig};
pub use spark::{SparkBackend, SparkConfig};
pub use stainless::{StainlessBackend, StainlessConfig};
pub use vcc::{VccBackend, VccConfig};
pub use verifast::{VeriFastBackend, VeriFastConfig};
pub use why3::{Why3Backend, Why3Config};

// Re-export distributed systems verification backends
pub use cadp::{CadpBackend, CadpConfig};
pub use ivy::{IvyBackend, IvyConfig};
pub use mcrl2::{Mcrl2Backend, Mcrl2Config};
pub use plang::{PLangBackend, PLangConfig};

// Re-export cryptographic verification backends
pub use cryptoverif::{CryptoVerifBackend, CryptoVerifConfig};
pub use easycrypt::{EasyCryptBackend, EasyCryptConfig};
pub use jasmin::{JasminBackend, JasminConfig};

// Re-export hardware verification backends
pub use cadence_eda::{CadenceEdaBackend, CadenceEdaConfig};
pub use jaspergold::{JasperGoldBackend, JasperGoldConfig};
pub use symbiyosys::{SymbiYosysBackend, SymbiYosysConfig};
pub use yosys::{YosysBackend, YosysConfig};

// Re-export symbolic execution and binary analysis backends
pub use angr::{AngrBackend, AngrConfig};
pub use bap::{BapAnalysisMode, BapBackend, BapConfig};
pub use ghidra::{GhidraAnalysisMode, GhidraBackend, GhidraConfig};
pub use isabil::{IsaBilBackend, IsaBilConfig, IsaBilMode};
pub use manticore::{ManticoreBackend, ManticoreConfig};
pub use soteria::{SoteriaBackend, SoteriaConfig, SoteriaMemoryModel};
pub use triton_dba::{TritonDbaBackend, TritonDbaConfig};

// Re-export abstract interpretation backends
pub use astree::{AstreeBackend, AstreeConfig};
pub use codesonar::{CodeSonarBackend, CodeSonarConfig};
pub use framac_eva::{FramaCEvaBackend, FramaCEvaConfig};
pub use polyspace::{PolyspaceBackend, PolyspaceConfig};

// Re-export Rust code coverage backends
pub use grcov::{CoverageStats, FileCoverage, GrcovBackend, GrcovConfig, GrcovOutputFormat};
pub use llvm_cov::{
    LlvmCovBackend, LlvmCovConfig, LlvmCovFileCoverage, LlvmCovOutputFormat, LlvmCovStats,
};
pub use tarpaulin::{TarpaulinBackend, TarpaulinConfig};

// Re-export Rust testing infrastructure backends
pub use insta::{InstaBackend, InstaConfig};
pub use mockall::{MockallBackend, MockallConfig};
pub use nextest::{NextestBackend, NextestConfig};
pub use rstest::{RstestBackend, RstestConfig};
pub use test_case::{TestCaseBackend, TestCaseConfig};

// Re-export Rust documentation quality backends
pub use deadlinks::{DeadlinksBackend, DeadlinksConfig};
pub use rdme::{RdmeBackend, RdmeConfig};
pub use spellcheck::{SpellcheckBackend, SpellcheckConfig};
