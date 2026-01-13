//! Backend trait definitions and core types
//!
//! This module contains:
//! - Backend-related types (BackendId, BackendResult, BackendError, etc.)
//! - The VerificationBackend trait
//! - HTML export helpers (shared by visualization modules)
//!
//! Counterexample types have been moved to the `counterexample` module.

use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use thiserror::Error;

// Re-export counterexample types for backward compatibility
pub use crate::counterexample::*;

// ============================================================================
// HTML Export Helpers
// ============================================================================

/// Generate download buttons HTML for HTML visualizations
/// Takes Mermaid code and optionally DOT code for download
pub fn html_download_buttons(mermaid_code: &str, dot_code: Option<&str>) -> String {
    html_download_buttons_with_container(mermaid_code, dot_code, None)
}

/// Generate download buttons HTML for HTML visualizations with a specific container ID
///
/// The container_id is used to locate the SVG element for download.
/// If None, uses the default ".mermaid" selector.
pub fn html_download_buttons_with_container(
    mermaid_code: &str,
    dot_code: Option<&str>,
    container_id: Option<&str>,
) -> String {
    static DOWNLOAD_BUTTON_ID: AtomicUsize = AtomicUsize::new(0);
    let id = DOWNLOAD_BUTTON_ID.fetch_add(1, Ordering::Relaxed);

    let mermaid_fn = format!("downloadMermaid_{id}");
    let dot_fn = format!("downloadDot_{id}");
    let svg_fn = format!("downloadSvg_{id}");

    let dot_button = if dot_code.is_some() {
        format!(r#"<button onclick="{dot_fn}()">Download DOT</button>"#)
    } else {
        String::new()
    };

    let dot_script = if let Some(dot) = dot_code {
        // Escape the DOT code for JavaScript
        let escaped_dot = dot
            .replace('\\', "\\\\")
            .replace('`', "\\`")
            .replace("${", "\\${");
        format!(
            r#"
        function {dot_fn}() {{
            const dot = `{escaped_dot}`;
            const blob = new Blob([dot], {{ type: 'text/plain' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'diagram.dot';
            a.click();
            URL.revokeObjectURL(url);
        }}"#
        )
    } else {
        String::new()
    };

    // Escape the Mermaid code for JavaScript
    let escaped_mermaid = mermaid_code
        .replace('\\', "\\\\")
        .replace('`', "\\`")
        .replace("${", "\\${");

    // SVG selector based on container_id
    let svg_selector = container_id.map_or_else(
        || ".mermaid svg".to_string(),
        |container| format!("#{container} svg"),
    );

    format!(
        r#"
    <div class="download-buttons">
        <button onclick="{mermaid_fn}()">Download Mermaid</button>
        <button onclick="{svg_fn}()">Download SVG</button>
        {dot_button}
    </div>
    <script>
        function {mermaid_fn}() {{
            const mermaid = `{escaped_mermaid}`;
            const blob = new Blob([mermaid], {{ type: 'text/plain' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'diagram.mmd';
            a.click();
            URL.revokeObjectURL(url);
        }}
        function {svg_fn}() {{
            const svg = document.querySelector('{svg_selector}');
            if (!svg) {{
                alert('SVG not yet rendered. Please wait for the diagram to load.');
                return;
            }}
            const svgData = new XMLSerializer().serializeToString(svg);
            const blob = new Blob([svgData], {{ type: 'image/svg+xml' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'diagram.svg';
            a.click();
            URL.revokeObjectURL(url);
        }}
        {dot_script}
    </script>"#
    )
}

/// CSS styles for download buttons
pub const DOWNLOAD_BUTTON_CSS: &str = r#"
        .download-buttons {
            margin: 15px 0;
            display: flex;
            gap: 10px;
        }
        .download-buttons button {
            padding: 8px 16px;
            background: #1976d2;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .download-buttons button:hover {
            background: #1565c0;
        }
"#;

// ============================================================================
// Backend Types
// ============================================================================

/// Backend identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendId {
    // ==========================================================================
    // THEOREM PROVERS & FORMAL VERIFICATION (Existing: 8)
    // ==========================================================================
    /// LEAN 4 theorem prover
    Lean4,
    /// TLA+ model checker (TLC)
    TlaPlus,
    /// Apalache symbolic model checker for TLA+
    Apalache,
    /// Kani Rust model checker
    Kani,
    /// Alloy relational analyzer
    Alloy,
    /// Isabelle/HOL theorem prover
    Isabelle,
    /// Coq proof assistant
    Coq,
    /// Dafny verification language
    Dafny,

    // ==========================================================================
    // PLATFORM API (Existing: 1)
    // ==========================================================================
    /// Platform API static checker generator (codegen)
    PlatformApi,

    // ==========================================================================
    // NEURAL NETWORK VERIFICATION (Existing: 3, New: 9)
    // ==========================================================================
    /// Marabou - SMT-based neural network verifier
    Marabou,
    /// alpha-beta-CROWN - GPU-accelerated neural network verifier
    AlphaBetaCrown,
    /// ERAN - Abstract interpretation for neural networks
    Eran,
    /// NNV - Neural network verification library
    NNV,
    /// nnenum - Enumeration-based neural network verifier
    Nnenum,
    /// VeriNet - Complete neural network verifier
    VeriNet,
    /// Venus - Complete DNN verifier
    Venus,
    /// DNNV - DNN verification framework
    DNNV,
    /// Auto-LiRPA - Linear relaxation based perturbation analysis
    AutoLiRPA,
    /// MN-BaB - Multi-neuron branch and bound
    MNBaB,
    /// Neurify - Symbolic interval propagation
    Neurify,
    /// ReluVal - Interval arithmetic for ReLU networks
    ReluVal,

    // ==========================================================================
    // ADVERSARIAL ROBUSTNESS (New: 5)
    // ==========================================================================
    /// ART - IBM Adversarial Robustness Toolbox
    ART,
    /// Foolbox - Adversarial attacks
    Foolbox,
    /// CleverHans - Adversarial examples library
    CleverHans,
    /// TextAttack - NLP adversarial attacks
    TextAttack,
    /// RobustBench - Robustness benchmarks
    RobustBench,

    // ==========================================================================
    // PROBABILISTIC VERIFICATION (Existing: 2)
    // ==========================================================================
    /// Storm - Probabilistic model checker
    Storm,
    /// PRISM - Probabilistic symbolic model checker
    Prism,

    // ==========================================================================
    // SECURITY PROTOCOL VERIFICATION (Existing: 3)
    // ==========================================================================
    /// Tamarin - Security protocol prover
    Tamarin,
    /// ProVerif - Cryptographic protocol verifier
    ProVerif,
    /// Verifpal - Modern security protocol verifier
    Verifpal,

    // ==========================================================================
    // MODEL CHECKERS (New: 8)
    // ==========================================================================
    /// SPIN - Protocol verification model checker
    SPIN,
    /// CBMC - C Bounded Model Checker
    CBMC,
    /// Infer - Facebook static analyzer
    Infer,
    /// KLEE - LLVM symbolic execution
    KLEE,
    /// NuSMV - Symbolic model checker
    NuSMV,
    /// CPAchecker - Configurable software verification
    CPAchecker,
    /// SeaHorn - LLVM-based verification framework
    SeaHorn,
    /// Frama-C - C analysis framework
    FramaC,
    /// Symbiotic - LLVM instrumentation + KLEE verification
    Symbiotic,
    /// 2LS - Template-based synthesis verifier for C
    TwoLS,

    // ==========================================================================
    // RUST FORMAL VERIFICATION (Existing: 3, New: 4)
    // ==========================================================================
    /// Verus - Rust verification via Z3
    Verus,
    /// Creusot - Rust verification via Why3
    Creusot,
    /// Prusti - Rust verification via Viper
    Prusti,
    /// Flux - Refinement types for Rust
    Flux,
    /// MIRAI - Facebook abstract interpreter for Rust
    Mirai,
    /// Rudra - Memory safety bug finder for unsafe Rust
    Rudra,
    /// Miri - Rust undefined behavior detector
    Miri,
    /// Haybale - LLVM symbolic execution for Rust
    Haybale,
    /// Crux-mir - Galois symbolic testing for Rust
    CruxMir,
    /// RustHorn - CHC-based Rust verifier
    RustHorn,
    /// RustBelt - Coq-based unsafe Rust verifier
    RustBelt,

    // ==========================================================================
    // GO VERIFICATION (New)
    // ==========================================================================
    /// Gobra - Viper-based Go verifier
    Gobra,

    // ==========================================================================
    // SMT SOLVERS (Existing: 2, New: 3)
    // ==========================================================================
    /// Z3 - Microsoft Research SMT solver
    Z3,
    /// CVC5 - Stanford/Iowa SMT solver
    Cvc5,
    /// Yices - SRI International SMT solver
    Yices,
    /// Boolector - Bit-vector and array SMT solver
    Boolector,
    /// MathSAT - FBK SMT solver with floating-point support
    MathSAT,

    // ==========================================================================
    // SAT SOLVERS (New: 3)
    // ==========================================================================
    /// MiniSat - Minimalistic CDCL SAT solver
    MiniSat,
    /// Glucose - High-performance SAT solver with LBD restarts
    Glucose,
    /// CaDiCaL - Award-winning modern SAT solver
    CaDiCaL,

    // ==========================================================================
    // RUST SANITIZERS & MEMORY TOOLS (New: 5)
    // ==========================================================================
    /// Address Sanitizer - Memory error detector
    AddressSanitizer,
    /// Memory Sanitizer - Uninitialized read detector
    MemorySanitizer,
    /// Thread Sanitizer - Data race detector
    ThreadSanitizer,
    /// Leak Sanitizer - Memory leak detector
    LeakSanitizer,
    /// Valgrind - Memory debugging and profiling toolkit
    Valgrind,

    // ==========================================================================
    // RUST CONCURRENCY TESTING (New: 4)
    // ==========================================================================
    /// Loom - Deterministic concurrency testing
    Loom,
    /// Shuttle - Randomized concurrency testing
    Shuttle,
    /// CDSChecker - C++11 memory model checker
    CDSChecker,
    /// GenMC - Stateless model checking for concurrency
    GenMC,

    // ==========================================================================
    // RUST FUZZING (New: 4)
    // ==========================================================================
    /// LibFuzzer via cargo-fuzz
    LibFuzzer,
    /// AFL - American Fuzzy Lop
    AFL,
    /// Honggfuzz - Coverage-guided fuzzer
    Honggfuzz,
    /// Bolero - Unified fuzzing/PBT framework
    Bolero,

    // ==========================================================================
    // RUST PROPERTY-BASED TESTING (New: 2)
    // ==========================================================================
    /// Proptest - Property-based testing
    Proptest,
    /// QuickCheck - Haskell-style property testing
    QuickCheck,

    // ==========================================================================
    // RUST STATIC ANALYSIS (New: 7)
    // ==========================================================================
    /// Clippy - Rust lint collection
    Clippy,
    /// cargo-semver-checks - API compatibility checker
    SemverChecks,
    /// cargo-geiger - Unsafe code auditor
    Geiger,
    /// cargo-audit - Security vulnerability scanner
    Audit,
    /// cargo-deny - Dependency policy enforcer
    Deny,
    /// cargo-vet - Supply chain auditor
    Vet,
    /// cargo-mutants - Mutation testing
    Mutants,

    // ==========================================================================
    // AI/ML OPTIMIZATION (New: 6)
    // ==========================================================================
    /// ONNX Runtime - Cross-platform ML inference
    ONNXRuntime,
    /// TensorRT - NVIDIA inference optimizer
    TensorRT,
    /// OpenVINO - Intel inference optimizer
    OpenVINO,
    /// Apache TVM - ML compiler stack
    TVM,
    /// IREE - Intermediate Representation Execution Environment
    IREE,
    /// Triton - GPU programming language
    Triton,

    // ==========================================================================
    // AI/ML COMPRESSION (New: 4)
    // ==========================================================================
    /// Intel Neural Compressor - Model quantization
    NeuralCompressor,
    /// NNCF - Neural Network Compression Framework
    NNCF,
    /// AIMET - AI Model Efficiency Toolkit
    AIMET,
    /// Brevitas - Quantization-aware training
    Brevitas,

    // ==========================================================================
    // DATA QUALITY (New: 4)
    // ==========================================================================
    /// Great Expectations - Data validation
    GreatExpectations,
    /// Deepchecks - ML validation suite
    Deepchecks,
    /// Evidently - ML monitoring
    Evidently,
    /// WhyLogs - Data profiling
    WhyLogs,

    // ==========================================================================
    // FAIRNESS & BIAS (New: 3)
    // ==========================================================================
    /// Fairlearn - Fairness assessment
    Fairlearn,
    /// AI Fairness 360 - IBM bias toolkit
    AIF360,
    /// Aequitas - Bias audit toolkit
    Aequitas,

    // ==========================================================================
    // INTERPRETABILITY (New: 5)
    // ==========================================================================
    /// SHAP - Shapley additive explanations
    SHAP,
    /// LIME - Local interpretable model explanations
    LIME,
    /// Captum - PyTorch interpretability
    Captum,
    /// InterpretML - Microsoft interpretability toolkit
    InterpretML,
    /// Alibi Explain - ML explanations
    Alibi,

    // ==========================================================================
    // LLM GUARDRAILS (New: 3)
    // ==========================================================================
    /// Guardrails AI - Output validation
    GuardrailsAI,
    /// NeMo Guardrails - NVIDIA guardrails
    NeMoGuardrails,
    /// Guidance - Structured LLM generation
    Guidance,

    // ==========================================================================
    // LLM EVALUATION (New: 5)
    // ==========================================================================
    /// Promptfoo - Prompt evaluation
    Promptfoo,
    /// TruLens - LLM evaluation
    TruLens,
    /// LangSmith - LLM testing
    LangSmith,
    /// Ragas - RAG evaluation
    Ragas,
    /// DeepEval - LLM testing framework
    DeepEval,

    // ==========================================================================
    // HALLUCINATION DETECTION (New: 2)
    // ==========================================================================
    /// SelfCheckGPT - Self-consistency hallucination check
    SelfCheckGPT,
    /// FactScore - Factual precision scoring
    FactScore,

    // ==========================================================================
    // DEPENDENTLY TYPED THEOREM PROVERS (New: 5)
    // ==========================================================================
    /// Agda - Dependently typed functional programming language
    Agda,
    /// Idris 2 - Dependently typed with quantitative type theory
    Idris,
    /// ACL2 - Applicative Common Lisp theorem prover
    ACL2,
    /// HOL4 - Higher-Order Logic theorem prover
    HOL4,
    /// F* - Proof-oriented ML with refinement types
    FStar,

    // ==========================================================================
    // ADDITIONAL THEOREM PROVERS (New: 5)
    // ==========================================================================
    /// HOL Light - HOL theorem prover (Cambridge)
    HOLLight,
    /// PVS - Prototype Verification System (SRI)
    PVS,
    /// Mizar - Mathematical proof system
    Mizar,
    /// Metamath - Proof verifier for formal systems
    Metamath,
    /// ATS - Applied Type System
    ATS,

    // ==========================================================================
    // ADDITIONAL SMT SOLVERS (New: 3)
    // ==========================================================================
    /// OpenSMT - Open-source SMT solver
    OpenSMT,
    /// veriT - SMT solver for proof production
    VeriT,
    /// Alt-Ergo - SMT solver for Why3
    AltErgo,

    // ==========================================================================
    // ADDITIONAL SAT SOLVERS (New: 2)
    // ==========================================================================
    /// Kissat - High-performance SAT solver
    Kissat,
    /// CryptoMiniSat - SAT solver with XOR reasoning
    CryptoMiniSat,

    // ==========================================================================
    // ADDITIONAL MODEL CHECKERS (New: 7)
    // ==========================================================================
    /// nuXmv - Extended symbolic model checker
    NuXmv,
    /// UPPAAL - Timed automata model checker
    UPPAAL,
    /// DIVINE - Parallel LTL model checker
    DIVINE,
    /// ESBMC - Efficient SMT-based bounded model checker
    ESBMC,
    /// Ultimate - Software model checker (Automizer/Taipan)
    Ultimate,
    /// SMACK - LLVM-to-Boogie translator
    SMACK,
    /// Java PathFinder - Java model checker
    JPF,

    // ==========================================================================
    // PROGRAM VERIFICATION FRAMEWORKS (New: 10)
    // ==========================================================================
    /// VCC - Microsoft C verifier
    VCC,
    /// VeriFast - C/Java separation logic verifier
    VeriFast,
    /// KeY - Java verification system
    KeY,
    /// OpenJML - Java Modeling Language tools
    OpenJML,
    /// Krakatoa - Java to Why3 translator
    Krakatoa,
    /// SPARK - Ada verification tools
    SPARK,
    /// Why3 - Verification platform
    Why3,
    /// Stainless - Scala verification
    Stainless,
    /// LiquidHaskell - Refinement types for Haskell
    LiquidHaskell,
    /// Boogie - Intermediate verification language
    Boogie,

    // ==========================================================================
    // DISTRIBUTED SYSTEMS VERIFICATION (New: 4)
    // ==========================================================================
    /// P - State machine programming language
    PLang,
    /// Ivy - Protocol verification language
    Ivy,
    /// mCRL2 - Process algebra toolset
    MCRL2,
    /// CADP - Construction and Analysis of Distributed Processes
    CADP,

    // ==========================================================================
    // CRYPTOGRAPHIC VERIFICATION (New: 3)
    // ==========================================================================
    /// EasyCrypt - Cryptographic proofs
    EasyCrypt,
    /// CryptoVerif - Computational security proofs
    CryptoVerif,
    /// Jasmin - Crypto implementation verification
    Jasmin,

    // ==========================================================================
    // HARDWARE VERIFICATION (New: 4)
    // ==========================================================================
    /// Yosys - Open source RTL synthesis
    Yosys,
    /// SymbiYosys - Formal verification with Yosys
    SymbiYosys,
    /// JasperGold - Cadence formal verification
    JasperGold,
    /// Cadence - EDA verification suite
    CadenceEDA,

    // ==========================================================================
    // SYMBOLIC EXECUTION & BINARY ANALYSIS (New: 7)
    // ==========================================================================
    /// Angr - Binary analysis platform (Python)
    Angr,
    /// Manticore - Symbolic execution engine
    Manticore,
    /// Triton - Dynamic binary analysis
    TritonDBA,
    /// BAP - Binary Analysis Platform (OCaml)
    Bap,
    /// Ghidra - NSA reverse engineering framework
    Ghidra,
    /// IsaBIL - Isabelle/HOL verification for BAP BIL IR
    IsaBIL,
    /// Soteria - Smart contract binary analysis
    Soteria,

    // ==========================================================================
    // ABSTRACT INTERPRETATION (New: 4)
    // ==========================================================================
    /// Astrée - Sound static analyzer
    Astree,
    /// Polyspace - MathWorks code verification
    Polyspace,
    /// CodeSonar - Binary and source code analysis
    CodeSonar,
    /// Frama-C EVA - Value analysis plugin
    FramaCEva,

    // ==========================================================================
    // RUST CODE COVERAGE (New: 3)
    // ==========================================================================
    /// cargo-tarpaulin - Code coverage tool
    Tarpaulin,
    /// cargo-llvm-cov - LLVM-based coverage
    LlvmCov,
    /// grcov - Mozilla coverage aggregator
    Grcov,

    // ==========================================================================
    // RUST TESTING INFRASTRUCTURE (New: 5)
    // ==========================================================================
    /// cargo-nextest - Fast test runner
    Nextest,
    /// cargo-insta - Snapshot testing
    Insta,
    /// rstest - Fixture-based testing
    Rstest,
    /// test-case - Parameterized tests
    TestCase,
    /// mockall - Mocking framework
    Mockall,

    // ==========================================================================
    // RUST DOCUMENTATION QUALITY (New: 3)
    // ==========================================================================
    /// cargo-deadlinks - Dead link checker
    Deadlinks,
    /// cargo-spellcheck - Documentation spell checker
    Spellcheck,
    /// cargo-rdme - README from rustdoc
    Rdme,

    // ==========================================================================
    // KANI FAST (Enhanced Kani Verification)
    // ==========================================================================
    /// Kani Fast - Enhanced Kani with k-induction, CHC solving, portfolio
    KaniFast,
}
// TOTAL: 181 backends

/// Property types that backends can verify
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyType {
    // ==========================================================================
    // THEOREM PROVING
    // ==========================================================================
    /// Mathematical theorem (LEAN 4, Coq, Isabelle)
    Theorem,
    /// Pre/post contract (Kani, Dafny, Verus, Creusot, Prusti)
    Contract,
    /// State invariant
    Invariant,
    /// Refinement relation between specs
    Refinement,

    // ==========================================================================
    // MODEL CHECKING
    // ==========================================================================
    /// Temporal logic property (TLA+, Apalache)
    Temporal,
    /// Probabilistic property (Storm, PRISM)
    Probabilistic,

    // ==========================================================================
    // NEURAL NETWORKS
    // ==========================================================================
    /// Neural network robustness property (Marabou, ERAN, alpha-beta-CROWN)
    NeuralRobustness,
    /// Neural network reachability (input-output constraints)
    NeuralReachability,
    /// Adversarial robustness (ART, Foolbox, CleverHans)
    AdversarialRobustness,

    // ==========================================================================
    // SECURITY
    // ==========================================================================
    /// Security/authentication property (Tamarin, ProVerif, Verifpal)
    SecurityProtocol,
    /// Platform API constraint (external API state machines for Metal, CUDA, etc.)
    PlatformApi,

    // ==========================================================================
    // RUST MEMORY SAFETY
    // ==========================================================================
    /// Memory safety (sanitizers, Miri, MIRAI, Rudra)
    MemorySafety,
    /// Undefined behavior detection (Miri)
    UndefinedBehavior,
    /// Data race detection (ThreadSanitizer, Loom, Shuttle)
    DataRace,
    /// Memory leak detection (LeakSanitizer)
    MemoryLeak,

    // ==========================================================================
    // RUST TESTING
    // ==========================================================================
    /// Fuzzing target (LibFuzzer, AFL, Honggfuzz, Bolero)
    Fuzzing,
    /// Property-based test (Proptest, QuickCheck)
    PropertyBased,
    /// Mutation testing (cargo-mutants)
    MutationTesting,

    // ==========================================================================
    // RUST STATIC ANALYSIS
    // ==========================================================================
    /// Lint/style check (Clippy)
    Lint,
    /// API compatibility (cargo-semver-checks)
    ApiCompatibility,
    /// Security vulnerability (cargo-audit)
    SecurityVulnerability,
    /// Dependency policy (cargo-deny)
    DependencyPolicy,
    /// Supply chain audit (cargo-vet)
    SupplyChain,
    /// Unsafe code audit (cargo-geiger)
    UnsafeAudit,

    // ==========================================================================
    // AI/ML OPTIMIZATION
    // ==========================================================================
    /// Model inference optimization (ONNX Runtime, TensorRT, TVM)
    ModelOptimization,
    /// Model compression/quantization (NNCF, Brevitas)
    ModelCompression,

    // ==========================================================================
    // AI/ML QUALITY
    // ==========================================================================
    /// Data quality validation (Great Expectations, Deepchecks)
    DataQuality,
    /// Model fairness (Fairlearn, AIF360, Aequitas)
    Fairness,
    /// Model interpretability (SHAP, LIME, Captum)
    Interpretability,

    // ==========================================================================
    // LLM
    // ==========================================================================
    /// LLM output guardrails (Guardrails AI, NeMo Guardrails)
    LLMGuardrails,
    /// LLM evaluation (Promptfoo, TruLens, DeepEval)
    LLMEvaluation,
    /// Hallucination detection (SelfCheckGPT, FactScore)
    HallucinationDetection,
}

impl PropertyType {
    /// Convert a USL `Property` to its corresponding `PropertyType` for backend selection.
    ///
    /// This is the canonical mapping from parsed USL properties to backend capability types.
    /// It centralizes the logic previously in `BackendSelector::property_type()`.
    ///
    /// # Semantic mappings
    /// - `Bisimulation` → `Refinement`: Bisimulation is behavioral refinement/equivalence
    /// - `Version` → `Contract`: Version compatibility is contract-based verification
    /// - `Capability` → `Contract`: Capability checking is contract-based verification
    /// - `Semantic` → `Theorem`: Semantic proofs are theorem-like
    ///
    /// # Example
    /// ```
    /// use dashprove_backends::PropertyType;
    /// use dashprove_usl::ast::Property;
    ///
    /// let property = Property::Theorem(dashprove_usl::ast::Theorem {
    ///     name: "example".to_string(),
    ///     body: dashprove_usl::ast::Expr::Bool(true),
    /// });
    /// assert_eq!(PropertyType::from_property(&property), PropertyType::Theorem);
    /// ```
    #[must_use]
    pub fn from_property(property: &dashprove_usl::ast::Property) -> Self {
        use dashprove_usl::ast::Property;

        match property {
            Property::Theorem(_) => Self::Theorem,
            Property::Temporal(_) => Self::Temporal,
            Property::Contract(_) => Self::Contract,
            Property::Invariant(_) => Self::Invariant,
            Property::Refinement(_) => Self::Refinement,
            Property::Probabilistic(_) => Self::Probabilistic,
            Property::Security(_) => Self::SecurityProtocol,
            Property::Semantic(_) => Self::Theorem,
            Property::PlatformApi(_) => Self::PlatformApi,
            Property::Bisimulation(_) => Self::Refinement, // Bisimulation is behavioral refinement
            Property::Version(_) => Self::Contract,        // Version compatibility is a contract
            Property::Capability(_) => Self::Contract,     // Capability is a form of contract
            Property::DistributedInvariant(_) => Self::Temporal, // Multi-agent invariants use temporal checking
            Property::DistributedTemporal(_) => Self::Temporal,  // Multi-agent temporal props
            Property::Composed(_) => Self::Theorem, // Composed theorems are theorem-like
            Property::ImprovementProposal(_) => Self::Contract, // Improvement proposals are contract-like
            Property::VerificationGate(_) => Self::Contract, // Verification gates are contract-like
            Property::Rollback(_) => Self::Temporal, // Rollback specs have temporal guarantees
        }
    }

    /// Parse a property type string to `PropertyType`.
    ///
    /// This parses string representations of property types, typically from
    /// feature extraction or serialized data. Case-insensitive.
    ///
    /// # Returns
    /// - `Some(PropertyType)` if the string matches a known property type
    /// - `None` if the string is unknown
    ///
    /// # Example
    /// ```
    /// use dashprove_backends::PropertyType;
    ///
    /// assert_eq!(PropertyType::parse("theorem"), Some(PropertyType::Theorem));
    /// assert_eq!(PropertyType::parse("TEMPORAL"), Some(PropertyType::Temporal));
    /// assert_eq!(PropertyType::parse("unknown"), None);
    /// ```
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "theorem" => Some(Self::Theorem),
            "contract" => Some(Self::Contract),
            "invariant" => Some(Self::Invariant),
            "refinement" => Some(Self::Refinement),
            "temporal" => Some(Self::Temporal),
            "probabilistic" => Some(Self::Probabilistic),
            "security" => Some(Self::SecurityProtocol),
            "semantic" => Some(Self::Theorem), // semantic proofs are theorem-like
            "platform_api" => Some(Self::PlatformApi),
            // Additional mappings for learning/reputation strings
            "memory_safety" => Some(Self::MemorySafety),
            "data_race" => Some(Self::DataRace),
            "fuzzing" => Some(Self::Fuzzing),
            "property_based" => Some(Self::PropertyBased),
            "lint" => Some(Self::Lint),
            "neural_robustness" => Some(Self::NeuralRobustness),
            _ => None,
        }
    }
}

/// Verification status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationStatus {
    /// Property was proven to hold
    Proven,
    /// Property was disproven (counterexample found)
    Disproven,
    /// Verification could not determine result
    Unknown {
        /// Reason why verification was inconclusive
        reason: String,
    },
    /// Partial verification (some cases verified)
    Partial {
        /// Percentage of cases verified (0.0 to 100.0)
        verified_percentage: f64,
    },
}

/// Result from a backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendResult {
    /// Which backend produced this result
    pub backend: BackendId,
    /// Verification status
    pub status: VerificationStatus,
    /// Proof output (if proven)
    pub proof: Option<String>,
    /// Structured counterexample (preferred) - contains parsed witness values
    pub counterexample: Option<StructuredCounterexample>,
    /// Diagnostic messages from the backend
    pub diagnostics: Vec<String>,
    /// Time taken for verification
    pub time_taken: Duration,
}

/// Backend health status
#[derive(Debug, Clone)]
pub enum HealthStatus {
    /// Backend is working normally
    Healthy,
    /// Backend is partially available with issues
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

/// Errors from verification backends
#[derive(Error, Debug)]
pub enum BackendError {
    /// Backend tool is not installed or not accessible
    #[error("Backend not available: {0}")]
    Unavailable(String),
    /// Compilation/parsing of the spec failed
    #[error("Compilation failed: {0}")]
    CompilationFailed(String),
    /// Verification execution failed
    #[error("Verification failed: {0}")]
    VerificationFailed(String),
    /// Verification exceeded configured timeout
    #[error("Timeout after {0:?}")]
    Timeout(Duration),
}

/// Trait that all verification backends must implement
#[async_trait]
pub trait VerificationBackend: Send + Sync {
    /// Unique identifier
    fn id(&self) -> BackendId;

    /// What property types this backend supports
    fn supports(&self) -> Vec<PropertyType>;

    /// Run verification
    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError>;

    /// Check if backend is available
    async fn health_check(&self) -> HealthStatus;
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::ast::{
        Bisimulation, CapabilitySpec, ComparisonOp, Contract, Expr, Invariant, PlatformApi,
        Probabilistic, Property, Refinement, Security, SemanticProperty, Temporal, TemporalExpr,
        Theorem, VersionSpec,
    };

    // ==========================================================================
    // PropertyType::from_property tests
    // ==========================================================================

    #[test]
    fn test_from_property_theorem() {
        let prop = Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        });
        assert_eq!(PropertyType::from_property(&prop), PropertyType::Theorem);
    }

    #[test]
    fn test_from_property_temporal() {
        let prop = Property::Temporal(Temporal {
            name: "test".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
            fairness: vec![],
        });
        assert_eq!(PropertyType::from_property(&prop), PropertyType::Temporal);
    }

    #[test]
    fn test_from_property_contract() {
        let prop = Property::Contract(Contract {
            type_path: vec!["Foo".to_string()],
            params: vec![],
            return_type: None,
            requires: vec![Expr::Bool(true)],
            ensures: vec![Expr::Bool(true)],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        });
        assert_eq!(PropertyType::from_property(&prop), PropertyType::Contract);
    }

    #[test]
    fn test_from_property_invariant() {
        let prop = Property::Invariant(Invariant {
            name: "test".to_string(),
            body: Expr::Bool(true),
        });
        assert_eq!(PropertyType::from_property(&prop), PropertyType::Invariant);
    }

    #[test]
    fn test_from_property_refinement() {
        let prop = Property::Refinement(Refinement {
            name: "test".to_string(),
            refines: "abstract_spec".to_string(),
            mappings: vec![],
            invariants: vec![],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![],
        });
        assert_eq!(PropertyType::from_property(&prop), PropertyType::Refinement);
    }

    #[test]
    fn test_from_property_probabilistic() {
        let prop = Property::Probabilistic(Probabilistic {
            name: "test".to_string(),
            condition: Expr::Bool(true),
            comparison: ComparisonOp::Ge,
            bound: 0.99,
        });
        assert_eq!(
            PropertyType::from_property(&prop),
            PropertyType::Probabilistic
        );
    }

    #[test]
    fn test_from_property_security() {
        let prop = Property::Security(Security {
            name: "test".to_string(),
            body: Expr::Bool(true),
        });
        assert_eq!(
            PropertyType::from_property(&prop),
            PropertyType::SecurityProtocol
        );
    }

    #[test]
    fn test_from_property_semantic() {
        let prop = Property::Semantic(SemanticProperty {
            name: "test".to_string(),
            body: Expr::Bool(true),
        });
        // Semantic maps to Theorem
        assert_eq!(PropertyType::from_property(&prop), PropertyType::Theorem);
    }

    #[test]
    fn test_from_property_platform_api() {
        let prop = Property::PlatformApi(PlatformApi {
            name: "Metal".to_string(),
            states: vec![],
        });
        assert_eq!(
            PropertyType::from_property(&prop),
            PropertyType::PlatformApi
        );
    }

    #[test]
    fn test_from_property_bisimulation() {
        let prop = Property::Bisimulation(Bisimulation {
            name: "test".to_string(),
            oracle: "oracle_impl".to_string(),
            subject: "subject_impl".to_string(),
            equivalent_on: vec!["api_requests".to_string()],
            tolerance: None,
            property: None,
        });
        // Bisimulation maps to Refinement
        assert_eq!(PropertyType::from_property(&prop), PropertyType::Refinement);
    }

    #[test]
    fn test_from_property_version() {
        let prop = Property::Version(VersionSpec {
            name: "V2".to_string(),
            improves: "V1".to_string(),
            capabilities: vec![],
            preserves: vec![],
        });
        // Version maps to Contract
        assert_eq!(PropertyType::from_property(&prop), PropertyType::Contract);
    }

    #[test]
    fn test_from_property_capability() {
        let prop = Property::Capability(CapabilitySpec {
            name: "test".to_string(),
            abilities: vec![],
            requires: vec![],
        });
        // Capability maps to Contract
        assert_eq!(PropertyType::from_property(&prop), PropertyType::Contract);
    }

    // ==========================================================================
    // PropertyType::parse tests
    // ==========================================================================

    #[test]
    fn test_parse_theorem() {
        assert_eq!(PropertyType::parse("theorem"), Some(PropertyType::Theorem));
        assert_eq!(PropertyType::parse("THEOREM"), Some(PropertyType::Theorem));
        assert_eq!(PropertyType::parse("Theorem"), Some(PropertyType::Theorem));
    }

    #[test]
    fn test_parse_contract() {
        assert_eq!(
            PropertyType::parse("contract"),
            Some(PropertyType::Contract)
        );
    }

    #[test]
    fn test_parse_invariant() {
        assert_eq!(
            PropertyType::parse("invariant"),
            Some(PropertyType::Invariant)
        );
    }

    #[test]
    fn test_parse_refinement() {
        assert_eq!(
            PropertyType::parse("refinement"),
            Some(PropertyType::Refinement)
        );
    }

    #[test]
    fn test_parse_temporal() {
        assert_eq!(
            PropertyType::parse("temporal"),
            Some(PropertyType::Temporal)
        );
    }

    #[test]
    fn test_parse_probabilistic() {
        assert_eq!(
            PropertyType::parse("probabilistic"),
            Some(PropertyType::Probabilistic)
        );
    }

    #[test]
    fn test_parse_security() {
        assert_eq!(
            PropertyType::parse("security"),
            Some(PropertyType::SecurityProtocol)
        );
    }

    #[test]
    fn test_parse_semantic() {
        // "semantic" maps to Theorem
        assert_eq!(PropertyType::parse("semantic"), Some(PropertyType::Theorem));
    }

    #[test]
    fn test_parse_platform_api() {
        assert_eq!(
            PropertyType::parse("platform_api"),
            Some(PropertyType::PlatformApi)
        );
    }

    #[test]
    fn test_parse_memory_safety() {
        assert_eq!(
            PropertyType::parse("memory_safety"),
            Some(PropertyType::MemorySafety)
        );
    }

    #[test]
    fn test_parse_data_race() {
        assert_eq!(
            PropertyType::parse("data_race"),
            Some(PropertyType::DataRace)
        );
    }

    #[test]
    fn test_parse_fuzzing() {
        assert_eq!(PropertyType::parse("fuzzing"), Some(PropertyType::Fuzzing));
    }

    #[test]
    fn test_parse_property_based() {
        assert_eq!(
            PropertyType::parse("property_based"),
            Some(PropertyType::PropertyBased)
        );
    }

    #[test]
    fn test_parse_lint() {
        assert_eq!(PropertyType::parse("lint"), Some(PropertyType::Lint));
    }

    #[test]
    fn test_parse_neural_robustness() {
        assert_eq!(
            PropertyType::parse("neural_robustness"),
            Some(PropertyType::NeuralRobustness)
        );
    }

    #[test]
    fn test_parse_unknown() {
        assert_eq!(PropertyType::parse("unknown_type"), None);
        assert_eq!(PropertyType::parse(""), None);
        assert_eq!(PropertyType::parse("invalid"), None);
    }

    #[test]
    fn download_buttons_use_unique_function_names() {
        let first = html_download_buttons("graph TB; A-->B;", None);
        let second = html_download_buttons("graph TB; C-->D;", Some("digraph { A -> B }"));

        let extract_id = |html: &str, marker: &str| -> String {
            html.split(marker)
                .nth(1)
                .map(|s| {
                    s.chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect::<String>()
                })
                .expect("marker not found")
        };

        let first_mermaid_id = extract_id(&first, "downloadMermaid_");
        let second_mermaid_id = extract_id(&second, "downloadMermaid_");
        assert_ne!(
            first_mermaid_id, second_mermaid_id,
            "Mermaid function IDs should be unique"
        );

        let first_svg_id = extract_id(&first, "downloadSvg_");
        let second_svg_id = extract_id(&second, "downloadSvg_");
        assert_ne!(
            first_svg_id, second_svg_id,
            "SVG function IDs should be unique"
        );

        // First call should NOT have DOT button, second should
        assert!(
            !first.contains("downloadDot_"),
            "First call should not have DOT button"
        );
        assert!(
            second.contains("downloadDot_"),
            "Second call should have DOT button"
        );
    }
}

// =============================================================================
// Kani Proofs for Self-Verification
// =============================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that VerificationStatus::Proven has a unique discriminant
    #[kani::proof]
    fn verify_status_proven_distinct() {
        let proven = VerificationStatus::Proven;
        let disproven = VerificationStatus::Disproven;
        let unknown = VerificationStatus::Unknown {
            reason: String::new(),
        };
        let partial = VerificationStatus::Partial {
            verified_percentage: 50.0,
        };

        kani::assert(
            std::mem::discriminant(&proven) != std::mem::discriminant(&disproven),
            "Proven != Disproven",
        );
        kani::assert(
            std::mem::discriminant(&proven) != std::mem::discriminant(&unknown),
            "Proven != Unknown",
        );
        kani::assert(
            std::mem::discriminant(&proven) != std::mem::discriminant(&partial),
            "Proven != Partial",
        );
    }

    /// Prove that all VerificationStatus variants have distinct discriminants
    #[kani::proof]
    fn verify_status_all_distinct() {
        let variants = [
            std::mem::discriminant(&VerificationStatus::Proven),
            std::mem::discriminant(&VerificationStatus::Disproven),
            std::mem::discriminant(&VerificationStatus::Unknown {
                reason: String::new(),
            }),
            std::mem::discriminant(&VerificationStatus::Partial {
                verified_percentage: 0.0,
            }),
        ];

        // All pairs should be distinct
        for i in 0..4 {
            for j in (i + 1)..4 {
                kani::assert(
                    variants[i] != variants[j],
                    "All VerificationStatus variants must be distinct",
                );
            }
        }
    }

    /// Prove that HealthStatus variants are distinct
    #[kani::proof]
    fn verify_health_status_distinct() {
        let healthy = std::mem::discriminant(&HealthStatus::Healthy);
        let unavailable = std::mem::discriminant(&HealthStatus::Unavailable {
            reason: String::new(),
        });
        let degraded = std::mem::discriminant(&HealthStatus::Degraded {
            reason: String::new(),
        });

        kani::assert(healthy != unavailable, "Healthy != Unavailable");
        kani::assert(healthy != degraded, "Healthy != Degraded");
        kani::assert(unavailable != degraded, "Unavailable != Degraded");
    }

    /// Prove that PropertyType variants are all distinct
    #[kani::proof]
    fn verify_property_type_distinct() {
        let variants = [
            std::mem::discriminant(&PropertyType::Invariant),
            std::mem::discriminant(&PropertyType::Temporal),
            std::mem::discriminant(&PropertyType::Contract),
            std::mem::discriminant(&PropertyType::Theorem),
            std::mem::discriminant(&PropertyType::Refinement),
        ];

        // All pairs should be distinct
        for i in 0..5 {
            for j in (i + 1)..5 {
                kani::assert(
                    variants[i] != variants[j],
                    "All PropertyType variants must be distinct",
                );
            }
        }
    }

    /// Prove that BackendResult preserves all field values
    #[kani::proof]
    fn verify_backend_result_preserves_fields() {
        let status = VerificationStatus::Unknown {
            reason: "timeout".to_string(),
        };
        let result = BackendResult {
            backend: BackendId::Lean4,
            status,
            proof: Some("proof".to_string()),
            counterexample: None,
            diagnostics: vec!["diag".to_string()],
            time_taken: Duration::from_secs(2),
        };

        kani::assert(
            result.backend == BackendId::Lean4,
            "backend should be preserved",
        );
        match &result.status {
            VerificationStatus::Unknown { reason } => {
                kani::assert(reason == "timeout", "reason should be preserved")
            }
            _ => kani::assert(false, "status must remain Unknown"),
        }
        kani::assert(result.proof.as_deref() == Some("proof"), "proof preserved");
        kani::assert(
            result.counterexample.is_none(),
            "counterexample can remain None",
        );
        kani::assert(
            result.diagnostics.len() == 1,
            "diagnostics length preserved",
        );
        kani::assert(
            result.time_taken == Duration::from_secs(2),
            "time_taken preserved",
        );
    }

    /// Prove that BackendError::Timeout keeps the duration value
    #[kani::proof]
    fn verify_backend_error_timeout_preserves_duration() {
        let duration = Duration::from_secs(5);
        let err = BackendError::Timeout(duration);
        match err {
            BackendError::Timeout(d) => {
                kani::assert(d == duration, "Timeout must preserve duration")
            }
            _ => kani::assert(false, "Unexpected BackendError variant"),
        }
    }

    /// Prove that download buttons render DOT button when code is provided
    #[kani::proof]
    fn verify_download_buttons_include_dot_when_present() {
        let html = html_download_buttons("graph TB; A-->B;", Some("digraph { A -> B }"));
        kani::assert(
            html.contains("Download DOT"),
            "DOT button should be present when dot_code is provided",
        );
    }

    /// Prove that a custom container id is used for SVG selection
    #[kani::proof]
    fn verify_download_buttons_use_container_selector() {
        let html = html_download_buttons_with_container(
            "graph TB; A-->B;",
            None,
            Some("custom_container"),
        );
        kani::assert(
            html.contains("#custom_container svg"),
            "Custom container id must be used in SVG selector",
        );
    }
}
