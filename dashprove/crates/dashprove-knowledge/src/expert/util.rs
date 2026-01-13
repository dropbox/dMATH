//! Utility functions for expert modules

use dashprove_backends::BackendId;

/// Get all backend IDs (169 total)
pub fn all_backends() -> Vec<BackendId> {
    vec![
        // ==========================================================================
        // THEOREM PROVERS & FORMAL VERIFICATION (8)
        // ==========================================================================
        BackendId::Lean4,
        BackendId::TlaPlus,
        BackendId::Apalache,
        BackendId::Kani,
        BackendId::Alloy,
        BackendId::Isabelle,
        BackendId::Coq,
        BackendId::Dafny,
        // ==========================================================================
        // PLATFORM API (1)
        // ==========================================================================
        BackendId::PlatformApi,
        // ==========================================================================
        // NEURAL NETWORK VERIFICATION (12)
        // ==========================================================================
        BackendId::Marabou,
        BackendId::AlphaBetaCrown,
        BackendId::Eran,
        BackendId::NNV,
        BackendId::Nnenum,
        BackendId::VeriNet,
        BackendId::Venus,
        BackendId::DNNV,
        BackendId::AutoLiRPA,
        BackendId::MNBaB,
        BackendId::Neurify,
        BackendId::ReluVal,
        // ==========================================================================
        // ADVERSARIAL ROBUSTNESS (5)
        // ==========================================================================
        BackendId::ART,
        BackendId::Foolbox,
        BackendId::CleverHans,
        BackendId::TextAttack,
        BackendId::RobustBench,
        // ==========================================================================
        // PROBABILISTIC VERIFICATION (2)
        // ==========================================================================
        BackendId::Storm,
        BackendId::Prism,
        // ==========================================================================
        // SECURITY PROTOCOL VERIFICATION (3)
        // ==========================================================================
        BackendId::Tamarin,
        BackendId::ProVerif,
        BackendId::Verifpal,
        // ==========================================================================
        // MODEL CHECKERS (8)
        // ==========================================================================
        BackendId::SPIN,
        BackendId::CBMC,
        BackendId::Infer,
        BackendId::KLEE,
        BackendId::NuSMV,
        BackendId::CPAchecker,
        BackendId::SeaHorn,
        BackendId::FramaC,
        // ==========================================================================
        // RUST FORMAL VERIFICATION (7)
        // ==========================================================================
        BackendId::Verus,
        BackendId::Creusot,
        BackendId::Prusti,
        BackendId::Flux,
        BackendId::Mirai,
        BackendId::Rudra,
        BackendId::Miri,
        // ==========================================================================
        // SMT SOLVERS (5)
        // ==========================================================================
        BackendId::Z3,
        BackendId::Cvc5,
        BackendId::Yices,
        BackendId::Boolector,
        BackendId::MathSAT,
        // ==========================================================================
        // SAT SOLVERS (3)
        // ==========================================================================
        BackendId::MiniSat,
        BackendId::Glucose,
        BackendId::CaDiCaL,
        // ==========================================================================
        // RUST SANITIZERS & MEMORY TOOLS (5)
        // ==========================================================================
        BackendId::AddressSanitizer,
        BackendId::MemorySanitizer,
        BackendId::ThreadSanitizer,
        BackendId::LeakSanitizer,
        BackendId::Valgrind,
        // ==========================================================================
        // RUST CONCURRENCY TESTING (4)
        // ==========================================================================
        BackendId::Loom,
        BackendId::Shuttle,
        BackendId::CDSChecker,
        BackendId::GenMC,
        // ==========================================================================
        // RUST FUZZING (4)
        // ==========================================================================
        BackendId::LibFuzzer,
        BackendId::AFL,
        BackendId::Honggfuzz,
        BackendId::Bolero,
        // ==========================================================================
        // RUST PROPERTY-BASED TESTING (2)
        // ==========================================================================
        BackendId::Proptest,
        BackendId::QuickCheck,
        // ==========================================================================
        // RUST STATIC ANALYSIS (7)
        // ==========================================================================
        BackendId::Clippy,
        BackendId::SemverChecks,
        BackendId::Geiger,
        BackendId::Audit,
        BackendId::Deny,
        BackendId::Vet,
        BackendId::Mutants,
        // ==========================================================================
        // AI/ML OPTIMIZATION (6)
        // ==========================================================================
        BackendId::ONNXRuntime,
        BackendId::TensorRT,
        BackendId::OpenVINO,
        BackendId::TVM,
        BackendId::IREE,
        BackendId::Triton,
        // ==========================================================================
        // AI/ML COMPRESSION (4)
        // ==========================================================================
        BackendId::NeuralCompressor,
        BackendId::NNCF,
        BackendId::AIMET,
        BackendId::Brevitas,
        // ==========================================================================
        // DATA QUALITY (4)
        // ==========================================================================
        BackendId::GreatExpectations,
        BackendId::Deepchecks,
        BackendId::Evidently,
        BackendId::WhyLogs,
        // ==========================================================================
        // FAIRNESS & BIAS (3)
        // ==========================================================================
        BackendId::Fairlearn,
        BackendId::AIF360,
        BackendId::Aequitas,
        // ==========================================================================
        // INTERPRETABILITY (5)
        // ==========================================================================
        BackendId::SHAP,
        BackendId::LIME,
        BackendId::Captum,
        BackendId::InterpretML,
        BackendId::Alibi,
        // ==========================================================================
        // LLM GUARDRAILS (3)
        // ==========================================================================
        BackendId::GuardrailsAI,
        BackendId::NeMoGuardrails,
        BackendId::Guidance,
        // ==========================================================================
        // LLM EVALUATION (5)
        // ==========================================================================
        BackendId::Promptfoo,
        BackendId::TruLens,
        BackendId::LangSmith,
        BackendId::Ragas,
        BackendId::DeepEval,
        // ==========================================================================
        // HALLUCINATION DETECTION (2)
        // ==========================================================================
        BackendId::SelfCheckGPT,
        BackendId::FactScore,
        // ==========================================================================
        // DEPENDENTLY TYPED THEOREM PROVERS (5)
        // ==========================================================================
        BackendId::Agda,
        BackendId::Idris,
        BackendId::ACL2,
        BackendId::HOL4,
        BackendId::FStar,
        // ==========================================================================
        // ADDITIONAL THEOREM PROVERS (5)
        // ==========================================================================
        BackendId::HOLLight,
        BackendId::PVS,
        BackendId::Mizar,
        BackendId::Metamath,
        BackendId::ATS,
        // ==========================================================================
        // ADDITIONAL SMT SOLVERS (3)
        // ==========================================================================
        BackendId::OpenSMT,
        BackendId::VeriT,
        BackendId::AltErgo,
        // ==========================================================================
        // ADDITIONAL SAT SOLVERS (2)
        // ==========================================================================
        BackendId::Kissat,
        BackendId::CryptoMiniSat,
        // ==========================================================================
        // ADDITIONAL MODEL CHECKERS (7)
        // ==========================================================================
        BackendId::NuXmv,
        BackendId::UPPAAL,
        BackendId::DIVINE,
        BackendId::ESBMC,
        BackendId::Ultimate,
        BackendId::SMACK,
        BackendId::JPF,
        // ==========================================================================
        // PROGRAM VERIFICATION FRAMEWORKS (10)
        // ==========================================================================
        BackendId::VCC,
        BackendId::VeriFast,
        BackendId::KeY,
        BackendId::OpenJML,
        BackendId::Krakatoa,
        BackendId::SPARK,
        BackendId::Why3,
        BackendId::Stainless,
        BackendId::LiquidHaskell,
        BackendId::Boogie,
        // ==========================================================================
        // DISTRIBUTED SYSTEMS VERIFICATION (4)
        // ==========================================================================
        BackendId::PLang,
        BackendId::Ivy,
        BackendId::MCRL2,
        BackendId::CADP,
        // ==========================================================================
        // CRYPTOGRAPHIC VERIFICATION (3)
        // ==========================================================================
        BackendId::EasyCrypt,
        BackendId::CryptoVerif,
        BackendId::Jasmin,
        // ==========================================================================
        // HARDWARE VERIFICATION (4)
        // ==========================================================================
        BackendId::Yosys,
        BackendId::SymbiYosys,
        BackendId::JasperGold,
        BackendId::CadenceEDA,
        // ==========================================================================
        // SYMBOLIC EXECUTION & BINARY ANALYSIS (7)
        // ==========================================================================
        BackendId::Angr,
        BackendId::Manticore,
        BackendId::TritonDBA,
        BackendId::Bap,
        BackendId::Ghidra,
        BackendId::IsaBIL,
        BackendId::Soteria,
        // ==========================================================================
        // ABSTRACT INTERPRETATION (4)
        // ==========================================================================
        BackendId::Astree,
        BackendId::Polyspace,
        BackendId::CodeSonar,
        BackendId::FramaCEva,
        // ==========================================================================
        // RUST CODE COVERAGE (3)
        // ==========================================================================
        BackendId::Tarpaulin,
        BackendId::LlvmCov,
        BackendId::Grcov,
        // ==========================================================================
        // RUST TESTING INFRASTRUCTURE (5)
        // ==========================================================================
        BackendId::Nextest,
        BackendId::Insta,
        BackendId::Rstest,
        BackendId::TestCase,
        BackendId::Mockall,
        // ==========================================================================
        // RUST DOCUMENTATION QUALITY (3)
        // ==========================================================================
        BackendId::Deadlinks,
        BackendId::Spellcheck,
        BackendId::Rdme,
        // ==========================================================================
        // ADDITIONAL RUST VERIFICATION (4)
        // ==========================================================================
        BackendId::Haybale,
        BackendId::CruxMir,
        BackendId::RustHorn,
        BackendId::RustBelt,
        // ==========================================================================
        // GO VERIFICATION (1)
        // ==========================================================================
        BackendId::Gobra,
        // ==========================================================================
        // ADDITIONAL C/C++ VERIFICATION (2)
        // ==========================================================================
        BackendId::Symbiotic,
        BackendId::TwoLS,
    ]
}

/// Get the tactic domain description for a backend
pub fn backend_tactic_domain(backend: BackendId) -> &'static str {
    match backend {
        BackendId::Lean4 => "Lean 4 theorem proving",
        BackendId::Coq => "Coq proof assistant",
        BackendId::Isabelle => "Isabelle/HOL",
        BackendId::TlaPlus => "TLA+ TLAPS",
        BackendId::Dafny => "Dafny auto-active verification",
        _ => "verification",
    }
}

/// Extract a tactic name from chunk content
pub fn extract_tactic_from_chunk(content: &str) -> String {
    // Try to extract a tactic name from content
    // Look for common patterns like "by <tactic>" or "apply <tactic>"
    let content_lower = content.to_lowercase();

    for tactic in [
        "simp",
        "rfl",
        "auto",
        "induction",
        "cases",
        "apply",
        "exact",
    ] {
        if content_lower.contains(tactic) {
            return tactic.to_string();
        }
    }

    "auto".to_string()
}

/// Convert a BackendId to the tool ID used in the ToolKnowledgeStore
///
/// Tool IDs in JSON files use lowercase with underscores for consistency.
pub fn backend_id_to_tool_id(backend: BackendId) -> String {
    match backend {
        // Theorem provers & formal verification
        BackendId::Lean4 => "lean4".to_string(),
        BackendId::TlaPlus => "tlaplus".to_string(),
        BackendId::Kani => "kani".to_string(),
        BackendId::Alloy => "alloy".to_string(),
        BackendId::Coq => "coq".to_string(),
        BackendId::Isabelle => "isabelle".to_string(),
        BackendId::Dafny => "dafny".to_string(),
        BackendId::Apalache => "apalache".to_string(),

        // Platform API
        BackendId::PlatformApi => "platform_api".to_string(),

        // SMT solvers
        BackendId::Z3 => "z3".to_string(),
        BackendId::Cvc5 => "cvc5".to_string(),
        BackendId::Yices => "yices".to_string(),
        BackendId::Boolector => "boolector".to_string(),
        BackendId::MathSAT => "mathsat".to_string(),

        // Rust formal verification
        BackendId::Verus => "verus".to_string(),
        BackendId::Creusot => "creusot".to_string(),
        BackendId::Prusti => "prusti".to_string(),
        BackendId::Miri => "miri".to_string(),
        BackendId::Flux => "flux".to_string(),
        BackendId::Mirai => "mirai".to_string(),
        BackendId::Rudra => "rudra".to_string(),

        // Neural network verification
        BackendId::Marabou => "marabou".to_string(),
        BackendId::AlphaBetaCrown => "alphabetacrown".to_string(),
        BackendId::Eran => "eran".to_string(),
        BackendId::NNV => "nnv".to_string(),
        BackendId::Nnenum => "nnenum".to_string(),
        BackendId::VeriNet => "verinet".to_string(),
        BackendId::Venus => "venus".to_string(),
        BackendId::DNNV => "dnnv".to_string(),
        BackendId::AutoLiRPA => "autolirpa".to_string(),
        BackendId::MNBaB => "mnbab".to_string(),
        BackendId::Neurify => "neurify".to_string(),
        BackendId::ReluVal => "reluval".to_string(),

        // Probabilistic verification
        BackendId::Storm => "storm".to_string(),
        BackendId::Prism => "prism".to_string(),

        // Security protocol verification
        BackendId::Tamarin => "tamarin".to_string(),
        BackendId::ProVerif => "proverif".to_string(),
        BackendId::Verifpal => "verifpal".to_string(),

        // Model checkers
        BackendId::SPIN => "spin".to_string(),
        BackendId::CBMC => "cbmc".to_string(),
        BackendId::Infer => "infer".to_string(),
        BackendId::KLEE => "klee".to_string(),
        BackendId::NuSMV => "nusmv".to_string(),
        BackendId::CPAchecker => "cpachecker".to_string(),
        BackendId::SeaHorn => "seahorn".to_string(),
        BackendId::FramaC => "framac".to_string(),

        // Rust sanitizers & memory tools
        BackendId::AddressSanitizer => "address_sanitizer".to_string(),
        BackendId::MemorySanitizer => "memory_sanitizer".to_string(),
        BackendId::ThreadSanitizer => "thread_sanitizer".to_string(),
        BackendId::LeakSanitizer => "leak_sanitizer".to_string(),
        BackendId::Valgrind => "valgrind".to_string(),

        // Rust concurrency testing
        BackendId::Loom => "loom".to_string(),
        BackendId::Shuttle => "shuttle".to_string(),
        BackendId::CDSChecker => "cdschecker".to_string(),
        BackendId::GenMC => "genmc".to_string(),

        // Rust fuzzing
        BackendId::LibFuzzer => "libfuzzer".to_string(),
        BackendId::AFL => "afl".to_string(),
        BackendId::Honggfuzz => "honggfuzz".to_string(),
        BackendId::Bolero => "bolero".to_string(),

        // Rust property-based testing
        BackendId::Proptest => "proptest".to_string(),
        BackendId::QuickCheck => "quickcheck".to_string(),

        // Rust static analysis
        BackendId::Clippy => "clippy".to_string(),
        BackendId::SemverChecks => "semver_checks".to_string(),
        BackendId::Geiger => "geiger".to_string(),
        BackendId::Audit => "cargo_audit".to_string(),
        BackendId::Deny => "cargo_deny".to_string(),
        BackendId::Vet => "cargo_vet".to_string(),
        BackendId::Mutants => "cargo_mutants".to_string(),

        // AI/ML optimization
        BackendId::ONNXRuntime => "onnxruntime".to_string(),
        BackendId::TensorRT => "tensorrt".to_string(),
        BackendId::OpenVINO => "openvino".to_string(),
        BackendId::TVM => "tvm".to_string(),
        BackendId::IREE => "iree".to_string(),
        BackendId::Triton => "triton".to_string(),

        // AI/ML compression
        BackendId::NeuralCompressor => "neural_compressor".to_string(),
        BackendId::NNCF => "nncf".to_string(),
        BackendId::AIMET => "aimet".to_string(),
        BackendId::Brevitas => "brevitas".to_string(),

        // Data quality
        BackendId::GreatExpectations => "great_expectations".to_string(),
        BackendId::Deepchecks => "deepchecks".to_string(),
        BackendId::Evidently => "evidently".to_string(),
        BackendId::WhyLogs => "whylogs".to_string(),

        // Fairness & bias
        BackendId::Fairlearn => "fairlearn".to_string(),
        BackendId::AIF360 => "aif360".to_string(),
        BackendId::Aequitas => "aequitas".to_string(),

        // Interpretability
        BackendId::SHAP => "shap".to_string(),
        BackendId::LIME => "lime".to_string(),
        BackendId::Captum => "captum".to_string(),
        BackendId::InterpretML => "interpretml".to_string(),
        BackendId::Alibi => "alibi".to_string(),

        // LLM guardrails
        BackendId::GuardrailsAI => "guardrails_ai".to_string(),
        BackendId::NeMoGuardrails => "nemo_guardrails".to_string(),
        BackendId::Guidance => "guidance".to_string(),

        // LLM evaluation
        BackendId::Promptfoo => "promptfoo".to_string(),
        BackendId::TruLens => "trulens".to_string(),
        BackendId::LangSmith => "langsmith".to_string(),
        BackendId::Ragas => "ragas".to_string(),
        BackendId::DeepEval => "deepeval".to_string(),

        // Hallucination detection
        BackendId::SelfCheckGPT => "selfcheckgpt".to_string(),
        BackendId::FactScore => "factscore".to_string(),

        // Adversarial robustness
        BackendId::ART => "art".to_string(),
        BackendId::Foolbox => "foolbox".to_string(),
        BackendId::CleverHans => "cleverhans".to_string(),
        BackendId::TextAttack => "textattack".to_string(),
        BackendId::RobustBench => "robustbench".to_string(),

        // SAT solvers
        BackendId::MiniSat => "minisat".to_string(),
        BackendId::Glucose => "glucose".to_string(),
        BackendId::CaDiCaL => "cadical".to_string(),

        // Dependently typed theorem provers
        BackendId::Agda => "agda".to_string(),
        BackendId::Idris => "idris".to_string(),
        BackendId::ACL2 => "acl2".to_string(),
        BackendId::HOL4 => "hol4".to_string(),
        BackendId::FStar => "fstar".to_string(),

        // Additional theorem provers
        BackendId::HOLLight => "hollight".to_string(),
        BackendId::PVS => "pvs".to_string(),
        BackendId::Mizar => "mizar".to_string(),
        BackendId::Metamath => "metamath".to_string(),
        BackendId::ATS => "ats".to_string(),

        // Additional SMT solvers
        BackendId::OpenSMT => "opensmt".to_string(),
        BackendId::VeriT => "verit".to_string(),
        BackendId::AltErgo => "altergo".to_string(),

        // Additional SAT solvers
        BackendId::Kissat => "kissat".to_string(),
        BackendId::CryptoMiniSat => "cryptominisat".to_string(),

        // Additional model checkers
        BackendId::NuXmv => "nuxmv".to_string(),
        BackendId::UPPAAL => "uppaal".to_string(),
        BackendId::DIVINE => "divine".to_string(),
        BackendId::ESBMC => "esbmc".to_string(),
        BackendId::Ultimate => "ultimate".to_string(),
        BackendId::SMACK => "smack".to_string(),
        BackendId::JPF => "jpf".to_string(),

        // Program verification frameworks
        BackendId::VCC => "vcc".to_string(),
        BackendId::VeriFast => "verifast".to_string(),
        BackendId::KeY => "key".to_string(),
        BackendId::OpenJML => "openjml".to_string(),
        BackendId::Krakatoa => "krakatoa".to_string(),
        BackendId::SPARK => "spark".to_string(),
        BackendId::Why3 => "why3".to_string(),
        BackendId::Stainless => "stainless".to_string(),
        BackendId::LiquidHaskell => "liquidhaskell".to_string(),
        BackendId::Boogie => "boogie".to_string(),

        // Distributed systems verification
        BackendId::PLang => "plang".to_string(),
        BackendId::Ivy => "ivy".to_string(),
        BackendId::MCRL2 => "mcrl2".to_string(),
        BackendId::CADP => "cadp".to_string(),

        // Cryptographic verification
        BackendId::EasyCrypt => "easycrypt".to_string(),
        BackendId::CryptoVerif => "cryptoverif".to_string(),
        BackendId::Jasmin => "jasmin".to_string(),

        // Hardware verification
        BackendId::Yosys => "yosys".to_string(),
        BackendId::SymbiYosys => "symbiyosys".to_string(),
        BackendId::JasperGold => "jaspergold".to_string(),
        BackendId::CadenceEDA => "cadence_eda".to_string(),

        // Symbolic execution and binary analysis
        BackendId::Angr => "angr".to_string(),
        BackendId::Manticore => "manticore".to_string(),
        BackendId::TritonDBA => "triton_dba".to_string(),
        BackendId::Bap => "bap".to_string(),
        BackendId::Ghidra => "ghidra".to_string(),
        BackendId::IsaBIL => "isabil".to_string(),
        BackendId::Soteria => "soteria".to_string(),

        // Abstract interpretation
        BackendId::Astree => "astree".to_string(),
        BackendId::Polyspace => "polyspace".to_string(),
        BackendId::CodeSonar => "codesonar".to_string(),
        BackendId::FramaCEva => "framac_eva".to_string(),

        // Rust code coverage
        BackendId::Tarpaulin => "tarpaulin".to_string(),
        BackendId::LlvmCov => "llvm_cov".to_string(),
        BackendId::Grcov => "grcov".to_string(),

        // Rust testing infrastructure
        BackendId::Nextest => "nextest".to_string(),
        BackendId::Insta => "insta".to_string(),
        BackendId::Rstest => "rstest".to_string(),
        BackendId::TestCase => "test_case".to_string(),
        BackendId::Mockall => "mockall".to_string(),

        // Rust documentation quality
        BackendId::Deadlinks => "deadlinks".to_string(),
        BackendId::Spellcheck => "spellcheck".to_string(),
        BackendId::Rdme => "rdme".to_string(),
        // Additional Rust verification
        BackendId::Haybale => "haybale".to_string(),
        BackendId::CruxMir => "crux-mir".to_string(),
        BackendId::RustHorn => "rusthorn".to_string(),
        BackendId::RustBelt => "rustbelt".to_string(),
        // Go verification
        BackendId::Gobra => "gobra".to_string(),
        // Additional C/C++ verification
        BackendId::Symbiotic => "symbiotic".to_string(),
        BackendId::TwoLS => "2ls".to_string(),
        // Kani Fast (enhanced Kani)
        BackendId::KaniFast => "kanifast".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_id_to_tool_id() {
        // Test common backends
        assert_eq!(backend_id_to_tool_id(BackendId::Lean4), "lean4");
        assert_eq!(backend_id_to_tool_id(BackendId::Kani), "kani");
        assert_eq!(backend_id_to_tool_id(BackendId::Z3), "z3");
        assert_eq!(backend_id_to_tool_id(BackendId::TlaPlus), "tlaplus");
        assert_eq!(backend_id_to_tool_id(BackendId::CBMC), "cbmc");
        assert_eq!(backend_id_to_tool_id(BackendId::Verus), "verus");
        assert_eq!(backend_id_to_tool_id(BackendId::Creusot), "creusot");
        assert_eq!(backend_id_to_tool_id(BackendId::Miri), "miri");
    }

    #[test]
    fn test_backend_id_to_tool_id_coverage() {
        // Ensure all backends in all_backends() have a tool_id mapping
        for backend in all_backends() {
            let tool_id = backend_id_to_tool_id(backend);
            assert!(
                !tool_id.is_empty(),
                "Backend {:?} should have non-empty tool_id",
                backend
            );
            // Tool IDs should be lowercase
            assert_eq!(
                tool_id,
                tool_id.to_lowercase(),
                "Tool ID for {:?} should be lowercase",
                backend
            );
        }
    }

    #[test]
    fn test_backend_tactic_domain() {
        assert_eq!(
            backend_tactic_domain(BackendId::Lean4),
            "Lean 4 theorem proving"
        );
        assert_eq!(backend_tactic_domain(BackendId::Coq), "Coq proof assistant");
        assert_eq!(backend_tactic_domain(BackendId::Z3), "verification");
    }

    #[test]
    fn test_extract_tactic_from_chunk() {
        assert_eq!(extract_tactic_from_chunk("use simp to simplify"), "simp");
        assert_eq!(
            extract_tactic_from_chunk("apply rfl for reflexivity"),
            "rfl"
        );
        assert_eq!(extract_tactic_from_chunk("try induction on n"), "induction");
        assert_eq!(extract_tactic_from_chunk("no tactic here"), "auto"); // fallback
    }

    #[test]
    fn test_all_backends_count() {
        let backends = all_backends();
        // Sanity check that we have a reasonable number of backends
        assert!(
            backends.len() > 100,
            "Should have at least 100 backends, got {}",
            backends.len()
        );
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: all_backends()
    // ==========================================================================

    #[test]
    fn test_all_backends_exact_count() {
        let backends = all_backends();
        // The exact count is 180 based on the function:
        // - Original 169 backends
        // - Plus 7 new backends added by MANAGER (Dec 25, 2025):
        //   Haybale, CruxMir, RustHorn, RustBelt (Rust), Gobra (Go), Symbiotic, TwoLS (C/C++)
        // - Plus 4 additional backends added subsequently
        assert_eq!(
            backends.len(),
            180,
            "all_backends() must return exactly 180 backends"
        );
    }

    #[test]
    fn test_all_backends_contains_theorem_provers() {
        let backends = all_backends();
        // Theorem provers - must be present
        assert!(backends.contains(&BackendId::Lean4), "Must contain Lean4");
        assert!(backends.contains(&BackendId::Coq), "Must contain Coq");
        assert!(
            backends.contains(&BackendId::Isabelle),
            "Must contain Isabelle"
        );
        assert!(backends.contains(&BackendId::Dafny), "Must contain Dafny");
    }

    #[test]
    fn test_all_backends_contains_smt_solvers() {
        let backends = all_backends();
        assert!(backends.contains(&BackendId::Z3), "Must contain Z3");
        assert!(backends.contains(&BackendId::Cvc5), "Must contain CVC5");
        assert!(backends.contains(&BackendId::Yices), "Must contain Yices");
    }

    #[test]
    fn test_all_backends_contains_rust_verification() {
        let backends = all_backends();
        assert!(backends.contains(&BackendId::Kani), "Must contain Kani");
        assert!(backends.contains(&BackendId::Verus), "Must contain Verus");
        assert!(
            backends.contains(&BackendId::Creusot),
            "Must contain Creusot"
        );
        assert!(backends.contains(&BackendId::Prusti), "Must contain Prusti");
        assert!(backends.contains(&BackendId::Miri), "Must contain Miri");
    }

    #[test]
    fn test_all_backends_contains_neural_network_verifiers() {
        let backends = all_backends();
        assert!(
            backends.contains(&BackendId::Marabou),
            "Must contain Marabou"
        );
        assert!(
            backends.contains(&BackendId::AlphaBetaCrown),
            "Must contain AlphaBetaCrown"
        );
        assert!(backends.contains(&BackendId::Eran), "Must contain Eran");
    }

    #[test]
    fn test_all_backends_contains_model_checkers() {
        let backends = all_backends();
        assert!(
            backends.contains(&BackendId::TlaPlus),
            "Must contain TlaPlus"
        );
        assert!(backends.contains(&BackendId::SPIN), "Must contain SPIN");
        assert!(backends.contains(&BackendId::CBMC), "Must contain CBMC");
    }

    #[test]
    fn test_all_backends_starts_with_lean4() {
        let backends = all_backends();
        assert_eq!(backends[0], BackendId::Lean4, "First backend must be Lean4");
    }

    #[test]
    fn test_all_backends_ends_with_twols() {
        let backends = all_backends();
        assert_eq!(
            backends[backends.len() - 1],
            BackendId::TwoLS,
            "Last backend must be TwoLS"
        );
    }

    #[test]
    fn test_all_backends_no_duplicates() {
        let backends = all_backends();
        let mut seen = std::collections::HashSet::new();
        for backend in backends {
            assert!(
                seen.insert(backend),
                "Backend {:?} appears more than once",
                backend
            );
        }
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: backend_tactic_domain()
    // ==========================================================================

    #[test]
    fn test_backend_tactic_domain_lean4() {
        assert_eq!(
            backend_tactic_domain(BackendId::Lean4),
            "Lean 4 theorem proving"
        );
    }

    #[test]
    fn test_backend_tactic_domain_coq() {
        assert_eq!(backend_tactic_domain(BackendId::Coq), "Coq proof assistant");
    }

    #[test]
    fn test_backend_tactic_domain_isabelle() {
        assert_eq!(backend_tactic_domain(BackendId::Isabelle), "Isabelle/HOL");
    }

    #[test]
    fn test_backend_tactic_domain_tlaplus() {
        assert_eq!(backend_tactic_domain(BackendId::TlaPlus), "TLA+ TLAPS");
    }

    #[test]
    fn test_backend_tactic_domain_dafny() {
        assert_eq!(
            backend_tactic_domain(BackendId::Dafny),
            "Dafny auto-active verification"
        );
    }

    #[test]
    fn test_backend_tactic_domain_fallback() {
        // All other backends should return "verification"
        assert_eq!(backend_tactic_domain(BackendId::Z3), "verification");
        assert_eq!(backend_tactic_domain(BackendId::Kani), "verification");
        assert_eq!(backend_tactic_domain(BackendId::Marabou), "verification");
        assert_eq!(backend_tactic_domain(BackendId::CBMC), "verification");
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: extract_tactic_from_chunk()
    // ==========================================================================

    #[test]
    fn test_extract_tactic_simp_is_first_in_priority() {
        // "simp" should match before other tactics
        assert_eq!(extract_tactic_from_chunk("simp"), "simp");
        assert_eq!(extract_tactic_from_chunk("Simp"), "simp");
        assert_eq!(extract_tactic_from_chunk("SIMP"), "simp");
    }

    #[test]
    fn test_extract_tactic_rfl_second_priority() {
        // "rfl" should match (but simp comes first if both present)
        assert_eq!(extract_tactic_from_chunk("use rfl"), "rfl");
        // If both simp and rfl present, simp wins (first in list)
        assert_eq!(extract_tactic_from_chunk("simp then rfl"), "simp");
    }

    #[test]
    fn test_extract_tactic_auto_third_priority() {
        assert_eq!(extract_tactic_from_chunk("try auto"), "auto");
        // If rfl and auto present, rfl wins
        assert_eq!(extract_tactic_from_chunk("rfl or auto"), "rfl");
    }

    #[test]
    fn test_extract_tactic_induction_fourth_priority() {
        assert_eq!(extract_tactic_from_chunk("use induction"), "induction");
        // If auto and induction present, auto wins
        assert_eq!(extract_tactic_from_chunk("auto or induction"), "auto");
    }

    #[test]
    fn test_extract_tactic_cases_fifth_priority() {
        assert_eq!(extract_tactic_from_chunk("use cases"), "cases");
        // If induction and cases present, induction wins
        assert_eq!(
            extract_tactic_from_chunk("induction then cases"),
            "induction"
        );
    }

    #[test]
    fn test_extract_tactic_apply_sixth_priority() {
        assert_eq!(extract_tactic_from_chunk("apply lemma"), "apply");
        // If cases and apply present, cases wins
        assert_eq!(extract_tactic_from_chunk("cases or apply"), "cases");
    }

    #[test]
    fn test_extract_tactic_exact_seventh_priority() {
        assert_eq!(extract_tactic_from_chunk("exact proof"), "exact");
        // If apply and exact present, apply wins
        assert_eq!(extract_tactic_from_chunk("apply then exact"), "apply");
    }

    #[test]
    fn test_extract_tactic_fallback_to_auto() {
        // When no tactic found, return "auto"
        assert_eq!(extract_tactic_from_chunk(""), "auto");
        assert_eq!(extract_tactic_from_chunk("xyz"), "auto");
        assert_eq!(extract_tactic_from_chunk("no matching tactics"), "auto");
    }

    #[test]
    fn test_extract_tactic_case_insensitive() {
        // The function converts to lowercase before matching
        assert_eq!(extract_tactic_from_chunk("SIMP"), "simp");
        assert_eq!(extract_tactic_from_chunk("RFL"), "rfl");
        assert_eq!(extract_tactic_from_chunk("AUTO"), "auto");
        assert_eq!(extract_tactic_from_chunk("INDUCTION"), "induction");
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: backend_id_to_tool_id() - all match arms
    // ==========================================================================

    #[test]
    fn test_tool_id_theorem_provers() {
        assert_eq!(backend_id_to_tool_id(BackendId::Lean4), "lean4");
        assert_eq!(backend_id_to_tool_id(BackendId::TlaPlus), "tlaplus");
        assert_eq!(backend_id_to_tool_id(BackendId::Kani), "kani");
        assert_eq!(backend_id_to_tool_id(BackendId::Alloy), "alloy");
        assert_eq!(backend_id_to_tool_id(BackendId::Coq), "coq");
        assert_eq!(backend_id_to_tool_id(BackendId::Isabelle), "isabelle");
        assert_eq!(backend_id_to_tool_id(BackendId::Dafny), "dafny");
        assert_eq!(backend_id_to_tool_id(BackendId::Apalache), "apalache");
    }

    #[test]
    fn test_tool_id_platform_api() {
        assert_eq!(
            backend_id_to_tool_id(BackendId::PlatformApi),
            "platform_api"
        );
    }

    #[test]
    fn test_tool_id_smt_solvers() {
        assert_eq!(backend_id_to_tool_id(BackendId::Z3), "z3");
        assert_eq!(backend_id_to_tool_id(BackendId::Cvc5), "cvc5");
        assert_eq!(backend_id_to_tool_id(BackendId::Yices), "yices");
        assert_eq!(backend_id_to_tool_id(BackendId::Boolector), "boolector");
        assert_eq!(backend_id_to_tool_id(BackendId::MathSAT), "mathsat");
    }

    #[test]
    fn test_tool_id_rust_formal_verification() {
        assert_eq!(backend_id_to_tool_id(BackendId::Verus), "verus");
        assert_eq!(backend_id_to_tool_id(BackendId::Creusot), "creusot");
        assert_eq!(backend_id_to_tool_id(BackendId::Prusti), "prusti");
        assert_eq!(backend_id_to_tool_id(BackendId::Miri), "miri");
        assert_eq!(backend_id_to_tool_id(BackendId::Flux), "flux");
        assert_eq!(backend_id_to_tool_id(BackendId::Mirai), "mirai");
        assert_eq!(backend_id_to_tool_id(BackendId::Rudra), "rudra");
    }

    #[test]
    fn test_tool_id_neural_network_verifiers() {
        assert_eq!(backend_id_to_tool_id(BackendId::Marabou), "marabou");
        assert_eq!(
            backend_id_to_tool_id(BackendId::AlphaBetaCrown),
            "alphabetacrown"
        );
        assert_eq!(backend_id_to_tool_id(BackendId::Eran), "eran");
        assert_eq!(backend_id_to_tool_id(BackendId::NNV), "nnv");
        assert_eq!(backend_id_to_tool_id(BackendId::Nnenum), "nnenum");
        assert_eq!(backend_id_to_tool_id(BackendId::VeriNet), "verinet");
        assert_eq!(backend_id_to_tool_id(BackendId::Venus), "venus");
        assert_eq!(backend_id_to_tool_id(BackendId::DNNV), "dnnv");
        assert_eq!(backend_id_to_tool_id(BackendId::AutoLiRPA), "autolirpa");
        assert_eq!(backend_id_to_tool_id(BackendId::MNBaB), "mnbab");
        assert_eq!(backend_id_to_tool_id(BackendId::Neurify), "neurify");
        assert_eq!(backend_id_to_tool_id(BackendId::ReluVal), "reluval");
    }

    #[test]
    fn test_tool_id_probabilistic_verification() {
        assert_eq!(backend_id_to_tool_id(BackendId::Storm), "storm");
        assert_eq!(backend_id_to_tool_id(BackendId::Prism), "prism");
    }

    #[test]
    fn test_tool_id_security_protocol() {
        assert_eq!(backend_id_to_tool_id(BackendId::Tamarin), "tamarin");
        assert_eq!(backend_id_to_tool_id(BackendId::ProVerif), "proverif");
        assert_eq!(backend_id_to_tool_id(BackendId::Verifpal), "verifpal");
    }

    #[test]
    fn test_tool_id_model_checkers() {
        assert_eq!(backend_id_to_tool_id(BackendId::SPIN), "spin");
        assert_eq!(backend_id_to_tool_id(BackendId::CBMC), "cbmc");
        assert_eq!(backend_id_to_tool_id(BackendId::Infer), "infer");
        assert_eq!(backend_id_to_tool_id(BackendId::KLEE), "klee");
        assert_eq!(backend_id_to_tool_id(BackendId::NuSMV), "nusmv");
        assert_eq!(backend_id_to_tool_id(BackendId::CPAchecker), "cpachecker");
        assert_eq!(backend_id_to_tool_id(BackendId::SeaHorn), "seahorn");
        assert_eq!(backend_id_to_tool_id(BackendId::FramaC), "framac");
    }

    #[test]
    fn test_tool_id_sanitizers() {
        assert_eq!(
            backend_id_to_tool_id(BackendId::AddressSanitizer),
            "address_sanitizer"
        );
        assert_eq!(
            backend_id_to_tool_id(BackendId::MemorySanitizer),
            "memory_sanitizer"
        );
        assert_eq!(
            backend_id_to_tool_id(BackendId::ThreadSanitizer),
            "thread_sanitizer"
        );
        assert_eq!(
            backend_id_to_tool_id(BackendId::LeakSanitizer),
            "leak_sanitizer"
        );
        assert_eq!(backend_id_to_tool_id(BackendId::Valgrind), "valgrind");
    }

    #[test]
    fn test_tool_id_concurrency_testing() {
        assert_eq!(backend_id_to_tool_id(BackendId::Loom), "loom");
        assert_eq!(backend_id_to_tool_id(BackendId::Shuttle), "shuttle");
        assert_eq!(backend_id_to_tool_id(BackendId::CDSChecker), "cdschecker");
        assert_eq!(backend_id_to_tool_id(BackendId::GenMC), "genmc");
    }

    #[test]
    fn test_tool_id_fuzzing() {
        assert_eq!(backend_id_to_tool_id(BackendId::LibFuzzer), "libfuzzer");
        assert_eq!(backend_id_to_tool_id(BackendId::AFL), "afl");
        assert_eq!(backend_id_to_tool_id(BackendId::Honggfuzz), "honggfuzz");
        assert_eq!(backend_id_to_tool_id(BackendId::Bolero), "bolero");
    }

    #[test]
    fn test_tool_id_property_based_testing() {
        assert_eq!(backend_id_to_tool_id(BackendId::Proptest), "proptest");
        assert_eq!(backend_id_to_tool_id(BackendId::QuickCheck), "quickcheck");
    }

    #[test]
    fn test_tool_id_static_analysis() {
        assert_eq!(backend_id_to_tool_id(BackendId::Clippy), "clippy");
        assert_eq!(
            backend_id_to_tool_id(BackendId::SemverChecks),
            "semver_checks"
        );
        assert_eq!(backend_id_to_tool_id(BackendId::Geiger), "geiger");
        assert_eq!(backend_id_to_tool_id(BackendId::Audit), "cargo_audit");
        assert_eq!(backend_id_to_tool_id(BackendId::Deny), "cargo_deny");
        assert_eq!(backend_id_to_tool_id(BackendId::Vet), "cargo_vet");
        assert_eq!(backend_id_to_tool_id(BackendId::Mutants), "cargo_mutants");
    }

    #[test]
    fn test_tool_id_ai_ml_optimization() {
        assert_eq!(backend_id_to_tool_id(BackendId::ONNXRuntime), "onnxruntime");
        assert_eq!(backend_id_to_tool_id(BackendId::TensorRT), "tensorrt");
        assert_eq!(backend_id_to_tool_id(BackendId::OpenVINO), "openvino");
        assert_eq!(backend_id_to_tool_id(BackendId::TVM), "tvm");
        assert_eq!(backend_id_to_tool_id(BackendId::IREE), "iree");
        assert_eq!(backend_id_to_tool_id(BackendId::Triton), "triton");
    }

    #[test]
    fn test_tool_id_ai_ml_compression() {
        assert_eq!(
            backend_id_to_tool_id(BackendId::NeuralCompressor),
            "neural_compressor"
        );
        assert_eq!(backend_id_to_tool_id(BackendId::NNCF), "nncf");
        assert_eq!(backend_id_to_tool_id(BackendId::AIMET), "aimet");
        assert_eq!(backend_id_to_tool_id(BackendId::Brevitas), "brevitas");
    }

    #[test]
    fn test_tool_id_data_quality() {
        assert_eq!(
            backend_id_to_tool_id(BackendId::GreatExpectations),
            "great_expectations"
        );
        assert_eq!(backend_id_to_tool_id(BackendId::Deepchecks), "deepchecks");
        assert_eq!(backend_id_to_tool_id(BackendId::Evidently), "evidently");
        assert_eq!(backend_id_to_tool_id(BackendId::WhyLogs), "whylogs");
    }

    #[test]
    fn test_tool_id_fairness_bias() {
        assert_eq!(backend_id_to_tool_id(BackendId::Fairlearn), "fairlearn");
        assert_eq!(backend_id_to_tool_id(BackendId::AIF360), "aif360");
        assert_eq!(backend_id_to_tool_id(BackendId::Aequitas), "aequitas");
    }

    #[test]
    fn test_tool_id_interpretability() {
        assert_eq!(backend_id_to_tool_id(BackendId::SHAP), "shap");
        assert_eq!(backend_id_to_tool_id(BackendId::LIME), "lime");
        assert_eq!(backend_id_to_tool_id(BackendId::Captum), "captum");
        assert_eq!(backend_id_to_tool_id(BackendId::InterpretML), "interpretml");
        assert_eq!(backend_id_to_tool_id(BackendId::Alibi), "alibi");
    }

    #[test]
    fn test_tool_id_llm_guardrails() {
        assert_eq!(
            backend_id_to_tool_id(BackendId::GuardrailsAI),
            "guardrails_ai"
        );
        assert_eq!(
            backend_id_to_tool_id(BackendId::NeMoGuardrails),
            "nemo_guardrails"
        );
        assert_eq!(backend_id_to_tool_id(BackendId::Guidance), "guidance");
    }

    #[test]
    fn test_tool_id_llm_evaluation() {
        assert_eq!(backend_id_to_tool_id(BackendId::Promptfoo), "promptfoo");
        assert_eq!(backend_id_to_tool_id(BackendId::TruLens), "trulens");
        assert_eq!(backend_id_to_tool_id(BackendId::LangSmith), "langsmith");
        assert_eq!(backend_id_to_tool_id(BackendId::Ragas), "ragas");
        assert_eq!(backend_id_to_tool_id(BackendId::DeepEval), "deepeval");
    }

    #[test]
    fn test_tool_id_hallucination_detection() {
        assert_eq!(
            backend_id_to_tool_id(BackendId::SelfCheckGPT),
            "selfcheckgpt"
        );
        assert_eq!(backend_id_to_tool_id(BackendId::FactScore), "factscore");
    }

    #[test]
    fn test_tool_id_adversarial_robustness() {
        assert_eq!(backend_id_to_tool_id(BackendId::ART), "art");
        assert_eq!(backend_id_to_tool_id(BackendId::Foolbox), "foolbox");
        assert_eq!(backend_id_to_tool_id(BackendId::CleverHans), "cleverhans");
        assert_eq!(backend_id_to_tool_id(BackendId::TextAttack), "textattack");
        assert_eq!(backend_id_to_tool_id(BackendId::RobustBench), "robustbench");
    }

    #[test]
    fn test_tool_id_sat_solvers() {
        assert_eq!(backend_id_to_tool_id(BackendId::MiniSat), "minisat");
        assert_eq!(backend_id_to_tool_id(BackendId::Glucose), "glucose");
        assert_eq!(backend_id_to_tool_id(BackendId::CaDiCaL), "cadical");
    }

    #[test]
    fn test_tool_id_dependently_typed() {
        assert_eq!(backend_id_to_tool_id(BackendId::Agda), "agda");
        assert_eq!(backend_id_to_tool_id(BackendId::Idris), "idris");
        assert_eq!(backend_id_to_tool_id(BackendId::ACL2), "acl2");
        assert_eq!(backend_id_to_tool_id(BackendId::HOL4), "hol4");
        assert_eq!(backend_id_to_tool_id(BackendId::FStar), "fstar");
    }

    #[test]
    fn test_tool_id_additional_theorem_provers() {
        assert_eq!(backend_id_to_tool_id(BackendId::HOLLight), "hollight");
        assert_eq!(backend_id_to_tool_id(BackendId::PVS), "pvs");
        assert_eq!(backend_id_to_tool_id(BackendId::Mizar), "mizar");
        assert_eq!(backend_id_to_tool_id(BackendId::Metamath), "metamath");
        assert_eq!(backend_id_to_tool_id(BackendId::ATS), "ats");
    }

    #[test]
    fn test_tool_id_additional_smt_solvers() {
        assert_eq!(backend_id_to_tool_id(BackendId::OpenSMT), "opensmt");
        assert_eq!(backend_id_to_tool_id(BackendId::VeriT), "verit");
        assert_eq!(backend_id_to_tool_id(BackendId::AltErgo), "altergo");
    }

    #[test]
    fn test_tool_id_additional_sat_solvers() {
        assert_eq!(backend_id_to_tool_id(BackendId::Kissat), "kissat");
        assert_eq!(
            backend_id_to_tool_id(BackendId::CryptoMiniSat),
            "cryptominisat"
        );
    }

    #[test]
    fn test_tool_id_additional_model_checkers() {
        assert_eq!(backend_id_to_tool_id(BackendId::NuXmv), "nuxmv");
        assert_eq!(backend_id_to_tool_id(BackendId::UPPAAL), "uppaal");
        assert_eq!(backend_id_to_tool_id(BackendId::DIVINE), "divine");
        assert_eq!(backend_id_to_tool_id(BackendId::ESBMC), "esbmc");
        assert_eq!(backend_id_to_tool_id(BackendId::Ultimate), "ultimate");
        assert_eq!(backend_id_to_tool_id(BackendId::SMACK), "smack");
        assert_eq!(backend_id_to_tool_id(BackendId::JPF), "jpf");
    }

    #[test]
    fn test_tool_id_program_verification() {
        assert_eq!(backend_id_to_tool_id(BackendId::VCC), "vcc");
        assert_eq!(backend_id_to_tool_id(BackendId::VeriFast), "verifast");
        assert_eq!(backend_id_to_tool_id(BackendId::KeY), "key");
        assert_eq!(backend_id_to_tool_id(BackendId::OpenJML), "openjml");
        assert_eq!(backend_id_to_tool_id(BackendId::Krakatoa), "krakatoa");
        assert_eq!(backend_id_to_tool_id(BackendId::SPARK), "spark");
        assert_eq!(backend_id_to_tool_id(BackendId::Why3), "why3");
        assert_eq!(backend_id_to_tool_id(BackendId::Stainless), "stainless");
        assert_eq!(
            backend_id_to_tool_id(BackendId::LiquidHaskell),
            "liquidhaskell"
        );
        assert_eq!(backend_id_to_tool_id(BackendId::Boogie), "boogie");
    }

    #[test]
    fn test_tool_id_distributed_systems() {
        assert_eq!(backend_id_to_tool_id(BackendId::PLang), "plang");
        assert_eq!(backend_id_to_tool_id(BackendId::Ivy), "ivy");
        assert_eq!(backend_id_to_tool_id(BackendId::MCRL2), "mcrl2");
        assert_eq!(backend_id_to_tool_id(BackendId::CADP), "cadp");
    }

    #[test]
    fn test_tool_id_cryptographic() {
        assert_eq!(backend_id_to_tool_id(BackendId::EasyCrypt), "easycrypt");
        assert_eq!(backend_id_to_tool_id(BackendId::CryptoVerif), "cryptoverif");
        assert_eq!(backend_id_to_tool_id(BackendId::Jasmin), "jasmin");
    }

    #[test]
    fn test_tool_id_hardware() {
        assert_eq!(backend_id_to_tool_id(BackendId::Yosys), "yosys");
        assert_eq!(backend_id_to_tool_id(BackendId::SymbiYosys), "symbiyosys");
        assert_eq!(backend_id_to_tool_id(BackendId::JasperGold), "jaspergold");
        assert_eq!(backend_id_to_tool_id(BackendId::CadenceEDA), "cadence_eda");
    }

    #[test]
    fn test_tool_id_symbolic_execution() {
        assert_eq!(backend_id_to_tool_id(BackendId::Angr), "angr");
        assert_eq!(backend_id_to_tool_id(BackendId::Manticore), "manticore");
        assert_eq!(backend_id_to_tool_id(BackendId::TritonDBA), "triton_dba");
    }

    #[test]
    fn test_tool_id_abstract_interpretation() {
        assert_eq!(backend_id_to_tool_id(BackendId::Astree), "astree");
        assert_eq!(backend_id_to_tool_id(BackendId::Polyspace), "polyspace");
        assert_eq!(backend_id_to_tool_id(BackendId::CodeSonar), "codesonar");
        assert_eq!(backend_id_to_tool_id(BackendId::FramaCEva), "framac_eva");
    }

    #[test]
    fn test_tool_id_code_coverage() {
        assert_eq!(backend_id_to_tool_id(BackendId::Tarpaulin), "tarpaulin");
        assert_eq!(backend_id_to_tool_id(BackendId::LlvmCov), "llvm_cov");
        assert_eq!(backend_id_to_tool_id(BackendId::Grcov), "grcov");
    }

    #[test]
    fn test_tool_id_testing_infrastructure() {
        assert_eq!(backend_id_to_tool_id(BackendId::Nextest), "nextest");
        assert_eq!(backend_id_to_tool_id(BackendId::Insta), "insta");
        assert_eq!(backend_id_to_tool_id(BackendId::Rstest), "rstest");
        assert_eq!(backend_id_to_tool_id(BackendId::TestCase), "test_case");
        assert_eq!(backend_id_to_tool_id(BackendId::Mockall), "mockall");
    }

    #[test]
    fn test_tool_id_documentation_quality() {
        assert_eq!(backend_id_to_tool_id(BackendId::Deadlinks), "deadlinks");
        assert_eq!(backend_id_to_tool_id(BackendId::Spellcheck), "spellcheck");
        assert_eq!(backend_id_to_tool_id(BackendId::Rdme), "rdme");
    }
}
