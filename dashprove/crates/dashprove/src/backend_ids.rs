use dashprove_backends::BackendId;
use serde::{Deserialize, Serialize};

// Kani proofs for backend ID conversions
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify default_backends returns the same length as SUPPORTED_BACKENDS
    #[kani::proof]
    fn verify_default_backends_length() {
        let defaults = default_backends();
        kani::assert(
            defaults.len() == SUPPORTED_BACKENDS.len(),
            "default_backends() should match SUPPORTED_BACKENDS length",
        );
    }

    /// Verify BackendIdParam to BackendId roundtrip for Lean4
    #[kani::proof]
    fn verify_backend_id_param_roundtrip_lean4() {
        let param = BackendIdParam::Lean4;
        let id: BackendId = param.clone().into();
        let back: BackendIdParam = id.into();
        kani::assert(
            matches!(back, BackendIdParam::Lean4),
            "Lean4 roundtrip should preserve variant",
        );
    }

    /// Verify BackendIdParam to BackendId roundtrip for TlaPlus
    #[kani::proof]
    fn verify_backend_id_param_roundtrip_tlaplus() {
        let param = BackendIdParam::TlaPlus;
        let id: BackendId = param.clone().into();
        let back: BackendIdParam = id.into();
        kani::assert(
            matches!(back, BackendIdParam::TlaPlus),
            "TlaPlus roundtrip should preserve variant",
        );
    }

    /// Verify backend_metric_label returns non-empty string for Lean4
    #[kani::proof]
    fn verify_metric_label_lean4_non_empty() {
        let label = backend_metric_label(BackendId::Lean4);
        kani::assert(!label.is_empty(), "Lean4 metric label should be non-empty");
    }

    /// Verify backend_metric_label returns non-empty string for Kani
    #[kani::proof]
    fn verify_metric_label_kani_non_empty() {
        let label = backend_metric_label(BackendId::Kani);
        kani::assert(!label.is_empty(), "Kani metric label should be non-empty");
    }

    /// Verify backend_metric_label returns "lean4" for Lean4
    #[kani::proof]
    fn verify_metric_label_lean4_value() {
        let label = backend_metric_label(BackendId::Lean4);
        kani::assert(label == "lean4", "Lean4 metric label should be 'lean4'");
    }

    /// Verify backend_metric_label returns "kani" for Kani
    #[kani::proof]
    fn verify_metric_label_kani_value() {
        let label = backend_metric_label(BackendId::Kani);
        kani::assert(label == "kani", "Kani metric label should be 'kani'");
    }

    /// Verify SUPPORTED_BACKENDS is not empty
    #[kani::proof]
    fn verify_supported_backends_not_empty() {
        kani::assert(
            !SUPPORTED_BACKENDS.is_empty(),
            "SUPPORTED_BACKENDS should not be empty",
        );
    }
}

// Utility macro to count the number of backends for fixed-size arrays
macro_rules! count_backends {
    ($($variant:ident),* $(,)?) => {
        <[()]>::len(&[$(count_backends!(@sub $variant)),*])
    };
    (@sub $variant:ident) => {
        ()
    };
}

// Declare BackendIdParam variants, conversions, and helpers from a single list
macro_rules! define_backend_params {
    ($($variant:ident => $slug:expr),+ $(,)?) => {
        /// Backend identifier exposed over the public API (string-based)
        #[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
        pub enum BackendIdParam {
            $(#[serde(rename = $slug)] $variant,)+
        }

        impl From<BackendIdParam> for BackendId {
            fn from(p: BackendIdParam) -> Self {
                match p {
                    $(BackendIdParam::$variant => BackendId::$variant,)+
                }
            }
        }

        impl From<BackendId> for BackendIdParam {
            fn from(id: BackendId) -> Self {
                match id {
                    $(BackendId::$variant => BackendIdParam::$variant,)+
                }
            }
        }

        /// All supported backends in deterministic order (mirrors BackendId)
        pub const SUPPORTED_BACKENDS: [BackendId; count_backends!($( $variant ),+)] = [
            $(BackendId::$variant,)+
        ];

        /// Default backend list when none is specified by the caller
        pub fn default_backends() -> Vec<BackendId> {
            SUPPORTED_BACKENDS.to_vec()
        }

        /// Stable metric/backend slug for dashboards and API clients
        pub fn backend_metric_label(backend: BackendId) -> &'static str {
            match backend {
                $(BackendId::$variant => $slug,)+
            }
        }
    };
}

define_backend_params! {
    // THEOREM PROVERS & FORMAL VERIFICATION
    Lean4 => "lean4",
    TlaPlus => "tlaplus",
    Apalache => "apalache",
    Kani => "kani",
    Alloy => "alloy",
    Isabelle => "isabelle",
    Coq => "coq",
    Dafny => "dafny",

    // PLATFORM API
    PlatformApi => "platform_api",

    // NEURAL NETWORK VERIFICATION
    Marabou => "marabou",
    AlphaBetaCrown => "alphabetacrown",
    Eran => "eran",
    NNV => "nnv",
    Nnenum => "nnenum",
    VeriNet => "verinet",
    Venus => "venus",
    DNNV => "dnnv",
    AutoLiRPA => "autolirpa",
    MNBaB => "mnbab",
    Neurify => "neurify",
    ReluVal => "reluval",

    // ADVERSARIAL ROBUSTNESS
    ART => "art",
    Foolbox => "foolbox",
    CleverHans => "cleverhans",
    TextAttack => "textattack",
    RobustBench => "robustbench",

    // PROBABILISTIC VERIFICATION
    Storm => "storm",
    Prism => "prism",

    // SECURITY PROTOCOL VERIFICATION
    Tamarin => "tamarin",
    ProVerif => "proverif",
    Verifpal => "verifpal",

    // MODEL CHECKERS
    SPIN => "spin",
    CBMC => "cbmc",
    Infer => "infer",
    KLEE => "klee",
    NuSMV => "nusmv",
    CPAchecker => "cpachecker",
    SeaHorn => "seahorn",
    FramaC => "framac",
    Symbiotic => "symbiotic",
    TwoLS => "twols",

    // RUST FORMAL VERIFICATION
    Verus => "verus",
    Creusot => "creusot",
    Prusti => "prusti",
    Flux => "flux",
    Mirai => "mirai",
    Rudra => "rudra",
    Miri => "miri",
    Haybale => "haybale",
    CruxMir => "cruxmir",
    RustHorn => "rusthorn",
    RustBelt => "rustbelt",

    // GO VERIFICATION
    Gobra => "gobra",

    // SMT SOLVERS
    Z3 => "z3",
    Cvc5 => "cvc5",
    Yices => "yices",
    Boolector => "boolector",
    MathSAT => "mathsat",

    // SAT SOLVERS
    MiniSat => "minisat",
    Glucose => "glucose",
    CaDiCaL => "cadical",

    // RUST SANITIZERS & MEMORY TOOLS
    AddressSanitizer => "addresssanitizer",
    MemorySanitizer => "memorysanitizer",
    ThreadSanitizer => "threadsanitizer",
    LeakSanitizer => "leaksanitizer",
    Valgrind => "valgrind",

    // RUST CONCURRENCY TESTING
    Loom => "loom",
    Shuttle => "shuttle",
    CDSChecker => "cdschecker",
    GenMC => "genmc",

    // RUST FUZZING
    LibFuzzer => "libfuzzer",
    AFL => "afl",
    Honggfuzz => "honggfuzz",
    Bolero => "bolero",

    // RUST PROPERTY-BASED TESTING
    Proptest => "proptest",
    QuickCheck => "quickcheck",

    // RUST STATIC ANALYSIS
    Clippy => "clippy",
    SemverChecks => "semverchecks",
    Geiger => "geiger",
    Audit => "audit",
    Deny => "deny",
    Vet => "vet",
    Mutants => "mutants",

    // AI/ML OPTIMIZATION
    ONNXRuntime => "onnxruntime",
    TensorRT => "tensorrt",
    OpenVINO => "openvino",
    TVM => "tvm",
    IREE => "iree",
    Triton => "triton",

    // AI/ML COMPRESSION
    NeuralCompressor => "neuralcompressor",
    NNCF => "nncf",
    AIMET => "aimet",
    Brevitas => "brevitas",

    // DATA QUALITY
    GreatExpectations => "greatexpectations",
    Deepchecks => "deepchecks",
    Evidently => "evidently",
    WhyLogs => "whylogs",

    // FAIRNESS & BIAS
    Fairlearn => "fairlearn",
    AIF360 => "aif360",
    Aequitas => "aequitas",

    // INTERPRETABILITY
    SHAP => "shap",
    LIME => "lime",
    Captum => "captum",
    InterpretML => "interpretml",
    Alibi => "alibi",

    // LLM GUARDRAILS
    GuardrailsAI => "guardrailsai",
    NeMoGuardrails => "nemoguardrails",
    Guidance => "guidance",

    // LLM EVALUATION
    Promptfoo => "promptfoo",
    TruLens => "trulens",
    LangSmith => "langsmith",
    Ragas => "ragas",
    DeepEval => "deepeval",

    // HALLUCINATION DETECTION
    SelfCheckGPT => "selfcheckgpt",
    FactScore => "factscore",

    // DEPENDENTLY TYPED THEOREM PROVERS
    Agda => "agda",
    Idris => "idris",
    ACL2 => "acl2",
    HOL4 => "hol4",
    FStar => "fstar",

    // ADDITIONAL THEOREM PROVERS
    HOLLight => "hollight",
    PVS => "pvs",
    Mizar => "mizar",
    Metamath => "metamath",
    ATS => "ats",

    // ADDITIONAL SMT SOLVERS
    OpenSMT => "opensmt",
    VeriT => "verit",
    AltErgo => "altergo",

    // ADDITIONAL SAT SOLVERS
    Kissat => "kissat",
    CryptoMiniSat => "cryptominisat",

    // ADDITIONAL MODEL CHECKERS
    NuXmv => "nuxmv",
    UPPAAL => "uppaal",
    DIVINE => "divine",
    ESBMC => "esbmc",
    Ultimate => "ultimate",
    SMACK => "smack",
    JPF => "jpf",

    // PROGRAM VERIFICATION FRAMEWORKS
    VCC => "vcc",
    VeriFast => "verifast",
    KeY => "key",
    OpenJML => "openjml",
    Krakatoa => "krakatoa",
    SPARK => "spark",
    Why3 => "why3",
    Stainless => "stainless",
    LiquidHaskell => "liquidhaskell",
    Boogie => "boogie",

    // DISTRIBUTED SYSTEMS VERIFICATION
    PLang => "plang",
    Ivy => "ivy",
    MCRL2 => "mcrl2",
    CADP => "cadp",

    // CRYPTOGRAPHIC VERIFICATION
    EasyCrypt => "easycrypt",
    CryptoVerif => "cryptoverif",
    Jasmin => "jasmin",

    // HARDWARE VERIFICATION
    Yosys => "yosys",
    SymbiYosys => "symbiyosys",
    JasperGold => "jaspergold",
    CadenceEDA => "cadenceeda",

    // SYMBOLIC EXECUTION & BINARY ANALYSIS
    Angr => "angr",
    Manticore => "manticore",
    TritonDBA => "tritondba",
    Bap => "bap",
    Ghidra => "ghidra",
    IsaBIL => "isabil",
    Soteria => "soteria",

    // ABSTRACT INTERPRETATION
    Astree => "astree",
    Polyspace => "polyspace",
    CodeSonar => "codesonar",
    FramaCEva => "framaceva",

    // RUST CODE COVERAGE
    Tarpaulin => "tarpaulin",
    LlvmCov => "llvmcov",
    Grcov => "grcov",

    // RUST TESTING FRAMEWORKS
    Nextest => "nextest",
    Insta => "insta",
    Rstest => "rstest",
    TestCase => "testcase",
    Mockall => "mockall",

    // RUST DOCUMENTATION TOOLS
    Deadlinks => "deadlinks",
    Spellcheck => "spellcheck",
    Rdme => "rdme",

    // KANI FAST (Enhanced Kani Verification)
    KaniFast => "kanifast",
}
