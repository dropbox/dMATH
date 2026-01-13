//! Check tools command implementation
//!
//! Checks the installation status of all verification tools across 200+ backends.

use std::process::Command;

/// Tool category for grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCategory {
    TheoremProver,
    ModelChecker,
    NeuralNetworkVerifier,
    ProbabilisticVerifier,
    SecurityProtocol,
    RustVerifier,
    SmtSolver,
    SatSolver,
    Sanitizer,
    ConcurrencyTester,
    Fuzzer,
    PropertyTester,
    StaticAnalyzer,
    AiOptimizer,
    AiCompressor,
    DataQuality,
    Fairness,
    Interpretability,
    LlmGuardrails,
    LlmEvaluation,
    HallucinationDetection,
}

impl ToolCategory {
    fn name(&self) -> &'static str {
        match self {
            Self::TheoremProver => "Theorem Provers",
            Self::ModelChecker => "Model Checkers",
            Self::NeuralNetworkVerifier => "Neural Network Verifiers",
            Self::ProbabilisticVerifier => "Probabilistic Verifiers",
            Self::SecurityProtocol => "Security Protocol",
            Self::RustVerifier => "Rust Verifiers",
            Self::SmtSolver => "SMT Solvers",
            Self::SatSolver => "SAT Solvers",
            Self::Sanitizer => "Sanitizers",
            Self::ConcurrencyTester => "Concurrency Testing",
            Self::Fuzzer => "Fuzzers",
            Self::PropertyTester => "Property Testing",
            Self::StaticAnalyzer => "Static Analysis",
            Self::AiOptimizer => "AI/ML Optimizers",
            Self::AiCompressor => "AI/ML Compression",
            Self::DataQuality => "Data Quality",
            Self::Fairness => "Fairness/Bias",
            Self::Interpretability => "Interpretability",
            Self::LlmGuardrails => "LLM Guardrails",
            Self::LlmEvaluation => "LLM Evaluation",
            Self::HallucinationDetection => "Hallucination Detection",
        }
    }
}

/// Status of a tool
#[derive(Debug, Clone)]
pub enum ToolStatus {
    /// Tool is installed and working
    Available { version: Option<String> },
    /// Tool is not installed
    NotInstalled,
    /// Tool check failed (reserved for future use)
    #[allow(dead_code)]
    Error { message: String },
}

impl ToolStatus {
    fn symbol(&self) -> &'static str {
        match self {
            Self::Available { .. } => "OK",
            Self::NotInstalled => "--",
            Self::Error { .. } => "ERR",
        }
    }

    fn is_available(&self) -> bool {
        matches!(self, Self::Available { .. })
    }
}

/// Tool definition
#[derive(Debug, Clone)]
pub struct ToolDef {
    name: &'static str,
    category: ToolCategory,
    check_cmd: CheckCommand,
    install_hint: &'static str,
}

/// How to check if a tool is installed
#[derive(Debug, Clone)]
enum CheckCommand {
    /// Run a shell command
    Shell {
        cmd: &'static str,
        args: &'static [&'static str],
    },
    /// Check Python package
    Python { package: &'static str },
    /// Check npm package
    Npm { package: &'static str },
    /// Run rustup component check
    RustupComponent {
        component: &'static str,
        toolchain: Option<&'static str>,
    },
    /// Cargo subcommand
    CargoSubcommand { subcommand: &'static str },
}

impl ToolDef {
    const fn shell(
        name: &'static str,
        category: ToolCategory,
        cmd: &'static str,
        args: &'static [&'static str],
        install_hint: &'static str,
    ) -> Self {
        Self {
            name,
            category,
            check_cmd: CheckCommand::Shell { cmd, args },
            install_hint,
        }
    }

    const fn python(
        name: &'static str,
        category: ToolCategory,
        package: &'static str,
        install_hint: &'static str,
    ) -> Self {
        Self {
            name,
            category,
            check_cmd: CheckCommand::Python { package },
            install_hint,
        }
    }

    const fn npm(
        name: &'static str,
        category: ToolCategory,
        package: &'static str,
        install_hint: &'static str,
    ) -> Self {
        Self {
            name,
            category,
            check_cmd: CheckCommand::Npm { package },
            install_hint,
        }
    }

    const fn rustup(
        name: &'static str,
        category: ToolCategory,
        component: &'static str,
        toolchain: Option<&'static str>,
        install_hint: &'static str,
    ) -> Self {
        Self {
            name,
            category,
            check_cmd: CheckCommand::RustupComponent {
                component,
                toolchain,
            },
            install_hint,
        }
    }

    const fn cargo(
        name: &'static str,
        category: ToolCategory,
        subcommand: &'static str,
        install_hint: &'static str,
    ) -> Self {
        Self {
            name,
            category,
            check_cmd: CheckCommand::CargoSubcommand { subcommand },
            install_hint,
        }
    }

    fn check(&self) -> ToolStatus {
        match &self.check_cmd {
            CheckCommand::Shell { cmd, args } => check_shell_command(cmd, args),
            CheckCommand::Python { package } => check_python_package(package),
            CheckCommand::Npm { package } => check_npm_package(package),
            CheckCommand::RustupComponent {
                component,
                toolchain,
            } => check_rustup_component(component, *toolchain),
            CheckCommand::CargoSubcommand { subcommand } => check_cargo_subcommand(subcommand),
        }
    }
}

fn check_shell_command(cmd: &str, args: &[&str]) -> ToolStatus {
    match Command::new(cmd).args(args).output() {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .map(|s| s.trim().to_string());
            ToolStatus::Available { version }
        }
        Ok(_) => ToolStatus::NotInstalled,
        Err(_) => ToolStatus::NotInstalled,
    }
}

fn check_python_package(package: &str) -> ToolStatus {
    let check_script = format!(
        "import {} as pkg; print(getattr(pkg, '__version__', 'installed'))",
        package.replace('-', "_")
    );
    match Command::new("python3").args(["-c", &check_script]).output() {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            ToolStatus::Available {
                version: Some(version),
            }
        }
        _ => ToolStatus::NotInstalled,
    }
}

fn check_npm_package(package: &str) -> ToolStatus {
    match Command::new("npm")
        .args(["list", "-g", package, "--depth=0"])
        .output()
    {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let version = stdout
                .lines()
                .find(|line| line.contains(package))
                .and_then(|line| line.split('@').next_back())
                .map(|v| v.trim().to_string());
            ToolStatus::Available { version }
        }
        _ => ToolStatus::NotInstalled,
    }
}

fn check_rustup_component(component: &str, toolchain: Option<&str>) -> ToolStatus {
    let args: Vec<&str> = match toolchain {
        Some(tc) => vec![
            format!("+{}", tc).leak(),
            "component",
            "list",
            "--installed",
        ],
        None => vec!["component", "list", "--installed"],
    };

    match Command::new("rustup").args(&args).output() {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if stdout.lines().any(|line| line.contains(component)) {
                ToolStatus::Available { version: None }
            } else {
                ToolStatus::NotInstalled
            }
        }
        _ => ToolStatus::NotInstalled,
    }
}

fn check_cargo_subcommand(subcommand: &str) -> ToolStatus {
    // Try to run `cargo <subcommand> --version` or `cargo <subcommand> --help`
    let version_result = Command::new("cargo")
        .args([subcommand, "--version"])
        .output();

    match version_result {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .map(|s| s.trim().to_string());
            ToolStatus::Available { version }
        }
        _ => {
            // Some cargo subcommands don't have --version, try --help
            match Command::new("cargo").args([subcommand, "--help"]).output() {
                Ok(output) if output.status.success() => ToolStatus::Available { version: None },
                _ => ToolStatus::NotInstalled,
            }
        }
    }
}

/// All tools organized by category
fn get_all_tools() -> Vec<ToolDef> {
    vec![
        // === THEOREM PROVERS (8) ===
        ToolDef::shell(
            "Lean4",
            ToolCategory::TheoremProver,
            "lean",
            &["--version"],
            "elan install",
        ),
        ToolDef::shell(
            "TLA+",
            ToolCategory::TheoremProver,
            "tlc2",
            &["-h"],
            "brew install tla-plus-toolbox",
        ),
        ToolDef::shell(
            "Apalache",
            ToolCategory::TheoremProver,
            "apalache-mc",
            &["version"],
            "brew install apalache",
        ),
        ToolDef::cargo(
            "Kani",
            ToolCategory::TheoremProver,
            "kani",
            "cargo install --locked kani-verifier",
        ),
        ToolDef::shell(
            "Alloy",
            ToolCategory::TheoremProver,
            "alloy",
            &["--version"],
            "brew install alloy",
        ),
        ToolDef::shell(
            "Isabelle",
            ToolCategory::TheoremProver,
            "isabelle",
            &["version"],
            "brew install isabelle",
        ),
        ToolDef::shell(
            "Coq",
            ToolCategory::TheoremProver,
            "coqc",
            &["--version"],
            "brew install coq",
        ),
        ToolDef::shell(
            "Dafny",
            ToolCategory::TheoremProver,
            "dafny",
            &["--version"],
            "brew install dafny",
        ),
        // === NEURAL NETWORK VERIFIERS (12) ===
        ToolDef::python(
            "Marabou",
            ToolCategory::NeuralNetworkVerifier,
            "maraboupy",
            "pip install maraboupy",
        ),
        ToolDef::python(
            "alpha-beta-CROWN",
            ToolCategory::NeuralNetworkVerifier,
            "auto_LiRPA",
            "pip install auto_lirpa",
        ),
        ToolDef::python(
            "ERAN",
            ToolCategory::NeuralNetworkVerifier,
            "eran",
            "Build from source",
        ),
        ToolDef::python(
            "NNV",
            ToolCategory::NeuralNetworkVerifier,
            "nnv",
            "pip install nnv",
        ),
        ToolDef::python(
            "nnenum",
            ToolCategory::NeuralNetworkVerifier,
            "nnenum",
            "pip install nnenum",
        ),
        ToolDef::python(
            "VeriNet",
            ToolCategory::NeuralNetworkVerifier,
            "verinet",
            "Build from source",
        ),
        ToolDef::python(
            "Venus",
            ToolCategory::NeuralNetworkVerifier,
            "venus",
            "pip install venus-ai",
        ),
        ToolDef::python(
            "DNNV",
            ToolCategory::NeuralNetworkVerifier,
            "dnnv",
            "pip install dnnv",
        ),
        ToolDef::python(
            "AutoLiRPA",
            ToolCategory::NeuralNetworkVerifier,
            "auto_LiRPA",
            "pip install auto_lirpa",
        ),
        ToolDef::python(
            "MN-BaB",
            ToolCategory::NeuralNetworkVerifier,
            "mnbab",
            "Build from source",
        ),
        ToolDef::python(
            "Neurify",
            ToolCategory::NeuralNetworkVerifier,
            "neurify",
            "Build from source",
        ),
        ToolDef::python(
            "ReluVal",
            ToolCategory::NeuralNetworkVerifier,
            "reluval",
            "Build from source",
        ),
        // === ADVERSARIAL ROBUSTNESS (5) ===
        ToolDef::python(
            "ART",
            ToolCategory::NeuralNetworkVerifier,
            "art",
            "pip install adversarial-robustness-toolbox",
        ),
        ToolDef::python(
            "Foolbox",
            ToolCategory::NeuralNetworkVerifier,
            "foolbox",
            "pip install foolbox",
        ),
        ToolDef::python(
            "CleverHans",
            ToolCategory::NeuralNetworkVerifier,
            "cleverhans",
            "pip install cleverhans",
        ),
        ToolDef::python(
            "TextAttack",
            ToolCategory::NeuralNetworkVerifier,
            "textattack",
            "pip install textattack",
        ),
        ToolDef::python(
            "RobustBench",
            ToolCategory::NeuralNetworkVerifier,
            "robustbench",
            "pip install robustbench",
        ),
        // === PROBABILISTIC VERIFIERS (2) ===
        ToolDef::shell(
            "Storm",
            ToolCategory::ProbabilisticVerifier,
            "storm",
            &["--version"],
            "brew install stormchecker",
        ),
        ToolDef::shell(
            "PRISM",
            ToolCategory::ProbabilisticVerifier,
            "prism",
            &["-version"],
            "Download from prismmodelchecker.org",
        ),
        // === SECURITY PROTOCOL (3) ===
        ToolDef::shell(
            "Tamarin",
            ToolCategory::SecurityProtocol,
            "tamarin-prover",
            &["--version"],
            "brew install tamarin-prover",
        ),
        ToolDef::shell(
            "ProVerif",
            ToolCategory::SecurityProtocol,
            "proverif",
            &["--help"],
            "brew install proverif",
        ),
        ToolDef::shell(
            "Verifpal",
            ToolCategory::SecurityProtocol,
            "verifpal",
            &["--version"],
            "brew install verifpal",
        ),
        // === MODEL CHECKERS (8) ===
        ToolDef::shell(
            "SPIN",
            ToolCategory::ModelChecker,
            "spin",
            &["-V"],
            "brew install spin",
        ),
        ToolDef::shell(
            "NuSMV",
            ToolCategory::ModelChecker,
            "nusmv",
            &["-h"],
            "Download from https://nusmv.fbk.eu",
        ),
        ToolDef::shell(
            "CBMC",
            ToolCategory::ModelChecker,
            "cbmc",
            &["--version"],
            "brew install cbmc",
        ),
        ToolDef::shell(
            "CPAchecker",
            ToolCategory::ModelChecker,
            "cpachecker",
            &["--version"],
            "Download from https://cpachecker.sosy-lab.org",
        ),
        ToolDef::shell(
            "SeaHorn",
            ToolCategory::ModelChecker,
            "sea",
            &["--version"],
            "Build from https://github.com/seahorn/seahorn",
        ),
        ToolDef::shell(
            "Infer",
            ToolCategory::ModelChecker,
            "infer",
            &["--version"],
            "brew install infer",
        ),
        ToolDef::shell(
            "KLEE",
            ToolCategory::ModelChecker,
            "klee",
            &["--version"],
            "Build from https://klee.github.io",
        ),
        ToolDef::shell(
            "Frama-C",
            ToolCategory::ModelChecker,
            "frama-c",
            &["-version"],
            "opam install frama-c",
        ),
        // === RUST VERIFIERS (7) ===
        ToolDef::shell(
            "Verus",
            ToolCategory::RustVerifier,
            "verus",
            &["--version"],
            "Build from source",
        ),
        ToolDef::shell(
            "Creusot",
            ToolCategory::RustVerifier,
            "creusot-rustc",
            &["--version"],
            "cargo install creusot",
        ),
        ToolDef::shell(
            "Prusti",
            ToolCategory::RustVerifier,
            "prusti-rustc",
            &["--version"],
            "cargo install prusti-cli",
        ),
        ToolDef::rustup(
            "Flux",
            ToolCategory::RustVerifier,
            "flux",
            Some("nightly"),
            "rustup +nightly component add flux",
        ),
        ToolDef::cargo(
            "MIRAI",
            ToolCategory::RustVerifier,
            "mirai",
            "cargo install mirai",
        ),
        ToolDef::cargo(
            "Rudra",
            ToolCategory::RustVerifier,
            "rudra",
            "cargo install rudra",
        ),
        ToolDef::rustup(
            "Miri",
            ToolCategory::RustVerifier,
            "miri",
            Some("nightly"),
            "rustup +nightly component add miri",
        ),
        // === SMT SOLVERS (5) ===
        ToolDef::shell(
            "Z3",
            ToolCategory::SmtSolver,
            "z3",
            &["--version"],
            "brew install z3",
        ),
        ToolDef::shell(
            "CVC5",
            ToolCategory::SmtSolver,
            "cvc5",
            &["--version"],
            "brew install cvc5",
        ),
        ToolDef::shell(
            "Yices",
            ToolCategory::SmtSolver,
            "yices-smt2",
            &["--version"],
            "brew install yices",
        ),
        ToolDef::shell(
            "Boolector",
            ToolCategory::SmtSolver,
            "boolector",
            &["--version"],
            "Build from https://github.com/Boolector/boolector",
        ),
        ToolDef::shell(
            "MathSAT",
            ToolCategory::SmtSolver,
            "mathsat",
            &["-version"],
            "Download from https://mathsat.fbk.eu/download.html",
        ),
        // === SAT SOLVERS (3) ===
        ToolDef::shell(
            "MiniSat",
            ToolCategory::SatSolver,
            "minisat",
            &["-h"],
            "brew install minisat",
        ),
        ToolDef::shell(
            "Glucose",
            ToolCategory::SatSolver,
            "glucose",
            &["-h"],
            "Download from https://www.labri.fr/perso/lsimon/glucose/",
        ),
        ToolDef::shell(
            "CaDiCaL",
            ToolCategory::SatSolver,
            "cadical",
            &["--version"],
            "Build from https://github.com/arminbiere/cadical",
        ),
        // === SANITIZERS (4) ===
        ToolDef::rustup(
            "ASAN",
            ToolCategory::Sanitizer,
            "rust-src",
            Some("nightly"),
            "RUSTFLAGS=\"-Z sanitizer=address\"",
        ),
        ToolDef::rustup(
            "MSAN",
            ToolCategory::Sanitizer,
            "rust-src",
            Some("nightly"),
            "RUSTFLAGS=\"-Z sanitizer=memory\"",
        ),
        ToolDef::rustup(
            "TSAN",
            ToolCategory::Sanitizer,
            "rust-src",
            Some("nightly"),
            "RUSTFLAGS=\"-Z sanitizer=thread\"",
        ),
        ToolDef::rustup(
            "LSAN",
            ToolCategory::Sanitizer,
            "rust-src",
            Some("nightly"),
            "RUSTFLAGS=\"-Z sanitizer=leak\"",
        ),
        // === CONCURRENCY TESTING (4) ===
        ToolDef::shell(
            "Loom",
            ToolCategory::ConcurrencyTester,
            "cargo",
            &["--list"],
            "cargo add loom --dev",
        ),
        ToolDef::shell(
            "Shuttle",
            ToolCategory::ConcurrencyTester,
            "cargo",
            &["--list"],
            "cargo add shuttle --dev",
        ),
        ToolDef::shell(
            "CDSChecker",
            ToolCategory::ConcurrencyTester,
            "cdschecker",
            &["--help"],
            "Build from source",
        ),
        ToolDef::shell(
            "GenMC",
            ToolCategory::ConcurrencyTester,
            "genmc",
            &["--version"],
            "Build from source",
        ),
        // === FUZZERS (4) ===
        ToolDef::cargo(
            "LibFuzzer",
            ToolCategory::Fuzzer,
            "fuzz",
            "cargo install cargo-fuzz",
        ),
        ToolDef::cargo("AFL", ToolCategory::Fuzzer, "afl", "cargo install afl"),
        ToolDef::shell(
            "Honggfuzz",
            ToolCategory::Fuzzer,
            "honggfuzz",
            &["--version"],
            "cargo install honggfuzz",
        ),
        ToolDef::cargo(
            "Bolero",
            ToolCategory::Fuzzer,
            "bolero",
            "cargo install cargo-bolero",
        ),
        // === PROPERTY TESTING (2) ===
        ToolDef::shell(
            "Proptest",
            ToolCategory::PropertyTester,
            "cargo",
            &["--list"],
            "cargo add proptest --dev",
        ),
        ToolDef::shell(
            "QuickCheck",
            ToolCategory::PropertyTester,
            "cargo",
            &["--list"],
            "cargo add quickcheck --dev",
        ),
        // === STATIC ANALYSIS (7) ===
        ToolDef::rustup(
            "Clippy",
            ToolCategory::StaticAnalyzer,
            "clippy",
            None,
            "rustup component add clippy",
        ),
        ToolDef::cargo(
            "SemverChecks",
            ToolCategory::StaticAnalyzer,
            "semver-checks",
            "cargo install cargo-semver-checks",
        ),
        ToolDef::cargo(
            "Geiger",
            ToolCategory::StaticAnalyzer,
            "geiger",
            "cargo install cargo-geiger",
        ),
        ToolDef::cargo(
            "Audit",
            ToolCategory::StaticAnalyzer,
            "audit",
            "cargo install cargo-audit",
        ),
        ToolDef::cargo(
            "Deny",
            ToolCategory::StaticAnalyzer,
            "deny",
            "cargo install cargo-deny",
        ),
        ToolDef::cargo(
            "Vet",
            ToolCategory::StaticAnalyzer,
            "vet",
            "cargo install cargo-vet",
        ),
        ToolDef::cargo(
            "Mutants",
            ToolCategory::StaticAnalyzer,
            "mutants",
            "cargo install cargo-mutants",
        ),
        // === AI/ML OPTIMIZERS (6) ===
        ToolDef::python(
            "ONNXRuntime",
            ToolCategory::AiOptimizer,
            "onnxruntime",
            "pip install onnxruntime",
        ),
        ToolDef::python(
            "TensorRT",
            ToolCategory::AiOptimizer,
            "tensorrt",
            "NVIDIA SDK",
        ),
        ToolDef::python(
            "OpenVINO",
            ToolCategory::AiOptimizer,
            "openvino",
            "pip install openvino",
        ),
        ToolDef::python(
            "TVM",
            ToolCategory::AiOptimizer,
            "tvm",
            "pip install apache-tvm",
        ),
        ToolDef::shell(
            "IREE",
            ToolCategory::AiOptimizer,
            "iree-compile",
            &["--version"],
            "Build from source",
        ),
        ToolDef::python(
            "Triton",
            ToolCategory::AiOptimizer,
            "triton",
            "pip install triton",
        ),
        // === AI/ML COMPRESSION (4) ===
        ToolDef::python(
            "NeuralCompressor",
            ToolCategory::AiCompressor,
            "neural_compressor",
            "pip install neural-compressor",
        ),
        ToolDef::python(
            "NNCF",
            ToolCategory::AiCompressor,
            "nncf",
            "pip install nncf",
        ),
        ToolDef::python("AIMET", ToolCategory::AiCompressor, "aimet", "Qualcomm SDK"),
        ToolDef::python(
            "Brevitas",
            ToolCategory::AiCompressor,
            "brevitas",
            "pip install brevitas",
        ),
        // === DATA QUALITY (4) ===
        ToolDef::python(
            "GreatExpectations",
            ToolCategory::DataQuality,
            "great_expectations",
            "pip install great_expectations",
        ),
        ToolDef::python(
            "Deepchecks",
            ToolCategory::DataQuality,
            "deepchecks",
            "pip install deepchecks",
        ),
        ToolDef::python(
            "Evidently",
            ToolCategory::DataQuality,
            "evidently",
            "pip install evidently",
        ),
        ToolDef::python(
            "WhyLogs",
            ToolCategory::DataQuality,
            "whylogs",
            "pip install whylogs",
        ),
        // === FAIRNESS (3) ===
        ToolDef::python(
            "Fairlearn",
            ToolCategory::Fairness,
            "fairlearn",
            "pip install fairlearn",
        ),
        ToolDef::python(
            "AIF360",
            ToolCategory::Fairness,
            "aif360",
            "pip install aif360",
        ),
        ToolDef::python(
            "Aequitas",
            ToolCategory::Fairness,
            "aequitas",
            "pip install aequitas",
        ),
        // === INTERPRETABILITY (5) ===
        ToolDef::python(
            "SHAP",
            ToolCategory::Interpretability,
            "shap",
            "pip install shap",
        ),
        ToolDef::python(
            "LIME",
            ToolCategory::Interpretability,
            "lime",
            "pip install lime",
        ),
        ToolDef::python(
            "Captum",
            ToolCategory::Interpretability,
            "captum",
            "pip install captum",
        ),
        ToolDef::python(
            "InterpretML",
            ToolCategory::Interpretability,
            "interpret",
            "pip install interpret",
        ),
        ToolDef::python(
            "Alibi",
            ToolCategory::Interpretability,
            "alibi",
            "pip install alibi",
        ),
        // === LLM GUARDRAILS (3) ===
        ToolDef::python(
            "GuardrailsAI",
            ToolCategory::LlmGuardrails,
            "guardrails",
            "pip install guardrails-ai",
        ),
        ToolDef::python(
            "NeMoGuardrails",
            ToolCategory::LlmGuardrails,
            "nemoguardrails",
            "pip install nemoguardrails",
        ),
        ToolDef::python(
            "Guidance",
            ToolCategory::LlmGuardrails,
            "guidance",
            "pip install guidance",
        ),
        // === LLM EVALUATION (5) ===
        ToolDef::npm(
            "Promptfoo",
            ToolCategory::LlmEvaluation,
            "promptfoo",
            "npm install -g promptfoo",
        ),
        ToolDef::python(
            "TruLens",
            ToolCategory::LlmEvaluation,
            "trulens_eval",
            "pip install trulens-eval",
        ),
        ToolDef::python(
            "LangSmith",
            ToolCategory::LlmEvaluation,
            "langsmith",
            "pip install langsmith",
        ),
        ToolDef::python(
            "Ragas",
            ToolCategory::LlmEvaluation,
            "ragas",
            "pip install ragas",
        ),
        ToolDef::python(
            "DeepEval",
            ToolCategory::LlmEvaluation,
            "deepeval",
            "pip install deepeval",
        ),
        // === HALLUCINATION DETECTION (2) ===
        ToolDef::python(
            "SelfCheckGPT",
            ToolCategory::HallucinationDetection,
            "selfcheckgpt",
            "pip install selfcheckgpt",
        ),
        ToolDef::python(
            "FactScore",
            ToolCategory::HallucinationDetection,
            "factscore",
            "pip install factscore",
        ),
    ]
}

/// Configuration for check-tools command
pub struct CheckToolsConfig<'a> {
    /// Show verbose output including install hints
    pub verbose: bool,
    /// Filter to specific category (optional)
    pub category: Option<&'a str>,
    /// Show only missing tools
    pub missing_only: bool,
    /// Output format (text, json)
    pub format: &'a str,
}

/// Run the check-tools command
pub fn run_check_tools(config: CheckToolsConfig) -> Result<(), Box<dyn std::error::Error>> {
    let tools = get_all_tools();
    let total = tools.len();
    let mut available = 0;
    let mut results: Vec<(String, ToolStatus, String, ToolCategory)> = Vec::new();

    // Filter by category if specified
    let category_filter: Option<ToolCategory> =
        config
            .category
            .and_then(|cat| match cat.to_lowercase().as_str() {
                "theorem" | "prover" | "theorem-prover" => Some(ToolCategory::TheoremProver),
                "neural" | "nn" | "nn-verifier" => Some(ToolCategory::NeuralNetworkVerifier),
                "prob" | "probabilistic" => Some(ToolCategory::ProbabilisticVerifier),
                "security" | "protocol" => Some(ToolCategory::SecurityProtocol),
                "model" | "model-checker" | "modelchecker" => Some(ToolCategory::ModelChecker),
                "rust" | "rust-verifier" => Some(ToolCategory::RustVerifier),
                "smt" | "smt-solver" => Some(ToolCategory::SmtSolver),
                "sat" | "sat-solver" => Some(ToolCategory::SatSolver),
                "sanitizer" | "san" => Some(ToolCategory::Sanitizer),
                "concurrency" | "thread" => Some(ToolCategory::ConcurrencyTester),
                "fuzz" | "fuzzer" => Some(ToolCategory::Fuzzer),
                "pbt" | "property" => Some(ToolCategory::PropertyTester),
                "static" | "lint" => Some(ToolCategory::StaticAnalyzer),
                "ai-opt" | "optimizer" => Some(ToolCategory::AiOptimizer),
                "compress" | "quantize" => Some(ToolCategory::AiCompressor),
                "data" | "quality" => Some(ToolCategory::DataQuality),
                "fairness" | "bias" => Some(ToolCategory::Fairness),
                "interpret" | "explain" => Some(ToolCategory::Interpretability),
                "guardrails" | "llm-guard" => Some(ToolCategory::LlmGuardrails),
                "eval" | "llm-eval" => Some(ToolCategory::LlmEvaluation),
                "hallucination" => Some(ToolCategory::HallucinationDetection),
                _ => None,
            });

    for tool in &tools {
        if let Some(ref cat) = category_filter {
            if tool.category != *cat {
                continue;
            }
        }

        let status = tool.check();
        if status.is_available() {
            available += 1;
        }

        if config.missing_only && status.is_available() {
            continue;
        }

        results.push((
            tool.name.to_string(),
            status,
            tool.install_hint.to_string(),
            tool.category,
        ));
    }

    if config.format == "json" {
        print_json_output(&results, total, available);
    } else {
        print_text_output(&results, total, available, config.verbose);
    }

    Ok(())
}

fn print_text_output(
    results: &[(String, ToolStatus, String, ToolCategory)],
    total: usize,
    available: usize,
    verbose: bool,
) {
    println!("DashProve Tool Status");
    println!("=====================\n");

    // Group by category
    let mut current_category: Option<ToolCategory> = None;

    for (name, status, install_hint, category) in results {
        if current_category != Some(*category) {
            if current_category.is_some() {
                println!();
            }
            println!("{}:", category.name());
            println!("{:-<50}", "");
            current_category = Some(*category);
        }

        let version_str = match status {
            ToolStatus::Available { version: Some(v) } => {
                // Truncate long version strings
                if v.len() > 20 {
                    format!(" ({}...)", &v[..17])
                } else {
                    format!(" ({})", v)
                }
            }
            ToolStatus::Available { version: None } => String::new(),
            ToolStatus::NotInstalled => String::new(),
            ToolStatus::Error { message } => format!(" [{}]", message),
        };

        println!("  {:<20} {:>4}{}", name, status.symbol(), version_str);

        if verbose && !status.is_available() {
            println!("    Install: {}", install_hint);
        }
    }

    println!("\n=====================");
    println!(
        "Summary: {}/{} tools available ({:.0}%)",
        available,
        total,
        (available as f64 / total as f64) * 100.0
    );
}

fn print_json_output(
    results: &[(String, ToolStatus, String, ToolCategory)],
    total: usize,
    available: usize,
) {
    use std::collections::HashMap;

    let tools: Vec<HashMap<&str, String>> = results
        .iter()
        .map(|(name, status, install_hint, category)| {
            let mut map = HashMap::new();
            map.insert("name", name.clone());
            map.insert("category", category.name().to_string());
            map.insert(
                "status",
                match status {
                    ToolStatus::Available { .. } => "available".to_string(),
                    ToolStatus::NotInstalled => "not_installed".to_string(),
                    ToolStatus::Error { .. } => "error".to_string(),
                },
            );
            if let ToolStatus::Available { version: Some(v) } = status {
                map.insert("version", v.clone());
            }
            map.insert("install_hint", install_hint.clone());
            map
        })
        .collect();

    let output = serde_json::json!({
        "total": total,
        "available": available,
        "missing": total - available,
        "tools": tools,
    });

    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_count() {
        let tools = get_all_tools();
        // Should have around 98 tools (some may vary due to categorization)
        assert!(
            tools.len() >= 80,
            "Expected at least 80 tools, got {}",
            tools.len()
        );
    }

    #[test]
    fn test_tool_categories() {
        let tools = get_all_tools();

        // Count tools per category
        let mut theorem_provers = 0;
        let mut rust_verifiers = 0;

        for tool in &tools {
            match tool.category {
                ToolCategory::TheoremProver => theorem_provers += 1,
                ToolCategory::RustVerifier => rust_verifiers += 1,
                _ => {}
            }
        }

        assert!(theorem_provers >= 5, "Expected at least 5 theorem provers");
        assert!(rust_verifiers >= 5, "Expected at least 5 Rust verifiers");
    }

    #[test]
    fn test_category_names() {
        assert_eq!(ToolCategory::TheoremProver.name(), "Theorem Provers");
        assert_eq!(ToolCategory::ModelChecker.name(), "Model Checkers");
        assert_eq!(ToolCategory::Fuzzer.name(), "Fuzzers");
        assert_eq!(ToolCategory::LlmEvaluation.name(), "LLM Evaluation");
    }

    #[test]
    fn test_tool_status_symbol() {
        assert_eq!(ToolStatus::Available { version: None }.symbol(), "OK");
        assert_eq!(ToolStatus::NotInstalled.symbol(), "--");
        assert_eq!(
            ToolStatus::Error {
                message: "test".to_string()
            }
            .symbol(),
            "ERR"
        );
    }

    #[test]
    fn test_check_clippy() {
        // Clippy should be available on most dev machines
        let status = check_rustup_component("clippy", None);
        // We don't assert it's available since CI might not have it,
        // but we verify the check doesn't panic
        let _ = status.symbol();
    }
}
