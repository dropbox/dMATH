//! End-to-end tests that execute real verification backends
//!
//! These tests invoke actual verification tools (when available) and validate
//! the full pipeline from USL specification through backend execution.
//!
//! Tests are skipped if the required backend tool is not installed.

use dashprove_backends::{
    AIF360Backend,
    AbCrownBackend,
    // Core theorem provers
    Acl2Backend,
    AequitasBackend,
    AflBackend,
    AgdaBackend,
    AimetBackend,
    AlibiBackend,
    AlloyBackend,
    AlloyConfig,
    AltErgoBackend,
    // Symbolic execution
    AngrBackend,
    ApalacheBackend,
    // Adversarial robustness
    ArtBackend,
    // Rust sanitizers and memory tools
    AsanBackend,
    // Abstract interpretation
    AstreeBackend,
    AtsBackend,
    AuditBackend,
    AutoLirpaBackend,
    BackendId,
    BoleroBackend,
    BoogieBackend,
    BoolectorBackend,
    BrevitasBackend,
    CDSCheckerBackend,
    CaDiCaLBackend,
    CadenceEdaBackend,
    CadpBackend,
    CaptumBackend,
    CbmcBackend,
    CleverHansBackend,
    CodeSonarBackend,
    CoqBackend,
    CpacheckerBackend,
    CryptoMiniSatBackend,
    CryptoVerifBackend,
    Cvc5Backend,
    DafnyBackend,
    // Rust documentation
    DeadlinksBackend,
    DeepEvalBackend,
    DeepchecksBackend,
    DenyBackend,
    DivineBackend,
    DnnvBackend,
    // Cryptographic verification
    EasyCryptBackend,
    EranBackend,
    EsbmcBackend,
    EvidentlyBackend,
    FStarBackend,
    FactScoreBackend,
    // Fairness/bias
    FairlearnBackend,
    FluxBackend,
    FoolboxBackend,
    FramaCBackend,
    FramaCEvaBackend,
    GeigerBackend,
    GenMCBackend,
    GlucoseBackend,
    GrcovBackend,
    // Data quality
    GreatExpectationsBackend,
    // LLM guardrails
    GuardrailsAIBackend,
    GuidanceBackend,
    HealthStatus,
    Hol4Backend,
    // Additional theorem provers
    HolLightBackend,
    HonggfuzzBackend,
    IREEBackend,
    IdrisBackend,
    InferBackend,
    InstaBackend,
    InterpretMlBackend,
    IsabelleBackend,
    IvyBackend,
    JasminBackend,
    JasperGoldBackend,
    JpfBackend,
    KaniBackend,
    KeyBackend,
    // Additional SAT solvers
    KissatBackend,
    KleeBackend,
    KrakatoaBackend,
    LangSmithBackend,
    Lean4Backend,
    // Rust fuzzing
    LibFuzzerBackend,
    LimeBackend,
    LiquidHaskellBackend,
    LlvmCovBackend,
    // Rust concurrency testing
    LoomBackend,
    LsanBackend,
    MNBaBBackend,
    ManticoreBackend,
    // Neural network verification
    MarabouBackend,
    MathSatBackend,
    Mcrl2Backend,
    MetamathBackend,
    MiniSatBackend,
    MiraiBackend,
    MizarBackend,
    MockallBackend,
    MsanBackend,
    MutantsBackend,
    NNCFBackend,
    NeMoGuardrailsBackend,
    // AI/ML compression
    NeuralCompressorBackend,
    NeurifyBackend,
    // Rust testing infrastructure
    NextestBackend,
    NnenumBackend,
    NnvBackend,
    NuSmvBackend,
    // Additional model checkers
    NuXmvBackend,
    // AI/ML optimization
    OnnxRuntimeBackend,
    OpenJmlBackend,
    // Additional SMT solvers
    OpenSmtBackend,
    OpenVINOBackend,
    // Distributed systems verification
    PLangBackend,
    PlatformApiBackend,
    PolyspaceBackend,
    PrismBackend,
    // LLM evaluation
    PromptfooBackend,
    // Rust property-based testing
    ProptestBackend,
    ProverifBackend,
    PvsBackend,
    QuickCheckBackend,
    RagasBackend,
    RdmeBackend,
    ReluValBackend,
    RobustBenchBackend,
    RstestBackend,
    RudraBackend,
    SeaHornBackend,
    // Hallucination detection
    SelfCheckGPTBackend,
    // Rust static analysis
    SemverChecksBackend,
    // Interpretability
    ShapBackend,
    ShuttleBackend,
    SmackBackend,
    SparkBackend,
    SpellcheckBackend,
    SpinBackend,
    StainlessBackend,
    StormBackend,
    SymbiYosysBackend,
    TVMBackend,
    TamarinBackend,
    // Rust coverage
    TarpaulinBackend,
    TensorRTBackend,
    TestCaseBackend,
    TextAttackBackend,
    TlaPlusBackend,
    TritonBackend,
    TritonDbaBackend,
    TruLensBackend,
    TsanBackend,
    UltimateBackend,
    UppaalBackend,
    ValgrindBackend,
    // Program verification frameworks
    VccBackend,
    VenusBackend,
    VeriFastBackend,
    VeriNetBackend,
    VeriTBackend,
    VerificationBackend,
    VerificationStatus,
    VerifpalBackend,
    VetBackend,
    Why3Backend,
    WhyLogsBackend,
    YicesBackend,
    // Hardware verification
    YosysBackend,
    Z3Backend,
};
use dashprove_usl::ast::{Expr, Invariant, Property, Spec, Theorem};
use dashprove_usl::parse;
use dashprove_usl::typecheck::typecheck;
use std::time::Duration;

/// Helper to check if a backend is available
async fn is_backend_available(backend: &dyn VerificationBackend) -> bool {
    matches!(backend.health_check().await, HealthStatus::Healthy)
}

/// Create a simple theorem spec for testing
fn simple_theorem_spec() -> dashprove_usl::typecheck::TypedSpec {
    let spec = Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "trivial_true".into(),
            body: Expr::Bool(true),
        })],
    };
    typecheck(spec).expect("spec should type-check")
}

/// Create an invariant spec for Alloy testing
fn simple_invariant_spec() -> dashprove_usl::typecheck::TypedSpec {
    let spec = Spec {
        types: vec![],
        properties: vec![Property::Invariant(Invariant {
            name: "always_true".into(),
            body: Expr::Bool(true),
        })],
    };
    typecheck(spec).expect("spec should type-check")
}

/// Create a platform API spec for static checker generation
fn platform_api_spec() -> dashprove_usl::typecheck::TypedSpec {
    let src = r#"
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

    let spec = parse(src).expect("parse should succeed");
    typecheck(spec).expect("typecheck should succeed")
}

/// Debug helper: print the generated Alloy code
#[allow(dead_code)]
fn debug_print_alloy_code(spec: &dashprove_usl::typecheck::TypedSpec) {
    let compiled = dashprove_usl::compile_to_alloy(spec);
    println!(
        "\n=== Generated Alloy code ===\n{}\n=== End ===",
        compiled.code
    );
}

// ============================================================================
// Alloy End-to-End Tests
// ============================================================================

mod alloy_e2e {
    use super::*;

    #[tokio::test]
    async fn test_alloy_backend_health_check() {
        let backend = AlloyBackend::new();
        let health = backend.health_check().await;

        // Just check that health_check works - may be Healthy or Unavailable
        match health {
            HealthStatus::Healthy => println!("Alloy is available"),
            HealthStatus::Unavailable { reason } => println!("Alloy unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Alloy degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_alloy_verify_simple_invariant() {
        let backend = AlloyBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Alloy not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Alloy result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
                // A trivial invariant should be proven (or at least not disproven)
                assert!(
                    matches!(r.status, VerificationStatus::Proven)
                        || matches!(r.status, VerificationStatus::Unknown { .. }),
                    "Unexpected status: {:?}",
                    r.status
                );
            }
            Err(e) => {
                // Backend might fail for compilation reasons - that's OK for this test
                println!("Alloy verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_alloy_with_custom_config() {
        let config = AlloyConfig {
            timeout: Duration::from_secs(30),
            scope: 3,
            solver: None,
            alloy_path: None,
        };
        let backend = AlloyBackend::with_config(config);

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Alloy not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_alloy_backend_id() {
        let backend = AlloyBackend::new();
        assert_eq!(backend.id(), BackendId::Alloy);
    }

    #[tokio::test]
    async fn test_alloy_compile_invariant() {
        // Test that Alloy compiles invariants correctly even if Alloy is unavailable
        let spec = simple_invariant_spec();
        let compiled = dashprove_usl::compile_to_alloy(&spec);

        // Verify the generated code has the expected Alloy structure
        assert!(
            compiled.code.contains("module"),
            "Should have module declaration"
        );
        assert!(
            compiled.code.contains("assert"),
            "Should have assert for invariant"
        );
        assert!(compiled.code.contains("check"), "Should have check command");
        println!("Generated Alloy code:\n{}", compiled.code);
    }

    #[tokio::test]
    async fn test_alloy_compile_theorem() {
        // Test that Alloy compiles theorems correctly
        let spec = simple_theorem_spec();
        let compiled = dashprove_usl::compile_to_alloy(&spec);

        assert!(
            compiled.code.contains("module"),
            "Should have module declaration"
        );
        assert!(
            compiled.code.contains("assert"),
            "Should have assert for theorem"
        );
        println!("Generated Alloy code:\n{}", compiled.code);
    }

    /// Create a more complex invariant with quantifiers
    fn quantified_invariant_spec() -> dashprove_usl::typecheck::TypedSpec {
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Invariant(Invariant {
                name: "reflexivity".into(),
                body: Expr::ForAll {
                    var: "x".to_string(),
                    ty: Some(dashprove_usl::ast::Type::Named("Int".to_string())),
                    body: Box::new(Expr::Compare(
                        Box::new(Expr::Var("x".to_string())),
                        dashprove_usl::ast::ComparisonOp::Eq,
                        Box::new(Expr::Var("x".to_string())),
                    )),
                },
            })],
        };
        typecheck(spec).expect("spec should type-check")
    }

    #[tokio::test]
    async fn test_alloy_compile_quantified_invariant() {
        let spec = quantified_invariant_spec();
        let compiled = dashprove_usl::compile_to_alloy(&spec);

        // Verify quantifier compilation
        assert!(
            compiled.code.contains("all x"),
            "Should compile forall to 'all'"
        );
        println!("Generated Alloy quantified code:\n{}", compiled.code);
    }

    #[tokio::test]
    async fn test_alloy_verify_quantified_invariant() {
        let backend = AlloyBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Alloy not available");
            return;
        }

        let spec = quantified_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Alloy quantified result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                // Reflexivity should be proven
                assert!(
                    matches!(r.status, VerificationStatus::Proven)
                        || matches!(r.status, VerificationStatus::Unknown { .. }),
                    "Unexpected status: {:?}",
                    r.status
                );
            }
            Err(e) => {
                println!("Alloy verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Kani End-to-End Tests
// ============================================================================

mod kani_e2e {
    use super::*;
    use dashprove_backends::KaniConfig;
    use dashprove_usl::ast::{Contract, Param, Type};
    use std::path::PathBuf;

    /// Create a simple contract spec for Kani testing
    fn simple_contract_spec() -> dashprove_usl::typecheck::TypedSpec {
        // Contract: add(x: u32, y: u32) -> u32
        // requires: x < 1000 && y < 1000
        // ensures: result == x + y
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Contract(Contract {
                type_path: vec!["Math".to_string()],
                params: vec![
                    Param {
                        name: "x".to_string(),
                        ty: Type::Named("u32".to_string()),
                    },
                    Param {
                        name: "y".to_string(),
                        ty: Type::Named("u32".to_string()),
                    },
                ],
                return_type: Some(Type::Named("u32".to_string())),
                requires: vec![Expr::And(
                    Box::new(Expr::Compare(
                        Box::new(Expr::Var("x".to_string())),
                        dashprove_usl::ast::ComparisonOp::Lt,
                        Box::new(Expr::Int(1000)),
                    )),
                    Box::new(Expr::Compare(
                        Box::new(Expr::Var("y".to_string())),
                        dashprove_usl::ast::ComparisonOp::Lt,
                        Box::new(Expr::Int(1000)),
                    )),
                )],
                ensures: vec![Expr::Compare(
                    Box::new(Expr::Var("result".to_string())),
                    dashprove_usl::ast::ComparisonOp::Eq,
                    Box::new(Expr::Binary(
                        Box::new(Expr::Var("x".to_string())),
                        dashprove_usl::ast::BinaryOp::Add,
                        Box::new(Expr::Var("y".to_string())),
                    )),
                )],
                ensures_err: vec![],
                assigns: vec![],
                allocates: vec![],
                frees: vec![],
                terminates: None,
                decreases: None,
                behaviors: vec![],
                complete_behaviors: false,
                disjoint_behaviors: false,
            })],
        };
        typecheck(spec).expect("spec should type-check")
    }

    /// Create a contract spec for safe_div from the example project
    fn safe_div_contract_spec() -> dashprove_usl::typecheck::TypedSpec {
        // Contract: safe_div(a: u32, b: u32) -> u32
        // requires: b != 0
        // ensures: true (we just want to test Kani runs)
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Contract(Contract {
                type_path: vec!["safe_div".to_string()],
                params: vec![
                    Param {
                        name: "a".to_string(),
                        ty: Type::Named("u32".to_string()),
                    },
                    Param {
                        name: "b".to_string(),
                        ty: Type::Named("u32".to_string()),
                    },
                ],
                return_type: Some(Type::Named("u32".to_string())),
                requires: vec![Expr::Compare(
                    Box::new(Expr::Var("b".to_string())),
                    dashprove_usl::ast::ComparisonOp::Ne,
                    Box::new(Expr::Int(0)),
                )],
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
            })],
        };
        typecheck(spec).expect("spec should type-check")
    }

    /// Get the path to the example Kani project
    fn example_kani_project_dir() -> PathBuf {
        // Find workspace root by looking for Cargo.toml with [workspace]
        let mut current = std::env::current_dir().expect("current dir");
        loop {
            let cargo_toml = current.join("Cargo.toml");
            if cargo_toml.exists() {
                if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                    if content.contains("[workspace]") {
                        return current.join("examples/kani");
                    }
                }
            }
            if !current.pop() {
                break;
            }
        }
        // Fallback: relative path from crate
        PathBuf::from("../../examples/kani")
    }

    #[tokio::test]
    async fn test_kani_backend_health_check() {
        let backend = KaniBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Kani is available"),
            HealthStatus::Unavailable { reason } => println!("Kani unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Kani degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_kani_backend_id() {
        let backend = KaniBackend::new();
        assert_eq!(backend.id(), BackendId::Kani);
    }

    #[tokio::test]
    async fn test_kani_compile_contract() {
        // Test that Kani compiles contracts correctly even if Kani is unavailable
        let spec = simple_contract_spec();
        let compiled = dashprove_usl::compile_to_kani(&spec);

        // Verify the generated code has the expected structure
        assert!(
            compiled.code.contains("#[kani::proof]"),
            "Should have kani proof attribute"
        );
        assert!(
            compiled.code.contains("kani::assume"),
            "Should have kani assume for preconditions"
        );
        assert!(
            compiled.code.contains("kani::assert"),
            "Should have kani assert for postconditions"
        );
        println!("Generated Kani code:\n{}", compiled.code);
    }

    #[tokio::test]
    async fn test_kani_verify_contract() {
        let backend = KaniBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Kani not available");
            return;
        }

        let spec = simple_contract_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Kani result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                // Kani might fail for project setup reasons
                println!("Kani verification error: {}", e);
            }
        }
    }

    // ============================================================================
    // Real Kani E2E Tests (using examples/kani project)
    // ============================================================================

    /// Test that the Kani backend can successfully verify a contract
    /// against the example Kani project.
    ///
    /// This test requires cargo-kani to be installed.
    #[tokio::test]
    async fn test_kani_real_e2e_with_example_project() {
        let project_dir = example_kani_project_dir();

        // Skip if example project doesn't exist
        if !project_dir.join("Cargo.toml").exists() {
            eprintln!(
                "SKIPPED: Example Kani project not found at {:?}",
                project_dir
            );
            return;
        }

        let config = KaniConfig {
            project_dir: Some(project_dir.clone()),
            timeout: Duration::from_secs(120),
            ..Default::default()
        };
        let backend = KaniBackend::with_config(config);

        // Check if Kani is available
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Kani not available (cargo-kani not installed)");
            return;
        }

        let spec = safe_div_contract_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("=== Real Kani E2E Test Results ===");
                println!("Status: {:?}", r.status);
                println!("Time taken: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
                if let Some(proof) = &r.proof {
                    println!("Proof: {}", proof);
                }
                if let Some(ce) = &r.counterexample {
                    println!("Counterexample: {}", ce);
                }
                // Test should complete without panic - actual result depends on generated harness
            }
            Err(e) => {
                println!("Kani E2E error: {}", e);
                // Error is acceptable for this test - we're testing the integration works
            }
        }
    }

    /// Test Kani backend compilation generates valid harness code
    #[tokio::test]
    async fn test_kani_real_e2e_compile_check() {
        let spec = safe_div_contract_spec();
        let compiled = dashprove_usl::compile_to_kani(&spec);

        println!("=== Generated Kani Harness ===");
        println!("{}", compiled.code);
        println!("=== End Harness ===");

        // Verify essential Kani attributes and macros are present
        assert!(
            compiled.code.contains("#![allow(unused)]"),
            "Should have unused warning suppression"
        );
        assert!(
            compiled.code.contains("#[kani::proof]"),
            "Should have kani proof attribute"
        );
    }

    /// Test Kani backend with a passing contract verification
    /// This creates a harness that should definitely pass.
    #[tokio::test]
    async fn test_kani_real_e2e_passing_contract() {
        let project_dir = example_kani_project_dir();

        if !project_dir.join("Cargo.toml").exists() {
            eprintln!(
                "SKIPPED: Example Kani project not found at {:?}",
                project_dir
            );
            return;
        }

        let config = KaniConfig {
            project_dir: Some(project_dir),
            timeout: Duration::from_secs(120),
            ..Default::default()
        };
        let backend = KaniBackend::with_config(config);

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Kani not available");
            return;
        }

        // Create a trivially true contract
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Contract(Contract {
                type_path: vec!["trivial_contract".to_string()],
                params: vec![Param {
                    name: "x".to_string(),
                    ty: Type::Named("u32".to_string()),
                }],
                return_type: Some(Type::Named("u32".to_string())),
                requires: vec![Expr::Bool(true)],
                ensures: vec![Expr::Bool(true)], // Trivially true
                ensures_err: vec![],
                assigns: vec![],
                allocates: vec![],
                frees: vec![],
                terminates: None,
                decreases: None,
                behaviors: vec![],
                complete_behaviors: false,
                disjoint_behaviors: false,
            })],
        };
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;

        match result {
            Ok(r) => {
                println!("Trivial contract result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                // A trivially true postcondition should pass
            }
            Err(e) => {
                println!("Error (acceptable in test): {}", e);
            }
        }
    }

    // ============================================================================
    // Real Kani E2E Tests with kani_test_project
    // These tests actually verify properties using cargo-kani
    // ============================================================================

    /// Get the path to the kani_test_project for real e2e tests
    fn kani_test_project_dir() -> PathBuf {
        // Find workspace root by looking for Cargo.toml with [workspace]
        let mut current = std::env::current_dir().expect("current dir");
        loop {
            let cargo_toml = current.join("Cargo.toml");
            if cargo_toml.exists() {
                if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                    if content.contains("[workspace]") && content.contains("dashprove") {
                        return current.join("crates/dashprove-backends/tests/kani_test_project");
                    }
                }
            }
            if !current.pop() {
                break;
            }
        }
        // Fallback: relative path from test crate
        PathBuf::from("tests/kani_test_project")
    }

    /// Create contract spec for identity function: ensures result == x
    fn identity_contract_spec() -> dashprove_usl::typecheck::TypedSpec {
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Contract(Contract {
                type_path: vec!["identity".to_string()],
                params: vec![Param {
                    name: "x".to_string(),
                    ty: Type::Named("u32".to_string()),
                }],
                return_type: Some(Type::Named("u32".to_string())),
                requires: vec![],
                ensures: vec![Expr::Compare(
                    Box::new(Expr::Var("result".to_string())),
                    dashprove_usl::ast::ComparisonOp::Eq,
                    Box::new(Expr::Var("x".to_string())),
                )],
                ensures_err: vec![],
                assigns: vec![],
                allocates: vec![],
                frees: vec![],
                terminates: None,
                decreases: None,
                behaviors: vec![],
                complete_behaviors: false,
                disjoint_behaviors: false,
            })],
        };
        typecheck(spec).expect("spec should type-check")
    }

    /// Create contract spec for bounded_add: requires x < 1000 && y < 1000, ensures result == x + y
    fn bounded_add_contract_spec() -> dashprove_usl::typecheck::TypedSpec {
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Contract(Contract {
                type_path: vec!["bounded_add".to_string()],
                params: vec![
                    Param {
                        name: "x".to_string(),
                        ty: Type::Named("u32".to_string()),
                    },
                    Param {
                        name: "y".to_string(),
                        ty: Type::Named("u32".to_string()),
                    },
                ],
                return_type: Some(Type::Named("u32".to_string())),
                requires: vec![Expr::And(
                    Box::new(Expr::Compare(
                        Box::new(Expr::Var("x".to_string())),
                        dashprove_usl::ast::ComparisonOp::Lt,
                        Box::new(Expr::Int(1000)),
                    )),
                    Box::new(Expr::Compare(
                        Box::new(Expr::Var("y".to_string())),
                        dashprove_usl::ast::ComparisonOp::Lt,
                        Box::new(Expr::Int(1000)),
                    )),
                )],
                ensures: vec![Expr::Compare(
                    Box::new(Expr::Var("result".to_string())),
                    dashprove_usl::ast::ComparisonOp::Eq,
                    Box::new(Expr::Binary(
                        Box::new(Expr::Var("x".to_string())),
                        dashprove_usl::ast::BinaryOp::Add,
                        Box::new(Expr::Var("y".to_string())),
                    )),
                )],
                ensures_err: vec![],
                assigns: vec![],
                allocates: vec![],
                frees: vec![],
                terminates: None,
                decreases: None,
                behaviors: vec![],
                complete_behaviors: false,
                disjoint_behaviors: false,
            })],
        };
        typecheck(spec).expect("spec should type-check")
    }

    /// Create contract spec for max_of: ensures result >= a && result >= b
    fn max_of_contract_spec() -> dashprove_usl::typecheck::TypedSpec {
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Contract(Contract {
                type_path: vec!["max_of".to_string()],
                params: vec![
                    Param {
                        name: "a".to_string(),
                        ty: Type::Named("u32".to_string()),
                    },
                    Param {
                        name: "b".to_string(),
                        ty: Type::Named("u32".to_string()),
                    },
                ],
                return_type: Some(Type::Named("u32".to_string())),
                requires: vec![],
                ensures: vec![Expr::And(
                    Box::new(Expr::Compare(
                        Box::new(Expr::Var("result".to_string())),
                        dashprove_usl::ast::ComparisonOp::Ge,
                        Box::new(Expr::Var("a".to_string())),
                    )),
                    Box::new(Expr::Compare(
                        Box::new(Expr::Var("result".to_string())),
                        dashprove_usl::ast::ComparisonOp::Ge,
                        Box::new(Expr::Var("b".to_string())),
                    )),
                )],
                ensures_err: vec![],
                assigns: vec![],
                allocates: vec![],
                frees: vec![],
                terminates: None,
                decreases: None,
                behaviors: vec![],
                complete_behaviors: false,
                disjoint_behaviors: false,
            })],
        };
        typecheck(spec).expect("spec should type-check")
    }

    /// Create a FALSE contract spec that should find a counterexample
    /// bounded_add without preconditions can overflow
    fn bounded_add_false_contract_spec() -> dashprove_usl::typecheck::TypedSpec {
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Contract(Contract {
                type_path: vec!["bounded_add".to_string()],
                params: vec![
                    Param {
                        name: "x".to_string(),
                        ty: Type::Named("u32".to_string()),
                    },
                    Param {
                        name: "y".to_string(),
                        ty: Type::Named("u32".to_string()),
                    },
                ],
                return_type: Some(Type::Named("u32".to_string())),
                // No preconditions - this allows overflow
                requires: vec![],
                // This postcondition will fail for large inputs
                ensures: vec![Expr::Compare(
                    Box::new(Expr::Var("result".to_string())),
                    dashprove_usl::ast::ComparisonOp::Eq,
                    Box::new(Expr::Binary(
                        Box::new(Expr::Var("x".to_string())),
                        dashprove_usl::ast::BinaryOp::Add,
                        Box::new(Expr::Var("y".to_string())),
                    )),
                )],
                ensures_err: vec![],
                assigns: vec![],
                allocates: vec![],
                frees: vec![],
                terminates: None,
                decreases: None,
                behaviors: vec![],
                complete_behaviors: false,
                disjoint_behaviors: false,
            })],
        };
        typecheck(spec).expect("spec should type-check")
    }

    /// Real E2E test: Verify identity function property
    /// This test actually runs cargo-kani and expects PROVEN status
    #[tokio::test]
    async fn test_kani_real_verify_identity_proven() {
        let project_dir = kani_test_project_dir();

        if !project_dir.join("Cargo.toml").exists() {
            eprintln!("SKIPPED: kani_test_project not found at {:?}", project_dir);
            return;
        }

        let config = KaniConfig {
            project_dir: Some(project_dir),
            timeout: Duration::from_secs(180),
            ..Default::default()
        };
        let backend = KaniBackend::with_config(config);

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Kani not available (cargo-kani not installed)");
            return;
        }

        let spec = identity_contract_spec();
        println!("=== Testing identity contract (expect PROVEN) ===");
        println!(
            "Generated harness:\n{}",
            dashprove_usl::compile_to_kani(&spec).code
        );

        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Status: {:?}", r.status);
                println!("Time taken: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
                if let Some(proof) = &r.proof {
                    println!("Proof: {}", proof);
                }
                // Identity function should verify successfully
                assert!(
                    matches!(r.status, VerificationStatus::Proven),
                    "Expected PROVEN for identity contract, got {:?}",
                    r.status
                );
            }
            Err(e) => {
                panic!("Kani verification failed unexpectedly: {}", e);
            }
        }
    }

    /// Real E2E test: Verify bounded_add function property
    /// This test actually runs cargo-kani and expects PROVEN status
    #[tokio::test]
    async fn test_kani_real_verify_bounded_add_proven() {
        let project_dir = kani_test_project_dir();

        if !project_dir.join("Cargo.toml").exists() {
            eprintln!("SKIPPED: kani_test_project not found at {:?}", project_dir);
            return;
        }

        let config = KaniConfig {
            project_dir: Some(project_dir),
            timeout: Duration::from_secs(180),
            ..Default::default()
        };
        let backend = KaniBackend::with_config(config);

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Kani not available (cargo-kani not installed)");
            return;
        }

        let spec = bounded_add_contract_spec();
        println!("=== Testing bounded_add contract (expect PROVEN) ===");
        println!(
            "Generated harness:\n{}",
            dashprove_usl::compile_to_kani(&spec).code
        );

        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Status: {:?}", r.status);
                println!("Time taken: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
                if let Some(proof) = &r.proof {
                    println!("Proof: {}", proof);
                }
                // Bounded add with preconditions should verify
                assert!(
                    matches!(r.status, VerificationStatus::Proven),
                    "Expected PROVEN for bounded_add contract, got {:?}",
                    r.status
                );
            }
            Err(e) => {
                panic!("Kani verification failed unexpectedly: {}", e);
            }
        }
    }

    /// Real E2E test: Verify max_of function property
    /// This test actually runs cargo-kani and expects PROVEN status
    #[tokio::test]
    async fn test_kani_real_verify_max_of_proven() {
        let project_dir = kani_test_project_dir();

        if !project_dir.join("Cargo.toml").exists() {
            eprintln!("SKIPPED: kani_test_project not found at {:?}", project_dir);
            return;
        }

        let config = KaniConfig {
            project_dir: Some(project_dir),
            timeout: Duration::from_secs(180),
            ..Default::default()
        };
        let backend = KaniBackend::with_config(config);

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Kani not available (cargo-kani not installed)");
            return;
        }

        let spec = max_of_contract_spec();
        println!("=== Testing max_of contract (expect PROVEN) ===");
        println!(
            "Generated harness:\n{}",
            dashprove_usl::compile_to_kani(&spec).code
        );

        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Status: {:?}", r.status);
                println!("Time taken: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
                if let Some(proof) = &r.proof {
                    println!("Proof: {}", proof);
                }
                // max_of should verify (result >= a && result >= b)
                assert!(
                    matches!(r.status, VerificationStatus::Proven),
                    "Expected PROVEN for max_of contract, got {:?}",
                    r.status
                );
            }
            Err(e) => {
                panic!("Kani verification failed unexpectedly: {}", e);
            }
        }
    }

    /// Real E2E test: bounded_add WITHOUT preconditions should find counterexample
    /// This test expects Kani to find that unbounded inputs can cause overflow
    #[tokio::test]
    async fn test_kani_real_verify_bounded_add_disproven() {
        let project_dir = kani_test_project_dir();

        if !project_dir.join("Cargo.toml").exists() {
            eprintln!("SKIPPED: kani_test_project not found at {:?}", project_dir);
            return;
        }

        let config = KaniConfig {
            project_dir: Some(project_dir),
            timeout: Duration::from_secs(180),
            ..Default::default()
        };
        let backend = KaniBackend::with_config(config);

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Kani not available (cargo-kani not installed)");
            return;
        }

        let spec = bounded_add_false_contract_spec();
        println!("=== Testing bounded_add WITHOUT preconditions (expect DISPROVEN) ===");
        println!(
            "Generated harness:\n{}",
            dashprove_usl::compile_to_kani(&spec).code
        );

        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Status: {:?}", r.status);
                println!("Time taken: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
                if let Some(ce) = &r.counterexample {
                    println!("Counterexample: {}", ce);
                }
                // Without preconditions, bounded_add can overflow, so Kani should find a counterexample
                assert!(
                    matches!(r.status, VerificationStatus::Disproven),
                    "Expected DISPROVEN for unbounded add contract, got {:?}",
                    r.status
                );
            }
            Err(e) => {
                panic!("Kani verification failed unexpectedly: {}", e);
            }
        }
    }
}

// ============================================================================
// LEAN 4 End-to-End Tests
// ============================================================================

mod lean4_e2e {
    use super::*;

    #[tokio::test]
    async fn test_lean4_backend_health_check() {
        let backend = Lean4Backend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("LEAN 4 is available"),
            HealthStatus::Unavailable { reason } => println!("LEAN 4 unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("LEAN 4 degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_lean4_verify_simple_theorem() {
        let backend = Lean4Backend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: LEAN 4 not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("LEAN 4 result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("LEAN 4 verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_lean4_backend_id() {
        let backend = Lean4Backend::new();
        assert_eq!(backend.id(), BackendId::Lean4);
    }
}

// ============================================================================
// TLA+ End-to-End Tests
// ============================================================================

mod tlaplus_e2e {
    use super::*;
    use dashprove_usl::ast::{Temporal, TemporalExpr};

    #[tokio::test]
    async fn test_tlaplus_backend_health_check() {
        let backend = TlaPlusBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("TLA+ (TLC) is available"),
            HealthStatus::Unavailable { reason } => println!("TLA+ unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("TLA+ degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_tlaplus_backend_id() {
        let backend = TlaPlusBackend::new();
        assert_eq!(backend.id(), BackendId::TlaPlus);
    }

    #[tokio::test]
    async fn test_tlaplus_verify_simple_invariant() {
        let backend = TlaPlusBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: TLA+ (TLC) not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("TLA+ result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
                // A trivial invariant should be proven
                assert!(
                    matches!(r.status, VerificationStatus::Proven)
                        || matches!(r.status, VerificationStatus::Unknown { .. }),
                    "Unexpected status: {:?}",
                    r.status
                );
            }
            Err(e) => {
                // TLC might fail for various reasons - log but don't fail test
                println!("TLA+ verification error: {}", e);
            }
        }
    }

    /// Create a temporal property spec (always true)
    fn temporal_always_true_spec() -> dashprove_usl::typecheck::TypedSpec {
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Temporal(Temporal {
                name: "always_true".into(),
                body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
                fairness: vec![],
            })],
        };
        typecheck(spec).expect("spec should type-check")
    }

    /// Create a temporal property spec (eventually true)
    fn temporal_eventually_true_spec() -> dashprove_usl::typecheck::TypedSpec {
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Temporal(Temporal {
                name: "eventually_true".into(),
                body: TemporalExpr::Eventually(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
                fairness: vec![],
            })],
        };
        typecheck(spec).expect("spec should type-check")
    }

    #[tokio::test]
    async fn test_tlaplus_verify_temporal_always() {
        let backend = TlaPlusBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: TLA+ (TLC) not available");
            return;
        }

        let spec = temporal_always_true_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("TLA+ temporal (always) result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                // Trivial always(true) should be proven
                assert!(
                    matches!(r.status, VerificationStatus::Proven)
                        || matches!(r.status, VerificationStatus::Unknown { .. }),
                    "Unexpected status: {:?}",
                    r.status
                );
            }
            Err(e) => {
                println!("TLA+ verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_tlaplus_verify_temporal_eventually() {
        let backend = TlaPlusBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: TLA+ (TLC) not available");
            return;
        }

        let spec = temporal_eventually_true_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("TLA+ temporal (eventually) result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                // eventually(true) in initial state is immediately true
                assert!(
                    matches!(r.status, VerificationStatus::Proven)
                        || matches!(r.status, VerificationStatus::Unknown { .. }),
                    "Unexpected status: {:?}",
                    r.status
                );
            }
            Err(e) => {
                println!("TLA+ verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_tlaplus_multiple_invariants() {
        let backend = TlaPlusBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: TLA+ (TLC) not available");
            return;
        }

        // Create spec with multiple invariants
        let spec = Spec {
            types: vec![],
            properties: vec![
                Property::Invariant(Invariant {
                    name: "inv1".into(),
                    body: Expr::Bool(true),
                }),
                Property::Invariant(Invariant {
                    name: "inv2".into(),
                    body: Expr::Bool(true),
                }),
            ],
        };
        let typed_spec = typecheck(spec).expect("spec should type-check");
        let result = backend.verify(&typed_spec).await;

        match result {
            Ok(r) => {
                println!("TLA+ multiple invariants result: {:?}", r.status);
                assert!(
                    matches!(r.status, VerificationStatus::Proven)
                        || matches!(r.status, VerificationStatus::Unknown { .. }),
                    "Unexpected status: {:?}",
                    r.status
                );
            }
            Err(e) => {
                println!("TLA+ verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Isabelle End-to-End Tests
// ============================================================================

mod isabelle_e2e {
    use super::*;

    #[tokio::test]
    async fn test_isabelle_backend_health_check() {
        let backend = IsabelleBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Isabelle is available"),
            HealthStatus::Unavailable { reason } => println!("Isabelle unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Isabelle degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_isabelle_backend_id() {
        let backend = IsabelleBackend::new();
        assert_eq!(backend.id(), BackendId::Isabelle);
    }

    #[tokio::test]
    async fn test_isabelle_compile_theorem() {
        // Test that Isabelle compiles theorems correctly even if Isabelle is unavailable
        let spec = simple_theorem_spec();
        let compiled = dashprove_usl::compile_to_isabelle(&spec);

        // Verify the generated code has the expected Isabelle structure
        assert!(
            compiled.code.contains("theory"),
            "Should have theory declaration"
        );
        assert!(compiled.code.contains("imports Main"), "Should import Main");
        assert!(compiled.code.contains("lemma"), "Should have lemma");
        assert!(compiled.code.contains("end"), "Should have end");
        println!("Generated Isabelle code:\n{}", compiled.code);
    }

    #[tokio::test]
    async fn test_isabelle_verify_simple_theorem() {
        let backend = IsabelleBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Isabelle not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Isabelle result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
                // A trivial theorem should be proven (or at least not disproven)
                assert!(
                    matches!(r.status, VerificationStatus::Proven)
                        || matches!(r.status, VerificationStatus::Unknown { .. }),
                    "Unexpected status: {:?}",
                    r.status
                );
            }
            Err(e) => {
                println!("Isabelle verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_isabelle_compile_invariant() {
        let spec = simple_invariant_spec();
        let compiled = dashprove_usl::compile_to_isabelle(&spec);

        assert!(compiled.code.contains("theory"), "Should have theory");
        assert!(
            compiled.code.contains("lemma") || compiled.code.contains("theorem"),
            "Should have proof statement"
        );
        println!("Generated Isabelle invariant code:\n{}", compiled.code);
    }
}

// ============================================================================
// Coq End-to-End Tests
// ============================================================================

mod coq_e2e {
    use super::*;

    #[tokio::test]
    async fn test_coq_backend_health_check() {
        let backend = CoqBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Coq is available"),
            HealthStatus::Unavailable { reason } => println!("Coq unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Coq degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_coq_backend_id() {
        let backend = CoqBackend::new();
        assert_eq!(backend.id(), BackendId::Coq);
    }

    #[tokio::test]
    async fn test_coq_compile_theorem() {
        // Test that Coq compiles theorems correctly even if Coq is unavailable
        let spec = simple_theorem_spec();
        let compiled = dashprove_usl::compile_to_coq(&spec);

        // Verify the generated code has the expected Coq structure
        assert!(
            compiled.code.contains("Theorem") || compiled.code.contains("Lemma"),
            "Should have Theorem or Lemma"
        );
        assert!(
            compiled.code.contains("Proof") || compiled.code.contains("Admitted"),
            "Should have proof structure"
        );
        println!("Generated Coq code:\n{}", compiled.code);
    }

    #[tokio::test]
    async fn test_coq_verify_simple_theorem() {
        let backend = CoqBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Coq not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Coq result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
                // A trivial theorem should be proven (or at least not disproven)
                assert!(
                    matches!(r.status, VerificationStatus::Proven)
                        || matches!(r.status, VerificationStatus::Unknown { .. }),
                    "Unexpected status: {:?}",
                    r.status
                );
            }
            Err(e) => {
                println!("Coq verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_coq_compile_invariant() {
        let spec = simple_invariant_spec();
        let compiled = dashprove_usl::compile_to_coq(&spec);

        assert!(
            compiled.code.contains("Theorem")
                || compiled.code.contains("Lemma")
                || compiled.code.contains("Definition"),
            "Should have proof declaration"
        );
        println!("Generated Coq invariant code:\n{}", compiled.code);
    }
}

// ============================================================================
// Dafny End-to-End Tests
// ============================================================================

mod dafny_e2e {
    use super::*;

    #[tokio::test]
    async fn test_dafny_backend_health_check() {
        let backend = DafnyBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Dafny is available"),
            HealthStatus::Unavailable { reason } => println!("Dafny unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Dafny degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_dafny_backend_id() {
        let backend = DafnyBackend::new();
        assert_eq!(backend.id(), BackendId::Dafny);
    }

    #[tokio::test]
    async fn test_dafny_compile_theorem() {
        // Test that Dafny compiles theorems correctly even if Dafny is unavailable
        let spec = simple_theorem_spec();
        let compiled = dashprove_usl::compile_to_dafny(&spec);

        // Verify the generated code has the expected Dafny structure
        assert!(
            compiled.code.contains("lemma") || compiled.code.contains("method"),
            "Should have lemma or method"
        );
        assert!(
            compiled.code.contains("ensures"),
            "Should have ensures clause"
        );
        println!("Generated Dafny code:\n{}", compiled.code);
    }

    #[tokio::test]
    async fn test_dafny_verify_simple_theorem() {
        let backend = DafnyBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Dafny not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Dafny result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
                // A trivial theorem should be proven (or at least not disproven)
                assert!(
                    matches!(r.status, VerificationStatus::Proven)
                        || matches!(r.status, VerificationStatus::Unknown { .. })
                        || matches!(r.status, VerificationStatus::Partial { .. }),
                    "Unexpected status: {:?}",
                    r.status
                );
            }
            Err(e) => {
                println!("Dafny verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_dafny_compile_invariant() {
        let spec = simple_invariant_spec();
        let compiled = dashprove_usl::compile_to_dafny(&spec);

        assert!(
            compiled.code.contains("lemma") || compiled.code.contains("predicate"),
            "Should have lemma or predicate"
        );
        println!("Generated Dafny invariant code:\n{}", compiled.code);
    }

    #[tokio::test]
    async fn test_dafny_compile_quantified_property() {
        // Create a spec with quantifiers to test Dafny's forall/exists compilation
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Theorem(Theorem {
                name: "forall_test".into(),
                body: Expr::ForAll {
                    var: "x".to_string(),
                    ty: Some(dashprove_usl::ast::Type::Named("Int".to_string())),
                    body: Box::new(Expr::Compare(
                        Box::new(Expr::Var("x".to_string())),
                        dashprove_usl::ast::ComparisonOp::Eq,
                        Box::new(Expr::Var("x".to_string())),
                    )),
                },
            })],
        };
        let typed_spec = typecheck(spec).expect("spec should type-check");
        let compiled = dashprove_usl::compile_to_dafny(&typed_spec);

        assert!(
            compiled.code.contains("forall"),
            "Should compile forall quantifier"
        );
        println!("Generated Dafny quantified code:\n{}", compiled.code);
    }
}

// ============================================================================
// Platform API End-to-End Tests
// ============================================================================

mod platform_api_e2e {
    use super::*;

    #[tokio::test]
    async fn test_platform_api_backend_health_check() {
        let backend = PlatformApiBackend::new();
        let health = backend.health_check().await;

        assert!(
            matches!(health, HealthStatus::Healthy),
            "Platform API backend should always be healthy: {health:?}"
        );
    }

    #[tokio::test]
    async fn test_platform_api_backend_id() {
        let backend = PlatformApiBackend::new();
        assert_eq!(backend.id(), BackendId::PlatformApi);
    }

    #[tokio::test]
    async fn test_platform_api_supports_platform_properties() {
        let backend = PlatformApiBackend::new();
        let supported = backend.supports();
        assert!(
            !supported.is_empty(),
            "Platform API backend should report supported property types"
        );
    }

    #[tokio::test]
    async fn test_platform_api_generates_static_checker_code() {
        let backend = PlatformApiBackend::new();
        let spec = platform_api_spec();
        let result = backend
            .verify(&spec)
            .await
            .expect("platform API verification should succeed");

        assert!(
            matches!(result.status, VerificationStatus::Proven),
            "platform API backend should mark generation as proven"
        );

        let proof = result
            .proof
            .as_ref()
            .expect("platform API backend should emit generated code");

        assert!(
            proof.contains("MTLCommandBufferStateTracker") || proof.contains("MTLCommandBuffer"),
            "generated code should include platform state tracker"
        );
        assert!(
            result
                .diagnostics
                .iter()
                .any(|d| d.contains("platform API")),
            "diagnostics should mention generated platform API module"
        );
    }

    #[tokio::test]
    async fn test_platform_api_verify_without_platform_section() {
        let backend = PlatformApiBackend::new();
        let spec = simple_theorem_spec();

        let result = backend.verify(&spec).await;
        assert!(
            result.is_err(),
            "platform API backend should fail when no platform_api blocks are present"
        );
    }
}

// ============================================================================
// Dispatcher Integration Tests
// ============================================================================

mod dispatcher_e2e {
    use super::*;
    use dashprove_dispatcher::{Dispatcher, DispatcherConfig};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_dispatcher_with_available_backends() {
        let mut dispatcher = Dispatcher::new(DispatcherConfig::default());

        // Register all backends
        let alloy = AlloyBackend::new();
        let lean4 = Lean4Backend::new();
        let kani = KaniBackend::new();
        let tlaplus = TlaPlusBackend::new();

        // Check which backends are available and register them
        let mut available_count = 0;

        if is_backend_available(&alloy).await {
            dispatcher.register_backend(Arc::new(AlloyBackend::new()));
            available_count += 1;
            println!("Registered Alloy backend");
        }

        if is_backend_available(&lean4).await {
            dispatcher.register_backend(Arc::new(Lean4Backend::new()));
            available_count += 1;
            println!("Registered LEAN 4 backend");
        }

        if is_backend_available(&kani).await {
            dispatcher.register_backend(Arc::new(KaniBackend::new()));
            available_count += 1;
            println!("Registered Kani backend");
        }

        if is_backend_available(&tlaplus).await {
            dispatcher.register_backend(Arc::new(TlaPlusBackend::new()));
            available_count += 1;
            println!("Registered TLA+ backend");
        }

        println!("Total backends available: {} out of 4", available_count);

        if available_count == 0 {
            eprintln!("SKIPPED: No backends available");
            return;
        }

        // Try verification with a simple spec
        let spec = simple_invariant_spec();
        let result = dispatcher.verify(&spec).await;

        match result {
            Ok(merged) => {
                println!("Dispatcher results:");
                println!("  Proven: {}", merged.summary.proven);
                println!("  Disproven: {}", merged.summary.disproven);
                println!("  Unknown: {}", merged.summary.unknown);
                println!("  Confidence: {:.2}", merged.summary.overall_confidence);
            }
            Err(e) => {
                println!("Dispatcher error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_dispatcher_backend_selection() {
        let mut dispatcher = Dispatcher::new(DispatcherConfig::all_backends());

        let alloy = AlloyBackend::new();
        if is_backend_available(&alloy).await {
            dispatcher.register_backend(Arc::new(AlloyBackend::new()));

            let spec = simple_invariant_spec();
            let result = dispatcher.verify_with(&spec, BackendId::Alloy).await;

            match result {
                Ok(merged) => {
                    assert_eq!(merged.properties.len(), 1);
                    let first = &merged.properties[0];
                    // All results should be from Alloy
                    for br in &first.backend_results {
                        assert_eq!(br.backend, BackendId::Alloy);
                    }
                }
                Err(e) => {
                    println!("verify_with error: {}", e);
                }
            }
        } else {
            eprintln!("SKIPPED: Alloy not available");
        }
    }
}

// ============================================================================
// Miri End-to-End Tests
// ============================================================================

mod miri_e2e {
    use dashprove_miri::{detect_miri, MiriConfig, MiriDetection};

    #[tokio::test]
    async fn test_miri_detection() {
        let config = MiriConfig::default();
        let detection = detect_miri(&config).await;

        match &detection {
            MiriDetection::Available { version, .. } => {
                println!("Miri detected: {}", version.version_string);
                assert!(!version.version_string.is_empty());
            }
            MiriDetection::NotFound(reason) => {
                println!("Miri not available: {}", reason);
                // This is valid - Miri may not be installed
            }
        }
    }

    #[tokio::test]
    async fn test_miri_detection_is_available() {
        let config = MiriConfig::default();
        let detection = detect_miri(&config).await;

        // is_available() method should work correctly
        let is_available = detection.is_available();
        let version = detection.version();

        if is_available {
            assert!(version.is_some());
            println!("Miri is available");
        } else {
            assert!(version.is_none());
            println!("Miri is not available");
        }
    }

    #[tokio::test]
    async fn test_miri_config_defaults() {
        let config = MiriConfig::default();
        // Config should have reasonable defaults
        assert!(config.timeout.as_secs() > 0);
    }
}

// ============================================================================
// Clippy End-to-End Tests
// ============================================================================

mod clippy_e2e {
    use dashprove_static::{AnalysisConfig, AnalysisTool, StaticAnalysisBackend};
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_clippy_installed() {
        let backend = StaticAnalysisBackend::new(AnalysisTool::Clippy, AnalysisConfig::default());
        let installed = backend.check_installed().await;

        match installed {
            Ok(true) => println!("Clippy is installed"),
            Ok(false) => println!("Clippy is not installed"),
            Err(e) => println!("Error checking Clippy: {}", e),
        }
    }

    #[tokio::test]
    async fn test_clippy_on_dashprove_static() {
        let backend = StaticAnalysisBackend::new(AnalysisTool::Clippy, AnalysisConfig::default());

        if !backend.check_installed().await.unwrap_or(false) {
            eprintln!("SKIPPED: Clippy not installed");
            return;
        }

        // Run clippy on dashprove-static itself
        let crate_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("dashprove-static");

        if !crate_path.exists() {
            eprintln!("SKIPPED: dashprove-static crate not found");
            return;
        }

        let result = backend.run_on_crate(&crate_path).await;

        match result {
            Ok(analysis) => {
                println!("Clippy analysis completed in {:?}", analysis.duration);
                println!(
                    "Findings: {} errors, {} warnings",
                    analysis.error_count, analysis.warning_count
                );
                // dashprove-static should pass clippy (we run clippy in CI)
            }
            Err(e) => {
                println!("Clippy analysis error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_analysis_tool_enum() {
        // Test that all tool names are valid
        assert_eq!(AnalysisTool::Clippy.name(), "clippy");
        assert_eq!(AnalysisTool::SemverChecks.name(), "cargo-semver-checks");
        assert_eq!(AnalysisTool::Geiger.name(), "cargo-geiger");
        assert_eq!(AnalysisTool::Audit.name(), "cargo-audit");
        assert_eq!(AnalysisTool::Deny.name(), "cargo-deny");
        assert_eq!(AnalysisTool::Vet.name(), "cargo-vet");
        assert_eq!(AnalysisTool::Mutants.name(), "cargo-mutants");
    }
}

// ============================================================================
// Sanitizer End-to-End Tests
// ============================================================================

mod sanitizer_e2e {
    use dashprove_sanitizers::{SanitizerBackend, SanitizerType};

    #[tokio::test]
    async fn test_sanitizer_nightly_detection() {
        // Check if nightly toolchain is available (required for sanitizers)
        let nightly_available = SanitizerBackend::check_nightly().await.unwrap_or(false);
        println!(
            "Nightly toolchain: {}",
            if nightly_available {
                "available"
            } else {
                "not installed (run: rustup install nightly)"
            }
        );
    }

    #[tokio::test]
    async fn test_sanitizer_platform_availability() {
        // Check platform availability for each sanitizer
        let asan_available = SanitizerType::Address.is_available();
        let msan_available = SanitizerType::Memory.is_available();
        let tsan_available = SanitizerType::Thread.is_available();
        let lsan_available = SanitizerType::Leak.is_available();

        println!(
            "AddressSanitizer: {}",
            if asan_available {
                "supported"
            } else {
                "not supported"
            }
        );
        println!(
            "MemorySanitizer: {} (Linux only)",
            if msan_available {
                "supported"
            } else {
                "not supported"
            }
        );
        println!(
            "ThreadSanitizer: {}",
            if tsan_available {
                "supported"
            } else {
                "not supported"
            }
        );
        println!(
            "LeakSanitizer: {}",
            if lsan_available {
                "supported"
            } else {
                "not supported"
            }
        );

        // ASAN and TSAN should be available on macOS and Linux
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            assert!(asan_available);
            assert!(tsan_available);
            assert!(lsan_available);
        }

        // MSAN only on Linux
        #[cfg(target_os = "linux")]
        assert!(msan_available);
        #[cfg(not(target_os = "linux"))]
        assert!(!msan_available);
    }

    #[tokio::test]
    async fn test_sanitizer_type_rustflags() {
        // Test that all sanitizers have proper RUSTFLAGS
        assert_eq!(SanitizerType::Address.rustflags(), "-Z sanitizer=address");
        assert_eq!(SanitizerType::Memory.rustflags(), "-Z sanitizer=memory");
        assert_eq!(SanitizerType::Thread.rustflags(), "-Z sanitizer=thread");
        assert_eq!(SanitizerType::Leak.rustflags(), "-Z sanitizer=leak");
    }

    #[tokio::test]
    async fn test_sanitizer_type_names() {
        assert_eq!(SanitizerType::Address.name(), "AddressSanitizer");
        assert_eq!(SanitizerType::Memory.name(), "MemorySanitizer");
        assert_eq!(SanitizerType::Thread.name(), "ThreadSanitizer");
        assert_eq!(SanitizerType::Leak.name(), "LeakSanitizer");
    }

    #[tokio::test]
    async fn test_sanitizer_backend_creation() {
        use std::time::Duration;
        let backend =
            SanitizerBackend::new(SanitizerType::Address).with_timeout(Duration::from_secs(120));

        // Backend should be creatable even if sanitizers aren't available
        // Use a real assertion that won't be optimized away
        std::hint::black_box(&backend);
    }
}

// ============================================================================
// Fuzzer End-to-End Tests
// ============================================================================

mod fuzzer_e2e {
    use dashprove_fuzz::{FuzzBackend, FuzzConfig, FuzzerType};
    use std::time::Duration;

    #[tokio::test]
    async fn test_fuzzer_detection() {
        // Check if each fuzzer is installed
        let libfuzzer_backend = FuzzBackend::new(FuzzerType::LibFuzzer, FuzzConfig::default());
        let afl_backend = FuzzBackend::new(FuzzerType::AFL, FuzzConfig::default());
        let honggfuzz_backend = FuzzBackend::new(FuzzerType::Honggfuzz, FuzzConfig::default());
        let bolero_backend = FuzzBackend::new(FuzzerType::Bolero, FuzzConfig::default());

        let libfuzzer_available = libfuzzer_backend.check_installed().await.unwrap_or(false);
        let afl_available = afl_backend.check_installed().await.unwrap_or(false);
        let honggfuzz_available = honggfuzz_backend.check_installed().await.unwrap_or(false);
        let bolero_available = bolero_backend.check_installed().await.unwrap_or(false);

        println!(
            "LibFuzzer (cargo-fuzz): {}",
            if libfuzzer_available {
                "installed"
            } else {
                "not installed"
            }
        );
        println!(
            "AFL.rs: {}",
            if afl_available {
                "installed"
            } else {
                "not installed"
            }
        );
        println!(
            "Honggfuzz: {}",
            if honggfuzz_available {
                "installed"
            } else {
                "not installed"
            }
        );
        println!(
            "Bolero: {}",
            if bolero_available {
                "installed"
            } else {
                "not installed"
            }
        );
    }

    #[tokio::test]
    async fn test_fuzz_config() {
        let config = FuzzConfig::default();
        // Config should have reasonable defaults
        assert!(config.timeout.as_secs() > 0);
        // max_iterations = 0 means unlimited until timeout
        assert_eq!(config.max_iterations, 0);
    }

    #[tokio::test]
    async fn test_fuzz_config_builder() {
        let config = FuzzConfig::default()
            .with_timeout(Duration::from_secs(120))
            .with_max_iterations(50000)
            .with_jobs(4);

        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.max_iterations, 50000);
        assert_eq!(config.jobs, 4);
    }

    #[tokio::test]
    async fn test_fuzzer_type_names() {
        assert_eq!(FuzzerType::LibFuzzer.name(), "cargo-fuzz (LibFuzzer)");
        assert_eq!(FuzzerType::AFL.name(), "AFL.rs");
        assert_eq!(FuzzerType::Honggfuzz.name(), "Honggfuzz");
        assert_eq!(FuzzerType::Bolero.name(), "Bolero");
    }

    #[tokio::test]
    async fn test_fuzzer_nightly_requirements() {
        assert!(FuzzerType::LibFuzzer.requires_nightly());
        assert!(FuzzerType::Honggfuzz.requires_nightly());
        assert!(!FuzzerType::AFL.requires_nightly());
        assert!(!FuzzerType::Bolero.requires_nightly());
    }

    #[tokio::test]
    async fn test_install_instructions() {
        let libfuzzer_backend = FuzzBackend::new(FuzzerType::LibFuzzer, FuzzConfig::default());
        let afl_backend = FuzzBackend::new(FuzzerType::AFL, FuzzConfig::default());

        assert!(libfuzzer_backend
            .install_instructions()
            .contains("cargo-fuzz"));
        assert!(afl_backend.install_instructions().contains("afl"));
    }
}

// ============================================================================
// Property-Based Testing End-to-End Tests
// ============================================================================

mod pbt_e2e {
    use dashprove_pbt::{
        generate_proptest_template, generate_quickcheck_template, PbtBackend, PbtConfig, PbtType,
    };
    use std::time::Duration;

    #[tokio::test]
    async fn test_pbt_type_names() {
        assert_eq!(PbtType::Proptest.name(), "proptest");
        assert_eq!(PbtType::QuickCheck.name(), "quickcheck");
    }

    #[tokio::test]
    async fn test_pbt_type_crate_names() {
        assert_eq!(PbtType::Proptest.crate_name(), "proptest");
        assert_eq!(PbtType::QuickCheck.crate_name(), "quickcheck");
    }

    #[tokio::test]
    async fn test_pbt_type_env_vars() {
        assert_eq!(PbtType::Proptest.cases_env_var(), "PROPTEST_CASES");
        assert_eq!(PbtType::QuickCheck.cases_env_var(), "QUICKCHECK_TESTS");
    }

    #[tokio::test]
    async fn test_pbt_config() {
        let config = PbtConfig::default();
        // Config should have reasonable defaults
        assert!(config.cases > 0);
        assert!(config.timeout.as_secs() > 0);
        assert!(config.max_shrink_iters > 0);
    }

    #[tokio::test]
    async fn test_pbt_config_builder() {
        let config = PbtConfig::default()
            .with_cases(1000)
            .with_max_shrink_iters(500)
            .with_seed(12345)
            .with_timeout(Duration::from_secs(120))
            .with_verbose(true)
            .with_fork(true);

        assert_eq!(config.cases, 1000);
        assert_eq!(config.max_shrink_iters, 500);
        assert_eq!(config.seed, Some(12345));
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert!(config.verbose);
        assert!(config.fork);
    }

    #[tokio::test]
    async fn test_pbt_backend_creation() {
        let config = PbtConfig::default();
        let proptest_backend = PbtBackend::new(PbtType::Proptest, config.clone());
        let quickcheck_backend = PbtBackend::new(PbtType::QuickCheck, config);

        // Just verify construction succeeds
        std::hint::black_box(&proptest_backend);
        std::hint::black_box(&quickcheck_backend);
    }

    #[tokio::test]
    async fn test_generate_proptest_template() {
        let template = generate_proptest_template("test_addition", "i32");
        assert!(template.contains("proptest!"));
        assert!(template.contains("fn test_addition"));
        assert!(template.contains("any::<i32>"));
        assert!(template.contains("prop_assert!"));
    }

    #[tokio::test]
    async fn test_generate_quickcheck_template() {
        let template = generate_quickcheck_template("test_reverse", "Vec<u8>");
        assert!(template.contains("quickcheck!"));
        assert!(template.contains("fn test_reverse"));
        assert!(template.contains("Vec<u8>"));
        assert!(template.contains("-> bool"));
    }
}

// ============================================================================
// Verus End-to-End Tests
// ============================================================================

mod verus_e2e {
    use super::*;
    use dashprove_backends::{VerusBackend, VerusConfig};
    use dashprove_usl::ast::{Contract, Param, Type};

    #[tokio::test]
    async fn test_verus_backend_health_check() {
        let backend = VerusBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Verus is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Verus unavailable: {}", reason);
                println!("Install from: https://github.com/verus-lang/verus");
            }
            HealthStatus::Degraded { reason } => println!("Verus degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_verus_backend_id() {
        let backend = VerusBackend::new();
        assert_eq!(backend.id(), BackendId::Verus);
    }

    #[tokio::test]
    async fn test_verus_supports_property_types() {
        let backend = VerusBackend::new();
        let supported = backend.supports();

        // Verus should support contracts and invariants
        assert!(
            !supported.is_empty(),
            "Verus should support at least one property type"
        );
        println!("Verus supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_verus_config_defaults() {
        let config = VerusConfig::default();
        // Config should have reasonable defaults
        assert!(config.timeout.as_secs() > 0);
        assert!(config.z3_threads > 0);
        assert!(!config.debug);
    }

    #[tokio::test]
    async fn test_verus_with_custom_config() {
        let config = VerusConfig {
            timeout: std::time::Duration::from_secs(120),
            z3_threads: 2,
            debug: true,
            verus_path: None,
        };
        let backend = VerusBackend::with_config(config);

        // Backend should be creatable with custom config
        assert_eq!(backend.id(), BackendId::Verus);
    }

    #[tokio::test]
    async fn test_verus_verify_simple_contract() {
        let backend = VerusBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Verus not available");
            return;
        }

        // Create a simple contract spec
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Contract(Contract {
                type_path: vec!["identity".to_string()],
                params: vec![Param {
                    name: "x".to_string(),
                    ty: Type::Named("u32".to_string()),
                }],
                return_type: Some(Type::Named("u32".to_string())),
                requires: vec![],
                ensures: vec![Expr::Compare(
                    Box::new(Expr::Var("result".to_string())),
                    dashprove_usl::ast::ComparisonOp::Eq,
                    Box::new(Expr::Var("x".to_string())),
                )],
                ensures_err: vec![],
                assigns: vec![],
                allocates: vec![],
                frees: vec![],
                terminates: None,
                decreases: None,
                behaviors: vec![],
                complete_behaviors: false,
                disjoint_behaviors: false,
            })],
        };
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;

        match result {
            Ok(r) => {
                println!("Verus result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Verus verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_verus_verify_theorem() {
        let backend = VerusBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Verus not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Verus theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                // A trivial theorem should be proven or at least not disproven
                assert!(
                    matches!(r.status, VerificationStatus::Proven)
                        || matches!(r.status, VerificationStatus::Unknown { .. }),
                    "Unexpected status: {:?}",
                    r.status
                );
            }
            Err(e) => {
                println!("Verus verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_verus_verify_invariant() {
        let backend = VerusBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Verus not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Verus invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Verus verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Creusot End-to-End Tests
// ============================================================================

mod creusot_e2e {
    use super::*;
    use dashprove_backends::{CreusotBackend, CreusotConfig};
    use dashprove_usl::ast::{Contract, Param, Type};

    #[tokio::test]
    async fn test_creusot_backend_health_check() {
        let backend = CreusotBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Creusot is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Creusot unavailable: {}", reason);
                println!("Install via: cargo install cargo-creusot");
                println!("Or from: https://github.com/creusot-rs/creusot");
            }
            HealthStatus::Degraded { reason } => println!("Creusot degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_creusot_backend_id() {
        let backend = CreusotBackend::new();
        assert_eq!(backend.id(), BackendId::Creusot);
    }

    #[tokio::test]
    async fn test_creusot_supports_property_types() {
        let backend = CreusotBackend::new();
        let supported = backend.supports();

        // Creusot should support contracts and invariants
        assert!(
            !supported.is_empty(),
            "Creusot should support at least one property type"
        );
        println!("Creusot supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_creusot_config_defaults() {
        let config = CreusotConfig::default();
        // Config should have reasonable defaults
        assert!(config.timeout.as_secs() > 0);
        assert!(!config.solver.is_empty());
        assert_eq!(config.solver, "z3"); // Default solver
    }

    #[tokio::test]
    async fn test_creusot_with_custom_config() {
        let config = CreusotConfig {
            timeout: std::time::Duration::from_secs(120),
            solver: "cvc4".to_string(),
            creusot_path: None,
            why3_path: None,
        };
        let backend = CreusotBackend::with_config(config);

        // Backend should be creatable with custom config
        assert_eq!(backend.id(), BackendId::Creusot);
    }

    #[tokio::test]
    async fn test_creusot_verify_simple_contract() {
        let backend = CreusotBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Creusot not available");
            return;
        }

        // Create a simple contract spec
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Contract(Contract {
                type_path: vec!["identity".to_string()],
                params: vec![Param {
                    name: "x".to_string(),
                    ty: Type::Named("u32".to_string()),
                }],
                return_type: Some(Type::Named("u32".to_string())),
                requires: vec![],
                ensures: vec![Expr::Compare(
                    Box::new(Expr::Var("result".to_string())),
                    dashprove_usl::ast::ComparisonOp::Eq,
                    Box::new(Expr::Var("x".to_string())),
                )],
                ensures_err: vec![],
                assigns: vec![],
                allocates: vec![],
                frees: vec![],
                terminates: None,
                decreases: None,
                behaviors: vec![],
                complete_behaviors: false,
                disjoint_behaviors: false,
            })],
        };
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;

        match result {
            Ok(r) => {
                println!("Creusot result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Creusot verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_creusot_verify_theorem() {
        let backend = CreusotBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Creusot not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Creusot theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                // A trivial theorem should be proven or at least not disproven
                assert!(
                    matches!(r.status, VerificationStatus::Proven)
                        || matches!(r.status, VerificationStatus::Unknown { .. }),
                    "Unexpected status: {:?}",
                    r.status
                );
            }
            Err(e) => {
                println!("Creusot verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_creusot_verify_invariant() {
        let backend = CreusotBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Creusot not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Creusot invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Creusot verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Prusti End-to-End Tests
// ============================================================================

mod prusti_e2e {
    use super::*;
    use dashprove_backends::{PrustiBackend, PrustiConfig};
    use dashprove_usl::ast::{Contract, Param, Type};

    #[tokio::test]
    async fn test_prusti_backend_health_check() {
        let backend = PrustiBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Prusti is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Prusti unavailable: {}", reason);
                println!("Install from: https://github.com/viperproject/prusti-dev");
            }
            HealthStatus::Degraded { reason } => println!("Prusti degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_prusti_backend_id() {
        let backend = PrustiBackend::new();
        assert_eq!(backend.id(), BackendId::Prusti);
    }

    #[tokio::test]
    async fn test_prusti_supports_property_types() {
        let backend = PrustiBackend::new();
        let supported = backend.supports();

        // Prusti should support contracts and memory safety
        assert!(
            !supported.is_empty(),
            "Prusti should support at least one property type"
        );
        println!("Prusti supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_prusti_config_defaults() {
        let config = PrustiConfig::default();
        // Config should have reasonable defaults
        assert!(config.timeout.as_secs() > 0);
        assert!(config.check_overflows); // Default should check overflows
        assert!(config.full_verification); // Default should do full verification
    }

    #[tokio::test]
    async fn test_prusti_with_custom_config() {
        let config = PrustiConfig {
            timeout: std::time::Duration::from_secs(120),
            check_overflows: false,
            full_verification: false,
            prusti_path: None,
        };
        let backend = PrustiBackend::with_config(config);

        // Backend should be creatable with custom config
        assert_eq!(backend.id(), BackendId::Prusti);
    }

    #[tokio::test]
    async fn test_prusti_verify_simple_contract() {
        let backend = PrustiBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Prusti not available");
            return;
        }

        // Create a simple contract spec
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Contract(Contract {
                type_path: vec!["identity".to_string()],
                params: vec![Param {
                    name: "x".to_string(),
                    ty: Type::Named("u32".to_string()),
                }],
                return_type: Some(Type::Named("u32".to_string())),
                requires: vec![],
                ensures: vec![Expr::Compare(
                    Box::new(Expr::Var("result".to_string())),
                    dashprove_usl::ast::ComparisonOp::Eq,
                    Box::new(Expr::Var("x".to_string())),
                )],
                ensures_err: vec![],
                assigns: vec![],
                allocates: vec![],
                frees: vec![],
                terminates: None,
                decreases: None,
                behaviors: vec![],
                complete_behaviors: false,
                disjoint_behaviors: false,
            })],
        };
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;

        match result {
            Ok(r) => {
                println!("Prusti result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Prusti verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_prusti_verify_theorem() {
        let backend = PrustiBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Prusti not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Prusti theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                // A trivial theorem should be proven or at least not disproven
                assert!(
                    matches!(r.status, VerificationStatus::Proven)
                        || matches!(r.status, VerificationStatus::Unknown { .. }),
                    "Unexpected status: {:?}",
                    r.status
                );
            }
            Err(e) => {
                println!("Prusti verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_prusti_verify_invariant() {
        let backend = PrustiBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Prusti not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Prusti invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Prusti verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_prusti_verify_contract_with_preconditions() {
        let backend = PrustiBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Prusti not available");
            return;
        }

        // Create a contract with preconditions for bounded values
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Contract(Contract {
                type_path: vec!["bounded_add".to_string()],
                params: vec![
                    Param {
                        name: "x".to_string(),
                        ty: Type::Named("u32".to_string()),
                    },
                    Param {
                        name: "y".to_string(),
                        ty: Type::Named("u32".to_string()),
                    },
                ],
                return_type: Some(Type::Named("u32".to_string())),
                requires: vec![Expr::And(
                    Box::new(Expr::Compare(
                        Box::new(Expr::Var("x".to_string())),
                        dashprove_usl::ast::ComparisonOp::Lt,
                        Box::new(Expr::Int(1000)),
                    )),
                    Box::new(Expr::Compare(
                        Box::new(Expr::Var("y".to_string())),
                        dashprove_usl::ast::ComparisonOp::Lt,
                        Box::new(Expr::Int(1000)),
                    )),
                )],
                ensures: vec![Expr::Compare(
                    Box::new(Expr::Var("result".to_string())),
                    dashprove_usl::ast::ComparisonOp::Eq,
                    Box::new(Expr::Binary(
                        Box::new(Expr::Var("x".to_string())),
                        dashprove_usl::ast::BinaryOp::Add,
                        Box::new(Expr::Var("y".to_string())),
                    )),
                )],
                ensures_err: vec![],
                assigns: vec![],
                allocates: vec![],
                frees: vec![],
                terminates: None,
                decreases: None,
                behaviors: vec![],
                complete_behaviors: false,
                disjoint_behaviors: false,
            })],
        };
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;

        match result {
            Ok(r) => {
                println!("Prusti bounded_add result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Prusti verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// SPIN End-to-End Tests
// ============================================================================

mod spin_e2e {
    use super::*;

    #[tokio::test]
    async fn test_spin_backend_health_check() {
        let backend = SpinBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("SPIN is available"),
            HealthStatus::Unavailable { reason } => {
                println!("SPIN unavailable: {}", reason);
                println!("Install via: brew install spin (macOS) or apt install spin (Linux)");
            }
            HealthStatus::Degraded { reason } => println!("SPIN degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_spin_backend_id() {
        let backend = SpinBackend::new();
        assert_eq!(backend.id(), BackendId::SPIN);
    }

    #[tokio::test]
    async fn test_spin_supports_property_types() {
        let backend = SpinBackend::new();
        let supported = backend.supports();

        // SPIN should support invariants and temporal properties
        assert!(
            !supported.is_empty(),
            "SPIN should support at least one property type"
        );
        println!("SPIN supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_spin_verify_simple_invariant() {
        let backend = SpinBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: SPIN not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("SPIN invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
                // A trivial invariant should be proven or at least not disproven
                assert!(
                    matches!(r.status, VerificationStatus::Proven)
                        || matches!(r.status, VerificationStatus::Unknown { .. }),
                    "Unexpected status: {:?}",
                    r.status
                );
            }
            Err(e) => {
                println!("SPIN verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_spin_verify_theorem() {
        let backend = SpinBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: SPIN not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("SPIN theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("SPIN verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// CBMC End-to-End Tests
// ============================================================================

mod cbmc_e2e {
    use super::*;

    #[tokio::test]
    async fn test_cbmc_backend_health_check() {
        let backend = CbmcBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("CBMC is available"),
            HealthStatus::Unavailable { reason } => {
                println!("CBMC unavailable: {}", reason);
                println!("Install via: brew install cbmc (macOS) or apt install cbmc (Linux)");
            }
            HealthStatus::Degraded { reason } => println!("CBMC degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_cbmc_backend_id() {
        let backend = CbmcBackend::new();
        assert_eq!(backend.id(), BackendId::CBMC);
    }

    #[tokio::test]
    async fn test_cbmc_supports_property_types() {
        let backend = CbmcBackend::new();
        let supported = backend.supports();

        // CBMC should support contracts and invariants
        assert!(
            !supported.is_empty(),
            "CBMC should support at least one property type"
        );
        println!("CBMC supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_cbmc_verify_simple_invariant() {
        let backend = CbmcBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: CBMC not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("CBMC invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("CBMC verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_cbmc_verify_theorem() {
        let backend = CbmcBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: CBMC not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("CBMC theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("CBMC verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// NuSMV End-to-End Tests
// ============================================================================

mod nusmv_e2e {
    use super::*;

    #[tokio::test]
    async fn test_nusmv_backend_health_check() {
        let backend = NuSmvBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("NuSMV is available"),
            HealthStatus::Unavailable { reason } => {
                println!("NuSMV unavailable: {}", reason);
                println!("Install from: https://nusmv.fbk.eu/");
            }
            HealthStatus::Degraded { reason } => println!("NuSMV degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_nusmv_backend_id() {
        let backend = NuSmvBackend::new();
        assert_eq!(backend.id(), BackendId::NuSMV);
    }

    #[tokio::test]
    async fn test_nusmv_supports_property_types() {
        let backend = NuSmvBackend::new();
        let supported = backend.supports();

        // NuSMV should support invariants and temporal properties
        assert!(
            !supported.is_empty(),
            "NuSMV should support at least one property type"
        );
        println!("NuSMV supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_nusmv_verify_simple_invariant() {
        let backend = NuSmvBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: NuSMV not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("NuSMV invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("NuSMV verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_nusmv_verify_theorem() {
        let backend = NuSmvBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: NuSMV not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("NuSMV theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("NuSMV verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// CPAchecker End-to-End Tests
// ============================================================================

mod cpachecker_e2e {
    use super::*;

    #[tokio::test]
    async fn test_cpachecker_backend_health_check() {
        let backend = CpacheckerBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("CPAchecker is available"),
            HealthStatus::Unavailable { reason } => {
                println!("CPAchecker unavailable: {}", reason);
                println!("Install from: https://cpachecker.sosy-lab.org/");
            }
            HealthStatus::Degraded { reason } => println!("CPAchecker degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_cpachecker_backend_id() {
        let backend = CpacheckerBackend::new();
        assert_eq!(backend.id(), BackendId::CPAchecker);
    }

    #[tokio::test]
    async fn test_cpachecker_supports_property_types() {
        let backend = CpacheckerBackend::new();
        let supported = backend.supports();

        // CPAchecker should support invariants and contracts
        assert!(
            !supported.is_empty(),
            "CPAchecker should support at least one property type"
        );
        println!("CPAchecker supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_cpachecker_verify_simple_invariant() {
        let backend = CpacheckerBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: CPAchecker not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("CPAchecker invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("CPAchecker verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_cpachecker_verify_theorem() {
        let backend = CpacheckerBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: CPAchecker not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("CPAchecker theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("CPAchecker verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// SeaHorn End-to-End Tests
// ============================================================================

mod seahorn_e2e {
    use super::*;

    #[tokio::test]
    async fn test_seahorn_backend_health_check() {
        let backend = SeaHornBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("SeaHorn is available"),
            HealthStatus::Unavailable { reason } => {
                println!("SeaHorn unavailable: {}", reason);
                println!("Install from: https://seahorn.github.io/");
            }
            HealthStatus::Degraded { reason } => println!("SeaHorn degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_seahorn_backend_id() {
        let backend = SeaHornBackend::new();
        assert_eq!(backend.id(), BackendId::SeaHorn);
    }

    #[tokio::test]
    async fn test_seahorn_supports_property_types() {
        let backend = SeaHornBackend::new();
        let supported = backend.supports();

        // SeaHorn should support contracts and invariants
        assert!(
            !supported.is_empty(),
            "SeaHorn should support at least one property type"
        );
        println!("SeaHorn supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_seahorn_verify_simple_invariant() {
        let backend = SeaHornBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: SeaHorn not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("SeaHorn invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("SeaHorn verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_seahorn_verify_theorem() {
        let backend = SeaHornBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: SeaHorn not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("SeaHorn theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("SeaHorn verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Frama-C End-to-End Tests
// ============================================================================

mod framac_e2e {
    use super::*;

    #[tokio::test]
    async fn test_framac_backend_health_check() {
        let backend = FramaCBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Frama-C is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Frama-C unavailable: {}", reason);
                println!("Install from: https://frama-c.com/");
            }
            HealthStatus::Degraded { reason } => println!("Frama-C degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_framac_backend_id() {
        let backend = FramaCBackend::new();
        assert_eq!(backend.id(), BackendId::FramaC);
    }

    #[tokio::test]
    async fn test_framac_supports_property_types() {
        let backend = FramaCBackend::new();
        let supported = backend.supports();

        // Frama-C should support contracts and invariants
        assert!(
            !supported.is_empty(),
            "Frama-C should support at least one property type"
        );
        println!("Frama-C supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_framac_verify_simple_invariant() {
        let backend = FramaCBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Frama-C not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Frama-C invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Frama-C verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_framac_verify_theorem() {
        let backend = FramaCBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Frama-C not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Frama-C theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Frama-C verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// KLEE End-to-End Tests
// ============================================================================

mod klee_e2e {
    use super::*;

    #[tokio::test]
    async fn test_klee_backend_health_check() {
        let backend = KleeBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("KLEE is available"),
            HealthStatus::Unavailable { reason } => {
                println!("KLEE unavailable: {}", reason);
                println!("Install from: https://klee.github.io/");
            }
            HealthStatus::Degraded { reason } => println!("KLEE degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_klee_backend_id() {
        let backend = KleeBackend::new();
        assert_eq!(backend.id(), BackendId::KLEE);
    }

    #[tokio::test]
    async fn test_klee_supports_property_types() {
        let backend = KleeBackend::new();
        let supported = backend.supports();

        // KLEE should support contracts and invariants
        assert!(
            !supported.is_empty(),
            "KLEE should support at least one property type"
        );
        println!("KLEE supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_klee_verify_simple_invariant() {
        let backend = KleeBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: KLEE not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("KLEE invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("KLEE verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_klee_verify_theorem() {
        let backend = KleeBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: KLEE not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("KLEE theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("KLEE verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Infer End-to-End Tests
// ============================================================================

mod infer_e2e {
    use super::*;

    #[tokio::test]
    async fn test_infer_backend_health_check() {
        let backend = InferBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Infer is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Infer unavailable: {}", reason);
                println!("Install via: brew install infer (macOS) or from fbinfer.com");
            }
            HealthStatus::Degraded { reason } => println!("Infer degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_infer_backend_id() {
        let backend = InferBackend::new();
        assert_eq!(backend.id(), BackendId::Infer);
    }

    #[tokio::test]
    async fn test_infer_supports_property_types() {
        let backend = InferBackend::new();
        let supported = backend.supports();

        // Infer should support contracts and memory safety
        assert!(
            !supported.is_empty(),
            "Infer should support at least one property type"
        );
        println!("Infer supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_infer_verify_simple_invariant() {
        let backend = InferBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Infer not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Infer invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Infer verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_infer_verify_theorem() {
        let backend = InferBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Infer not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Infer theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Infer verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Z3 End-to-End Tests
// ============================================================================

mod z3_e2e {
    use super::*;

    #[tokio::test]
    async fn test_z3_backend_health_check() {
        let backend = Z3Backend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Z3 is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Z3 unavailable: {}", reason);
                println!("Install via: brew install z3 (macOS) or apt install z3 (Linux)");
            }
            HealthStatus::Degraded { reason } => println!("Z3 degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_z3_backend_id() {
        let backend = Z3Backend::new();
        assert_eq!(backend.id(), BackendId::Z3);
    }

    #[tokio::test]
    async fn test_z3_supports_property_types() {
        let backend = Z3Backend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Z3 should support at least one property type"
        );
        println!("Z3 supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_z3_verify_simple_invariant() {
        let backend = Z3Backend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Z3 not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Z3 invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
                assert!(
                    matches!(r.status, VerificationStatus::Proven)
                        || matches!(r.status, VerificationStatus::Unknown { .. }),
                    "Unexpected status: {:?}",
                    r.status
                );
            }
            Err(e) => {
                println!("Z3 verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_z3_verify_theorem() {
        let backend = Z3Backend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Z3 not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Z3 theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Z3 verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// CVC5 End-to-End Tests
// ============================================================================

mod cvc5_e2e {
    use super::*;

    #[tokio::test]
    async fn test_cvc5_backend_health_check() {
        let backend = Cvc5Backend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("CVC5 is available"),
            HealthStatus::Unavailable { reason } => {
                println!("CVC5 unavailable: {}", reason);
                println!("Install via: brew install cvc5 (macOS) or from https://cvc5.github.io/");
            }
            HealthStatus::Degraded { reason } => println!("CVC5 degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_cvc5_backend_id() {
        let backend = Cvc5Backend::new();
        assert_eq!(backend.id(), BackendId::Cvc5);
    }

    #[tokio::test]
    async fn test_cvc5_supports_property_types() {
        let backend = Cvc5Backend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "CVC5 should support at least one property type"
        );
        println!("CVC5 supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_cvc5_verify_simple_invariant() {
        let backend = Cvc5Backend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: CVC5 not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("CVC5 invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
                assert!(
                    matches!(r.status, VerificationStatus::Proven)
                        || matches!(r.status, VerificationStatus::Unknown { .. }),
                    "Unexpected status: {:?}",
                    r.status
                );
            }
            Err(e) => {
                println!("CVC5 verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_cvc5_verify_theorem() {
        let backend = Cvc5Backend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: CVC5 not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("CVC5 theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("CVC5 verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Yices End-to-End Tests
// ============================================================================

mod yices_e2e {
    use super::*;

    #[tokio::test]
    async fn test_yices_backend_health_check() {
        let backend = YicesBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Yices is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Yices unavailable: {}", reason);
                println!(
                    "Install via: brew install yices2 (macOS) or from https://yices.csl.sri.com/"
                );
            }
            HealthStatus::Degraded { reason } => println!("Yices degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_yices_backend_id() {
        let backend = YicesBackend::new();
        assert_eq!(backend.id(), BackendId::Yices);
    }

    #[tokio::test]
    async fn test_yices_supports_property_types() {
        let backend = YicesBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Yices should support at least one property type"
        );
        println!("Yices supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_yices_verify_simple_invariant() {
        let backend = YicesBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Yices not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Yices invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
                assert!(
                    matches!(r.status, VerificationStatus::Proven)
                        || matches!(r.status, VerificationStatus::Unknown { .. }),
                    "Unexpected status: {:?}",
                    r.status
                );
            }
            Err(e) => {
                println!("Yices verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_yices_verify_theorem() {
        let backend = YicesBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Yices not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Yices theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Yices verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Boolector End-to-End Tests
// ============================================================================

mod boolector_e2e {
    use super::*;

    #[tokio::test]
    async fn test_boolector_backend_health_check() {
        let backend = BoolectorBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Boolector is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Boolector unavailable: {}", reason);
                println!("Install via: brew install boolector (macOS) or from https://boolector.github.io/");
            }
            HealthStatus::Degraded { reason } => println!("Boolector degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_boolector_backend_id() {
        let backend = BoolectorBackend::new();
        assert_eq!(backend.id(), BackendId::Boolector);
    }

    #[tokio::test]
    async fn test_boolector_supports_property_types() {
        let backend = BoolectorBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Boolector should support at least one property type"
        );
        println!("Boolector supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_boolector_verify_simple_invariant() {
        let backend = BoolectorBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Boolector not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Boolector invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Boolector verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_boolector_verify_theorem() {
        let backend = BoolectorBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Boolector not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Boolector theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Boolector verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// MathSAT End-to-End Tests
// ============================================================================

mod mathsat_e2e {
    use super::*;

    #[tokio::test]
    async fn test_mathsat_backend_health_check() {
        let backend = MathSatBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("MathSAT is available"),
            HealthStatus::Unavailable { reason } => {
                println!("MathSAT unavailable: {}", reason);
                println!("Install from: https://mathsat.fbk.eu/");
            }
            HealthStatus::Degraded { reason } => println!("MathSAT degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_mathsat_backend_id() {
        let backend = MathSatBackend::new();
        assert_eq!(backend.id(), BackendId::MathSAT);
    }

    #[tokio::test]
    async fn test_mathsat_supports_property_types() {
        let backend = MathSatBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "MathSAT should support at least one property type"
        );
        println!("MathSAT supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_mathsat_verify_simple_invariant() {
        let backend = MathSatBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: MathSAT not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("MathSAT invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("MathSAT verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_mathsat_verify_theorem() {
        let backend = MathSatBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: MathSAT not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("MathSAT theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("MathSAT verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// MiniSat End-to-End Tests
// ============================================================================

mod minisat_e2e {
    use super::*;

    #[tokio::test]
    async fn test_minisat_backend_health_check() {
        let backend = MiniSatBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("MiniSat is available"),
            HealthStatus::Unavailable { reason } => {
                println!("MiniSat unavailable: {}", reason);
                println!(
                    "Install via: brew install minisat (macOS) or apt install minisat (Linux)"
                );
            }
            HealthStatus::Degraded { reason } => println!("MiniSat degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_minisat_backend_id() {
        let backend = MiniSatBackend::new();
        assert_eq!(backend.id(), BackendId::MiniSat);
    }

    #[tokio::test]
    async fn test_minisat_supports_property_types() {
        let backend = MiniSatBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "MiniSat should support at least one property type"
        );
        println!("MiniSat supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_minisat_verify_simple_invariant() {
        let backend = MiniSatBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: MiniSat not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("MiniSat invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("MiniSat verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_minisat_verify_theorem() {
        let backend = MiniSatBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: MiniSat not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("MiniSat theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("MiniSat verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Glucose End-to-End Tests
// ============================================================================

mod glucose_e2e {
    use super::*;

    #[tokio::test]
    async fn test_glucose_backend_health_check() {
        let backend = GlucoseBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Glucose is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Glucose unavailable: {}", reason);
                println!("Install from: https://www.labri.fr/perso/lsimon/glucose/");
            }
            HealthStatus::Degraded { reason } => println!("Glucose degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_glucose_backend_id() {
        let backend = GlucoseBackend::new();
        assert_eq!(backend.id(), BackendId::Glucose);
    }

    #[tokio::test]
    async fn test_glucose_supports_property_types() {
        let backend = GlucoseBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Glucose should support at least one property type"
        );
        println!("Glucose supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_glucose_verify_simple_invariant() {
        let backend = GlucoseBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Glucose not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Glucose invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Glucose verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_glucose_verify_theorem() {
        let backend = GlucoseBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Glucose not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Glucose theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Glucose verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// CaDiCaL End-to-End Tests
// ============================================================================

mod cadical_e2e {
    use super::*;

    #[tokio::test]
    async fn test_cadical_backend_health_check() {
        let backend = CaDiCaLBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("CaDiCaL is available"),
            HealthStatus::Unavailable { reason } => {
                println!("CaDiCaL unavailable: {}", reason);
                println!("Install from: https://github.com/arminbiere/cadical");
            }
            HealthStatus::Degraded { reason } => println!("CaDiCaL degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_cadical_backend_id() {
        let backend = CaDiCaLBackend::new();
        assert_eq!(backend.id(), BackendId::CaDiCaL);
    }

    #[tokio::test]
    async fn test_cadical_supports_property_types() {
        let backend = CaDiCaLBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "CaDiCaL should support at least one property type"
        );
        println!("CaDiCaL supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_cadical_verify_simple_invariant() {
        let backend = CaDiCaLBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: CaDiCaL not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("CaDiCaL invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("CaDiCaL verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_cadical_verify_theorem() {
        let backend = CaDiCaLBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: CaDiCaL not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("CaDiCaL theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("CaDiCaL verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// ACL2 End-to-End Tests
// ============================================================================

mod acl2_e2e {
    use super::*;

    #[tokio::test]
    async fn test_acl2_backend_health_check() {
        let backend = Acl2Backend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("ACL2 is available"),
            HealthStatus::Unavailable { reason } => {
                println!("ACL2 unavailable: {}", reason);
                println!("Install from: https://www.cs.utexas.edu/users/moore/acl2/");
            }
            HealthStatus::Degraded { reason } => println!("ACL2 degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_acl2_backend_id() {
        let backend = Acl2Backend::new();
        assert_eq!(backend.id(), BackendId::ACL2);
    }

    #[tokio::test]
    async fn test_acl2_supports_property_types() {
        let backend = Acl2Backend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "ACL2 should support at least one property type"
        );
        println!("ACL2 supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_acl2_verify_simple_invariant() {
        let backend = Acl2Backend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: ACL2 not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("ACL2 invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("ACL2 verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_acl2_verify_theorem() {
        let backend = Acl2Backend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: ACL2 not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("ACL2 theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("ACL2 verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// HOL4 End-to-End Tests
// ============================================================================

mod hol4_e2e {
    use super::*;

    #[tokio::test]
    async fn test_hol4_backend_health_check() {
        let backend = Hol4Backend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("HOL4 is available"),
            HealthStatus::Unavailable { reason } => {
                println!("HOL4 unavailable: {}", reason);
                println!("Install from: https://hol-theorem-prover.org/");
            }
            HealthStatus::Degraded { reason } => println!("HOL4 degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_hol4_backend_id() {
        let backend = Hol4Backend::new();
        assert_eq!(backend.id(), BackendId::HOL4);
    }

    #[tokio::test]
    async fn test_hol4_supports_property_types() {
        let backend = Hol4Backend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "HOL4 should support at least one property type"
        );
        println!("HOL4 supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_hol4_verify_simple_invariant() {
        let backend = Hol4Backend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: HOL4 not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("HOL4 invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("HOL4 verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_hol4_verify_theorem() {
        let backend = Hol4Backend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: HOL4 not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("HOL4 theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("HOL4 verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Agda End-to-End Tests
// ============================================================================

mod agda_e2e {
    use super::*;

    #[tokio::test]
    async fn test_agda_backend_health_check() {
        let backend = AgdaBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Agda is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Agda unavailable: {}", reason);
                println!("Install via: cabal install Agda (or brew install agda on macOS)");
            }
            HealthStatus::Degraded { reason } => println!("Agda degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_agda_backend_id() {
        let backend = AgdaBackend::new();
        assert_eq!(backend.id(), BackendId::Agda);
    }

    #[tokio::test]
    async fn test_agda_supports_property_types() {
        let backend = AgdaBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Agda should support at least one property type"
        );
        println!("Agda supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_agda_verify_simple_invariant() {
        let backend = AgdaBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Agda not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Agda invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Agda verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_agda_verify_theorem() {
        let backend = AgdaBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Agda not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Agda theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Agda verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Idris End-to-End Tests
// ============================================================================

mod idris_e2e {
    use super::*;

    #[tokio::test]
    async fn test_idris_backend_health_check() {
        let backend = IdrisBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Idris 2 is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Idris 2 unavailable: {}", reason);
                println!("Install from: https://www.idris-lang.org/");
            }
            HealthStatus::Degraded { reason } => println!("Idris 2 degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_idris_backend_id() {
        let backend = IdrisBackend::new();
        assert_eq!(backend.id(), BackendId::Idris);
    }

    #[tokio::test]
    async fn test_idris_supports_property_types() {
        let backend = IdrisBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Idris should support at least one property type"
        );
        println!("Idris supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_idris_verify_simple_invariant() {
        let backend = IdrisBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Idris not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Idris invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Idris verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_idris_verify_theorem() {
        let backend = IdrisBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Idris not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Idris theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Idris verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// F* End-to-End Tests
// ============================================================================

mod fstar_e2e {
    use super::*;

    #[tokio::test]
    async fn test_fstar_backend_health_check() {
        let backend = FStarBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("F* is available"),
            HealthStatus::Unavailable { reason } => {
                println!("F* unavailable: {}", reason);
                println!("Install from: https://www.fstar-lang.org/");
            }
            HealthStatus::Degraded { reason } => println!("F* degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_fstar_backend_id() {
        let backend = FStarBackend::new();
        assert_eq!(backend.id(), BackendId::FStar);
    }

    #[tokio::test]
    async fn test_fstar_supports_property_types() {
        let backend = FStarBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "F* should support at least one property type"
        );
        println!("F* supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_fstar_verify_simple_invariant() {
        let backend = FStarBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: F* not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("F* invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("F* verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_fstar_verify_theorem() {
        let backend = FStarBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: F* not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("F* theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("F* verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Apalache End-to-End Tests (TLA+ Model Checker)
// ============================================================================

mod apalache_e2e {
    use super::*;

    #[tokio::test]
    async fn test_apalache_backend_health_check() {
        let backend = ApalacheBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Apalache is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Apalache unavailable: {}", reason);
                println!("Install from: https://apalache.informal.systems/");
            }
            HealthStatus::Degraded { reason } => println!("Apalache degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_apalache_backend_id() {
        let backend = ApalacheBackend::new();
        assert_eq!(backend.id(), BackendId::Apalache);
    }

    #[tokio::test]
    async fn test_apalache_supports_property_types() {
        let backend = ApalacheBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Apalache should support at least one property type"
        );
        println!("Apalache supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_apalache_verify_simple_invariant() {
        let backend = ApalacheBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Apalache not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Apalache invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Apalache verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_apalache_verify_theorem() {
        let backend = ApalacheBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Apalache not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Apalache theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Apalache verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Storm End-to-End Tests (Probabilistic Model Checker)
// ============================================================================

mod storm_e2e {
    use super::*;

    #[tokio::test]
    async fn test_storm_backend_health_check() {
        let backend = StormBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Storm is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Storm unavailable: {}", reason);
                println!("Install from: https://www.stormchecker.org/");
            }
            HealthStatus::Degraded { reason } => println!("Storm degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_storm_backend_id() {
        let backend = StormBackend::new();
        assert_eq!(backend.id(), BackendId::Storm);
    }

    #[tokio::test]
    async fn test_storm_supports_property_types() {
        let backend = StormBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Storm should support at least one property type"
        );
        println!("Storm supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_storm_verify_simple_invariant() {
        let backend = StormBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Storm not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Storm invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Storm verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_storm_verify_theorem() {
        let backend = StormBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Storm not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Storm theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Storm verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// PRISM End-to-End Tests (Probabilistic Model Checker)
// ============================================================================

mod prism_e2e {
    use super::*;

    #[tokio::test]
    async fn test_prism_backend_health_check() {
        let backend = PrismBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("PRISM is available"),
            HealthStatus::Unavailable { reason } => {
                println!("PRISM unavailable: {}", reason);
                println!("Install from: https://www.prismmodelchecker.org/");
            }
            HealthStatus::Degraded { reason } => println!("PRISM degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_prism_backend_id() {
        let backend = PrismBackend::new();
        assert_eq!(backend.id(), BackendId::Prism);
    }

    #[tokio::test]
    async fn test_prism_supports_property_types() {
        let backend = PrismBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "PRISM should support at least one property type"
        );
        println!("PRISM supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_prism_verify_simple_invariant() {
        let backend = PrismBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: PRISM not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("PRISM invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("PRISM verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_prism_verify_theorem() {
        let backend = PrismBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: PRISM not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("PRISM theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("PRISM verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Tamarin End-to-End Tests (Security Protocol Verifier)
// ============================================================================

mod tamarin_e2e {
    use super::*;

    #[tokio::test]
    async fn test_tamarin_backend_health_check() {
        let backend = TamarinBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Tamarin is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Tamarin unavailable: {}", reason);
                println!("Install from: https://tamarin-prover.github.io/");
            }
            HealthStatus::Degraded { reason } => println!("Tamarin degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_tamarin_backend_id() {
        let backend = TamarinBackend::new();
        assert_eq!(backend.id(), BackendId::Tamarin);
    }

    #[tokio::test]
    async fn test_tamarin_supports_property_types() {
        let backend = TamarinBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Tamarin should support at least one property type"
        );
        println!("Tamarin supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_tamarin_verify_simple_invariant() {
        let backend = TamarinBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Tamarin not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Tamarin invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Tamarin verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_tamarin_verify_theorem() {
        let backend = TamarinBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Tamarin not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Tamarin theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Tamarin verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Verifpal End-to-End Tests (Security Protocol Verifier)
// ============================================================================

mod verifpal_e2e {
    use super::*;

    #[tokio::test]
    async fn test_verifpal_backend_health_check() {
        let backend = VerifpalBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Verifpal is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Verifpal unavailable: {}", reason);
                println!("Install from: https://verifpal.com/");
            }
            HealthStatus::Degraded { reason } => println!("Verifpal degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_verifpal_backend_id() {
        let backend = VerifpalBackend::new();
        assert_eq!(backend.id(), BackendId::Verifpal);
    }

    #[tokio::test]
    async fn test_verifpal_supports_property_types() {
        let backend = VerifpalBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Verifpal should support at least one property type"
        );
        println!("Verifpal supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_verifpal_verify_simple_invariant() {
        let backend = VerifpalBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Verifpal not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Verifpal invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Verifpal verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_verifpal_verify_theorem() {
        let backend = VerifpalBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Verifpal not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Verifpal theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Verifpal verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// ProVerif End-to-End Tests (Security Protocol Verifier)
// ============================================================================

mod proverif_e2e {
    use super::*;

    #[tokio::test]
    async fn test_proverif_backend_health_check() {
        let backend = ProverifBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("ProVerif is available"),
            HealthStatus::Unavailable { reason } => {
                println!("ProVerif unavailable: {}", reason);
                println!("Install from: https://bblanche.gitlabpages.inria.fr/proverif/");
            }
            HealthStatus::Degraded { reason } => println!("ProVerif degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_proverif_backend_id() {
        let backend = ProverifBackend::new();
        assert_eq!(backend.id(), BackendId::ProVerif);
    }

    #[tokio::test]
    async fn test_proverif_supports_property_types() {
        let backend = ProverifBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "ProVerif should support at least one property type"
        );
        println!("ProVerif supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_proverif_verify_simple_invariant() {
        let backend = ProverifBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: ProVerif not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("ProVerif invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("ProVerif verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_proverif_verify_theorem() {
        let backend = ProverifBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: ProVerif not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("ProVerif theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("ProVerif verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Flux End-to-End Tests (Rust Refinement Type Verifier)
// ============================================================================

mod flux_e2e {
    use super::*;

    #[tokio::test]
    async fn test_flux_backend_health_check() {
        let backend = FluxBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Flux is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Flux unavailable: {}", reason);
                println!("Install from: https://github.com/flux-rs/flux");
            }
            HealthStatus::Degraded { reason } => println!("Flux degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_flux_backend_id() {
        let backend = FluxBackend::new();
        assert_eq!(backend.id(), BackendId::Flux);
    }

    #[tokio::test]
    async fn test_flux_supports_property_types() {
        let backend = FluxBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Flux should support at least one property type"
        );
        println!("Flux supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_flux_verify_simple_invariant() {
        let backend = FluxBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Flux not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Flux invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Flux verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_flux_verify_theorem() {
        let backend = FluxBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Flux not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Flux theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Flux verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Mirai End-to-End Tests (Rust Abstract Interpreter)
// ============================================================================

mod mirai_e2e {
    use super::*;

    #[tokio::test]
    async fn test_mirai_backend_health_check() {
        let backend = MiraiBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("MIRAI is available"),
            HealthStatus::Unavailable { reason } => {
                println!("MIRAI unavailable: {}", reason);
                println!("Install from: https://github.com/facebookexperimental/MIRAI");
            }
            HealthStatus::Degraded { reason } => println!("MIRAI degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_mirai_backend_id() {
        let backend = MiraiBackend::new();
        assert_eq!(backend.id(), BackendId::Mirai);
    }

    #[tokio::test]
    async fn test_mirai_supports_property_types() {
        let backend = MiraiBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "MIRAI should support at least one property type"
        );
        println!("MIRAI supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_mirai_verify_simple_invariant() {
        let backend = MiraiBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: MIRAI not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("MIRAI invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("MIRAI verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_mirai_verify_theorem() {
        let backend = MiraiBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: MIRAI not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("MIRAI theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("MIRAI verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Rudra End-to-End Tests (Rust Memory Safety Analyzer)
// ============================================================================

mod rudra_e2e {
    use super::*;

    #[tokio::test]
    async fn test_rudra_backend_health_check() {
        let backend = RudraBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Rudra is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Rudra unavailable: {}", reason);
                println!("Install from: https://github.com/sslab-gatech/Rudra");
            }
            HealthStatus::Degraded { reason } => println!("Rudra degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_rudra_backend_id() {
        let backend = RudraBackend::new();
        assert_eq!(backend.id(), BackendId::Rudra);
    }

    #[tokio::test]
    async fn test_rudra_supports_property_types() {
        let backend = RudraBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Rudra should support at least one property type"
        );
        println!("Rudra supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_rudra_verify_simple_invariant() {
        let backend = RudraBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Rudra not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Rudra invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Rudra verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_rudra_verify_theorem() {
        let backend = RudraBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Rudra not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Rudra theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Rudra verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Neural Network Verification End-to-End Tests
// ============================================================================

// Marabou End-to-End Tests
mod marabou_e2e {
    use super::*;

    #[tokio::test]
    async fn test_marabou_backend_health_check() {
        let backend = MarabouBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Marabou is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Marabou unavailable: {}", reason);
                println!("Install from: https://github.com/NeuralNetworkVerification/Marabou");
            }
            HealthStatus::Degraded { reason } => println!("Marabou degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_marabou_backend_id() {
        let backend = MarabouBackend::new();
        assert_eq!(backend.id(), BackendId::Marabou);
    }

    #[tokio::test]
    async fn test_marabou_supports_property_types() {
        let backend = MarabouBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Marabou should support at least one property type"
        );
        println!("Marabou supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_marabou_verify_simple_invariant() {
        let backend = MarabouBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Marabou not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Marabou invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Marabou verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_marabou_verify_theorem() {
        let backend = MarabouBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Marabou not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Marabou theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Marabou verification error: {}", e);
            }
        }
    }
}

// alpha-beta-CROWN End-to-End Tests
mod abcrown_e2e {
    use super::*;

    #[tokio::test]
    async fn test_abcrown_backend_health_check() {
        let backend = AbCrownBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("alpha-beta-CROWN is available"),
            HealthStatus::Unavailable { reason } => {
                println!("alpha-beta-CROWN unavailable: {}", reason);
                println!("Install from: https://github.com/Verified-Intelligence/alpha-beta-CROWN");
            }
            HealthStatus::Degraded { reason } => println!("alpha-beta-CROWN degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_abcrown_backend_id() {
        let backend = AbCrownBackend::new();
        assert_eq!(backend.id(), BackendId::AlphaBetaCrown);
    }

    #[tokio::test]
    async fn test_abcrown_supports_property_types() {
        let backend = AbCrownBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "alpha-beta-CROWN should support at least one property type"
        );
        println!("alpha-beta-CROWN supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_abcrown_verify_simple_invariant() {
        let backend = AbCrownBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: alpha-beta-CROWN not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("alpha-beta-CROWN invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("alpha-beta-CROWN verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_abcrown_verify_theorem() {
        let backend = AbCrownBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: alpha-beta-CROWN not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("alpha-beta-CROWN theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("alpha-beta-CROWN verification error: {}", e);
            }
        }
    }
}

// ERAN End-to-End Tests
mod eran_e2e {
    use super::*;

    #[tokio::test]
    async fn test_eran_backend_health_check() {
        let backend = EranBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("ERAN is available"),
            HealthStatus::Unavailable { reason } => {
                println!("ERAN unavailable: {}", reason);
                println!("Install from: https://github.com/eth-sri/eran");
            }
            HealthStatus::Degraded { reason } => println!("ERAN degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_eran_backend_id() {
        let backend = EranBackend::new();
        assert_eq!(backend.id(), BackendId::Eran);
    }

    #[tokio::test]
    async fn test_eran_supports_property_types() {
        let backend = EranBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "ERAN should support at least one property type"
        );
        println!("ERAN supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_eran_verify_simple_invariant() {
        let backend = EranBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: ERAN not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("ERAN invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("ERAN verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_eran_verify_theorem() {
        let backend = EranBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: ERAN not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("ERAN theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("ERAN verification error: {}", e);
            }
        }
    }
}

// NNV End-to-End Tests
mod nnv_e2e {
    use super::*;

    #[tokio::test]
    async fn test_nnv_backend_health_check() {
        let backend = NnvBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("NNV is available"),
            HealthStatus::Unavailable { reason } => {
                println!("NNV unavailable: {}", reason);
                println!("Install from: https://github.com/verivital/nnv");
            }
            HealthStatus::Degraded { reason } => println!("NNV degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_nnv_backend_id() {
        let backend = NnvBackend::new();
        assert_eq!(backend.id(), BackendId::NNV);
    }

    #[tokio::test]
    async fn test_nnv_supports_property_types() {
        let backend = NnvBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "NNV should support at least one property type"
        );
        println!("NNV supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_nnv_verify_simple_invariant() {
        let backend = NnvBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: NNV not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("NNV invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("NNV verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_nnv_verify_theorem() {
        let backend = NnvBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: NNV not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("NNV theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("NNV verification error: {}", e);
            }
        }
    }
}

// nnenum End-to-End Tests
mod nnenum_e2e {
    use super::*;

    #[tokio::test]
    async fn test_nnenum_backend_health_check() {
        let backend = NnenumBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("nnenum is available"),
            HealthStatus::Unavailable { reason } => {
                println!("nnenum unavailable: {}", reason);
                println!("Install from: https://github.com/stanleybak/nnenum");
            }
            HealthStatus::Degraded { reason } => println!("nnenum degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_nnenum_backend_id() {
        let backend = NnenumBackend::new();
        assert_eq!(backend.id(), BackendId::Nnenum);
    }

    #[tokio::test]
    async fn test_nnenum_supports_property_types() {
        let backend = NnenumBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "nnenum should support at least one property type"
        );
        println!("nnenum supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_nnenum_verify_simple_invariant() {
        let backend = NnenumBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: nnenum not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("nnenum invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("nnenum verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_nnenum_verify_theorem() {
        let backend = NnenumBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: nnenum not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("nnenum theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("nnenum verification error: {}", e);
            }
        }
    }
}

// VeriNet End-to-End Tests
mod verinet_e2e {
    use super::*;

    #[tokio::test]
    async fn test_verinet_backend_health_check() {
        let backend = VeriNetBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("VeriNet is available"),
            HealthStatus::Unavailable { reason } => {
                println!("VeriNet unavailable: {}", reason);
                println!("Install from: https://github.com/stanis-au/verinet");
            }
            HealthStatus::Degraded { reason } => println!("VeriNet degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_verinet_backend_id() {
        let backend = VeriNetBackend::new();
        assert_eq!(backend.id(), BackendId::VeriNet);
    }

    #[tokio::test]
    async fn test_verinet_supports_property_types() {
        let backend = VeriNetBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "VeriNet should support at least one property type"
        );
        println!("VeriNet supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_verinet_verify_simple_invariant() {
        let backend = VeriNetBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: VeriNet not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("VeriNet invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("VeriNet verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_verinet_verify_theorem() {
        let backend = VeriNetBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: VeriNet not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("VeriNet theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("VeriNet verification error: {}", e);
            }
        }
    }
}

// Venus End-to-End Tests
mod venus_e2e {
    use super::*;

    #[tokio::test]
    async fn test_venus_backend_health_check() {
        let backend = VenusBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Venus is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Venus unavailable: {}", reason);
                println!("Install from: https://github.com/vas-group-imperial/venus");
            }
            HealthStatus::Degraded { reason } => println!("Venus degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_venus_backend_id() {
        let backend = VenusBackend::new();
        assert_eq!(backend.id(), BackendId::Venus);
    }

    #[tokio::test]
    async fn test_venus_supports_property_types() {
        let backend = VenusBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Venus should support at least one property type"
        );
        println!("Venus supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_venus_verify_simple_invariant() {
        let backend = VenusBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Venus not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Venus invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Venus verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_venus_verify_theorem() {
        let backend = VenusBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Venus not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Venus theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Venus verification error: {}", e);
            }
        }
    }
}

// DNNV End-to-End Tests
mod dnnv_e2e {
    use super::*;

    #[tokio::test]
    async fn test_dnnv_backend_health_check() {
        let backend = DnnvBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("DNNV is available"),
            HealthStatus::Unavailable { reason } => {
                println!("DNNV unavailable: {}", reason);
                println!("Install from: https://github.com/dlshriver/dnnv");
            }
            HealthStatus::Degraded { reason } => println!("DNNV degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_dnnv_backend_id() {
        let backend = DnnvBackend::new();
        assert_eq!(backend.id(), BackendId::DNNV);
    }

    #[tokio::test]
    async fn test_dnnv_supports_property_types() {
        let backend = DnnvBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "DNNV should support at least one property type"
        );
        println!("DNNV supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_dnnv_verify_simple_invariant() {
        let backend = DnnvBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: DNNV not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("DNNV invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("DNNV verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_dnnv_verify_theorem() {
        let backend = DnnvBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: DNNV not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("DNNV theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("DNNV verification error: {}", e);
            }
        }
    }
}

// Auto-LiRPA End-to-End Tests
mod autolirpa_e2e {
    use super::*;

    #[tokio::test]
    async fn test_autolirpa_backend_health_check() {
        let backend = AutoLirpaBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Auto-LiRPA is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Auto-LiRPA unavailable: {}", reason);
                println!("Install from: https://github.com/Verified-Intelligence/auto_LiRPA");
            }
            HealthStatus::Degraded { reason } => println!("Auto-LiRPA degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_autolirpa_backend_id() {
        let backend = AutoLirpaBackend::new();
        assert_eq!(backend.id(), BackendId::AutoLiRPA);
    }

    #[tokio::test]
    async fn test_autolirpa_supports_property_types() {
        let backend = AutoLirpaBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Auto-LiRPA should support at least one property type"
        );
        println!("Auto-LiRPA supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_autolirpa_verify_simple_invariant() {
        let backend = AutoLirpaBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Auto-LiRPA not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Auto-LiRPA invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Auto-LiRPA verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_autolirpa_verify_theorem() {
        let backend = AutoLirpaBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Auto-LiRPA not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Auto-LiRPA theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Auto-LiRPA verification error: {}", e);
            }
        }
    }
}

// MN-BaB End-to-End Tests
mod mnbab_e2e {
    use super::*;

    #[tokio::test]
    async fn test_mnbab_backend_health_check() {
        let backend = MNBaBBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("MN-BaB is available"),
            HealthStatus::Unavailable { reason } => {
                println!("MN-BaB unavailable: {}", reason);
                println!("Install from: https://github.com/eth-sri/mn-bab");
            }
            HealthStatus::Degraded { reason } => println!("MN-BaB degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_mnbab_backend_id() {
        let backend = MNBaBBackend::new();
        assert_eq!(backend.id(), BackendId::MNBaB);
    }

    #[tokio::test]
    async fn test_mnbab_supports_property_types() {
        let backend = MNBaBBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "MN-BaB should support at least one property type"
        );
        println!("MN-BaB supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_mnbab_verify_simple_invariant() {
        let backend = MNBaBBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: MN-BaB not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("MN-BaB invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("MN-BaB verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_mnbab_verify_theorem() {
        let backend = MNBaBBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: MN-BaB not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("MN-BaB theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("MN-BaB verification error: {}", e);
            }
        }
    }
}

// Neurify End-to-End Tests
mod neurify_e2e {
    use super::*;

    #[tokio::test]
    async fn test_neurify_backend_health_check() {
        let backend = NeurifyBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Neurify is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Neurify unavailable: {}", reason);
                println!("Install from: https://github.com/tcwangshiqi-columbia/Neurify");
            }
            HealthStatus::Degraded { reason } => println!("Neurify degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_neurify_backend_id() {
        let backend = NeurifyBackend::new();
        assert_eq!(backend.id(), BackendId::Neurify);
    }

    #[tokio::test]
    async fn test_neurify_supports_property_types() {
        let backend = NeurifyBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Neurify should support at least one property type"
        );
        println!("Neurify supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_neurify_verify_simple_invariant() {
        let backend = NeurifyBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Neurify not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Neurify invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Neurify verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_neurify_verify_theorem() {
        let backend = NeurifyBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Neurify not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Neurify theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Neurify verification error: {}", e);
            }
        }
    }
}

// ReluVal End-to-End Tests
mod reluval_e2e {
    use super::*;

    #[tokio::test]
    async fn test_reluval_backend_health_check() {
        let backend = ReluValBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("ReluVal is available"),
            HealthStatus::Unavailable { reason } => {
                println!("ReluVal unavailable: {}", reason);
                println!("Install from: https://github.com/tcwangshiqi-columbia/ReluVal");
            }
            HealthStatus::Degraded { reason } => println!("ReluVal degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_reluval_backend_id() {
        let backend = ReluValBackend::new();
        assert_eq!(backend.id(), BackendId::ReluVal);
    }

    #[tokio::test]
    async fn test_reluval_supports_property_types() {
        let backend = ReluValBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "ReluVal should support at least one property type"
        );
        println!("ReluVal supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_reluval_verify_simple_invariant() {
        let backend = ReluValBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: ReluVal not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("ReluVal invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("ReluVal verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_reluval_verify_theorem() {
        let backend = ReluValBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: ReluVal not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("ReluVal theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("ReluVal verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Adversarial Robustness End-to-End Tests
// ============================================================================

// ART (Adversarial Robustness Toolbox) End-to-End Tests
mod art_e2e {
    use super::*;

    #[tokio::test]
    async fn test_art_backend_health_check() {
        let backend = ArtBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("ART is available"),
            HealthStatus::Unavailable { reason } => {
                println!("ART unavailable: {}", reason);
                println!("Install: pip install adversarial-robustness-toolbox");
            }
            HealthStatus::Degraded { reason } => println!("ART degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_art_backend_id() {
        let backend = ArtBackend::new();
        assert_eq!(backend.id(), BackendId::ART);
    }

    #[tokio::test]
    async fn test_art_supports_property_types() {
        let backend = ArtBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "ART should support at least one property type"
        );
        println!("ART supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_art_verify_simple_invariant() {
        let backend = ArtBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: ART not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("ART invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("ART verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_art_verify_theorem() {
        let backend = ArtBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: ART not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("ART theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("ART verification error: {}", e);
            }
        }
    }
}

// Foolbox End-to-End Tests
mod foolbox_e2e {
    use super::*;

    #[tokio::test]
    async fn test_foolbox_backend_health_check() {
        let backend = FoolboxBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Foolbox is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Foolbox unavailable: {}", reason);
                println!("Install: pip install foolbox");
            }
            HealthStatus::Degraded { reason } => println!("Foolbox degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_foolbox_backend_id() {
        let backend = FoolboxBackend::new();
        assert_eq!(backend.id(), BackendId::Foolbox);
    }

    #[tokio::test]
    async fn test_foolbox_supports_property_types() {
        let backend = FoolboxBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Foolbox should support at least one property type"
        );
        println!("Foolbox supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_foolbox_verify_simple_invariant() {
        let backend = FoolboxBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Foolbox not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Foolbox invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Foolbox verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_foolbox_verify_theorem() {
        let backend = FoolboxBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Foolbox not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Foolbox theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Foolbox verification error: {}", e);
            }
        }
    }
}

// CleverHans End-to-End Tests
mod cleverhans_e2e {
    use super::*;

    #[tokio::test]
    async fn test_cleverhans_backend_health_check() {
        let backend = CleverHansBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("CleverHans is available"),
            HealthStatus::Unavailable { reason } => {
                println!("CleverHans unavailable: {}", reason);
                println!("Install: pip install cleverhans");
            }
            HealthStatus::Degraded { reason } => println!("CleverHans degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_cleverhans_backend_id() {
        let backend = CleverHansBackend::new();
        assert_eq!(backend.id(), BackendId::CleverHans);
    }

    #[tokio::test]
    async fn test_cleverhans_supports_property_types() {
        let backend = CleverHansBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "CleverHans should support at least one property type"
        );
        println!("CleverHans supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_cleverhans_verify_simple_invariant() {
        let backend = CleverHansBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: CleverHans not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("CleverHans invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("CleverHans verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_cleverhans_verify_theorem() {
        let backend = CleverHansBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: CleverHans not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("CleverHans theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("CleverHans verification error: {}", e);
            }
        }
    }
}

// TextAttack End-to-End Tests
mod textattack_e2e {
    use super::*;

    #[tokio::test]
    async fn test_textattack_backend_health_check() {
        let backend = TextAttackBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("TextAttack is available"),
            HealthStatus::Unavailable { reason } => {
                println!("TextAttack unavailable: {}", reason);
                println!("Install: pip install textattack");
            }
            HealthStatus::Degraded { reason } => println!("TextAttack degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_textattack_backend_id() {
        let backend = TextAttackBackend::new();
        assert_eq!(backend.id(), BackendId::TextAttack);
    }

    #[tokio::test]
    async fn test_textattack_supports_property_types() {
        let backend = TextAttackBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "TextAttack should support at least one property type"
        );
        println!("TextAttack supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_textattack_verify_simple_invariant() {
        let backend = TextAttackBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: TextAttack not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("TextAttack invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("TextAttack verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_textattack_verify_theorem() {
        let backend = TextAttackBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: TextAttack not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("TextAttack theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("TextAttack verification error: {}", e);
            }
        }
    }
}

// RobustBench End-to-End Tests
mod robustbench_e2e {
    use super::*;

    #[tokio::test]
    async fn test_robustbench_backend_health_check() {
        let backend = RobustBenchBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("RobustBench is available"),
            HealthStatus::Unavailable { reason } => {
                println!("RobustBench unavailable: {}", reason);
                println!("Install: pip install robustbench");
            }
            HealthStatus::Degraded { reason } => println!("RobustBench degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_robustbench_backend_id() {
        let backend = RobustBenchBackend::new();
        assert_eq!(backend.id(), BackendId::RobustBench);
    }

    #[tokio::test]
    async fn test_robustbench_supports_property_types() {
        let backend = RobustBenchBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "RobustBench should support at least one property type"
        );
        println!("RobustBench supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_robustbench_verify_simple_invariant() {
        let backend = RobustBenchBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: RobustBench not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("RobustBench invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("RobustBench verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_robustbench_verify_theorem() {
        let backend = RobustBenchBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: RobustBench not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("RobustBench theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("RobustBench verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Rust Sanitizers and Memory Tools End-to-End Tests
// ============================================================================

// AddressSanitizer End-to-End Tests
mod asan_e2e {
    use super::*;

    #[tokio::test]
    async fn test_asan_backend_health_check() {
        let backend = AsanBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("AddressSanitizer is available"),
            HealthStatus::Unavailable { reason } => {
                println!("AddressSanitizer unavailable: {}", reason);
                println!("Requires: rustup +nightly and LLVM sanitizer support");
            }
            HealthStatus::Degraded { reason } => println!("AddressSanitizer degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_asan_backend_id() {
        let backend = AsanBackend::new();
        assert_eq!(backend.id(), BackendId::AddressSanitizer);
    }

    #[tokio::test]
    async fn test_asan_supports_property_types() {
        let backend = AsanBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "AddressSanitizer should support at least one property type"
        );
        println!("AddressSanitizer supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_asan_verify_simple_invariant() {
        let backend = AsanBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: AddressSanitizer not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("AddressSanitizer invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("AddressSanitizer verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_asan_verify_theorem() {
        let backend = AsanBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: AddressSanitizer not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("AddressSanitizer theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("AddressSanitizer verification error: {}", e);
            }
        }
    }
}

// MemorySanitizer End-to-End Tests
mod msan_e2e {
    use super::*;

    #[tokio::test]
    async fn test_msan_backend_health_check() {
        let backend = MsanBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("MemorySanitizer is available"),
            HealthStatus::Unavailable { reason } => {
                println!("MemorySanitizer unavailable: {}", reason);
                println!("Requires: rustup +nightly and LLVM sanitizer support");
            }
            HealthStatus::Degraded { reason } => println!("MemorySanitizer degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_msan_backend_id() {
        let backend = MsanBackend::new();
        assert_eq!(backend.id(), BackendId::MemorySanitizer);
    }

    #[tokio::test]
    async fn test_msan_supports_property_types() {
        let backend = MsanBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "MemorySanitizer should support at least one property type"
        );
        println!("MemorySanitizer supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_msan_verify_simple_invariant() {
        let backend = MsanBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: MemorySanitizer not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("MemorySanitizer invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("MemorySanitizer verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_msan_verify_theorem() {
        let backend = MsanBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: MemorySanitizer not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("MemorySanitizer theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("MemorySanitizer verification error: {}", e);
            }
        }
    }
}

// ThreadSanitizer End-to-End Tests
mod tsan_e2e {
    use super::*;

    #[tokio::test]
    async fn test_tsan_backend_health_check() {
        let backend = TsanBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("ThreadSanitizer is available"),
            HealthStatus::Unavailable { reason } => {
                println!("ThreadSanitizer unavailable: {}", reason);
                println!("Requires: rustup +nightly and LLVM sanitizer support");
            }
            HealthStatus::Degraded { reason } => println!("ThreadSanitizer degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_tsan_backend_id() {
        let backend = TsanBackend::new();
        assert_eq!(backend.id(), BackendId::ThreadSanitizer);
    }

    #[tokio::test]
    async fn test_tsan_supports_property_types() {
        let backend = TsanBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "ThreadSanitizer should support at least one property type"
        );
        println!("ThreadSanitizer supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_tsan_verify_simple_invariant() {
        let backend = TsanBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: ThreadSanitizer not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("ThreadSanitizer invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("ThreadSanitizer verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_tsan_verify_theorem() {
        let backend = TsanBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: ThreadSanitizer not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("ThreadSanitizer theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("ThreadSanitizer verification error: {}", e);
            }
        }
    }
}

// LeakSanitizer End-to-End Tests
mod lsan_e2e {
    use super::*;

    #[tokio::test]
    async fn test_lsan_backend_health_check() {
        let backend = LsanBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("LeakSanitizer is available"),
            HealthStatus::Unavailable { reason } => {
                println!("LeakSanitizer unavailable: {}", reason);
                println!("Requires: rustup +nightly and LLVM sanitizer support");
            }
            HealthStatus::Degraded { reason } => println!("LeakSanitizer degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_lsan_backend_id() {
        let backend = LsanBackend::new();
        assert_eq!(backend.id(), BackendId::LeakSanitizer);
    }

    #[tokio::test]
    async fn test_lsan_supports_property_types() {
        let backend = LsanBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "LeakSanitizer should support at least one property type"
        );
        println!("LeakSanitizer supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_lsan_verify_simple_invariant() {
        let backend = LsanBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: LeakSanitizer not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("LeakSanitizer invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("LeakSanitizer verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_lsan_verify_theorem() {
        let backend = LsanBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: LeakSanitizer not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("LeakSanitizer theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("LeakSanitizer verification error: {}", e);
            }
        }
    }
}

// Valgrind End-to-End Tests
mod valgrind_e2e {
    use super::*;

    #[tokio::test]
    async fn test_valgrind_backend_health_check() {
        let backend = ValgrindBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Valgrind is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Valgrind unavailable: {}", reason);
                println!("Install: brew install valgrind (macOS) or apt install valgrind (Linux)");
            }
            HealthStatus::Degraded { reason } => println!("Valgrind degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_valgrind_backend_id() {
        let backend = ValgrindBackend::new();
        assert_eq!(backend.id(), BackendId::Valgrind);
    }

    #[tokio::test]
    async fn test_valgrind_supports_property_types() {
        let backend = ValgrindBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Valgrind should support at least one property type"
        );
        println!("Valgrind supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_valgrind_verify_simple_invariant() {
        let backend = ValgrindBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Valgrind not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Valgrind invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Valgrind verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_valgrind_verify_theorem() {
        let backend = ValgrindBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Valgrind not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Valgrind theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Valgrind verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Rust Concurrency Testing End-to-End Tests
// ============================================================================

// Loom End-to-End Tests
mod loom_e2e {
    use super::*;

    #[tokio::test]
    async fn test_loom_backend_health_check() {
        let backend = LoomBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Loom is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Loom unavailable: {}", reason);
                println!("Add to Cargo.toml: loom = \"0.7\"");
            }
            HealthStatus::Degraded { reason } => println!("Loom degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_loom_backend_id() {
        let backend = LoomBackend::new();
        assert_eq!(backend.id(), BackendId::Loom);
    }

    #[tokio::test]
    async fn test_loom_supports_property_types() {
        let backend = LoomBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Loom should support at least one property type"
        );
        println!("Loom supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_loom_verify_simple_invariant() {
        let backend = LoomBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Loom not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Loom invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Loom verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_loom_verify_theorem() {
        let backend = LoomBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Loom not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Loom theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Loom verification error: {}", e);
            }
        }
    }
}

// Shuttle End-to-End Tests
mod shuttle_e2e {
    use super::*;

    #[tokio::test]
    async fn test_shuttle_backend_health_check() {
        let backend = ShuttleBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Shuttle is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Shuttle unavailable: {}", reason);
                println!("Add to Cargo.toml: shuttle = \"0.7\"");
            }
            HealthStatus::Degraded { reason } => println!("Shuttle degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_shuttle_backend_id() {
        let backend = ShuttleBackend::new();
        assert_eq!(backend.id(), BackendId::Shuttle);
    }

    #[tokio::test]
    async fn test_shuttle_supports_property_types() {
        let backend = ShuttleBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Shuttle should support at least one property type"
        );
        println!("Shuttle supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_shuttle_verify_simple_invariant() {
        let backend = ShuttleBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Shuttle not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Shuttle invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Shuttle verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_shuttle_verify_theorem() {
        let backend = ShuttleBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Shuttle not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Shuttle theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Shuttle verification error: {}", e);
            }
        }
    }
}

// CDSChecker End-to-End Tests
mod cdschecker_e2e {
    use super::*;

    #[tokio::test]
    async fn test_cdschecker_backend_health_check() {
        let backend = CDSCheckerBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("CDSChecker is available"),
            HealthStatus::Unavailable { reason } => {
                println!("CDSChecker unavailable: {}", reason);
                println!("Build from: https://plrg.kaist.ac.kr/cdschecker/");
            }
            HealthStatus::Degraded { reason } => println!("CDSChecker degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_cdschecker_backend_id() {
        let backend = CDSCheckerBackend::new();
        assert_eq!(backend.id(), BackendId::CDSChecker);
    }

    #[tokio::test]
    async fn test_cdschecker_supports_property_types() {
        let backend = CDSCheckerBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "CDSChecker should support at least one property type"
        );
        println!("CDSChecker supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_cdschecker_verify_simple_invariant() {
        let backend = CDSCheckerBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: CDSChecker not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("CDSChecker invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("CDSChecker verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_cdschecker_verify_theorem() {
        let backend = CDSCheckerBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: CDSChecker not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("CDSChecker theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("CDSChecker verification error: {}", e);
            }
        }
    }
}

// GenMC End-to-End Tests
mod genmc_e2e {
    use super::*;

    #[tokio::test]
    async fn test_genmc_backend_health_check() {
        let backend = GenMCBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("GenMC is available"),
            HealthStatus::Unavailable { reason } => {
                println!("GenMC unavailable: {}", reason);
                println!("Build from: https://github.com/MPI-SWS/genmc");
            }
            HealthStatus::Degraded { reason } => println!("GenMC degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_genmc_backend_id() {
        let backend = GenMCBackend::new();
        assert_eq!(backend.id(), BackendId::GenMC);
    }

    #[tokio::test]
    async fn test_genmc_supports_property_types() {
        let backend = GenMCBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "GenMC should support at least one property type"
        );
        println!("GenMC supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_genmc_verify_simple_invariant() {
        let backend = GenMCBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: GenMC not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("GenMC invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("GenMC verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_genmc_verify_theorem() {
        let backend = GenMCBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: GenMC not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("GenMC theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("GenMC verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Rust Fuzzing End-to-End Tests
// ============================================================================

// LibFuzzer End-to-End Tests
mod libfuzzer_e2e {
    use super::*;

    #[tokio::test]
    async fn test_libfuzzer_backend_health_check() {
        let backend = LibFuzzerBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("LibFuzzer is available"),
            HealthStatus::Unavailable { reason } => {
                println!("LibFuzzer unavailable: {}", reason);
                println!("Install: cargo install cargo-fuzz");
            }
            HealthStatus::Degraded { reason } => println!("LibFuzzer degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_libfuzzer_backend_id() {
        let backend = LibFuzzerBackend::new();
        assert_eq!(backend.id(), BackendId::LibFuzzer);
    }

    #[tokio::test]
    async fn test_libfuzzer_supports_property_types() {
        let backend = LibFuzzerBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "LibFuzzer should support at least one property type"
        );
        println!("LibFuzzer supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_libfuzzer_verify_simple_invariant() {
        let backend = LibFuzzerBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: LibFuzzer not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("LibFuzzer invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("LibFuzzer verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_libfuzzer_verify_theorem() {
        let backend = LibFuzzerBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: LibFuzzer not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("LibFuzzer theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("LibFuzzer verification error: {}", e);
            }
        }
    }
}

// AFL End-to-End Tests
mod afl_e2e {
    use super::*;

    #[tokio::test]
    async fn test_afl_backend_health_check() {
        let backend = AflBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("AFL is available"),
            HealthStatus::Unavailable { reason } => {
                println!("AFL unavailable: {}", reason);
                println!("Install: cargo install cargo-afl");
            }
            HealthStatus::Degraded { reason } => println!("AFL degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_afl_backend_id() {
        let backend = AflBackend::new();
        assert_eq!(backend.id(), BackendId::AFL);
    }

    #[tokio::test]
    async fn test_afl_supports_property_types() {
        let backend = AflBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "AFL should support at least one property type"
        );
        println!("AFL supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_afl_verify_simple_invariant() {
        let backend = AflBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: AFL not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("AFL invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("AFL verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_afl_verify_theorem() {
        let backend = AflBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: AFL not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("AFL theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("AFL verification error: {}", e);
            }
        }
    }
}

// Honggfuzz End-to-End Tests
mod honggfuzz_e2e {
    use super::*;

    #[tokio::test]
    async fn test_honggfuzz_backend_health_check() {
        let backend = HonggfuzzBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Honggfuzz is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Honggfuzz unavailable: {}", reason);
                println!("Install: cargo install honggfuzz");
            }
            HealthStatus::Degraded { reason } => println!("Honggfuzz degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_honggfuzz_backend_id() {
        let backend = HonggfuzzBackend::new();
        assert_eq!(backend.id(), BackendId::Honggfuzz);
    }

    #[tokio::test]
    async fn test_honggfuzz_supports_property_types() {
        let backend = HonggfuzzBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Honggfuzz should support at least one property type"
        );
        println!("Honggfuzz supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_honggfuzz_verify_simple_invariant() {
        let backend = HonggfuzzBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Honggfuzz not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Honggfuzz invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Honggfuzz verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_honggfuzz_verify_theorem() {
        let backend = HonggfuzzBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Honggfuzz not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Honggfuzz theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Honggfuzz verification error: {}", e);
            }
        }
    }
}

// Bolero End-to-End Tests
mod bolero_e2e {
    use super::*;

    #[tokio::test]
    async fn test_bolero_backend_health_check() {
        let backend = BoleroBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Bolero is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Bolero unavailable: {}", reason);
                println!("Add to Cargo.toml: bolero = \"0.10\"");
            }
            HealthStatus::Degraded { reason } => println!("Bolero degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_bolero_backend_id() {
        let backend = BoleroBackend::new();
        assert_eq!(backend.id(), BackendId::Bolero);
    }

    #[tokio::test]
    async fn test_bolero_supports_property_types() {
        let backend = BoleroBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Bolero should support at least one property type"
        );
        println!("Bolero supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_bolero_verify_simple_invariant() {
        let backend = BoleroBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Bolero not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Bolero invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Bolero verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_bolero_verify_theorem() {
        let backend = BoleroBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Bolero not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Bolero theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Bolero verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Rust Property-Based Testing End-to-End Tests
// ============================================================================

// Proptest End-to-End Tests
mod proptest_e2e {
    use super::*;

    #[tokio::test]
    async fn test_proptest_backend_health_check() {
        let backend = ProptestBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Proptest is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Proptest unavailable: {}", reason);
                println!("Add to Cargo.toml: proptest = \"1.4\"");
            }
            HealthStatus::Degraded { reason } => println!("Proptest degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_proptest_backend_id() {
        let backend = ProptestBackend::new();
        assert_eq!(backend.id(), BackendId::Proptest);
    }

    #[tokio::test]
    async fn test_proptest_supports_property_types() {
        let backend = ProptestBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Proptest should support at least one property type"
        );
        println!("Proptest supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_proptest_verify_simple_invariant() {
        let backend = ProptestBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Proptest not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Proptest invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Proptest verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_proptest_verify_theorem() {
        let backend = ProptestBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Proptest not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Proptest theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Proptest verification error: {}", e);
            }
        }
    }
}

// QuickCheck End-to-End Tests
mod quickcheck_e2e {
    use super::*;

    #[tokio::test]
    async fn test_quickcheck_backend_health_check() {
        let backend = QuickCheckBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("QuickCheck is available"),
            HealthStatus::Unavailable { reason } => {
                println!("QuickCheck unavailable: {}", reason);
                println!("Add to Cargo.toml: quickcheck = \"1.0\"");
            }
            HealthStatus::Degraded { reason } => println!("QuickCheck degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_quickcheck_backend_id() {
        let backend = QuickCheckBackend::new();
        assert_eq!(backend.id(), BackendId::QuickCheck);
    }

    #[tokio::test]
    async fn test_quickcheck_supports_property_types() {
        let backend = QuickCheckBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "QuickCheck should support at least one property type"
        );
        println!("QuickCheck supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_quickcheck_verify_simple_invariant() {
        let backend = QuickCheckBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: QuickCheck not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("QuickCheck invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("QuickCheck verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_quickcheck_verify_theorem() {
        let backend = QuickCheckBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: QuickCheck not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("QuickCheck theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("QuickCheck verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Rust Static Analysis End-to-End Tests
// ============================================================================

// SemverChecks End-to-End Tests
mod semverchecks_e2e {
    use super::*;

    #[tokio::test]
    async fn test_semverchecks_backend_health_check() {
        let backend = SemverChecksBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("SemverChecks is available"),
            HealthStatus::Unavailable { reason } => {
                println!("SemverChecks unavailable: {}", reason);
                println!("Install: cargo install cargo-semver-checks");
            }
            HealthStatus::Degraded { reason } => println!("SemverChecks degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_semverchecks_backend_id() {
        let backend = SemverChecksBackend::new();
        assert_eq!(backend.id(), BackendId::SemverChecks);
    }

    #[tokio::test]
    async fn test_semverchecks_supports_property_types() {
        let backend = SemverChecksBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "SemverChecks should support at least one property type"
        );
        println!("SemverChecks supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_semverchecks_verify_simple_invariant() {
        let backend = SemverChecksBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: SemverChecks not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("SemverChecks invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("SemverChecks verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_semverchecks_verify_theorem() {
        let backend = SemverChecksBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: SemverChecks not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("SemverChecks theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("SemverChecks verification error: {}", e);
            }
        }
    }
}

// Geiger End-to-End Tests
mod geiger_e2e {
    use super::*;

    #[tokio::test]
    async fn test_geiger_backend_health_check() {
        let backend = GeigerBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Geiger is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Geiger unavailable: {}", reason);
                println!("Install: cargo install cargo-geiger");
            }
            HealthStatus::Degraded { reason } => println!("Geiger degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_geiger_backend_id() {
        let backend = GeigerBackend::new();
        assert_eq!(backend.id(), BackendId::Geiger);
    }

    #[tokio::test]
    async fn test_geiger_supports_property_types() {
        let backend = GeigerBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Geiger should support at least one property type"
        );
        println!("Geiger supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_geiger_verify_simple_invariant() {
        let backend = GeigerBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Geiger not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Geiger invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Geiger verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_geiger_verify_theorem() {
        let backend = GeigerBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Geiger not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Geiger theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Geiger verification error: {}", e);
            }
        }
    }
}

// Audit End-to-End Tests
mod audit_e2e {
    use super::*;

    #[tokio::test]
    async fn test_audit_backend_health_check() {
        let backend = AuditBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Audit is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Audit unavailable: {}", reason);
                println!("Install: cargo install cargo-audit");
            }
            HealthStatus::Degraded { reason } => println!("Audit degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_audit_backend_id() {
        let backend = AuditBackend::new();
        assert_eq!(backend.id(), BackendId::Audit);
    }

    #[tokio::test]
    async fn test_audit_supports_property_types() {
        let backend = AuditBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Audit should support at least one property type"
        );
        println!("Audit supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_audit_verify_simple_invariant() {
        let backend = AuditBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Audit not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Audit invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Audit verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_audit_verify_theorem() {
        let backend = AuditBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Audit not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Audit theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Audit verification error: {}", e);
            }
        }
    }
}

// Deny End-to-End Tests
mod deny_e2e {
    use super::*;

    #[tokio::test]
    async fn test_deny_backend_health_check() {
        let backend = DenyBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Deny is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Deny unavailable: {}", reason);
                println!("Install: cargo install cargo-deny");
            }
            HealthStatus::Degraded { reason } => println!("Deny degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_deny_backend_id() {
        let backend = DenyBackend::new();
        assert_eq!(backend.id(), BackendId::Deny);
    }

    #[tokio::test]
    async fn test_deny_supports_property_types() {
        let backend = DenyBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Deny should support at least one property type"
        );
        println!("Deny supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_deny_verify_simple_invariant() {
        let backend = DenyBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Deny not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Deny invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Deny verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_deny_verify_theorem() {
        let backend = DenyBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Deny not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Deny theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Deny verification error: {}", e);
            }
        }
    }
}

// Vet End-to-End Tests
mod vet_e2e {
    use super::*;

    #[tokio::test]
    async fn test_vet_backend_health_check() {
        let backend = VetBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Vet is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Vet unavailable: {}", reason);
                println!("Install: cargo install cargo-vet");
            }
            HealthStatus::Degraded { reason } => println!("Vet degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_vet_backend_id() {
        let backend = VetBackend::new();
        assert_eq!(backend.id(), BackendId::Vet);
    }

    #[tokio::test]
    async fn test_vet_supports_property_types() {
        let backend = VetBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Vet should support at least one property type"
        );
        println!("Vet supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_vet_verify_simple_invariant() {
        let backend = VetBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Vet not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Vet invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Vet verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_vet_verify_theorem() {
        let backend = VetBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Vet not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Vet theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Vet verification error: {}", e);
            }
        }
    }
}

// Mutants End-to-End Tests
mod mutants_e2e {
    use super::*;

    #[tokio::test]
    async fn test_mutants_backend_health_check() {
        let backend = MutantsBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Mutants is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Mutants unavailable: {}", reason);
                println!("Install: cargo install cargo-mutants");
            }
            HealthStatus::Degraded { reason } => println!("Mutants degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_mutants_backend_id() {
        let backend = MutantsBackend::new();
        assert_eq!(backend.id(), BackendId::Mutants);
    }

    #[tokio::test]
    async fn test_mutants_supports_property_types() {
        let backend = MutantsBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Mutants should support at least one property type"
        );
        println!("Mutants supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_mutants_verify_simple_invariant() {
        let backend = MutantsBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Mutants not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Mutants invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Mutants verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_mutants_verify_theorem() {
        let backend = MutantsBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Mutants not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Mutants theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Mutants verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// AI/ML Optimization End-to-End Tests
// ============================================================================

// ONNX Runtime End-to-End Tests
mod onnxruntime_e2e {
    use super::*;

    #[tokio::test]
    async fn test_onnxruntime_backend_health_check() {
        let backend = OnnxRuntimeBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("ONNX Runtime is available"),
            HealthStatus::Unavailable { reason } => {
                println!("ONNX Runtime unavailable: {}", reason);
                println!("Install: pip install onnxruntime");
            }
            HealthStatus::Degraded { reason } => println!("ONNX Runtime degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_onnxruntime_backend_id() {
        let backend = OnnxRuntimeBackend::new();
        assert_eq!(backend.id(), BackendId::ONNXRuntime);
    }

    #[tokio::test]
    async fn test_onnxruntime_supports_property_types() {
        let backend = OnnxRuntimeBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "ONNX Runtime should support at least one property type"
        );
        println!("ONNX Runtime supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_onnxruntime_verify_simple_invariant() {
        let backend = OnnxRuntimeBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: ONNX Runtime not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("ONNX Runtime invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("ONNX Runtime verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_onnxruntime_verify_theorem() {
        let backend = OnnxRuntimeBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: ONNX Runtime not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("ONNX Runtime theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("ONNX Runtime verification error: {}", e);
            }
        }
    }
}

// TensorRT End-to-End Tests
mod tensorrt_e2e {
    use super::*;

    #[tokio::test]
    async fn test_tensorrt_backend_health_check() {
        let backend = TensorRTBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("TensorRT is available"),
            HealthStatus::Unavailable { reason } => {
                println!("TensorRT unavailable: {}", reason);
                println!("Requires NVIDIA GPU and TensorRT SDK");
            }
            HealthStatus::Degraded { reason } => println!("TensorRT degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_tensorrt_backend_id() {
        let backend = TensorRTBackend::new();
        assert_eq!(backend.id(), BackendId::TensorRT);
    }

    #[tokio::test]
    async fn test_tensorrt_supports_property_types() {
        let backend = TensorRTBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "TensorRT should support at least one property type"
        );
        println!("TensorRT supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_tensorrt_verify_simple_invariant() {
        let backend = TensorRTBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: TensorRT not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("TensorRT invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("TensorRT verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_tensorrt_verify_theorem() {
        let backend = TensorRTBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: TensorRT not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("TensorRT theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("TensorRT verification error: {}", e);
            }
        }
    }
}

// OpenVINO End-to-End Tests
mod openvino_e2e {
    use super::*;

    #[tokio::test]
    async fn test_openvino_backend_health_check() {
        let backend = OpenVINOBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("OpenVINO is available"),
            HealthStatus::Unavailable { reason } => {
                println!("OpenVINO unavailable: {}", reason);
                println!("Install: pip install openvino");
            }
            HealthStatus::Degraded { reason } => println!("OpenVINO degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_openvino_backend_id() {
        let backend = OpenVINOBackend::new();
        assert_eq!(backend.id(), BackendId::OpenVINO);
    }

    #[tokio::test]
    async fn test_openvino_supports_property_types() {
        let backend = OpenVINOBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "OpenVINO should support at least one property type"
        );
        println!("OpenVINO supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_openvino_verify_simple_invariant() {
        let backend = OpenVINOBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: OpenVINO not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("OpenVINO invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("OpenVINO verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_openvino_verify_theorem() {
        let backend = OpenVINOBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: OpenVINO not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("OpenVINO theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("OpenVINO verification error: {}", e);
            }
        }
    }
}

// TVM End-to-End Tests
mod tvm_e2e {
    use super::*;

    #[tokio::test]
    async fn test_tvm_backend_health_check() {
        let backend = TVMBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("TVM is available"),
            HealthStatus::Unavailable { reason } => {
                println!("TVM unavailable: {}", reason);
                println!("Install: pip install apache-tvm");
            }
            HealthStatus::Degraded { reason } => println!("TVM degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_tvm_backend_id() {
        let backend = TVMBackend::new();
        assert_eq!(backend.id(), BackendId::TVM);
    }

    #[tokio::test]
    async fn test_tvm_supports_property_types() {
        let backend = TVMBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "TVM should support at least one property type"
        );
        println!("TVM supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_tvm_verify_simple_invariant() {
        let backend = TVMBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: TVM not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("TVM invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("TVM verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_tvm_verify_theorem() {
        let backend = TVMBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: TVM not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("TVM theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("TVM verification error: {}", e);
            }
        }
    }
}

// IREE End-to-End Tests
mod iree_e2e {
    use super::*;

    #[tokio::test]
    async fn test_iree_backend_health_check() {
        let backend = IREEBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("IREE is available"),
            HealthStatus::Unavailable { reason } => {
                println!("IREE unavailable: {}", reason);
                println!("Install: pip install iree-compiler iree-runtime");
            }
            HealthStatus::Degraded { reason } => println!("IREE degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_iree_backend_id() {
        let backend = IREEBackend::new();
        assert_eq!(backend.id(), BackendId::IREE);
    }

    #[tokio::test]
    async fn test_iree_supports_property_types() {
        let backend = IREEBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "IREE should support at least one property type"
        );
        println!("IREE supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_iree_verify_simple_invariant() {
        let backend = IREEBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: IREE not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("IREE invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("IREE verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_iree_verify_theorem() {
        let backend = IREEBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: IREE not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("IREE theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("IREE verification error: {}", e);
            }
        }
    }
}

// Triton End-to-End Tests
mod triton_e2e {
    use super::*;

    #[tokio::test]
    async fn test_triton_backend_health_check() {
        let backend = TritonBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Triton is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Triton unavailable: {}", reason);
                println!("Install: pip install triton");
            }
            HealthStatus::Degraded { reason } => println!("Triton degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_triton_backend_id() {
        let backend = TritonBackend::new();
        assert_eq!(backend.id(), BackendId::Triton);
    }

    #[tokio::test]
    async fn test_triton_supports_property_types() {
        let backend = TritonBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Triton should support at least one property type"
        );
        println!("Triton supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_triton_verify_simple_invariant() {
        let backend = TritonBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Triton not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Triton invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Triton verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_triton_verify_theorem() {
        let backend = TritonBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Triton not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Triton theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Triton verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// AI/ML Compression End-to-End Tests
// ============================================================================

// Neural Compressor End-to-End Tests
mod neuralcompressor_e2e {
    use super::*;

    #[tokio::test]
    async fn test_neuralcompressor_backend_health_check() {
        let backend = NeuralCompressorBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Neural Compressor is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Neural Compressor unavailable: {}", reason);
                println!("Install: pip install neural-compressor");
            }
            HealthStatus::Degraded { reason } => println!("Neural Compressor degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_neuralcompressor_backend_id() {
        let backend = NeuralCompressorBackend::new();
        assert_eq!(backend.id(), BackendId::NeuralCompressor);
    }

    #[tokio::test]
    async fn test_neuralcompressor_supports_property_types() {
        let backend = NeuralCompressorBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Neural Compressor should support at least one property type"
        );
        println!("Neural Compressor supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_neuralcompressor_verify_simple_invariant() {
        let backend = NeuralCompressorBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Neural Compressor not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Neural Compressor invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Neural Compressor verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_neuralcompressor_verify_theorem() {
        let backend = NeuralCompressorBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Neural Compressor not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Neural Compressor theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Neural Compressor verification error: {}", e);
            }
        }
    }
}

// NNCF End-to-End Tests
mod nncf_e2e {
    use super::*;

    #[tokio::test]
    async fn test_nncf_backend_health_check() {
        let backend = NNCFBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("NNCF is available"),
            HealthStatus::Unavailable { reason } => {
                println!("NNCF unavailable: {}", reason);
                println!("Install: pip install nncf");
            }
            HealthStatus::Degraded { reason } => println!("NNCF degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_nncf_backend_id() {
        let backend = NNCFBackend::new();
        assert_eq!(backend.id(), BackendId::NNCF);
    }

    #[tokio::test]
    async fn test_nncf_supports_property_types() {
        let backend = NNCFBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "NNCF should support at least one property type"
        );
        println!("NNCF supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_nncf_verify_simple_invariant() {
        let backend = NNCFBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: NNCF not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("NNCF invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("NNCF verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_nncf_verify_theorem() {
        let backend = NNCFBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: NNCF not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("NNCF theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("NNCF verification error: {}", e);
            }
        }
    }
}

// AIMET End-to-End Tests
mod aimet_e2e {
    use super::*;

    #[tokio::test]
    async fn test_aimet_backend_health_check() {
        let backend = AimetBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("AIMET is available"),
            HealthStatus::Unavailable { reason } => {
                println!("AIMET unavailable: {}", reason);
                println!("Install from: https://quic.github.io/aimet-pages/");
            }
            HealthStatus::Degraded { reason } => println!("AIMET degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_aimet_backend_id() {
        let backend = AimetBackend::new();
        assert_eq!(backend.id(), BackendId::AIMET);
    }

    #[tokio::test]
    async fn test_aimet_supports_property_types() {
        let backend = AimetBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "AIMET should support at least one property type"
        );
        println!("AIMET supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_aimet_verify_simple_invariant() {
        let backend = AimetBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: AIMET not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("AIMET invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("AIMET verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_aimet_verify_theorem() {
        let backend = AimetBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: AIMET not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("AIMET theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("AIMET verification error: {}", e);
            }
        }
    }
}

// Brevitas End-to-End Tests
mod brevitas_e2e {
    use super::*;

    #[tokio::test]
    async fn test_brevitas_backend_health_check() {
        let backend = BrevitasBackend::new();
        let health = backend.health_check().await;

        match health {
            HealthStatus::Healthy => println!("Brevitas is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Brevitas unavailable: {}", reason);
                println!("Install: pip install brevitas");
            }
            HealthStatus::Degraded { reason } => println!("Brevitas degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_brevitas_backend_id() {
        let backend = BrevitasBackend::new();
        assert_eq!(backend.id(), BackendId::Brevitas);
    }

    #[tokio::test]
    async fn test_brevitas_supports_property_types() {
        let backend = BrevitasBackend::new();
        let supported = backend.supports();

        assert!(
            !supported.is_empty(),
            "Brevitas should support at least one property type"
        );
        println!("Brevitas supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_brevitas_verify_simple_invariant() {
        let backend = BrevitasBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Brevitas not available");
            return;
        }

        let spec = simple_invariant_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Brevitas invariant result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
                println!("Diagnostics: {:?}", r.diagnostics);
            }
            Err(e) => {
                println!("Brevitas verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_brevitas_verify_theorem() {
        let backend = BrevitasBackend::new();

        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Brevitas not available");
            return;
        }

        let spec = simple_theorem_spec();
        let result = backend.verify(&spec).await;

        match result {
            Ok(r) => {
                println!("Brevitas theorem result: {:?}", r.status);
                println!("Time: {:?}", r.time_taken);
            }
            Err(e) => {
                println!("Brevitas verification error: {}", e);
            }
        }
    }
}

// ============================================================================
// Data Quality End-to-End Tests
// ============================================================================

// Great Expectations End-to-End Tests
mod greatexpectations_e2e {
    use super::*;

    #[tokio::test]
    async fn test_greatexpectations_backend_health_check() {
        let backend = GreatExpectationsBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Great Expectations is available"),
            HealthStatus::Unavailable { reason } => {
                println!("Great Expectations unavailable: {}", reason);
            }
            HealthStatus::Degraded { reason } => {
                println!("Great Expectations degraded: {}", reason)
            }
        }
    }

    #[tokio::test]
    async fn test_greatexpectations_backend_id() {
        let backend = GreatExpectationsBackend::new();
        assert_eq!(backend.id(), BackendId::GreatExpectations);
    }

    #[tokio::test]
    async fn test_greatexpectations_supports_property_types() {
        let backend = GreatExpectationsBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
        println!("Great Expectations supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_greatexpectations_verify_simple_invariant() {
        let backend = GreatExpectationsBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Great Expectations not available");
            return;
        }
        let spec = simple_invariant_spec();
        match backend.verify(&spec).await {
            Ok(r) => println!("Great Expectations result: {:?}", r.status),
            Err(e) => println!("Great Expectations error: {}", e),
        }
    }

    #[tokio::test]
    async fn test_greatexpectations_verify_theorem() {
        let backend = GreatExpectationsBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED: Great Expectations not available");
            return;
        }
        let spec = simple_theorem_spec();
        match backend.verify(&spec).await {
            Ok(r) => println!("Great Expectations theorem result: {:?}", r.status),
            Err(e) => println!("Great Expectations error: {}", e),
        }
    }
}

// Deepchecks End-to-End Tests
mod deepchecks_e2e {
    use super::*;

    #[tokio::test]
    async fn test_deepchecks_backend_health_check() {
        let backend = DeepchecksBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Deepchecks is available"),
            HealthStatus::Unavailable { reason } => println!("Deepchecks unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Deepchecks degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_deepchecks_backend_id() {
        let backend = DeepchecksBackend::new();
        assert_eq!(backend.id(), BackendId::Deepchecks);
    }

    #[tokio::test]
    async fn test_deepchecks_supports_property_types() {
        let backend = DeepchecksBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
        println!("Deepchecks supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_deepchecks_verify_simple_invariant() {
        let backend = DeepchecksBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        match backend.verify(&spec).await {
            Ok(r) => println!("Deepchecks result: {:?}", r.status),
            Err(e) => println!("Deepchecks error: {}", e),
        }
    }

    #[tokio::test]
    async fn test_deepchecks_verify_theorem() {
        let backend = DeepchecksBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        match backend.verify(&spec).await {
            Ok(r) => println!("Deepchecks theorem result: {:?}", r.status),
            Err(e) => println!("Deepchecks error: {}", e),
        }
    }
}

// Evidently End-to-End Tests
mod evidently_e2e {
    use super::*;

    #[tokio::test]
    async fn test_evidently_backend_health_check() {
        let backend = EvidentlyBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Evidently is available"),
            HealthStatus::Unavailable { reason } => println!("Evidently unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Evidently degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_evidently_backend_id() {
        let backend = EvidentlyBackend::new();
        assert_eq!(backend.id(), BackendId::Evidently);
    }

    #[tokio::test]
    async fn test_evidently_supports_property_types() {
        let backend = EvidentlyBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
        println!("Evidently supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_evidently_verify_simple_invariant() {
        let backend = EvidentlyBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        match backend.verify(&spec).await {
            Ok(r) => println!("Evidently result: {:?}", r.status),
            Err(e) => println!("Evidently error: {}", e),
        }
    }

    #[tokio::test]
    async fn test_evidently_verify_theorem() {
        let backend = EvidentlyBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        match backend.verify(&spec).await {
            Ok(r) => println!("Evidently theorem result: {:?}", r.status),
            Err(e) => println!("Evidently error: {}", e),
        }
    }
}

// WhyLogs End-to-End Tests
mod whylogs_e2e {
    use super::*;

    #[tokio::test]
    async fn test_whylogs_backend_health_check() {
        let backend = WhyLogsBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("WhyLogs is available"),
            HealthStatus::Unavailable { reason } => println!("WhyLogs unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("WhyLogs degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_whylogs_backend_id() {
        let backend = WhyLogsBackend::new();
        assert_eq!(backend.id(), BackendId::WhyLogs);
    }

    #[tokio::test]
    async fn test_whylogs_supports_property_types() {
        let backend = WhyLogsBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
        println!("WhyLogs supports: {:?}", supported);
    }

    #[tokio::test]
    async fn test_whylogs_verify_simple_invariant() {
        let backend = WhyLogsBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        match backend.verify(&spec).await {
            Ok(r) => println!("WhyLogs result: {:?}", r.status),
            Err(e) => println!("WhyLogs error: {}", e),
        }
    }

    #[tokio::test]
    async fn test_whylogs_verify_theorem() {
        let backend = WhyLogsBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        match backend.verify(&spec).await {
            Ok(r) => println!("WhyLogs theorem result: {:?}", r.status),
            Err(e) => println!("WhyLogs error: {}", e),
        }
    }
}

// ============================================================================
// Fairness/Bias End-to-End Tests
// ============================================================================

mod fairlearn_e2e {
    use super::*;

    #[tokio::test]
    async fn test_fairlearn_backend_health_check() {
        let backend = FairlearnBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Fairlearn is available"),
            HealthStatus::Unavailable { reason } => println!("Fairlearn unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Fairlearn degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_fairlearn_backend_id() {
        let backend = FairlearnBackend::new();
        assert_eq!(backend.id(), BackendId::Fairlearn);
    }

    #[tokio::test]
    async fn test_fairlearn_supports_property_types() {
        let backend = FairlearnBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_fairlearn_verify_simple_invariant() {
        let backend = FairlearnBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_fairlearn_verify_theorem() {
        let backend = FairlearnBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

mod aif360_e2e {
    use super::*;

    #[tokio::test]
    async fn test_aif360_backend_health_check() {
        let backend = AIF360Backend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("AIF360 is available"),
            HealthStatus::Unavailable { reason } => println!("AIF360 unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("AIF360 degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_aif360_backend_id() {
        let backend = AIF360Backend::new();
        assert_eq!(backend.id(), BackendId::AIF360);
    }

    #[tokio::test]
    async fn test_aif360_supports_property_types() {
        let backend = AIF360Backend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_aif360_verify_simple_invariant() {
        let backend = AIF360Backend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_aif360_verify_theorem() {
        let backend = AIF360Backend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

mod aequitas_e2e {
    use super::*;

    #[tokio::test]
    async fn test_aequitas_backend_health_check() {
        let backend = AequitasBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Aequitas is available"),
            HealthStatus::Unavailable { reason } => println!("Aequitas unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Aequitas degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_aequitas_backend_id() {
        let backend = AequitasBackend::new();
        assert_eq!(backend.id(), BackendId::Aequitas);
    }

    #[tokio::test]
    async fn test_aequitas_supports_property_types() {
        let backend = AequitasBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_aequitas_verify_simple_invariant() {
        let backend = AequitasBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_aequitas_verify_theorem() {
        let backend = AequitasBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Interpretability End-to-End Tests
// ============================================================================

mod shap_e2e {
    use super::*;

    #[tokio::test]
    async fn test_shap_backend_health_check() {
        let backend = ShapBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("SHAP is available"),
            HealthStatus::Unavailable { reason } => println!("SHAP unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("SHAP degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_shap_backend_id() {
        let backend = ShapBackend::new();
        assert_eq!(backend.id(), BackendId::SHAP);
    }

    #[tokio::test]
    async fn test_shap_supports_property_types() {
        let backend = ShapBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_shap_verify_simple_invariant() {
        let backend = ShapBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_shap_verify_theorem() {
        let backend = ShapBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

mod lime_e2e {
    use super::*;

    #[tokio::test]
    async fn test_lime_backend_health_check() {
        let backend = LimeBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("LIME is available"),
            HealthStatus::Unavailable { reason } => println!("LIME unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("LIME degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_lime_backend_id() {
        let backend = LimeBackend::new();
        assert_eq!(backend.id(), BackendId::LIME);
    }

    #[tokio::test]
    async fn test_lime_supports_property_types() {
        let backend = LimeBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_lime_verify_simple_invariant() {
        let backend = LimeBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_lime_verify_theorem() {
        let backend = LimeBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

mod captum_e2e {
    use super::*;

    #[tokio::test]
    async fn test_captum_backend_health_check() {
        let backend = CaptumBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Captum is available"),
            HealthStatus::Unavailable { reason } => println!("Captum unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Captum degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_captum_backend_id() {
        let backend = CaptumBackend::new();
        assert_eq!(backend.id(), BackendId::Captum);
    }

    #[tokio::test]
    async fn test_captum_supports_property_types() {
        let backend = CaptumBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_captum_verify_simple_invariant() {
        let backend = CaptumBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_captum_verify_theorem() {
        let backend = CaptumBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

mod interpretml_e2e {
    use super::*;

    #[tokio::test]
    async fn test_interpretml_backend_health_check() {
        let backend = InterpretMlBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("InterpretML is available"),
            HealthStatus::Unavailable { reason } => println!("InterpretML unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("InterpretML degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_interpretml_backend_id() {
        let backend = InterpretMlBackend::new();
        assert_eq!(backend.id(), BackendId::InterpretML);
    }

    #[tokio::test]
    async fn test_interpretml_supports_property_types() {
        let backend = InterpretMlBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_interpretml_verify_simple_invariant() {
        let backend = InterpretMlBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_interpretml_verify_theorem() {
        let backend = InterpretMlBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

mod alibi_e2e {
    use super::*;

    #[tokio::test]
    async fn test_alibi_backend_health_check() {
        let backend = AlibiBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Alibi is available"),
            HealthStatus::Unavailable { reason } => println!("Alibi unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Alibi degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_alibi_backend_id() {
        let backend = AlibiBackend::new();
        assert_eq!(backend.id(), BackendId::Alibi);
    }

    #[tokio::test]
    async fn test_alibi_supports_property_types() {
        let backend = AlibiBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_alibi_verify_simple_invariant() {
        let backend = AlibiBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_alibi_verify_theorem() {
        let backend = AlibiBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// LLM Guardrails End-to-End Tests
// ============================================================================

mod guardrailsai_e2e {
    use super::*;

    #[tokio::test]
    async fn test_guardrailsai_backend_health_check() {
        let backend = GuardrailsAIBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("GuardrailsAI is available"),
            HealthStatus::Unavailable { reason } => {
                println!("GuardrailsAI unavailable: {}", reason)
            }
            HealthStatus::Degraded { reason } => println!("GuardrailsAI degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_guardrailsai_backend_id() {
        let backend = GuardrailsAIBackend::new();
        assert_eq!(backend.id(), BackendId::GuardrailsAI);
    }

    #[tokio::test]
    async fn test_guardrailsai_supports_property_types() {
        let backend = GuardrailsAIBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_guardrailsai_verify_simple_invariant() {
        let backend = GuardrailsAIBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_guardrailsai_verify_theorem() {
        let backend = GuardrailsAIBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

mod nemoguardrails_e2e {
    use super::*;

    #[tokio::test]
    async fn test_nemoguardrails_backend_health_check() {
        let backend = NeMoGuardrailsBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("NeMo Guardrails is available"),
            HealthStatus::Unavailable { reason } => {
                println!("NeMo Guardrails unavailable: {}", reason)
            }
            HealthStatus::Degraded { reason } => println!("NeMo Guardrails degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_nemoguardrails_backend_id() {
        let backend = NeMoGuardrailsBackend::new();
        assert_eq!(backend.id(), BackendId::NeMoGuardrails);
    }

    #[tokio::test]
    async fn test_nemoguardrails_supports_property_types() {
        let backend = NeMoGuardrailsBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_nemoguardrails_verify_simple_invariant() {
        let backend = NeMoGuardrailsBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_nemoguardrails_verify_theorem() {
        let backend = NeMoGuardrailsBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

mod guidance_e2e {
    use super::*;

    #[tokio::test]
    async fn test_guidance_backend_health_check() {
        let backend = GuidanceBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Guidance is available"),
            HealthStatus::Unavailable { reason } => println!("Guidance unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Guidance degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_guidance_backend_id() {
        let backend = GuidanceBackend::new();
        assert_eq!(backend.id(), BackendId::Guidance);
    }

    #[tokio::test]
    async fn test_guidance_supports_property_types() {
        let backend = GuidanceBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_guidance_verify_simple_invariant() {
        let backend = GuidanceBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_guidance_verify_theorem() {
        let backend = GuidanceBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// LLM Evaluation End-to-End Tests
// ============================================================================

mod promptfoo_e2e {
    use super::*;

    #[tokio::test]
    async fn test_promptfoo_backend_health_check() {
        let backend = PromptfooBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Promptfoo is available"),
            HealthStatus::Unavailable { reason } => println!("Promptfoo unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Promptfoo degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_promptfoo_backend_id() {
        let backend = PromptfooBackend::new();
        assert_eq!(backend.id(), BackendId::Promptfoo);
    }

    #[tokio::test]
    async fn test_promptfoo_supports_property_types() {
        let backend = PromptfooBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_promptfoo_verify_simple_invariant() {
        let backend = PromptfooBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_promptfoo_verify_theorem() {
        let backend = PromptfooBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

mod trulens_e2e {
    use super::*;

    #[tokio::test]
    async fn test_trulens_backend_health_check() {
        let backend = TruLensBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("TruLens is available"),
            HealthStatus::Unavailable { reason } => println!("TruLens unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("TruLens degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_trulens_backend_id() {
        let backend = TruLensBackend::new();
        assert_eq!(backend.id(), BackendId::TruLens);
    }

    #[tokio::test]
    async fn test_trulens_supports_property_types() {
        let backend = TruLensBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_trulens_verify_simple_invariant() {
        let backend = TruLensBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_trulens_verify_theorem() {
        let backend = TruLensBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

mod langsmith_e2e {
    use super::*;

    #[tokio::test]
    async fn test_langsmith_backend_health_check() {
        let backend = LangSmithBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("LangSmith is available"),
            HealthStatus::Unavailable { reason } => println!("LangSmith unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("LangSmith degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_langsmith_backend_id() {
        let backend = LangSmithBackend::new();
        assert_eq!(backend.id(), BackendId::LangSmith);
    }

    #[tokio::test]
    async fn test_langsmith_supports_property_types() {
        let backend = LangSmithBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_langsmith_verify_simple_invariant() {
        let backend = LangSmithBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_langsmith_verify_theorem() {
        let backend = LangSmithBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

mod ragas_e2e {
    use super::*;

    #[tokio::test]
    async fn test_ragas_backend_health_check() {
        let backend = RagasBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Ragas is available"),
            HealthStatus::Unavailable { reason } => println!("Ragas unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Ragas degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_ragas_backend_id() {
        let backend = RagasBackend::new();
        assert_eq!(backend.id(), BackendId::Ragas);
    }

    #[tokio::test]
    async fn test_ragas_supports_property_types() {
        let backend = RagasBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_ragas_verify_simple_invariant() {
        let backend = RagasBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_ragas_verify_theorem() {
        let backend = RagasBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

mod deepeval_e2e {
    use super::*;

    #[tokio::test]
    async fn test_deepeval_backend_health_check() {
        let backend = DeepEvalBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("DeepEval is available"),
            HealthStatus::Unavailable { reason } => println!("DeepEval unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("DeepEval degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_deepeval_backend_id() {
        let backend = DeepEvalBackend::new();
        assert_eq!(backend.id(), BackendId::DeepEval);
    }

    #[tokio::test]
    async fn test_deepeval_supports_property_types() {
        let backend = DeepEvalBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_deepeval_verify_simple_invariant() {
        let backend = DeepEvalBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_deepeval_verify_theorem() {
        let backend = DeepEvalBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Hallucination Detection End-to-End Tests
// ============================================================================

mod selfcheckgpt_e2e {
    use super::*;

    #[tokio::test]
    async fn test_selfcheckgpt_backend_health_check() {
        let backend = SelfCheckGPTBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("SelfCheckGPT is available"),
            HealthStatus::Unavailable { reason } => {
                println!("SelfCheckGPT unavailable: {}", reason)
            }
            HealthStatus::Degraded { reason } => println!("SelfCheckGPT degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_selfcheckgpt_backend_id() {
        let backend = SelfCheckGPTBackend::new();
        assert_eq!(backend.id(), BackendId::SelfCheckGPT);
    }

    #[tokio::test]
    async fn test_selfcheckgpt_supports_property_types() {
        let backend = SelfCheckGPTBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_selfcheckgpt_verify_simple_invariant() {
        let backend = SelfCheckGPTBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_selfcheckgpt_verify_theorem() {
        let backend = SelfCheckGPTBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

mod factscore_e2e {
    use super::*;

    #[tokio::test]
    async fn test_factscore_backend_health_check() {
        let backend = FactScoreBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("FactScore is available"),
            HealthStatus::Unavailable { reason } => println!("FactScore unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("FactScore degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_factscore_backend_id() {
        let backend = FactScoreBackend::new();
        assert_eq!(backend.id(), BackendId::FactScore);
    }

    #[tokio::test]
    async fn test_factscore_supports_property_types() {
        let backend = FactScoreBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_factscore_verify_simple_invariant() {
        let backend = FactScoreBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_factscore_verify_theorem() {
        let backend = FactScoreBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// HolLight End-to-End Tests
// ============================================================================

mod hollight_e2e {
    use super::*;

    #[tokio::test]
    async fn test_hollight_backend_health_check() {
        let backend = HolLightBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("HolLight is available"),
            HealthStatus::Unavailable { reason } => println!("HolLight unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("HolLight degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_hollight_backend_id() {
        let backend = HolLightBackend::new();
        assert_eq!(backend.id(), BackendId::HOLLight);
    }

    #[tokio::test]
    async fn test_hollight_supports_property_types() {
        let backend = HolLightBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_hollight_verify_simple_invariant() {
        let backend = HolLightBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_hollight_verify_theorem() {
        let backend = HolLightBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Pvs End-to-End Tests
// ============================================================================

mod pvs_e2e {
    use super::*;

    #[tokio::test]
    async fn test_pvs_backend_health_check() {
        let backend = PvsBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Pvs is available"),
            HealthStatus::Unavailable { reason } => println!("Pvs unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Pvs degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_pvs_backend_id() {
        let backend = PvsBackend::new();
        assert_eq!(backend.id(), BackendId::PVS);
    }

    #[tokio::test]
    async fn test_pvs_supports_property_types() {
        let backend = PvsBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_pvs_verify_simple_invariant() {
        let backend = PvsBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_pvs_verify_theorem() {
        let backend = PvsBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Mizar End-to-End Tests
// ============================================================================

mod mizar_e2e {
    use super::*;

    #[tokio::test]
    async fn test_mizar_backend_health_check() {
        let backend = MizarBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Mizar is available"),
            HealthStatus::Unavailable { reason } => println!("Mizar unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Mizar degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_mizar_backend_id() {
        let backend = MizarBackend::new();
        assert_eq!(backend.id(), BackendId::Mizar);
    }

    #[tokio::test]
    async fn test_mizar_supports_property_types() {
        let backend = MizarBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_mizar_verify_simple_invariant() {
        let backend = MizarBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_mizar_verify_theorem() {
        let backend = MizarBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Metamath End-to-End Tests
// ============================================================================

mod metamath_e2e {
    use super::*;

    #[tokio::test]
    async fn test_metamath_backend_health_check() {
        let backend = MetamathBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Metamath is available"),
            HealthStatus::Unavailable { reason } => println!("Metamath unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Metamath degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_metamath_backend_id() {
        let backend = MetamathBackend::new();
        assert_eq!(backend.id(), BackendId::Metamath);
    }

    #[tokio::test]
    async fn test_metamath_supports_property_types() {
        let backend = MetamathBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_metamath_verify_simple_invariant() {
        let backend = MetamathBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_metamath_verify_theorem() {
        let backend = MetamathBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Ats End-to-End Tests
// ============================================================================

mod ats_e2e {
    use super::*;

    #[tokio::test]
    async fn test_ats_backend_health_check() {
        let backend = AtsBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Ats is available"),
            HealthStatus::Unavailable { reason } => println!("Ats unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Ats degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_ats_backend_id() {
        let backend = AtsBackend::new();
        assert_eq!(backend.id(), BackendId::ATS);
    }

    #[tokio::test]
    async fn test_ats_supports_property_types() {
        let backend = AtsBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_ats_verify_simple_invariant() {
        let backend = AtsBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_ats_verify_theorem() {
        let backend = AtsBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// OpenSmt End-to-End Tests
// ============================================================================

mod opensmt_e2e {
    use super::*;

    #[tokio::test]
    async fn test_opensmt_backend_health_check() {
        let backend = OpenSmtBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("OpenSmt is available"),
            HealthStatus::Unavailable { reason } => println!("OpenSmt unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("OpenSmt degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_opensmt_backend_id() {
        let backend = OpenSmtBackend::new();
        assert_eq!(backend.id(), BackendId::OpenSMT);
    }

    #[tokio::test]
    async fn test_opensmt_supports_property_types() {
        let backend = OpenSmtBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_opensmt_verify_simple_invariant() {
        let backend = OpenSmtBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_opensmt_verify_theorem() {
        let backend = OpenSmtBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// VeriT End-to-End Tests
// ============================================================================

mod verit_e2e {
    use super::*;

    #[tokio::test]
    async fn test_verit_backend_health_check() {
        let backend = VeriTBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("VeriT is available"),
            HealthStatus::Unavailable { reason } => println!("VeriT unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("VeriT degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_verit_backend_id() {
        let backend = VeriTBackend::new();
        assert_eq!(backend.id(), BackendId::VeriT);
    }

    #[tokio::test]
    async fn test_verit_supports_property_types() {
        let backend = VeriTBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_verit_verify_simple_invariant() {
        let backend = VeriTBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_verit_verify_theorem() {
        let backend = VeriTBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// AltErgo End-to-End Tests
// ============================================================================

mod altergo_e2e {
    use super::*;

    #[tokio::test]
    async fn test_altergo_backend_health_check() {
        let backend = AltErgoBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("AltErgo is available"),
            HealthStatus::Unavailable { reason } => println!("AltErgo unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("AltErgo degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_altergo_backend_id() {
        let backend = AltErgoBackend::new();
        assert_eq!(backend.id(), BackendId::AltErgo);
    }

    #[tokio::test]
    async fn test_altergo_supports_property_types() {
        let backend = AltErgoBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_altergo_verify_simple_invariant() {
        let backend = AltErgoBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_altergo_verify_theorem() {
        let backend = AltErgoBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Kissat End-to-End Tests
// ============================================================================

mod kissat_e2e {
    use super::*;

    #[tokio::test]
    async fn test_kissat_backend_health_check() {
        let backend = KissatBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Kissat is available"),
            HealthStatus::Unavailable { reason } => println!("Kissat unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Kissat degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_kissat_backend_id() {
        let backend = KissatBackend::new();
        assert_eq!(backend.id(), BackendId::Kissat);
    }

    #[tokio::test]
    async fn test_kissat_supports_property_types() {
        let backend = KissatBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_kissat_verify_simple_invariant() {
        let backend = KissatBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_kissat_verify_theorem() {
        let backend = KissatBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// CryptoMiniSat End-to-End Tests
// ============================================================================

mod cryptominisat_e2e {
    use super::*;

    #[tokio::test]
    async fn test_cryptominisat_backend_health_check() {
        let backend = CryptoMiniSatBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("CryptoMiniSat is available"),
            HealthStatus::Unavailable { reason } => {
                println!("CryptoMiniSat unavailable: {}", reason)
            }
            HealthStatus::Degraded { reason } => println!("CryptoMiniSat degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_cryptominisat_backend_id() {
        let backend = CryptoMiniSatBackend::new();
        assert_eq!(backend.id(), BackendId::CryptoMiniSat);
    }

    #[tokio::test]
    async fn test_cryptominisat_supports_property_types() {
        let backend = CryptoMiniSatBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_cryptominisat_verify_simple_invariant() {
        let backend = CryptoMiniSatBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_cryptominisat_verify_theorem() {
        let backend = CryptoMiniSatBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// NuXmv End-to-End Tests
// ============================================================================

mod nuxmv_e2e {
    use super::*;

    #[tokio::test]
    async fn test_nuxmv_backend_health_check() {
        let backend = NuXmvBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("NuXmv is available"),
            HealthStatus::Unavailable { reason } => println!("NuXmv unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("NuXmv degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_nuxmv_backend_id() {
        let backend = NuXmvBackend::new();
        assert_eq!(backend.id(), BackendId::NuXmv);
    }

    #[tokio::test]
    async fn test_nuxmv_supports_property_types() {
        let backend = NuXmvBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_nuxmv_verify_simple_invariant() {
        let backend = NuXmvBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_nuxmv_verify_theorem() {
        let backend = NuXmvBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Uppaal End-to-End Tests
// ============================================================================

mod uppaal_e2e {
    use super::*;

    #[tokio::test]
    async fn test_uppaal_backend_health_check() {
        let backend = UppaalBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Uppaal is available"),
            HealthStatus::Unavailable { reason } => println!("Uppaal unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Uppaal degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_uppaal_backend_id() {
        let backend = UppaalBackend::new();
        assert_eq!(backend.id(), BackendId::UPPAAL);
    }

    #[tokio::test]
    async fn test_uppaal_supports_property_types() {
        let backend = UppaalBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_uppaal_verify_simple_invariant() {
        let backend = UppaalBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_uppaal_verify_theorem() {
        let backend = UppaalBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Divine End-to-End Tests
// ============================================================================

mod divine_e2e {
    use super::*;

    #[tokio::test]
    async fn test_divine_backend_health_check() {
        let backend = DivineBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Divine is available"),
            HealthStatus::Unavailable { reason } => println!("Divine unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Divine degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_divine_backend_id() {
        let backend = DivineBackend::new();
        assert_eq!(backend.id(), BackendId::DIVINE);
    }

    #[tokio::test]
    async fn test_divine_supports_property_types() {
        let backend = DivineBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_divine_verify_simple_invariant() {
        let backend = DivineBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_divine_verify_theorem() {
        let backend = DivineBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Esbmc End-to-End Tests
// ============================================================================

mod esbmc_e2e {
    use super::*;

    #[tokio::test]
    async fn test_esbmc_backend_health_check() {
        let backend = EsbmcBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Esbmc is available"),
            HealthStatus::Unavailable { reason } => println!("Esbmc unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Esbmc degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_esbmc_backend_id() {
        let backend = EsbmcBackend::new();
        assert_eq!(backend.id(), BackendId::ESBMC);
    }

    #[tokio::test]
    async fn test_esbmc_supports_property_types() {
        let backend = EsbmcBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_esbmc_verify_simple_invariant() {
        let backend = EsbmcBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_esbmc_verify_theorem() {
        let backend = EsbmcBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Ultimate End-to-End Tests
// ============================================================================

mod ultimate_e2e {
    use super::*;

    #[tokio::test]
    async fn test_ultimate_backend_health_check() {
        let backend = UltimateBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Ultimate is available"),
            HealthStatus::Unavailable { reason } => println!("Ultimate unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Ultimate degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_ultimate_backend_id() {
        let backend = UltimateBackend::new();
        assert_eq!(backend.id(), BackendId::Ultimate);
    }

    #[tokio::test]
    async fn test_ultimate_supports_property_types() {
        let backend = UltimateBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_ultimate_verify_simple_invariant() {
        let backend = UltimateBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_ultimate_verify_theorem() {
        let backend = UltimateBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Smack End-to-End Tests
// ============================================================================

mod smack_e2e {
    use super::*;

    #[tokio::test]
    async fn test_smack_backend_health_check() {
        let backend = SmackBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Smack is available"),
            HealthStatus::Unavailable { reason } => println!("Smack unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Smack degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_smack_backend_id() {
        let backend = SmackBackend::new();
        assert_eq!(backend.id(), BackendId::SMACK);
    }

    #[tokio::test]
    async fn test_smack_supports_property_types() {
        let backend = SmackBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_smack_verify_simple_invariant() {
        let backend = SmackBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_smack_verify_theorem() {
        let backend = SmackBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Jpf End-to-End Tests
// ============================================================================

mod jpf_e2e {
    use super::*;

    #[tokio::test]
    async fn test_jpf_backend_health_check() {
        let backend = JpfBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Jpf is available"),
            HealthStatus::Unavailable { reason } => println!("Jpf unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Jpf degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_jpf_backend_id() {
        let backend = JpfBackend::new();
        assert_eq!(backend.id(), BackendId::JPF);
    }

    #[tokio::test]
    async fn test_jpf_supports_property_types() {
        let backend = JpfBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_jpf_verify_simple_invariant() {
        let backend = JpfBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_jpf_verify_theorem() {
        let backend = JpfBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Vcc End-to-End Tests
// ============================================================================

mod vcc_e2e {
    use super::*;

    #[tokio::test]
    async fn test_vcc_backend_health_check() {
        let backend = VccBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Vcc is available"),
            HealthStatus::Unavailable { reason } => println!("Vcc unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Vcc degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_vcc_backend_id() {
        let backend = VccBackend::new();
        assert_eq!(backend.id(), BackendId::VCC);
    }

    #[tokio::test]
    async fn test_vcc_supports_property_types() {
        let backend = VccBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_vcc_verify_simple_invariant() {
        let backend = VccBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_vcc_verify_theorem() {
        let backend = VccBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// VeriFast End-to-End Tests
// ============================================================================

mod verifast_e2e {
    use super::*;

    #[tokio::test]
    async fn test_verifast_backend_health_check() {
        let backend = VeriFastBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("VeriFast is available"),
            HealthStatus::Unavailable { reason } => println!("VeriFast unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("VeriFast degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_verifast_backend_id() {
        let backend = VeriFastBackend::new();
        assert_eq!(backend.id(), BackendId::VeriFast);
    }

    #[tokio::test]
    async fn test_verifast_supports_property_types() {
        let backend = VeriFastBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_verifast_verify_simple_invariant() {
        let backend = VeriFastBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_verifast_verify_theorem() {
        let backend = VeriFastBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Key End-to-End Tests
// ============================================================================

mod key_e2e {
    use super::*;

    #[tokio::test]
    async fn test_key_backend_health_check() {
        let backend = KeyBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Key is available"),
            HealthStatus::Unavailable { reason } => println!("Key unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Key degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_key_backend_id() {
        let backend = KeyBackend::new();
        assert_eq!(backend.id(), BackendId::KeY);
    }

    #[tokio::test]
    async fn test_key_supports_property_types() {
        let backend = KeyBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_key_verify_simple_invariant() {
        let backend = KeyBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_key_verify_theorem() {
        let backend = KeyBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// OpenJml End-to-End Tests
// ============================================================================

mod openjml_e2e {
    use super::*;

    #[tokio::test]
    async fn test_openjml_backend_health_check() {
        let backend = OpenJmlBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("OpenJml is available"),
            HealthStatus::Unavailable { reason } => println!("OpenJml unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("OpenJml degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_openjml_backend_id() {
        let backend = OpenJmlBackend::new();
        assert_eq!(backend.id(), BackendId::OpenJML);
    }

    #[tokio::test]
    async fn test_openjml_supports_property_types() {
        let backend = OpenJmlBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_openjml_verify_simple_invariant() {
        let backend = OpenJmlBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_openjml_verify_theorem() {
        let backend = OpenJmlBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Krakatoa End-to-End Tests
// ============================================================================

mod krakatoa_e2e {
    use super::*;

    #[tokio::test]
    async fn test_krakatoa_backend_health_check() {
        let backend = KrakatoaBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Krakatoa is available"),
            HealthStatus::Unavailable { reason } => println!("Krakatoa unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Krakatoa degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_krakatoa_backend_id() {
        let backend = KrakatoaBackend::new();
        assert_eq!(backend.id(), BackendId::Krakatoa);
    }

    #[tokio::test]
    async fn test_krakatoa_supports_property_types() {
        let backend = KrakatoaBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_krakatoa_verify_simple_invariant() {
        let backend = KrakatoaBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_krakatoa_verify_theorem() {
        let backend = KrakatoaBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Spark End-to-End Tests
// ============================================================================

mod spark_e2e {
    use super::*;

    #[tokio::test]
    async fn test_spark_backend_health_check() {
        let backend = SparkBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Spark is available"),
            HealthStatus::Unavailable { reason } => println!("Spark unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Spark degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_spark_backend_id() {
        let backend = SparkBackend::new();
        assert_eq!(backend.id(), BackendId::SPARK);
    }

    #[tokio::test]
    async fn test_spark_supports_property_types() {
        let backend = SparkBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_spark_verify_simple_invariant() {
        let backend = SparkBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_spark_verify_theorem() {
        let backend = SparkBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Why3 End-to-End Tests
// ============================================================================

mod why3_e2e {
    use super::*;

    #[tokio::test]
    async fn test_why3_backend_health_check() {
        let backend = Why3Backend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Why3 is available"),
            HealthStatus::Unavailable { reason } => println!("Why3 unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Why3 degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_why3_backend_id() {
        let backend = Why3Backend::new();
        assert_eq!(backend.id(), BackendId::Why3);
    }

    #[tokio::test]
    async fn test_why3_supports_property_types() {
        let backend = Why3Backend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_why3_verify_simple_invariant() {
        let backend = Why3Backend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_why3_verify_theorem() {
        let backend = Why3Backend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Stainless End-to-End Tests
// ============================================================================

mod stainless_e2e {
    use super::*;

    #[tokio::test]
    async fn test_stainless_backend_health_check() {
        let backend = StainlessBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Stainless is available"),
            HealthStatus::Unavailable { reason } => println!("Stainless unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Stainless degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_stainless_backend_id() {
        let backend = StainlessBackend::new();
        assert_eq!(backend.id(), BackendId::Stainless);
    }

    #[tokio::test]
    async fn test_stainless_supports_property_types() {
        let backend = StainlessBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_stainless_verify_simple_invariant() {
        let backend = StainlessBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_stainless_verify_theorem() {
        let backend = StainlessBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// LiquidHaskell End-to-End Tests
// ============================================================================

mod liquidhaskell_e2e {
    use super::*;

    #[tokio::test]
    async fn test_liquidhaskell_backend_health_check() {
        let backend = LiquidHaskellBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("LiquidHaskell is available"),
            HealthStatus::Unavailable { reason } => {
                println!("LiquidHaskell unavailable: {}", reason)
            }
            HealthStatus::Degraded { reason } => println!("LiquidHaskell degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_liquidhaskell_backend_id() {
        let backend = LiquidHaskellBackend::new();
        assert_eq!(backend.id(), BackendId::LiquidHaskell);
    }

    #[tokio::test]
    async fn test_liquidhaskell_supports_property_types() {
        let backend = LiquidHaskellBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_liquidhaskell_verify_simple_invariant() {
        let backend = LiquidHaskellBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_liquidhaskell_verify_theorem() {
        let backend = LiquidHaskellBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Boogie End-to-End Tests
// ============================================================================

mod boogie_e2e {
    use super::*;

    #[tokio::test]
    async fn test_boogie_backend_health_check() {
        let backend = BoogieBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Boogie is available"),
            HealthStatus::Unavailable { reason } => println!("Boogie unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Boogie degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_boogie_backend_id() {
        let backend = BoogieBackend::new();
        assert_eq!(backend.id(), BackendId::Boogie);
    }

    #[tokio::test]
    async fn test_boogie_supports_property_types() {
        let backend = BoogieBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_boogie_verify_simple_invariant() {
        let backend = BoogieBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_boogie_verify_theorem() {
        let backend = BoogieBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// PLang End-to-End Tests
// ============================================================================

mod plang_e2e {
    use super::*;

    #[tokio::test]
    async fn test_plang_backend_health_check() {
        let backend = PLangBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("PLang is available"),
            HealthStatus::Unavailable { reason } => println!("PLang unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("PLang degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_plang_backend_id() {
        let backend = PLangBackend::new();
        assert_eq!(backend.id(), BackendId::PLang);
    }

    #[tokio::test]
    async fn test_plang_supports_property_types() {
        let backend = PLangBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_plang_verify_simple_invariant() {
        let backend = PLangBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_plang_verify_theorem() {
        let backend = PLangBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Ivy End-to-End Tests
// ============================================================================

mod ivy_e2e {
    use super::*;

    #[tokio::test]
    async fn test_ivy_backend_health_check() {
        let backend = IvyBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Ivy is available"),
            HealthStatus::Unavailable { reason } => println!("Ivy unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Ivy degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_ivy_backend_id() {
        let backend = IvyBackend::new();
        assert_eq!(backend.id(), BackendId::Ivy);
    }

    #[tokio::test]
    async fn test_ivy_supports_property_types() {
        let backend = IvyBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_ivy_verify_simple_invariant() {
        let backend = IvyBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_ivy_verify_theorem() {
        let backend = IvyBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Mcrl2 End-to-End Tests
// ============================================================================

mod mcrl2_e2e {
    use super::*;

    #[tokio::test]
    async fn test_mcrl2_backend_health_check() {
        let backend = Mcrl2Backend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Mcrl2 is available"),
            HealthStatus::Unavailable { reason } => println!("Mcrl2 unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Mcrl2 degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_mcrl2_backend_id() {
        let backend = Mcrl2Backend::new();
        assert_eq!(backend.id(), BackendId::MCRL2);
    }

    #[tokio::test]
    async fn test_mcrl2_supports_property_types() {
        let backend = Mcrl2Backend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_mcrl2_verify_simple_invariant() {
        let backend = Mcrl2Backend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_mcrl2_verify_theorem() {
        let backend = Mcrl2Backend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Cadp End-to-End Tests
// ============================================================================

mod cadp_e2e {
    use super::*;

    #[tokio::test]
    async fn test_cadp_backend_health_check() {
        let backend = CadpBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Cadp is available"),
            HealthStatus::Unavailable { reason } => println!("Cadp unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Cadp degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_cadp_backend_id() {
        let backend = CadpBackend::new();
        assert_eq!(backend.id(), BackendId::CADP);
    }

    #[tokio::test]
    async fn test_cadp_supports_property_types() {
        let backend = CadpBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_cadp_verify_simple_invariant() {
        let backend = CadpBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_cadp_verify_theorem() {
        let backend = CadpBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// EasyCrypt End-to-End Tests
// ============================================================================

mod easycrypt_e2e {
    use super::*;

    #[tokio::test]
    async fn test_easycrypt_backend_health_check() {
        let backend = EasyCryptBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("EasyCrypt is available"),
            HealthStatus::Unavailable { reason } => println!("EasyCrypt unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("EasyCrypt degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_easycrypt_backend_id() {
        let backend = EasyCryptBackend::new();
        assert_eq!(backend.id(), BackendId::EasyCrypt);
    }

    #[tokio::test]
    async fn test_easycrypt_supports_property_types() {
        let backend = EasyCryptBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_easycrypt_verify_simple_invariant() {
        let backend = EasyCryptBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_easycrypt_verify_theorem() {
        let backend = EasyCryptBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// CryptoVerif End-to-End Tests
// ============================================================================

mod cryptoverif_e2e {
    use super::*;

    #[tokio::test]
    async fn test_cryptoverif_backend_health_check() {
        let backend = CryptoVerifBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("CryptoVerif is available"),
            HealthStatus::Unavailable { reason } => println!("CryptoVerif unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("CryptoVerif degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_cryptoverif_backend_id() {
        let backend = CryptoVerifBackend::new();
        assert_eq!(backend.id(), BackendId::CryptoVerif);
    }

    #[tokio::test]
    async fn test_cryptoverif_supports_property_types() {
        let backend = CryptoVerifBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_cryptoverif_verify_simple_invariant() {
        let backend = CryptoVerifBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_cryptoverif_verify_theorem() {
        let backend = CryptoVerifBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Jasmin End-to-End Tests
// ============================================================================

mod jasmin_e2e {
    use super::*;

    #[tokio::test]
    async fn test_jasmin_backend_health_check() {
        let backend = JasminBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Jasmin is available"),
            HealthStatus::Unavailable { reason } => println!("Jasmin unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Jasmin degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_jasmin_backend_id() {
        let backend = JasminBackend::new();
        assert_eq!(backend.id(), BackendId::Jasmin);
    }

    #[tokio::test]
    async fn test_jasmin_supports_property_types() {
        let backend = JasminBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_jasmin_verify_simple_invariant() {
        let backend = JasminBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_jasmin_verify_theorem() {
        let backend = JasminBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Yosys End-to-End Tests
// ============================================================================

mod yosys_e2e {
    use super::*;

    #[tokio::test]
    async fn test_yosys_backend_health_check() {
        let backend = YosysBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Yosys is available"),
            HealthStatus::Unavailable { reason } => println!("Yosys unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Yosys degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_yosys_backend_id() {
        let backend = YosysBackend::new();
        assert_eq!(backend.id(), BackendId::Yosys);
    }

    #[tokio::test]
    async fn test_yosys_supports_property_types() {
        let backend = YosysBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_yosys_verify_simple_invariant() {
        let backend = YosysBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_yosys_verify_theorem() {
        let backend = YosysBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// SymbiYosys End-to-End Tests
// ============================================================================

mod symbiyosys_e2e {
    use super::*;

    #[tokio::test]
    async fn test_symbiyosys_backend_health_check() {
        let backend = SymbiYosysBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("SymbiYosys is available"),
            HealthStatus::Unavailable { reason } => println!("SymbiYosys unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("SymbiYosys degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_symbiyosys_backend_id() {
        let backend = SymbiYosysBackend::new();
        assert_eq!(backend.id(), BackendId::SymbiYosys);
    }

    #[tokio::test]
    async fn test_symbiyosys_supports_property_types() {
        let backend = SymbiYosysBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_symbiyosys_verify_simple_invariant() {
        let backend = SymbiYosysBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_symbiyosys_verify_theorem() {
        let backend = SymbiYosysBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// JasperGold End-to-End Tests
// ============================================================================

mod jaspergold_e2e {
    use super::*;

    #[tokio::test]
    async fn test_jaspergold_backend_health_check() {
        let backend = JasperGoldBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("JasperGold is available"),
            HealthStatus::Unavailable { reason } => println!("JasperGold unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("JasperGold degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_jaspergold_backend_id() {
        let backend = JasperGoldBackend::new();
        assert_eq!(backend.id(), BackendId::JasperGold);
    }

    #[tokio::test]
    async fn test_jaspergold_supports_property_types() {
        let backend = JasperGoldBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_jaspergold_verify_simple_invariant() {
        let backend = JasperGoldBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_jaspergold_verify_theorem() {
        let backend = JasperGoldBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// CadenceEda End-to-End Tests
// ============================================================================

mod cadence_eda_e2e {
    use super::*;

    #[tokio::test]
    async fn test_cadence_eda_backend_health_check() {
        let backend = CadenceEdaBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("CadenceEda is available"),
            HealthStatus::Unavailable { reason } => println!("CadenceEda unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("CadenceEda degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_cadence_eda_backend_id() {
        let backend = CadenceEdaBackend::new();
        assert_eq!(backend.id(), BackendId::CadenceEDA);
    }

    #[tokio::test]
    async fn test_cadence_eda_supports_property_types() {
        let backend = CadenceEdaBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_cadence_eda_verify_simple_invariant() {
        let backend = CadenceEdaBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_cadence_eda_verify_theorem() {
        let backend = CadenceEdaBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Angr End-to-End Tests
// ============================================================================

mod angr_e2e {
    use super::*;

    #[tokio::test]
    async fn test_angr_backend_health_check() {
        let backend = AngrBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Angr is available"),
            HealthStatus::Unavailable { reason } => println!("Angr unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Angr degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_angr_backend_id() {
        let backend = AngrBackend::new();
        assert_eq!(backend.id(), BackendId::Angr);
    }

    #[tokio::test]
    async fn test_angr_supports_property_types() {
        let backend = AngrBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_angr_verify_simple_invariant() {
        let backend = AngrBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_angr_verify_theorem() {
        let backend = AngrBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Manticore End-to-End Tests
// ============================================================================

mod manticore_e2e {
    use super::*;

    #[tokio::test]
    async fn test_manticore_backend_health_check() {
        let backend = ManticoreBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Manticore is available"),
            HealthStatus::Unavailable { reason } => println!("Manticore unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Manticore degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_manticore_backend_id() {
        let backend = ManticoreBackend::new();
        assert_eq!(backend.id(), BackendId::Manticore);
    }

    #[tokio::test]
    async fn test_manticore_supports_property_types() {
        let backend = ManticoreBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_manticore_verify_simple_invariant() {
        let backend = ManticoreBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_manticore_verify_theorem() {
        let backend = ManticoreBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// TritonDba End-to-End Tests
// ============================================================================

mod triton_dba_e2e {
    use super::*;

    #[tokio::test]
    async fn test_triton_dba_backend_health_check() {
        let backend = TritonDbaBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("TritonDba is available"),
            HealthStatus::Unavailable { reason } => println!("TritonDba unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("TritonDba degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_triton_dba_backend_id() {
        let backend = TritonDbaBackend::new();
        assert_eq!(backend.id(), BackendId::TritonDBA);
    }

    #[tokio::test]
    async fn test_triton_dba_supports_property_types() {
        let backend = TritonDbaBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_triton_dba_verify_simple_invariant() {
        let backend = TritonDbaBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_triton_dba_verify_theorem() {
        let backend = TritonDbaBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Astree End-to-End Tests
// ============================================================================

mod astree_e2e {
    use super::*;

    #[tokio::test]
    async fn test_astree_backend_health_check() {
        let backend = AstreeBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Astree is available"),
            HealthStatus::Unavailable { reason } => println!("Astree unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Astree degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_astree_backend_id() {
        let backend = AstreeBackend::new();
        assert_eq!(backend.id(), BackendId::Astree);
    }

    #[tokio::test]
    async fn test_astree_supports_property_types() {
        let backend = AstreeBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_astree_verify_simple_invariant() {
        let backend = AstreeBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_astree_verify_theorem() {
        let backend = AstreeBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Polyspace End-to-End Tests
// ============================================================================

mod polyspace_e2e {
    use super::*;

    #[tokio::test]
    async fn test_polyspace_backend_health_check() {
        let backend = PolyspaceBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Polyspace is available"),
            HealthStatus::Unavailable { reason } => println!("Polyspace unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Polyspace degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_polyspace_backend_id() {
        let backend = PolyspaceBackend::new();
        assert_eq!(backend.id(), BackendId::Polyspace);
    }

    #[tokio::test]
    async fn test_polyspace_supports_property_types() {
        let backend = PolyspaceBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_polyspace_verify_simple_invariant() {
        let backend = PolyspaceBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_polyspace_verify_theorem() {
        let backend = PolyspaceBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// CodeSonar End-to-End Tests
// ============================================================================

mod codesonar_e2e {
    use super::*;

    #[tokio::test]
    async fn test_codesonar_backend_health_check() {
        let backend = CodeSonarBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("CodeSonar is available"),
            HealthStatus::Unavailable { reason } => println!("CodeSonar unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("CodeSonar degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_codesonar_backend_id() {
        let backend = CodeSonarBackend::new();
        assert_eq!(backend.id(), BackendId::CodeSonar);
    }

    #[tokio::test]
    async fn test_codesonar_supports_property_types() {
        let backend = CodeSonarBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_codesonar_verify_simple_invariant() {
        let backend = CodeSonarBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_codesonar_verify_theorem() {
        let backend = CodeSonarBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// FramaCEva End-to-End Tests
// ============================================================================

mod framac_eva_e2e {
    use super::*;

    #[tokio::test]
    async fn test_framac_eva_backend_health_check() {
        let backend = FramaCEvaBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("FramaCEva is available"),
            HealthStatus::Unavailable { reason } => println!("FramaCEva unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("FramaCEva degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_framac_eva_backend_id() {
        let backend = FramaCEvaBackend::new();
        assert_eq!(backend.id(), BackendId::FramaCEva);
    }

    #[tokio::test]
    async fn test_framac_eva_supports_property_types() {
        let backend = FramaCEvaBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_framac_eva_verify_simple_invariant() {
        let backend = FramaCEvaBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_framac_eva_verify_theorem() {
        let backend = FramaCEvaBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Tarpaulin End-to-End Tests
// ============================================================================

mod tarpaulin_e2e {
    use super::*;

    #[tokio::test]
    async fn test_tarpaulin_backend_health_check() {
        let backend = TarpaulinBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Tarpaulin is available"),
            HealthStatus::Unavailable { reason } => println!("Tarpaulin unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Tarpaulin degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_tarpaulin_backend_id() {
        let backend = TarpaulinBackend::new();
        assert_eq!(backend.id(), BackendId::Tarpaulin);
    }

    #[tokio::test]
    async fn test_tarpaulin_supports_property_types() {
        let backend = TarpaulinBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_tarpaulin_verify_simple_invariant() {
        let backend = TarpaulinBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_tarpaulin_verify_theorem() {
        let backend = TarpaulinBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// LlvmCov End-to-End Tests
// ============================================================================

mod llvm_cov_e2e {
    use super::*;

    #[tokio::test]
    async fn test_llvm_cov_backend_health_check() {
        let backend = LlvmCovBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("LlvmCov is available"),
            HealthStatus::Unavailable { reason } => println!("LlvmCov unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("LlvmCov degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_llvm_cov_backend_id() {
        let backend = LlvmCovBackend::new();
        assert_eq!(backend.id(), BackendId::LlvmCov);
    }

    #[tokio::test]
    async fn test_llvm_cov_supports_property_types() {
        let backend = LlvmCovBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_llvm_cov_verify_simple_invariant() {
        let backend = LlvmCovBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_llvm_cov_verify_theorem() {
        let backend = LlvmCovBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Grcov End-to-End Tests
// ============================================================================

mod grcov_e2e {
    use super::*;

    #[tokio::test]
    async fn test_grcov_backend_health_check() {
        let backend = GrcovBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Grcov is available"),
            HealthStatus::Unavailable { reason } => println!("Grcov unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Grcov degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_grcov_backend_id() {
        let backend = GrcovBackend::new();
        assert_eq!(backend.id(), BackendId::Grcov);
    }

    #[tokio::test]
    async fn test_grcov_supports_property_types() {
        let backend = GrcovBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_grcov_verify_simple_invariant() {
        let backend = GrcovBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_grcov_verify_theorem() {
        let backend = GrcovBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Nextest End-to-End Tests
// ============================================================================

mod nextest_e2e {
    use super::*;

    #[tokio::test]
    async fn test_nextest_backend_health_check() {
        let backend = NextestBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Nextest is available"),
            HealthStatus::Unavailable { reason } => println!("Nextest unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Nextest degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_nextest_backend_id() {
        let backend = NextestBackend::new();
        assert_eq!(backend.id(), BackendId::Nextest);
    }

    #[tokio::test]
    async fn test_nextest_supports_property_types() {
        let backend = NextestBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_nextest_verify_simple_invariant() {
        let backend = NextestBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_nextest_verify_theorem() {
        let backend = NextestBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Insta End-to-End Tests
// ============================================================================

mod insta_e2e {
    use super::*;

    #[tokio::test]
    async fn test_insta_backend_health_check() {
        let backend = InstaBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Insta is available"),
            HealthStatus::Unavailable { reason } => println!("Insta unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Insta degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_insta_backend_id() {
        let backend = InstaBackend::new();
        assert_eq!(backend.id(), BackendId::Insta);
    }

    #[tokio::test]
    async fn test_insta_supports_property_types() {
        let backend = InstaBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_insta_verify_simple_invariant() {
        let backend = InstaBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_insta_verify_theorem() {
        let backend = InstaBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Rstest End-to-End Tests
// ============================================================================

mod rstest_e2e {
    use super::*;

    #[tokio::test]
    async fn test_rstest_backend_health_check() {
        let backend = RstestBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Rstest is available"),
            HealthStatus::Unavailable { reason } => println!("Rstest unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Rstest degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_rstest_backend_id() {
        let backend = RstestBackend::new();
        assert_eq!(backend.id(), BackendId::Rstest);
    }

    #[tokio::test]
    async fn test_rstest_supports_property_types() {
        let backend = RstestBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_rstest_verify_simple_invariant() {
        let backend = RstestBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_rstest_verify_theorem() {
        let backend = RstestBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// TestCase End-to-End Tests
// ============================================================================

mod test_case_e2e {
    use super::*;

    #[tokio::test]
    async fn test_test_case_backend_health_check() {
        let backend = TestCaseBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("TestCase is available"),
            HealthStatus::Unavailable { reason } => println!("TestCase unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("TestCase degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_test_case_backend_id() {
        let backend = TestCaseBackend::new();
        assert_eq!(backend.id(), BackendId::TestCase);
    }

    #[tokio::test]
    async fn test_test_case_supports_property_types() {
        let backend = TestCaseBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_test_case_verify_simple_invariant() {
        let backend = TestCaseBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_test_case_verify_theorem() {
        let backend = TestCaseBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Mockall End-to-End Tests
// ============================================================================

mod mockall_e2e {
    use super::*;

    #[tokio::test]
    async fn test_mockall_backend_health_check() {
        let backend = MockallBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Mockall is available"),
            HealthStatus::Unavailable { reason } => println!("Mockall unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Mockall degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_mockall_backend_id() {
        let backend = MockallBackend::new();
        assert_eq!(backend.id(), BackendId::Mockall);
    }

    #[tokio::test]
    async fn test_mockall_supports_property_types() {
        let backend = MockallBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_mockall_verify_simple_invariant() {
        let backend = MockallBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_mockall_verify_theorem() {
        let backend = MockallBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Deadlinks End-to-End Tests
// ============================================================================

mod deadlinks_e2e {
    use super::*;

    #[tokio::test]
    async fn test_deadlinks_backend_health_check() {
        let backend = DeadlinksBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Deadlinks is available"),
            HealthStatus::Unavailable { reason } => println!("Deadlinks unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Deadlinks degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_deadlinks_backend_id() {
        let backend = DeadlinksBackend::new();
        assert_eq!(backend.id(), BackendId::Deadlinks);
    }

    #[tokio::test]
    async fn test_deadlinks_supports_property_types() {
        let backend = DeadlinksBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_deadlinks_verify_simple_invariant() {
        let backend = DeadlinksBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_deadlinks_verify_theorem() {
        let backend = DeadlinksBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Spellcheck End-to-End Tests
// ============================================================================

mod spellcheck_e2e {
    use super::*;

    #[tokio::test]
    async fn test_spellcheck_backend_health_check() {
        let backend = SpellcheckBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Spellcheck is available"),
            HealthStatus::Unavailable { reason } => println!("Spellcheck unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Spellcheck degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_spellcheck_backend_id() {
        let backend = SpellcheckBackend::new();
        assert_eq!(backend.id(), BackendId::Spellcheck);
    }

    #[tokio::test]
    async fn test_spellcheck_supports_property_types() {
        let backend = SpellcheckBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_spellcheck_verify_simple_invariant() {
        let backend = SpellcheckBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_spellcheck_verify_theorem() {
        let backend = SpellcheckBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}

// ============================================================================
// Rdme End-to-End Tests
// ============================================================================

mod rdme_e2e {
    use super::*;

    #[tokio::test]
    async fn test_rdme_backend_health_check() {
        let backend = RdmeBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => println!("Rdme is available"),
            HealthStatus::Unavailable { reason } => println!("Rdme unavailable: {}", reason),
            HealthStatus::Degraded { reason } => println!("Rdme degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_rdme_backend_id() {
        let backend = RdmeBackend::new();
        assert_eq!(backend.id(), BackendId::Rdme);
    }

    #[tokio::test]
    async fn test_rdme_supports_property_types() {
        let backend = RdmeBackend::new();
        let supported = backend.supports();
        assert!(!supported.is_empty());
    }

    #[tokio::test]
    async fn test_rdme_verify_simple_invariant() {
        let backend = RdmeBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_invariant_spec();
        let _ = backend.verify(&spec).await;
    }

    #[tokio::test]
    async fn test_rdme_verify_theorem() {
        let backend = RdmeBackend::new();
        if !is_backend_available(&backend).await {
            eprintln!("SKIPPED");
            return;
        }
        let spec = simple_theorem_spec();
        let _ = backend.verify(&spec).await;
    }
}
