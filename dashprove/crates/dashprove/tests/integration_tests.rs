//! Integration tests for dashprove library crate
//!
//! These tests verify the public API works correctly end-to-end.

use dashprove::{
    DashProve, DashProveConfig, DashProveError, MonitorConfig, RuntimeMonitor, VerificationResult,
};
use dashprove_backends::BackendId;

// =============================================================================
// DashProveConfig tests
// =============================================================================

#[test]
fn test_config_default_creates_empty_backends_list() {
    let config = DashProveConfig::default();
    assert!(config.backends.is_empty());
    assert!(!config.learning_enabled);
    assert!(config.api_url.is_none());
}

#[test]
fn test_config_all_backends_matches_supported_backends() {
    let config = DashProveConfig::all_backends();
    let expected = dashprove::backend_ids::SUPPORTED_BACKENDS.len();
    assert_eq!(
        config.backends.len(),
        expected,
        "Expected {} backends, got {}",
        expected,
        config.backends.len()
    );

    // Verify specific backend categories are included
    assert!(config.backends.contains(&BackendId::Lean4));
    assert!(config.backends.contains(&BackendId::TlaPlus));
    assert!(config.backends.contains(&BackendId::Kani));
    assert!(config.backends.contains(&BackendId::Z3));
    assert!(config.backends.contains(&BackendId::Storm));
    assert!(config.backends.contains(&BackendId::Marabou));
    assert!(config.backends.contains(&BackendId::Tamarin));
    assert!(config.backends.contains(&BackendId::PlatformApi));
}

#[test]
fn test_config_with_backend_creates_single_backend_config() {
    let config = DashProveConfig::with_backend(BackendId::Lean4);
    assert_eq!(config.backends.len(), 1);
    assert_eq!(config.backends[0], BackendId::Lean4);
}

#[test]
fn test_config_with_learning_enables_learning() {
    let config = DashProveConfig::default().with_learning();
    assert!(config.learning_enabled);
}

#[test]
fn test_config_remote_sets_api_url() {
    let config = DashProveConfig::remote("http://localhost:3000");
    assert_eq!(config.api_url, Some("http://localhost:3000".to_string()));
}

// =============================================================================
// DashProve client tests
// =============================================================================

#[test]
fn test_client_creation_with_default_config() {
    let client = DashProve::new(DashProveConfig::default());
    // Auto-detect mode should register some backends
    assert!(!client.backends().is_empty());
}

#[test]
fn test_client_default_client_method() {
    let client = DashProve::default_client();
    assert!(!client.backends().is_empty());
}

#[test]
fn test_client_with_specific_backend() {
    let config = DashProveConfig::with_backend(BackendId::Lean4);
    let client = DashProve::new(config);
    assert!(client.backends().contains(&BackendId::Lean4));
}

#[test]
fn test_client_config_accessor() {
    let original_config = DashProveConfig::default().with_learning();
    let client = DashProve::new(original_config);
    assert!(client.config().learning_enabled);
}

// =============================================================================
// VerificationResult tests
// =============================================================================

#[test]
fn test_verification_result_is_proven() {
    let result = VerificationResult {
        status: dashprove_backends::VerificationStatus::Proven,
        properties: vec![],
        proof: None,
        counterexample: None,
        suggestions: vec![],
        confidence: 1.0,
    };
    assert!(result.is_proven());
    assert!(!result.is_disproven());
}

#[test]
fn test_verification_result_is_disproven() {
    let result = VerificationResult {
        status: dashprove_backends::VerificationStatus::Disproven,
        properties: vec![],
        proof: None,
        counterexample: Some("x = 5".to_string()),
        suggestions: vec![],
        confidence: 1.0,
    };
    assert!(!result.is_proven());
    assert!(result.is_disproven());
}

#[test]
fn test_verification_result_counting() {
    use dashprove::client::PropertyResult;
    use dashprove_backends::VerificationStatus;

    let result = VerificationResult {
        status: VerificationStatus::Unknown {
            reason: "mixed".to_string(),
        },
        properties: vec![
            PropertyResult {
                name: "p1".to_string(),
                status: VerificationStatus::Proven,
                backends_used: vec![],
                proof: None,
                counterexample: None,
            },
            PropertyResult {
                name: "p2".to_string(),
                status: VerificationStatus::Proven,
                backends_used: vec![],
                proof: None,
                counterexample: None,
            },
            PropertyResult {
                name: "p3".to_string(),
                status: VerificationStatus::Disproven,
                backends_used: vec![],
                proof: None,
                counterexample: None,
            },
        ],
        proof: None,
        counterexample: None,
        suggestions: vec![],
        confidence: 0.66,
    };

    assert_eq!(result.proven_count(), 2);
    assert_eq!(result.disproven_count(), 1);
}

// =============================================================================
// RuntimeMonitor integration tests
// =============================================================================

#[test]
fn test_monitor_from_spec_rust_target() {
    use dashprove_usl::{parse, typecheck};

    let spec = parse("theorem test { forall x: Bool . x or not x }").unwrap();
    let typed = typecheck(spec).unwrap();

    let config = MonitorConfig::default();
    let monitor = RuntimeMonitor::from_spec(&typed, &config);

    assert_eq!(monitor.name, "test");
    assert_eq!(monitor.property_count(), 1);
    assert!(monitor.code.contains("TestMonitor"));
    assert!(monitor.code.contains("check_test"));
    assert!(monitor.code.contains("[false, true]"));
}

#[test]
fn test_monitor_from_spec_typescript_target() {
    use dashprove::monitor::MonitorTarget;
    use dashprove_usl::{parse, typecheck};

    let spec = parse("theorem my_property { true }").unwrap();
    let typed = typecheck(spec).unwrap();

    let config = MonitorConfig {
        target: MonitorTarget::TypeScript,
        ..Default::default()
    };
    let monitor = RuntimeMonitor::from_spec(&typed, &config);

    assert!(monitor.code.contains("export class MyPropertyMonitor"));
    assert!(monitor.code.contains("checkMyProperty()"));
    assert!(monitor.code.contains("checkAll()"));
}

#[test]
fn test_monitor_from_spec_python_target() {
    use dashprove::monitor::MonitorTarget;
    use dashprove_usl::{parse, typecheck};

    let spec = parse("theorem example { true }").unwrap();
    let typed = typecheck(spec).unwrap();

    let config = MonitorConfig {
        target: MonitorTarget::Python,
        ..Default::default()
    };
    let monitor = RuntimeMonitor::from_spec(&typed, &config);

    assert!(monitor.code.contains("class ExampleMonitor"));
    assert!(monitor.code.contains("def check_example(self)"));
    assert!(monitor.code.contains("def check_all(self)"));
}

#[test]
fn test_monitor_with_assertions() {
    use dashprove_usl::{parse, typecheck};

    let spec = parse("theorem safe { true }").unwrap();
    let typed = typecheck(spec).unwrap();

    let config = MonitorConfig {
        generate_assertions: true,
        ..Default::default()
    };
    let monitor = RuntimeMonitor::from_spec(&typed, &config);

    // Assertions should influence the check_code in properties
    assert!(monitor
        .properties
        .iter()
        .any(|p| p.check_code.contains("assert")));
}

#[test]
fn test_monitor_with_logging() {
    use dashprove_usl::{parse, typecheck};

    let spec = parse("theorem logged { true }").unwrap();
    let typed = typecheck(spec).unwrap();

    let config = MonitorConfig {
        generate_logging: true,
        ..Default::default()
    };
    let monitor = RuntimeMonitor::from_spec(&typed, &config);

    // Logging should be imported
    assert!(monitor.code.contains("tracing"));
}

#[test]
fn test_monitor_multiple_properties() {
    use dashprove_usl::{parse, typecheck};

    let spec = parse(
        r#"
        theorem prop1 { true }
        theorem prop2 { false implies true }
        invariant inv1 { true }
    "#,
    )
    .unwrap();
    let typed = typecheck(spec).unwrap();

    let config = MonitorConfig::default();
    let monitor = RuntimeMonitor::from_spec(&typed, &config);

    assert_eq!(monitor.property_count(), 3);
    assert!(monitor.code.contains("check_prop1"));
    assert!(monitor.code.contains("check_prop2"));
    assert!(monitor.code.contains("check_inv1"));
}

#[test]
fn test_monitor_contract_generates_requires_ensures() {
    use dashprove_usl::{parse, typecheck};

    let spec = parse(
        r#"
        contract divide(x: Int, y: Int) -> Result<Int> {
            requires { y != 0 }
            ensures { result >= 0 implies result * y <= x }
        }
    "#,
    )
    .unwrap();
    let typed = typecheck(spec).unwrap();

    let config = MonitorConfig::default();
    let monitor = RuntimeMonitor::from_spec(&typed, &config);

    assert!(monitor.code.contains("check_divide_requires"));
    assert!(monitor.code.contains("check_divide_ensures"));
    assert!(monitor.code.contains("(y != 0)"));
}

// =============================================================================
// Error handling tests
// =============================================================================

#[tokio::test]
async fn test_verify_invalid_spec_returns_parse_error() {
    let mut client = DashProve::default_client();
    let result = client.verify("this is not valid USL").await;

    assert!(matches!(result, Err(DashProveError::Parse(_))));
}

#[tokio::test]
async fn test_verify_type_error() {
    let mut client = DashProve::default_client();
    // Type error: adding int and string
    let result = client.verify("theorem test { 1 + \"string\" }").await;

    // This should fail at type checking with a type error
    assert!(matches!(result, Err(DashProveError::Type(_))));
}

#[tokio::test]
async fn test_verify_code_invalid_spec_returns_parse_error() {
    let mut client = DashProve::default_client();
    let code = "pub fn test() {}";
    let invalid_spec = "not valid USL";

    let result = client.verify_code(code, invalid_spec).await;

    assert!(matches!(result, Err(DashProveError::Parse(_))));
}

// =============================================================================
// Re-export verification tests
// =============================================================================

#[test]
fn test_reexports_available() {
    // Verify that key types are re-exported at crate root
    let _backend_id: BackendId = BackendId::Lean4;
    let _status: dashprove::VerificationStatus = dashprove::VerificationStatus::Proven;
}

#[test]
fn test_sub_crate_access() {
    // Verify that sub-crates are accessible
    let _backend_id = dashprove::backends::BackendId::TlaPlus;
    let _parse_fn = dashprove::usl::parse;
}

// =============================================================================
// USL parsing and type checking through library
// =============================================================================

#[test]
fn test_parse_and_typecheck_through_reexports() {
    use dashprove::usl::{parse, typecheck};

    let spec_source = r#"
        theorem test1 { forall x: Bool . x or not x }
        theorem test2 { exists x: Bool . x }
        invariant safety { true }
    "#;

    let parsed = parse(spec_source).expect("parsing failed");
    assert_eq!(parsed.properties.len(), 3);

    let typed = typecheck(parsed).expect("type checking failed");
    assert_eq!(typed.spec.properties.len(), 3);
}

#[test]
fn test_parse_complex_expressions() {
    use dashprove::usl::{parse, typecheck};

    let spec_source = r#"
        theorem arithmetic {
            forall x: Int . forall y: Int .
                (x > 0 and y > 0) implies (x + y > 0)
        }
    "#;

    let parsed = parse(spec_source).expect("parsing failed");
    let typed = typecheck(parsed).expect("type checking failed");
    assert_eq!(typed.spec.properties.len(), 1);
}

#[test]
fn test_parse_contracts() {
    use dashprove::usl::{parse, typecheck};

    let spec_source = r#"
        contract MyType::method(self: MyType, value: Int) -> Result<Int> {
            requires { value >= 0 }
            ensures { result >= value }
            ensures_err { value < 0 }
        }
    "#;

    let parsed = parse(spec_source).expect("parsing failed");
    let typed = typecheck(parsed).expect("type checking failed");
    assert_eq!(typed.spec.properties.len(), 1);
}

// =============================================================================
// Async verification tests (basic API flow)
// =============================================================================

#[tokio::test]
async fn test_verify_valid_spec() {
    let mut client = DashProve::default_client();

    // Simple tautology that should parse and type-check
    let result = client
        .verify("theorem excluded_middle { forall x: Bool . x or not x }")
        .await;

    // Either succeeds or fails due to backend unavailability
    // But should not fail at parse/type-check
    match result {
        Ok(_) => {} // Great
        Err(DashProveError::Parse(e)) => panic!("Unexpected parse error: {:?}", e),
        Err(DashProveError::Type(e)) => panic!("Unexpected type error: {:?}", e),
        Err(_) => {} // Backend/Dispatcher error is acceptable
    }
}

#[tokio::test]
async fn test_verify_with_specific_backend() {
    let mut client = DashProve::default_client();

    let result = client
        .verify_with_backend("theorem test { true }", BackendId::Lean4)
        .await;

    // Either succeeds or fails due to backend unavailability
    match result {
        Ok(_) => {}
        Err(DashProveError::Parse(e)) => panic!("Unexpected parse error: {:?}", e),
        Err(DashProveError::Type(e)) => panic!("Unexpected type error: {:?}", e),
        Err(_) => {} // Backend error is acceptable
    }
}

// =============================================================================
// ML integration tests
// =============================================================================

#[test]
fn test_ml_strategy_config() {
    use dashprove_ai::StrategyPredictor;

    let predictor = StrategyPredictor::new();
    let config = DashProveConfig::default().with_ml_strategy(predictor, 0.5);

    assert!(config.ml_predictor.is_some());
    assert!((config.ml_min_confidence - 0.5).abs() < f64::EPSILON);
}

#[test]
fn test_set_ml_predictor_after_construction() {
    use dashprove_ai::StrategyPredictor;

    let mut client = DashProve::default_client();
    let predictor = StrategyPredictor::new();

    // Initially no ML predictor
    assert!(client.config().ml_predictor.is_none());

    // Set ML predictor
    client.set_ml_predictor(predictor, 0.7);

    // Now should have ML predictor
    assert!(client.config().ml_predictor.is_some());
    assert!((client.config().ml_min_confidence - 0.7).abs() < f64::EPSILON);
}

// =============================================================================
// Health check tests
// =============================================================================

#[tokio::test]
async fn test_check_health_returns_results() {
    let mut client = DashProve::default_client();
    let health = client.check_health().await;

    // Should return at least one backend health result
    assert!(!health.is_empty());

    // Each result should be a tuple of (BackendId, bool)
    for (backend_id, _healthy) in health {
        // Just verify the backend_id is valid (not panic)
        let _ = format!("{:?}", backend_id);
    }
}
