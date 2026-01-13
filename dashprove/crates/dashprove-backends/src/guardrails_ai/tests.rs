//! Tests for GuardrailsAI backend

use super::*;
use crate::traits::VerificationStatus;

fn create_test_spec() -> dashprove_usl::typecheck::TypedSpec {
    use dashprove_usl::ast::Spec;
    use dashprove_usl::typecheck::TypedSpec;
    use std::collections::HashMap;

    TypedSpec {
        spec: Spec::default(),
        type_info: HashMap::new(),
    }
}

#[test]
fn test_backend_id() {
    let backend = GuardrailsAIBackend::new();
    assert_eq!(backend.id(), BackendId::GuardrailsAI);
}

#[test]
fn test_supports_llm_guardrails() {
    let backend = GuardrailsAIBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::LLMGuardrails));
}

#[test]
fn test_default_config() {
    let config = GuardrailsAIConfig::default();
    assert_eq!(config.guardrail_type, GuardrailType::Schema);
    assert_eq!(config.strictness, StrictnessLevel::Standard);
    assert!(config.use_fallbacks);
    assert_eq!(config.max_retries, 3);
    assert!((config.pass_rate_threshold - 0.8).abs() < 0.01);
}

#[test]
fn test_safety_config() {
    let config = GuardrailsAIConfig::safety();
    assert_eq!(config.guardrail_type, GuardrailType::Safety);
    assert_eq!(config.strictness, StrictnessLevel::Strict);
    assert!((config.pass_rate_threshold - 0.95).abs() < 0.01);
}

#[test]
fn test_schema_strict_config() {
    let config = GuardrailsAIConfig::schema_strict();
    assert_eq!(config.guardrail_type, GuardrailType::Schema);
    assert_eq!(config.strictness, StrictnessLevel::Strict);
    assert!(!config.use_fallbacks);
    assert!((config.pass_rate_threshold - 1.0).abs() < 0.01);
}

#[test]
fn test_guardrail_type_strings() {
    assert_eq!(GuardrailType::Schema.as_str(), "schema");
    assert_eq!(GuardrailType::Quality.as_str(), "quality");
    assert_eq!(GuardrailType::Safety.as_str(), "safety");
    assert_eq!(GuardrailType::Factual.as_str(), "factual");
    assert_eq!(GuardrailType::Custom.as_str(), "custom");
}

#[test]
fn test_strictness_strings() {
    assert_eq!(StrictnessLevel::Lenient.as_str(), "lenient");
    assert_eq!(StrictnessLevel::Standard.as_str(), "standard");
    assert_eq!(StrictnessLevel::Strict.as_str(), "strict");
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
GUARDRAILS_RESULT_START
{
    "status": "success",
    "guardrail_type": "schema",
    "strictness": "standard",
    "use_fallbacks": true,
    "passed": 8,
    "failed": 2,
    "total": 10,
    "pass_rate": 0.8,
    "pass_threshold": 0.8,
    "errors": [],
    "duration_s": 1.5
}
GUARDRAILS_RESULT_END
GUARDRAILS_STATUS: VERIFIED
"#;

    let (status, counterexample) = script::parse_guardrails_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(counterexample.is_none());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
GUARDRAILS_RESULT_START
{
    "status": "success",
    "guardrail_type": "schema",
    "strictness": "strict",
    "use_fallbacks": false,
    "passed": 4,
    "failed": 6,
    "total": 10,
    "pass_rate": 0.4,
    "pass_threshold": 0.8,
    "errors": ["Case 2: validation failed", "Case 4: missing field"],
    "duration_s": 1.2
}
GUARDRAILS_RESULT_END
GUARDRAILS_STATUS: NOT_VERIFIED
"#;

    let (status, counterexample) = script::parse_guardrails_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(counterexample.is_some());
    let cx = counterexample.unwrap();
    assert!(cx.raw.is_some());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
GUARDRAILS_RESULT_START
{
    "status": "success",
    "guardrail_type": "quality",
    "strictness": "standard",
    "use_fallbacks": true,
    "passed": 6,
    "failed": 4,
    "total": 10,
    "pass_rate": 0.6,
    "pass_threshold": 0.8,
    "errors": ["Case 1: too short"],
    "duration_s": 0.8
}
GUARDRAILS_RESULT_END
GUARDRAILS_STATUS: PARTIALLY_VERIFIED
"#;

    let (status, counterexample) = script::parse_guardrails_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(counterexample.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "GUARDRAILS_ERROR: Missing dependencies: No module named 'guardrails'";
    let (status, _) = script::parse_guardrails_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[test]
fn test_script_generation() {
    let config = GuardrailsAIConfig::default();
    let spec = create_test_spec();
    let script = script::generate_guardrails_script(&spec, &config);
    assert!(script.is_ok());
    let script_content = script.unwrap();
    assert!(script_content.contains("guardrail_type = \"schema\""));
    assert!(script_content.contains("strictness = \"standard\""));
    assert!(script_content.contains("use_fallbacks = True"));
}

#[tokio::test]
async fn test_guardrails_health_check() {
    let backend = GuardrailsAIBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => (),
        HealthStatus::Unavailable { reason } => {
            assert!(reason.contains("guardrails") || reason.contains("not installed"));
        }
        HealthStatus::Degraded { .. } => (),
    }
}

#[tokio::test]
async fn test_guardrails_verify_returns_result_or_unavailable() {
    let backend = GuardrailsAIBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;
    match result {
        Ok(r) => {
            assert_eq!(r.backend, BackendId::GuardrailsAI);
        }
        Err(BackendError::Unavailable(_)) => (),
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}
