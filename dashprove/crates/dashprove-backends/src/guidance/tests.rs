//! Tests for Guidance backend

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
    let backend = GuidanceBackend::new();
    assert_eq!(backend.id(), BackendId::Guidance);
}

#[test]
fn test_supports_llm_guardrails() {
    let backend = GuidanceBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::LLMGuardrails));
}

#[test]
fn test_default_config() {
    let config = GuidanceConfig::default();
    assert_eq!(config.generation_mode, GenerationMode::Constrained);
    assert_eq!(config.validation_mode, ValidationMode::Structure);
    assert!(!config.allow_partial);
    assert_eq!(config.max_tokens, 512);
    assert!((config.pass_rate_threshold - 0.85).abs() < 0.01);
}

#[test]
fn test_json_schema_config() {
    let config = GuidanceConfig::json_schema();
    assert_eq!(config.generation_mode, GenerationMode::JsonSchema);
    assert_eq!(config.validation_mode, ValidationMode::TypeChecked);
    assert!((config.pass_rate_threshold - 0.95).abs() < 0.01);
}

#[test]
fn test_grammar_config() {
    let config = GuidanceConfig::grammar();
    assert_eq!(config.generation_mode, GenerationMode::Grammar);
    assert!(config.allow_partial);
    assert!((config.pass_rate_threshold - 0.8).abs() < 0.01);
}

#[test]
fn test_semantic_strict_config() {
    let config = GuidanceConfig::semantic_strict();
    assert_eq!(config.validation_mode, ValidationMode::Semantic);
    assert!(!config.allow_partial);
    assert!((config.pass_rate_threshold - 1.0).abs() < 0.01);
}

#[test]
fn test_generation_mode_strings() {
    assert_eq!(GenerationMode::Constrained.as_str(), "constrained");
    assert_eq!(GenerationMode::Grammar.as_str(), "grammar");
    assert_eq!(GenerationMode::Regex.as_str(), "regex");
    assert_eq!(GenerationMode::JsonSchema.as_str(), "json_schema");
}

#[test]
fn test_validation_mode_strings() {
    assert_eq!(ValidationMode::Structure.as_str(), "structure");
    assert_eq!(ValidationMode::TypeChecked.as_str(), "type_checked");
    assert_eq!(ValidationMode::Semantic.as_str(), "semantic");
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
GUIDANCE_RESULT_START
{
    "status": "success",
    "generation_mode": "constrained",
    "validation_mode": "structure",
    "allow_partial": false,
    "max_tokens": 512,
    "passed": 7,
    "failed": 1,
    "total": 8,
    "pass_rate": 0.875,
    "pass_threshold": 0.85,
    "errors": [],
    "duration_s": 0.3
}
GUIDANCE_RESULT_END
GUIDANCE_STATUS: VERIFIED
"#;

    let (status, counterexample) = script::parse_guidance_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(counterexample.is_none());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
GUIDANCE_RESULT_START
{
    "status": "success",
    "generation_mode": "json_schema",
    "validation_mode": "type_checked",
    "allow_partial": false,
    "max_tokens": 512,
    "passed": 4,
    "failed": 4,
    "total": 8,
    "pass_rate": 0.5,
    "pass_threshold": 0.95,
    "errors": ["Case 2: JSON schema validation failed", "Case 3: JSON schema validation failed"],
    "duration_s": 0.2
}
GUIDANCE_RESULT_END
GUIDANCE_STATUS: NOT_VERIFIED
"#;

    let (status, counterexample) = script::parse_guidance_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(counterexample.is_some());
    let cx = counterexample.unwrap();
    assert!(cx.raw.is_some());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
GUIDANCE_RESULT_START
{
    "status": "success",
    "generation_mode": "grammar",
    "validation_mode": "structure",
    "allow_partial": true,
    "max_tokens": 512,
    "passed": 6,
    "failed": 2,
    "total": 8,
    "pass_rate": 0.75,
    "pass_threshold": 0.85,
    "errors": ["Case 3: expected True, got False"],
    "duration_s": 0.25
}
GUIDANCE_RESULT_END
GUIDANCE_STATUS: PARTIALLY_VERIFIED
"#;

    let (status, counterexample) = script::parse_guidance_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(counterexample.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "GUIDANCE_ERROR: Missing dependencies: No module named 'guidance'";
    let (status, _) = script::parse_guidance_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[test]
fn test_script_generation() {
    let config = GuidanceConfig::default();
    let spec = create_test_spec();
    let script = script::generate_guidance_script(&spec, &config);
    assert!(script.is_ok());
    let script_content = script.unwrap();
    assert!(script_content.contains("generation_mode = \"constrained\""));
    assert!(script_content.contains("validation_mode = \"structure\""));
    assert!(script_content.contains("allow_partial = False"));
}

#[tokio::test]
async fn test_guidance_health_check() {
    let backend = GuidanceBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => (),
        HealthStatus::Unavailable { reason } => {
            assert!(
                reason.contains("guidance")
                    || reason.contains("not installed")
                    || reason.contains("Guidance")
            );
        }
        HealthStatus::Degraded { .. } => (),
    }
}

#[tokio::test]
async fn test_guidance_verify_returns_result_or_unavailable() {
    let backend = GuidanceBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;
    match result {
        Ok(r) => {
            assert_eq!(r.backend, BackendId::Guidance);
        }
        Err(BackendError::Unavailable(_)) => (),
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}
