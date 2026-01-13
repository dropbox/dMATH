//! Tests for Promptfoo backend

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
    let backend = PromptfooBackend::new();
    assert_eq!(backend.id(), BackendId::Promptfoo);
}

#[test]
fn test_supports_llm_evaluation() {
    let backend = PromptfooBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::LLMEvaluation));
}

#[test]
fn test_default_config() {
    let config = PromptfooConfig::default();
    assert_eq!(config.assertion_type, AssertionType::Contains);
    assert_eq!(config.output_format, OutputFormat::Json);
    assert_eq!(config.iterations, 1);
    assert_eq!(config.max_concurrency, 4);
    assert!((config.pass_rate_threshold - 0.8).abs() < 0.01);
}

#[test]
fn test_strict_config() {
    let config = PromptfooConfig::strict();
    assert_eq!(config.assertion_type, AssertionType::Equals);
    assert!((config.pass_rate_threshold - 1.0).abs() < 0.01);
}

#[test]
fn test_llm_eval_config() {
    let config = PromptfooConfig::llm_eval();
    assert_eq!(config.assertion_type, AssertionType::LlmRubric);
    assert_eq!(config.iterations, 3);
    assert!((config.pass_rate_threshold - 0.85).abs() < 0.01);
}

#[test]
fn test_json_validation_config() {
    let config = PromptfooConfig::json_validation();
    assert_eq!(config.assertion_type, AssertionType::JsonSchema);
    assert!((config.pass_rate_threshold - 0.95).abs() < 0.01);
}

#[test]
fn test_assertion_type_strings() {
    assert_eq!(AssertionType::Contains.as_str(), "contains");
    assert_eq!(AssertionType::Equals.as_str(), "equals");
    assert_eq!(AssertionType::Regex.as_str(), "regex");
    assert_eq!(AssertionType::LlmRubric.as_str(), "llm-rubric");
    assert_eq!(AssertionType::JsonSchema.as_str(), "is-json");
    assert_eq!(AssertionType::Similar.as_str(), "similar");
}

#[test]
fn test_output_format_strings() {
    assert_eq!(OutputFormat::Json.as_str(), "json");
    assert_eq!(OutputFormat::Yaml.as_str(), "yaml");
    assert_eq!(OutputFormat::Csv.as_str(), "csv");
    assert_eq!(OutputFormat::Html.as_str(), "html");
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
PROMPTFOO_RESULT_START
{
    "status": "success",
    "assertion_type": "contains",
    "output_format": "json",
    "iterations": 1,
    "max_concurrency": 4,
    "passed": 7,
    "failed": 1,
    "total": 8,
    "pass_rate": 0.875,
    "pass_threshold": 0.8,
    "errors": [],
    "duration_s": 0.1
}
PROMPTFOO_RESULT_END
PROMPTFOO_STATUS: VERIFIED
"#;

    let (status, counterexample) = script::parse_promptfoo_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(counterexample.is_none());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
PROMPTFOO_RESULT_START
{
    "status": "success",
    "assertion_type": "equals",
    "output_format": "json",
    "iterations": 1,
    "max_concurrency": 4,
    "passed": 3,
    "failed": 5,
    "total": 8,
    "pass_rate": 0.375,
    "pass_threshold": 1.0,
    "errors": ["Case 0: assertion failed", "Case 1: assertion failed"],
    "duration_s": 0.05
}
PROMPTFOO_RESULT_END
PROMPTFOO_STATUS: NOT_VERIFIED
"#;

    let (status, counterexample) = script::parse_promptfoo_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(counterexample.is_some());
    let cx = counterexample.unwrap();
    assert!(cx.raw.is_some());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
PROMPTFOO_RESULT_START
{
    "status": "success",
    "assertion_type": "contains",
    "output_format": "json",
    "iterations": 1,
    "max_concurrency": 4,
    "passed": 5,
    "failed": 3,
    "total": 8,
    "pass_rate": 0.625,
    "pass_threshold": 0.8,
    "errors": ["Case 5: assertion failed"],
    "duration_s": 0.08
}
PROMPTFOO_RESULT_END
PROMPTFOO_STATUS: PARTIALLY_VERIFIED
"#;

    let (status, counterexample) = script::parse_promptfoo_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(counterexample.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "PROMPTFOO_ERROR: Cannot find module 'promptfoo'";
    let (status, _) = script::parse_promptfoo_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[test]
fn test_script_generation() {
    let config = PromptfooConfig::default();
    let spec = create_test_spec();
    let script = script::generate_promptfoo_script(&spec, &config);
    assert!(script.is_ok());
    let script_content = script.unwrap();
    assert!(script_content.contains("assertionType = \"contains\""));
    assert!(script_content.contains("outputFormat = \"json\""));
    assert!(script_content.contains("iterations = 1"));
}

#[tokio::test]
async fn test_promptfoo_health_check() {
    let backend = PromptfooBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => (),
        HealthStatus::Unavailable { reason } => {
            assert!(
                reason.contains("promptfoo")
                    || reason.contains("not installed")
                    || reason.contains("Node")
                    || reason.contains("Promptfoo")
            );
        }
        HealthStatus::Degraded { .. } => (),
    }
}

#[tokio::test]
async fn test_promptfoo_verify_returns_result_or_unavailable() {
    let backend = PromptfooBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;
    match result {
        Ok(r) => {
            assert_eq!(r.backend, BackendId::Promptfoo);
        }
        Err(BackendError::Unavailable(_)) => (),
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}
