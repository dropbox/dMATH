//! Tests for NeMo Guardrails backend

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
    let backend = NeMoGuardrailsBackend::new();
    assert_eq!(backend.id(), BackendId::NeMoGuardrails);
}

#[test]
fn test_supports_llm_guardrails() {
    let backend = NeMoGuardrailsBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::LLMGuardrails));
}

#[test]
fn test_default_config() {
    let config = NeMoGuardrailsConfig::default();
    assert_eq!(config.rail_type, RailType::Output);
    assert_eq!(config.colang_version, ColangVersion::V2);
    assert!(config.jailbreak_detection);
    assert!(!config.topical_rail);
    assert!(!config.fact_checking);
    assert!((config.pass_rate_threshold - 0.85).abs() < 0.01);
}

#[test]
fn test_input_filter_config() {
    let config = NeMoGuardrailsConfig::input_filter();
    assert_eq!(config.rail_type, RailType::Input);
    assert!(config.jailbreak_detection);
    assert!((config.pass_rate_threshold - 0.9).abs() < 0.01);
}

#[test]
fn test_dialog_config() {
    let config = NeMoGuardrailsConfig::dialog();
    assert_eq!(config.rail_type, RailType::Dialog);
    assert!(config.topical_rail);
    assert!((config.pass_rate_threshold - 0.8).abs() < 0.01);
}

#[test]
fn test_fact_checking_config() {
    let config = NeMoGuardrailsConfig::with_fact_checking();
    assert!(config.fact_checking);
    assert!((config.pass_rate_threshold - 0.9).abs() < 0.01);
}

#[test]
fn test_rail_type_strings() {
    assert_eq!(RailType::Input.as_str(), "input");
    assert_eq!(RailType::Output.as_str(), "output");
    assert_eq!(RailType::Dialog.as_str(), "dialog");
    assert_eq!(RailType::Retrieval.as_str(), "retrieval");
}

#[test]
fn test_colang_version_strings() {
    assert_eq!(ColangVersion::V1.as_str(), "1.0");
    assert_eq!(ColangVersion::V2.as_str(), "2.0");
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
NEMO_RESULT_START
{
    "status": "success",
    "rail_type": "output",
    "colang_version": "2.0",
    "jailbreak_detection": true,
    "topical_rail": false,
    "fact_checking": false,
    "passed": 7,
    "failed": 1,
    "total": 8,
    "pass_rate": 0.875,
    "pass_threshold": 0.85,
    "errors": [],
    "duration_s": 0.5
}
NEMO_RESULT_END
NEMO_STATUS: VERIFIED
"#;

    let (status, counterexample) = script::parse_nemo_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(counterexample.is_none());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
NEMO_RESULT_START
{
    "status": "success",
    "rail_type": "input",
    "colang_version": "2.0",
    "jailbreak_detection": true,
    "topical_rail": false,
    "fact_checking": false,
    "passed": 4,
    "failed": 4,
    "total": 8,
    "pass_rate": 0.5,
    "pass_threshold": 0.85,
    "errors": ["Input 3: potential jailbreak detected", "Input 5: potential jailbreak detected"],
    "duration_s": 0.3
}
NEMO_RESULT_END
NEMO_STATUS: NOT_VERIFIED
"#;

    let (status, counterexample) = script::parse_nemo_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(counterexample.is_some());
    let cx = counterexample.unwrap();
    assert!(cx.raw.is_some());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
NEMO_RESULT_START
{
    "status": "success",
    "rail_type": "dialog",
    "colang_version": "2.0",
    "jailbreak_detection": false,
    "topical_rail": true,
    "fact_checking": false,
    "passed": 6,
    "failed": 2,
    "total": 8,
    "pass_rate": 0.75,
    "pass_threshold": 0.85,
    "errors": ["Dialog 2: expected topic_query, got None"],
    "duration_s": 0.4
}
NEMO_RESULT_END
NEMO_STATUS: PARTIALLY_VERIFIED
"#;

    let (status, counterexample) = script::parse_nemo_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(counterexample.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "NEMO_ERROR: Missing dependencies: No module named 'nemoguardrails'";
    let (status, _) = script::parse_nemo_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[test]
fn test_script_generation() {
    let config = NeMoGuardrailsConfig::default();
    let spec = create_test_spec();
    let script = script::generate_nemo_script(&spec, &config);
    assert!(script.is_ok());
    let script_content = script.unwrap();
    assert!(script_content.contains("rail_type = \"output\""));
    assert!(script_content.contains("colang_version = \"2.0\""));
    assert!(script_content.contains("jailbreak_detection = True"));
}

#[tokio::test]
async fn test_nemo_health_check() {
    let backend = NeMoGuardrailsBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => (),
        HealthStatus::Unavailable { reason } => {
            assert!(
                reason.contains("nemoguardrails")
                    || reason.contains("not installed")
                    || reason.contains("NeMo")
            );
        }
        HealthStatus::Degraded { .. } => (),
    }
}

#[tokio::test]
async fn test_nemo_verify_returns_result_or_unavailable() {
    let backend = NeMoGuardrailsBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;
    match result {
        Ok(r) => {
            assert_eq!(r.backend, BackendId::NeMoGuardrails);
        }
        Err(BackendError::Unavailable(_)) => (),
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}
