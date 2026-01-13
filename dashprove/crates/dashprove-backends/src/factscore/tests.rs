//! Tests for FactScore backend

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
    let backend = FactScoreBackend::new();
    assert_eq!(backend.id(), BackendId::FactScore);
}

#[test]
fn test_supports_hallucination_detection() {
    let backend = FactScoreBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::HallucinationDetection));
}

#[test]
fn test_default_config() {
    let config = FactScoreConfig::default();
    assert_eq!(config.knowledge_source, KnowledgeSource::Wikipedia);
    assert_eq!(config.extraction_method, ExtractionMethod::Sentence);
    assert!(config.atomic_facts);
    assert!((config.confidence_threshold - 0.7).abs() < 0.01);
    assert!((config.pass_rate_threshold - 0.8).abs() < 0.01);
}

#[test]
fn test_claim_level_config() {
    let config = FactScoreConfig::claim_level();
    assert_eq!(config.extraction_method, ExtractionMethod::Claim);
    assert!(config.atomic_facts);
    assert!((config.confidence_threshold - 0.75).abs() < 0.01);
    assert!((config.pass_rate_threshold - 0.85).abs() < 0.01);
}

#[test]
fn test_strict_config() {
    let config = FactScoreConfig::strict();
    assert_eq!(config.extraction_method, ExtractionMethod::Claim);
    assert!((config.confidence_threshold - 0.9).abs() < 0.01);
    assert!((config.pass_rate_threshold - 0.95).abs() < 0.01);
}

#[test]
fn test_entity_focused_config() {
    let config = FactScoreConfig::entity_focused();
    assert_eq!(config.extraction_method, ExtractionMethod::Entity);
    assert!((config.confidence_threshold - 0.8).abs() < 0.01);
}

#[test]
fn test_knowledge_graph_config() {
    let config = FactScoreConfig::knowledge_graph();
    assert_eq!(config.extraction_method, ExtractionMethod::Triple);
    assert!(!config.atomic_facts);
    assert!((config.confidence_threshold - 0.75).abs() < 0.01);
}

#[test]
fn test_knowledge_source_strings() {
    assert_eq!(KnowledgeSource::Wikipedia.as_str(), "wikipedia");
    assert_eq!(KnowledgeSource::Custom.as_str(), "custom");
    assert_eq!(KnowledgeSource::WebSearch.as_str(), "web_search");
    assert_eq!(KnowledgeSource::Retrieved.as_str(), "retrieved");
}

#[test]
fn test_extraction_method_strings() {
    assert_eq!(ExtractionMethod::Sentence.as_str(), "sentence");
    assert_eq!(ExtractionMethod::Claim.as_str(), "claim");
    assert_eq!(ExtractionMethod::Entity.as_str(), "entity");
    assert_eq!(ExtractionMethod::Triple.as_str(), "triple");
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
FACTSCORE_RESULT_START
{
    "status": "success",
    "knowledge_source": "wikipedia",
    "extraction_method": "sentence",
    "atomic_facts": true,
    "confidence_threshold": 0.7,
    "passed": 7,
    "failed": 1,
    "total": 8,
    "pass_rate": 0.875,
    "pass_threshold": 0.8,
    "errors": [],
    "duration_s": 0.5
}
FACTSCORE_RESULT_END
FACTSCORE_STATUS: VERIFIED
"#;

    let (status, counterexample) = script::parse_factscore_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(counterexample.is_none());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
FACTSCORE_RESULT_START
{
    "status": "success",
    "knowledge_source": "wikipedia",
    "extraction_method": "claim",
    "atomic_facts": true,
    "confidence_threshold": 0.9,
    "passed": 4,
    "failed": 4,
    "total": 8,
    "pass_rate": 0.5,
    "pass_threshold": 0.95,
    "errors": ["Case 3: claim 'Mars is the closest...' confidence 0.15"],
    "duration_s": 0.4
}
FACTSCORE_RESULT_END
FACTSCORE_STATUS: NOT_VERIFIED
"#;

    let (status, counterexample) = script::parse_factscore_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(counterexample.is_some());
    let cx = counterexample.unwrap();
    assert!(cx.raw.is_some());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
FACTSCORE_RESULT_START
{
    "status": "success",
    "knowledge_source": "wikipedia",
    "extraction_method": "entity",
    "atomic_facts": true,
    "confidence_threshold": 0.8,
    "passed": 6,
    "failed": 2,
    "total": 8,
    "pass_rate": 0.75,
    "pass_threshold": 0.85,
    "errors": ["Case 6: (Great Wall, location, visible from space) confidence 0.30 < 0.8"],
    "duration_s": 0.45
}
FACTSCORE_RESULT_END
FACTSCORE_STATUS: PARTIALLY_VERIFIED
"#;

    let (status, counterexample) = script::parse_factscore_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(counterexample.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "FACTSCORE_ERROR: Missing dependencies: No module named 'factscore'";
    let (status, _) = script::parse_factscore_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[test]
fn test_script_generation() {
    let config = FactScoreConfig::default();
    let spec = create_test_spec();
    let script = script::generate_factscore_script(&spec, &config);
    assert!(script.is_ok());
    let script_content = script.unwrap();
    assert!(script_content.contains("knowledge_source = \"wikipedia\""));
    assert!(script_content.contains("extraction_method = \"sentence\""));
    assert!(script_content.contains("atomic_facts = True"));
}

#[tokio::test]
async fn test_factscore_health_check() {
    let backend = FactScoreBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => (),
        HealthStatus::Unavailable { reason } => {
            assert!(
                reason.contains("factscore")
                    || reason.contains("not installed")
                    || reason.contains("FactScore")
            );
        }
        HealthStatus::Degraded { .. } => (),
    }
}

#[tokio::test]
async fn test_factscore_verify_returns_result_or_unavailable() {
    let backend = FactScoreBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;
    match result {
        Ok(r) => {
            assert_eq!(r.backend, BackendId::FactScore);
        }
        Err(BackendError::Unavailable(_)) => (),
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}
