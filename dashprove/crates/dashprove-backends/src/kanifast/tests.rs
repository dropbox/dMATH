//! Tests for Kani Fast backend

use super::*;
use crate::traits::{BackendId, PropertyType, VerificationBackend};

#[test]
fn test_kanifast_backend_id() {
    let backend = KaniFastBackend::new();
    assert_eq!(backend.id(), BackendId::KaniFast);
}

#[test]
fn test_kanifast_supports_contract() {
    let backend = KaniFastBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::Contract));
}

#[test]
fn test_kanifast_supports_invariant() {
    let backend = KaniFastBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::Invariant));
}

#[test]
fn test_kanifast_supports_memory_safety() {
    let backend = KaniFastBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::MemorySafety));
}

#[test]
fn test_kanifast_config_default() {
    let config = KaniFastConfig::default();
    assert!(config.cli_path.is_none());
    assert!(config.project_dir.is_none());
    assert_eq!(config.timeout, std::time::Duration::from_secs(300));
    assert_eq!(config.mode, config::VerificationMode::Bounded);
    assert!(!config.enable_ai);
    assert!(config.enable_explain);
    assert!(!config.enable_lean5);
}

#[test]
fn test_verification_mode_as_flag() {
    use config::VerificationMode;

    assert_eq!(VerificationMode::Bounded.as_flag(), None);
    assert_eq!(VerificationMode::KInduction.as_flag(), Some("--kinduction"));
    assert_eq!(VerificationMode::Chc.as_flag(), Some("--chc"));
    assert_eq!(VerificationMode::Portfolio.as_flag(), Some("--portfolio"));
    assert_eq!(VerificationMode::Auto.as_flag(), Some("--auto"));
}

#[test]
fn test_kanifast_with_config() {
    let config = KaniFastConfig {
        enable_ai: true,
        enable_lean5: true,
        mode: config::VerificationMode::Portfolio,
        ..Default::default()
    };
    let backend = KaniFastBackend::with_config(config);
    assert_eq!(backend.id(), BackendId::KaniFast);
}

mod execution_tests {
    use super::execution;
    use crate::traits::{BackendId, VerificationStatus};
    use std::time::Duration;

    #[test]
    fn test_parse_json_verified() {
        let json = serde_json::json!({
            "status": "verified",
            "proof": "All checks passed"
        });
        let result = execution::parse_json_output(&json, Duration::from_secs(1));
        assert!(matches!(result.status, VerificationStatus::Proven));
        assert_eq!(result.backend, BackendId::KaniFast);
        assert!(result.proof.is_some());
    }

    #[test]
    fn test_parse_json_counterexample() {
        let json = serde_json::json!({
            "status": "counterexample",
            "counterexample": {
                "witness": {
                    "x": 42,
                    "y": true
                },
                "failed_checks": [{
                    "id": "overflow.1",
                    "description": "attempt to add with overflow",
                    "location": {
                        "file": "src/lib.rs",
                        "line": 10,
                        "column": 5
                    },
                    "function": "add_numbers"
                }]
            }
        });
        let result = execution::parse_json_output(&json, Duration::from_secs(2));
        assert!(matches!(result.status, VerificationStatus::Disproven));
        assert!(result.counterexample.is_some());

        let ce = result.counterexample.unwrap();
        assert!(!ce.witness.is_empty());
        assert!(!ce.failed_checks.is_empty());
        assert_eq!(ce.failed_checks[0].check_id, "overflow.1");
    }

    #[test]
    fn test_parse_json_timeout() {
        let json = serde_json::json!({
            "status": "timeout"
        });
        let result = execution::parse_json_output(&json, Duration::from_secs(300));
        assert!(matches!(
            result.status,
            VerificationStatus::Unknown { reason } if reason.contains("timed out")
        ));
    }

    #[test]
    fn test_parse_json_error() {
        let json = serde_json::json!({
            "status": "error",
            "message": "Compilation failed"
        });
        let result = execution::parse_json_output(&json, Duration::from_secs(1));
        assert!(matches!(
            result.status,
            VerificationStatus::Unknown { reason } if reason.contains("Compilation failed")
        ));
    }
}

// Kani proofs for backend behavior
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn proof_kanifast_backend_id_is_kanifast() {
        let backend = KaniFastBackend::new();
        kani::assert(
            matches!(backend.id(), BackendId::KaniFast),
            "Backend ID should be KaniFast",
        );
    }

    #[kani::proof]
    fn proof_kanifast_supports_at_least_one_property() {
        let backend = KaniFastBackend::new();
        let supported = backend.supports();
        kani::assert(
            !supported.is_empty(),
            "KaniFast should support at least one property type",
        );
    }

    #[kani::proof]
    fn proof_kanifast_supports_contract() {
        let backend = KaniFastBackend::new();
        let supported = backend.supports();
        let has_contract = supported
            .iter()
            .any(|p| matches!(p, PropertyType::Contract));
        kani::assert(has_contract, "KaniFast should support Contract property");
    }

    #[kani::proof]
    fn proof_default_backend_equals_new() {
        let backend1 = KaniFastBackend::new();
        let backend2 = KaniFastBackend::default();
        kani::assert(
            backend1.id() == backend2.id(),
            "new() and default() should produce same ID",
        );
    }
}
