//! Property-based tests for dashprove main library crate using proptest

use dashprove::{
    client::PropertyResult,
    monitor::{MonitorConfig, MonitorTarget, MonitoredProperty, PropertyKind, RuntimeMonitor},
    BackendId, DashProveConfig, VerificationResult, VerificationStatus,
};
use proptest::prelude::*;

// ============================================================================
// Strategy generators
// ============================================================================

fn arbitrary_backend() -> impl Strategy<Value = BackendId> {
    prop_oneof![
        Just(BackendId::Lean4),
        Just(BackendId::TlaPlus),
        Just(BackendId::Kani),
        Just(BackendId::Alloy),
        Just(BackendId::Isabelle),
        Just(BackendId::Coq),
        Just(BackendId::Dafny),
        Just(BackendId::Marabou),
        Just(BackendId::AlphaBetaCrown),
        Just(BackendId::Eran),
        Just(BackendId::Storm),
        Just(BackendId::Prism),
        Just(BackendId::Tamarin),
        Just(BackendId::ProVerif),
        Just(BackendId::Verifpal),
        Just(BackendId::Verus),
        Just(BackendId::Creusot),
        Just(BackendId::Prusti),
        Just(BackendId::Z3),
        Just(BackendId::Cvc5),
    ]
}

fn arbitrary_verification_status() -> impl Strategy<Value = VerificationStatus> {
    prop_oneof![
        Just(VerificationStatus::Proven),
        Just(VerificationStatus::Disproven),
        "[a-z ]{5,30}".prop_map(|reason| VerificationStatus::Unknown { reason }),
    ]
}

fn arbitrary_monitor_target() -> impl Strategy<Value = MonitorTarget> {
    prop_oneof![
        Just(MonitorTarget::Rust),
        Just(MonitorTarget::TypeScript),
        Just(MonitorTarget::Python),
    ]
}

fn arbitrary_property_kind() -> impl Strategy<Value = PropertyKind> {
    prop_oneof![
        Just(PropertyKind::Invariant),
        Just(PropertyKind::Precondition),
        Just(PropertyKind::Postcondition),
        Just(PropertyKind::Temporal),
    ]
}

fn arbitrary_monitor_config() -> impl Strategy<Value = MonitorConfig> {
    (
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        arbitrary_monitor_target(),
    )
        .prop_map(
            |(generate_assertions, generate_logging, generate_metrics, target)| MonitorConfig {
                generate_assertions,
                generate_logging,
                generate_metrics,
                target,
            },
        )
}

fn arbitrary_monitored_property() -> impl Strategy<Value = MonitoredProperty> {
    (
        "[a-z_]{3,20}",
        arbitrary_property_kind(),
        "[a-z0-9_().; ]{10,50}",
    )
        .prop_map(|(name, kind, check_code)| MonitoredProperty {
            name,
            kind,
            check_code,
        })
}

fn arbitrary_runtime_monitor() -> impl Strategy<Value = RuntimeMonitor> {
    (
        "[A-Z][a-z]{3,15}Monitor",
        "[a-z0-9_{}();\\s]{50,200}",
        prop::collection::vec(arbitrary_monitored_property(), 0..5),
        arbitrary_monitor_target(),
    )
        .prop_map(|(name, code, properties, target)| RuntimeMonitor {
            name,
            code,
            properties,
            target,
        })
}

fn arbitrary_property_result() -> impl Strategy<Value = PropertyResult> {
    (
        "[a-z_]{3,20}",
        arbitrary_verification_status(),
        prop::collection::vec(arbitrary_backend(), 0..3),
        prop::option::of("[a-z ]{10,50}"),
        prop::option::of("[a-z ]{10,50}"),
    )
        .prop_map(
            |(name, status, backends_used, proof, counterexample)| PropertyResult {
                name,
                status,
                backends_used,
                proof,
                counterexample,
            },
        )
}

fn arbitrary_verification_result() -> impl Strategy<Value = VerificationResult> {
    (
        arbitrary_verification_status(),
        prop::collection::vec(arbitrary_property_result(), 0..5),
        prop::option::of("[a-z ]{10,50}"),
        prop::option::of("[a-z ]{10,50}"),
        prop::collection::vec("[a-z ]{5,20}", 0..3),
        0.0f64..1.0f64,
    )
        .prop_map(
            |(status, properties, proof, counterexample, suggestions, confidence)| {
                VerificationResult {
                    status,
                    properties,
                    proof,
                    counterexample,
                    suggestions,
                    confidence,
                }
            },
        )
}

// ============================================================================
// DashProveConfig tests
// ============================================================================

#[test]
fn config_default_has_empty_backends() {
    let config = DashProveConfig::default();
    assert!(config.backends.is_empty());
    assert!(!config.learning_enabled);
    assert!(config.api_url.is_none());
    assert!(config.ml_predictor.is_none());
}

#[test]
fn config_with_learning_enables_learning() {
    let config = DashProveConfig::default().with_learning();
    assert!(config.learning_enabled);
}

#[test]
fn config_all_backends_matches_supported_backends() {
    let config = DashProveConfig::all_backends();
    let expected = dashprove::backend_ids::SUPPORTED_BACKENDS.len();
    assert_eq!(config.backends.len(), expected);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn config_with_backend_contains_single_backend(backend in arbitrary_backend()) {
        let config = DashProveConfig::with_backend(backend);
        prop_assert_eq!(config.backends.len(), 1);
        prop_assert_eq!(config.backends[0], backend);
    }

    #[test]
    fn config_remote_sets_url(url in "[a-z]{3,10}://[a-z]{5,15}:[0-9]{2,5}") {
        let config = DashProveConfig::remote(&url);
        prop_assert_eq!(config.api_url, Some(url));
    }
}

// ============================================================================
// VerificationResult tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn verification_result_json_roundtrip(result in arbitrary_verification_result()) {
        let json = serde_json::to_string(&result).unwrap();
        let parsed: VerificationResult = serde_json::from_str(&json).unwrap();

        // Use approximate comparison for f64 due to JSON serialization precision
        prop_assert!((result.confidence - parsed.confidence).abs() < 1e-10);
        prop_assert_eq!(result.proof, parsed.proof);
        prop_assert_eq!(result.counterexample, parsed.counterexample);
        prop_assert_eq!(result.suggestions.len(), parsed.suggestions.len());
        prop_assert_eq!(result.properties.len(), parsed.properties.len());
    }

    #[test]
    fn verification_result_is_proven_matches_status(result in arbitrary_verification_result()) {
        let is_proven = result.is_proven();
        let matches_proven = matches!(result.status, VerificationStatus::Proven);
        prop_assert_eq!(is_proven, matches_proven);
    }

    #[test]
    fn verification_result_is_disproven_matches_status(result in arbitrary_verification_result()) {
        let is_disproven = result.is_disproven();
        let matches_disproven = matches!(result.status, VerificationStatus::Disproven);
        prop_assert_eq!(is_disproven, matches_disproven);
    }

    #[test]
    fn verification_result_proven_count_correct(result in arbitrary_verification_result()) {
        let count = result.proven_count();
        let manual_count = result
            .properties
            .iter()
            .filter(|p| matches!(p.status, VerificationStatus::Proven))
            .count();
        prop_assert_eq!(count, manual_count);
    }

    #[test]
    fn verification_result_disproven_count_correct(result in arbitrary_verification_result()) {
        let count = result.disproven_count();
        let manual_count = result
            .properties
            .iter()
            .filter(|p| matches!(p.status, VerificationStatus::Disproven))
            .count();
        prop_assert_eq!(count, manual_count);
    }

    #[test]
    fn verification_result_confidence_in_range(result in arbitrary_verification_result()) {
        prop_assert!(result.confidence >= 0.0);
        prop_assert!(result.confidence <= 1.0);
    }
}

// ============================================================================
// PropertyResult tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn property_result_json_roundtrip(result in arbitrary_property_result()) {
        let json = serde_json::to_string(&result).unwrap();
        let parsed: PropertyResult = serde_json::from_str(&json).unwrap();

        prop_assert_eq!(result.name, parsed.name);
        prop_assert_eq!(result.proof, parsed.proof);
        prop_assert_eq!(result.counterexample, parsed.counterexample);
        prop_assert_eq!(result.backends_used.len(), parsed.backends_used.len());
    }

    #[test]
    fn property_result_name_not_empty(result in arbitrary_property_result()) {
        prop_assert!(!result.name.is_empty());
    }
}

// ============================================================================
// MonitorConfig tests
// ============================================================================

#[test]
fn monitor_config_default_is_rust_target() {
    let config = MonitorConfig::default();
    assert!(matches!(config.target, MonitorTarget::Rust));
}

#[test]
fn monitor_config_default_no_assertions() {
    let config = MonitorConfig::default();
    assert!(!config.generate_assertions);
    assert!(!config.generate_logging);
    assert!(!config.generate_metrics);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn monitor_config_json_roundtrip(config in arbitrary_monitor_config()) {
        let json = serde_json::to_string(&config).unwrap();
        let parsed: MonitorConfig = serde_json::from_str(&json).unwrap();

        prop_assert_eq!(config.generate_assertions, parsed.generate_assertions);
        prop_assert_eq!(config.generate_logging, parsed.generate_logging);
        prop_assert_eq!(config.generate_metrics, parsed.generate_metrics);
    }
}

// ============================================================================
// MonitorTarget tests
// ============================================================================

#[test]
fn monitor_target_default_is_rust() {
    let target = MonitorTarget::default();
    assert!(matches!(target, MonitorTarget::Rust));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn monitor_target_json_roundtrip(target in arbitrary_monitor_target()) {
        let json = serde_json::to_string(&target).unwrap();
        let parsed: MonitorTarget = serde_json::from_str(&json).unwrap();

        let target_str = format!("{:?}", target);
        let parsed_str = format!("{:?}", parsed);
        prop_assert_eq!(target_str, parsed_str);
    }
}

// ============================================================================
// PropertyKind tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn property_kind_json_roundtrip(kind in arbitrary_property_kind()) {
        let json = serde_json::to_string(&kind).unwrap();
        let parsed: PropertyKind = serde_json::from_str(&json).unwrap();

        let kind_str = format!("{:?}", kind);
        let parsed_str = format!("{:?}", parsed);
        prop_assert_eq!(kind_str, parsed_str);
    }
}

// ============================================================================
// RuntimeMonitor tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn runtime_monitor_json_roundtrip(monitor in arbitrary_runtime_monitor()) {
        let json = serde_json::to_string(&monitor).unwrap();
        let parsed: RuntimeMonitor = serde_json::from_str(&json).unwrap();

        prop_assert_eq!(monitor.name, parsed.name);
        prop_assert_eq!(monitor.code, parsed.code);
        prop_assert_eq!(monitor.properties.len(), parsed.properties.len());
    }

    #[test]
    fn runtime_monitor_property_count_matches_len(monitor in arbitrary_runtime_monitor()) {
        prop_assert_eq!(monitor.property_count(), monitor.properties.len());
    }

    #[test]
    fn runtime_monitor_name_not_empty(monitor in arbitrary_runtime_monitor()) {
        prop_assert!(!monitor.name.is_empty());
    }
}

// ============================================================================
// MonitoredProperty tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn monitored_property_json_roundtrip(prop in arbitrary_monitored_property()) {
        let json = serde_json::to_string(&prop).unwrap();
        let parsed: MonitoredProperty = serde_json::from_str(&json).unwrap();

        prop_assert_eq!(prop.name, parsed.name);
        prop_assert_eq!(prop.check_code, parsed.check_code);
    }

    #[test]
    fn monitored_property_name_not_empty(prop in arbitrary_monitored_property()) {
        prop_assert!(!prop.name.is_empty());
    }
}

// ============================================================================
// VerificationStatus tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn verification_status_json_roundtrip(status in arbitrary_verification_status()) {
        let json = serde_json::to_string(&status).unwrap();
        let parsed: VerificationStatus = serde_json::from_str(&json).unwrap();

        // Compare via debug format since VerificationStatus may not implement PartialEq
        let status_str = format!("{:?}", status);
        let parsed_str = format!("{:?}", parsed);
        prop_assert_eq!(status_str, parsed_str);
    }

    #[test]
    fn verification_status_proven_disproven_mutually_exclusive(status in arbitrary_verification_status()) {
        let is_proven = matches!(status, VerificationStatus::Proven);
        let is_disproven = matches!(status, VerificationStatus::Disproven);
        // Can't be both proven and disproven
        prop_assert!(!(is_proven && is_disproven));
    }
}

// ============================================================================
// BackendId tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn backend_id_json_roundtrip(backend in arbitrary_backend()) {
        let json = serde_json::to_string(&backend).unwrap();
        let parsed: BackendId = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(backend, parsed);
    }

    #[test]
    fn backend_id_debug_not_empty(backend in arbitrary_backend()) {
        let debug = format!("{:?}", backend);
        prop_assert!(!debug.is_empty());
    }
}

// ============================================================================
// Integration property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn config_builder_chain_preserves_backend(backend in arbitrary_backend()) {
        let config = DashProveConfig::with_backend(backend).with_learning();
        prop_assert!(config.learning_enabled);
        prop_assert_eq!(config.backends.len(), 1);
        prop_assert_eq!(config.backends[0], backend);
    }
}
