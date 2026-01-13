//! Property-based tests for dashprove-dispatcher
//!
//! Uses proptest to verify:
//! - MergeStrategy properties
//! - VerificationSummary consistency
//! - DispatcherConfig preset invariants
//! - SelectionStrategy properties
//! - BackendRegistry operations

use dashprove_backends::{BackendId, VerificationStatus};
use dashprove_dispatcher::{
    DispatcherConfig, MergeStrategy, SelectionStrategy, VerificationSummary,
};
use proptest::prelude::*;
use std::time::Duration;

// ============================================================================
// Generators for dispatcher types
// ============================================================================

/// Generate a BackendId
fn backend_id_strategy() -> impl Strategy<Value = BackendId> {
    prop_oneof![
        Just(BackendId::Lean4),
        Just(BackendId::TlaPlus),
        Just(BackendId::Kani),
        Just(BackendId::Alloy),
        Just(BackendId::Isabelle),
        Just(BackendId::Coq),
        Just(BackendId::Dafny),
    ]
}

/// Generate a MergeStrategy
fn merge_strategy_strategy() -> impl Strategy<Value = MergeStrategy> {
    prop_oneof![
        Just(MergeStrategy::FirstSuccess),
        Just(MergeStrategy::Unanimous),
        Just(MergeStrategy::Majority),
        Just(MergeStrategy::MostConfident),
        Just(MergeStrategy::Pessimistic),
        Just(MergeStrategy::Optimistic),
        (0usize..3).prop_map(|f| MergeStrategy::ByzantineFaultTolerant { max_faulty: f }),
        // Generate WeightedConsensus with small weight maps
        proptest::collection::hash_map(backend_id_strategy(), 0.0f64..1.0, 0..3)
            .prop_map(|weights| MergeStrategy::WeightedConsensus { weights }),
    ]
}

/// Generate a SelectionStrategy
fn selection_strategy_strategy() -> impl Strategy<Value = SelectionStrategy> {
    prop_oneof![
        Just(SelectionStrategy::Single),
        Just(SelectionStrategy::All),
        (1usize..5).prop_map(|n| SelectionStrategy::Redundant { min_backends: n }),
        backend_id_strategy().prop_map(SelectionStrategy::Specific),
        (0.0f64..1.0).prop_map(|c| SelectionStrategy::MlBased { min_confidence: c }),
    ]
}

/// Generate a VerificationStatus
fn verification_status_strategy() -> impl Strategy<Value = VerificationStatus> {
    prop_oneof![
        Just(VerificationStatus::Proven),
        Just(VerificationStatus::Disproven),
        "[a-z ]{1,20}"
            .prop_map(|s| s.to_string())
            .prop_map(|reason| VerificationStatus::Unknown { reason }),
        (0.0f64..100.0).prop_map(|pct| VerificationStatus::Partial {
            verified_percentage: pct
        }),
    ]
}

/// Generate a VerificationSummary
fn verification_summary_strategy() -> impl Strategy<Value = VerificationSummary> {
    (
        0usize..100,
        0usize..100,
        0usize..100,
        0usize..100,
        0.0f64..1.0,
    )
        .prop_map(
            |(proven, disproven, unknown, partial, confidence)| VerificationSummary {
                proven,
                disproven,
                unknown,
                partial,
                overall_confidence: confidence,
            },
        )
}

/// Generate a DispatcherConfig
fn dispatcher_config_strategy() -> impl Strategy<Value = DispatcherConfig> {
    (
        selection_strategy_strategy(),
        merge_strategy_strategy(),
        1usize..16,
        1u64..600,
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(
            |(
                selection_strategy,
                merge_strategy,
                max_concurrent,
                timeout_secs,
                check_health,
                auto_update_reputation,
            )| {
                DispatcherConfig {
                    selection_strategy,
                    merge_strategy,
                    max_concurrent,
                    task_timeout: Duration::from_secs(timeout_secs),
                    check_health,
                    auto_update_reputation,
                }
            },
        )
}

// ============================================================================
// Property tests for MergeStrategy
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: MergeStrategy::default() returns FirstSuccess
    #[test]
    fn merge_strategy_default_is_first_success(_dummy in 0..1u8) {
        let default = MergeStrategy::default();
        prop_assert_eq!(default, MergeStrategy::FirstSuccess);
    }

    /// Property: MergeStrategy equality is reflexive
    #[test]
    fn merge_strategy_eq_reflexive(strategy in merge_strategy_strategy()) {
        let cloned = strategy.clone();
        prop_assert_eq!(cloned, strategy);
    }

    /// Property: MergeStrategy clone equals original
    #[test]
    fn merge_strategy_clone_equals(strategy in merge_strategy_strategy()) {
        let cloned = strategy.clone();
        prop_assert_eq!(strategy, cloned);
    }
}

// ============================================================================
// Property tests for SelectionStrategy
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: SelectionStrategy clone equals original
    #[test]
    fn selection_strategy_clone_equals(strategy in selection_strategy_strategy()) {
        let cloned = strategy.clone();
        // Compare by debug representation since we don't have PartialEq for all variants
        prop_assert_eq!(format!("{:?}", strategy), format!("{:?}", cloned));
    }

    /// Property: MlBased confidence is always in [0, 1]
    #[test]
    fn ml_based_confidence_bounded(confidence in -10.0f64..10.0) {
        let config = DispatcherConfig::ml_based(confidence);
        match config.selection_strategy {
            SelectionStrategy::MlBased { min_confidence } => {
                prop_assert!((0.0..=1.0).contains(&min_confidence),
                    "Confidence should be clamped to [0,1], got {}", min_confidence);
            }
            _ => prop_assert!(false, "Expected MlBased strategy"),
        }
    }

    /// Property: Redundant min_backends is preserved
    #[test]
    fn redundant_min_backends_preserved(n in 1usize..10) {
        let config = DispatcherConfig::redundant(n);
        match config.selection_strategy {
            SelectionStrategy::Redundant { min_backends } => {
                prop_assert_eq!(min_backends, n);
            }
            _ => prop_assert!(false, "Expected Redundant strategy"),
        }
    }
}

// ============================================================================
// Property tests for VerificationSummary
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: VerificationSummary overall_confidence is in [0, 1]
    #[test]
    fn summary_confidence_bounded(summary in verification_summary_strategy()) {
        prop_assert!(summary.overall_confidence >= 0.0 && summary.overall_confidence <= 1.0);
    }

    /// Property: Clone equals original
    #[test]
    fn summary_clone_equals(summary in verification_summary_strategy()) {
        let cloned = summary.clone();
        prop_assert_eq!(summary.proven, cloned.proven);
        prop_assert_eq!(summary.disproven, cloned.disproven);
        prop_assert_eq!(summary.unknown, cloned.unknown);
        prop_assert_eq!(summary.partial, cloned.partial);
        prop_assert!((summary.overall_confidence - cloned.overall_confidence).abs() < 1e-10);
    }

    /// Property: Serialization round-trips
    #[test]
    fn summary_serialization_roundtrip(summary in verification_summary_strategy()) {
        let json = serde_json::to_string(&summary).expect("serialize failed");
        let roundtrip: VerificationSummary = serde_json::from_str(&json).expect("deserialize failed");

        prop_assert_eq!(summary.proven, roundtrip.proven);
        prop_assert_eq!(summary.disproven, roundtrip.disproven);
        prop_assert_eq!(summary.unknown, roundtrip.unknown);
        prop_assert_eq!(summary.partial, roundtrip.partial);
    }
}

// ============================================================================
// Property tests for DispatcherConfig
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: DispatcherConfig::default() has sensible values
    #[test]
    fn config_default_sensible(_dummy in 0..1u8) {
        let config = DispatcherConfig::default();

        prop_assert!(config.max_concurrent > 0);
        prop_assert!(config.task_timeout.as_secs() > 0);
        prop_assert!(matches!(config.selection_strategy, SelectionStrategy::Single));
        prop_assert!(matches!(config.merge_strategy, MergeStrategy::FirstSuccess));
    }

    /// Property: Config preset specific() uses correct backend
    #[test]
    fn config_specific_uses_backend(backend in backend_id_strategy()) {
        let config = DispatcherConfig::specific(backend);

        match config.selection_strategy {
            SelectionStrategy::Specific(b) => prop_assert_eq!(b, backend),
            _ => prop_assert!(false, "Expected Specific strategy"),
        }
    }

    /// Property: Config preset all_backends() uses All strategy
    #[test]
    fn config_all_backends_uses_all(_dummy in 0..1u8) {
        let config = DispatcherConfig::all_backends();

        prop_assert!(matches!(config.selection_strategy, SelectionStrategy::All));
        prop_assert!(matches!(config.merge_strategy, MergeStrategy::Majority));
    }

    /// Property: Config preset redundant() uses Redundant strategy
    #[test]
    fn config_redundant_uses_redundant(n in 1usize..10) {
        let config = DispatcherConfig::redundant(n);

        let is_redundant = matches!(
            config.selection_strategy,
            SelectionStrategy::Redundant { min_backends: _ }
        );
        prop_assert!(is_redundant, "Expected Redundant strategy");
        prop_assert!(matches!(config.merge_strategy, MergeStrategy::Unanimous));
    }

    /// Property: Config clone equals original
    #[test]
    fn config_clone_equals(config in dispatcher_config_strategy()) {
        let cloned = config.clone();

        prop_assert_eq!(config.max_concurrent, cloned.max_concurrent);
        prop_assert_eq!(config.task_timeout, cloned.task_timeout);
        prop_assert_eq!(config.check_health, cloned.check_health);
        prop_assert_eq!(format!("{:?}", config.merge_strategy), format!("{:?}", cloned.merge_strategy));
    }
}

// ============================================================================
// Property tests for VerificationStatus
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: VerificationStatus clone equals original
    #[test]
    fn status_clone_equals(status in verification_status_strategy()) {
        let cloned = status.clone();
        prop_assert_eq!(format!("{:?}", status), format!("{:?}", cloned));
    }

    /// Property: Partial verified_percentage is in [0, 100]
    #[test]
    fn partial_percentage_bounded(pct in 0.0f64..100.0) {
        let status = VerificationStatus::Partial {
            verified_percentage: pct,
        };

        match status {
            VerificationStatus::Partial { verified_percentage } => {
                prop_assert!((0.0..=100.0).contains(&verified_percentage));
            }
            _ => prop_assert!(false, "Expected Partial status"),
        }
    }
}

// ============================================================================
// Edge case tests
// ============================================================================

#[test]
fn config_ml_based_clamps_high() {
    let config = DispatcherConfig::ml_based(1.5);
    match config.selection_strategy {
        SelectionStrategy::MlBased { min_confidence } => {
            assert!((min_confidence - 1.0).abs() < 1e-10);
        }
        _ => panic!("Expected MlBased strategy"),
    }
}

#[test]
fn config_ml_based_clamps_low() {
    let config = DispatcherConfig::ml_based(-0.5);
    match config.selection_strategy {
        SelectionStrategy::MlBased { min_confidence } => {
            assert!((min_confidence - 0.0).abs() < 1e-10);
        }
        _ => panic!("Expected MlBased strategy"),
    }
}

#[test]
fn config_default_has_reasonable_timeout() {
    let config = DispatcherConfig::default();
    assert!(config.task_timeout.as_secs() >= 60);
    assert!(config.task_timeout.as_secs() <= 600);
}

#[test]
fn merge_strategies_are_all_distinct() {
    let strategies = [
        MergeStrategy::FirstSuccess,
        MergeStrategy::Unanimous,
        MergeStrategy::Majority,
        MergeStrategy::MostConfident,
        MergeStrategy::Pessimistic,
        MergeStrategy::Optimistic,
    ];

    for i in 0..strategies.len() {
        for j in (i + 1)..strategies.len() {
            assert_ne!(
                strategies[i], strategies[j],
                "Strategies {:?} and {:?} should be different",
                strategies[i], strategies[j]
            );
        }
    }
}

#[test]
fn verification_summary_zero_properties() {
    let summary = VerificationSummary {
        proven: 0,
        disproven: 0,
        unknown: 0,
        partial: 0,
        overall_confidence: 0.0,
    };

    let json = serde_json::to_string(&summary).expect("serialize failed");
    let roundtrip: VerificationSummary = serde_json::from_str(&json).expect("deserialize failed");

    assert_eq!(
        summary.proven + summary.disproven + summary.unknown + summary.partial,
        0
    );
    assert_eq!(
        roundtrip.proven + roundtrip.disproven + roundtrip.unknown + roundtrip.partial,
        0
    );
}

#[test]
fn verification_status_unknown_preserves_reason() {
    let reason = "solver timeout".to_string();
    let status = VerificationStatus::Unknown {
        reason: reason.clone(),
    };

    match status {
        VerificationStatus::Unknown { reason: r } => assert_eq!(r, reason),
        _ => panic!("Expected Unknown status"),
    }
}

#[test]
fn selection_strategy_single_is_default_like() {
    // Single strategy should work similar to default behavior
    let config = DispatcherConfig {
        selection_strategy: SelectionStrategy::Single,
        ..Default::default()
    };

    assert_eq!(
        config.max_concurrent,
        DispatcherConfig::default().max_concurrent
    );
}
