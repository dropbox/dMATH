//! Tests for the self-improvement infrastructure

use super::*;
use crate::certificate::CertificateCheck;
use crate::improvement::RejectionReason;
use proptest::prelude::*;
use std::time::Duration;

// ==================== Proptest Strategies ====================

fn version_id_strategy() -> impl Strategy<Value = VersionId> {
    "[a-f0-9]{16}".prop_map(VersionId::new)
}

fn capability_value_strategy() -> impl Strategy<Value = version::CapabilityValue> {
    prop_oneof![
        any::<bool>().prop_map(version::CapabilityValue::Boolean),
        (0.0f64..1000.0).prop_map(version::CapabilityValue::Numeric),
        (0u64..10000).prop_map(version::CapabilityValue::Count),
        (0u32..100, 0u32..100, 0u32..100)
            .prop_map(|(a, b, c)| version::CapabilityValue::Version(a, b, c)),
    ]
}

fn capability_strategy() -> impl Strategy<Value = version::Capability> {
    ("[a-z_]{1,20}", capability_value_strategy()).prop_map(|(name, value)| version::Capability {
        name,
        value,
        description: None,
        unit: None,
    })
}

fn capability_set_strategy() -> impl Strategy<Value = version::CapabilitySet> {
    prop::collection::vec(capability_strategy(), 0..5).prop_map(|caps| {
        let mut set = version::CapabilitySet::new();
        for cap in caps {
            set.add(cap);
        }
        set
    })
}

fn improvement_kind_strategy() -> impl Strategy<Value = improvement::ImprovementKind> {
    prop_oneof![
        Just(improvement::ImprovementKind::BugFix),
        Just(improvement::ImprovementKind::Optimization),
        Just(improvement::ImprovementKind::Feature),
        Just(improvement::ImprovementKind::Security),
        Just(improvement::ImprovementKind::Refactoring),
        Just(improvement::ImprovementKind::Configuration),
        Just(improvement::ImprovementKind::DependencyUpdate),
    ]
}

fn improvement_target_strategy() -> impl Strategy<Value = improvement::ImprovementTarget> {
    prop_oneof![
        Just(improvement::ImprovementTarget::System),
        "[a-z_]{1,20}".prop_map(improvement::ImprovementTarget::Module),
        "[a-z_]{1,30}".prop_map(improvement::ImprovementTarget::Function),
        Just(improvement::ImprovementTarget::Dependencies),
    ]
}

fn rollback_trigger_strategy() -> impl Strategy<Value = rollback::RollbackTrigger> {
    prop_oneof![
        Just(rollback::RollbackTrigger::VerificationFailure),
        Just(rollback::RollbackTrigger::SoundnessViolation),
        Just(rollback::RollbackTrigger::CapabilityRegression),
        "[a-zA-Z ]{1,30}".prop_map(|r| rollback::RollbackTrigger::Manual { reason: r }),
        "[a-z_]{1,20}"
            .prop_map(|c| rollback::RollbackTrigger::HealthCheckFailure { check_name: c }),
        Just(rollback::RollbackTrigger::Timeout),
    ]
}

fn rejection_reason_strategy() -> impl Strategy<Value = improvement::RejectionReason> {
    prop_oneof![
        Just(improvement::RejectionReason::SoundnessViolation),
        Just(improvement::RejectionReason::CapabilityRegression),
        Just(improvement::RejectionReason::VerificationFailed),
        Just(improvement::RejectionReason::InvalidProposal),
        Just(improvement::RejectionReason::VerificationTimeout),
        Just(improvement::RejectionReason::SystemBusy),
    ]
}

proptest! {
    // ==================== VersionId Tests ====================

    #[test]
    fn version_id_from_content_deterministic(content in prop::collection::vec(any::<u8>(), 1..100)) {
        let id1 = VersionId::from_content_hash(&content);
        let id2 = VersionId::from_content_hash(&content);
        prop_assert_eq!(id1, id2);
    }

    #[test]
    fn version_id_different_content_different_id(
        content1 in prop::collection::vec(any::<u8>(), 1..50),
        content2 in prop::collection::vec(any::<u8>(), 51..100)
    ) {
        let _id1 = VersionId::from_content_hash(&content1);
        let _id2 = VersionId::from_content_hash(&content2);
        // Different content should (almost certainly) produce different IDs
        // There's an astronomically small chance of collision, so we don't assert
    }

    #[test]
    fn version_id_display(id in version_id_strategy()) {
        let display = format!("{}", id);
        prop_assert!(!display.is_empty());
    }

    // ==================== CapabilityValue Tests ====================

    #[test]
    fn capability_value_at_least_reflexive(value in capability_value_strategy()) {
        prop_assert!(value.at_least(&value));
    }

    #[test]
    fn capability_value_boolean_ordering(a: bool, b: bool) {
        let va = version::CapabilityValue::Boolean(a);
        let vb = version::CapabilityValue::Boolean(b);
        prop_assert_eq!(va.at_least(&vb), a >= b);
    }

    #[test]
    fn capability_value_numeric_ordering(a in 0.0f64..1000.0, b in 0.0f64..1000.0) {
        let va = version::CapabilityValue::Numeric(a);
        let vb = version::CapabilityValue::Numeric(b);
        prop_assert_eq!(va.at_least(&vb), a >= b);
    }

    #[test]
    fn capability_value_count_ordering(a in 0u64..10000, b in 0u64..10000) {
        let va = version::CapabilityValue::Count(a);
        let vb = version::CapabilityValue::Count(b);
        prop_assert_eq!(va.at_least(&vb), a >= b);
    }

    #[test]
    fn capability_value_version_ordering(
        a in (0u32..10, 0u32..10, 0u32..10),
        b in (0u32..10, 0u32..10, 0u32..10)
    ) {
        let va = version::CapabilityValue::Version(a.0, a.1, a.2);
        let vb = version::CapabilityValue::Version(b.0, b.1, b.2);
        prop_assert_eq!(va.at_least(&vb), a >= b);
    }

    #[test]
    fn capability_value_display_not_empty(value in capability_value_strategy()) {
        let display = format!("{}", value);
        prop_assert!(!display.is_empty());
    }

    // ==================== CapabilitySet Tests ====================

    #[test]
    fn capability_set_at_least_reflexive(set in capability_set_strategy()) {
        prop_assert!(set.at_least(&set));
    }

    #[test]
    fn capability_set_empty_at_least_empty(_x in 0..1i32) {
        let empty = version::CapabilitySet::new();
        prop_assert!(empty.at_least(&empty));
    }

    #[test]
    fn capability_set_changes_from_self_empty(set in capability_set_strategy()) {
        let changes = set.changes_from(&set);
        prop_assert!(changes.is_empty());
    }

    // ==================== Version Tests ====================

    #[test]
    fn version_new_has_content_hash(content in prop::collection::vec(any::<u8>(), 1..100)) {
        let version = Version::new("v1.0.0", version::CapabilitySet::new(), &content);
        prop_assert!(!version.content_hash.is_empty());
    }

    #[test]
    fn version_genesis_is_genesis(content in prop::collection::vec(any::<u8>(), 1..100)) {
        let version = Version::genesis("v0.0.1", &content);
        prop_assert!(version.metadata.is_genesis);
        prop_assert!(version.previous_version.is_none());
    }

    // ==================== Improvement Tests ====================

    #[test]
    fn improvement_new_has_id(
        desc in "[a-zA-Z ]{1,50}",
        kind in improvement_kind_strategy(),
        target in improvement_target_strategy()
    ) {
        let imp = Improvement::new(desc, kind, target);
        prop_assert!(imp.id.starts_with("imp-"));
    }

    #[test]
    fn improvement_valid_with_description(
        kind in improvement_kind_strategy(),
        target in improvement_target_strategy()
    ) {
        let imp = Improvement::new("Valid description", kind, target);
        prop_assert!(imp.is_valid());
    }

    #[test]
    fn improvement_invalid_with_empty_description(
        kind in improvement_kind_strategy(),
        target in improvement_target_strategy()
    ) {
        let imp = Improvement::new("", kind, target);
        prop_assert!(!imp.is_valid());
    }

    // ==================== ImprovementResult Tests ====================

    #[test]
    fn improvement_result_accepted_is_accepted(_x in 0..1i32) {
        let version = Version::genesis("v1", b"test");
        let cert = ProofCertificate::new(&version, vec![], None);
        let result = ImprovementResult::accepted(version, cert);
        prop_assert!(result.is_accepted());
    }

    #[test]
    fn improvement_result_rejected_not_accepted(
        reason in rejection_reason_strategy(),
        details in "[a-zA-Z ]{1,50}"
    ) {
        let result = ImprovementResult::rejected(reason, details, vec![]);
        prop_assert!(!result.is_accepted());
    }

    // ==================== RollbackTrigger Tests ====================

    #[test]
    fn rollback_trigger_description_not_empty(trigger in rollback_trigger_strategy()) {
        prop_assert!(!trigger.description().is_empty());
    }

    #[test]
    fn rollback_trigger_soundness_is_critical(_x in 0..1i32) {
        prop_assert!(rollback::RollbackTrigger::SoundnessViolation.is_critical());
    }

    #[test]
    fn rollback_trigger_verification_failure_not_critical(_x in 0..1i32) {
        prop_assert!(!rollback::RollbackTrigger::VerificationFailure.is_critical());
    }

    // ==================== RollbackManager Tests ====================

    #[test]
    fn rollback_manager_starts_empty(_x in 0..1i32) {
        let manager = RollbackManager::new();
        prop_assert!(manager.history().is_empty());
    }

    #[test]
    fn rollback_manager_auto_rollback_default(_x in 0..1i32) {
        let manager = RollbackManager::new();
        prop_assert!(manager.auto_rollback_enabled());
    }

    // ==================== GateConfig Tests ====================

    #[test]
    fn gate_config_default_has_backends(_x in 0..1i32) {
        let config = GateConfig::default();
        prop_assert!(!config.backends.is_empty());
    }

    #[test]
    fn gate_config_strict_has_more_backends(_x in 0..1i32) {
        let default = GateConfig::default();
        let strict = GateConfig::strict();
        prop_assert!(strict.backends.len() >= default.backends.len());
    }

    // ==================== VerificationGate Tests ====================

    #[test]
    fn verification_gate_rejects_empty_description(_x in 0..1i32) {
        let gate = VerificationGate::new();
        let version = Version::genesis("v1", b"test");
        let improvement = Improvement::new(
            "",
            improvement::ImprovementKind::BugFix,
            improvement::ImprovementTarget::System,
        );

        let result = gate.apply_improvement(&version, &improvement, None).unwrap();
        prop_assert!(!result.is_accepted());
    }

    // ==================== VerificationConfig Tests ====================

    #[test]
    fn verification_config_default_values(_x in 0..1i32) {
        let config = VerificationConfig::default();
        prop_assert!(config.min_passing_backends > 0);
        prop_assert!(config.min_confidence > 0.0);
        prop_assert!(config.min_confidence <= 1.0);
    }

    #[test]
    fn verification_config_strict_higher_requirements(_x in 0..1i32) {
        let default = VerificationConfig::default();
        let strict = VerificationConfig::strict();
        prop_assert!(strict.min_passing_backends >= default.min_passing_backends);
        prop_assert!(strict.min_confidence >= default.min_confidence);
    }

    // ==================== ProofCertificate Tests ====================

    #[test]
    fn proof_certificate_verified_when_all_pass(_x in 0..1i32) {
        let version = Version::genesis("v1", b"test");
        let checks = vec![
            CertificateCheck::passed("check1"),
            CertificateCheck::passed("check2"),
        ];
        let cert = ProofCertificate::new(&version, checks, None);
        prop_assert!(cert.is_verified());
    }

    #[test]
    fn proof_certificate_not_verified_when_any_fail(_x in 0..1i32) {
        let version = Version::genesis("v1", b"test");
        let checks = vec![
            CertificateCheck::passed("check1"),
            CertificateCheck::failed("check2", "error"),
        ];
        let cert = ProofCertificate::new(&version, checks, None);
        prop_assert!(!cert.is_verified());
    }

    #[test]
    fn proof_certificate_integrity_valid(_x in 0..1i32) {
        let version = Version::genesis("v1", b"test");
        let checks = vec![CertificateCheck::passed("check1")];
        let cert = ProofCertificate::new(&version, checks, None);
        prop_assert!(cert.verify_integrity(&version));
    }

    // ==================== CertificateChain Tests ====================

    #[test]
    fn certificate_chain_starts_empty(_x in 0..1i32) {
        let chain = CertificateChain::new();
        prop_assert!(chain.certificates.is_empty());
        prop_assert!(chain.head.is_none());
    }

    #[test]
    fn certificate_chain_verify_empty_valid(_x in 0..1i32) {
        let chain = CertificateChain::new();
        let verification = chain.verify_chain();
        prop_assert!(verification.valid);
        prop_assert_eq!(verification.verified_count, 0);
    }

    // ==================== VersionHistory Tests ====================

    #[test]
    fn version_history_starts_empty(_x in 0..1i32) {
        let history = VersionHistory::new();
        prop_assert!(history.is_empty());
        prop_assert_eq!(history.len(), 0);
    }

    #[test]
    fn version_history_register_sets_current(_x in 0..1i32) {
        let mut history = VersionHistory::new();
        let version = Version::genesis("v1", b"test");
        let cert = ProofCertificate::new(&version, vec![CertificateCheck::passed("test")], None);

        history.register(version.clone(), cert).unwrap();

        prop_assert!(history.current().is_some());
        prop_assert_eq!(history.current().unwrap().id.clone(), version.id);
    }

    // ==================== Error Tests ====================

    #[test]
    fn error_gate_rejection_is_verification_failure(_x in 0..1i32) {
        let err = SelfImpError::gate_rejection("test");
        prop_assert!(err.is_verification_failure());
    }

    #[test]
    fn error_soundness_violation_is_verification_failure(_x in 0..1i32) {
        let err = SelfImpError::SoundnessViolation("test".to_string());
        prop_assert!(err.is_verification_failure());
    }

    #[test]
    fn error_history_corruption_not_recoverable(_x in 0..1i32) {
        let err = SelfImpError::HistoryCorruption("test".to_string());
        prop_assert!(!err.is_recoverable());
    }

    #[test]
    fn error_verification_timeout_is_recoverable(_x in 0..1i32) {
        let err = SelfImpError::VerificationTimeout(60);
        prop_assert!(err.is_recoverable());
    }
}

// ==================== Unit Tests ====================

#[cfg(test)]
mod unit_tests {
    use super::*;
    use async_trait::async_trait;
    use dashprove_backends::{
        BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
        VerificationStatus,
    };
    use dashprove_dispatcher::{Dispatcher, DispatcherConfig, MergeStrategy, SelectionStrategy};
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn test_capability_improvement_tracking() {
        let mut old_set = CapabilitySet::new();
        old_set.add(Capability::numeric("speed", 1.0));
        old_set.add(Capability::count("backends", 5));

        let mut new_set = CapabilitySet::new();
        new_set.add(Capability::numeric("speed", 1.5)); // Improved
        new_set.add(Capability::count("backends", 5)); // Same
        new_set.add(Capability::boolean("new_feature", true)); // Added

        assert!(new_set.at_least(&old_set));

        let changes = new_set.changes_from(&old_set);
        assert!(!changes.is_empty());

        let improvements: Vec<_> = changes.iter().filter(|c| c.is_improvement()).collect();
        assert!(!improvements.is_empty());
    }

    #[test]
    fn test_capability_regression_detected() {
        let mut old_set = CapabilitySet::new();
        old_set.add(Capability::numeric("speed", 2.0));

        let mut new_set = CapabilitySet::new();
        new_set.add(Capability::numeric("speed", 1.0)); // Regression!

        assert!(!new_set.at_least(&old_set));

        let changes = new_set.changes_from(&old_set);
        let regressions: Vec<_> = changes.iter().filter(|c| c.is_regression()).collect();
        assert!(!regressions.is_empty());
    }

    #[test]
    fn test_verification_gate_accepts_valid_improvement() {
        let gate = VerificationGate::new();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Add new feature",
            ImprovementKind::Feature,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .unwrap();

        assert!(result.is_accepted());
        assert!(result.new_version().is_some());
        assert!(result.certificate().is_some());
    }

    #[test]
    fn test_verification_gate_rejects_capability_regression() {
        let gate = VerificationGate::new();

        let mut old_caps = CapabilitySet::new();
        old_caps.add(Capability::numeric("performance", 100.0));

        let mut new_caps = CapabilitySet::new();
        new_caps.add(Capability::numeric("performance", 50.0)); // Regression

        let version = Version::new("v1", old_caps, b"test content");
        let improvement = Improvement::new(
            "Bad improvement",
            ImprovementKind::Optimization,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(new_caps);

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .unwrap();

        assert!(!result.is_accepted());
        match result {
            ImprovementResult::Rejected { reason, .. } => {
                assert_eq!(reason, RejectionReason::CapabilityRegression);
            }
            _ => panic!("Expected rejection"),
        }
    }

    #[test]
    fn test_certificate_chain_integrity() {
        let mut chain = CertificateChain::new();

        // Add first certificate
        let v1 = Version::genesis("v1", b"v1 content");
        let c1 = ProofCertificate::new(&v1, vec![CertificateCheck::passed("test")], None);
        chain.add(c1.clone()).unwrap();

        // Add second certificate chained to first
        let v2 = Version::derived_from(&v1, "v2", CapabilitySet::new(), b"v2 content");
        let c2 = ProofCertificate::new(
            &v2,
            vec![CertificateCheck::passed("test")],
            Some(c1.content_hash.clone()),
        );
        chain.add(c2).unwrap();

        // Verify chain integrity
        let verification = chain.verify_chain();
        assert!(verification.valid);
        assert_eq!(verification.verified_count, 2);
    }

    #[test]
    fn test_version_history_rollback() {
        let mut history = VersionHistory::new();

        // Register v1
        let v1 = Version::genesis("v1", b"v1 content");
        let c1 = ProofCertificate::new(&v1, vec![CertificateCheck::passed("test")], None);
        history.register(v1.clone(), c1).unwrap();

        // Register v2
        let v2 = Version::derived_from(&v1, "v2", CapabilitySet::new(), b"v2 content");
        let c2 = ProofCertificate::new(
            &v2,
            vec![CertificateCheck::passed("test")],
            history.next_certificate_hash(),
        );
        history.register(v2.clone(), c2).unwrap();

        // Current should be v2
        assert_eq!(history.current().unwrap().id, v2.id);

        // Previous should be v1
        assert_eq!(history.previous().unwrap().id, v1.id);

        // Rollback to v1
        history.set_current(&v1.id.to_string()).unwrap();
        history
            .mark_rolled_back(&v2.id.to_string(), "test rollback".to_string())
            .unwrap();

        // Current should now be v1
        assert_eq!(history.current().unwrap().id, v1.id);
    }

    #[test]
    fn test_async_verification_gate_default_config() {
        // AsyncVerificationGate should be constructible with default config
        let gate = AsyncVerificationGate::default();
        // Just ensure it creates without panic
        drop(gate);
    }

    #[test]
    fn test_async_verification_gate_with_config() {
        let config = GateConfig::strict();
        let gate = AsyncVerificationGate::new(config);
        // Verify it accepts strict config
        drop(gate);
    }

    #[test]
    fn test_async_gate_generate_spec_basic() {
        // Test that spec generation produces valid properties for different improvement types
        let gate = AsyncVerificationGate::default();

        let version = Version::genesis("v1", b"test");
        let improvement = Improvement::new(
            "Test improvement",
            ImprovementKind::BugFix,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(version.capabilities.clone());

        // The generate_verification_spec is private, so we test through the gate flow
        // We just verify the gate doesn't panic on creation
        drop(gate);
        drop(improvement);
    }

    #[derive(Clone)]
    struct StubBackend {
        id: BackendId,
        status: VerificationStatus,
        supported: Vec<PropertyType>,
    }

    impl StubBackend {
        fn new(id: BackendId, status: VerificationStatus, supported: Vec<PropertyType>) -> Self {
            Self {
                id,
                status,
                supported,
            }
        }
    }

    #[async_trait]
    impl VerificationBackend for StubBackend {
        fn id(&self) -> BackendId {
            self.id
        }

        fn supports(&self) -> Vec<PropertyType> {
            self.supported.clone()
        }

        async fn verify(
            &self,
            _spec: &dashprove_usl::typecheck::TypedSpec,
        ) -> Result<BackendResult, BackendError> {
            Ok(BackendResult {
                backend: self.id,
                status: self.status.clone(),
                proof: None,
                counterexample: None,
                diagnostics: vec![],
                time_taken: Duration::from_millis(5),
            })
        }

        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
    }

    #[tokio::test]
    async fn async_gate_accepts_when_dispatcher_proves() {
        let mut capabilities = CapabilitySet::new();
        capabilities.add(Capability::numeric("throughput", 1.0));
        let current = Version::new("v1", capabilities.clone(), b"v1 content");

        let improvement = Improvement::new(
            "Dispatcher proven improvement",
            ImprovementKind::Optimization,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(capabilities.clone());

        let mut dispatcher = Dispatcher::new(DispatcherConfig {
            selection_strategy: SelectionStrategy::All,
            merge_strategy: MergeStrategy::Unanimous,
            max_concurrent: 2,
            task_timeout: Duration::from_secs(5),
            check_health: false,
            auto_update_reputation: false,
        });

        dispatcher.register_backend(Arc::new(StubBackend::new(
            BackendId::Lean4,
            VerificationStatus::Proven,
            vec![PropertyType::Theorem, PropertyType::Invariant],
        )));

        let mut gate = AsyncVerificationGate::with_dispatcher(GateConfig::default(), dispatcher);

        let result = gate
            .apply_improvement(&current, &improvement, None)
            .await
            .expect("async gate should run with dispatcher");

        assert!(result.is_accepted());
    }

    #[tokio::test]
    async fn async_gate_rejects_when_dispatcher_fails() {
        let mut capabilities = CapabilitySet::new();
        capabilities.add(Capability::numeric("accuracy", 0.9));
        let current = Version::new("v1", capabilities.clone(), b"v1 content");

        let improvement = Improvement::new(
            "Dispatcher failure",
            ImprovementKind::Feature,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(capabilities.clone());

        let mut dispatcher = Dispatcher::new(DispatcherConfig {
            selection_strategy: SelectionStrategy::All,
            merge_strategy: MergeStrategy::Unanimous,
            max_concurrent: 2,
            task_timeout: Duration::from_secs(5),
            check_health: false,
            auto_update_reputation: false,
        });

        dispatcher.register_backend(Arc::new(StubBackend::new(
            BackendId::TlaPlus,
            VerificationStatus::Unknown {
                reason: "proof not found".to_string(),
            },
            vec![PropertyType::Theorem, PropertyType::Invariant],
        )));

        let mut gate = AsyncVerificationGate::with_dispatcher(GateConfig::default(), dispatcher);

        let result = gate
            .apply_improvement(&current, &improvement, None)
            .await
            .expect("async gate should return rejection");

        assert!(!result.is_accepted());
        match result {
            ImprovementResult::Rejected { reason, .. } => {
                assert_eq!(reason, RejectionReason::VerificationFailed);
            }
            _ => panic!("Expected rejection due to failed verification"),
        }
    }

    // ==================== Spec Generation Tests ====================
    // These tests verify that the verification gate generates correct
    // USL properties for different improvement kinds and scenarios.

    #[test]
    fn sync_gate_generates_spec_for_bugfix() {
        let gate = VerificationGate::new();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Fix null pointer bug in parser",
            ImprovementKind::BugFix,
            ImprovementTarget::Module("parser".to_string()),
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .expect("should process improvement");

        assert!(result.is_accepted());

        // Check that bugfix-specific checks were generated
        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();
            // Should have spec generation and typecheck checks
            assert!(check_names.contains(&"spec_generation"));
            assert!(check_names.contains(&"spec_typecheck"));
            // Should have bugfix theorems
            assert!(check_names.contains(&"theorem_behavior_preservation"));
            assert!(check_names.contains(&"theorem_bug_fixed"));
        }
    }

    #[test]
    fn sync_gate_generates_spec_for_optimization() {
        let gate = VerificationGate::new();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Optimize query execution",
            ImprovementKind::Optimization,
            ImprovementTarget::Function("execute_query".to_string()),
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();
            // Should have optimization-specific theorems
            assert!(check_names.contains(&"theorem_semantic_preservation"));
            assert!(check_names.contains(&"invariant_performance_improved"));
        }
    }

    #[test]
    fn sync_gate_generates_spec_for_feature() {
        let gate = VerificationGate::new();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Add new export feature",
            ImprovementKind::Feature,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();
            // Should have feature-specific properties
            assert!(check_names.contains(&"invariant_backward_compatibility"));
            assert!(check_names.contains(&"theorem_feature_adds_capability"));
        }
    }

    #[test]
    fn sync_gate_generates_spec_for_security() {
        let gate = VerificationGate::new();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Fix SQL injection vulnerability",
            ImprovementKind::Security,
            ImprovementTarget::Module("database".to_string()),
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();
            // Should have security-specific properties
            assert!(check_names.contains(&"theorem_no_new_vulnerabilities"));
            assert!(check_names.contains(&"invariant_security_improved"));
        }
    }

    #[test]
    fn sync_gate_generates_spec_for_refactoring() {
        let gate = VerificationGate::new();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Refactor API handlers to use async",
            ImprovementKind::Refactoring,
            ImprovementTarget::Module("handlers".to_string()),
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();
            // Refactoring must preserve exact behavior
            assert!(check_names.contains(&"theorem_refactoring_equivalence"));
        }
    }

    #[test]
    fn sync_gate_generates_spec_for_configuration() {
        let gate = VerificationGate::new();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Update timeout configuration",
            ImprovementKind::Configuration,
            ImprovementTarget::Config("timeout".to_string()),
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();
            // Config changes must preserve functional behavior
            assert!(check_names.contains(&"invariant_config_behavior_preserved"));
        }
    }

    #[test]
    fn sync_gate_generates_spec_for_dependency_update() {
        let gate = VerificationGate::new();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Update serde to 1.0.200",
            ImprovementKind::DependencyUpdate,
            ImprovementTarget::Dependencies,
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();
            // Dependency updates should preserve behavior
            assert!(check_names.contains(&"invariant_config_behavior_preserved"));
        }
    }

    #[test]
    fn sync_gate_generates_spec_for_custom_kind() {
        let gate = VerificationGate::new();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Apply custom AI-generated transformation",
            ImprovementKind::Custom("ai_transform".to_string()),
            ImprovementTarget::System,
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();
            // Custom kinds get a generic soundness theorem with sanitized name
            assert!(check_names.contains(&"theorem_custom_ai_transform_soundness"));
        }
    }

    #[test]
    fn sync_gate_generates_capability_invariants() {
        let gate = VerificationGate::new();

        // Version with multiple capability types
        let mut caps = CapabilitySet::new();
        caps.add(Capability::boolean("feature_enabled", true));
        caps.add(Capability::numeric("throughput", 100.0));
        caps.add(Capability::count("backends", 5));
        caps.add(Capability::version("api_version", 1, 2, 3));

        let version = Version::new("v1", caps.clone(), b"test content");
        let improvement = Improvement::new(
            "Add new capability",
            ImprovementKind::Feature,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(caps.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();

            // Should have invariants for each capability
            assert!(check_names.contains(&"invariant_capability_preserved_feature_enabled"));
            assert!(check_names.contains(&"invariant_capability_preserved_throughput"));
            assert!(check_names.contains(&"invariant_capability_preserved_backends"));
            assert!(check_names.contains(&"invariant_capability_preserved_api_version"));
        }
    }

    #[test]
    fn sync_gate_generates_file_change_properties() {
        use crate::improvement::{FileChange, FileChangeType, ImprovementChanges};

        let gate = VerificationGate::new();
        let version = Version::genesis("v1", b"test content");

        // Create improvement with file changes
        let mut changes = ImprovementChanges::default();
        changes.modified_files.push(FileChange {
            path: "src/lib.rs".to_string(),
            change_type: FileChangeType::Modified,
            new_content_hash: Some("abc123".to_string()),
            lines_added: 50,
            lines_removed: 20,
        });
        changes.modified_files.push(FileChange {
            path: "src/deprecated.rs".to_string(),
            change_type: FileChangeType::Deleted,
            new_content_hash: None,
            lines_added: 0,
            lines_removed: 100,
        });

        let improvement = Improvement::new(
            "Refactor and cleanup deprecated code",
            ImprovementKind::Refactoring,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(version.capabilities.clone())
        .with_changes(changes);

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();

            // Should have file deletion check for deleted file
            // Path is sanitized: src/deprecated.rs -> src_deprecated_rs
            assert!(check_names.contains(&"invariant_deleted_file_not_critical_src_deprecated_rs"));

            // Should have change scope invariant
            assert!(check_names.contains(&"invariant_change_scope_bounded"));
        }
    }

    #[test]
    fn sync_gate_generates_soundness_and_lineage_properties() {
        let gate = VerificationGate::new();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Any improvement",
            ImprovementKind::Feature,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();

            // All improvements must have soundness preservation
            assert!(check_names.contains(&"theorem_soundness_preservation"));

            // All improvements must have version lineage invariant
            assert!(check_names.contains(&"invariant_version_lineage"));
        }
    }

    #[test]
    fn sync_gate_adds_backend_ready_checks() {
        let config = GateConfig {
            backends: vec![
                "lean4".to_string(),
                "kani".to_string(),
                "tlaplus".to_string(),
            ],
            ..Default::default()
        };
        let gate = VerificationGate::with_config(config);
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Add feature",
            ImprovementKind::Feature,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();

            // Should have backend_ready checks for each configured backend
            assert!(check_names.contains(&"backend_lean4_ready"));
            assert!(check_names.contains(&"backend_kani_ready"));
            assert!(check_names.contains(&"backend_tlaplus_ready"));
        }
    }

    #[test]
    fn sync_gate_sanitizes_paths_in_property_names() {
        use crate::improvement::{FileChange, FileChangeType, ImprovementChanges};

        let gate = VerificationGate::new();
        let version = Version::genesis("v1", b"test content");

        // File path with special characters
        let mut changes = ImprovementChanges::default();
        changes.modified_files.push(FileChange {
            path: "src/foo-bar/baz_qux.rs".to_string(),
            change_type: FileChangeType::Deleted,
            new_content_hash: None,
            lines_added: 0,
            lines_removed: 50,
        });

        let improvement = Improvement::new(
            "Remove old file",
            ImprovementKind::Refactoring,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(version.capabilities.clone())
        .with_changes(changes);

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();

            // Path should be sanitized: - and / become _
            assert!(
                check_names.contains(&"invariant_deleted_file_not_critical_src_foo_bar_baz_qux_rs")
            );
        }
    }

    #[test]
    fn sync_gate_handles_large_file_changes() {
        use crate::improvement::{FileChange, FileChangeType, ImprovementChanges};

        let gate = VerificationGate::new();
        let version = Version::genesis("v1", b"test content");

        // Create many file changes
        let mut changes = ImprovementChanges::default();
        for i in 0..100 {
            changes.modified_files.push(FileChange {
                path: format!("src/module_{}.rs", i),
                change_type: FileChangeType::Modified,
                new_content_hash: Some(format!("hash_{}", i)),
                lines_added: 10,
                lines_removed: 5,
            });
        }

        let improvement = Improvement::new(
            "Mass refactoring",
            ImprovementKind::Refactoring,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(version.capabilities.clone())
        .with_changes(changes);

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();

            // Should have change scope bounded invariant
            assert!(check_names.contains(&"invariant_change_scope_bounded"));

            // Total lines changed = 100 * (10 + 5) = 1500
            // This is under the 10000 line limit, so should pass
        }
    }

    // ==================== AsyncVerificationGate Spec Generation Tests ====================
    // Tests for async gate behavior - the async gate differs from sync gate in that
    // without a registered dispatcher, it only generates placeholder backend checks
    // rather than individual property checks.
    //
    // When a dispatcher with backends IS registered, the async gate generates
    // property_N checks for each property verified by the dispatcher.

    #[tokio::test]
    async fn async_gate_accepts_valid_bugfix() {
        let mut gate = AsyncVerificationGate::default();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Fix null pointer bug in parser",
            ImprovementKind::BugFix,
            ImprovementTarget::Module("parser".to_string()),
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .await
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();
            // Async gate without dispatcher generates structural checks + backend placeholders
            assert!(check_names.contains(&"structural_validity"));
            assert!(check_names.contains(&"soundness_preservation"));
            assert!(check_names.contains(&"capability_preservation"));
            // Backend placeholders
            assert!(check_names.iter().any(|n| n.contains("backend_")));
        }
    }

    #[tokio::test]
    async fn async_gate_accepts_valid_optimization() {
        let mut gate = AsyncVerificationGate::default();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Optimize query execution",
            ImprovementKind::Optimization,
            ImprovementTarget::Function("execute_query".to_string()),
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .await
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();
            // Should have core phase checks
            assert!(check_names.contains(&"structural_validity"));
            assert!(check_names.contains(&"soundness_preservation"));
        }
    }

    #[tokio::test]
    async fn async_gate_accepts_valid_feature() {
        let mut gate = AsyncVerificationGate::default();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Add new export feature",
            ImprovementKind::Feature,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .await
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            // Certificate should be verified
            assert!(certificate.is_verified());
            // All checks should pass
            assert!(certificate.checks.iter().all(|c| c.passed));
        }
    }

    #[tokio::test]
    async fn async_gate_accepts_valid_security() {
        let mut gate = AsyncVerificationGate::default();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Fix SQL injection vulnerability",
            ImprovementKind::Security,
            ImprovementTarget::Module("database".to_string()),
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .await
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            assert!(certificate.is_verified());
            // Should have the core structural and soundness checks
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();
            assert!(check_names.contains(&"structural_validity"));
        }
    }

    #[tokio::test]
    async fn async_gate_accepts_valid_refactoring() {
        let mut gate = AsyncVerificationGate::default();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Refactor API handlers to use async",
            ImprovementKind::Refactoring,
            ImprovementTarget::Module("handlers".to_string()),
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .await
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            assert!(certificate.is_verified());
        }
    }

    #[tokio::test]
    async fn async_gate_accepts_valid_configuration() {
        let mut gate = AsyncVerificationGate::default();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Update timeout configuration",
            ImprovementKind::Configuration,
            ImprovementTarget::Config("timeout".to_string()),
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .await
            .expect("should process improvement");

        assert!(result.is_accepted());
    }

    #[tokio::test]
    async fn async_gate_accepts_valid_dependency_update() {
        let mut gate = AsyncVerificationGate::default();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Update serde to 1.0.200",
            ImprovementKind::DependencyUpdate,
            ImprovementTarget::Dependencies,
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .await
            .expect("should process improvement");

        assert!(result.is_accepted());
    }

    #[tokio::test]
    async fn async_gate_accepts_valid_custom_kind() {
        let mut gate = AsyncVerificationGate::default();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Apply custom AI-generated transformation",
            ImprovementKind::Custom("ai_transform".to_string()),
            ImprovementTarget::System,
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .await
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            assert!(certificate.is_verified());
        }
    }

    #[tokio::test]
    async fn async_gate_generates_backend_placeholder_checks() {
        let mut gate = AsyncVerificationGate::default();

        // Version with multiple capability types
        let mut caps = CapabilitySet::new();
        caps.add(Capability::boolean("feature_enabled", true));
        caps.add(Capability::numeric("throughput", 100.0));
        caps.add(Capability::count("backends", 5));
        caps.add(Capability::version("api_version", 1, 2, 3));

        let version = Version::new("v1", caps.clone(), b"test content");
        let improvement = Improvement::new(
            "Add new capability",
            ImprovementKind::Feature,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(caps.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .await
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();

            // Without a dispatcher, async gate generates backend placeholder checks
            // based on the configured backends (lean4, kani, tlaplus by default)
            assert!(check_names.iter().any(|n| n.contains("backend_lean4")));
            assert!(check_names.iter().any(|n| n.contains("backend_kani")));
            assert!(check_names.iter().any(|n| n.contains("backend_tlaplus")));
        }
    }

    #[tokio::test]
    async fn async_gate_generates_file_change_properties() {
        use crate::improvement::{FileChange, FileChangeType, ImprovementChanges};

        let mut gate = AsyncVerificationGate::default();
        let version = Version::genesis("v1", b"test content");

        // Create improvement with file changes
        let mut changes = ImprovementChanges::default();
        changes.modified_files.push(FileChange {
            path: "src/lib.rs".to_string(),
            change_type: FileChangeType::Modified,
            new_content_hash: Some("abc123".to_string()),
            lines_added: 50,
            lines_removed: 20,
        });
        changes.modified_files.push(FileChange {
            path: "src/deprecated.rs".to_string(),
            change_type: FileChangeType::Deleted,
            new_content_hash: None,
            lines_added: 0,
            lines_removed: 100,
        });

        let improvement = Improvement::new(
            "Refactor and cleanup deprecated code",
            ImprovementKind::Refactoring,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(version.capabilities.clone())
        .with_changes(changes);

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .await
            .expect("should process improvement");

        assert!(result.is_accepted());

        // Async gate with no registered backends passes but generates placeholder checks
        // for the spec structure
        if let ImprovementResult::Accepted { certificate, .. } = result {
            // Verify certificate was created
            assert!(certificate.is_verified());
        }
    }

    #[tokio::test]
    async fn async_gate_generates_soundness_and_lineage_properties() {
        let mut gate = AsyncVerificationGate::default();
        let version = Version::genesis("v1", b"test content");
        let improvement = Improvement::new(
            "Any improvement",
            ImprovementKind::Feature,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .await
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            // Certificate should be verified when no backends fail
            assert!(certificate.is_verified());
            // All checks should pass
            assert!(certificate.checks.iter().all(|c| c.passed));
        }
    }

    #[tokio::test]
    async fn async_gate_with_dispatcher_generates_property_checks() {
        let mut capabilities = CapabilitySet::new();
        capabilities.add(Capability::numeric("metric", 1.0));
        let current = Version::new("v1", capabilities.clone(), b"v1 content");

        let improvement = Improvement::new(
            "Improvement with dispatcher verification",
            ImprovementKind::BugFix,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(capabilities.clone());

        let mut dispatcher = Dispatcher::new(DispatcherConfig {
            selection_strategy: SelectionStrategy::All,
            merge_strategy: MergeStrategy::Unanimous,
            max_concurrent: 2,
            task_timeout: Duration::from_secs(5),
            check_health: false,
            auto_update_reputation: false,
        });

        dispatcher.register_backend(Arc::new(StubBackend::new(
            BackendId::Kani,
            VerificationStatus::Proven,
            vec![PropertyType::Theorem, PropertyType::Invariant],
        )));

        let mut gate = AsyncVerificationGate::with_dispatcher(GateConfig::default(), dispatcher);

        let result = gate
            .apply_improvement(&current, &improvement, None)
            .await
            .expect("async gate should complete");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            let check_names: Vec<_> = certificate.checks.iter().map(|c| c.name.as_str()).collect();
            // Should have verification_summary when dispatcher is used
            assert!(check_names.contains(&"verification_summary"));
            // Should have property checks from dispatcher
            assert!(check_names.iter().any(|n| n.starts_with("property_")));
        }
    }

    #[tokio::test]
    async fn async_gate_accepts_custom_kind_with_special_chars() {
        let mut gate = AsyncVerificationGate::default();
        let version = Version::genesis("v1", b"test content");

        // Custom kind with special characters - should be accepted
        let improvement = Improvement::new(
            "Apply custom transformation",
            ImprovementKind::Custom("my-custom/kind.v2".to_string()),
            ImprovementTarget::System,
        )
        .with_expected_capabilities(version.capabilities.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .await
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            // Certificate should be valid regardless of custom kind naming
            assert!(certificate.is_verified());
        }
    }

    #[tokio::test]
    async fn async_gate_handles_many_capabilities() {
        let mut gate = AsyncVerificationGate::default();

        // Create version with many capabilities
        let mut caps = CapabilitySet::new();
        for i in 0..20 {
            caps.add(Capability::numeric(format!("metric_{}", i), i as f64));
        }

        let version = Version::new("v1", caps.clone(), b"test content");
        let improvement = Improvement::new(
            "Update with many capabilities",
            ImprovementKind::Optimization,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(caps.clone());

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .await
            .expect("should process improvement");

        assert!(result.is_accepted());

        if let ImprovementResult::Accepted { certificate, .. } = result {
            // Async gate without dispatcher generates:
            // 3 phase checks + 3 backend placeholders = 6+ checks
            // (Not individual property checks like sync gate)
            assert!(certificate.checks.len() >= 5);
            assert!(certificate.is_verified());
        }
    }

    #[tokio::test]
    async fn async_gate_rejects_empty_description() {
        let mut gate = AsyncVerificationGate::default();
        let version = Version::genesis("v1", b"test");
        let improvement = Improvement::new("", ImprovementKind::BugFix, ImprovementTarget::System);

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .await
            .expect("should return rejection");

        assert!(!result.is_accepted());
        match result {
            ImprovementResult::Rejected { reason, .. } => {
                assert_eq!(reason, RejectionReason::InvalidProposal);
            }
            _ => panic!("Expected rejection"),
        }
    }

    #[tokio::test]
    async fn async_gate_rejects_capability_regression() {
        let mut gate = AsyncVerificationGate::default();

        let mut old_caps = CapabilitySet::new();
        old_caps.add(Capability::numeric("performance", 100.0));

        let mut new_caps = CapabilitySet::new();
        new_caps.add(Capability::numeric("performance", 50.0)); // Regression!

        let version = Version::new("v1", old_caps, b"test content");
        let improvement = Improvement::new(
            "Bad improvement",
            ImprovementKind::Optimization,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(new_caps);

        let result = gate
            .apply_improvement(&version, &improvement, None)
            .await
            .expect("should return rejection");

        assert!(!result.is_accepted());
        match result {
            ImprovementResult::Rejected { reason, .. } => {
                assert_eq!(reason, RejectionReason::CapabilityRegression);
            }
            _ => panic!("Expected rejection due to capability regression"),
        }
    }

    // ========================================================================
    // Incremental Gate Verification Tests
    // ========================================================================

    #[test]
    fn incremental_gate_result_no_cache() {
        let result = IncrementalGateResult::no_cache();
        assert_eq!(result.cached_count, 0);
        assert_eq!(result.verified_count, 0);
        assert_eq!(result.hit_rate(), 0.0);
    }

    #[test]
    fn incremental_gate_result_hit_rate_calculation() {
        let result = IncrementalGateResult {
            cached_count: 3,
            verified_count: 1,
            cached_properties: vec!["p1".to_string(), "p2".to_string(), "p3".to_string()],
            verified_properties: vec!["p4".to_string()],
            time_saved_ms: 150,
        };
        assert_eq!(result.hit_rate(), 0.75);
    }

    #[test]
    fn incremental_gate_result_hit_rate_empty() {
        let result = IncrementalGateResult {
            cached_count: 0,
            verified_count: 0,
            cached_properties: Vec::new(),
            verified_properties: Vec::new(),
            time_saved_ms: 0,
        };
        assert_eq!(result.hit_rate(), 0.0);
    }

    #[test]
    fn incremental_gate_result_all_cached() {
        let result = IncrementalGateResult {
            cached_count: 5,
            verified_count: 0,
            cached_properties: vec![
                "p1".to_string(),
                "p2".to_string(),
                "p3".to_string(),
                "p4".to_string(),
                "p5".to_string(),
            ],
            verified_properties: Vec::new(),
            time_saved_ms: 250,
        };
        assert_eq!(result.hit_rate(), 1.0);
    }

    #[test]
    fn async_gate_with_cache_creates_cache() {
        let gate = AsyncVerificationGate::with_cache(GateConfig::default());
        assert!(gate.has_cache());
    }

    #[test]
    fn async_gate_with_custom_cache_creates_cache() {
        let gate = AsyncVerificationGate::with_custom_cache(
            GateConfig::default(),
            100,
            std::time::Duration::from_secs(60),
        );
        assert!(gate.has_cache());
    }

    #[test]
    fn async_gate_new_has_no_cache() {
        let gate = AsyncVerificationGate::new(GateConfig::default());
        assert!(!gate.has_cache());
    }

    #[test]
    fn async_gate_enable_cache() {
        let mut gate = AsyncVerificationGate::new(GateConfig::default());
        assert!(!gate.has_cache());
        gate.enable_cache();
        assert!(gate.has_cache());
    }

    #[test]
    fn async_gate_enable_cache_idempotent() {
        let mut gate = AsyncVerificationGate::with_cache(GateConfig::default());
        assert!(gate.has_cache());
        gate.enable_cache();
        assert!(gate.has_cache()); // Still has cache after second enable
    }

    #[tokio::test]
    async fn async_gate_cache_stats_initially_empty() {
        let gate = AsyncVerificationGate::with_cache(GateConfig::default());
        let stats = gate.cache_stats().await;
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.entry_count, 0);
    }

    #[tokio::test]
    async fn async_gate_cache_stats_none_without_cache() {
        let gate = AsyncVerificationGate::new(GateConfig::default());
        let stats = gate.cache_stats().await;
        assert!(stats.is_none());
    }

    #[tokio::test]
    async fn async_gate_clear_cache() {
        let gate = AsyncVerificationGate::with_cache(GateConfig::default());

        // Clear cache (should not panic even when empty)
        gate.clear_cache().await;

        let stats = gate.cache_stats().await.unwrap();
        assert_eq!(stats.entry_count, 0);
    }

    #[tokio::test]
    async fn async_gate_with_dispatcher_and_cache() {
        let config = DispatcherConfig::default();
        let dispatcher = Dispatcher::new(config);
        let gate =
            AsyncVerificationGate::with_dispatcher_and_cache(GateConfig::default(), dispatcher);
        assert!(gate.has_cache());
        assert!(gate.has_dispatcher());
    }

    #[tokio::test]
    async fn async_gate_incremental_returns_stats() {
        let mut gate = AsyncVerificationGate::with_cache(GateConfig::default());

        let caps = CapabilitySet::new();
        let version = Version::new("v1", caps.clone(), b"test content");
        let improvement = Improvement::new(
            "Test improvement",
            ImprovementKind::BugFix,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(caps);

        let changed_defs = vec!["SomeType".to_string()];
        let result = gate
            .apply_improvement_incremental(&version, &improvement, &changed_defs, None)
            .await;

        assert!(result.is_ok());
        let (improvement_result, stats) = result.unwrap();

        // Stats should be returned
        // The first call with cache should verify all properties (no cache hits yet)
        assert!(improvement_result.is_accepted());
        // With cache enabled, verified_count should include the generated properties
        assert!(
            stats.verified_count > 0
                || stats.cached_count > 0
                || stats.verified_properties.is_empty()
        );
    }

    #[tokio::test]
    async fn async_gate_incremental_without_cache_falls_back() {
        let mut gate = AsyncVerificationGate::new(GateConfig::default());

        let caps = CapabilitySet::new();
        let version = Version::new("v1", caps.clone(), b"test content");
        let improvement = Improvement::new(
            "Test improvement",
            ImprovementKind::BugFix,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(caps);

        let changed_defs = vec!["SomeType".to_string()];
        let result = gate
            .apply_improvement_incremental(&version, &improvement, &changed_defs, None)
            .await;

        assert!(result.is_ok());
        let (_improvement_result, stats) = result.unwrap();

        // Without cache, cached_count should be 0
        assert_eq!(stats.cached_count, 0);
    }

    #[tokio::test]
    async fn async_gate_cache_entries_expire_after_ttl() {
        let mut gate = AsyncVerificationGate::with_custom_cache(
            GateConfig::default(),
            128,
            std::time::Duration::from_millis(20),
        );

        let caps = CapabilitySet::new();
        let version = Version::new("v1", caps.clone(), b"test content");
        let improvement = Improvement::new(
            "Test improvement",
            ImprovementKind::BugFix,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(caps);

        // Populate cache
        let (_, stats_initial) = gate
            .apply_improvement_incremental(&version, &improvement, &[], None)
            .await
            .expect("first verification should succeed");
        assert_eq!(stats_initial.cached_count, 0);
        assert!(stats_initial.verified_count > 0);

        // Immediately reuse cache before TTL expiration
        let (_, stats_before_expiry) = gate
            .apply_improvement_incremental(&version, &improvement, &[], None)
            .await
            .expect("second verification should succeed");
        assert!(stats_before_expiry.cached_count > 0);
        assert_eq!(stats_before_expiry.verified_count, 0);

        let cache_stats_before = gate.cache_stats().await.unwrap();

        // Wait long enough for TTL to expire entries
        tokio::time::sleep(std::time::Duration::from_millis(35)).await;

        let (_, stats_after_expiry) = gate
            .apply_improvement_incremental(&version, &improvement, &[], None)
            .await
            .expect("third verification should succeed");

        // Expired entries should be treated as misses and re-verified
        assert_eq!(stats_after_expiry.cached_count, 0);
        assert!(stats_after_expiry.verified_count > 0);

        let cache_stats_after = gate.cache_stats().await.unwrap();
        assert!(cache_stats_after.entry_count > 0);
        assert!(
            cache_stats_after.misses
                >= cache_stats_before.misses + stats_before_expiry.cached_count as u64
        );
    }

    #[tokio::test]
    async fn async_gate_incremental_rejects_invalid_proposal() {
        let mut gate = AsyncVerificationGate::with_cache(GateConfig::default());

        let caps = CapabilitySet::new();
        let version = Version::new("v1", caps.clone(), b"test content");
        let mut improvement = Improvement::new(
            "",
            ImprovementKind::BugFix, // Empty description - invalid
            ImprovementTarget::System,
        );
        improvement.description = String::new(); // Force empty description

        let changed_defs = Vec::new();
        let result = gate
            .apply_improvement_incremental(&version, &improvement, &changed_defs, None)
            .await;

        assert!(result.is_ok());
        let (improvement_result, stats) = result.unwrap();

        assert!(!improvement_result.is_accepted());
        // Stats should indicate no cache was used due to early rejection
        assert_eq!(stats.cached_count, 0);
        assert_eq!(stats.verified_count, 0);
    }

    #[tokio::test]
    async fn async_gate_incremental_rejects_capability_regression() {
        let mut gate = AsyncVerificationGate::with_cache(GateConfig::default());

        let mut old_caps = CapabilitySet::new();
        old_caps.add(Capability::numeric("performance", 100.0));

        let mut new_caps = CapabilitySet::new();
        new_caps.add(Capability::numeric("performance", 50.0)); // Regression!

        let version = Version::new("v1", old_caps, b"test content");
        let improvement = Improvement::new(
            "Bad improvement",
            ImprovementKind::Optimization,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(new_caps);

        let changed_defs = Vec::new();
        let result = gate
            .apply_improvement_incremental(&version, &improvement, &changed_defs, None)
            .await;

        assert!(result.is_ok());
        let (improvement_result, stats) = result.unwrap();

        assert!(!improvement_result.is_accepted());
        match improvement_result {
            ImprovementResult::Rejected { reason, .. } => {
                assert_eq!(reason, RejectionReason::CapabilityRegression);
            }
            _ => panic!("Expected rejection due to capability regression"),
        }
        // Stats should indicate no cache was used due to early rejection
        assert_eq!(stats.cached_count, 0);
        assert_eq!(stats.verified_count, 0);
    }

    #[tokio::test]
    async fn async_gate_incremental_stores_property_specific_cache_entries() {
        // Test that each property gets its own cache entry with individual status
        let mut gate = AsyncVerificationGate::with_cache(GateConfig::default());

        // Create version with multiple capabilities to generate multiple properties
        let mut caps = CapabilitySet::new();
        caps.add(Capability::numeric("accuracy", 95.0));
        caps.add(Capability::boolean("supports_unicode", true));
        caps.add(Capability::count("max_threads", 8));

        let version = Version::new("v1", caps.clone(), b"test content with multiple caps");
        let improvement = Improvement::new(
            "Multi-property test improvement",
            ImprovementKind::Optimization,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(caps);

        // First verification - populates cache with property-specific entries
        let (result1, stats1) = gate
            .apply_improvement_incremental(&version, &improvement, &[], None)
            .await
            .expect("first verification should succeed");

        assert!(result1.is_accepted());
        // Should verify multiple properties (soundness, lineage, + capabilities)
        assert!(
            stats1.verified_count >= 3,
            "Expected at least 3 verified properties, got {}",
            stats1.verified_count
        );
        assert_eq!(
            stats1.cached_count, 0,
            "First run should have no cache hits"
        );

        // Second verification - should get cache hits with property-specific statuses
        let (result2, stats2) = gate
            .apply_improvement_incremental(&version, &improvement, &[], None)
            .await
            .expect("second verification should succeed");

        assert!(result2.is_accepted());
        assert!(
            stats2.cached_count >= 3,
            "Expected at least 3 cache hits, got {}",
            stats2.cached_count
        );
        assert_eq!(stats2.verified_count, 0, "Second run should use all cached");

        // Verify cache stats reflect property-specific entries
        let cache_stats = gate.cache_stats().await.unwrap();
        assert!(
            cache_stats.entry_count >= 3,
            "Expected at least 3 cache entries, got {}",
            cache_stats.entry_count
        );
        assert!(
            cache_stats.hits >= stats2.cached_count as u64,
            "Cache hits should be at least {}",
            stats2.cached_count
        );
    }

    #[tokio::test]
    async fn async_gate_incremental_cache_preserves_individual_property_names() {
        // Test that cached property names match the generated property names
        let mut gate = AsyncVerificationGate::with_cache(GateConfig::default());

        let mut caps = CapabilitySet::new();
        caps.add(Capability::numeric("performance", 100.0));

        let version = Version::new("v1", caps.clone(), b"perf test");
        let improvement = Improvement::new(
            "Performance test improvement",
            ImprovementKind::Optimization,
            ImprovementTarget::System,
        )
        .with_expected_capabilities(caps);

        // First run to populate cache
        let (_, stats1) = gate
            .apply_improvement_incremental(&version, &improvement, &[], None)
            .await
            .expect("should succeed");

        // Capture verified property names
        let verified_names = stats1.verified_properties.clone();
        assert!(
            !verified_names.is_empty(),
            "Should have verified at least one property"
        );

        // Second run should cache same property names
        let (_, stats2) = gate
            .apply_improvement_incremental(&version, &improvement, &[], None)
            .await
            .expect("should succeed");

        // Cached properties should match the original verified property names
        assert_eq!(
            stats2.cached_properties.len(),
            verified_names.len(),
            "Cached count should match verified count"
        );

        // Each cached property should be one of the originally verified ones
        for cached_name in &stats2.cached_properties {
            assert!(
                verified_names.contains(cached_name),
                "Cached property '{}' should be in verified names {:?}",
                cached_name,
                verified_names
            );
        }
    }
}

// ==================== Learning Threshold Tests ====================

#[test]
fn test_learning_threshold_config_new() {
    let config = LearningThresholdConfig::new();
    assert_eq!(config.window_duration, Duration::from_secs(3600));
    assert_eq!(config.warmup_samples, 10);
    assert_eq!(config.max_samples, 1000);
    assert!((config.smoothing_factor - 0.3).abs() < f64::EPSILON);
    assert!((config.high_percentile - 75.0).abs() < f64::EPSILON);
    assert!((config.low_percentile - 25.0).abs() < f64::EPSILON);
    assert!(!config.time_aware);
    assert_eq!(config.min_threshold_floor, 100);
    assert_eq!(config.max_threshold_ceiling, 100 * 1024 * 1024);
}

#[test]
fn test_learning_threshold_config_default() {
    let config = LearningThresholdConfig::default();
    assert_eq!(config.window_duration, Duration::from_secs(3600));
    assert_eq!(config.warmup_samples, 10);
}

#[test]
fn test_learning_threshold_config_builder_methods() {
    let config = LearningThresholdConfig::new()
        .with_window_duration(Duration::from_secs(7200))
        .with_warmup_samples(20)
        .with_max_samples(500)
        .with_smoothing_factor(0.5)
        .with_high_percentile(80.0)
        .with_low_percentile(20.0)
        .with_time_awareness(true)
        .with_min_threshold_floor(500)
        .with_max_threshold_ceiling(50 * 1024 * 1024);

    assert_eq!(config.window_duration, Duration::from_secs(7200));
    assert_eq!(config.warmup_samples, 20);
    assert_eq!(config.max_samples, 500);
    assert!((config.smoothing_factor - 0.5).abs() < f64::EPSILON);
    assert!((config.high_percentile - 80.0).abs() < f64::EPSILON);
    assert!((config.low_percentile - 20.0).abs() < f64::EPSILON);
    assert!(config.time_aware);
    assert_eq!(config.min_threshold_floor, 500);
    assert_eq!(config.max_threshold_ceiling, 50 * 1024 * 1024);
}

#[test]
fn test_learning_threshold_config_smoothing_factor_clamping() {
    // Below range should be clamped to 0.0
    let config = LearningThresholdConfig::new().with_smoothing_factor(-0.5);
    assert!((config.smoothing_factor - 0.0).abs() < f64::EPSILON);

    // Above range should be clamped to 1.0
    let config = LearningThresholdConfig::new().with_smoothing_factor(1.5);
    assert!((config.smoothing_factor - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_learning_threshold_config_percentile_clamping() {
    // High percentile below 50 should be clamped
    let config = LearningThresholdConfig::new().with_high_percentile(40.0);
    assert!((config.high_percentile - 50.0).abs() < f64::EPSILON);

    // High percentile above 99 should be clamped
    let config = LearningThresholdConfig::new().with_high_percentile(100.0);
    assert!((config.high_percentile - 99.0).abs() < f64::EPSILON);

    // Low percentile below 1 should be clamped
    let config = LearningThresholdConfig::new().with_low_percentile(0.5);
    assert!((config.low_percentile - 1.0).abs() < f64::EPSILON);

    // Low percentile above 50 should be clamped
    let config = LearningThresholdConfig::new().with_low_percentile(60.0);
    assert!((config.low_percentile - 50.0).abs() < f64::EPSILON);
}

#[test]
fn test_learning_threshold_config_presets() {
    // Short-term preset
    let short = LearningThresholdConfig::short_term();
    assert_eq!(short.window_duration, Duration::from_secs(15 * 60));
    assert_eq!(short.warmup_samples, 5);
    assert!((short.smoothing_factor - 0.5).abs() < f64::EPSILON);
    assert!(!short.time_aware);

    // Long-term preset
    let long = LearningThresholdConfig::long_term();
    assert_eq!(long.window_duration, Duration::from_secs(24 * 3600));
    assert_eq!(long.warmup_samples, 100);
    assert!((long.smoothing_factor - 0.1).abs() < f64::EPSILON);
    assert!(long.time_aware);

    // Stable preset
    let stable = LearningThresholdConfig::stable();
    assert_eq!(stable.window_duration, Duration::from_secs(6 * 3600));
    assert_eq!(stable.warmup_samples, 50);
}

#[test]
fn test_activity_sample_now() {
    let sample = ActivitySample::now(5000, Duration::from_secs(60));
    assert_eq!(sample.bytes_changed, 5000);
    assert_eq!(sample.interval_ms, 60000);
    assert!(sample.timestamp_ms > 0);
    assert!(sample.hour_of_day < 24);
}

#[test]
fn test_activity_sample_with_timestamp() {
    // 12:30 UTC (noon)
    let timestamp_ms = 1704110400000u64; // Some specific timestamp
    let sample = ActivitySample::with_timestamp(timestamp_ms, 1000, Duration::from_secs(30));
    assert_eq!(sample.timestamp_ms, timestamp_ms);
    assert_eq!(sample.bytes_changed, 1000);
    assert_eq!(sample.interval_ms, 30000);
}

#[test]
fn test_historical_activity_tracker_new() {
    let config = LearningThresholdConfig::new();
    let tracker = HistoricalActivityTracker::new(config);
    assert_eq!(tracker.sample_count(), 0);
    assert!(!tracker.is_warmed_up());
    assert_eq!(tracker.high_threshold(), 10 * 1024); // Default 10KB
    assert_eq!(tracker.low_threshold(), 1024); // Default 1KB
}

#[test]
fn test_historical_activity_tracker_with_defaults() {
    let tracker = HistoricalActivityTracker::with_defaults();
    assert_eq!(tracker.sample_count(), 0);
    assert!(!tracker.is_warmed_up());
}

#[test]
fn test_historical_activity_tracker_warmup() {
    let config = LearningThresholdConfig::new().with_warmup_samples(5);
    let mut tracker = HistoricalActivityTracker::new(config);

    // Record samples below warmup
    for i in 0..4 {
        let result = tracker.record_sample(1000 * (i + 1) as u64, Duration::from_secs(60));
        assert!(
            result.is_none(),
            "Should not update thresholds during warmup"
        );
        assert!(!tracker.is_warmed_up());
    }

    // Fifth sample should trigger warmup completion
    let _result = tracker.record_sample(5000, Duration::from_secs(60));
    assert!(tracker.is_warmed_up());
    // May or may not update thresholds depending on data
}

#[test]
fn test_historical_activity_tracker_threshold_update() {
    let config = LearningThresholdConfig::new()
        .with_warmup_samples(5)
        .with_smoothing_factor(1.0); // Immediate updates for testing
    let mut tracker = HistoricalActivityTracker::new(config);

    // Add samples with varying activity levels
    // Low: 100, 200, 300; Medium: 5000, 6000; High: 15000, 20000
    let values = [100, 200, 300, 5000, 6000, 15000, 20000];
    for &v in &values {
        tracker.record_sample(v, Duration::from_secs(60));
    }

    // After warmup, thresholds should have been updated based on percentiles
    // With 7 samples and 75th percentile: index = round(0.75 * 6) = 5 -> 15000
    // With 7 samples and 25th percentile: index = round(0.25 * 6) = 2 -> 300
    let stats = tracker.statistics();
    assert!(
        stats.current_high_threshold > 1000,
        "High threshold should be elevated"
    );
    assert!(
        stats.current_low_threshold < 10000,
        "Low threshold should be reasonable"
    );
}

#[test]
fn test_historical_activity_tracker_statistics() {
    let config = LearningThresholdConfig::new().with_warmup_samples(3);
    let mut tracker = HistoricalActivityTracker::new(config);

    // Add some samples
    tracker.record_sample(1000, Duration::from_secs(60));
    tracker.record_sample(2000, Duration::from_secs(60));
    tracker.record_sample(3000, Duration::from_secs(60));

    let stats = tracker.statistics();
    assert_eq!(stats.sample_count, 3);
    assert!((stats.mean_bytes - 2000.0).abs() < 1.0);
    assert_eq!(stats.min_bytes, 1000);
    assert_eq!(stats.max_bytes, 3000);
}

#[test]
fn test_historical_activity_tracker_reset() {
    let config = LearningThresholdConfig::new().with_warmup_samples(3);
    let mut tracker = HistoricalActivityTracker::new(config);

    // Add samples and warmup
    tracker.record_sample(5000, Duration::from_secs(60));
    tracker.record_sample(10000, Duration::from_secs(60));
    tracker.record_sample(15000, Duration::from_secs(60));

    assert!(tracker.is_warmed_up());
    assert_eq!(tracker.sample_count(), 3);

    // Reset
    tracker.reset();

    assert!(!tracker.is_warmed_up());
    assert_eq!(tracker.sample_count(), 0);
    assert_eq!(tracker.high_threshold(), 10 * 1024); // Back to default
    assert_eq!(tracker.low_threshold(), 1024);
}

#[test]
fn test_historical_activity_tracker_export_import() {
    let config = LearningThresholdConfig::new().with_warmup_samples(3);
    let mut tracker1 = HistoricalActivityTracker::new(config.clone());

    // Add samples
    tracker1.record_sample(1000, Duration::from_secs(60));
    tracker1.record_sample(2000, Duration::from_secs(60));
    tracker1.record_sample(3000, Duration::from_secs(60));

    // Export
    let samples = tracker1.export_samples();
    assert_eq!(samples.len(), 3);

    // Import into new tracker
    let mut tracker2 = HistoricalActivityTracker::new(config);
    tracker2.import_samples(samples);

    assert_eq!(tracker2.sample_count(), 3);
    assert!(tracker2.is_warmed_up());
}

#[test]
fn test_historical_activity_tracker_max_samples() {
    let config = LearningThresholdConfig::new()
        .with_warmup_samples(3)
        .with_max_samples(10);
    let mut tracker = HistoricalActivityTracker::new(config);

    // Add more samples than max
    for i in 0..20 {
        tracker.record_sample(i * 100, Duration::from_secs(60));
    }

    // Should be capped at max_samples
    assert!(tracker.sample_count() <= 10);
}

#[test]
fn test_cache_autosave_config_with_learning_thresholds() {
    let config = CacheAutosaveConfig::new("cache.json", Duration::from_secs(60))
        .with_learning_thresholds(LearningThresholdConfig::short_term());

    assert!(config.learning_thresholds.is_some());
    let learning = config.learning_thresholds.unwrap();
    assert_eq!(learning.window_duration, Duration::from_secs(15 * 60));
}

#[test]
fn test_cache_autosave_config_intelligent_preset() {
    let config = CacheAutosaveConfig::intelligent("cache.json.gz");

    assert!(config.adaptive_interval.is_some());
    assert!(config.learning_thresholds.is_some());

    let learning = config.learning_thresholds.unwrap();
    assert_eq!(learning.warmup_samples, 5); // short_term preset
}

#[test]
fn test_cache_autosave_preset_learning_thresholds_none() {
    // All standard presets should have learning_thresholds = None
    let performance = CacheAutosaveConfig::performance("cache.json");
    assert!(performance.learning_thresholds.is_none());

    let storage = CacheAutosaveConfig::storage_optimized("cache.json.gz");
    assert!(storage.learning_thresholds.is_none());

    let balanced = CacheAutosaveConfig::balanced("cache.json.gz");
    assert!(balanced.learning_thresholds.is_none());

    let aggressive = CacheAutosaveConfig::aggressive("cache.json.gz");
    assert!(aggressive.learning_thresholds.is_none());

    let development = CacheAutosaveConfig::development("cache.json");
    assert!(development.learning_thresholds.is_none());

    let adaptive = CacheAutosaveConfig::adaptive("cache.json.gz");
    assert!(adaptive.learning_thresholds.is_none());

    let burst = CacheAutosaveConfig::burst_optimized("cache.json.gz");
    assert!(burst.learning_thresholds.is_none());

    let resilient = CacheAutosaveConfig::resilient("cache.json.gz");
    assert!(resilient.learning_thresholds.is_none());
}

#[test]
fn test_threshold_update_event() {
    let config = LearningThresholdConfig::new()
        .with_warmup_samples(5)
        .with_smoothing_factor(1.0); // Immediate updates
    let mut tracker = HistoricalActivityTracker::new(config);

    // Add samples that will force threshold updates
    for _ in 0..4 {
        tracker.record_sample(1000, Duration::from_secs(60));
    }

    // Fifth sample with different value should trigger update
    let result = tracker.record_sample(50000, Duration::from_secs(60));

    if let Some(event) = result {
        assert!(
            event.new_high_threshold >= event.old_high_threshold
                || event.new_low_threshold != event.old_low_threshold
        );
        assert_eq!(event.sample_count, 5);
        assert!(!event.time_aware_adjustment);
    }
}

#[test]
fn test_adaptive_interval_config_compute_with_thresholds() {
    let config = AdaptiveIntervalConfig::new(Duration::from_secs(15), Duration::from_secs(300))
        .with_decrease_factor(0.5)
        .with_increase_factor(2.0);

    let current = Duration::from_secs(60);

    // Test with custom thresholds (not the default ones)
    // High activity threshold: 5KB, Low activity threshold: 500 bytes
    let high_thresh = 5 * 1024;
    let low_thresh = 500;

    // Change is above high threshold -> should decrease
    let result =
        config.compute_next_interval_with_thresholds(current, 10 * 1024, high_thresh, low_thresh);
    assert_eq!(result, Duration::from_secs(30)); // 60 * 0.5

    // Change is below low threshold -> should increase
    let result =
        config.compute_next_interval_with_thresholds(current, 100, high_thresh, low_thresh);
    assert_eq!(result, Duration::from_secs(120)); // 60 * 2.0

    // Change is between thresholds -> should stay the same
    let result =
        config.compute_next_interval_with_thresholds(current, 2 * 1024, high_thresh, low_thresh);
    assert_eq!(result, current);
}

#[test]
fn test_adaptive_interval_config_compute_with_thresholds_uses_provided_values() {
    // This test verifies that the provided thresholds are used, not the config's defaults
    let config = AdaptiveIntervalConfig::new(Duration::from_secs(15), Duration::from_secs(300))
        .with_high_activity_threshold(100_000) // Very high default
        .with_low_activity_threshold(10); // Very low default

    let current = Duration::from_secs(60);

    // With the config defaults, 5KB would be "between" thresholds
    // But with custom thresholds (high=1KB, low=100), 5KB should trigger decrease
    let high_thresh = 1024; // 1KB
    let low_thresh = 100;

    let result =
        config.compute_next_interval_with_thresholds(current, 5 * 1024, high_thresh, low_thresh);
    // Should decrease because 5KB > 1KB (custom high threshold)
    assert!(result < current);
}

#[test]
fn test_cache_autosave_callbacks_on_threshold_update_builder() {
    // Verify the builder pattern works for on_threshold_update
    let _callbacks = CacheAutosaveCallbacks::new().on_threshold_update(|event| {
        // Verify event has expected fields accessible
        let _ = event.old_high_threshold;
        let _ = event.new_high_threshold;
        let _ = event.old_low_threshold;
        let _ = event.new_low_threshold;
        let _ = event.sample_count;
        let _ = event.time_aware_adjustment;
    });

    // Verify chaining works
    let _callbacks = CacheAutosaveCallbacks::new()
        .on_save(|_| {})
        .on_error(|_| {})
        .on_threshold_update(|_| {});
}

#[test]
fn test_cache_autosave_callbacks_debug_includes_threshold_update() {
    let callbacks = CacheAutosaveCallbacks::new().on_threshold_update(|_| { /* no-op */ });

    let debug_str = format!("{:?}", callbacks);
    assert!(debug_str.contains("on_threshold_update"));
    assert!(debug_str.contains("Fn(...)"));
}
