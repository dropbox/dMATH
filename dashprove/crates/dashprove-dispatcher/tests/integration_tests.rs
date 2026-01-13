//! Integration tests for dashprove-dispatcher
//!
//! These tests verify the public API of the dispatcher crate including:
//! - Dispatcher creation and configuration
//! - Backend registration and selection
//! - Parallel execution with concurrency control
//! - Result merging strategies
//! - ML-based backend selection integration

use dashprove_ai::{StrategyModel, StrategyPredictor};
use dashprove_backends::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use dashprove_dispatcher::{
    BackendRegistry, BackendSelector, Dispatcher, DispatcherConfig, ExecutorConfig, MergeStrategy,
    ParallelExecutor, PropertyAssignment, ResultMerger, Selection, SelectionMethod,
    SelectionMetrics, SelectionStrategy, VerificationSummary,
};
use dashprove_usl::ast::{
    ComparisonOp, Contract, Expr, Invariant, Param, Probabilistic, Property, Security, Spec,
    Temporal, TemporalExpr, Theorem, Type,
};
use dashprove_usl::typecheck::{typecheck, TypedSpec};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

// =============================================================================
// Test Mock Backend
// =============================================================================

/// Mock backend for testing dispatcher functionality
struct MockBackend {
    id: BackendId,
    supported: Vec<PropertyType>,
    health: HealthStatus,
    status: VerificationStatus,
    delay: Duration,
    should_fail: bool,
}

impl MockBackend {
    fn new(id: BackendId, supported: Vec<PropertyType>) -> Self {
        MockBackend {
            id,
            supported,
            health: HealthStatus::Healthy,
            status: VerificationStatus::Proven,
            delay: Duration::from_millis(10),
            should_fail: false,
        }
    }

    fn with_health(mut self, health: HealthStatus) -> Self {
        self.health = health;
        self
    }

    fn with_status(mut self, status: VerificationStatus) -> Self {
        self.status = status;
        self
    }

    fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    fn failing(mut self) -> Self {
        self.should_fail = true;
        self
    }
}

#[async_trait::async_trait]
impl VerificationBackend for MockBackend {
    fn id(&self) -> BackendId {
        self.id
    }

    fn supports(&self) -> Vec<PropertyType> {
        self.supported.clone()
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        tokio::time::sleep(self.delay).await;

        if self.should_fail {
            return Err(BackendError::VerificationFailed("Mock failure".into()));
        }

        Ok(BackendResult {
            backend: self.id,
            status: self.status.clone(),
            proof: Some("mock proof".into()),
            counterexample: None,
            diagnostics: vec![],
            time_taken: self.delay,
        })
    }

    async fn health_check(&self) -> HealthStatus {
        self.health.clone()
    }
}

// =============================================================================
// Test Fixtures
// =============================================================================

fn make_theorem_spec() -> TypedSpec {
    let spec = Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "test_theorem".into(),
            body: Expr::Bool(true),
        })],
    };
    typecheck(spec).unwrap()
}

fn make_invariant_spec() -> TypedSpec {
    let spec = Spec {
        types: vec![],
        properties: vec![Property::Invariant(Invariant {
            name: "test_invariant".into(),
            body: Expr::Bool(true),
        })],
    };
    typecheck(spec).unwrap()
}

fn make_temporal_spec() -> TypedSpec {
    let spec = Spec {
        types: vec![],
        properties: vec![Property::Temporal(Temporal {
            name: "test_temporal".into(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
            fairness: vec![],
        })],
    };
    typecheck(spec).unwrap()
}

fn make_contract_spec() -> TypedSpec {
    let spec = Spec {
        types: vec![],
        properties: vec![Property::Contract(Contract {
            type_path: vec!["TestType".into(), "test_method".into()],
            params: vec![Param {
                name: "x".into(),
                ty: Type::Named("Int".into()),
            }],
            return_type: Some(Type::Named("Bool".into())),
            requires: vec![Expr::Bool(true)],
            ensures: vec![Expr::Bool(true)],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        })],
    };
    typecheck(spec).unwrap()
}

fn make_probabilistic_spec() -> TypedSpec {
    let spec = Spec {
        types: vec![],
        properties: vec![Property::Probabilistic(Probabilistic {
            name: "test_prob".into(),
            condition: Expr::Bool(true),
            comparison: ComparisonOp::Ge,
            bound: 0.95,
        })],
    };
    typecheck(spec).unwrap()
}

fn make_security_spec() -> TypedSpec {
    let spec = Spec {
        types: vec![],
        properties: vec![Property::Security(Security {
            name: "test_security".into(),
            body: Expr::Bool(true),
        })],
    };
    typecheck(spec).unwrap()
}

fn make_multi_property_spec() -> TypedSpec {
    let spec = Spec {
        types: vec![],
        properties: vec![
            Property::Theorem(Theorem {
                name: "theorem1".into(),
                body: Expr::Bool(true),
            }),
            Property::Invariant(Invariant {
                name: "invariant1".into(),
                body: Expr::Bool(true),
            }),
            Property::Temporal(Temporal {
                name: "temporal1".into(),
                body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
                fairness: vec![],
            }),
        ],
    };
    typecheck(spec).unwrap()
}

// =============================================================================
// Dispatcher Configuration Tests
// =============================================================================

#[test]
fn test_dispatcher_config_default() {
    let config = DispatcherConfig::default();
    assert!(matches!(
        config.selection_strategy,
        SelectionStrategy::Single
    ));
    assert!(matches!(config.merge_strategy, MergeStrategy::FirstSuccess));
    assert_eq!(config.max_concurrent, 4);
    assert!(config.check_health);
}

#[test]
fn test_dispatcher_config_redundant() {
    let config = DispatcherConfig::redundant(3);
    assert!(matches!(
        config.selection_strategy,
        SelectionStrategy::Redundant { min_backends: 3 }
    ));
    assert!(matches!(config.merge_strategy, MergeStrategy::Unanimous));
}

#[test]
fn test_dispatcher_config_all_backends() {
    let config = DispatcherConfig::all_backends();
    assert!(matches!(config.selection_strategy, SelectionStrategy::All));
    assert!(matches!(config.merge_strategy, MergeStrategy::Majority));
}

#[test]
fn test_dispatcher_config_specific() {
    let config = DispatcherConfig::specific(BackendId::Lean4);
    assert!(matches!(
        config.selection_strategy,
        SelectionStrategy::Specific(BackendId::Lean4)
    ));
}

#[test]
fn test_dispatcher_config_ml_based() {
    let config = DispatcherConfig::ml_based(0.5);
    assert!(matches!(
        config.selection_strategy,
        SelectionStrategy::MlBased { min_confidence } if min_confidence == 0.5
    ));
}

#[test]
fn test_dispatcher_config_ml_based_clamping() {
    // Test clamping of confidence values
    let config_high = DispatcherConfig::ml_based(1.5);
    assert!(matches!(
        config_high.selection_strategy,
        SelectionStrategy::MlBased { min_confidence } if min_confidence == 1.0
    ));

    let config_low = DispatcherConfig::ml_based(-0.5);
    assert!(matches!(
        config_low.selection_strategy,
        SelectionStrategy::MlBased { min_confidence } if min_confidence == 0.0
    ));
}

#[test]
fn test_dispatcher_config_knowledge_enhanced() {
    let config = DispatcherConfig::knowledge_enhanced(0.5);
    assert!(matches!(
        config.selection_strategy,
        SelectionStrategy::KnowledgeEnhanced { min_confidence } if min_confidence == 0.5
    ));
}

#[test]
fn test_dispatcher_config_knowledge_enhanced_clamping() {
    // Test clamping of confidence values
    let config_high = DispatcherConfig::knowledge_enhanced(1.5);
    assert!(matches!(
        config_high.selection_strategy,
        SelectionStrategy::KnowledgeEnhanced { min_confidence } if min_confidence == 1.0
    ));

    let config_low = DispatcherConfig::knowledge_enhanced(-0.5);
    assert!(matches!(
        config_low.selection_strategy,
        SelectionStrategy::KnowledgeEnhanced { min_confidence } if min_confidence == 0.0
    ));
}

#[test]
fn test_dispatcher_has_knowledge_stores() {
    // Dispatcher without knowledge stores
    let dispatcher = Dispatcher::new(DispatcherConfig::default());
    assert!(!dispatcher.has_knowledge_stores());
}

// =============================================================================
// Backend Registry Tests
// =============================================================================

#[test]
fn test_registry_register_and_get() {
    let mut registry = BackendRegistry::new();
    assert!(registry.is_empty());

    registry.register(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    )));

    assert_eq!(registry.len(), 1);
    assert!(!registry.is_empty());
    assert!(registry.get(BackendId::Lean4).is_some());
    assert!(registry.get(BackendId::TlaPlus).is_none());
}

#[test]
fn test_registry_backends_for_type() {
    let mut registry = BackendRegistry::new();

    registry.register(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem, PropertyType::Invariant],
    )));
    registry.register(Arc::new(MockBackend::new(
        BackendId::Alloy,
        vec![PropertyType::Invariant],
    )));
    registry.register(Arc::new(MockBackend::new(
        BackendId::TlaPlus,
        vec![PropertyType::Temporal],
    )));

    let theorem_backends = registry.backends_for_type(PropertyType::Theorem);
    assert_eq!(theorem_backends.len(), 1);
    assert!(theorem_backends.contains(&BackendId::Lean4));

    let invariant_backends = registry.backends_for_type(PropertyType::Invariant);
    assert_eq!(invariant_backends.len(), 2);

    let temporal_backends = registry.backends_for_type(PropertyType::Temporal);
    assert_eq!(temporal_backends.len(), 1);
    assert!(temporal_backends.contains(&BackendId::TlaPlus));
}

#[test]
fn test_registry_all_backends() {
    let mut registry = BackendRegistry::new();

    registry.register(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    )));
    registry.register(Arc::new(MockBackend::new(
        BackendId::TlaPlus,
        vec![PropertyType::Temporal],
    )));
    registry.register(Arc::new(MockBackend::new(
        BackendId::Kani,
        vec![PropertyType::Contract],
    )));

    let all = registry.all_backends();
    assert_eq!(all.len(), 3);
    assert!(all.contains(&BackendId::Lean4));
    assert!(all.contains(&BackendId::TlaPlus));
    assert!(all.contains(&BackendId::Kani));
}

#[test]
fn test_registry_get_info() {
    let mut registry = BackendRegistry::new();

    registry.register(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem, PropertyType::Invariant],
    )));

    let info = registry.get_info(BackendId::Lean4);
    assert!(info.is_some());
    let info = info.unwrap();
    assert_eq!(info.id, BackendId::Lean4);
    assert_eq!(info.supported_types.len(), 2);
    assert!(matches!(info.health, HealthStatus::Healthy));
}

#[test]
fn test_registry_update_health() {
    let mut registry = BackendRegistry::new();

    registry.register(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    )));

    registry.update_health(
        BackendId::Lean4,
        HealthStatus::Degraded {
            reason: "high latency".into(),
        },
    );

    let info = registry.get_info(BackendId::Lean4).unwrap();
    assert!(matches!(info.health, HealthStatus::Degraded { .. }));
}

#[test]
fn test_registry_healthy_backends_sorted_by_priority() {
    let mut registry = BackendRegistry::new();

    // Alloy has priority 90, Lean4 has priority 100
    registry.register(Arc::new(MockBackend::new(
        BackendId::Alloy,
        vec![PropertyType::Invariant],
    )));
    registry.register(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Invariant],
    )));

    let healthy = registry.healthy_backends_for_type(PropertyType::Invariant);
    // Lean4 (priority 100) should come before Alloy (priority 90)
    assert_eq!(healthy[0], BackendId::Lean4);
    assert_eq!(healthy[1], BackendId::Alloy);
}

#[tokio::test]
async fn test_registry_check_all_health() {
    let mut registry = BackendRegistry::new();

    registry.register(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    )));
    registry.register(Arc::new(
        MockBackend::new(BackendId::Alloy, vec![PropertyType::Invariant]).with_health(
            HealthStatus::Unavailable {
                reason: "not installed".into(),
            },
        ),
    ));

    registry.check_all_health().await;

    // After health check, the mock backend returns its preset health
    let lean_info = registry.get_info(BackendId::Lean4).unwrap();
    assert!(matches!(lean_info.health, HealthStatus::Healthy));
}

// =============================================================================
// Backend Selector Tests
// =============================================================================

#[test]
fn test_selector_single_strategy() {
    let mut registry = BackendRegistry::new();
    registry.register(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    )));

    let selector = BackendSelector::new(&registry, SelectionStrategy::Single);
    let properties = vec![Property::Theorem(Theorem {
        name: "test".into(),
        body: Expr::Bool(true),
    })];

    let selection = selector.select(&properties).unwrap();
    assert_eq!(selection.assignments.len(), 1);
    assert_eq!(selection.assignments[0].backends, vec![BackendId::Lean4]);
}

#[test]
fn test_selector_all_strategy() {
    let mut registry = BackendRegistry::new();
    registry.register(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Invariant],
    )));
    registry.register(Arc::new(MockBackend::new(
        BackendId::Alloy,
        vec![PropertyType::Invariant],
    )));

    let selector = BackendSelector::new(&registry, SelectionStrategy::All);
    let properties = vec![Property::Invariant(Invariant {
        name: "test".into(),
        body: Expr::Bool(true),
    })];

    let selection = selector.select(&properties).unwrap();
    assert_eq!(selection.assignments[0].backends.len(), 2);
}

#[test]
fn test_selector_specific_strategy() {
    let mut registry = BackendRegistry::new();
    registry.register(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    )));
    registry.register(Arc::new(MockBackend::new(
        BackendId::Coq,
        vec![PropertyType::Theorem],
    )));

    let selector = BackendSelector::new(&registry, SelectionStrategy::Specific(BackendId::Coq));
    let properties = vec![Property::Theorem(Theorem {
        name: "test".into(),
        body: Expr::Bool(true),
    })];

    let selection = selector.select(&properties).unwrap();
    assert_eq!(selection.assignments[0].backends, vec![BackendId::Coq]);
}

#[test]
fn test_selector_redundant_strategy() {
    let mut registry = BackendRegistry::new();
    registry.register(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    )));
    registry.register(Arc::new(MockBackend::new(
        BackendId::Coq,
        vec![PropertyType::Theorem],
    )));
    registry.register(Arc::new(MockBackend::new(
        BackendId::Isabelle,
        vec![PropertyType::Theorem],
    )));

    let selector =
        BackendSelector::new(&registry, SelectionStrategy::Redundant { min_backends: 2 });
    let properties = vec![Property::Theorem(Theorem {
        name: "test".into(),
        body: Expr::Bool(true),
    })];

    let selection = selector.select(&properties).unwrap();
    assert_eq!(selection.assignments[0].backends.len(), 2);
}

#[test]
fn test_selector_ml_based_with_predictor() {
    let mut registry = BackendRegistry::new();
    registry.register(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    )));
    registry.register(Arc::new(MockBackend::new(
        BackendId::Coq,
        vec![PropertyType::Theorem],
    )));

    let predictor = Arc::new(StrategyModel::from(StrategyPredictor::new()));
    let selector = BackendSelector::with_ml_predictor(
        &registry,
        SelectionStrategy::MlBased {
            min_confidence: 0.0,
        },
        predictor,
    );

    let properties = vec![Property::Theorem(Theorem {
        name: "test".into(),
        body: Expr::Bool(true),
    })];

    let selection = selector.select(&properties).unwrap();
    assert!(!selection.assignments[0].backends.is_empty());
}

#[test]
fn test_selector_property_type_mapping() {
    // Verify all property types are correctly mapped
    assert_eq!(
        BackendSelector::property_type(&Property::Theorem(Theorem {
            name: "t".into(),
            body: Expr::Bool(true)
        })),
        PropertyType::Theorem
    );

    assert_eq!(
        BackendSelector::property_type(&Property::Invariant(Invariant {
            name: "i".into(),
            body: Expr::Bool(true)
        })),
        PropertyType::Invariant
    );

    assert_eq!(
        BackendSelector::property_type(&Property::Temporal(Temporal {
            name: "t".into(),
            body: TemporalExpr::Atom(Expr::Bool(true)),
            fairness: vec![],
        })),
        PropertyType::Temporal
    );

    assert_eq!(
        BackendSelector::property_type(&Property::Contract(Contract {
            type_path: vec!["T".into()],
            params: vec![],
            return_type: None,
            requires: vec![],
            ensures: vec![],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        })),
        PropertyType::Contract
    );

    assert_eq!(
        BackendSelector::property_type(&Property::Probabilistic(Probabilistic {
            name: "p".into(),
            condition: Expr::Bool(true),
            comparison: ComparisonOp::Ge,
            bound: 0.5
        })),
        PropertyType::Probabilistic
    );

    assert_eq!(
        BackendSelector::property_type(&Property::Security(Security {
            name: "s".into(),
            body: Expr::Bool(true)
        })),
        PropertyType::SecurityProtocol
    );
}

// =============================================================================
// Result Merger Tests
// =============================================================================

#[test]
fn test_merger_first_success_strategy() {
    let merger = ResultMerger::new(MergeStrategy::FirstSuccess);

    let mut by_property = HashMap::new();
    by_property.insert(
        0,
        vec![dashprove_dispatcher::TaskResult {
            property_index: 0,
            backend: BackendId::Lean4,
            result: Ok(BackendResult {
                backend: BackendId::Lean4,
                status: VerificationStatus::Proven,
                proof: Some("proof".into()),
                counterexample: None,
                diagnostics: vec![],
                time_taken: Duration::from_millis(100),
            }),
        }],
    );

    let property_types = HashMap::from([(0usize, PropertyType::Theorem)]);

    let exec_results = dashprove_dispatcher::ExecutionResults {
        by_property,
        property_types,
        total_time: Duration::from_secs(1),
        successful: 1,
        failed: 0,
    };

    let merged = merger.merge(exec_results);
    assert_eq!(merged.summary.proven, 1);
    assert_eq!(merged.summary.disproven, 0);
}

#[test]
fn test_merger_unanimous_strategy() {
    let merger = ResultMerger::new(MergeStrategy::Unanimous);

    let mut by_property = HashMap::new();
    by_property.insert(
        0,
        vec![
            dashprove_dispatcher::TaskResult {
                property_index: 0,
                backend: BackendId::Lean4,
                result: Ok(BackendResult {
                    backend: BackendId::Lean4,
                    status: VerificationStatus::Proven,
                    proof: Some("proof1".into()),
                    counterexample: None,
                    diagnostics: vec![],
                    time_taken: Duration::from_millis(100),
                }),
            },
            dashprove_dispatcher::TaskResult {
                property_index: 0,
                backend: BackendId::Coq,
                result: Ok(BackendResult {
                    backend: BackendId::Coq,
                    status: VerificationStatus::Proven,
                    proof: Some("proof2".into()),
                    counterexample: None,
                    diagnostics: vec![],
                    time_taken: Duration::from_millis(150),
                }),
            },
        ],
    );

    let property_types = HashMap::from([(0usize, PropertyType::Theorem)]);

    let exec_results = dashprove_dispatcher::ExecutionResults {
        by_property,
        property_types,
        total_time: Duration::from_secs(1),
        successful: 2,
        failed: 0,
    };

    let merged = merger.merge(exec_results);
    assert_eq!(merged.summary.proven, 1);
    assert_eq!(merged.properties[0].confidence, 1.0);
}

#[test]
fn test_merger_majority_strategy() {
    let merger = ResultMerger::new(MergeStrategy::Majority);

    let mut by_property = HashMap::new();
    by_property.insert(
        0,
        vec![
            dashprove_dispatcher::TaskResult {
                property_index: 0,
                backend: BackendId::Lean4,
                result: Ok(BackendResult {
                    backend: BackendId::Lean4,
                    status: VerificationStatus::Proven,
                    proof: Some("proof".into()),
                    counterexample: None,
                    diagnostics: vec![],
                    time_taken: Duration::from_millis(100),
                }),
            },
            dashprove_dispatcher::TaskResult {
                property_index: 0,
                backend: BackendId::Coq,
                result: Ok(BackendResult {
                    backend: BackendId::Coq,
                    status: VerificationStatus::Proven,
                    proof: Some("proof".into()),
                    counterexample: None,
                    diagnostics: vec![],
                    time_taken: Duration::from_millis(100),
                }),
            },
            dashprove_dispatcher::TaskResult {
                property_index: 0,
                backend: BackendId::Isabelle,
                result: Ok(BackendResult {
                    backend: BackendId::Isabelle,
                    status: VerificationStatus::Disproven,
                    proof: None,
                    counterexample: None,
                    diagnostics: vec![],
                    time_taken: Duration::from_millis(100),
                }),
            },
        ],
    );

    let property_types = HashMap::from([(0usize, PropertyType::Theorem)]);

    let exec_results = dashprove_dispatcher::ExecutionResults {
        by_property,
        property_types,
        total_time: Duration::from_secs(1),
        successful: 3,
        failed: 0,
    };

    let merged = merger.merge(exec_results);
    // 2 proven vs 1 disproven = majority says proven
    assert_eq!(merged.summary.proven, 1);
}

#[test]
fn test_merger_pessimistic_strategy() {
    let merger = ResultMerger::new(MergeStrategy::Pessimistic);

    let mut by_property = HashMap::new();
    by_property.insert(
        0,
        vec![
            dashprove_dispatcher::TaskResult {
                property_index: 0,
                backend: BackendId::Lean4,
                result: Ok(BackendResult {
                    backend: BackendId::Lean4,
                    status: VerificationStatus::Proven,
                    proof: Some("proof".into()),
                    counterexample: None,
                    diagnostics: vec![],
                    time_taken: Duration::from_millis(100),
                }),
            },
            dashprove_dispatcher::TaskResult {
                property_index: 0,
                backend: BackendId::Alloy,
                result: Ok(BackendResult {
                    backend: BackendId::Alloy,
                    status: VerificationStatus::Disproven,
                    proof: None,
                    counterexample: None,
                    diagnostics: vec![],
                    time_taken: Duration::from_millis(100),
                }),
            },
        ],
    );

    let property_types = HashMap::from([(0usize, PropertyType::Theorem)]);

    let exec_results = dashprove_dispatcher::ExecutionResults {
        by_property,
        property_types,
        total_time: Duration::from_secs(1),
        successful: 2,
        failed: 0,
    };

    let merged = merger.merge(exec_results);
    // Pessimistic prefers disproven
    assert_eq!(merged.summary.disproven, 1);
}

#[test]
fn test_merger_optimistic_strategy() {
    let merger = ResultMerger::new(MergeStrategy::Optimistic);

    let mut by_property = HashMap::new();
    by_property.insert(
        0,
        vec![
            dashprove_dispatcher::TaskResult {
                property_index: 0,
                backend: BackendId::Lean4,
                result: Ok(BackendResult {
                    backend: BackendId::Lean4,
                    status: VerificationStatus::Unknown {
                        reason: "timeout".into(),
                    },
                    proof: None,
                    counterexample: None,
                    diagnostics: vec![],
                    time_taken: Duration::from_millis(100),
                }),
            },
            dashprove_dispatcher::TaskResult {
                property_index: 0,
                backend: BackendId::Alloy,
                result: Ok(BackendResult {
                    backend: BackendId::Alloy,
                    status: VerificationStatus::Proven,
                    proof: Some("proof".into()),
                    counterexample: None,
                    diagnostics: vec![],
                    time_taken: Duration::from_millis(100),
                }),
            },
        ],
    );

    let property_types = HashMap::from([(0usize, PropertyType::Theorem)]);

    let exec_results = dashprove_dispatcher::ExecutionResults {
        by_property,
        property_types,
        total_time: Duration::from_secs(1),
        successful: 2,
        failed: 0,
    };

    let merged = merger.merge(exec_results);
    // Optimistic prefers proven
    assert_eq!(merged.summary.proven, 1);
}

// =============================================================================
// Parallel Executor Tests
// =============================================================================

#[tokio::test]
async fn test_executor_single_task() {
    let executor = ParallelExecutor::default();
    let backend: Arc<dyn VerificationBackend> = Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    ));
    let spec = make_theorem_spec();

    let result = executor.execute_single(&backend, &spec).await;
    assert!(result.is_ok());
    assert!(matches!(result.unwrap().status, VerificationStatus::Proven));
}

#[tokio::test]
async fn test_executor_parallel_tasks() {
    let executor = ParallelExecutor::new(ExecutorConfig {
        max_concurrent: 4,
        task_timeout: Duration::from_secs(30),
        fail_fast: false,
        ..Default::default()
    });

    let mut backends: HashMap<BackendId, Arc<dyn VerificationBackend>> = HashMap::new();
    backends.insert(
        BackendId::Lean4,
        Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Invariant],
        )),
    );
    backends.insert(
        BackendId::Alloy,
        Arc::new(MockBackend::new(
            BackendId::Alloy,
            vec![PropertyType::Invariant],
        )),
    );

    let selection = Selection {
        assignments: vec![PropertyAssignment {
            property_index: 0,
            property_type: PropertyType::Invariant,
            backends: vec![BackendId::Lean4, BackendId::Alloy],
            selection_method: SelectionMethod::RuleBased,
        }],
        warnings: vec![],
        metrics: SelectionMetrics::default(),
    };

    let spec = make_invariant_spec();
    let results = executor
        .execute(&selection, &spec, &backends)
        .await
        .unwrap();

    assert_eq!(results.successful, 2);
    assert_eq!(results.failed, 0);
}

#[tokio::test]
async fn test_executor_timeout() {
    let executor = ParallelExecutor::new(ExecutorConfig {
        max_concurrent: 1,
        task_timeout: Duration::from_millis(10),
        fail_fast: false,
        ..Default::default()
    });

    let backend: Arc<dyn VerificationBackend> = Arc::new(
        MockBackend::new(BackendId::Lean4, vec![PropertyType::Theorem])
            .with_delay(Duration::from_secs(1)),
    );
    let spec = make_theorem_spec();

    let result = executor.execute_single(&backend, &spec).await;
    assert!(matches!(
        result,
        Err(dashprove_dispatcher::ExecutorError::Timeout(_))
    ));
}

#[tokio::test]
async fn test_executor_concurrency_limit() {
    let executor = ParallelExecutor::new(ExecutorConfig {
        max_concurrent: 1, // Only one at a time
        task_timeout: Duration::from_secs(30),
        fail_fast: false,
        ..Default::default()
    });

    let mut backends: HashMap<BackendId, Arc<dyn VerificationBackend>> = HashMap::new();
    backends.insert(
        BackendId::Lean4,
        Arc::new(
            MockBackend::new(BackendId::Lean4, vec![PropertyType::Invariant])
                .with_delay(Duration::from_millis(50)),
        ),
    );
    backends.insert(
        BackendId::Alloy,
        Arc::new(
            MockBackend::new(BackendId::Alloy, vec![PropertyType::Invariant])
                .with_delay(Duration::from_millis(50)),
        ),
    );

    let selection = Selection {
        assignments: vec![PropertyAssignment {
            property_index: 0,
            property_type: PropertyType::Invariant,
            backends: vec![BackendId::Lean4, BackendId::Alloy],
            selection_method: SelectionMethod::RuleBased,
        }],
        warnings: vec![],
        metrics: SelectionMetrics::default(),
    };

    let spec = make_invariant_spec();
    let start = std::time::Instant::now();
    let results = executor
        .execute(&selection, &spec, &backends)
        .await
        .unwrap();
    let elapsed = start.elapsed();

    assert_eq!(results.successful, 2);
    // With max_concurrent=1, should run sequentially (at least 100ms total)
    assert!(elapsed >= Duration::from_millis(100));
}

// =============================================================================
// Full Dispatcher Integration Tests
// =============================================================================

#[tokio::test]
async fn test_dispatcher_basic_verification() {
    let mut dispatcher = Dispatcher::default();

    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    )));

    let spec = make_theorem_spec();
    let results = dispatcher.verify(&spec).await.unwrap();

    assert_eq!(results.summary.proven, 1);
    assert_eq!(results.properties.len(), 1);
}

#[tokio::test]
async fn test_dispatcher_no_backends_error() {
    let mut dispatcher = Dispatcher::default();
    let spec = make_theorem_spec();

    let result = dispatcher.verify(&spec).await;
    assert!(matches!(
        result,
        Err(dashprove_dispatcher::DispatcherError::NoBackends)
    ));
}

#[tokio::test]
async fn test_dispatcher_specific_backend() {
    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::Alloy));

    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Invariant],
    )));
    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Alloy,
        vec![PropertyType::Invariant],
    )));

    let spec = make_invariant_spec();
    let results = dispatcher.verify(&spec).await.unwrap();

    assert_eq!(results.properties[0].backend_results.len(), 1);
    assert_eq!(
        results.properties[0].backend_results[0].backend,
        BackendId::Alloy
    );
}

#[tokio::test]
async fn test_dispatcher_all_backends() {
    let mut dispatcher = Dispatcher::new(DispatcherConfig::all_backends());

    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Invariant],
    )));
    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Alloy,
        vec![PropertyType::Invariant],
    )));

    let spec = make_invariant_spec();
    let results = dispatcher.verify(&spec).await.unwrap();

    assert_eq!(results.properties[0].backend_results.len(), 2);
}

#[tokio::test]
async fn test_dispatcher_redundant_verification() {
    let mut dispatcher = Dispatcher::new(DispatcherConfig::redundant(2));

    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    )));
    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Coq,
        vec![PropertyType::Theorem],
    )));

    let spec = make_theorem_spec();
    let results = dispatcher.verify(&spec).await.unwrap();

    assert_eq!(results.properties[0].backend_results.len(), 2);
    assert_eq!(results.summary.overall_confidence, 1.0);
}

#[tokio::test]
async fn test_dispatcher_verify_with() {
    let mut dispatcher = Dispatcher::default();

    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    )));
    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Coq,
        vec![PropertyType::Theorem],
    )));

    let spec = make_theorem_spec();
    let results = dispatcher.verify_with(&spec, BackendId::Coq).await.unwrap();

    assert_eq!(results.properties[0].backend_results.len(), 1);
    assert_eq!(
        results.properties[0].backend_results[0].backend,
        BackendId::Coq
    );
}

#[tokio::test]
async fn test_dispatcher_disproven_result() {
    let mut dispatcher = Dispatcher::default();

    dispatcher.register_backend(Arc::new(
        MockBackend::new(BackendId::Lean4, vec![PropertyType::Theorem])
            .with_status(VerificationStatus::Disproven),
    ));

    let spec = make_theorem_spec();
    let results = dispatcher.verify(&spec).await.unwrap();

    assert_eq!(results.summary.disproven, 1);
    assert_eq!(results.summary.proven, 0);
}

#[tokio::test]
async fn test_dispatcher_ml_based_selection() {
    let predictor = Arc::new(StrategyModel::from(StrategyPredictor::new()));
    let config = DispatcherConfig::ml_based(0.0);
    let mut dispatcher = Dispatcher::with_ml_predictor(config, predictor);

    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    )));

    let spec = make_theorem_spec();
    let results = dispatcher.verify(&spec).await.unwrap();

    assert_eq!(results.summary.proven, 1);
}

#[tokio::test]
async fn test_dispatcher_set_ml_predictor() {
    let mut dispatcher = Dispatcher::default();
    assert!(dispatcher.ml_predictor().is_none());

    let predictor = Arc::new(StrategyModel::from(StrategyPredictor::new()));
    dispatcher.set_ml_predictor(predictor);
    assert!(dispatcher.ml_predictor().is_some());
}

#[tokio::test]
async fn test_dispatcher_multi_property_spec() {
    let mut dispatcher = Dispatcher::default();

    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem, PropertyType::Invariant],
    )));
    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::TlaPlus,
        vec![PropertyType::Temporal],
    )));

    let spec = make_multi_property_spec();
    let results = dispatcher.verify(&spec).await.unwrap();

    assert_eq!(results.summary.proven, 3);
    assert_eq!(results.properties.len(), 3);
}

#[tokio::test]
async fn test_dispatcher_config_update() {
    let mut dispatcher = Dispatcher::default();

    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    )));

    // Initial config
    assert!(matches!(
        dispatcher.config().selection_strategy,
        SelectionStrategy::Single
    ));

    // Update config
    dispatcher.set_config(DispatcherConfig::all_backends());
    assert!(matches!(
        dispatcher.config().selection_strategy,
        SelectionStrategy::All
    ));
}

#[tokio::test]
async fn test_dispatcher_registry_access() {
    let mut dispatcher = Dispatcher::default();

    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    )));

    // Read-only access
    assert_eq!(dispatcher.registry().len(), 1);

    // Mutable access
    dispatcher
        .registry_mut()
        .register(Arc::new(MockBackend::new(
            BackendId::TlaPlus,
            vec![PropertyType::Temporal],
        )));

    assert_eq!(dispatcher.registry().len(), 2);
}

// =============================================================================
// Property-Type Specific Backend Tests
// =============================================================================

#[tokio::test]
async fn test_temporal_verification_with_tlaplus() {
    let mut dispatcher = Dispatcher::default();

    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::TlaPlus,
        vec![PropertyType::Temporal],
    )));

    let spec = make_temporal_spec();
    let results = dispatcher.verify(&spec).await.unwrap();

    assert_eq!(results.summary.proven, 1);
    assert_eq!(
        results.properties[0].backend_results[0].backend,
        BackendId::TlaPlus
    );
}

#[tokio::test]
async fn test_contract_verification_with_kani() {
    let mut dispatcher = Dispatcher::default();

    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Kani,
        vec![PropertyType::Contract],
    )));

    let spec = make_contract_spec();
    let results = dispatcher.verify(&spec).await.unwrap();

    assert_eq!(results.summary.proven, 1);
    assert_eq!(
        results.properties[0].backend_results[0].backend,
        BackendId::Kani
    );
}

#[tokio::test]
async fn test_probabilistic_verification_with_storm() {
    let mut dispatcher = Dispatcher::default();

    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Storm,
        vec![PropertyType::Probabilistic],
    )));

    let spec = make_probabilistic_spec();
    let results = dispatcher.verify(&spec).await.unwrap();

    assert_eq!(results.summary.proven, 1);
    assert_eq!(
        results.properties[0].backend_results[0].backend,
        BackendId::Storm
    );
}

#[tokio::test]
async fn test_security_verification_with_tamarin() {
    let mut dispatcher = Dispatcher::default();

    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Tamarin,
        vec![PropertyType::SecurityProtocol],
    )));

    let spec = make_security_spec();
    let results = dispatcher.verify(&spec).await.unwrap();

    assert_eq!(results.summary.proven, 1);
    assert_eq!(
        results.properties[0].backend_results[0].backend,
        BackendId::Tamarin
    );
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[tokio::test]
async fn test_dispatcher_handles_backend_failure() {
    let mut dispatcher = Dispatcher::default();

    dispatcher.register_backend(Arc::new(
        MockBackend::new(BackendId::Lean4, vec![PropertyType::Theorem]).failing(),
    ));

    let spec = make_theorem_spec();
    let results = dispatcher.verify(&spec).await.unwrap();

    // Backend failure should result in unknown status
    assert_eq!(results.summary.unknown, 1);
    assert!(results.properties[0].backend_results[0].error.is_some());
}

#[tokio::test]
async fn test_dispatcher_partial_verification() {
    let mut dispatcher = Dispatcher::new(DispatcherConfig::all_backends());

    // One succeeds, one fails
    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    )));
    dispatcher.register_backend(Arc::new(
        MockBackend::new(BackendId::Coq, vec![PropertyType::Theorem]).failing(),
    ));

    let spec = make_theorem_spec();
    let results = dispatcher.verify(&spec).await.unwrap();

    // Should still produce a result even with partial failure
    assert_eq!(results.properties[0].backend_results.len(), 2);
}

// =============================================================================
// Re-export Tests
// =============================================================================

#[test]
fn test_public_api_exports() {
    // Verify all expected types are exported from the crate root
    let _config: DispatcherConfig = DispatcherConfig::default();
    let _selection_strategy: SelectionStrategy = SelectionStrategy::Single;
    let _merge_strategy: MergeStrategy = MergeStrategy::FirstSuccess;

    // Verify builder-style config creation
    let _redundant = DispatcherConfig::redundant(2);
    let _all = DispatcherConfig::all_backends();
    let _specific = DispatcherConfig::specific(BackendId::Lean4);
    let _ml = DispatcherConfig::ml_based(0.5);
}

#[test]
fn test_verification_summary_fields() {
    let summary = VerificationSummary {
        proven: 5,
        disproven: 1,
        unknown: 2,
        partial: 1,
        overall_confidence: 0.85,
    };

    assert_eq!(summary.proven, 5);
    assert_eq!(summary.disproven, 1);
    assert_eq!(summary.unknown, 2);
    assert_eq!(summary.partial, 1);
    assert_eq!(summary.overall_confidence, 0.85);
}

#[test]
fn test_executor_config_default() {
    let config = ExecutorConfig::default();
    assert_eq!(config.max_concurrent, 4);
    assert_eq!(config.task_timeout, Duration::from_secs(300));
    assert!(!config.fail_fast);
}

// =============================================================================
// Full Pipeline Integration Tests (Corpus -> Bootstrap -> Domain-Weighted Verify)
// =============================================================================

/// Test the full pipeline from proof corpus to domain-weighted verification
#[tokio::test]
async fn test_corpus_to_domain_weighted_verification_pipeline() {
    use dashprove_learning::{
        LearnableResult, ProofCorpus, ReputationConfig, ReputationFromCorpus,
    };

    // Step 1: Create a proof corpus with entries from multiple backends and property types
    let mut corpus = ProofCorpus::new();

    // Add several theorem proofs for Lean4 (good performance)
    for i in 0..10 {
        let result = LearnableResult {
            property: Property::Theorem(Theorem {
                name: format!("lean_theorem_{}", i),
                body: Expr::Bool(true),
            }),
            backend: BackendId::Lean4,
            status: VerificationStatus::Proven,
            tactics: vec!["simp".to_string()],
            time_taken: Duration::from_millis(100),
            proof_output: Some("proof".to_string()),
        };
        corpus.insert(&result);
    }

    // Add a few theorem proofs for Coq (fewer, to test weighting)
    for i in 0..3 {
        let result = LearnableResult {
            property: Property::Theorem(Theorem {
                name: format!("coq_theorem_{}", i),
                body: Expr::Bool(true),
            }),
            backend: BackendId::Coq,
            status: VerificationStatus::Proven,
            tactics: vec!["auto".to_string()],
            time_taken: Duration::from_millis(200),
            proof_output: Some("proof".to_string()),
        };
        corpus.insert(&result);
    }

    // Add invariant proofs for Alloy (specializes in invariants)
    for i in 0..8 {
        let result = LearnableResult {
            property: Property::Invariant(Invariant {
                name: format!("alloy_invariant_{}", i),
                body: Expr::Bool(true),
            }),
            backend: BackendId::Alloy,
            status: VerificationStatus::Proven,
            tactics: vec![],
            time_taken: Duration::from_millis(150),
            proof_output: Some("proof".to_string()),
        };
        corpus.insert(&result);
    }

    // Add temporal proofs for TLA+
    for i in 0..5 {
        let result = LearnableResult {
            property: Property::Temporal(Temporal {
                name: format!("tlaplus_temporal_{}", i),
                body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
                fairness: vec![],
            }),
            backend: BackendId::TlaPlus,
            status: VerificationStatus::Proven,
            tactics: vec![],
            time_taken: Duration::from_millis(300),
            proof_output: Some("proof".to_string()),
        };
        corpus.insert(&result);
    }

    assert_eq!(corpus.len(), 26);

    // Step 2: Bootstrap reputation tracker from corpus with domain stats
    let (tracker, stats) = ReputationFromCorpus::new(&corpus)
        .with_config(ReputationConfig::default())
        .with_domain_stats(true)
        .build_with_stats();

    // Verify bootstrapping captured domain stats
    assert_eq!(stats.total_entries, 26);
    assert!(stats.domain_coverage() > 90.0); // Most entries should have property types
    assert!(tracker.domain_count() > 0);

    // Verify domain-specific weights were computed
    let domain_weights = tracker.compute_domain_weights();
    let aggregate_weights = tracker.compute_weights();

    // Lean4 should have high weight for theorems
    let lean_theorem_key =
        dashprove_learning::DomainKey::new(BackendId::Lean4, PropertyType::Theorem);
    if let Some(&weight) = domain_weights.get(&lean_theorem_key) {
        assert!(
            weight > 0.5,
            "Lean4 should have good reputation for theorems"
        );
    }

    // Alloy should have high weight for invariants
    let alloy_invariant_key =
        dashprove_learning::DomainKey::new(BackendId::Alloy, PropertyType::Invariant);
    if let Some(&weight) = domain_weights.get(&alloy_invariant_key) {
        assert!(
            weight > 0.5,
            "Alloy should have good reputation for invariants"
        );
    }

    // Step 3: Create dispatcher with domain-weighted consensus
    let mut dispatcher = Dispatcher::new(DispatcherConfig {
        selection_strategy: SelectionStrategy::All,
        merge_strategy: MergeStrategy::DomainWeightedConsensus {
            domain_weights: domain_weights.clone(),
            aggregate_weights: aggregate_weights.clone(),
        },
        max_concurrent: 4,
        check_health: false,
        ..Default::default()
    });

    // Register mock backends that simulate the backends in our corpus
    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem, PropertyType::Invariant],
    )));
    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Coq,
        vec![PropertyType::Theorem],
    )));
    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Alloy,
        vec![PropertyType::Invariant],
    )));
    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::TlaPlus,
        vec![PropertyType::Temporal],
    )));

    // Step 4: Verify a theorem (Lean4 should be weighted higher)
    let theorem_spec = make_theorem_spec();
    let results = dispatcher.verify(&theorem_spec).await.unwrap();

    assert_eq!(results.summary.proven, 1);
    assert!(
        results.properties[0].confidence > 0.5,
        "Confidence should be reasonable with weighted consensus"
    );

    // Step 5: Verify an invariant (Alloy should be weighted higher in domain consensus)
    let invariant_spec = make_invariant_spec();
    let results = dispatcher.verify(&invariant_spec).await.unwrap();

    assert_eq!(results.summary.proven, 1);

    // Step 6: Verify a temporal property
    let temporal_spec = make_temporal_spec();
    let results = dispatcher.verify(&temporal_spec).await.unwrap();

    assert_eq!(results.summary.proven, 1);
}

/// Test that dispatcher with reputation tracker automatically updates during verification
#[tokio::test]
async fn test_dispatcher_auto_reputation_update_with_domain() {
    use dashprove_learning::{DomainKey, ReputationConfig, ReputationTracker};

    let tracker = ReputationTracker::new(ReputationConfig::default());

    // Configure dispatcher with auto-update enabled
    let config = DispatcherConfig {
        selection_strategy: SelectionStrategy::All,
        merge_strategy: MergeStrategy::FirstSuccess,
        auto_update_reputation: true,
        ..Default::default()
    };

    let mut dispatcher = Dispatcher::new(config);
    dispatcher.set_reputation_tracker(tracker);

    // Register backends
    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem],
    )));
    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Coq,
        vec![PropertyType::Theorem],
    )));

    // Initial state: no observations
    assert_eq!(
        dispatcher
            .reputation_tracker()
            .unwrap()
            .total_observations(),
        0
    );
    assert_eq!(
        dispatcher
            .reputation_tracker()
            .unwrap()
            .total_domain_observations(),
        0
    );

    // Run verification
    let spec = make_theorem_spec();
    let _results = dispatcher.verify(&spec).await.unwrap();

    // After verification: should have recorded observations
    let tracker = dispatcher.reputation_tracker().unwrap();
    assert!(
        tracker.total_observations() > 0,
        "Should have recorded aggregate observations"
    );
    assert!(
        tracker.total_domain_observations() > 0,
        "Should have recorded domain observations"
    );
    assert!(
        tracker.domain_count() > 0,
        "Should have tracked domain-specific stats"
    );

    // Verify domain-specific tracking
    let lean_theorem_key = DomainKey::new(BackendId::Lean4, PropertyType::Theorem);
    let lean_stats = tracker.get_domain_stats(&lean_theorem_key);
    assert!(
        lean_stats.is_some(),
        "Lean4 should have domain stats for theorems"
    );
    assert_eq!(lean_stats.unwrap().successes, 1);
}

/// Test the full learning workflow: verify -> record -> bootstrap -> domain-verify
#[tokio::test]
async fn test_complete_learning_workflow_with_domain_reputation() {
    use dashprove_learning::{
        LearnableResult, ProofCorpus, ReputationConfig, ReputationFromCorpus,
    };

    // Phase 1: Initial verification without reputation (simulating cold start)
    let mut corpus = ProofCorpus::new();

    // Simulate recording successful verifications to the corpus
    // In a real scenario, these would come from actual verification results
    let properties = vec![
        (BackendId::Lean4, PropertyType::Theorem, 100),
        (BackendId::Lean4, PropertyType::Theorem, 95),
        (BackendId::Lean4, PropertyType::Theorem, 110),
        (BackendId::Coq, PropertyType::Theorem, 200),
        (BackendId::Alloy, PropertyType::Invariant, 150),
        (BackendId::Alloy, PropertyType::Invariant, 140),
        (BackendId::Alloy, PropertyType::Invariant, 160),
        (BackendId::TlaPlus, PropertyType::Temporal, 300),
        (BackendId::TlaPlus, PropertyType::Temporal, 280),
    ];

    for (i, (backend, prop_type, time_ms)) in properties.iter().enumerate() {
        let property = match prop_type {
            PropertyType::Theorem => Property::Theorem(Theorem {
                name: format!("prop_{}", i),
                body: Expr::Bool(true),
            }),
            PropertyType::Invariant => Property::Invariant(Invariant {
                name: format!("prop_{}", i),
                body: Expr::Bool(true),
            }),
            PropertyType::Temporal => Property::Temporal(Temporal {
                name: format!("prop_{}", i),
                body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
                fairness: vec![],
            }),
            _ => continue,
        };

        let result = LearnableResult {
            property,
            backend: *backend,
            status: VerificationStatus::Proven,
            tactics: vec![],
            time_taken: Duration::from_millis(*time_ms as u64),
            proof_output: Some("proof".to_string()),
        };
        corpus.insert(&result);
    }

    // Phase 2: Bootstrap reputation from corpus
    let (tracker, stats) = ReputationFromCorpus::new(&corpus)
        .with_config(ReputationConfig::default())
        .build_with_stats();

    assert_eq!(stats.total_entries, 9);
    assert!(stats.domain_coverage() > 90.0);

    // Verify reputation weights reflect the corpus
    let domain_weights = tracker.compute_domain_weights();

    // Lean4 should have the best reputation for theorems (most proofs, fastest times)
    let lean_theorem = dashprove_learning::DomainKey::new(BackendId::Lean4, PropertyType::Theorem);
    let coq_theorem = dashprove_learning::DomainKey::new(BackendId::Coq, PropertyType::Theorem);

    if let (Some(&lean_w), Some(&coq_w)) = (
        domain_weights.get(&lean_theorem),
        domain_weights.get(&coq_theorem),
    ) {
        // Lean4 has more theorem proofs and faster times, so should have higher weight
        assert!(
            lean_w >= coq_w,
            "Lean4 should have >= weight than Coq for theorems: {} vs {}",
            lean_w,
            coq_w
        );
    }

    // Phase 3: Use bootstrapped reputation for new verifications
    let mut dispatcher = Dispatcher::new(DispatcherConfig {
        selection_strategy: SelectionStrategy::All,
        merge_strategy: MergeStrategy::DomainWeightedConsensus {
            domain_weights,
            aggregate_weights: tracker.compute_weights(),
        },
        ..Default::default()
    });

    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Lean4,
        vec![PropertyType::Theorem, PropertyType::Invariant],
    )));
    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::Alloy,
        vec![PropertyType::Invariant],
    )));
    dispatcher.register_backend(Arc::new(MockBackend::new(
        BackendId::TlaPlus,
        vec![PropertyType::Temporal],
    )));

    // Verify using domain-weighted consensus
    let invariant_spec = make_invariant_spec();
    let results = dispatcher.verify(&invariant_spec).await.unwrap();

    assert_eq!(results.summary.proven, 1);
    // Property type should be preserved in merged result
    assert_eq!(
        results.properties[0].property_type,
        Some(PropertyType::Invariant)
    );
}
