//! End-to-end pipeline tests
//!
//! Tests the full verification pipeline from USL spec to backend results.

use dashprove::dispatcher::{Dispatcher, DispatcherConfig};
use dashprove::learning::{LearnableResult, ProofLearningSystem};
use dashprove::usl::{parse, typecheck};
use dashprove_backends::{
    BackendId, BackendResult, HealthStatus, PropertyType, StructuredCounterexample,
    VerificationBackend, VerificationStatus,
};
use std::sync::Arc;
use std::time::Duration;

/// Mock backend for integration testing (doesn't require actual tools)
struct MockIntegrationBackend {
    id: BackendId,
    supported: Vec<PropertyType>,
    should_prove: bool,
}

impl MockIntegrationBackend {
    fn lean4_proving() -> Self {
        Self {
            id: BackendId::Lean4,
            supported: vec![
                PropertyType::Theorem,
                PropertyType::Invariant,
                PropertyType::Refinement,
            ],
            should_prove: true,
        }
    }

    fn tlaplus_proving() -> Self {
        Self {
            id: BackendId::TlaPlus,
            supported: vec![PropertyType::Temporal, PropertyType::Invariant],
            should_prove: true,
        }
    }

    fn alloy_disproving() -> Self {
        Self {
            id: BackendId::Alloy,
            supported: vec![PropertyType::Theorem, PropertyType::Invariant],
            should_prove: false,
        }
    }
}

#[async_trait::async_trait]
impl VerificationBackend for MockIntegrationBackend {
    fn id(&self) -> BackendId {
        self.id
    }

    fn supports(&self) -> Vec<PropertyType> {
        self.supported.clone()
    }

    async fn verify(
        &self,
        _spec: &dashprove::usl::typecheck::TypedSpec,
    ) -> Result<BackendResult, dashprove_backends::BackendError> {
        let status = if self.should_prove {
            VerificationStatus::Proven
        } else {
            VerificationStatus::Disproven
        };

        Ok(BackendResult {
            backend: self.id,
            status,
            proof: if self.should_prove {
                Some("Verified.".to_string())
            } else {
                None
            },
            counterexample: if !self.should_prove {
                Some(StructuredCounterexample::from_raw(
                    "Found counterexample at x=0".to_string(),
                ))
            } else {
                None
            },
            diagnostics: vec![],
            time_taken: Duration::from_millis(50),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        HealthStatus::Healthy
    }
}

#[tokio::test]
async fn test_full_pipeline_theorem_verification() {
    let spec_src = r#"
        theorem excluded_middle {
            forall x: Bool . x or not x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::default());
    dispatcher.register_backend(Arc::new(MockIntegrationBackend::lean4_proving()));

    let results = dispatcher
        .verify(&typed_spec)
        .await
        .expect("verification should complete");

    assert_eq!(results.summary.proven, 1);
    assert_eq!(results.summary.disproven, 0);
    assert_eq!(results.summary.unknown, 0);
}

#[tokio::test]
async fn test_full_pipeline_invariant_verification() {
    let spec_src = r#"
        invariant always_true {
            true
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::default());
    dispatcher.register_backend(Arc::new(MockIntegrationBackend::lean4_proving()));

    let results = dispatcher
        .verify(&typed_spec)
        .await
        .expect("verification should complete");

    assert_eq!(results.summary.proven, 1);
}

#[tokio::test]
async fn test_full_pipeline_temporal_verification() {
    let spec_src = r#"
        temporal eventually_done {
            always(eventually(done))
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::default());
    dispatcher.register_backend(Arc::new(MockIntegrationBackend::tlaplus_proving()));

    let results = dispatcher
        .verify(&typed_spec)
        .await
        .expect("verification should complete");

    assert_eq!(results.summary.proven, 1);
}

#[tokio::test]
async fn test_full_pipeline_contract_verification() {
    let spec_src = r#"
        contract increment(x: Int) -> Int {
            requires { x >= 0 }
            ensures { result == x + 1 }
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    // Contract requires Kani backend
    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::Kani));
    dispatcher.register_backend(Arc::new(MockIntegrationBackend {
        id: BackendId::Kani,
        supported: vec![PropertyType::Contract],
        should_prove: true,
    }));

    let results = dispatcher
        .verify(&typed_spec)
        .await
        .expect("verification should complete");

    assert_eq!(results.summary.proven, 1);
}

#[tokio::test]
async fn test_full_pipeline_disproven_property() {
    let spec_src = r#"
        theorem always_false {
            false
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::default());
    dispatcher.register_backend(Arc::new(MockIntegrationBackend::alloy_disproving()));

    let results = dispatcher
        .verify(&typed_spec)
        .await
        .expect("verification should complete");

    assert_eq!(results.summary.proven, 0);
    assert_eq!(results.summary.disproven, 1);
}

#[tokio::test]
async fn test_full_pipeline_multiple_properties() {
    let spec_src = r#"
        theorem prop1 { true }
        theorem prop2 { forall x: Bool . x == x }
        invariant inv1 { true }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::default());
    dispatcher.register_backend(Arc::new(MockIntegrationBackend::lean4_proving()));

    let results = dispatcher
        .verify(&typed_spec)
        .await
        .expect("verification should complete");

    assert_eq!(results.summary.proven, 3);
    assert_eq!(results.properties.len(), 3);
}

#[tokio::test]
async fn test_full_pipeline_redundant_verification() {
    let spec_src = r#"
        theorem test { true }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::redundant(2));
    dispatcher.register_backend(Arc::new(MockIntegrationBackend::lean4_proving()));
    dispatcher.register_backend(Arc::new(MockIntegrationBackend {
        id: BackendId::Alloy,
        supported: vec![PropertyType::Theorem, PropertyType::Invariant],
        should_prove: true,
    }));

    let results = dispatcher
        .verify(&typed_spec)
        .await
        .expect("verification should complete");

    // Both backends should agree
    assert_eq!(results.summary.proven, 1);
    assert_eq!(results.summary.overall_confidence, 1.0);
    assert_eq!(results.properties[0].backend_results.len(), 2);
}

#[tokio::test]
async fn test_learning_system_integration() {
    let mut learning = ProofLearningSystem::new();

    // Record a successful verification
    let result = LearnableResult {
        property: dashprove::usl::ast::Property::Theorem(dashprove::usl::ast::Theorem {
            name: "test_theorem".to_string(),
            body: dashprove::usl::ast::Expr::Bool(true),
        }),
        backend: BackendId::Lean4,
        status: VerificationStatus::Proven,
        tactics: vec!["decide".to_string(), "simp".to_string()],
        time_taken: Duration::from_millis(100),
        proof_output: Some("theorem test_theorem : True := trivial".to_string()),
    };

    learning.record(&result);
    assert_eq!(learning.proof_count(), 1);

    // Query similar proofs
    let query = dashprove::usl::ast::Property::Theorem(dashprove::usl::ast::Theorem {
        name: "similar_theorem".to_string(),
        body: dashprove::usl::ast::Expr::Bool(true),
    });

    let similar = learning.find_similar(&query, 5);
    assert!(!similar.is_empty());
    assert!(similar[0].similarity > 0.5);

    // Get tactic suggestions
    let suggestions = learning.suggest_tactics(&query, 3);
    assert!(!suggestions.is_empty());
    let tactic_names: Vec<_> = suggestions.iter().map(|(name, _)| name.as_str()).collect();
    assert!(tactic_names.contains(&"decide") || tactic_names.contains(&"simp"));
}

#[tokio::test]
async fn test_learning_system_persistence() {
    let temp_dir = std::env::temp_dir().join(format!(
        "dashprove_integration_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    {
        let mut learning = ProofLearningSystem::new();

        let result = LearnableResult {
            property: dashprove::usl::ast::Property::Invariant(dashprove::usl::ast::Invariant {
                name: "persist_inv".to_string(),
                body: dashprove::usl::ast::Expr::Bool(true),
            }),
            backend: BackendId::Lean4,
            status: VerificationStatus::Proven,
            tactics: vec!["trivial".to_string()],
            time_taken: Duration::from_millis(50),
            proof_output: None,
        };

        learning.record(&result);
        learning
            .save_to_dir(&temp_dir)
            .expect("save should succeed");
    }

    // Load in new instance
    let loaded = ProofLearningSystem::load_from_dir(&temp_dir).expect("load should succeed");
    assert_eq!(loaded.proof_count(), 1);

    // Cleanup
    std::fs::remove_dir_all(&temp_dir).ok();
}

#[tokio::test]
async fn test_ai_assistant_integration() {
    use dashprove::ai::ProofAssistant;

    let assistant = ProofAssistant::new();

    let property = dashprove::usl::ast::Property::Theorem(dashprove::usl::ast::Theorem {
        name: "test_prop".to_string(),
        body: dashprove::usl::ast::Expr::ForAll {
            var: "x".to_string(),
            ty: Some(dashprove::usl::ast::Type::Named("Bool".to_string())),
            body: Box::new(dashprove::usl::ast::Expr::Or(
                Box::new(dashprove::usl::ast::Expr::Var("x".to_string())),
                Box::new(dashprove::usl::ast::Expr::Not(Box::new(
                    dashprove::usl::ast::Expr::Var("x".to_string()),
                ))),
            )),
        },
    });

    // Get strategy recommendation
    let strategy = assistant.recommend_strategy(&property);
    assert_eq!(strategy.backend, BackendId::Lean4);
    assert!(!strategy.tactics.is_empty());
    assert!(!strategy.rationale.is_empty());

    // Create proof sketch
    let sketch = assistant.create_sketch(&property, &[]);
    assert!(!sketch.steps.is_empty());
}

#[tokio::test]
async fn test_compile_to_all_backends() {
    use dashprove::usl::{compile_to_alloy, compile_to_kani, compile_to_lean, compile_to_tlaplus};

    let spec_src = r#"
        theorem test {
            forall x: Bool . x or not x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    // Compile to all backends
    let lean = compile_to_lean(&typed_spec);
    let tlaplus = compile_to_tlaplus(&typed_spec);
    let kani = compile_to_kani(&typed_spec);
    let alloy = compile_to_alloy(&typed_spec);

    // All should produce non-empty code
    assert!(!lean.code.is_empty());
    assert!(!tlaplus.code.is_empty());
    assert!(!kani.code.is_empty());
    assert!(!alloy.code.is_empty());

    // LEAN code should have namespace
    assert!(lean.code.contains("namespace"));
    assert!(lean.code.contains("theorem"));

    // TLA+ should have MODULE
    assert!(tlaplus.code.contains("MODULE"));

    // Kani output depends on property type - for theorems it generates a comment
    // (Kani is primarily for contracts, not theorems)
    assert!(!kani.code.is_empty());
    assert!(
        kani.code.contains("Generated") || kani.code.contains("kani") || kani.code.contains("//")
    );

    // Alloy should have sig or fact or module comment
    assert!(
        alloy.code.contains("sig")
            || alloy.code.contains("fact")
            || alloy.code.contains("pred")
            || alloy.code.contains("module")
    );
}
