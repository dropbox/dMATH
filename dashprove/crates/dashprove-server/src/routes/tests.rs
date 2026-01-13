use super::*;
use axum::extract::State;
use axum::routing::{get, post};
use axum::Json;
use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use dashprove_backends::BackendId;
use tower::ServiceExt;

fn test_app() -> Router {
    let state = Arc::new(AppState::new());
    Router::new()
        .route("/health", get(health))
        .route("/version", get(version))
        .route("/verify", post(verify))
        .route("/verify/incremental", post(verify_incremental))
        .route("/corpus/search", get(corpus_search))
        .route("/tactics/suggest", post(tactics_suggest))
        .route("/sketch/elaborate", post(sketch_elaborate))
        .route("/explain", post(explain))
        .route("/backends", get(list_backends))
        .with_state(state)
}

#[tokio::test]
async fn test_verify_records_metrics() {
    let state = Arc::new(AppState::new());
    let request = VerifyRequest {
        spec: "theorem metrics_test { true }".to_string(),
        backend: Some(BackendIdParam::Lean4),
        use_ml: false,
        ml_min_confidence: 0.5,
    };

    let response = verify(State(state.clone()), Json(request)).await.unwrap();
    let result = response.0;
    assert!(result.valid);

    let cache_stats = state.proof_cache.read().await.stats();
    let metrics_output = state
        .metrics
        .export_prometheus(
            state.active_requests(),
            state.session_manager.active_count().await,
            cache_stats.total_entries,
            cache_stats.valid_entries,
            cache_stats.expired_entries,
        )
        .await;

    assert!(metrics_output.contains("dashprove_verification_duration_seconds_count 1"));
    assert!(metrics_output.contains("dashprove_verifications_success 1"));
    assert!(metrics_output
        .contains("dashprove_backend_verification_duration_seconds_count{backend=\"lean4\"} 1"));
    // Per-backend success/failure counters
    assert!(metrics_output.contains("dashprove_backend_verifications_success{backend=\"lean4\"} 1"));
    assert!(metrics_output.contains("dashprove_backend_verifications_failed{backend=\"lean4\"} 0"));
}

#[tokio::test]
async fn test_verify_failure_records_metrics() {
    let state = Arc::new(AppState::new());
    let request = VerifyRequest {
        spec: "invalid syntax {".to_string(),
        backend: None,
        use_ml: false,
        ml_min_confidence: 0.5,
    };

    let result = verify(State(state.clone()), Json(request)).await;
    assert!(result.is_err());

    let cache_stats = state.proof_cache.read().await.stats();
    let metrics_output = state
        .metrics
        .export_prometheus(
            state.active_requests(),
            state.session_manager.active_count().await,
            cache_stats.total_entries,
            cache_stats.valid_entries,
            cache_stats.expired_entries,
        )
        .await;

    assert!(metrics_output.contains("dashprove_verification_duration_seconds_count 1"));
    assert!(metrics_output.contains("dashprove_verifications_failed 1"));
}

#[tokio::test]
async fn test_health_endpoint_healthy() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Verify response body
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: HealthResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(result.status, "healthy");
    assert_eq!(result.shutdown_state, "running");
    assert!(result.ready);
    assert_eq!(result.in_flight_requests, 0);
    assert_eq!(result.active_websocket_sessions, 0);
}

#[tokio::test]
async fn test_health_endpoint_draining() {
    let state = Arc::new(AppState::new());
    // Set state to draining
    state.set_shutdown_state(ShutdownState::Draining);

    let app = Router::new()
        .route("/health", get(health))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return 503 when draining
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: HealthResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(result.status, "draining");
    assert_eq!(result.shutdown_state, "draining");
    assert!(!result.ready);
}

#[tokio::test]
async fn test_health_endpoint_shutting_down() {
    let state = Arc::new(AppState::new());
    // Set state to shutting down
    state.set_shutdown_state(ShutdownState::ShuttingDown);

    let app = Router::new()
        .route("/health", get(health))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return 503 when shutting down
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: HealthResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(result.status, "unhealthy");
    assert_eq!(result.shutdown_state, "shutting_down");
    assert!(!result.ready);
}

#[tokio::test]
async fn test_verify_simple_theorem() {
    let app = test_app();
    let body = serde_json::json!({
        "spec": "theorem test { true }"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/verify")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: VerifyResponse = serde_json::from_slice(&body).unwrap();
    assert!(result.valid);
    assert_eq!(result.property_count, 1);
    assert!(!result.compilations.is_empty());
}

#[tokio::test]
async fn test_verify_parse_error() {
    let app = test_app();
    let body = serde_json::json!({
        "spec": "invalid syntax {"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/verify")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_verify_with_specific_backend() {
    let app = test_app();
    let body = serde_json::json!({
        "spec": "theorem test { forall x: Bool . x or not x }",
        "backend": "lean4"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/verify")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: VerifyResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(result.compilations.len(), 1);
}

#[tokio::test]
async fn test_verify_with_new_backend() {
    let app = test_app();
    let body = serde_json::json!({
        "spec": "theorem test { true }",
        "backend": "coq"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/verify")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: VerifyResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(result.compilations.len(), 1);
    assert_eq!(result.compilations[0].backend, BackendIdParam::Coq);
}

#[tokio::test]
async fn test_tactics_suggest() {
    let app = test_app();
    let body = serde_json::json!({
        "property": "theorem test { forall x: Bool . x or not x }",
        "n": 3
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tactics/suggest")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: TacticSuggestResponse = serde_json::from_slice(&body).unwrap();
    // Should get some suggestions
    assert!(!result.suggestions.is_empty());
}

#[tokio::test]
async fn test_sketch_elaborate() {
    let app = test_app();
    let body = serde_json::json!({
        "property": "theorem test { true }",
        "hints": []
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/sketch/elaborate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: SketchElaborateResponse = serde_json::from_slice(&body).unwrap();
    assert!(!result.lean_code.is_empty());
}

#[tokio::test]
async fn test_corpus_search_no_learning() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/corpus/search?query=test&k=5")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    // Should return SERVICE_UNAVAILABLE when no learning system
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn test_corpus_search_with_learning() {
    let state = Arc::new(AppState::with_learning(ProofLearningSystem::new()));
    let app = Router::new()
        .route("/corpus/search", get(corpus_search))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/corpus/search?query=true&k=5")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: CorpusSearchResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(result.total_corpus_size, 0); // Empty corpus
}

#[test]
fn test_backend_id_conversion() {
    assert_eq!(BackendId::from(BackendIdParam::Lean4), BackendId::Lean4);
    assert_eq!(BackendId::from(BackendIdParam::TlaPlus), BackendId::TlaPlus);
    assert_eq!(BackendId::from(BackendIdParam::Kani), BackendId::Kani);
    assert_eq!(BackendId::from(BackendIdParam::Alloy), BackendId::Alloy);
    assert_eq!(
        BackendId::from(BackendIdParam::Isabelle),
        BackendId::Isabelle
    );
    assert_eq!(BackendId::from(BackendIdParam::Coq), BackendId::Coq);
    assert_eq!(BackendId::from(BackendIdParam::Dafny), BackendId::Dafny);
}

#[test]
fn test_property_name() {
    use dashprove_usl::ast::{Expr, Invariant, Property, Theorem};

    let theorem = Property::Theorem(Theorem {
        name: "my_theorem".to_string(),
        body: Expr::Bool(true),
    });
    assert_eq!(theorem.name(), "my_theorem");

    let invariant = Property::Invariant(Invariant {
        name: "my_invariant".to_string(),
        body: Expr::Bool(true),
    });
    assert_eq!(invariant.name(), "my_invariant");
}

// ============ Incremental Verification Tests ============

#[tokio::test]
async fn test_incremental_verify_no_changes() {
    let app = test_app();
    let body = serde_json::json!({
        "base_spec": "theorem test { true }",
        "current_spec": "theorem test { true }",
        "changes": []
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/verify/incremental")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: IncrementalVerifyResponse = serde_json::from_slice(&body).unwrap();
    assert!(result.valid);
    // With no changes, the property should be cached
    assert_eq!(result.cached_count, 1);
    assert_eq!(result.verified_count, 0);
    assert!(result.unchanged_properties.contains(&"test".to_string()));
}

#[tokio::test]
async fn test_incremental_verify_new_property() {
    let app = test_app();
    let body = serde_json::json!({
        "base_spec": "theorem test1 { true }",
        "current_spec": "theorem test1 { true } theorem test2 { false }",
        "changes": []
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/verify/incremental")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: IncrementalVerifyResponse = serde_json::from_slice(&body).unwrap();
    assert!(result.valid);
    // test1 unchanged, test2 is new (affected)
    assert_eq!(result.cached_count, 1);
    assert_eq!(result.verified_count, 1);
    assert!(result.affected_properties.contains(&"test2".to_string()));
}

#[tokio::test]
async fn test_incremental_verify_deleted_property() {
    let app = test_app();
    let body = serde_json::json!({
        "base_spec": "theorem test1 { true } theorem test2 { false }",
        "current_spec": "theorem test1 { true }",
        "changes": []
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/verify/incremental")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: IncrementalVerifyResponse = serde_json::from_slice(&body).unwrap();
    assert!(result.valid);
    // test2 was deleted
    assert!(result
        .affected_properties
        .iter()
        .any(|p| p.contains("test2") && p.contains("deleted")));
}

#[tokio::test]
async fn test_incremental_verify_with_change_target() {
    // Test that DependencyChanged affects properties that call the changed function
    let app = test_app();
    let body = serde_json::json!({
        "base_spec": "theorem uses_foo { foo(1) } theorem other { true }",
        "current_spec": "theorem uses_foo { foo(1) } theorem other { true }",
        "changes": [
            { "kind": "dependency_changed", "target": "foo", "details": "function body changed" }
        ]
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/verify/incremental")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: IncrementalVerifyResponse = serde_json::from_slice(&body).unwrap();
    assert!(result.valid);
    // uses_foo calls foo(), should be affected by foo changing
    assert!(result.affected_properties.contains(&"uses_foo".to_string()));
    // "other" doesn't call foo(), should be unchanged
    assert!(result.unchanged_properties.contains(&"other".to_string()));
}

#[tokio::test]
async fn test_incremental_verify_with_backend() {
    let app = test_app();
    let body = serde_json::json!({
        "base_spec": "theorem test { true }",
        "current_spec": "theorem test { true } theorem new { false }",
        "changes": [],
        "backend": "lean4"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/verify/incremental")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: IncrementalVerifyResponse = serde_json::from_slice(&body).unwrap();
    assert!(result.valid);
    // Should only have one compilation (LEAN 4)
    assert_eq!(result.compilations.len(), 1);
    assert!(matches!(
        result.compilations[0].backend,
        BackendIdParam::Lean4
    ));
}

#[tokio::test]
async fn test_incremental_verify_parse_error() {
    let app = test_app();
    let body = serde_json::json!({
        "base_spec": "invalid {",
        "current_spec": "theorem test { true }",
        "changes": []
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/verify/incremental")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn test_change_kind_deserialization() {
    let json = r#"{"kind": "file_modified", "target": "src/main.rs", "details": null}"#;
    let change: Change = serde_json::from_str(json).unwrap();
    assert!(matches!(change.kind, ChangeKind::FileModified));
    assert_eq!(change.target, "src/main.rs");
}

#[tokio::test]
async fn test_incremental_verify_type_dependency() {
    // Test that modifying a type affects properties that reference it
    let app = test_app();

    let base_spec = r#"
        type Counter = { value: Int }
        theorem counter_positive { forall c: Counter . c.value >= 0 }
        theorem unrelated { forall x: Int . x == x }
    "#;

    let current_spec = r#"
        type Counter = { value: Int, name: String }
        theorem counter_positive { forall c: Counter . c.value >= 0 }
        theorem unrelated { forall x: Int . x == x }
    "#;

    let body = serde_json::json!({
        "base_spec": base_spec,
        "current_spec": current_spec,
        "changes": [
            {"kind": "type_modified", "target": "Counter", "details": "added name field"}
        ]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/verify/incremental")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: IncrementalVerifyResponse = serde_json::from_slice(&body).unwrap();

    // counter_positive depends on Counter type, should be affected
    assert!(
        result
            .affected_properties
            .contains(&"counter_positive".to_string()),
        "counter_positive should be affected by Counter type change"
    );

    // unrelated does not depend on Counter, should not be affected
    assert!(
        result
            .unchanged_properties
            .contains(&"unrelated".to_string()),
        "unrelated should be unchanged"
    );
}

// ============ Explain Endpoint Tests ============

#[tokio::test]
async fn test_explain_lean_unsolved_goals() {
    let app = test_app();
    let body = serde_json::json!({
        "property": "theorem test_theorem { forall x: Bool . x or not x }",
        "counterexample": "unsolved goals\n⊢ P ∧ Q",
        "backend": "lean4"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/explain")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: ExplainResponse = serde_json::from_slice(&body).unwrap();
    assert!(matches!(result.kind, ExplanationKindResponse::MissingCase));
    assert!(result.summary.contains("incomplete"));
    assert!(!result.suggestions.is_empty());
}

#[tokio::test]
async fn test_explain_tlaplus_trace() {
    let app = test_app();
    let body = serde_json::json!({
        "property": "temporal safety { always(eventually(true)) }",
        "counterexample": "State 1:\n  x = 0\n  y = 1\nState 2:\n  /\\ Next_action\n  x = 1\n  y = 2",
        "backend": "tlaplus"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/explain")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: ExplainResponse = serde_json::from_slice(&body).unwrap();
    assert!(matches!(result.kind, ExplanationKindResponse::StateTrace));
    // Should have parsed the trace
    assert!(!result.trace.is_empty());
}

#[tokio::test]
async fn test_explain_kani_assertion() {
    let app = test_app();
    let body = serde_json::json!({
        "property": "contract add(x: Int, y: Int) -> Int { requires { x >= 0 } ensures { result > 0 } }",
        "counterexample": "assertion failed: result > 0\nconcrete value: x = 0, y = 0",
        "backend": "kani"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/explain")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: ExplainResponse = serde_json::from_slice(&body).unwrap();
    assert!(matches!(
        result.kind,
        ExplanationKindResponse::PostconditionViolation
    ));
    assert!(result.summary.contains("Postcondition"));
}

#[tokio::test]
async fn test_explain_alloy_counterexample() {
    let app = test_app();
    let body = serde_json::json!({
        "property": "invariant no_cycles { true }",
        "counterexample": "sig Node = {Node0, Node1}\nedges = {Node0->Node1, Node1->Node0}",
        "backend": "alloy"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/explain")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: ExplainResponse = serde_json::from_slice(&body).unwrap();
    assert!(matches!(
        result.kind,
        ExplanationKindResponse::VariableAssignment
    ));
    assert!(result.summary.contains("Counterexample"));
}

#[tokio::test]
async fn test_explain_parse_error() {
    let app = test_app();
    let body = serde_json::json!({
        "property": "invalid syntax {",
        "counterexample": "some error",
        "backend": "lean4"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/explain")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn test_explanation_kind_serialization() {
    let kind = ExplanationKindResponse::MissingCase;
    let json = serde_json::to_string(&kind).unwrap();
    assert_eq!(json, "\"missing_case\"");

    let kind = ExplanationKindResponse::PostconditionViolation;
    let json = serde_json::to_string(&kind).unwrap();
    assert_eq!(json, "\"postcondition_violation\"");
}

// ============ Backends Endpoint Tests ============

#[tokio::test]
async fn test_backends_returns_all_backends() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/backends")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: BackendsResponse = serde_json::from_slice(&body).unwrap();

    // Should have 7 backends
    assert_eq!(result.backends.len(), 7);

    // Check all expected backends are present
    let ids: Vec<_> = result.backends.iter().map(|b| &b.id).collect();
    assert!(ids.contains(&&BackendIdParam::Lean4));
    assert!(ids.contains(&&BackendIdParam::TlaPlus));
    assert!(ids.contains(&&BackendIdParam::Kani));
    assert!(ids.contains(&&BackendIdParam::Alloy));
    assert!(ids.contains(&&BackendIdParam::Isabelle));
    assert!(ids.contains(&&BackendIdParam::Coq));
    assert!(ids.contains(&&BackendIdParam::Dafny));
}

#[tokio::test]
async fn test_backends_have_names() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/backends")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: BackendsResponse = serde_json::from_slice(&body).unwrap();

    // Each backend should have a non-empty name
    for backend in &result.backends {
        assert!(!backend.name.is_empty());
    }
}

#[tokio::test]
async fn test_backends_have_supported_properties() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/backends")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: BackendsResponse = serde_json::from_slice(&body).unwrap();

    // Each backend should support at least one property type
    for backend in &result.backends {
        assert!(!backend.supports.is_empty());
    }

    // Verify specific mappings
    let lean4 = result
        .backends
        .iter()
        .find(|b| b.id == BackendIdParam::Lean4)
        .unwrap();
    assert!(lean4.supports.contains(&PropertyTypeResponse::Theorem));

    let tlaplus = result
        .backends
        .iter()
        .find(|b| b.id == BackendIdParam::TlaPlus)
        .unwrap();
    assert!(tlaplus.supports.contains(&PropertyTypeResponse::Temporal));

    let kani = result
        .backends
        .iter()
        .find(|b| b.id == BackendIdParam::Kani)
        .unwrap();
    assert!(kani.supports.contains(&PropertyTypeResponse::Contract));

    let alloy = result
        .backends
        .iter()
        .find(|b| b.id == BackendIdParam::Alloy)
        .unwrap();
    assert!(alloy.supports.contains(&PropertyTypeResponse::Invariant));

    let isabelle = result
        .backends
        .iter()
        .find(|b| b.id == BackendIdParam::Isabelle)
        .unwrap();
    assert!(isabelle.supports.contains(&PropertyTypeResponse::Theorem));
    assert!(isabelle
        .supports
        .contains(&PropertyTypeResponse::Refinement));

    let coq = result
        .backends
        .iter()
        .find(|b| b.id == BackendIdParam::Coq)
        .unwrap();
    assert!(coq.supports.contains(&PropertyTypeResponse::Theorem));

    let dafny = result
        .backends
        .iter()
        .find(|b| b.id == BackendIdParam::Dafny)
        .unwrap();
    assert!(dafny.supports.contains(&PropertyTypeResponse::Contract));
}

#[test]
fn test_health_status_serialization() {
    let healthy = HealthStatusResponse::Healthy;
    let json = serde_json::to_string(&healthy).unwrap();
    assert!(json.contains("\"status\":\"healthy\""));

    let unavailable = HealthStatusResponse::Unavailable {
        reason: "not installed".to_string(),
    };
    let json = serde_json::to_string(&unavailable).unwrap();
    assert!(json.contains("\"status\":\"unavailable\""));
    assert!(json.contains("not installed"));
}

#[test]
fn test_property_type_serialization() {
    let pt = PropertyTypeResponse::Theorem;
    let json = serde_json::to_string(&pt).unwrap();
    assert_eq!(json, "\"theorem\"");

    let pt = PropertyTypeResponse::Temporal;
    let json = serde_json::to_string(&pt).unwrap();
    assert_eq!(json, "\"temporal\"");
}

// ============ Version Endpoint Tests ============

#[tokio::test]
async fn test_version_returns_metadata() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/version")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: VersionResponse = serde_json::from_slice(&body).unwrap();

    // Check that version info is present
    assert_eq!(result.name, "dashprove-server");
    assert_eq!(result.version, "0.1.0");
    assert_eq!(result.api_version, "v1");
}

#[test]
fn test_version_response_serialization() {
    let version = VersionResponse {
        name: "test".to_string(),
        version: "1.0.0".to_string(),
        api_version: "v1".to_string(),
        rust_version: "1.75.0".to_string(),
        target: "x86_64-unknown-linux-gnu".to_string(),
    };
    let json = serde_json::to_string(&version).unwrap();
    assert!(json.contains("\"name\":\"test\""));
    assert!(json.contains("\"version\":\"1.0.0\""));
    assert!(json.contains("\"api_version\":\"v1\""));
}

// ============ Shutdown State Tests ============

#[test]
fn test_shutdown_state_default() {
    assert_eq!(ShutdownState::default(), ShutdownState::Running);
}

#[test]
fn test_shutdown_state_values() {
    // Verify enum values for atomic storage
    assert_eq!(ShutdownState::Running as u8, 0);
    assert_eq!(ShutdownState::Draining as u8, 1);
    assert_eq!(ShutdownState::ShuttingDown as u8, 2);
}

#[test]
fn test_app_state_shutdown_methods() {
    let state = AppState::new();

    // Initial state should be Running
    assert_eq!(state.get_shutdown_state(), ShutdownState::Running);
    assert!(!state.is_draining());

    // Set to Draining
    state.set_shutdown_state(ShutdownState::Draining);
    assert_eq!(state.get_shutdown_state(), ShutdownState::Draining);
    assert!(state.is_draining());

    // Set to ShuttingDown
    state.set_shutdown_state(ShutdownState::ShuttingDown);
    assert_eq!(state.get_shutdown_state(), ShutdownState::ShuttingDown);
    assert!(state.is_draining());

    // Can reset to Running
    state.set_shutdown_state(ShutdownState::Running);
    assert_eq!(state.get_shutdown_state(), ShutdownState::Running);
    assert!(!state.is_draining());
}

#[test]
fn test_health_response_serialization() {
    let response = HealthResponse {
        status: "healthy".to_string(),
        shutdown_state: "running".to_string(),
        in_flight_requests: 5,
        active_websocket_sessions: 2,
        ready: true,
    };
    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("\"status\":\"healthy\""));
    assert!(json.contains("\"shutdown_state\":\"running\""));
    assert!(json.contains("\"in_flight_requests\":5"));
    assert!(json.contains("\"active_websocket_sessions\":2"));
    assert!(json.contains("\"ready\":true"));
}

#[test]
fn test_health_response_draining_serialization() {
    let response = HealthResponse {
        status: "draining".to_string(),
        shutdown_state: "draining".to_string(),
        in_flight_requests: 10,
        active_websocket_sessions: 3,
        ready: false,
    };
    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("\"status\":\"draining\""));
    assert!(json.contains("\"ready\":false"));
}

#[tokio::test]
async fn test_health_response_from_state_running() {
    let state = AppState::new();
    let response = HealthResponse::from_state(&state).await;

    assert_eq!(response.status, "healthy");
    assert_eq!(response.shutdown_state, "running");
    assert!(response.ready);
    assert_eq!(response.in_flight_requests, 0);
    assert_eq!(response.active_websocket_sessions, 0);
}

#[tokio::test]
async fn test_health_response_from_state_draining() {
    let state = AppState::new();
    state.set_shutdown_state(ShutdownState::Draining);
    let response = HealthResponse::from_state(&state).await;

    assert_eq!(response.status, "draining");
    assert_eq!(response.shutdown_state, "draining");
    assert!(!response.ready);
}

#[tokio::test]
async fn test_health_response_from_state_shutting_down() {
    let state = AppState::new();
    state.set_shutdown_state(ShutdownState::ShuttingDown);
    let response = HealthResponse::from_state(&state).await;

    assert_eq!(response.status, "unhealthy");
    assert_eq!(response.shutdown_state, "shutting_down");
    assert!(!response.ready);
}

#[tokio::test]
async fn test_health_response_with_active_sessions() {
    let state = AppState::new();
    // Create some WebSocket sessions
    state.session_manager.create_session().await;
    state.session_manager.create_session().await;

    let response = HealthResponse::from_state(&state).await;
    assert_eq!(response.active_websocket_sessions, 2);
}

#[test]
fn test_health_response_with_in_flight_requests() {
    let state = AppState::new();
    state.request_started();
    state.request_started();
    state.request_started();

    assert_eq!(state.active_requests(), 3);
}

// ============ Expert API Tests ============

#[test]
fn test_parse_expert_property_types() {
    let types = parse_expert_property_types("safety,liveness,temporal");
    assert_eq!(types.len(), 3);
    assert!(types.contains(&ExpertPropertyType::Safety));
    assert!(types.contains(&ExpertPropertyType::Liveness));
    assert!(types.contains(&ExpertPropertyType::Temporal));
}

#[test]
fn test_parse_expert_property_types_with_aliases() {
    let types = parse_expert_property_types("nn,security");
    assert_eq!(types.len(), 2);
    assert!(types.contains(&ExpertPropertyType::NeuralNetwork));
    assert!(types.contains(&ExpertPropertyType::SecurityProtocol));
}

#[test]
fn test_parse_expert_property_types_empty_string() {
    let types = parse_expert_property_types("");
    assert!(types.is_empty());
}

#[test]
fn test_parse_expert_property_types_unknown() {
    let types = parse_expert_property_types("unknown,invalid");
    assert!(types.is_empty());
}

#[test]
fn test_get_knowledge_dir_returns_path() {
    let dir = get_knowledge_dir();
    assert!(dir.to_string_lossy().contains("knowledge"));
}

#[tokio::test]
async fn test_expert_backend_with_rust_code() {
    let request = ExpertBackendRequest {
        spec: Some("fn test() {}".to_string()),
        property_types: Some("safety".to_string()),
        code_lang: Some("rust".to_string()),
        tags: None,
    };

    let result = expert_backend(Json(request)).await;
    assert!(result.is_ok());

    let response = result.unwrap().0;
    // Rust code should recommend a Rust-capable backend
    assert!(
        response.backend == BackendIdParam::Kani
            || response.backend == BackendIdParam::Verus
            || response.backend == BackendIdParam::Creusot
            || response.backend == BackendIdParam::Prusti
    );
    assert!(response.confidence >= 0.0 && response.confidence <= 1.0);
    assert!(!response.rationale.is_empty());
}

#[tokio::test]
async fn test_expert_backend_with_temporal_properties() {
    let request = ExpertBackendRequest {
        spec: None,
        property_types: Some("temporal,liveness".to_string()),
        code_lang: None,
        tags: None,
    };

    let result = expert_backend(Json(request)).await;
    assert!(result.is_ok());

    let response = result.unwrap().0;
    // Temporal properties should recommend TLA+ or similar
    assert!(response.confidence >= 0.0 && response.confidence <= 1.0);
}

#[tokio::test]
async fn test_expert_error_type_mismatch() {
    let request = ExpertErrorRequest {
        message: "type mismatch: expected Int, found String".to_string(),
        backend: Some(BackendIdParam::Lean4),
    };

    let result = expert_error(Json(request)).await;
    assert!(result.is_ok());

    let response = result.unwrap().0;
    assert!(!response.explanation.is_empty());
    assert!(!response.root_cause.is_empty());
    assert!(!response.suggested_fixes.is_empty());
}

#[tokio::test]
async fn test_expert_error_timeout() {
    let request = ExpertErrorRequest {
        message: "verification timeout after 60s".to_string(),
        backend: None,
    };

    let result = expert_error(Json(request)).await;
    assert!(result.is_ok());

    let response = result.unwrap().0;
    // The explanation contains "timed out" not "timeout"
    assert!(response.explanation.to_lowercase().contains("time"));
}

#[tokio::test]
async fn test_expert_tactic_lean4_equality() {
    let request = ExpertTacticRequest {
        goal: "prove x = y".to_string(),
        backend: BackendIdParam::Lean4,
        context: None,
    };

    let result = expert_tactic(Json(request)).await;
    assert!(result.is_ok());

    let response = result.unwrap().0;
    assert!(!response.suggestions.is_empty());

    // Should suggest tactics like rfl or simp for equality
    let tactics: Vec<_> = response
        .suggestions
        .iter()
        .map(|s| s.tactic.as_str())
        .collect();
    assert!(tactics.contains(&"rfl") || tactics.contains(&"simp"));
}

#[tokio::test]
async fn test_expert_tactic_coq_forall() {
    let request = ExpertTacticRequest {
        goal: "forall x, P x".to_string(),
        backend: BackendIdParam::Coq,
        context: None,
    };

    let result = expert_tactic(Json(request)).await;
    assert!(result.is_ok());

    let response = result.unwrap().0;
    assert!(!response.suggestions.is_empty());

    // Should suggest intros for forall
    let tactics: Vec<_> = response
        .suggestions
        .iter()
        .map(|s| s.tactic.as_str())
        .collect();
    assert!(tactics.contains(&"intros"));
}

#[tokio::test]
async fn test_expert_compile_lean4() {
    let request = ExpertCompileRequest {
        spec: "theorem test { forall x: Int . x >= 0 implies x * x >= 0 }".to_string(),
        backend: BackendIdParam::Lean4,
    };

    let result = expert_compile(Json(request)).await;
    assert!(result.is_ok());

    let response = result.unwrap().0;
    assert_eq!(response.target_backend, BackendIdParam::Lean4);
    assert!(!response.steps.is_empty());
    assert!(!response.pitfalls.is_empty());
    assert!(!response.best_practices.is_empty());
}

#[tokio::test]
async fn test_expert_compile_kani() {
    let request = ExpertCompileRequest {
        spec: "fn check_bounds(x: u32) { assert!(x < 100); }".to_string(),
        backend: BackendIdParam::Kani,
    };

    let result = expert_compile(Json(request)).await;
    assert!(result.is_ok());

    let response = result.unwrap().0;
    assert_eq!(response.target_backend, BackendIdParam::Kani);
    assert!(!response.steps.is_empty());
}
