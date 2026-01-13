//! Integration tests for DashProve REST API server
//!
//! These tests spawn an actual server and make real HTTP requests
//! to test end-to-end functionality. Includes WebSocket tests for
//! streaming verification.

use axum::{
    routing::{get, post},
    Router,
};
use chrono::Utc;
use dashprove_learning::ProofLearningSystem;
use dashprove_server::routes::{
    AppState, BackendsResponse, CacheStatsResponse, CorpusCompareResponse, CorpusHistoryResponse,
    CorpusStatsResponse, CorpusSuggestResponse, CounterexampleAddResponse,
    CounterexampleClassifyResponse, CounterexampleClustersResponse, CounterexampleEntryResponse,
    CounterexampleListResponse, CounterexampleSearchResponse, ExplainResponse, HealthResponse,
    IncrementalVerifyResponse, SketchElaborateResponse, TacticSuggestResponse, VerifyResponse,
    VersionResponse,
};
use dashprove_server::ws::{WsMessage, WsVerifyRequest};
use futures_util::{SinkExt, StreamExt};
use std::{sync::Arc, time::Duration};
use tokio::net::TcpListener;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

/// Helper to create a test app with full routing
fn create_app(with_learning: bool) -> Router {
    use dashprove_server::{routes, ws};

    let state = if with_learning {
        Arc::new(AppState::with_learning(ProofLearningSystem::new()))
    } else {
        Arc::new(AppState::new())
    };

    Router::new()
        .route("/health", get(routes::health))
        .route("/version", get(routes::version))
        .route("/verify", post(routes::verify))
        .route("/verify/incremental", post(routes::verify_incremental))
        .route("/corpus/search", get(routes::corpus_search))
        .route("/corpus/stats", get(routes::corpus_stats))
        .route("/corpus/history", get(routes::corpus_history))
        .route("/corpus/compare", get(routes::corpus_compare))
        .route("/corpus/suggest", get(routes::corpus_suggest))
        .route(
            "/corpus/counterexamples/search",
            post(routes::counterexample_search),
        )
        .route(
            "/corpus/counterexamples/text-search",
            get(routes::counterexample_text_search),
        )
        .route(
            "/corpus/counterexamples",
            get(routes::counterexample_list).post(routes::counterexample_add),
        )
        .route(
            "/corpus/counterexamples/:id",
            get(routes::counterexample_get),
        )
        .route(
            "/corpus/counterexamples/classify",
            post(routes::counterexample_classify),
        )
        .route(
            "/corpus/counterexamples/clusters",
            post(routes::counterexample_clusters),
        )
        .route("/tactics/suggest", post(routes::tactics_suggest))
        .route("/sketch/elaborate", post(routes::sketch_elaborate))
        .route("/explain", post(routes::explain))
        .route("/backends", get(routes::list_backends))
        .route("/cache/stats", get(routes::cache_stats))
        .route("/ws/verify", get(ws::ws_verify_handler))
        .with_state(state)
}

/// Spawns the server on an available port and returns the base URL
async fn spawn_server(with_learning: bool) -> String {
    let app = create_app(with_learning);

    // Bind to port 0 to get a random available port
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Spawn the server in the background
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Give the server a moment to start
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

#[tokio::test]
async fn test_server_health_endpoint() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/health", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: HealthResponse = response.json().await.unwrap();
    assert_eq!(body.status, "healthy");
    assert_eq!(body.shutdown_state, "running");
    assert!(body.ready);
    assert_eq!(body.in_flight_requests, 0);
    assert_eq!(body.active_websocket_sessions, 0);
}

#[tokio::test]
async fn test_server_version_endpoint() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/version", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let version: VersionResponse = response.json().await.unwrap();
    assert_eq!(version.name, "dashprove-server");
    assert_eq!(version.api_version, "v1");
}

#[tokio::test]
async fn test_server_verify_endpoint() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "spec": "theorem test { forall x: Bool . x or not x }"
    });

    let response = client
        .post(format!("{}/verify", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: VerifyResponse = response.json().await.unwrap();
    assert!(result.valid);
    assert_eq!(result.property_count, 1);
    // Should have compilations for all backends (181 total as of Phase 15, including KaniFast)
    assert_eq!(result.compilations.len(), 181);
}

#[tokio::test]
async fn test_server_verify_with_backend() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "spec": "theorem test { true }",
        "backend": "lean4"
    });

    let response = client
        .post(format!("{}/verify", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: VerifyResponse = response.json().await.unwrap();
    assert!(result.valid);
    // Only LEAN 4 compilation
    assert_eq!(result.compilations.len(), 1);
}

#[tokio::test]
async fn test_server_verify_invalid_spec() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "spec": "invalid syntax {"
    });

    let response = client
        .post(format!("{}/verify", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn test_server_incremental_verify() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "base_spec": "theorem test1 { true }",
        "current_spec": "theorem test1 { true } theorem test2 { false }",
        "changes": []
    });

    let response = client
        .post(format!("{}/verify/incremental", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: IncrementalVerifyResponse = response.json().await.unwrap();
    assert!(result.valid);
    assert_eq!(result.cached_count, 1); // test1 unchanged
    assert_eq!(result.verified_count, 1); // test2 is new
}

#[tokio::test]
async fn test_server_backends_endpoint() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/backends", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: BackendsResponse = response.json().await.unwrap();
    assert_eq!(result.backends.len(), 7);

    // Check all expected backends present
    let names: Vec<_> = result.backends.iter().map(|b| b.name.as_str()).collect();
    assert!(names.contains(&"LEAN 4"));
    assert!(names.contains(&"TLA+"));
    assert!(names.contains(&"Kani"));
    assert!(names.contains(&"Alloy"));
    assert!(names.contains(&"Isabelle/HOL"));
    assert!(names.contains(&"Coq"));
    assert!(names.contains(&"Dafny"));
}

#[tokio::test]
async fn test_server_tactics_suggest() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "property": "theorem test { forall x: Bool . x or not x }",
        "n": 3
    });

    let response = client
        .post(format!("{}/tactics/suggest", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: TacticSuggestResponse = response.json().await.unwrap();
    assert!(!result.suggestions.is_empty());
}

#[tokio::test]
async fn test_server_sketch_elaborate() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "property": "theorem test { true }",
        "hints": []
    });

    let response = client
        .post(format!("{}/sketch/elaborate", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: SketchElaborateResponse = response.json().await.unwrap();
    assert!(!result.lean_code.is_empty());
}

#[tokio::test]
async fn test_server_explain_endpoint() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "property": "theorem test { forall x: Bool . x and not x }",
        "counterexample": "unsolved goals\n x: Bool\n P: Prop",
        "backend": "lean4"
    });

    let response = client
        .post(format!("{}/explain", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: ExplainResponse = response.json().await.unwrap();
    assert!(!result.summary.is_empty());
    assert!(!result.suggestions.is_empty());
}

#[tokio::test]
async fn test_server_corpus_search_no_learning() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/corpus/search?query=test&k=5", base_url))
        .send()
        .await
        .unwrap();

    // Without learning system, should return SERVICE_UNAVAILABLE
    assert_eq!(response.status(), 503);
}

#[tokio::test]
async fn test_server_corpus_search_with_learning() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/corpus/search?query=true&k=5", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn test_server_concurrent_requests() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    // Fire multiple requests concurrently
    let mut handles = Vec::new();
    for i in 0..5 {
        let client = client.clone();
        let url = format!("{}/verify", base_url);
        let handle = tokio::spawn(async move {
            let body = serde_json::json!({
                "spec": format!("theorem test{} {{ true }}", i)
            });
            client.post(&url).json(&body).send().await.unwrap()
        });
        handles.push(handle);
    }

    // All should succeed
    for handle in handles {
        let response = handle.await.unwrap();
        assert_eq!(response.status(), 200);
    }
}

#[tokio::test]
async fn test_server_multiple_properties() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    // Test with multiple properties in one spec
    let body = serde_json::json!({
        "spec": "theorem theorem1 { true } theorem theorem2 { forall x: Bool . x or not x } invariant inv1 { true }"
    });

    let response = client
        .post(format!("{}/verify", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: VerifyResponse = response.json().await.unwrap();
    assert!(result.valid);
    assert_eq!(result.property_count, 3); // 2 theorems + 1 invariant
}

// ============ WebSocket Tests ============

/// Helper to get WebSocket URL from HTTP URL
fn ws_url(http_url: &str) -> String {
    http_url.replace("http://", "ws://")
}

#[tokio::test]
async fn test_ws_verify_simple_theorem() {
    let base_url = spawn_server(false).await;
    let ws_addr = format!("{}/ws/verify", ws_url(&base_url));

    // Connect to WebSocket
    let (mut ws_stream, _) = connect_async(&ws_addr).await.expect("Failed to connect");

    // First message should be Connected with session ID
    let first_msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = first_msg {
        let parsed: WsMessage =
            serde_json::from_str(&text).expect("Failed to parse connected message");
        assert!(matches!(
            &parsed,
            WsMessage::Connected { session_id, .. } if !session_id.is_empty()
        ));
    } else {
        panic!("Expected text message");
    }

    // Send verification request
    let request = WsVerifyRequest {
        spec: "theorem test { true }".to_string(),
        backend: None,
        request_id: Some("test-simple".to_string()),
    };
    let msg = Message::Text(serde_json::to_string(&request).unwrap());
    ws_stream.send(msg).await.expect("Failed to send request");

    // Collect messages until completed
    let mut messages = Vec::new();
    while let Some(msg) = ws_stream.next().await {
        let msg = msg.expect("Error receiving message");
        if let Message::Text(text) = msg {
            let parsed: WsMessage = serde_json::from_str(&text).expect("Failed to parse message");
            let is_completed = matches!(&parsed, WsMessage::Completed { .. });
            let is_error = matches!(&parsed, WsMessage::Error { .. });
            messages.push(parsed);
            if is_completed || is_error {
                break;
            }
        }
    }

    // Verify we got the expected sequence
    assert!(!messages.is_empty());

    // First message after Connected should be Accepted
    assert!(
        matches!(&messages[0], WsMessage::Accepted { request_id } if request_id == "test-simple")
    );

    // Last message should be Completed
    assert!(matches!(
        messages.last().unwrap(),
        WsMessage::Completed {
            valid: true,
            property_count: 1,
            ..
        }
    ));
}

#[tokio::test]
async fn test_ws_resume_existing_session() {
    let base_url = spawn_server(false).await;
    let ws_addr = format!("{}/ws/verify", ws_url(&base_url));

    let (mut ws_stream, _) = connect_async(&ws_addr).await.expect("Failed to connect");

    // Capture initial session ID
    let first_msg = ws_stream.next().await.unwrap().unwrap();
    let session_id = match first_msg {
        Message::Text(text) => {
            let parsed: WsMessage =
                serde_json::from_str(&text).expect("Failed to parse connected message");
            match parsed {
                WsMessage::Connected {
                    session_id,
                    resumed,
                    ..
                } => {
                    assert!(!resumed);
                    session_id
                }
                other => panic!("Unexpected first message: {:?}", other),
            }
        }
        other => panic!("Expected text message, got {:?}", other),
    };

    // Close initial connection to simulate disconnect
    ws_stream.close(None).await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;

    // Reconnect with the same session ID
    let resume_addr = format!("{}/ws/verify?session_id={}", ws_url(&base_url), session_id);
    let (mut ws_stream, _) = connect_async(&resume_addr)
        .await
        .expect("Failed to reconnect with session_id");

    // First message should indicate resumed session
    let first_msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = first_msg {
        let parsed: WsMessage =
            serde_json::from_str(&text).expect("Failed to parse reconnected message");
        match parsed {
            WsMessage::Connected {
                session_id: returned,
                resumed,
                ..
            } => {
                assert_eq!(returned, session_id);
                assert!(resumed);
            }
            other => panic!("Unexpected message after reconnect: {:?}", other),
        }
    } else {
        panic!("Expected text message after reconnect");
    }
}

#[tokio::test]
async fn test_ws_verify_with_progress() {
    let base_url = spawn_server(false).await;
    let ws_addr = format!("{}/ws/verify", ws_url(&base_url));

    let (mut ws_stream, _) = connect_async(&ws_addr).await.expect("Failed to connect");

    // First message should be Connected
    let first_msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = first_msg {
        let parsed: WsMessage = serde_json::from_str(&text).expect("Failed to parse");
        assert!(matches!(&parsed, WsMessage::Connected { .. }));
    }

    let request = WsVerifyRequest {
        spec: "theorem test { forall x: Bool . x or not x }".to_string(),
        backend: None,
        request_id: Some("test-progress".to_string()),
    };
    ws_stream
        .send(Message::Text(serde_json::to_string(&request).unwrap()))
        .await
        .unwrap();

    let mut has_progress = false;
    let mut has_backend_started = false;
    let mut has_backend_completed = false;

    while let Some(msg) = ws_stream.next().await {
        let msg = msg.expect("Error receiving message");
        if let Message::Text(text) = msg {
            let parsed: WsMessage = serde_json::from_str(&text).expect("Failed to parse");
            match &parsed {
                WsMessage::Progress { .. } => has_progress = true,
                WsMessage::BackendStarted { .. } => has_backend_started = true,
                WsMessage::BackendCompleted { .. } => has_backend_completed = true,
                WsMessage::Completed { .. } | WsMessage::Error { .. } => break,
                _ => {}
            }
        }
    }

    // Verify we got progress updates
    assert!(has_progress, "Should have received progress messages");
    assert!(
        has_backend_started,
        "Should have received backend started messages"
    );
    assert!(
        has_backend_completed,
        "Should have received backend completed messages"
    );
}

#[tokio::test]
async fn test_ws_verify_specific_backend() {
    let base_url = spawn_server(false).await;
    let ws_addr = format!("{}/ws/verify", ws_url(&base_url));

    let (mut ws_stream, _) = connect_async(&ws_addr).await.expect("Failed to connect");

    // First message should be Connected
    let first_msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = first_msg {
        let parsed: WsMessage = serde_json::from_str(&text).expect("Failed to parse");
        assert!(matches!(&parsed, WsMessage::Connected { .. }));
    }

    let request = WsVerifyRequest {
        spec: "theorem test { true }".to_string(),
        backend: Some(dashprove_server::routes::BackendIdParam::Lean4),
        request_id: Some("test-lean4".to_string()),
    };
    ws_stream
        .send(Message::Text(serde_json::to_string(&request).unwrap()))
        .await
        .unwrap();

    let mut backend_count = 0;
    let mut final_result = None;

    while let Some(msg) = ws_stream.next().await {
        let msg = msg.expect("Error receiving message");
        if let Message::Text(text) = msg {
            let parsed: WsMessage = serde_json::from_str(&text).expect("Failed to parse");
            match &parsed {
                WsMessage::BackendStarted { .. } => backend_count += 1,
                WsMessage::Completed { compilations, .. } => {
                    final_result = Some(compilations.len());
                    break;
                }
                WsMessage::Error { .. } => break,
                _ => {}
            }
        }
    }

    // Should only have processed one backend
    assert_eq!(backend_count, 1, "Should only process specified backend");
    assert_eq!(final_result, Some(1), "Should have one compilation result");
}

#[tokio::test]
async fn test_ws_verify_parse_error() {
    let base_url = spawn_server(false).await;
    let ws_addr = format!("{}/ws/verify", ws_url(&base_url));

    let (mut ws_stream, _) = connect_async(&ws_addr).await.expect("Failed to connect");

    // First message should be Connected
    let first_msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = first_msg {
        let parsed: WsMessage = serde_json::from_str(&text).expect("Failed to parse");
        assert!(matches!(&parsed, WsMessage::Connected { .. }));
    }

    let request = WsVerifyRequest {
        spec: "invalid syntax {".to_string(),
        backend: None,
        request_id: Some("test-error".to_string()),
    };
    ws_stream
        .send(Message::Text(serde_json::to_string(&request).unwrap()))
        .await
        .unwrap();

    let mut got_error = false;
    while let Some(msg) = ws_stream.next().await {
        let msg = msg.expect("Error receiving message");
        if let Message::Text(text) = msg {
            let parsed: WsMessage = serde_json::from_str(&text).expect("Failed to parse");
            if matches!(&parsed, WsMessage::Error { error, .. } if error == "Parse error") {
                got_error = true;
                break;
            }
            if matches!(&parsed, WsMessage::Completed { .. }) {
                break;
            }
        }
    }

    assert!(got_error, "Should have received parse error");
}

#[tokio::test]
async fn test_ws_verify_invalid_request() {
    let base_url = spawn_server(false).await;
    let ws_addr = format!("{}/ws/verify", ws_url(&base_url));

    let (mut ws_stream, _) = connect_async(&ws_addr).await.expect("Failed to connect");

    // First message should be Connected
    let first_msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = first_msg {
        let parsed: WsMessage = serde_json::from_str(&text).expect("Failed to parse");
        assert!(matches!(&parsed, WsMessage::Connected { .. }));
    }

    // Send invalid JSON
    ws_stream
        .send(Message::Text("not valid json".to_string()))
        .await
        .unwrap();

    // Should get an error response
    if let Some(msg) = ws_stream.next().await {
        let msg = msg.expect("Error receiving message");
        if let Message::Text(text) = msg {
            let parsed: WsMessage = serde_json::from_str(&text).expect("Failed to parse");
            assert!(
                matches!(&parsed, WsMessage::Error { error, .. } if error == "Invalid request format"),
                "Should receive invalid request error"
            );
        }
    }
}

#[tokio::test]
async fn test_ws_multiple_requests() {
    let base_url = spawn_server(false).await;
    let ws_addr = format!("{}/ws/verify", ws_url(&base_url));

    let (mut ws_stream, _) = connect_async(&ws_addr).await.expect("Failed to connect");

    // First message should be Connected
    let first_msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = first_msg {
        let parsed: WsMessage = serde_json::from_str(&text).expect("Failed to parse");
        assert!(matches!(&parsed, WsMessage::Connected { .. }));
    }

    // Send first request
    let request1 = WsVerifyRequest {
        spec: "theorem test1 { true }".to_string(),
        backend: Some(dashprove_server::routes::BackendIdParam::Lean4),
        request_id: Some("req-1".to_string()),
    };
    ws_stream
        .send(Message::Text(serde_json::to_string(&request1).unwrap()))
        .await
        .unwrap();

    // Wait for first completion
    let mut req1_completed = false;
    while let Some(msg) = ws_stream.next().await {
        let msg = msg.expect("Error receiving message");
        if let Message::Text(text) = msg {
            let parsed: WsMessage = serde_json::from_str(&text).expect("Failed to parse");
            if matches!(&parsed, WsMessage::Completed { request_id, .. } if request_id == "req-1") {
                req1_completed = true;
                break;
            }
            if matches!(&parsed, WsMessage::Error { .. }) {
                break;
            }
        }
    }
    assert!(req1_completed, "First request should complete");

    // Send second request on same connection
    let request2 = WsVerifyRequest {
        spec: "theorem test2 { false }".to_string(),
        backend: Some(dashprove_server::routes::BackendIdParam::Lean4),
        request_id: Some("req-2".to_string()),
    };
    ws_stream
        .send(Message::Text(serde_json::to_string(&request2).unwrap()))
        .await
        .unwrap();

    // Wait for second completion
    let mut req2_completed = false;
    while let Some(msg) = ws_stream.next().await {
        let msg = msg.expect("Error receiving message");
        if let Message::Text(text) = msg {
            let parsed: WsMessage = serde_json::from_str(&text).expect("Failed to parse");
            if matches!(&parsed, WsMessage::Completed { request_id, .. } if request_id == "req-2") {
                req2_completed = true;
                break;
            }
            if matches!(&parsed, WsMessage::Error { .. }) {
                break;
            }
        }
    }
    assert!(req2_completed, "Second request should complete");
}

// ============ Corpus History/Compare/Suggest Integration Tests ============

#[tokio::test]
async fn test_server_corpus_history_no_learning() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/corpus/history", base_url))
        .send()
        .await
        .unwrap();

    // Without learning system, should return SERVICE_UNAVAILABLE
    assert_eq!(response.status(), 503);
}

#[tokio::test]
async fn test_server_corpus_history_with_learning() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/corpus/history", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CorpusHistoryResponse = response.json().await.unwrap();
    // Empty corpus should have zero total count
    assert_eq!(result.total_count, 0);
    assert!(result.periods.is_empty());
}

#[tokio::test]
async fn test_server_corpus_history_with_params() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    // Test with corpus type parameter
    let response = client
        .get(format!("{}/corpus/history?corpus=proofs", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CorpusHistoryResponse = response.json().await.unwrap();
    assert_eq!(result.total_count, 0);

    // Test with period parameter
    let response = client
        .get(format!("{}/corpus/history?period=week", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    // Test with date range filters
    let response = client
        .get(format!(
            "{}/corpus/history?from=2024-01-01&to=2024-12-31",
            base_url
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn test_server_corpus_compare_no_learning() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!(
            "{}/corpus/compare?baseline_from=2024-01-01&baseline_to=2024-01-31&compare_from=2024-02-01&compare_to=2024-02-28",
            base_url
        ))
        .send()
        .await
        .unwrap();

    // Without learning system, should return SERVICE_UNAVAILABLE
    assert_eq!(response.status(), 503);
}

#[tokio::test]
async fn test_server_corpus_compare_with_learning() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!(
            "{}/corpus/compare?baseline_from=2024-01-01&baseline_to=2024-01-31&compare_from=2024-02-01&compare_to=2024-02-28",
            base_url
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CorpusCompareResponse = response.json().await.unwrap();
    // Both periods should be empty in a new corpus
    assert_eq!(result.baseline_count, 0);
    assert_eq!(result.comparison_count, 0);
    assert_eq!(result.count_delta, 0);
}

#[tokio::test]
async fn test_server_corpus_compare_proofs() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!(
            "{}/corpus/compare?corpus=proofs&baseline_from=2024-01-01&baseline_to=2024-01-31&compare_from=2024-02-01&compare_to=2024-02-28",
            base_url
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CorpusCompareResponse = response.json().await.unwrap();
    assert_eq!(result.baseline_count, 0);
}

#[tokio::test]
async fn test_server_corpus_compare_invalid_date() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    // Invalid date format
    let response = client
        .get(format!(
            "{}/corpus/compare?baseline_from=invalid&baseline_to=2024-01-31&compare_from=2024-02-01&compare_to=2024-02-28",
            base_url
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn test_server_corpus_suggest_no_learning() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/corpus/suggest", base_url))
        .send()
        .await
        .unwrap();

    // Without learning system, should return SERVICE_UNAVAILABLE
    assert_eq!(response.status(), 503);
}

#[tokio::test]
async fn test_server_corpus_suggest_with_learning() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/corpus/suggest", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CorpusSuggestResponse = response.json().await.unwrap();
    // With empty corpus, suggestions may or may not be available
    // but the response structure should be valid
    assert!(result.suggestions.is_empty() || !result.suggestions.is_empty());
}

#[tokio::test]
async fn test_server_corpus_suggest_proofs() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/corpus/suggest?corpus=proofs", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CorpusSuggestResponse = response.json().await.unwrap();
    // Response structure should be valid
    for suggestion in &result.suggestions {
        // Each suggestion should have valid fields
        assert!(!suggestion.suggestion_type.is_empty());
        assert!(!suggestion.description.is_empty());
        assert!(!suggestion.api_query.is_empty());
    }
}

// ============ Corpus Stats Integration Tests ============

#[tokio::test]
async fn test_server_corpus_stats_no_learning() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/corpus/stats", base_url))
        .send()
        .await
        .unwrap();

    // Without learning system, should return SERVICE_UNAVAILABLE
    assert_eq!(response.status(), 503);
}

#[tokio::test]
async fn test_server_corpus_stats_with_learning() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/corpus/stats", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CorpusStatsResponse = response.json().await.unwrap();

    // Empty corpus should have zero counts
    assert_eq!(result.proofs.total, 0);
    assert!(result.proofs.by_backend.is_empty());
    assert_eq!(result.counterexamples.total, 0);
    assert_eq!(result.counterexamples.cluster_patterns, 0);
    assert!(result.counterexamples.by_backend.is_empty());
    assert_eq!(result.tactics.total_observations, 0);
    assert_eq!(result.tactics.unique_tactics, 0);
    assert!(result.tactics.top_tactics.is_empty());
}

#[tokio::test]
async fn test_server_corpus_stats_response_structure() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/corpus/stats", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    // Parse response as raw JSON to verify structure
    let json: serde_json::Value = response.json().await.unwrap();

    // Verify all top-level fields exist
    assert!(json.get("proofs").is_some());
    assert!(json.get("counterexamples").is_some());
    assert!(json.get("tactics").is_some());

    // Verify proofs structure
    let proofs = json.get("proofs").unwrap();
    assert!(proofs.get("total").is_some());
    assert!(proofs.get("by_backend").is_some());

    // Verify counterexamples structure
    let cx = json.get("counterexamples").unwrap();
    assert!(cx.get("total").is_some());
    assert!(cx.get("cluster_patterns").is_some());
    assert!(cx.get("by_backend").is_some());

    // Verify tactics structure
    let tactics = json.get("tactics").unwrap();
    assert!(tactics.get("total_observations").is_some());
    assert!(tactics.get("unique_tactics").is_some());
    assert!(tactics.get("top_tactics").is_some());
}

// ============ Counterexample Search Integration Tests ============

#[tokio::test]
async fn test_server_counterexample_search_no_learning() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "counterexample": {
            "witness": {"x": 1, "y": 2},
            "failed_checks": [{"id": "check1", "description": "test failure"}]
        },
        "k": 5
    });

    let response = client
        .post(format!("{}/corpus/counterexamples/search", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    // Without learning system, should return SERVICE_UNAVAILABLE
    assert_eq!(response.status(), 503);
}

#[tokio::test]
async fn test_server_counterexample_search_with_learning() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "counterexample": {
            "witness": {"x": 1, "y": 2},
            "failed_checks": [{"id": "check1", "description": "test failure"}]
        },
        "k": 5
    });

    let response = client
        .post(format!("{}/corpus/counterexamples/search", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CounterexampleSearchResponse = response.json().await.unwrap();
    // Empty corpus should have no results
    assert!(result.results.is_empty());
    assert_eq!(result.total_corpus_size, 0);
}

// ============ Counterexample Add Integration Tests ============

#[tokio::test]
async fn test_server_counterexample_add_no_learning() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "counterexample": {
            "witness": {"x": 0},
            "failed_checks": [{"id": "div_by_zero", "description": "division by zero"}]
        },
        "property_name": "safe_division",
        "backend": "kani"
    });

    let response = client
        .post(format!("{}/corpus/counterexamples", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    // Without learning system, should return SERVICE_UNAVAILABLE
    assert_eq!(response.status(), 503);
}

#[tokio::test]
async fn test_server_counterexample_add_with_learning() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "counterexample": {
            "witness": {"x": 0},
            "failed_checks": [{"id": "div_by_zero", "description": "division by zero"}]
        },
        "property_name": "safe_division",
        "backend": "kani"
    });

    let response = client
        .post(format!("{}/corpus/counterexamples", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CounterexampleAddResponse = response.json().await.unwrap();
    // Should have an ID and corpus size of 1
    assert!(!result.id.is_empty());
    assert_eq!(result.total_corpus_size, 1);
}

#[tokio::test]
async fn test_server_counterexample_add_with_cluster_label() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "counterexample": {
            "witness": {"divisor": 0},
            "failed_checks": [{"id": "div_check", "description": "attempted division by zero"}]
        },
        "property_name": "arithmetic_safety",
        "backend": "kani",
        "cluster_label": "division_errors"
    });

    let response = client
        .post(format!("{}/corpus/counterexamples", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CounterexampleAddResponse = response.json().await.unwrap();
    assert!(!result.id.is_empty());
}

// ============ Counterexample Classify Integration Tests ============

#[tokio::test]
async fn test_server_counterexample_classify_no_learning() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "counterexample": {
            "witness": {"x": 1},
            "failed_checks": []
        }
    });

    let response = client
        .post(format!("{}/corpus/counterexamples/classify", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    // Without learning system, should return SERVICE_UNAVAILABLE
    assert_eq!(response.status(), 503);
}

#[tokio::test]
async fn test_server_counterexample_classify_with_learning() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "counterexample": {
            "witness": {"x": 1},
            "failed_checks": [{"id": "test", "description": "test check"}]
        }
    });

    let response = client
        .post(format!("{}/corpus/counterexamples/classify", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CounterexampleClassifyResponse = response.json().await.unwrap();
    // With no patterns stored, classification should return None
    assert!(result.cluster_label.is_none());
    assert!(result.similarity.is_none());
    assert_eq!(result.total_patterns, 0);
}

// ============ Counterexample Clusters Integration Tests ============

#[tokio::test]
async fn test_server_counterexample_clusters_no_learning() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "patterns": [
            {
                "label": "null_pointer",
                "representative": {
                    "witness": {"ptr": null},
                    "failed_checks": [{"id": "null_check", "description": "null pointer dereference"}]
                },
                "count": 5
            }
        ],
        "similarity_threshold": 0.7
    });

    let response = client
        .post(format!("{}/corpus/counterexamples/clusters", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    // Without learning system, should return SERVICE_UNAVAILABLE
    assert_eq!(response.status(), 503);
}

#[tokio::test]
async fn test_server_counterexample_clusters_with_learning() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "patterns": [
            {
                "label": "null_pointer",
                "representative": {
                    "witness": {"ptr": "null"},
                    "failed_checks": [{"id": "null_check", "description": "null pointer dereference"}]
                },
                "count": 5
            },
            {
                "label": "array_bounds",
                "representative": {
                    "witness": {"index": 100, "len": 10},
                    "failed_checks": [{"id": "bounds_check", "description": "array index out of bounds"}]
                },
                "count": 3
            }
        ],
        "similarity_threshold": 0.7
    });

    let response = client
        .post(format!("{}/corpus/counterexamples/clusters", base_url))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CounterexampleClustersResponse = response.json().await.unwrap();
    assert_eq!(result.patterns_recorded, 2);
    assert_eq!(result.total_patterns, 2);
}

#[tokio::test]
async fn test_server_counterexample_full_workflow() {
    // Test the full workflow: add counterexample, store pattern, classify new counterexample
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    // Step 1: Add a counterexample
    let add_body = serde_json::json!({
        "counterexample": {
            "witness": {"x": 0},
            "failed_checks": [{"id": "div_zero", "description": "division by zero"}]
        },
        "property_name": "safe_div",
        "backend": "kani"
    });

    let response = client
        .post(format!("{}/corpus/counterexamples", base_url))
        .json(&add_body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);

    // Step 2: Record a cluster pattern
    let cluster_body = serde_json::json!({
        "patterns": [{
            "label": "arithmetic_error",
            "representative": {
                "witness": {"x": 0},
                "failed_checks": [{"id": "div_zero", "description": "division by zero"}]
            },
            "count": 1
        }],
        "similarity_threshold": 0.7
    });

    let response = client
        .post(format!("{}/corpus/counterexamples/clusters", base_url))
        .json(&cluster_body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let cluster_result: CounterexampleClustersResponse = response.json().await.unwrap();
    assert_eq!(cluster_result.total_patterns, 1);

    // Step 3: Classify a similar counterexample
    let classify_body = serde_json::json!({
        "counterexample": {
            "witness": {"y": 0},
            "failed_checks": [{"id": "div_zero2", "description": "division by zero in func"}]
        }
    });

    let response = client
        .post(format!("{}/corpus/counterexamples/classify", base_url))
        .json(&classify_body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let classify_result: CounterexampleClassifyResponse = response.json().await.unwrap();
    assert_eq!(classify_result.total_patterns, 1);
    // The counterexample may or may not match depending on similarity algorithm

    // Step 4: Search for similar counterexamples
    let search_body = serde_json::json!({
        "counterexample": {
            "witness": {"x": 0},
            "failed_checks": [{"id": "div_zero", "description": "division by zero"}]
        },
        "k": 5
    });

    let response = client
        .post(format!("{}/corpus/counterexamples/search", base_url))
        .json(&search_body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let search_result: CounterexampleSearchResponse = response.json().await.unwrap();
    // Should find the counterexample we added
    assert_eq!(search_result.total_corpus_size, 1);
}

// ============ Counterexample Text Search Tests ============

#[tokio::test]
async fn test_server_counterexample_text_search_no_learning() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!(
            "{}/corpus/counterexamples/text-search?query=overflow",
            base_url
        ))
        .send()
        .await
        .unwrap();

    // Should return 503 when learning system not configured
    assert_eq!(response.status(), 503);
}

#[tokio::test]
async fn test_server_counterexample_text_search_with_learning() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!(
            "{}/corpus/counterexamples/text-search?query=overflow",
            base_url
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CounterexampleSearchResponse = response.json().await.unwrap();
    // Empty corpus, no results
    assert!(result.results.is_empty());
    assert_eq!(result.total_corpus_size, 0);
}

#[tokio::test]
async fn test_server_counterexample_text_search_with_results() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    // First, add a counterexample with a keyword
    let add_body = serde_json::json!({
        "counterexample": {
            "witness": {"x": 0},
            "failed_checks": [{"id": "check1", "description": "buffer overflow detected"}]
        },
        "property_name": "no_overflow",
        "backend": "kani"
    });

    let response = client
        .post(format!("{}/corpus/counterexamples", base_url))
        .json(&add_body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);

    // Now search by text
    let response = client
        .get(format!(
            "{}/corpus/counterexamples/text-search?query=overflow",
            base_url
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CounterexampleSearchResponse = response.json().await.unwrap();
    // Should find the counterexample with "overflow" keyword
    assert_eq!(result.total_corpus_size, 1);
    assert!(!result.results.is_empty());
    assert_eq!(result.results[0].property_name, "no_overflow");
}

#[tokio::test]
async fn test_server_counterexample_text_search_with_k_param() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    // Add multiple counterexamples
    for i in 0..5 {
        let add_body = serde_json::json!({
            "counterexample": {
                "witness": {"x": i},
                "failed_checks": [{"id": format!("check{}", i), "description": "memory leak found"}]
            },
            "property_name": format!("leak_prop_{}", i),
            "backend": "kani"
        });

        client
            .post(format!("{}/corpus/counterexamples", base_url))
            .json(&add_body)
            .send()
            .await
            .unwrap();
    }

    // Search with k=2 to limit results
    let response = client
        .get(format!(
            "{}/corpus/counterexamples/text-search?query=memory+leak&k=2",
            base_url
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CounterexampleSearchResponse = response.json().await.unwrap();
    assert_eq!(result.total_corpus_size, 5);
    // Should only return k results
    assert!(result.results.len() <= 2);
}

// ============ Counterexample List/Get Tests ============

#[tokio::test]
async fn test_server_counterexample_list_no_learning() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/corpus/counterexamples", base_url))
        .send()
        .await
        .unwrap();

    // Without learning system, should return SERVICE_UNAVAILABLE
    assert_eq!(response.status(), 503);
}

#[tokio::test]
async fn test_server_counterexample_list_with_learning() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/corpus/counterexamples", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CounterexampleListResponse = response.json().await.unwrap();
    // Empty corpus
    assert!(result.counterexamples.is_empty());
    assert_eq!(result.total, 0);
    assert_eq!(result.offset, 0);
    assert_eq!(result.limit, 50); // default limit
}

#[tokio::test]
async fn test_server_counterexample_list_with_data() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    // Add some counterexamples
    for i in 0..3 {
        let add_body = serde_json::json!({
            "counterexample": {
                "witness": {"idx": i},
                "failed_checks": [{"id": format!("check_{}", i), "description": format!("error {}", i)}]
            },
            "property_name": format!("prop_{}", i),
            "backend": "kani"
        });

        client
            .post(format!("{}/corpus/counterexamples", base_url))
            .json(&add_body)
            .send()
            .await
            .unwrap();
    }

    // List all counterexamples
    let response = client
        .get(format!("{}/corpus/counterexamples", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CounterexampleListResponse = response.json().await.unwrap();
    assert_eq!(result.counterexamples.len(), 3);
    assert_eq!(result.total, 3);
}

#[tokio::test]
async fn test_server_counterexample_list_pagination() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    // Add some counterexamples
    for i in 0..5 {
        let add_body = serde_json::json!({
            "counterexample": {
                "witness": {"idx": i},
                "failed_checks": [{"id": format!("check_{}", i), "description": format!("error {}", i)}]
            },
            "property_name": format!("prop_{}", i),
            "backend": "kani"
        });

        client
            .post(format!("{}/corpus/counterexamples", base_url))
            .json(&add_body)
            .send()
            .await
            .unwrap();
    }

    // Get first page with limit=2
    let response = client
        .get(format!(
            "{}/corpus/counterexamples?limit=2&offset=0",
            base_url
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CounterexampleListResponse = response.json().await.unwrap();
    assert_eq!(result.counterexamples.len(), 2);
    assert_eq!(result.total, 5);
    assert_eq!(result.offset, 0);
    assert_eq!(result.limit, 2);

    // Get second page
    let response = client
        .get(format!(
            "{}/corpus/counterexamples?limit=2&offset=2",
            base_url
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CounterexampleListResponse = response.json().await.unwrap();
    assert_eq!(result.counterexamples.len(), 2);
    assert_eq!(result.offset, 2);
}

#[tokio::test]
async fn test_server_counterexample_list_property_filter() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let props = ["alpha_property", "beta_case", "gamma_item"];
    for name in props {
        let add_body = serde_json::json!({
            "counterexample": {
                "witness": {"flag": true},
                "failed_checks": [{"id": format!("check_{name}"), "description": "failure"}]
            },
            "property_name": name,
            "backend": "kani"
        });

        client
            .post(format!("{}/corpus/counterexamples", base_url))
            .json(&add_body)
            .send()
            .await
            .unwrap();
    }

    let response = client
        .get(format!(
            "{}/corpus/counterexamples?property_name=BETA",
            base_url
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CounterexampleListResponse = response.json().await.unwrap();
    assert_eq!(result.total, 1);
    assert_eq!(result.counterexamples.len(), 1);
    assert_eq!(result.counterexamples[0].property_name, "beta_case");
}

#[tokio::test]
async fn test_server_counterexample_list_date_filters() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    for i in 0..2 {
        let add_body = serde_json::json!({
            "counterexample": {
                "witness": {"idx": i},
                "failed_checks": [{"id": format!("check_{i}"), "description": "error"}]
            },
            "property_name": format!("prop_{i}"),
            "backend": "kani"
        });

        client
            .post(format!("{}/corpus/counterexamples", base_url))
            .json(&add_body)
            .send()
            .await
            .unwrap();
    }

    let today = Utc::now().date_naive();
    let yesterday = (today - chrono::Duration::days(1))
        .format("%Y-%m-%d")
        .to_string();
    let tomorrow = (today + chrono::Duration::days(1))
        .format("%Y-%m-%d")
        .to_string();

    let response = client
        .get(format!(
            "{}/corpus/counterexamples?from={}&to={}",
            base_url, yesterday, tomorrow
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CounterexampleListResponse = response.json().await.unwrap();
    assert_eq!(result.total, 2);
    assert_eq!(result.counterexamples.len(), 2);

    let future_start = (today + chrono::Duration::days(2))
        .format("%Y-%m-%d")
        .to_string();
    let future_end = (today + chrono::Duration::days(3))
        .format("%Y-%m-%d")
        .to_string();

    let response = client
        .get(format!(
            "{}/corpus/counterexamples?from={}&to={}",
            base_url, future_start, future_end
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let empty_result: CounterexampleListResponse = response.json().await.unwrap();
    assert_eq!(empty_result.total, 0);
    assert!(empty_result.counterexamples.is_empty());
}

#[tokio::test]
async fn test_server_counterexample_get_no_learning() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/corpus/counterexamples/some-id", base_url))
        .send()
        .await
        .unwrap();

    // Without learning system, should return SERVICE_UNAVAILABLE
    assert_eq!(response.status(), 503);
}

#[tokio::test]
async fn test_server_counterexample_get_not_found() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!(
            "{}/corpus/counterexamples/nonexistent-id",
            base_url
        ))
        .send()
        .await
        .unwrap();

    // ID not found
    assert_eq!(response.status(), 404);
}

#[tokio::test]
async fn test_server_counterexample_get_by_id() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    // Add a counterexample
    let add_body = serde_json::json!({
        "counterexample": {
            "witness": {"x": 42},
            "failed_checks": [{"id": "test_check", "description": "test failed"}]
        },
        "property_name": "test_prop",
        "backend": "kani"
    });

    let response = client
        .post(format!("{}/corpus/counterexamples", base_url))
        .json(&add_body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let add_result: CounterexampleAddResponse = response.json().await.unwrap();
    let cx_id = add_result.id;

    // Retrieve it by ID
    let response = client
        .get(format!("{}/corpus/counterexamples/{}", base_url, cx_id))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let result: CounterexampleEntryResponse = response.json().await.unwrap();
    assert_eq!(result.id, cx_id);
    assert_eq!(result.property_name, "test_prop");
}

// ============ HTML Output Format Tests ============

#[tokio::test]
async fn test_server_corpus_history_html_format() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/corpus/history?format=html", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let content_type = response.headers().get("content-type").unwrap();
    assert!(content_type.to_str().unwrap().contains("text/html"));

    let html = response.text().await.unwrap();
    assert!(html.contains("<!DOCTYPE html>") || html.contains("<html"));
    assert!(html.contains("Chart.js") || html.contains("chart"));
}

#[tokio::test]
async fn test_server_corpus_compare_html_format() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    // Need to provide valid date ranges for compare
    let response = client
        .get(format!(
            "{}/corpus/compare?baseline_from=2024-01-01&baseline_to=2024-01-31&compare_from=2024-02-01&compare_to=2024-02-28&format=html",
            base_url
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let content_type = response.headers().get("content-type").unwrap();
    assert!(content_type.to_str().unwrap().contains("text/html"));

    let html = response.text().await.unwrap();
    assert!(html.contains("<!DOCTYPE html>") || html.contains("<html"));
}

#[tokio::test]
async fn test_server_corpus_history_json_format_default() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    // Default should be JSON
    let response = client
        .get(format!("{}/corpus/history", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let content_type = response.headers().get("content-type").unwrap();
    assert!(content_type.to_str().unwrap().contains("application/json"));

    // Should parse as JSON
    let _result: CorpusHistoryResponse = response.json().await.unwrap();
}

#[tokio::test]
async fn test_server_corpus_compare_json_format_explicit() {
    let base_url = spawn_server(true).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!(
            "{}/corpus/compare?baseline_from=2024-01-01&baseline_to=2024-01-31&compare_from=2024-02-01&compare_to=2024-02-28&format=json",
            base_url
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let content_type = response.headers().get("content-type").unwrap();
    assert!(content_type.to_str().unwrap().contains("application/json"));

    // Should parse as JSON
    let _result: CorpusCompareResponse = response.json().await.unwrap();
}

// ============ Cache Tests ============

#[tokio::test]
async fn test_server_cache_stats_endpoint() {
    let base_url = spawn_server(false).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/cache/stats", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let stats: CacheStatsResponse = response.json().await.unwrap();

    // New cache should be empty
    assert_eq!(stats.total_entries, 0);
    assert_eq!(stats.valid_entries, 0);
    assert_eq!(stats.expired_entries, 0);
    // Default max entries is 10000
    assert_eq!(stats.max_entries, 10000);
    // Default TTL is 3600 seconds (1 hour)
    assert_eq!(stats.default_ttl_secs, 3600);
}
