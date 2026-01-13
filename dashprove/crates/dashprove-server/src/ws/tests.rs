//! WebSocket tests

use super::admin::{ListSessionsResponse, SessionInfoResponse};
use super::messages::{VerificationPhase, WsMessage, WsVerifyRequest};
use super::session::SessionManager;
use crate::routes::{AppState, BackendIdParam, CompilationResult};
use axum::{body::Body, http::Request, http::StatusCode, routing::get, Router};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

#[test]
fn test_ws_message_serialization() {
    let msg = WsMessage::Accepted {
        request_id: "test-123".to_string(),
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"accepted\""));
    assert!(json.contains("\"request_id\":\"test-123\""));
}

#[test]
fn test_ws_progress_message() {
    let msg = WsMessage::Progress {
        request_id: "test-456".to_string(),
        phase: VerificationPhase::Parsing,
        message: "Parsing...".to_string(),
        percentage: Some(25),
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"progress\""));
    assert!(json.contains("\"phase\":\"parsing\""));
    assert!(json.contains("\"percentage\":25"));
}

#[test]
fn test_ws_error_message() {
    let msg = WsMessage::Error {
        request_id: Some("test-789".to_string()),
        error: "Parse failed".to_string(),
        details: Some("unexpected token".to_string()),
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"error\""));
    assert!(json.contains("\"error\":\"Parse failed\""));
}

#[test]
fn test_ws_backend_completed_message() {
    let msg = WsMessage::BackendCompleted {
        request_id: "test-abc".to_string(),
        backend: BackendIdParam::Lean4,
        result: CompilationResult {
            backend: BackendIdParam::Lean4,
            code: "theorem test : True := trivial".to_string(),
        },
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"backend_completed\""));
    assert!(json.contains("\"backend\":\"lean4\""));
}

#[test]
fn test_ws_completed_message() {
    let msg = WsMessage::Completed {
        request_id: "test-final".to_string(),
        valid: true,
        property_count: 2,
        compilations: vec![],
        errors: vec![],
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"completed\""));
    assert!(json.contains("\"valid\":true"));
    assert!(json.contains("\"property_count\":2"));
}

#[test]
fn test_ws_verify_request_deserialization() {
    let json = r#"{"spec": "theorem test { true }", "backend": "lean4"}"#;
    let req: WsVerifyRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.spec, "theorem test { true }");
    assert_eq!(req.backend, Some(BackendIdParam::Lean4));
}

#[test]
fn test_ws_verify_request_minimal() {
    let json = r#"{"spec": "theorem test { true }"}"#;
    let req: WsVerifyRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.spec, "theorem test { true }");
    assert!(req.backend.is_none());
    assert!(req.request_id.is_none());
}

#[test]
fn test_verification_phase_serialization() {
    assert_eq!(
        serde_json::to_string(&VerificationPhase::Parsing).unwrap(),
        "\"parsing\""
    );
    assert_eq!(
        serde_json::to_string(&VerificationPhase::TypeChecking).unwrap(),
        "\"type_checking\""
    );
    assert_eq!(
        serde_json::to_string(&VerificationPhase::Compiling).unwrap(),
        "\"compiling\""
    );
    assert_eq!(
        serde_json::to_string(&VerificationPhase::Verifying).unwrap(),
        "\"verifying\""
    );
    assert_eq!(
        serde_json::to_string(&VerificationPhase::Merging).unwrap(),
        "\"merging\""
    );
}

#[test]
fn test_ws_connected_message() {
    let msg = WsMessage::Connected {
        session_id: "abc-123-def".to_string(),
        resumed: false,
        correlation_id: None,
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"connected\""));
    assert!(json.contains("\"session_id\":\"abc-123-def\""));
    assert!(json.contains("\"resumed\":false"));
    // correlation_id should be omitted when None
    assert!(!json.contains("correlation_id"));
}

#[test]
fn test_ws_connected_message_resumed() {
    let msg = WsMessage::Connected {
        session_id: "resumed-123".to_string(),
        resumed: true,
        correlation_id: None,
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"connected\""));
    assert!(json.contains("\"session_id\":\"resumed-123\""));
    assert!(json.contains("\"resumed\":true"));
}

#[test]
fn test_ws_connected_message_with_correlation_id() {
    let msg = WsMessage::Connected {
        session_id: "session-xyz".to_string(),
        resumed: false,
        correlation_id: Some("trace-abc-123".to_string()),
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"connected\""));
    assert!(json.contains("\"session_id\":\"session-xyz\""));
    assert!(json.contains("\"correlation_id\":\"trace-abc-123\""));
}

#[tokio::test]
async fn test_session_manager_create_session() {
    let manager = SessionManager::new();

    let session_id = manager.create_session().await;
    assert!(!session_id.is_empty());

    // Should be able to get the session
    let info = manager.get_session(&session_id).await;
    assert!(info.is_some());
    let info = info.unwrap();
    assert!(info.connected);
    assert_eq!(info.request_count, 0);
}

#[tokio::test]
async fn test_session_manager_disconnect() {
    let manager = SessionManager::new();

    let session_id = manager.create_session().await;
    assert_eq!(manager.active_count().await, 1);

    manager.disconnect_session(&session_id).await;
    assert_eq!(manager.active_count().await, 0);

    // Session should still exist but be disconnected
    let info = manager.get_session(&session_id).await.unwrap();
    assert!(!info.connected);
}

#[tokio::test]
async fn test_session_manager_reconnect_existing_session() {
    let manager = SessionManager::new();

    let session_id = manager.create_session().await;
    manager.disconnect_session(&session_id).await;

    assert_eq!(manager.active_count().await, 0);

    let info = manager.reconnect_session(&session_id).await;
    assert!(info.is_some());
    assert!(info.unwrap().connected);
    assert_eq!(manager.active_count().await, 1);
}

#[tokio::test]
async fn test_session_manager_request_count() {
    let manager = SessionManager::new();

    let session_id = manager.create_session().await;

    manager.increment_request_count(&session_id).await;
    manager.increment_request_count(&session_id).await;
    manager.increment_request_count(&session_id).await;

    let info = manager.get_session(&session_id).await.unwrap();
    assert_eq!(info.request_count, 3);
}

#[tokio::test]
async fn test_session_manager_reconnect_missing_session() {
    let manager = SessionManager::new();
    assert!(manager.reconnect_session("does-not-exist").await.is_none());
}

#[tokio::test]
async fn test_session_manager_multiple_sessions() {
    let manager = SessionManager::new();

    let session1 = manager.create_session().await;
    let session2 = manager.create_session().await;
    let session3 = manager.create_session().await;

    assert_eq!(manager.active_count().await, 3);

    manager.disconnect_session(&session2).await;
    assert_eq!(manager.active_count().await, 2);

    let active = manager.list_active_sessions().await;
    assert_eq!(active.len(), 2);
    assert!(active.iter().any(|s| s.session_id == session1));
    assert!(active.iter().any(|s| s.session_id == session3));
}

#[tokio::test]
async fn test_session_manager_cleanup() {
    let manager = SessionManager::new();

    let session1 = manager.create_session().await;
    let session2 = manager.create_session().await;

    // Disconnect one session
    manager.disconnect_session(&session1).await;

    // Cleanup with 1 hour retention - should keep both since they're recent
    manager.cleanup(Duration::from_secs(3600)).await;
    assert!(manager.get_session(&session1).await.is_some());
    assert!(manager.get_session(&session2).await.is_some());

    // Cleanup with 0 retention - should remove disconnected session
    manager.cleanup(Duration::from_secs(0)).await;
    assert!(manager.get_session(&session1).await.is_none()); // removed (disconnected)
    assert!(manager.get_session(&session2).await.is_some()); // kept (connected)
}

// ============ Session Info Response Tests ============

#[test]
fn test_session_info_response_serialization() {
    let response = SessionInfoResponse {
        session_id: "abc-123".to_string(),
        uptime_seconds: 120,
        request_count: 5,
        connected: true,
    };
    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("\"session_id\":\"abc-123\""));
    assert!(json.contains("\"uptime_seconds\":120"));
    assert!(json.contains("\"request_count\":5"));
    assert!(json.contains("\"connected\":true"));
}

#[test]
fn test_list_sessions_response_serialization() {
    let response = ListSessionsResponse {
        sessions: vec![
            SessionInfoResponse {
                session_id: "session-1".to_string(),
                uptime_seconds: 60,
                request_count: 3,
                connected: true,
            },
            SessionInfoResponse {
                session_id: "session-2".to_string(),
                uptime_seconds: 30,
                request_count: 1,
                connected: true,
            },
        ],
        active_count: 2,
    };
    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("\"active_count\":2"));
    assert!(json.contains("\"session_id\":\"session-1\""));
    assert!(json.contains("\"session_id\":\"session-2\""));
}

#[tokio::test]
async fn test_session_info_from_session_info() {
    let manager = SessionManager::new();
    let session_id = manager.create_session().await;

    // Increment request count
    manager.increment_request_count(&session_id).await;
    manager.increment_request_count(&session_id).await;

    let info = manager.get_session(&session_id).await.unwrap();
    let response = SessionInfoResponse::from_session_info(&info);

    assert_eq!(response.session_id, session_id);
    assert_eq!(response.request_count, 2);
    assert!(response.connected);
    // uptime should be very small since we just created it
    assert!(response.uptime_seconds < 5);
}

#[tokio::test]
async fn test_list_sessions_empty() {
    let state = Arc::new(AppState::new());
    let app = Router::new()
        .route("/admin/sessions", get(super::admin::list_sessions))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/admin/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: ListSessionsResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(result.active_count, 0);
    assert!(result.sessions.is_empty());
}

#[tokio::test]
async fn test_list_sessions_with_active_sessions() {
    let state = Arc::new(AppState::new());

    // Create some sessions directly via session manager
    let session1 = state.session_manager.create_session().await;
    let session2 = state.session_manager.create_session().await;
    state
        .session_manager
        .increment_request_count(&session1)
        .await;
    state
        .session_manager
        .increment_request_count(&session1)
        .await;
    state
        .session_manager
        .increment_request_count(&session2)
        .await;

    let app = Router::new()
        .route("/admin/sessions", get(super::admin::list_sessions))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/admin/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: ListSessionsResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(result.active_count, 2);
    assert_eq!(result.sessions.len(), 2);

    // Verify sessions are in response
    let session_ids: Vec<_> = result.sessions.iter().map(|s| &s.session_id).collect();
    assert!(session_ids.contains(&&session1));
    assert!(session_ids.contains(&&session2));

    // Find session1 and verify its request count
    let s1 = result
        .sessions
        .iter()
        .find(|s| s.session_id == session1)
        .unwrap();
    assert_eq!(s1.request_count, 2);

    // Find session2 and verify its request count
    let s2 = result
        .sessions
        .iter()
        .find(|s| s.session_id == session2)
        .unwrap();
    assert_eq!(s2.request_count, 1);
}

#[tokio::test]
async fn test_list_sessions_excludes_disconnected() {
    let state = Arc::new(AppState::new());

    // Create sessions and disconnect one
    let session1 = state.session_manager.create_session().await;
    let session2 = state.session_manager.create_session().await;
    state.session_manager.disconnect_session(&session2).await;

    let app = Router::new()
        .route("/admin/sessions", get(super::admin::list_sessions))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/admin/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: ListSessionsResponse = serde_json::from_slice(&body).unwrap();

    // Only session1 should be listed (session2 is disconnected)
    assert_eq!(result.active_count, 1);
    assert_eq!(result.sessions.len(), 1);
    assert_eq!(result.sessions[0].session_id, session1);
}
