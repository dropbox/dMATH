//! Status, metadata, and backend listing routes for DashProve server

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use dashprove_backends::{
    AlloyBackend, CoqBackend, DafnyBackend, IsabelleBackend, KaniBackend, Lean4Backend,
    TlaPlusBackend, VerificationBackend,
};
use std::sync::Arc;

use super::types::{
    BackendIdParam, BackendInfo, BackendsResponse, HealthResponse, HealthStatusResponse,
    VersionResponse,
};
use super::{AppState, ShutdownState};

impl HealthResponse {
    /// Create health response from current app state
    pub async fn from_state(state: &AppState) -> Self {
        let shutdown_state = state.get_shutdown_state();
        let in_flight = state.active_requests();
        let ws_sessions = state.session_manager.active_count().await;
        let (status, ready) = match shutdown_state {
            ShutdownState::Running => ("healthy".to_string(), true),
            ShutdownState::Draining => ("draining".to_string(), false),
            ShutdownState::ShuttingDown => ("unhealthy".to_string(), false),
        };
        let shutdown_state_str = match shutdown_state {
            ShutdownState::Running => "running",
            ShutdownState::Draining => "draining",
            ShutdownState::ShuttingDown => "shutting_down",
        };
        Self {
            status,
            shutdown_state: shutdown_state_str.to_string(),
            in_flight_requests: in_flight,
            active_websocket_sessions: ws_sessions,
            ready,
        }
    }
}

/// Health check endpoint
///
/// Returns detailed health status including:
/// - Overall status: "healthy", "draining", or "unhealthy"
/// - Shutdown state: "running", "draining", or "shutting_down"
/// - In-flight request count
/// - Active WebSocket session count
/// - Ready flag (false during drain/shutdown - signals load balancers to stop routing)
///
/// HTTP Status codes:
/// - 200 OK: Server is healthy and accepting requests
/// - 503 Service Unavailable: Server is draining or shutting down
pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let response = HealthResponse::from_state(&state).await;
    let status_code = if response.ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status_code, Json(response))
}

/// GET /version - API version and metadata
pub async fn version() -> Json<VersionResponse> {
    Json(VersionResponse::current())
}

/// GET /metrics - Prometheus-compatible metrics endpoint
///
/// Returns metrics in Prometheus text exposition format for scraping.
/// Includes:
/// - HTTP request counts and durations by endpoint
/// - Active connection counts (HTTP and WebSocket)
/// - Cache statistics
/// - Verification success/failure counts
/// - Server uptime
pub async fn prometheus_metrics(State(state): State<Arc<AppState>>) -> Response {
    let cache_stats = state.proof_cache.read().await.stats();
    let active_requests = state.active_requests();
    let active_websockets = state.session_manager.active_count().await;

    let output = state
        .metrics
        .export_prometheus(
            active_requests,
            active_websockets,
            cache_stats.total_entries,
            cache_stats.valid_entries,
            cache_stats.expired_entries,
        )
        .await;

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        .body(output.into())
        .unwrap()
}

/// GET /backends - List available verification backends with health status
///
/// Returns a list of all supported backends with their current health status.
/// This allows clients to check which verification tools are available.
pub async fn list_backends(State(_state): State<Arc<AppState>>) -> Json<BackendsResponse> {
    let mut backends = Vec::new();

    // Check each backend's health
    backends.push(backend_info::<Lean4Backend>(BackendIdParam::Lean4, "LEAN 4").await);
    backends.push(backend_info::<TlaPlusBackend>(BackendIdParam::TlaPlus, "TLA+").await);
    backends.push(backend_info::<KaniBackend>(BackendIdParam::Kani, "Kani").await);
    backends.push(backend_info::<AlloyBackend>(BackendIdParam::Alloy, "Alloy").await);
    backends.push(backend_info::<IsabelleBackend>(BackendIdParam::Isabelle, "Isabelle/HOL").await);
    backends.push(backend_info::<CoqBackend>(BackendIdParam::Coq, "Coq").await);
    backends.push(backend_info::<DafnyBackend>(BackendIdParam::Dafny, "Dafny").await);

    Json(BackendsResponse { backends })
}

/// Build backend info entry from backend implementation
async fn backend_info<B>(id: BackendIdParam, name: &str) -> BackendInfo
where
    B: VerificationBackend + Default,
{
    let backend = B::default();
    let supports = backend
        .supports()
        .into_iter()
        .map(super::types::PropertyTypeResponse::from)
        .collect();
    let health = health_to_response(backend.health_check().await);

    BackendInfo {
        id,
        name: name.to_string(),
        supports,
        health,
    }
}

/// Convert HealthStatus to response type
fn health_to_response(health: dashprove_backends::HealthStatus) -> HealthStatusResponse {
    use dashprove_backends::HealthStatus;
    match health {
        HealthStatus::Healthy => HealthStatusResponse::Healthy,
        HealthStatus::Degraded { reason } => HealthStatusResponse::Degraded { reason },
        HealthStatus::Unavailable { reason } => HealthStatusResponse::Unavailable { reason },
    }
}
