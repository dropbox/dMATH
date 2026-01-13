//! Admin endpoints for WebSocket session management

use super::session::SessionInfo;
use crate::routes::AppState;
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Response for listing active WebSocket sessions
#[derive(Debug, Serialize, Deserialize)]
pub struct ListSessionsResponse {
    /// Active (connected) sessions
    pub sessions: Vec<SessionInfoResponse>,
    /// Total number of active sessions
    pub active_count: usize,
}

/// Session info in API response (serializable version of SessionInfo)
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionInfoResponse {
    /// Session ID (UUID)
    pub session_id: String,
    /// Seconds since session was created
    pub uptime_seconds: u64,
    /// Number of verification requests made in this session
    pub request_count: u32,
    /// Whether the session is currently connected
    pub connected: bool,
}

impl SessionInfoResponse {
    /// Convert from internal SessionInfo to API response
    pub fn from_session_info(info: &SessionInfo) -> Self {
        Self {
            session_id: info.session_id.clone(),
            uptime_seconds: info.created_at.elapsed().as_secs(),
            request_count: info.request_count,
            connected: info.connected,
        }
    }
}

/// GET /admin/sessions - List active WebSocket sessions
///
/// Returns information about all currently connected WebSocket sessions,
/// including session IDs, uptime, and request counts.
pub async fn list_sessions(State(state): State<Arc<AppState>>) -> Json<ListSessionsResponse> {
    let sessions = state.session_manager.list_active_sessions().await;
    let active_count = sessions.len();

    let sessions: Vec<SessionInfoResponse> = sessions
        .iter()
        .map(SessionInfoResponse::from_session_info)
        .collect();

    Json(ListSessionsResponse {
        sessions,
        active_count,
    })
}
