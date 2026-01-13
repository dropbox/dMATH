//! WebSocket session management

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tracing::info;
use uuid::Uuid;

/// Session manager for tracking active WebSocket connections
#[derive(Debug, Default)]
pub struct SessionManager {
    sessions: RwLock<HashMap<String, SessionInfo>>,
}

/// Information about an active WebSocket session
#[derive(Debug, Clone)]
pub struct SessionInfo {
    /// Session ID
    pub session_id: String,
    /// When the session was created
    pub created_at: Instant,
    /// Number of verification requests made
    pub request_count: u32,
    /// Whether the session is currently connected
    pub connected: bool,
}

impl SessionManager {
    /// Create a new session manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new session and return its ID
    pub async fn create_session(&self) -> String {
        let session_id = Uuid::new_v4().to_string();
        let mut sessions = self.sessions.write().await;
        sessions.insert(
            session_id.clone(),
            SessionInfo {
                session_id: session_id.clone(),
                created_at: Instant::now(),
                request_count: 0,
                connected: true,
            },
        );
        info!(session_id = %session_id, "New WebSocket session created");
        session_id
    }

    /// Mark a session as disconnected
    pub async fn disconnect_session(&self, session_id: &str) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.connected = false;
            info!(session_id = %session_id, "WebSocket session disconnected");
        }
    }

    /// Increment the request count for a session
    pub async fn increment_request_count(&self, session_id: &str) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.request_count += 1;
        }
    }

    /// Get session info
    pub async fn get_session(&self, session_id: &str) -> Option<SessionInfo> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).cloned()
    }

    /// Reconnect to an existing session by ID, marking it connected again
    pub async fn reconnect_session(&self, session_id: &str) -> Option<SessionInfo> {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.connected = true;
            info!(session_id = %session_id, "WebSocket session reconnected");
            Some(session.clone())
        } else {
            None
        }
    }

    /// List all active (connected) sessions
    pub async fn list_active_sessions(&self) -> Vec<SessionInfo> {
        let sessions = self.sessions.read().await;
        sessions.values().filter(|s| s.connected).cloned().collect()
    }

    /// Get count of active sessions
    pub async fn active_count(&self) -> usize {
        let sessions = self.sessions.read().await;
        sessions.values().filter(|s| s.connected).count()
    }

    /// Clean up old disconnected sessions (older than duration)
    pub async fn cleanup(&self, max_age: Duration) {
        let mut sessions = self.sessions.write().await;
        let now = Instant::now();
        sessions.retain(|_, session| {
            session.connected || now.duration_since(session.created_at) < max_age
        });
    }
}
