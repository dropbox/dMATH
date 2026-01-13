//! WebSocket streaming for long-running verifications
//!
//! Provides real-time progress updates during verification via WebSocket.
//! Clients can connect to /ws/verify and send verification requests,
//! receiving progress events as verification proceeds.
//!
//! Features:
//! - Session IDs for connection tracking
//! - Multiple concurrent verification requests per session
//! - Graceful handling of disconnections
//! - Admin endpoint to monitor active sessions
//! - Correlation ID support for distributed tracing (via `correlation_id` query param)

mod admin;
mod handler;
mod messages;
mod session;

pub use admin::{list_sessions, ListSessionsResponse, SessionInfoResponse};
pub use handler::ws_verify_handler;
pub use messages::{VerificationPhase, WsMessage, WsSessionQuery, WsVerifyRequest};
pub use session::{SessionInfo, SessionManager};

#[cfg(test)]
mod tests;
