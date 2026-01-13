use serde::{Deserialize, Serialize};

// ============ Version Types ============

/// API version and metadata response
#[derive(Debug, Serialize, Deserialize)]
pub struct VersionResponse {
    /// Package name
    pub name: String,
    /// Package version
    pub version: String,
    /// API version for compatibility checking
    pub api_version: String,
    /// Rust version used to build
    pub rust_version: String,
    /// Build target
    pub target: String,
}

impl VersionResponse {
    /// Create version response with current build info
    pub fn current() -> Self {
        VersionResponse {
            name: env!("CARGO_PKG_NAME").to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            api_version: "v1".to_string(),
            rust_version: option_env!("CARGO_PKG_RUST_VERSION")
                .unwrap_or("unknown")
                .to_string(),
            target: std::env::consts::ARCH.to_string(),
        }
    }
}

// ============ Health Types ============

/// Health status response with detailed server state
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Overall health status: "healthy", "draining", or "unhealthy"
    pub status: String,
    /// Current shutdown state of the server
    pub shutdown_state: String,
    /// Number of currently in-flight HTTP requests
    pub in_flight_requests: usize,
    /// Number of active WebSocket sessions
    pub active_websocket_sessions: usize,
    /// Whether the server is ready to accept new requests
    pub ready: bool,
}
