//! WebSocket message types

use crate::routes::{BackendIdParam, CompilationResult};
use serde::{Deserialize, Serialize};

/// Optional query parameters for WebSocket session management
#[derive(Debug, Deserialize)]
pub struct WsSessionQuery {
    /// Existing session to resume (if any)
    pub session_id: Option<String>,
    /// Client-provided correlation ID for distributed tracing
    /// (equivalent to X-Request-ID header in HTTP)
    pub correlation_id: Option<String>,
}

/// WebSocket verification request
#[derive(Debug, Serialize, Deserialize)]
pub struct WsVerifyRequest {
    /// USL specification source code
    pub spec: String,
    /// Optional: specific backend to use
    pub backend: Option<BackendIdParam>,
    /// Request ID for correlation (optional, will be generated if not provided)
    pub request_id: Option<String>,
}

/// WebSocket message types sent to clients
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsMessage {
    /// Connection established, session ID assigned
    Connected {
        /// Unique session identifier
        session_id: String,
        /// True when reusing an existing session_id after reconnect
        resumed: bool,
        /// Client-provided correlation ID (if any) for distributed tracing
        #[serde(skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
    /// Acknowledgement that request was received
    Accepted {
        /// Request identifier
        request_id: String,
    },
    /// Progress update during verification
    Progress {
        /// Request identifier
        request_id: String,
        /// Current verification phase
        phase: VerificationPhase,
        /// Human-readable progress message
        message: String,
        /// Percentage complete (0-100), if determinable
        percentage: Option<u8>,
    },
    /// Backend compilation started
    BackendStarted {
        /// Request identifier
        request_id: String,
        /// Which backend started
        backend: BackendIdParam,
    },
    /// Backend compilation completed
    BackendCompleted {
        /// Request identifier
        request_id: String,
        /// Which backend completed
        backend: BackendIdParam,
        /// Compilation result
        result: CompilationResult,
    },
    /// Verification completed successfully
    Completed {
        /// Request identifier
        request_id: String,
        /// Whether parsing and type-checking succeeded
        valid: bool,
        /// Number of properties found
        property_count: usize,
        /// Compilation outputs per backend
        compilations: Vec<CompilationResult>,
        /// Errors (if any)
        errors: Vec<String>,
    },
    /// Error occurred
    Error {
        /// Request identifier (if known)
        request_id: Option<String>,
        /// Error message
        error: String,
        /// Additional details (optional)
        details: Option<String>,
    },
}

/// Phases of verification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationPhase {
    /// Parsing the specification
    Parsing,
    /// Type-checking the specification
    TypeChecking,
    /// Compiling to backends
    Compiling,
    /// Running verification
    Verifying,
    /// Merging results
    Merging,
}

/// Internal progress event for channel communication
#[derive(Debug)]
pub enum ProgressEvent {
    Phase(VerificationPhase, String, Option<u8>),
    BackendStarted(BackendIdParam),
    BackendCompleted(BackendIdParam, CompilationResult),
    Completed {
        valid: bool,
        property_count: usize,
        compilations: Vec<CompilationResult>,
        errors: Vec<String>,
    },
    Error(String, Option<String>),
}
