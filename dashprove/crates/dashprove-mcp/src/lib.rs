// Crate-level lint configuration for pedantic clippy
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::use_self)]
#![allow(clippy::unused_self)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::similar_names)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::single_match_else)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::significant_drop_tightening)]
#![allow(clippy::let_underscore_must_use)]
#![allow(clippy::or_fun_call)]
#![allow(clippy::unused_async)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::return_self_not_must_use)]

//! DashProve Model Context Protocol (MCP) Server
//!
//! Provides MCP-compliant tools for AI agents to interact with DashProve verification.
//! Implements the Model Context Protocol specification for tool discovery and invocation.
//!
//! ## Features
//!
//! - JSON-RPC 2.0 based protocol
//! - Tool discovery via `tools/list`
//! - Tool invocation via `tools/call`
//! - Multiple transports: stdio, HTTP, and WebSocket
//!
//! ## HTTP Endpoints (when using HTTP transport)
//!
//! - `POST /jsonrpc` - JSON-RPC 2.0 endpoint
//! - `POST /verify` - Direct single-spec verification (accepts VerifyUslArgs, returns VerifyUslResult)
//! - `POST /verify/stream` - Start streaming verification (accepts StreamingVerifyArgs, returns StreamingVerifyStartResult)
//! - `POST /batch` - Direct batch verification (accepts BatchVerifyArgs, returns BatchVerifyResult)
//! - `GET /events/:session_id` - SSE stream for verification progress
//! - `GET /sessions/:session_id` - Session status (polling alternative to SSE)
//! - `DELETE /sessions/:session_id` - Cancel a running session
//! - `POST /sessions/:session_id/cancel` - Cancel a running session (alternative)
//! - `GET /health` - Health check
//! - `GET /` - Server info
//! - `GET /metrics` - Prometheus-format metrics (requires --metrics flag)
//! - `GET /metrics/json` - JSON-format metrics snapshot (requires --metrics flag)
//!
//! ## Available Tools
//!
//! - `dashprove.verify_usl` - Verify a USL specification
//! - `dashprove.verify_usl_streaming` - Start streaming verification with real-time progress
//! - `dashprove.batch_verify` - Verify multiple USL specifications in batch
//! - `dashprove.select_backend` - Get recommended backends for a property type
//! - `dashprove.compile_to` - Compile USL to a specific backend
//! - `dashprove.get_suggestions` - Get proof suggestions for failed verification
//! - `dashprove.check_dependencies` - Check backend dependencies and availability
//! - `get_session_status` - Query status of a streaming verification session
//! - `cancel_session` - Cancel a running streaming verification session
//!
//! ## Supported Backends
//!
//! - **lean4** - LEAN 4 theorem prover
//! - **tlaplus** - TLA+ model checker (TLC)
//! - **kani** - Kani Rust model checker
//! - **coq** - Coq proof assistant
//! - **alloy** - Alloy relational analyzer
//! - **isabelle** - Isabelle/HOL theorem prover
//! - **dafny** - Dafny verification language
//!
//! ## Example
//!
//! ```ignore
//! use dashprove_mcp::McpServer;
//!
//! let server = McpServer::new();
//! server.run_stdio().await?;
//! ```

pub mod cache;
pub mod error;
pub mod logging;
pub mod metrics;
pub mod protocol;
pub mod ratelimit;
pub mod server;
pub mod streaming;
pub mod tools;
pub mod transport;

pub use cache::{
    CacheAutoSaver, CacheConfig, CacheKey, CacheStats, CachedResult, SharedVerificationCache,
    VerificationCache,
};
pub use error::McpError;
pub use logging::{
    LogConfig, LogFilter, LogStats, RequestLogEntry, RequestLogger, SharedRequestLogger,
};
pub use metrics::{
    McpMetrics, MetricsConfig, MetricsSnapshot, RateSnapshot, RateWindows, RatesSnapshot,
    RollingWindow, RollingWindowConfig, SharedMcpMetrics, SharedRollingWindow,
};
pub use protocol::{JsonRpcRequest, JsonRpcResponse, McpMessage};
pub use ratelimit::{RateLimitConfig, RateLimitStats, RateLimiter, SharedRateLimiter};
pub use server::McpServer;
pub use streaming::{SessionManager, VerificationEvent};
pub use tools::{
    Tool, ToolCall, ToolDefinition, ToolResult, VerifyUslResult, VerifyUslStreamingTool,
};
pub use transport::{
    ConnectionMetrics, HttpTransport, MultiplexedSession, MultiplexedSessionInfo,
    RecoverableSession, SessionRecoveryConfig, SessionRecoveryToken, SessionTimeoutConfig,
    Transport, WebSocketTransport, WsClientMessage, WsMultiplexManager, WsServerMessage,
};

#[cfg(test)]
mod tests;
