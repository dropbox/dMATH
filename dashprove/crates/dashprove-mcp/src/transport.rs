//! MCP transport implementations
//!
//! Provides stdio, HTTP, and WebSocket transport layers for the MCP server.
//!
//! ## Authentication
//!
//! HTTP and WebSocket transports support optional token-based authentication.
//! When an API token is configured:
//! - HTTP requests must include `Authorization: Bearer <token>` header
//! - WebSocket connections can authenticate via query param `?token=<token>`
//! - Health and info endpoints remain unauthenticated

use std::convert::Infallible;
use std::io::{BufRead, Write};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::body::to_bytes;
use axum::body::HttpBody;
use axum::{
    body::Body,
    extract::{ConnectInfo, Path, State},
    http::{header, Request, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{delete, get, post},
    Json, Router,
};
use futures::stream;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use tower_http::cors::{Any, CorsLayer};
use tracing::{debug, error, info, warn};

use crate::cache::CacheConfig;
use crate::error::McpError;
use crate::logging::{
    LogConfig, LogFilter, RequestLogEntry, RequestLogger, RequestStatus, RequestType,
    SharedRequestLogger,
};
use crate::metrics::{McpMetrics, MetricsConfig, SharedMcpMetrics};
use crate::protocol::{JsonRpcRequest, JsonRpcResponse, RequestId};
use crate::ratelimit::{RateLimitConfig, RateLimiter, SharedRateLimiter};
use crate::server::McpServer;
use crate::streaming::{
    SessionManager, StreamingVerifyArgs, StreamingVerifyStartResult, VerificationEvent,
};
use crate::tools::{BatchVerifyArgs, BatchVerifyResult, VerifyUslArgs, VerifyUslResult};

/// Request body for cache configuration updates
#[derive(Debug, Clone, serde::Deserialize)]
pub struct CacheConfigRequest {
    /// Time-to-live for cache entries in seconds (optional)
    #[serde(default)]
    pub ttl_secs: Option<u64>,
    /// Maximum number of entries (optional)
    #[serde(default)]
    pub max_entries: Option<usize>,
    /// Whether caching is enabled (optional)
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Whether to clear cache when configuration changes significantly
    #[serde(default)]
    pub clear_on_change: Option<bool>,
}

/// Transport trait for MCP communication
pub trait Transport {
    /// Run the transport loop
    fn run(&mut self, server: &mut McpServer) -> Result<(), McpError>;
}

/// Stdio transport for MCP
///
/// Reads JSON-RPC requests from stdin (one per line) and writes responses to stdout.
pub struct StdioTransport;

impl StdioTransport {
    /// Create a new stdio transport
    pub fn new() -> Self {
        Self
    }

    /// Run the stdio transport asynchronously
    pub async fn run_async(&self, server: &mut McpServer) -> Result<(), McpError> {
        info!("Starting MCP server on stdio transport");

        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line).await?;

            if bytes_read == 0 {
                // EOF
                info!("Stdin closed, shutting down");
                break;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            debug!("Received: {}", trimmed);

            match server.process_line(trimmed).await {
                Ok(response) => {
                    debug!("Sending: {}", response);
                    stdout.write_all(response.as_bytes()).await?;
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                }
                Err(e) => {
                    error!("Error processing request: {}", e);
                    let error_response = McpServer::parse_error_response(None, &e.to_string());
                    let response = serde_json::to_string(&error_response)
                        .map_err(|e| McpError::InternalError(e.to_string()))?;
                    stdout.write_all(response.as_bytes()).await?;
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                }
            }
        }

        Ok(())
    }

    /// Run synchronously (blocking)
    pub fn run_sync(&self, server: &mut McpServer) -> Result<(), McpError> {
        info!("Starting MCP server on stdio transport (sync)");

        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();
        let handle = stdin.lock();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| McpError::InternalError(e.to_string()))?;

        for line in handle.lines() {
            let line = line?;
            let trimmed = line.trim();

            if trimmed.is_empty() {
                continue;
            }

            debug!("Received: {}", trimmed);

            let response = rt.block_on(async { server.process_line(trimmed).await });

            match response {
                Ok(response) => {
                    debug!("Sending: {}", response);
                    writeln!(stdout, "{}", response)?;
                    stdout.flush()?;
                }
                Err(e) => {
                    error!("Error processing request: {}", e);
                    let error_response = McpServer::parse_error_response(None, &e.to_string());
                    let response = serde_json::to_string(&error_response)
                        .map_err(|e| McpError::InternalError(e.to_string()))?;
                    writeln!(stdout, "{}", response)?;
                    stdout.flush()?;
                }
            }
        }

        Ok(())
    }
}

impl Default for StdioTransport {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared state for HTTP transport
#[derive(Clone)]
pub struct HttpServerState {
    /// MCP server instance
    pub server: Arc<Mutex<McpServer>>,
    /// Session manager for streaming verification
    pub session_manager: Arc<SessionManager>,
    /// Optional API token for authentication
    pub api_token: Option<String>,
    /// Optional rate limiter
    pub rate_limiter: Option<SharedRateLimiter>,
    /// Optional request logger
    pub request_logger: Option<SharedRequestLogger>,
    /// Optional metrics collector
    pub metrics: Option<SharedMcpMetrics>,
}

/// Rate limiting middleware for HTTP requests
///
/// Applies per-client rate limiting based on IP address.
/// Requests exceeding the limit are rejected with 429 Too Many Requests.
pub async fn rate_limit_middleware_fn(
    State(state): State<HttpServerState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // If no rate limiter is configured, allow all requests
    let Some(limiter) = &state.rate_limiter else {
        return next.run(request).await;
    };

    // Check if rate limiting is enabled
    let config = limiter.config().await;
    if !config.enabled {
        return next.run(request).await;
    }

    // Extract client IP from request
    let ip = extract_client_ip(&request);

    // Check rate limit
    if limiter.check(ip).await {
        // Add rate limit headers
        let mut response = next.run(request).await;
        if let Some(remaining) = limiter.remaining_tokens(ip).await {
            if let Ok(val) = remaining.to_string().parse() {
                response.headers_mut().insert("X-RateLimit-Remaining", val);
            }
        }
        if let Ok(val) = format!("{}/s", config.requests_per_second).parse() {
            response.headers_mut().insert("X-RateLimit-Limit", val);
        }
        response
    } else {
        // Return 429 Too Many Requests
        warn!("Rate limit exceeded for client IP: {}", ip);

        let retry_after = (1.0 / config.requests_per_second).ceil() as u64;

        (
            StatusCode::TOO_MANY_REQUESTS,
            [
                ("Retry-After", retry_after.to_string()),
                (
                    "X-RateLimit-Limit",
                    format!("{}/s", config.requests_per_second),
                ),
                ("X-RateLimit-Remaining", "0".to_string()),
            ],
            Json(serde_json::json!({
                "error": "Too Many Requests",
                "message": format!(
                    "Rate limit exceeded. Maximum {} requests per second.",
                    config.requests_per_second
                ),
                "retry_after_seconds": retry_after
            })),
        )
            .into_response()
    }
}

/// Extract client IP from request
///
/// Checks in order:
/// 1. X-Forwarded-For header (first IP)
/// 2. X-Real-IP header
/// 3. Connected socket address
/// 4. Fallback to 0.0.0.0
fn extract_client_ip(request: &Request<Body>) -> std::net::IpAddr {
    // Try X-Forwarded-For header
    if let Some(xff) = request.headers().get("X-Forwarded-For") {
        if let Ok(xff_str) = xff.to_str() {
            if let Some(first_ip) = xff_str.split(',').next() {
                if let Ok(ip) = first_ip.trim().parse() {
                    return ip;
                }
            }
        }
    }

    // Try X-Real-IP header
    if let Some(real_ip) = request.headers().get("X-Real-IP") {
        if let Ok(ip_str) = real_ip.to_str() {
            if let Ok(ip) = ip_str.trim().parse() {
                return ip;
            }
        }
    }

    // Try to get from extensions (ConnectInfo)
    if let Some(connect_info) = request
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()
    {
        return connect_info.0.ip();
    }

    // Fallback
    "0.0.0.0".parse().unwrap()
}

const TRUNCATION_SUFFIX: &str = "...[truncated]";

/// Format a request/response body for logging, truncating if needed.
fn format_body_for_logging(bytes: &[u8], max_len: usize) -> String {
    if max_len == 0 {
        return String::new();
    }

    let body = String::from_utf8_lossy(bytes);
    if body.len() <= max_len {
        return body.into_owned();
    }

    let keep = max_len.saturating_sub(TRUNCATION_SUFFIX.len());
    let mut truncated: String = body.chars().take(keep).collect();
    truncated.push_str(TRUNCATION_SUFFIX);
    truncated
}

/// Determine if it is safe to buffer the response body for logging.
fn should_log_response_body(response: &Response, config: &LogConfig) -> bool {
    if !config.log_response_body {
        return false;
    }

    if let Some(content_type) = response.headers().get(header::CONTENT_TYPE) {
        if let Ok(ct) = content_type.to_str() {
            if ct.starts_with("text/event-stream") {
                // Avoid buffering streaming SSE responses
                return false;
            }
        }
    }

    let max_buffer = config
        .max_body_size
        .saturating_mul(8)
        .max(config.max_body_size);

    match response.body().size_hint().upper() {
        Some(upper) => upper <= max_buffer as u64,
        None => true, // Unknown size but not marked streaming
    }
}

/// Authentication middleware for HTTP requests
///
/// Validates the Authorization header against the configured API token.
/// Requests without valid auth are rejected with 401 Unauthorized.
pub async fn auth_middleware(
    State(state): State<HttpServerState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // If no token is configured, allow all requests
    let Some(expected_token) = &state.api_token else {
        return next.run(request).await;
    };

    // Check Authorization header
    if let Some(auth_header) = request.headers().get(header::AUTHORIZATION) {
        if let Ok(auth_str) = auth_header.to_str() {
            // Support "Bearer <token>" format or raw token
            let token = auth_str.strip_prefix("Bearer ").unwrap_or(auth_str);

            if token == expected_token {
                return next.run(request).await;
            }
        }
    }

    // Check query parameter as fallback (useful for WebSocket upgrade)
    if let Some(query) = request.uri().query() {
        for pair in query.split('&') {
            if let Some(token) = pair.strip_prefix("token=") {
                if token == expected_token {
                    return next.run(request).await;
                }
            }
        }
    }

    debug!(
        "Authentication failed for request to {}",
        request.uri().path()
    );
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({
            "error": "Unauthorized",
            "message": "Valid API token required. Provide via Authorization header (Bearer token) or ?token= query parameter."
        })),
    )
        .into_response()
}

/// Request logging middleware for HTTP requests
///
/// Logs all HTTP requests with timing, status codes, and client IPs.
/// Runs as the outermost middleware to capture the full request lifecycle.
pub async fn request_logging_middleware(
    State(state): State<HttpServerState>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    // If no logger is configured, skip logging
    let Some(logger) = &state.request_logger else {
        return next.run(request).await;
    };

    // Get logging config
    let config = logger.config().await;
    if !config.enabled {
        return next.run(request).await;
    }

    // Extract request info before passing to next handler
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let client_ip = extract_client_ip(&request);

    // Capture request body if enabled
    let mut request_body = None;
    if config.log_request_body {
        let body = std::mem::take(request.body_mut());
        match to_bytes(body, usize::MAX).await {
            Ok(bytes) => {
                request_body = Some(format_body_for_logging(&bytes, config.max_body_size));
                *request.body_mut() = Body::from(bytes);
            }
            Err(err) => {
                warn!("Failed to read request body for logging: {}", err);
            }
        }
    }

    let start = std::time::Instant::now();

    // Execute the request
    let mut response = next.run(request).await;

    // Log the request
    let status_code = response.status().as_u16();
    let duration = start.elapsed();

    let mut response_body = None;
    if should_log_response_body(&response, &config) {
        let body = std::mem::take(response.body_mut());
        match to_bytes(body, usize::MAX).await {
            Ok(bytes) => {
                response_body = Some(format_body_for_logging(&bytes, config.max_body_size));
                *response.body_mut() = Body::from(bytes);
            }
            Err(err) => {
                warn!("Failed to read response body for logging: {}", err);
            }
        }
    }

    let mut entry = RequestLogEntry::new(RequestType::Http, &method, &path)
        .with_client_ip(client_ip)
        .with_status_code(status_code)
        .with_duration(duration);

    if let Some(body) = request_body {
        entry.request_body = Some(body);
    }
    if let Some(body) = response_body {
        entry.response_body = Some(body);
    }

    // Determine request status based on HTTP status code
    entry.status = if status_code >= 500 {
        RequestStatus::Error
    } else if status_code == 401 || status_code == 403 || status_code == 429 {
        RequestStatus::Rejected
    } else if status_code >= 400 {
        RequestStatus::Error
    } else {
        RequestStatus::Success
    };

    logger.log(entry).await;

    response
}

/// HTTP transport for MCP
///
/// Provides an HTTP server endpoint for JSON-RPC requests.
/// Exposes POST /jsonrpc for JSON-RPC 2.0 requests and GET /health for health checks.
/// Supports optional token-based authentication, rate limiting, request logging, and metrics.
pub struct HttpTransport {
    /// Address to bind to
    pub bind_addr: String,
    /// Optional API token for authentication
    pub api_token: Option<String>,
    /// Optional rate limit configuration
    pub rate_limit_config: Option<RateLimitConfig>,
    /// Optional request logging configuration
    pub log_config: Option<LogConfig>,
    /// Optional metrics configuration
    pub metrics_config: Option<MetricsConfig>,
}

impl HttpTransport {
    /// Create a new HTTP transport
    pub fn new(bind_addr: impl Into<String>) -> Self {
        Self {
            bind_addr: bind_addr.into(),
            api_token: None,
            rate_limit_config: None,
            log_config: None,
            metrics_config: None,
        }
    }

    /// Create a new HTTP transport with authentication
    pub fn with_auth(bind_addr: impl Into<String>, api_token: Option<String>) -> Self {
        Self {
            bind_addr: bind_addr.into(),
            api_token,
            rate_limit_config: None,
            log_config: None,
            metrics_config: None,
        }
    }

    /// Create a new HTTP transport with rate limiting
    pub fn with_rate_limit(
        bind_addr: impl Into<String>,
        rate_limit_config: RateLimitConfig,
    ) -> Self {
        Self {
            bind_addr: bind_addr.into(),
            api_token: None,
            rate_limit_config: Some(rate_limit_config),
            log_config: None,
            metrics_config: None,
        }
    }

    /// Create a new HTTP transport with authentication and rate limiting
    pub fn with_auth_and_rate_limit(
        bind_addr: impl Into<String>,
        api_token: Option<String>,
        rate_limit_config: Option<RateLimitConfig>,
    ) -> Self {
        Self {
            bind_addr: bind_addr.into(),
            api_token,
            rate_limit_config,
            log_config: None,
            metrics_config: None,
        }
    }

    /// Create a fully configured HTTP transport
    pub fn with_all_options(
        bind_addr: impl Into<String>,
        api_token: Option<String>,
        rate_limit_config: Option<RateLimitConfig>,
        log_config: Option<LogConfig>,
        metrics_config: Option<MetricsConfig>,
    ) -> Self {
        Self {
            bind_addr: bind_addr.into(),
            api_token,
            rate_limit_config,
            log_config,
            metrics_config,
        }
    }

    /// Run the HTTP transport asynchronously
    ///
    /// This starts an axum HTTP server that accepts JSON-RPC requests at POST /jsonrpc
    /// and SSE streams at GET /events/{session_id}
    pub async fn run_async(&self, server: McpServer) -> Result<(), McpError> {
        // Get session manager from server's tool registry
        let session_manager = server.tools_session_manager();

        // Create rate limiter if configured
        let rate_limiter = self.rate_limit_config.clone().map(RateLimiter::new);

        // Start rate limiter cleanup task if rate limiting is enabled
        let _cleanup_handle = rate_limiter
            .as_ref()
            .map(|limiter| RateLimiter::start_cleanup_task(Arc::clone(limiter)));

        // Create request logger if configured
        let request_logger = self.log_config.clone().map(RequestLogger::new);

        // Create metrics collector if configured
        let metrics = self
            .metrics_config
            .clone()
            .filter(|c| c.enabled)
            .map(McpMetrics::new);

        let state = HttpServerState {
            server: Arc::new(Mutex::new(server)),
            session_manager,
            api_token: self.api_token.clone(),
            rate_limiter: rate_limiter.clone(),
            request_logger: request_logger.clone(),
            metrics: metrics.clone(),
        };

        // Configure CORS for browser-based clients
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);

        // Protected routes (require auth and rate limiting if configured)
        let protected_routes = Router::new()
            .route("/jsonrpc", post(handle_jsonrpc))
            .route("/verify", post(handle_verify))
            .route("/verify/stream", post(handle_verify_streaming))
            .route("/batch", post(handle_batch_verify))
            .route("/events/:session_id", get(handle_sse_events))
            .route("/sessions/:session_id", get(handle_session_status))
            .route("/sessions/:session_id/cancel", post(handle_cancel_session))
            .route("/sessions/:session_id", delete(handle_cancel_session))
            .route("/cache/stats", get(handle_cache_stats))
            .route("/cache/clear", post(handle_cache_clear))
            .route("/cache/config", get(handle_cache_config_get))
            .route("/cache/config", post(handle_cache_config_update))
            .route("/cache/save", post(handle_cache_save))
            .route("/cache/load", post(handle_cache_load))
            .route("/ratelimit/stats", get(handle_ratelimit_stats))
            .route("/ratelimit/config", get(handle_ratelimit_config_get))
            .route("/ratelimit/config", post(handle_ratelimit_config_update))
            .route("/logs", get(handle_logs_query))
            .route("/logs/recent", get(handle_logs_recent))
            .route("/logs/stats", get(handle_logs_stats))
            .route("/logs/config", get(handle_logs_config_get))
            .route("/logs/config", post(handle_logs_config_update))
            .route("/logs/clear", post(handle_logs_clear))
            .route("/logs/export", get(handle_logs_export))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                auth_middleware,
            ))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                rate_limit_middleware_fn,
            ))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                request_logging_middleware,
            ));

        // Public routes (no auth required, but still rate limited)
        let public_routes = Router::new()
            .route("/health", get(handle_health))
            .route("/", get(handle_info))
            .route("/metrics", get(handle_metrics))
            .route("/metrics/json", get(handle_metrics_json))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                rate_limit_middleware_fn,
            ))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                request_logging_middleware,
            ));

        let app = Router::new()
            .merge(protected_routes)
            .merge(public_routes)
            .with_state(state)
            .layer(cors);

        let addr: SocketAddr = self
            .bind_addr
            .parse()
            .map_err(|e| McpError::InternalError(format!("Invalid bind address: {}", e)))?;

        // Log startup info
        let auth_enabled = self.api_token.is_some();
        let rate_limit_enabled = self.rate_limit_config.as_ref().is_some_and(|c| c.enabled);
        let logging_enabled = self.log_config.as_ref().is_some_and(|c| c.enabled);
        let metrics_enabled = self.metrics_config.as_ref().is_some_and(|c| c.enabled);

        // Build features string for startup message
        let mut features = Vec::new();
        if auth_enabled {
            features.push("auth");
        }
        if rate_limit_enabled {
            features.push("rate-limit");
        }
        if logging_enabled {
            features.push("logging");
        }
        if metrics_enabled {
            features.push("metrics");
        }

        if features.is_empty() {
            info!("Starting MCP HTTP server on http://{}", addr);
        } else {
            info!(
                "Starting MCP HTTP server on http://{} ({})",
                addr,
                features.join(", ")
            );
        }

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| McpError::InternalError(format!("Failed to bind: {}", e)))?;

        axum::serve(listener, app)
            .await
            .map_err(|e| McpError::InternalError(format!("Server error: {}", e)))?;

        Ok(())
    }
}

impl Default for HttpTransport {
    fn default() -> Self {
        Self::new("127.0.0.1:3001")
    }
}

/// Handle JSON-RPC POST requests
pub async fn handle_jsonrpc(
    State(state): State<HttpServerState>,
    Json(request): Json<JsonRpcRequest>,
) -> impl IntoResponse {
    debug!("HTTP JSON-RPC request: {:?}", request.method);

    let mut server = state.server.lock().await;
    let response = server.handle_request(request).await;

    Json(response)
}

/// Handle single-spec verification POST requests
///
/// This endpoint provides direct HTTP access to USL verification without
/// requiring JSON-RPC wrapping. Accepts VerifyUslArgs and returns VerifyUslResult.
///
/// Example:
/// ```json
/// POST /verify
/// {
///     "spec": "property P1: always(x > 0)",
///     "strategy": "auto",
///     "backends": ["lean4"],
///     "timeout": 30
/// }
/// ```
pub async fn handle_verify(
    State(state): State<HttpServerState>,
    Json(args): Json<VerifyUslArgs>,
) -> impl IntoResponse {
    debug!(
        "HTTP verify request for spec: {}",
        &args.spec[..args.spec.len().min(50)]
    );

    let server = state.server.lock().await;
    match execute_tool_typed::<_, VerifyUslResult>(&server, "dashprove.verify_usl", &args).await {
        Ok(result) => (StatusCode::OK, Json(serde_json::to_value(result).unwrap())).into_response(),
        Err(response) => response,
    }
}

/// Handle streaming verification POST requests
///
/// Starts a streaming verification session and returns a session ID plus events URL.
/// Clients can then subscribe to `/events/{session_id}` for real-time updates or
/// poll `/sessions/{session_id}` to check completion status.
pub async fn handle_verify_streaming(
    State(state): State<HttpServerState>,
    Json(args): Json<StreamingVerifyArgs>,
) -> impl IntoResponse {
    debug!(
        "HTTP streaming verify request for spec: {}",
        &args.spec[..args.spec.len().min(50)]
    );

    let server = state.server.lock().await;
    match execute_tool_typed::<_, StreamingVerifyStartResult>(
        &server,
        "dashprove.verify_usl_streaming",
        &args,
    )
    .await
    {
        Ok(result) => (StatusCode::OK, Json(serde_json::to_value(result).unwrap())).into_response(),
        Err(response) => response,
    }
}

/// Handle batch verification POST requests
///
/// This endpoint provides direct HTTP access to batch verification without
/// requiring JSON-RPC wrapping. Accepts BatchVerifyArgs and returns BatchVerifyResult.
///
/// Example:
/// ```json
/// POST /batch
/// {
///     "specs": [
///         {"spec": "property P1: always(x > 0)"},
///         {"id": "my-spec", "spec": "property P2: eventually(ready)"}
///     ],
///     "timeout": 60,
///     "strategy": "auto"
/// }
/// ```
pub async fn handle_batch_verify(
    State(state): State<HttpServerState>,
    Json(args): Json<BatchVerifyArgs>,
) -> impl IntoResponse {
    debug!("HTTP batch verify request with {} specs", args.specs.len());

    let server = state.server.lock().await;
    match execute_tool_typed::<_, BatchVerifyResult>(&server, "dashprove.batch_verify", &args).await
    {
        Ok(result) => (StatusCode::OK, Json(serde_json::to_value(result).unwrap())).into_response(),
        Err(response) => response,
    }
}

/// Handle SSE events for streaming verification
pub async fn handle_sse_events(
    State(state): State<HttpServerState>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    debug!("SSE subscription request for session: {}", session_id);

    let session = state.session_manager.get_session(&session_id).await;

    match session {
        Some(session) => {
            let receiver = {
                let s = session.lock().await;
                s.subscribe()
            };

            let stream = BroadcastStream::new(receiver).filter_map(|result| match result {
                Ok(event) => {
                    let event_type = match &event {
                        VerificationEvent::Started { .. } => "started",
                        VerificationEvent::Progress { .. } => "progress",
                        VerificationEvent::BackendResult { .. } => "backend_result",
                        VerificationEvent::Completed { .. } => "completed",
                        VerificationEvent::Cancelled { .. } => "cancelled",
                        VerificationEvent::Error { .. } => "error",
                    };

                    match serde_json::to_string(&event) {
                        Ok(data) => Some(Ok::<_, Infallible>(
                            Event::default().event(event_type).data(data),
                        )),
                        Err(e) => {
                            warn!("Failed to serialize event: {}", e);
                            None
                        }
                    }
                }
                Err(e) => {
                    warn!("Broadcast receive error: {}", e);
                    None
                }
            });

            Sse::new(stream)
                .keep_alive(KeepAlive::default().interval(Duration::from_secs(15)))
                .into_response()
        }
        None => {
            // Session not found - return error as SSE
            let error_event = serde_json::json!({
                "type": "error",
                "session_id": session_id,
                "message": "Session not found"
            });

            let stream = stream::once(async move {
                Ok::<_, Infallible>(
                    Event::default()
                        .event("error")
                        .data(error_event.to_string()),
                )
            });

            Sse::new(stream).into_response()
        }
    }
}

/// Session status endpoint for polling-based clients
pub async fn handle_session_status(
    State(state): State<HttpServerState>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    debug!("Session status request for: {}", session_id);

    let status = state.session_manager.get_session_status(&session_id).await;

    if status.exists {
        (StatusCode::OK, Json(serde_json::to_value(status).unwrap())).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "Session not found",
                "session_id": session_id
            })),
        )
            .into_response()
    }
}

/// Cancel session endpoint
///
/// Supports both POST /sessions/:session_id/cancel and DELETE /sessions/:session_id
pub async fn handle_cancel_session(
    State(state): State<HttpServerState>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    debug!("Cancel session request for: {}", session_id);

    let result = state.session_manager.cancel_session(&session_id).await;

    if result.success {
        (StatusCode::OK, Json(serde_json::to_value(result).unwrap())).into_response()
    } else if result.message.contains("not found") {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::to_value(result).unwrap()),
        )
            .into_response()
    } else {
        // Session was already completed or cancelled
        (
            StatusCode::CONFLICT,
            Json(serde_json::to_value(result).unwrap()),
        )
            .into_response()
    }
}

/// Health check endpoint
pub async fn handle_health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "service": "dashprove-mcp",
        "transport": "http"
    }))
}

/// Prometheus-format metrics endpoint
///
/// Returns metrics in Prometheus text exposition format.
/// This endpoint is designed for scraping by Prometheus or compatible systems.
///
/// Returns 503 Service Unavailable if metrics collection is disabled.
pub async fn handle_metrics(State(state): State<HttpServerState>) -> impl IntoResponse {
    let Some(metrics) = &state.metrics else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::CONTENT_TYPE, "text/plain")],
            "Metrics collection is disabled. Start server with --metrics to enable.".to_string(),
        )
            .into_response();
    };

    // Get cache stats for the metrics export
    let server = state.server.lock().await;
    let cache_stats = server.tools_registry().cache_stats().await;
    let cache = server.tools_registry().verification_cache();
    let cache_config = cache.config().await;

    // Calculate valid vs expired entries
    let valid_entries = if cache_config.enabled {
        cache_stats
            .entries
            .saturating_sub(cache_stats.expirations as usize)
    } else {
        0
    };
    let expired_entries = cache_stats.expirations as usize;

    let output = metrics
        .export_prometheus(cache_stats.entries, valid_entries, expired_entries)
        .await;

    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        output,
    )
        .into_response()
}

/// JSON metrics endpoint
///
/// Returns a snapshot of current metrics in JSON format.
/// Useful for dashboards and custom integrations that prefer JSON over Prometheus format.
///
/// Example response:
/// ```json
/// {
///     "uptime_secs": 3600.5,
///     "requests_total": 1234,
///     "verifications_total": 100,
///     "verifications_success": 95,
///     "verifications_failed": 5,
///     "cache_hits": 80,
///     "cache_misses": 20,
///     "rate_limit_checks": 1234,
///     "rate_limit_rejected": 10,
///     "websocket_connections": 5,
///     "active_http": 2,
///     "active_websocket": 1
/// }
/// ```
pub async fn handle_metrics_json(State(state): State<HttpServerState>) -> impl IntoResponse {
    let Some(metrics) = &state.metrics else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "Metrics collection is disabled",
                "hint": "Start server with --metrics to enable metrics collection"
            })),
        )
            .into_response();
    };

    let snapshot = metrics.snapshot().await;
    (StatusCode::OK, Json(serde_json::json!(snapshot))).into_response()
}

/// Cache statistics endpoint
///
/// Returns current cache statistics including hit/miss rates, entry count,
/// and eviction counts.
///
/// Example response:
/// ```json
/// {
///     "hits": 42,
///     "misses": 8,
///     "entries": 15,
///     "evictions": 0,
///     "expirations": 2,
///     "hit_rate": 84.0,
///     "enabled": true
/// }
/// ```
pub async fn handle_cache_stats(State(state): State<HttpServerState>) -> impl IntoResponse {
    let server = state.server.lock().await;
    let stats = server.tools_registry().cache_stats().await;
    let cache = server.tools_registry().verification_cache();
    let config = cache.config().await;

    Json(serde_json::json!({
        "hits": stats.hits,
        "misses": stats.misses,
        "entries": stats.entries,
        "evictions": stats.evictions,
        "expirations": stats.expirations,
        "hit_rate": stats.hit_rate(),
        "enabled": config.enabled,
        "ttl_secs": config.ttl.as_secs(),
        "max_entries": config.max_entries
    }))
}

/// Clear cache endpoint
///
/// Clears all cached verification results. Returns the number of entries
/// that were cleared.
///
/// Example response:
/// ```json
/// {
///     "success": true,
///     "entries_cleared": 15
/// }
/// ```
pub async fn handle_cache_clear(State(state): State<HttpServerState>) -> impl IntoResponse {
    let server = state.server.lock().await;
    let cache = server.tools_registry().verification_cache();

    let entries_before = cache.len().await;
    cache.clear().await;

    Json(serde_json::json!({
        "success": true,
        "entries_cleared": entries_before
    }))
}

/// Get current cache configuration
///
/// Returns the current cache configuration including TTL, max entries, and enabled status.
///
/// Example response:
/// ```json
/// {
///     "ttl_secs": 300,
///     "max_entries": 1000,
///     "enabled": true
/// }
/// ```
pub async fn handle_cache_config_get(State(state): State<HttpServerState>) -> impl IntoResponse {
    let server = state.server.lock().await;
    let cache = server.tools_registry().verification_cache();
    let config = cache.config().await;

    Json(serde_json::json!({
        "ttl_secs": config.ttl.as_secs(),
        "max_entries": config.max_entries,
        "enabled": config.enabled
    }))
}

/// Update cache configuration
///
/// Updates cache configuration at runtime. All fields are optional - only provided
/// fields will be updated. Set `clear_on_change` to true to clear the cache when
/// configuration changes significantly (TTL decreased, cache disabled, or max entries reduced).
///
/// Example request:
/// ```json
/// POST /cache/config
/// {
///     "ttl_secs": 600,
///     "max_entries": 2000,
///     "enabled": true,
///     "clear_on_change": false
/// }
/// ```
///
/// Example response:
/// ```json
/// {
///     "success": true,
///     "old_config": {"ttl_secs": 300, "max_entries": 1000, "enabled": true},
///     "new_config": {"ttl_secs": 600, "max_entries": 2000, "enabled": true}
/// }
/// ```
pub async fn handle_cache_config_update(
    State(state): State<HttpServerState>,
    Json(request): Json<CacheConfigRequest>,
) -> impl IntoResponse {
    let server = state.server.lock().await;
    let cache = server.tools_registry().verification_cache();

    // Get current config
    let old_config = cache.config().await;

    // Build new config from request, using old values as defaults
    let new_config = CacheConfig {
        ttl: request
            .ttl_secs
            .map(std::time::Duration::from_secs)
            .unwrap_or(old_config.ttl),
        max_entries: request.max_entries.unwrap_or(old_config.max_entries),
        enabled: request.enabled.unwrap_or(old_config.enabled),
    };

    let clear_on_change = request.clear_on_change.unwrap_or(false);

    // Update config
    cache
        .update_config(new_config.clone(), clear_on_change)
        .await;

    Json(serde_json::json!({
        "success": true,
        "old_config": {
            "ttl_secs": old_config.ttl.as_secs(),
            "max_entries": old_config.max_entries,
            "enabled": old_config.enabled
        },
        "new_config": {
            "ttl_secs": new_config.ttl.as_secs(),
            "max_entries": new_config.max_entries,
            "enabled": new_config.enabled
        }
    }))
}

/// Request body for cache save endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSaveRequest {
    /// Path to save the cache file to
    pub path: String,
}

/// Save cache to file endpoint
///
/// Saves the verification cache to a file for persistence across server restarts.
/// Creates a JSON snapshot including all non-expired entries, configuration, and statistics.
///
/// Example request:
/// ```json
/// POST /cache/save
/// {
///     "path": "/tmp/dashprove_cache.json"
/// }
/// ```
///
/// Example response:
/// ```json
/// {
///     "success": true,
///     "entries_saved": 15,
///     "path": "/tmp/dashprove_cache.json",
///     "size_bytes": 12345
/// }
/// ```
pub async fn handle_cache_save(
    State(state): State<HttpServerState>,
    Json(request): Json<CacheSaveRequest>,
) -> impl IntoResponse {
    let server = state.server.lock().await;
    let cache = server.tools_registry().verification_cache();

    match cache.save_to_file(&request.path).await {
        Ok(result) => Json(serde_json::json!({
            "success": true,
            "entries_saved": result.entries_saved,
            "path": result.path,
            "size_bytes": result.size_bytes
        })),
        Err(e) => Json(serde_json::json!({
            "success": false,
            "error": e.to_string(),
            "entries_saved": 0,
            "path": request.path,
            "size_bytes": 0
        })),
    }
}

/// Request body for cache load endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLoadRequest {
    /// Path to load the cache file from
    pub path: String,
    /// Whether to merge with existing entries (default: false = replace)
    #[serde(default)]
    pub merge: bool,
}

/// Load cache from file endpoint
///
/// Loads the verification cache from a previously saved file. Entries that have
/// already expired (based on their age at save time plus time since save) are skipped.
/// Configuration is updated from the file.
///
/// Example request:
/// ```json
/// POST /cache/load
/// {
///     "path": "/tmp/dashprove_cache.json",
///     "merge": false
/// }
/// ```
///
/// Example response:
/// ```json
/// {
///     "success": true,
///     "entries_loaded": 12,
///     "entries_expired": 3,
///     "path": "/tmp/dashprove_cache.json",
///     "snapshot_age_secs": 3600
/// }
/// ```
pub async fn handle_cache_load(
    State(state): State<HttpServerState>,
    Json(request): Json<CacheLoadRequest>,
) -> impl IntoResponse {
    let server = state.server.lock().await;
    let cache = server.tools_registry().verification_cache();

    match cache.load_from_file(&request.path, request.merge).await {
        Ok(result) => Json(serde_json::json!({
            "success": true,
            "entries_loaded": result.entries_loaded,
            "entries_expired": result.entries_expired,
            "path": result.path,
            "snapshot_age_secs": result.snapshot_age_secs
        })),
        Err(e) => Json(serde_json::json!({
            "success": false,
            "error": e.to_string(),
            "entries_loaded": 0,
            "entries_expired": 0,
            "path": request.path,
            "snapshot_age_secs": 0
        })),
    }
}

/// Rate limit statistics endpoint
///
/// Returns current rate limiting statistics including allowed/rejected counts
/// and the current number of tracked clients.
///
/// Example response:
/// ```json
/// {
///     "allowed": 1000,
///     "rejected": 50,
///     "rejection_rate": 4.76,
///     "tracked_clients": 10,
///     "enabled": true
/// }
/// ```
pub async fn handle_ratelimit_stats(State(state): State<HttpServerState>) -> impl IntoResponse {
    let Some(limiter) = &state.rate_limiter else {
        return Json(serde_json::json!({
            "enabled": false,
            "message": "Rate limiting not configured"
        }));
    };

    let stats = limiter.stats().await;
    let config = limiter.config().await;

    Json(serde_json::json!({
        "allowed": stats.allowed,
        "rejected": stats.rejected,
        "rejection_rate": stats.rejection_rate(),
        "tracked_clients": stats.tracked_clients,
        "enabled": config.enabled,
        "requests_per_second": config.requests_per_second,
        "burst_size": config.burst_size
    }))
}

/// Get current rate limit configuration
///
/// Returns the current rate limiting configuration.
///
/// Example response:
/// ```json
/// {
///     "requests_per_second": 10.0,
///     "burst_size": 50,
///     "cleanup_interval_secs": 60,
///     "enabled": true
/// }
/// ```
pub async fn handle_ratelimit_config_get(
    State(state): State<HttpServerState>,
) -> impl IntoResponse {
    let Some(limiter) = &state.rate_limiter else {
        return Json(serde_json::json!({
            "enabled": false,
            "message": "Rate limiting not configured"
        }));
    };

    let config = limiter.config().await;

    Json(serde_json::json!({
        "requests_per_second": config.requests_per_second,
        "burst_size": config.burst_size,
        "cleanup_interval_secs": config.cleanup_interval_secs,
        "enabled": config.enabled
    }))
}

/// Request body for rate limit configuration updates
#[derive(Debug, Clone, Deserialize)]
pub struct RateLimitConfigRequest {
    /// Requests per second (optional)
    #[serde(default)]
    pub requests_per_second: Option<f64>,
    /// Burst size (optional)
    #[serde(default)]
    pub burst_size: Option<u32>,
    /// Cleanup interval in seconds (optional)
    #[serde(default)]
    pub cleanup_interval_secs: Option<u64>,
    /// Whether rate limiting is enabled (optional)
    #[serde(default)]
    pub enabled: Option<bool>,
}

/// Update rate limit configuration
///
/// Updates rate limiting configuration at runtime. All fields are optional -
/// only provided fields will be updated.
///
/// Example request:
/// ```json
/// POST /ratelimit/config
/// {
///     "requests_per_second": 20.0,
///     "burst_size": 100,
///     "enabled": true
/// }
/// ```
///
/// Example response:
/// ```json
/// {
///     "success": true,
///     "old_config": {"requests_per_second": 10.0, "burst_size": 50, "enabled": true},
///     "new_config": {"requests_per_second": 20.0, "burst_size": 100, "enabled": true}
/// }
/// ```
pub async fn handle_ratelimit_config_update(
    State(state): State<HttpServerState>,
    Json(request): Json<RateLimitConfigRequest>,
) -> impl IntoResponse {
    let Some(limiter) = &state.rate_limiter else {
        return Json(serde_json::json!({
            "success": false,
            "error": "Rate limiting not configured"
        }));
    };

    let old_config = limiter.config().await;

    let new_config = RateLimitConfig {
        requests_per_second: request
            .requests_per_second
            .unwrap_or(old_config.requests_per_second),
        burst_size: request.burst_size.unwrap_or(old_config.burst_size),
        cleanup_interval_secs: request
            .cleanup_interval_secs
            .unwrap_or(old_config.cleanup_interval_secs),
        enabled: request.enabled.unwrap_or(old_config.enabled),
    };

    limiter.update_config(new_config.clone()).await;

    Json(serde_json::json!({
        "success": true,
        "old_config": {
            "requests_per_second": old_config.requests_per_second,
            "burst_size": old_config.burst_size,
            "cleanup_interval_secs": old_config.cleanup_interval_secs,
            "enabled": old_config.enabled
        },
        "new_config": {
            "requests_per_second": new_config.requests_per_second,
            "burst_size": new_config.burst_size,
            "cleanup_interval_secs": new_config.cleanup_interval_secs,
            "enabled": new_config.enabled
        }
    }))
}

// ============================================================================
// Request Logging Endpoints
// ============================================================================

/// Query log entries with optional filtering
///
/// Query parameters:
/// - `request_type`: Filter by request type (http, websocket, json_rpc)
/// - `status`: Filter by status (success, error, rejected)
/// - `path_prefix`: Filter by path prefix
/// - `client_ip`: Filter by client IP
/// - `since`: Filter entries after this Unix timestamp (ms)
/// - `until`: Filter entries before this Unix timestamp (ms)
/// - `limit`: Maximum number of entries to return
/// - `offset`: Offset for pagination
///
/// Example: GET /logs?status=error&limit=50
pub async fn handle_logs_query(
    State(state): State<HttpServerState>,
    axum::extract::Query(filter): axum::extract::Query<LogFilter>,
) -> impl IntoResponse {
    let Some(logger) = &state.request_logger else {
        return Json(serde_json::json!({
            "enabled": false,
            "message": "Request logging not configured",
            "entries": []
        }));
    };

    let entries = logger.query(&filter).await;
    Json(serde_json::json!({
        "count": entries.len(),
        "entries": entries
    }))
}

/// Get recent log entries (most recent first)
///
/// Query parameters:
/// - `count`: Number of entries to return (default: 50, max: 500)
///
/// Example: GET /logs/recent?count=100
pub async fn handle_logs_recent(
    State(state): State<HttpServerState>,
    axum::extract::Query(params): axum::extract::Query<RecentLogsParams>,
) -> impl IntoResponse {
    let Some(logger) = &state.request_logger else {
        return Json(serde_json::json!({
            "enabled": false,
            "message": "Request logging not configured",
            "entries": []
        }));
    };

    let count = params.count.unwrap_or(50).min(500);
    let entries = logger.recent(count).await;
    Json(serde_json::json!({
        "count": entries.len(),
        "entries": entries
    }))
}

/// Query parameters for recent logs endpoint
#[derive(Debug, Clone, Deserialize)]
pub struct RecentLogsParams {
    /// Number of entries to return (default: 50)
    pub count: Option<usize>,
}

/// Get logging statistics
///
/// Returns statistics about logged requests including total count, status
/// breakdown, and average response time.
///
/// Example response:
/// ```json
/// {
///     "total_logged": 1000,
///     "entries_in_buffer": 500,
///     "buffer_capacity": 1000,
///     "success_count": 900,
///     "error_count": 50,
///     "rejected_count": 50,
///     "avg_duration_ms": 45.5,
///     "enabled": true
/// }
/// ```
pub async fn handle_logs_stats(State(state): State<HttpServerState>) -> impl IntoResponse {
    let Some(logger) = &state.request_logger else {
        return Json(serde_json::json!({
            "enabled": false,
            "message": "Request logging not configured"
        }));
    };

    let stats = logger.stats().await;
    Json(serde_json::to_value(stats).unwrap())
}

/// Get current logging configuration
///
/// Example response:
/// ```json
/// {
///     "max_entries": 1000,
///     "enabled": true,
///     "log_request_body": false,
///     "log_response_body": false,
///     "max_body_size": 4096
/// }
/// ```
pub async fn handle_logs_config_get(State(state): State<HttpServerState>) -> impl IntoResponse {
    let Some(logger) = &state.request_logger else {
        return Json(serde_json::json!({
            "enabled": false,
            "message": "Request logging not configured"
        }));
    };

    let config = logger.config().await;
    Json(serde_json::to_value(config).unwrap())
}

/// Request body for log configuration updates
#[derive(Debug, Clone, Deserialize)]
pub struct LogConfigRequest {
    /// Maximum number of log entries to retain (optional)
    #[serde(default)]
    pub max_entries: Option<usize>,
    /// Whether request logging is enabled (optional)
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Whether to log request bodies (optional)
    #[serde(default)]
    pub log_request_body: Option<bool>,
    /// Whether to log response bodies (optional)
    #[serde(default)]
    pub log_response_body: Option<bool>,
    /// Maximum body size to log in bytes (optional)
    #[serde(default)]
    pub max_body_size: Option<usize>,
}

/// Update logging configuration
///
/// Updates logging configuration at runtime. All fields are optional -
/// only provided fields will be updated.
///
/// Example request:
/// ```json
/// POST /logs/config
/// {
///     "max_entries": 2000,
///     "enabled": true,
///     "log_request_body": true
/// }
/// ```
pub async fn handle_logs_config_update(
    State(state): State<HttpServerState>,
    Json(request): Json<LogConfigRequest>,
) -> impl IntoResponse {
    let Some(logger) = &state.request_logger else {
        return Json(serde_json::json!({
            "success": false,
            "error": "Request logging not configured"
        }));
    };

    let old_config = logger.config().await;

    let new_config = LogConfig {
        max_entries: request.max_entries.unwrap_or(old_config.max_entries),
        enabled: request.enabled.unwrap_or(old_config.enabled),
        log_request_body: request
            .log_request_body
            .unwrap_or(old_config.log_request_body),
        log_response_body: request
            .log_response_body
            .unwrap_or(old_config.log_response_body),
        max_body_size: request.max_body_size.unwrap_or(old_config.max_body_size),
    };

    logger.update_config(new_config.clone()).await;

    Json(serde_json::json!({
        "success": true,
        "old_config": serde_json::to_value(&old_config).unwrap(),
        "new_config": serde_json::to_value(&new_config).unwrap()
    }))
}

/// Clear all log entries
///
/// Clears the log buffer but preserves all-time statistics (total_logged, etc.).
///
/// Example response:
/// ```json
/// {
///     "success": true,
///     "message": "Log entries cleared"
/// }
/// ```
pub async fn handle_logs_clear(State(state): State<HttpServerState>) -> impl IntoResponse {
    let Some(logger) = &state.request_logger else {
        return Json(serde_json::json!({
            "success": false,
            "error": "Request logging not configured"
        }));
    };

    logger.clear().await;
    Json(serde_json::json!({
        "success": true,
        "message": "Log entries cleared"
    }))
}

/// Export all log entries as JSON
///
/// Returns a JSON array of all log entries currently in the buffer.
/// Useful for external analysis or archiving.
pub async fn handle_logs_export(State(state): State<HttpServerState>) -> impl IntoResponse {
    let Some(logger) = &state.request_logger else {
        return (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            "[]".to_string(),
        )
            .into_response();
    };

    match logger.export_json().await {
        Ok(json) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            json,
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::json!({"error": e.to_string()}).to_string(),
        )
            .into_response(),
    }
}

/// Info endpoint at root
pub async fn handle_info() -> impl IntoResponse {
    Json(serde_json::json!({
        "name": "dashprove-mcp",
        "version": env!("CARGO_PKG_VERSION"),
        "protocol": "MCP",
        "transport": "HTTP",
        "endpoints": {
            "jsonrpc": "POST /jsonrpc - JSON-RPC 2.0 endpoint",
            "verify": "POST /verify - Direct single-spec verification endpoint (accepts VerifyUslArgs)",
            "verify_stream": "POST /verify/stream - Start streaming verification (accepts StreamingVerifyArgs)",
            "batch": "POST /batch - Direct batch verification endpoint (accepts BatchVerifyArgs)",
            "events": "GET /events/{session_id} - SSE stream for verification progress",
            "sessions": "GET /sessions/{session_id} - Session status (polling-based alternative to SSE)",
            "cancel_session": "DELETE /sessions/{session_id} or POST /sessions/{session_id}/cancel - Cancel a running session",
            "cache_stats": "GET /cache/stats - Cache statistics (hits, misses, hit rate, config)",
            "cache_clear": "POST /cache/clear - Clear all cached verification results",
            "cache_config": "GET/POST /cache/config - Get or update cache configuration (ttl_secs, max_entries, enabled)",
            "cache_save": "POST /cache/save - Save cache to file for persistence (path required)",
            "cache_load": "POST /cache/load - Load cache from file (path required, merge optional)",
            "ratelimit_stats": "GET /ratelimit/stats - Rate limiting statistics (allowed, rejected, rejection rate)",
            "ratelimit_config": "GET/POST /ratelimit/config - Get or update rate limit configuration",
            "logs": "GET /logs - Query log entries with optional filtering",
            "logs_recent": "GET /logs/recent - Get recent log entries (most recent first)",
            "logs_stats": "GET /logs/stats - Get logging statistics (total, by status, avg duration)",
            "logs_config": "GET/POST /logs/config - Get or update logging configuration",
            "logs_clear": "POST /logs/clear - Clear all log entries",
            "logs_export": "GET /logs/export - Export all log entries as JSON",
            "health": "GET /health - Health check",
            "info": "GET / - This info page"
        }
    }))
}

/// HTTP response type for errors
#[derive(Debug)]
pub struct HttpError {
    /// HTTP status code
    pub status: StatusCode,
    /// Error message
    pub message: String,
}

impl IntoResponse for HttpError {
    fn into_response(self) -> axum::response::Response {
        let body = serde_json::json!({
            "error": self.message
        });
        (self.status, Json(body)).into_response()
    }
}

impl From<McpError> for HttpError {
    fn from(e: McpError) -> Self {
        HttpError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: e.to_string(),
        }
    }
}

/// Helper function to execute a tool and parse its result as a typed response.
///
/// This eliminates duplication across HTTP handlers that all follow the pattern:
/// 1. Serialize args to JSON
/// 2. Execute tool through registry
/// 3. Parse result from Content::Text
/// 4. Return typed response
async fn execute_tool_typed<T, R>(
    server: &tokio::sync::MutexGuard<'_, McpServer>,
    tool_name: &str,
    args: &T,
) -> Result<R, axum::response::Response>
where
    T: serde::Serialize,
    R: serde::de::DeserializeOwned,
{
    let arguments = serde_json::to_value(args).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("Invalid arguments: {}", e)
            })),
        )
            .into_response()
    })?;

    let tool_result = server
        .tools_registry()
        .execute(tool_name, arguments)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response()
        })?;

    if let Some(crate::protocol::Content::Text { text }) = tool_result.content.first() {
        serde_json::from_str::<R>(text).map_err(|e| {
            warn!("Failed to parse tool result: {}", e);
            (StatusCode::OK, Json(serde_json::json!({ "result": text }))).into_response()
        })
    } else {
        Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "No result content" })),
        )
            .into_response())
    }
}

/// JSON-RPC batch request handler
///
/// Handles batch requests per JSON-RPC 2.0 spec
pub async fn handle_jsonrpc_batch(
    state: HttpServerState,
    requests: Vec<JsonRpcRequest>,
) -> Vec<JsonRpcResponse> {
    let mut server = state.server.lock().await;
    let mut responses = Vec::with_capacity(requests.len());

    for request in requests {
        let response = server.handle_request(request).await;
        responses.push(response);
    }

    responses
}

// ============================================================================
// WebSocket Transport
// ============================================================================

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};

/// WebSocket message types for client communication
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsClientMessage {
    /// JSON-RPC request (optionally scoped to a multiplexed session)
    JsonRpc(JsonRpcRequest),
    /// Subscribe to verification events for a session
    Subscribe { session_id: String },
    /// Unsubscribe from verification events
    Unsubscribe { session_id: String },
    /// Ping message for keep-alive
    Ping,
    /// Create a new multiplexed session within this WebSocket connection
    CreateSession {
        /// Client-provided session ID (optional, will be generated if not provided)
        #[serde(default)]
        session_id: Option<String>,
        /// Optional metadata for the session
        #[serde(default)]
        metadata: Option<serde_json::Value>,
    },
    /// Destroy a multiplexed session
    DestroySession {
        /// Session ID to destroy
        session_id: String,
    },
    /// List all multiplexed sessions in this connection
    ListSessions,
    /// Get status of a specific multiplexed session
    GetSessionInfo {
        /// Session ID to query
        session_id: String,
    },
    /// Update metadata for a multiplexed session
    UpdateSessionMetadata {
        /// Session ID to update
        session_id: String,
        /// New metadata (replaces existing metadata)
        metadata: Option<serde_json::Value>,
    },
    /// Touch a session to keep it alive (update last activity timestamp)
    TouchSession {
        /// Session ID to touch
        session_id: String,
    },
    /// Session-scoped JSON-RPC request (routes request to a specific multiplexed session)
    ///
    /// Unlike the regular JsonRpc variant, this tracks the request within a specific
    /// multiplexed session for metrics, session activity tracking, and request isolation.
    SessionScopedJsonRpc {
        /// The multiplexed session ID to route this request to
        session_id: String,
        /// The JSON-RPC request to execute
        request: JsonRpcRequest,
    },
    /// Batch of session-scoped JSON-RPC requests
    ///
    /// Allows sending multiple JSON-RPC requests in a single WebSocket message,
    /// all routed to the same multiplexed session. Reduces round-trip latency
    /// for clients that need to make multiple related requests.
    ///
    /// Requests are processed sequentially in order. The response will contain
    /// results for all requests, preserving order.
    BatchSessionScopedJsonRpc {
        /// The multiplexed session ID to route all requests to
        session_id: String,
        /// The JSON-RPC requests to execute (processed in order)
        requests: Vec<JsonRpcRequest>,
    },
    /// Get connection-level metrics aggregated across all sessions
    GetConnectionMetrics,
    /// Generate a recovery token for session recovery after reconnection
    GenerateRecoveryToken,
    /// Recover sessions from a previously generated recovery token
    RecoverSessions {
        /// The recovery token from a previous connection
        token: SessionRecoveryToken,
    },
}

/// WebSocket message types for server responses
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsServerMessage {
    /// JSON-RPC response
    JsonRpc(JsonRpcResponse),
    /// Verification event from a subscribed session
    Event {
        session_id: String,
        event: VerificationEvent,
    },
    /// Subscription confirmation
    Subscribed { session_id: String },
    /// Unsubscription confirmation
    Unsubscribed { session_id: String },
    /// Subscription error (e.g., session not found)
    SubscriptionError { session_id: String, message: String },
    /// Pong response to ping
    Pong,
    /// Error message
    Error { message: String },
    /// Multiplexed session created confirmation
    SessionCreated {
        /// The session ID (generated or client-provided)
        session_id: String,
        /// Session metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<serde_json::Value>,
    },
    /// Multiplexed session destroyed confirmation
    SessionDestroyed {
        /// The destroyed session ID
        session_id: String,
    },
    /// Session destruction error
    SessionDestroyError {
        /// The session ID that failed to destroy
        session_id: String,
        /// Error message
        message: String,
    },
    /// List of multiplexed sessions
    SessionList {
        /// All active session IDs in this connection
        sessions: Vec<MultiplexedSessionInfo>,
    },
    /// Single session information
    SessionInfo(MultiplexedSessionInfo),
    /// Session info error (session not found)
    SessionInfoError {
        /// Session ID that was queried
        session_id: String,
        /// Error message
        message: String,
    },
    /// Session metadata updated confirmation
    SessionMetadataUpdated {
        /// The session ID
        session_id: String,
        /// The new metadata
        metadata: Option<serde_json::Value>,
    },
    /// Session metadata update error
    SessionMetadataUpdateError {
        /// Session ID that failed to update
        session_id: String,
        /// Error message
        message: String,
    },
    /// Session touched (kept alive)
    SessionTouched {
        /// The session ID
        session_id: String,
        /// The new last_activity timestamp (Unix timestamp ms)
        last_activity: u64,
    },
    /// Session touch error
    SessionTouchError {
        /// Session ID that failed to touch
        session_id: String,
        /// Error message
        message: String,
    },
    /// Session expired notification (sent when session is cleaned up)
    SessionExpired {
        /// The expired session ID
        session_id: String,
        /// Idle duration before expiration (seconds)
        idle_duration_secs: u64,
    },
    /// Session-scoped JSON-RPC response
    ///
    /// Response to a SessionScopedJsonRpc request, including the session ID
    /// for client correlation.
    SessionScopedJsonRpc {
        /// The multiplexed session ID the response belongs to
        session_id: String,
        /// The JSON-RPC response
        response: JsonRpcResponse,
    },
    /// Session-scoped JSON-RPC error (session not found)
    SessionScopedJsonRpcError {
        /// The session ID that was requested
        session_id: String,
        /// The request ID from the original request (for correlation)
        request_id: RequestId,
        /// Error message
        message: String,
    },
    /// Batch session-scoped JSON-RPC response
    ///
    /// Response to a BatchSessionScopedJsonRpc request, containing all
    /// responses in the same order as the original requests.
    BatchSessionScopedJsonRpc {
        /// The multiplexed session ID the responses belong to
        session_id: String,
        /// The JSON-RPC responses (in same order as requests)
        responses: Vec<JsonRpcResponse>,
    },
    /// Batch session-scoped JSON-RPC error (session not found)
    BatchSessionScopedJsonRpcError {
        /// The session ID that was requested
        session_id: String,
        /// Error message
        message: String,
    },
    /// Connection-level metrics response
    ConnectionMetricsResponse {
        /// The unique connection ID
        connection_id: String,
        /// Aggregated metrics across all sessions
        metrics: ConnectionMetrics,
    },
    /// Recovery token generated response
    RecoveryTokenGenerated {
        /// The recovery token to save for reconnection
        token: SessionRecoveryToken,
    },
    /// Recovery token not available (recovery disabled or no sessions)
    RecoveryTokenUnavailable {
        /// Reason why token couldn't be generated
        message: String,
    },
    /// Sessions recovered from token
    SessionsRecovered {
        /// The session IDs that were successfully recovered
        recovered_sessions: Vec<String>,
        /// The connection ID that was recovered from
        original_connection_id: String,
    },
    /// Session recovery error
    SessionRecoveryError {
        /// Error message explaining why recovery failed
        message: String,
    },
}

/// Information about a multiplexed session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiplexedSessionInfo {
    /// Session ID
    pub session_id: String,
    /// Session metadata (client-provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// Number of active subscriptions in this session
    pub active_subscriptions: usize,
    /// When the session was created (Unix timestamp ms)
    pub created_at: u64,
    /// Number of requests processed in this session
    pub request_count: u64,
    /// Last activity timestamp (Unix timestamp ms)
    pub last_activity: u64,
    /// Whether the session has timed out (for informational purposes)
    #[serde(default)]
    pub is_expired: bool,
}

/// Internal state for a multiplexed session within a WebSocket connection
#[derive(Debug)]
pub struct MultiplexedSession {
    /// Session ID
    pub session_id: String,
    /// Session metadata (client-provided)
    pub metadata: Option<serde_json::Value>,
    /// Active subscriptions (verification session ID -> abort handle)
    pub subscriptions: std::collections::HashMap<String, tokio::task::JoinHandle<()>>,
    /// When the session was created
    pub created_at: std::time::Instant,
    /// Number of requests processed
    pub request_count: u64,
    /// Last activity time (updated on each request)
    pub last_activity: std::time::Instant,
}

impl MultiplexedSession {
    /// Create a new multiplexed session
    pub fn new(session_id: String, metadata: Option<serde_json::Value>) -> Self {
        let now = std::time::Instant::now();
        Self {
            session_id,
            metadata,
            subscriptions: std::collections::HashMap::new(),
            created_at: now,
            request_count: 0,
            last_activity: now,
        }
    }

    /// Get session info
    pub fn info(&self) -> MultiplexedSessionInfo {
        let now_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        MultiplexedSessionInfo {
            session_id: self.session_id.clone(),
            metadata: self.metadata.clone(),
            active_subscriptions: self.subscriptions.len(),
            created_at: now_unix.saturating_sub(self.created_at.elapsed().as_millis() as u64),
            request_count: self.request_count,
            last_activity: now_unix.saturating_sub(self.last_activity.elapsed().as_millis() as u64),
            is_expired: false,
        }
    }

    /// Get session info with expiration status
    pub fn info_with_timeout(&self, timeout: Duration) -> MultiplexedSessionInfo {
        let mut info = self.info();
        info.is_expired = self.is_expired(timeout);
        info
    }

    /// Increment request count and update last activity
    pub fn increment_request_count(&mut self) {
        self.request_count += 1;
        self.last_activity = std::time::Instant::now();
    }

    /// Update last activity timestamp
    pub fn touch(&mut self) {
        self.last_activity = std::time::Instant::now();
    }

    /// Update session metadata
    pub fn update_metadata(&mut self, metadata: Option<serde_json::Value>) {
        self.metadata = metadata;
        self.touch();
    }

    /// Check if session has expired based on timeout
    pub fn is_expired(&self, timeout: Duration) -> bool {
        self.last_activity.elapsed() > timeout
    }

    /// Get time since last activity
    pub fn idle_duration(&self) -> Duration {
        self.last_activity.elapsed()
    }

    /// Abort all subscriptions
    pub fn abort_all_subscriptions(&mut self) {
        for (_, handle) in self.subscriptions.drain() {
            handle.abort();
        }
    }
}

/// Configuration for session timeouts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTimeoutConfig {
    /// Session idle timeout (default: 30 minutes)
    /// Sessions with no activity for this duration will be expired
    pub idle_timeout: Duration,
    /// Whether timeout cleanup is enabled
    pub enabled: bool,
    /// Interval between cleanup runs (default: 60 seconds)
    pub cleanup_interval: Duration,
}

/// Connection-level metrics aggregated across all sessions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConnectionMetrics {
    /// Total number of sessions created during this connection's lifetime
    pub sessions_created: u64,
    /// Total number of sessions destroyed during this connection's lifetime
    pub sessions_destroyed: u64,
    /// Total number of sessions expired due to timeout
    pub sessions_expired: u64,
    /// Current number of active sessions
    pub active_sessions: usize,
    /// Total requests processed across all sessions
    pub total_requests: u64,
    /// Connection uptime in milliseconds
    pub uptime_ms: u64,
    /// Average session lifetime in milliseconds (for destroyed/expired sessions)
    pub avg_session_lifetime_ms: Option<u64>,
    /// Total number of subscriptions created
    pub subscriptions_created: u64,
    /// Total number of subscriptions removed
    pub subscriptions_removed: u64,
    /// Current active subscriptions across all sessions
    pub active_subscriptions: usize,
}

/// Recovery token for session persistence across reconnections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecoveryToken {
    /// Unique connection identifier
    pub connection_id: String,
    /// Session states to recover
    pub sessions: Vec<RecoverableSession>,
    /// When the token was created (Unix timestamp ms)
    pub created_at: u64,
    /// Token expiration time (Unix timestamp ms)
    pub expires_at: u64,
}

/// Recoverable session data for reconnection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoverableSession {
    /// Session ID
    pub session_id: String,
    /// Session metadata
    pub metadata: Option<serde_json::Value>,
    /// Request count at time of disconnect
    pub request_count: u64,
}

/// Configuration for session recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecoveryConfig {
    /// Whether session recovery is enabled
    pub enabled: bool,
    /// Recovery token TTL in seconds (default: 5 minutes)
    pub token_ttl_secs: u64,
}

impl Default for SessionRecoveryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            token_ttl_secs: 300, // 5 minutes
        }
    }
}

impl SessionRecoveryConfig {
    /// Create a disabled recovery config
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create a config with custom TTL
    pub fn with_ttl_secs(secs: u64) -> Self {
        Self {
            enabled: true,
            token_ttl_secs: secs,
        }
    }
}

impl Default for SessionTimeoutConfig {
    fn default() -> Self {
        Self {
            idle_timeout: Duration::from_secs(30 * 60), // 30 minutes
            enabled: true,
            cleanup_interval: Duration::from_secs(60), // 1 minute
        }
    }
}

impl SessionTimeoutConfig {
    /// Create a new timeout config with the given idle timeout in seconds
    pub fn with_idle_timeout_secs(secs: u64) -> Self {
        Self {
            idle_timeout: Duration::from_secs(secs),
            ..Default::default()
        }
    }

    /// Disable timeout cleanup
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

/// Internal metrics tracking for WsMultiplexManager
#[derive(Debug, Default)]
struct ConnectionMetricsTracker {
    /// Total sessions created
    sessions_created: u64,
    /// Total sessions destroyed
    sessions_destroyed: u64,
    /// Total sessions expired
    sessions_expired: u64,
    /// Total subscriptions created
    subscriptions_created: u64,
    /// Total subscriptions removed
    subscriptions_removed: u64,
    /// Sum of session lifetimes (ms) for calculating average
    total_session_lifetime_ms: u64,
    /// Number of sessions that have ended (for average calculation)
    ended_sessions: u64,
    /// Connection start time
    connection_start: Option<std::time::Instant>,
    /// Unique connection ID for recovery
    connection_id: String,
}

impl ConnectionMetricsTracker {
    fn new() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        Self {
            connection_start: Some(std::time::Instant::now()),
            connection_id: format!("conn-{:x}-{:04x}", ts, rand_suffix()),
            ..Default::default()
        }
    }
}

/// Manager for multiplexed sessions within a single WebSocket connection
#[derive(Debug)]
pub struct WsMultiplexManager {
    /// All multiplexed sessions in this connection
    sessions: std::collections::HashMap<String, MultiplexedSession>,
    /// Default session for backward compatibility (requests without session_id)
    pub default_session: Option<String>,
    /// Session timeout configuration
    pub timeout_config: SessionTimeoutConfig,
    /// Session recovery configuration
    pub recovery_config: SessionRecoveryConfig,
    /// Connection-level metrics
    metrics: ConnectionMetricsTracker,
}

impl WsMultiplexManager {
    /// Create a new multiplex manager
    pub fn new() -> Self {
        Self {
            sessions: std::collections::HashMap::new(),
            default_session: None,
            timeout_config: SessionTimeoutConfig::default(),
            recovery_config: SessionRecoveryConfig::default(),
            metrics: ConnectionMetricsTracker::new(),
        }
    }

    /// Create a new multiplex manager with custom timeout configuration
    pub fn with_timeout_config(timeout_config: SessionTimeoutConfig) -> Self {
        Self {
            sessions: std::collections::HashMap::new(),
            default_session: None,
            timeout_config,
            recovery_config: SessionRecoveryConfig::default(),
            metrics: ConnectionMetricsTracker::new(),
        }
    }

    /// Create a new multiplex manager with custom timeout and recovery configuration
    pub fn with_configs(
        timeout_config: SessionTimeoutConfig,
        recovery_config: SessionRecoveryConfig,
    ) -> Self {
        Self {
            sessions: std::collections::HashMap::new(),
            default_session: None,
            timeout_config,
            recovery_config,
            metrics: ConnectionMetricsTracker::new(),
        }
    }

    /// Create a new session
    pub fn create_session(
        &mut self,
        session_id: Option<String>,
        metadata: Option<serde_json::Value>,
    ) -> String {
        let session_id = session_id.unwrap_or_else(|| {
            // Generate a unique session ID using timestamp and random suffix
            use std::time::{SystemTime, UNIX_EPOCH};
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            format!("mux-{:x}-{:04x}", ts, rand_suffix())
        });

        // If this is the first session, make it the default
        if self.sessions.is_empty() {
            self.default_session = Some(session_id.clone());
        }

        let session = MultiplexedSession::new(session_id.clone(), metadata);
        self.sessions.insert(session_id.clone(), session);

        // Track metrics
        self.metrics.sessions_created += 1;

        session_id
    }

    /// Destroy a session
    pub fn destroy_session(&mut self, session_id: &str) -> bool {
        if let Some(mut session) = self.sessions.remove(session_id) {
            // Track metrics
            let lifetime_ms = session.created_at.elapsed().as_millis() as u64;
            self.metrics.sessions_destroyed += 1;
            self.metrics.total_session_lifetime_ms += lifetime_ms;
            self.metrics.ended_sessions += 1;
            self.metrics.subscriptions_removed += session.subscriptions.len() as u64;

            session.abort_all_subscriptions();

            // Update default session if we destroyed it
            if self.default_session.as_deref() == Some(session_id) {
                self.default_session = self.sessions.keys().next().cloned();
            }
            true
        } else {
            false
        }
    }

    /// Get a session by ID
    pub fn get_session(&self, session_id: &str) -> Option<&MultiplexedSession> {
        self.sessions.get(session_id)
    }

    /// Get a mutable session by ID
    pub fn get_session_mut(&mut self, session_id: &str) -> Option<&mut MultiplexedSession> {
        self.sessions.get_mut(session_id)
    }

    /// Get the default session (or create one if none exists)
    pub fn get_or_create_default_session(&mut self) -> &mut MultiplexedSession {
        if self.default_session.is_none() {
            let session_id = self.create_session(None, None);
            self.default_session = Some(session_id);
        }
        let default_id = self.default_session.as_ref().unwrap();
        self.sessions.get_mut(default_id).unwrap()
    }

    /// List all sessions
    pub fn list_sessions(&self) -> Vec<MultiplexedSessionInfo> {
        self.sessions.values().map(|s| s.info()).collect()
    }

    /// List all sessions with timeout status
    pub fn list_sessions_with_timeout(&self) -> Vec<MultiplexedSessionInfo> {
        let timeout = self.timeout_config.idle_timeout;
        self.sessions
            .values()
            .map(|s| s.info_with_timeout(timeout))
            .collect()
    }

    /// Check if a session exists
    pub fn has_session(&self, session_id: &str) -> bool {
        self.sessions.contains_key(session_id)
    }

    /// Abort all subscriptions in all sessions
    pub fn abort_all(&mut self) {
        for session in self.sessions.values_mut() {
            session.abort_all_subscriptions();
        }
    }

    /// Get total number of sessions
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Update session metadata
    pub fn update_session_metadata(
        &mut self,
        session_id: &str,
        metadata: Option<serde_json::Value>,
    ) -> bool {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.update_metadata(metadata);
            true
        } else {
            false
        }
    }

    /// Touch a session to update its last activity timestamp
    pub fn touch_session(&mut self, session_id: &str) -> bool {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.touch();
            true
        } else {
            false
        }
    }

    /// Clean up expired sessions based on configured timeout
    ///
    /// Returns a list of session IDs that were expired and removed.
    pub fn cleanup_expired_sessions(&mut self) -> Vec<String> {
        if !self.timeout_config.enabled {
            return Vec::new();
        }

        let timeout = self.timeout_config.idle_timeout;
        let expired_ids: Vec<String> = self
            .sessions
            .iter()
            .filter(|(_, session)| session.is_expired(timeout))
            .map(|(id, _)| id.clone())
            .collect();

        for id in &expired_ids {
            if let Some(mut session) = self.sessions.remove(id) {
                // Track metrics
                let lifetime_ms = session.created_at.elapsed().as_millis() as u64;
                self.metrics.sessions_expired += 1;
                self.metrics.total_session_lifetime_ms += lifetime_ms;
                self.metrics.ended_sessions += 1;
                self.metrics.subscriptions_removed += session.subscriptions.len() as u64;

                session.abort_all_subscriptions();
                debug!(
                    "Expired multiplexed session: {} (idle for {:?})",
                    id,
                    session.idle_duration()
                );
            }

            // Update default session if we removed it
            if self.default_session.as_deref() == Some(id.as_str()) {
                self.default_session = self.sessions.keys().next().cloned();
            }
        }

        expired_ids
    }

    /// Get list of sessions that would be expired (without removing them)
    pub fn get_expired_sessions(&self) -> Vec<String> {
        if !self.timeout_config.enabled {
            return Vec::new();
        }

        let timeout = self.timeout_config.idle_timeout;
        self.sessions
            .iter()
            .filter(|(_, session)| session.is_expired(timeout))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get timeout configuration
    pub fn timeout_config(&self) -> &SessionTimeoutConfig {
        &self.timeout_config
    }

    /// Update timeout configuration
    pub fn set_timeout_config(&mut self, config: SessionTimeoutConfig) {
        self.timeout_config = config;
    }

    /// Get expired sessions with their idle durations (in seconds)
    ///
    /// Returns a list of (session_id, idle_duration_secs) for all expired sessions.
    /// Useful for getting the idle duration before cleanup removes the sessions.
    pub fn get_expired_sessions_with_idle(&self) -> Vec<(String, u64)> {
        if !self.timeout_config.enabled {
            return Vec::new();
        }

        let timeout = self.timeout_config.idle_timeout;
        self.sessions
            .iter()
            .filter(|(_, session)| session.is_expired(timeout))
            .map(|(id, session)| (id.clone(), session.idle_duration().as_secs()))
            .collect()
    }

    /// Track a new subscription creation
    pub fn track_subscription_created(&mut self) {
        self.metrics.subscriptions_created += 1;
    }

    /// Track a subscription removal
    pub fn track_subscription_removed(&mut self) {
        self.metrics.subscriptions_removed += 1;
    }

    /// Get connection-level metrics aggregated across all sessions
    pub fn connection_metrics(&self) -> ConnectionMetrics {
        let uptime_ms = self
            .metrics
            .connection_start
            .map(|start| start.elapsed().as_millis() as u64)
            .unwrap_or(0);

        let total_requests: u64 = self.sessions.values().map(|s| s.request_count).sum();

        let active_subscriptions: usize =
            self.sessions.values().map(|s| s.subscriptions.len()).sum();

        let avg_session_lifetime_ms = if self.metrics.ended_sessions > 0 {
            Some(self.metrics.total_session_lifetime_ms / self.metrics.ended_sessions)
        } else {
            None
        };

        ConnectionMetrics {
            sessions_created: self.metrics.sessions_created,
            sessions_destroyed: self.metrics.sessions_destroyed,
            sessions_expired: self.metrics.sessions_expired,
            active_sessions: self.sessions.len(),
            total_requests,
            uptime_ms,
            avg_session_lifetime_ms,
            subscriptions_created: self.metrics.subscriptions_created,
            subscriptions_removed: self.metrics.subscriptions_removed,
            active_subscriptions,
        }
    }

    /// Get the unique connection ID
    pub fn connection_id(&self) -> &str {
        &self.metrics.connection_id
    }

    /// Generate a recovery token for this connection's sessions
    ///
    /// The token contains all session state needed to recover after reconnection.
    /// Returns None if recovery is disabled or there are no sessions.
    pub fn generate_recovery_token(&self) -> Option<SessionRecoveryToken> {
        if !self.recovery_config.enabled || self.sessions.is_empty() {
            return None;
        }

        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let sessions: Vec<RecoverableSession> = self
            .sessions
            .values()
            .map(|s| RecoverableSession {
                session_id: s.session_id.clone(),
                metadata: s.metadata.clone(),
                request_count: s.request_count,
            })
            .collect();

        Some(SessionRecoveryToken {
            connection_id: self.metrics.connection_id.clone(),
            sessions,
            created_at: now,
            expires_at: now + (self.recovery_config.token_ttl_secs * 1000),
        })
    }

    /// Recover sessions from a recovery token
    ///
    /// This restores sessions that were saved before a disconnect.
    /// Returns the list of recovered session IDs, or an error if the token is invalid.
    pub fn recover_sessions(
        &mut self,
        token: &SessionRecoveryToken,
    ) -> Result<Vec<String>, String> {
        if !self.recovery_config.enabled {
            return Err("Session recovery is disabled".to_string());
        }

        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        if now > token.expires_at {
            return Err("Recovery token has expired".to_string());
        }

        let mut recovered = Vec::new();

        for recoverable in &token.sessions {
            // Skip if session already exists
            if self.sessions.contains_key(&recoverable.session_id) {
                continue;
            }

            let mut session = MultiplexedSession::new(
                recoverable.session_id.clone(),
                recoverable.metadata.clone(),
            );
            session.request_count = recoverable.request_count;

            self.sessions
                .insert(recoverable.session_id.clone(), session);
            self.metrics.sessions_created += 1;
            recovered.push(recoverable.session_id.clone());

            // Set default if none exists
            if self.default_session.is_none() {
                self.default_session = Some(recoverable.session_id.clone());
            }
        }

        Ok(recovered)
    }

    /// Get recovery configuration
    pub fn recovery_config(&self) -> &SessionRecoveryConfig {
        &self.recovery_config
    }

    /// Update recovery configuration
    pub fn set_recovery_config(&mut self, config: SessionRecoveryConfig) {
        self.recovery_config = config;
    }
}

impl Default for WsMultiplexManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a random suffix for session IDs (simple XorShift-based PRNG)
fn rand_suffix() -> u16 {
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};

    thread_local! {
        static STATE: Cell<u64> = Cell::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        );
    }

    STATE.with(|state| {
        let mut x = state.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state.set(x);
        x as u16
    })
}

/// Handle WebSocket upgrade request
///
/// Upgrades the HTTP connection to a WebSocket connection for bidirectional
/// JSON-RPC communication and event streaming.
pub async fn handle_websocket_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<HttpServerState>,
) -> impl IntoResponse {
    info!("WebSocket upgrade request received");
    ws.on_upgrade(|socket| handle_websocket(socket, state))
}

/// Handle an established WebSocket connection
async fn handle_websocket(socket: WebSocket, state: HttpServerState) {
    use futures::{SinkExt, StreamExt as FuturesStreamExt};
    use tokio::sync::mpsc;

    let (mut ws_sender, mut ws_receiver) = socket.split();

    // Channel for sending messages to the WebSocket
    let (tx, mut rx) = mpsc::channel::<WsServerMessage>(100);

    // Multiplexing manager for this connection
    let multiplex_manager = Arc::new(tokio::sync::RwLock::new(WsMultiplexManager::new()));

    // Spawn task to forward messages to WebSocket
    let sender_multiplex_manager = multiplex_manager.clone();
    let sender_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            match serde_json::to_string(&msg) {
                Ok(text) => {
                    if ws_sender.send(Message::Text(text)).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    warn!("Failed to serialize WebSocket message: {}", e);
                }
            }
        }
        // Clean up all subscriptions when connection closes
        let mut manager = sender_multiplex_manager.write().await;
        manager.abort_all();
    });

    // Spawn automatic session cleanup task
    let cleanup_multiplex_manager = multiplex_manager.clone();
    let cleanup_tx = tx.clone();
    let cleanup_task = tokio::spawn(async move {
        // Get initial cleanup interval from manager
        let cleanup_interval = {
            let manager = cleanup_multiplex_manager.read().await;
            manager.timeout_config().cleanup_interval
        };

        let mut interval = tokio::time::interval(cleanup_interval);
        // Skip the first tick (fires immediately)
        interval.tick().await;

        loop {
            interval.tick().await;

            // Get idle durations before cleanup
            let expired_info: Vec<(String, u64)> = {
                let manager = cleanup_multiplex_manager.read().await;
                manager.get_expired_sessions_with_idle()
            };

            if expired_info.is_empty() {
                continue;
            }

            // Perform cleanup
            let expired_ids = {
                let mut manager = cleanup_multiplex_manager.write().await;
                manager.cleanup_expired_sessions()
            };

            // Send SessionExpired notifications for each expired session
            for session_id in expired_ids {
                let idle_secs = expired_info
                    .iter()
                    .find(|(id, _)| id == &session_id)
                    .map(|(_, secs)| *secs)
                    .unwrap_or(0);

                info!(
                    "Session {} expired after {} seconds of inactivity",
                    session_id, idle_secs
                );

                if cleanup_tx
                    .send(WsServerMessage::SessionExpired {
                        session_id,
                        idle_duration_secs: idle_secs,
                    })
                    .await
                    .is_err()
                {
                    // Channel closed, connection closing
                    break;
                }
            }
        }
    });

    // Process incoming messages
    while let Some(result) = FuturesStreamExt::next(&mut ws_receiver).await {
        match result {
            Ok(Message::Text(text)) => {
                let tx = tx.clone();
                let state = state.clone();
                let multiplex_manager = multiplex_manager.clone();

                // Parse and handle the message
                match serde_json::from_str::<WsClientMessage>(&text) {
                    Ok(client_msg) => {
                        handle_ws_message(client_msg, tx, state, multiplex_manager).await;
                    }
                    Err(e) => {
                        // Try parsing as raw JSON-RPC request for backward compatibility
                        match serde_json::from_str::<JsonRpcRequest>(&text) {
                            Ok(request) => {
                                let mut server = state.server.lock().await;
                                let response = server.handle_request(request).await;
                                let _ = tx.send(WsServerMessage::JsonRpc(response)).await;
                            }
                            Err(_) => {
                                let _ = tx
                                    .send(WsServerMessage::Error {
                                        message: format!("Invalid message format: {}", e),
                                    })
                                    .await;
                            }
                        }
                    }
                }
            }
            Ok(Message::Binary(_)) => {
                // Binary messages not supported
                let _ = tx
                    .send(WsServerMessage::Error {
                        message: "Binary messages not supported".to_string(),
                    })
                    .await;
            }
            Ok(Message::Ping(data)) => {
                // Axum handles ping/pong automatically, but we acknowledge anyway
                debug!("WebSocket ping received: {} bytes", data.len());
            }
            Ok(Message::Pong(_)) => {
                // Pong received, nothing to do
            }
            Ok(Message::Close(_)) => {
                info!("WebSocket close received");
                break;
            }
            Err(e) => {
                warn!("WebSocket error: {}", e);
                break;
            }
        }
    }

    // Clean up
    sender_task.abort();
    cleanup_task.abort();
    let mut manager = multiplex_manager.write().await;
    manager.abort_all();
    info!(
        "WebSocket connection closed ({} multiplexed sessions)",
        manager.session_count()
    );
}

/// Handle a parsed WebSocket client message
async fn handle_ws_message(
    msg: WsClientMessage,
    tx: tokio::sync::mpsc::Sender<WsServerMessage>,
    state: HttpServerState,
    multiplex_manager: Arc<tokio::sync::RwLock<WsMultiplexManager>>,
) {
    match msg {
        WsClientMessage::JsonRpc(request) => {
            let mut server = state.server.lock().await;
            let response = server.handle_request(request).await;
            let _ = tx.send(WsServerMessage::JsonRpc(response)).await;
        }
        WsClientMessage::Subscribe { session_id } => {
            handle_ws_subscribe(session_id, tx, state, multiplex_manager).await;
        }
        WsClientMessage::Unsubscribe { session_id } => {
            handle_ws_unsubscribe(session_id, tx, multiplex_manager).await;
        }
        WsClientMessage::Ping => {
            let _ = tx.send(WsServerMessage::Pong).await;
        }
        WsClientMessage::CreateSession {
            session_id,
            metadata,
        } => {
            let mut manager = multiplex_manager.write().await;
            let created_id = manager.create_session(session_id, metadata.clone());
            info!("Created multiplexed session: {}", created_id);
            let _ = tx
                .send(WsServerMessage::SessionCreated {
                    session_id: created_id,
                    metadata,
                })
                .await;
        }
        WsClientMessage::DestroySession { session_id } => {
            let mut manager = multiplex_manager.write().await;
            if manager.destroy_session(&session_id) {
                info!("Destroyed multiplexed session: {}", session_id);
                let _ = tx
                    .send(WsServerMessage::SessionDestroyed { session_id })
                    .await;
            } else {
                let _ = tx
                    .send(WsServerMessage::SessionDestroyError {
                        session_id,
                        message: "Session not found".to_string(),
                    })
                    .await;
            }
        }
        WsClientMessage::ListSessions => {
            let manager = multiplex_manager.read().await;
            let sessions = manager.list_sessions();
            let _ = tx.send(WsServerMessage::SessionList { sessions }).await;
        }
        WsClientMessage::GetSessionInfo { session_id } => {
            let manager = multiplex_manager.read().await;
            if let Some(session) = manager.get_session(&session_id) {
                let _ = tx.send(WsServerMessage::SessionInfo(session.info())).await;
            } else {
                let _ = tx
                    .send(WsServerMessage::SessionInfoError {
                        session_id,
                        message: "Session not found".to_string(),
                    })
                    .await;
            }
        }
        WsClientMessage::UpdateSessionMetadata {
            session_id,
            metadata,
        } => {
            let mut manager = multiplex_manager.write().await;
            if manager.update_session_metadata(&session_id, metadata.clone()) {
                debug!("Updated metadata for session: {}", session_id);
                let _ = tx
                    .send(WsServerMessage::SessionMetadataUpdated {
                        session_id,
                        metadata,
                    })
                    .await;
            } else {
                let _ = tx
                    .send(WsServerMessage::SessionMetadataUpdateError {
                        session_id,
                        message: "Session not found".to_string(),
                    })
                    .await;
            }
        }
        WsClientMessage::TouchSession { session_id } => {
            let mut manager = multiplex_manager.write().await;
            if manager.touch_session(&session_id) {
                let last_activity = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                debug!("Touched session: {}", session_id);
                let _ = tx
                    .send(WsServerMessage::SessionTouched {
                        session_id,
                        last_activity,
                    })
                    .await;
            } else {
                let _ = tx
                    .send(WsServerMessage::SessionTouchError {
                        session_id,
                        message: "Session not found".to_string(),
                    })
                    .await;
            }
        }
        WsClientMessage::SessionScopedJsonRpc {
            session_id,
            request,
        } => {
            // First check if the session exists and update its activity
            let session_exists = {
                let mut manager = multiplex_manager.write().await;
                if let Some(session) = manager.get_session_mut(&session_id) {
                    session.increment_request_count();
                    true
                } else {
                    false
                }
            };

            if session_exists {
                // Process the JSON-RPC request
                let mut server = state.server.lock().await;
                let response = server.handle_request(request).await;
                let _ = tx
                    .send(WsServerMessage::SessionScopedJsonRpc {
                        session_id,
                        response,
                    })
                    .await;
            } else {
                // Session not found
                let _ = tx
                    .send(WsServerMessage::SessionScopedJsonRpcError {
                        session_id,
                        request_id: request.id,
                        message: "Session not found".to_string(),
                    })
                    .await;
            }
        }
        WsClientMessage::BatchSessionScopedJsonRpc {
            session_id,
            requests,
        } => {
            // First check if the session exists and update its activity
            let session_exists = {
                let mut manager = multiplex_manager.write().await;
                if let Some(session) = manager.get_session_mut(&session_id) {
                    // Increment request count by the number of requests in the batch
                    for _ in 0..requests.len() {
                        session.increment_request_count();
                    }
                    true
                } else {
                    false
                }
            };

            if session_exists {
                // Process all JSON-RPC requests sequentially
                let mut server = state.server.lock().await;
                let mut responses = Vec::with_capacity(requests.len());
                for request in requests {
                    let response = server.handle_request(request).await;
                    responses.push(response);
                }
                let _ = tx
                    .send(WsServerMessage::BatchSessionScopedJsonRpc {
                        session_id,
                        responses,
                    })
                    .await;
            } else {
                // Session not found
                let _ = tx
                    .send(WsServerMessage::BatchSessionScopedJsonRpcError {
                        session_id,
                        message: "Session not found".to_string(),
                    })
                    .await;
            }
        }
        WsClientMessage::GetConnectionMetrics => {
            let manager = multiplex_manager.read().await;
            let metrics = manager.connection_metrics();
            let connection_id = manager.connection_id().to_string();
            let _ = tx
                .send(WsServerMessage::ConnectionMetricsResponse {
                    connection_id,
                    metrics,
                })
                .await;
        }
        WsClientMessage::GenerateRecoveryToken => {
            let manager = multiplex_manager.read().await;
            match manager.generate_recovery_token() {
                Some(token) => {
                    let _ = tx
                        .send(WsServerMessage::RecoveryTokenGenerated { token })
                        .await;
                }
                None => {
                    let message = if !manager.recovery_config().enabled {
                        "Session recovery is disabled".to_string()
                    } else {
                        "No sessions to recover".to_string()
                    };
                    let _ = tx
                        .send(WsServerMessage::RecoveryTokenUnavailable { message })
                        .await;
                }
            }
        }
        WsClientMessage::RecoverSessions { token } => {
            let original_connection_id = token.connection_id.clone();
            let mut manager = multiplex_manager.write().await;
            match manager.recover_sessions(&token) {
                Ok(recovered_sessions) => {
                    let _ = tx
                        .send(WsServerMessage::SessionsRecovered {
                            recovered_sessions,
                            original_connection_id,
                        })
                        .await;
                }
                Err(message) => {
                    let _ = tx
                        .send(WsServerMessage::SessionRecoveryError { message })
                        .await;
                }
            }
        }
    }
}

/// Handle WebSocket subscribe request
async fn handle_ws_subscribe(
    session_id: String,
    tx: tokio::sync::mpsc::Sender<WsServerMessage>,
    state: HttpServerState,
    multiplex_manager: Arc<tokio::sync::RwLock<WsMultiplexManager>>,
) {
    // Check if already subscribed (in the default session for backward compatibility)
    {
        let manager = multiplex_manager.read().await;
        let default_session = manager.get_session(
            manager
                .default_session
                .as_deref()
                .unwrap_or("__nonexistent__"),
        );
        if let Some(session) = default_session {
            if session.subscriptions.contains_key(&session_id) {
                let _ = tx
                    .send(WsServerMessage::Subscribed {
                        session_id: session_id.clone(),
                    })
                    .await;
                return;
            }
        }
    }

    // Find the verification session
    let session = state.session_manager.get_session(&session_id).await;
    match session {
        Some(session) => {
            let receiver = {
                let s = session.lock().await;
                s.subscribe()
            };

            // Spawn task to forward events
            let tx_clone = tx.clone();
            let session_id_clone = session_id.clone();
            let handle = tokio::spawn(async move {
                let mut stream = BroadcastStream::new(receiver);
                while let Some(result) = stream.next().await {
                    if let Ok(event) = result {
                        if tx_clone
                            .send(WsServerMessage::Event {
                                session_id: session_id_clone.clone(),
                                event,
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            });

            // Store the subscription in the default session
            {
                let mut manager = multiplex_manager.write().await;
                let default_session = manager.get_or_create_default_session();
                default_session
                    .subscriptions
                    .insert(session_id.clone(), handle);
            }

            let _ = tx.send(WsServerMessage::Subscribed { session_id }).await;
        }
        None => {
            let _ = tx
                .send(WsServerMessage::SubscriptionError {
                    session_id,
                    message: "Session not found".to_string(),
                })
                .await;
        }
    }
}

/// Handle WebSocket unsubscribe request
async fn handle_ws_unsubscribe(
    session_id: String,
    tx: tokio::sync::mpsc::Sender<WsServerMessage>,
    multiplex_manager: Arc<tokio::sync::RwLock<WsMultiplexManager>>,
) {
    let mut manager = multiplex_manager.write().await;
    // Search all multiplexed sessions for this subscription
    for mux_session in manager.sessions.values_mut() {
        if let Some(handle) = mux_session.subscriptions.remove(&session_id) {
            handle.abort();
            let _ = tx
                .send(WsServerMessage::Unsubscribed {
                    session_id: session_id.clone(),
                })
                .await;
            return;
        }
    }

    // Not found in any session
    let _ = tx
        .send(WsServerMessage::SubscriptionError {
            session_id,
            message: "Not subscribed to this session".to_string(),
        })
        .await;
}

/// WebSocket transport for MCP
///
/// Provides a WebSocket endpoint for bidirectional JSON-RPC communication.
/// Supports:
/// - JSON-RPC requests and responses
/// - Event subscriptions for streaming verification
/// - Keep-alive pings
/// - Optional token-based authentication
/// - Optional rate limiting
/// - Optional request logging
/// - Optional metrics collection
pub struct WebSocketTransport {
    /// Address to bind to
    pub bind_addr: String,
    /// Optional API token for authentication
    pub api_token: Option<String>,
    /// Optional rate limit configuration
    pub rate_limit_config: Option<RateLimitConfig>,
    /// Optional request logging configuration
    pub log_config: Option<LogConfig>,
    /// Optional metrics configuration
    pub metrics_config: Option<MetricsConfig>,
}

impl WebSocketTransport {
    /// Create a new WebSocket transport
    pub fn new(bind_addr: impl Into<String>) -> Self {
        Self {
            bind_addr: bind_addr.into(),
            api_token: None,
            rate_limit_config: None,
            log_config: None,
            metrics_config: None,
        }
    }

    /// Create a new WebSocket transport with authentication
    pub fn with_auth(bind_addr: impl Into<String>, api_token: Option<String>) -> Self {
        Self {
            bind_addr: bind_addr.into(),
            api_token,
            rate_limit_config: None,
            log_config: None,
            metrics_config: None,
        }
    }

    /// Create a new WebSocket transport with rate limiting
    pub fn with_rate_limit(
        bind_addr: impl Into<String>,
        rate_limit_config: RateLimitConfig,
    ) -> Self {
        Self {
            bind_addr: bind_addr.into(),
            api_token: None,
            rate_limit_config: Some(rate_limit_config),
            log_config: None,
            metrics_config: None,
        }
    }

    /// Create a new WebSocket transport with authentication and rate limiting
    pub fn with_auth_and_rate_limit(
        bind_addr: impl Into<String>,
        api_token: Option<String>,
        rate_limit_config: Option<RateLimitConfig>,
    ) -> Self {
        Self {
            bind_addr: bind_addr.into(),
            api_token,
            rate_limit_config,
            log_config: None,
            metrics_config: None,
        }
    }

    /// Create a fully configured WebSocket transport
    pub fn with_all_options(
        bind_addr: impl Into<String>,
        api_token: Option<String>,
        rate_limit_config: Option<RateLimitConfig>,
        log_config: Option<LogConfig>,
        metrics_config: Option<MetricsConfig>,
    ) -> Self {
        Self {
            bind_addr: bind_addr.into(),
            api_token,
            rate_limit_config,
            log_config,
            metrics_config,
        }
    }

    /// Run the WebSocket transport asynchronously
    ///
    /// This starts an axum HTTP server with WebSocket upgrade support at GET /ws
    pub async fn run_async(&self, server: McpServer) -> Result<(), McpError> {
        let session_manager = server.tools_session_manager();

        // Create rate limiter if configured
        let rate_limiter = self.rate_limit_config.clone().map(RateLimiter::new);

        // Start rate limiter cleanup task if rate limiting is enabled
        let _cleanup_handle = rate_limiter
            .as_ref()
            .map(|limiter| RateLimiter::start_cleanup_task(Arc::clone(limiter)));

        // Create request logger if configured
        let request_logger = self.log_config.clone().map(RequestLogger::new);

        // Create metrics collector if configured
        let metrics = self
            .metrics_config
            .clone()
            .filter(|c| c.enabled)
            .map(McpMetrics::new);

        let state = HttpServerState {
            server: Arc::new(Mutex::new(server)),
            session_manager,
            api_token: self.api_token.clone(),
            rate_limiter: rate_limiter.clone(),
            request_logger: request_logger.clone(),
            metrics: metrics.clone(),
        };

        // Configure CORS
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);

        // Protected routes (require auth and rate limiting if configured)
        // WebSocket authentication is checked via query param ?token=...
        let protected_routes = Router::new()
            .route("/ws", get(handle_websocket_upgrade))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                auth_middleware,
            ))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                rate_limit_middleware_fn,
            ))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                request_logging_middleware,
            ));

        // Public routes (no auth required, but still rate limited)
        let public_routes = Router::new()
            .route("/health", get(handle_health))
            .route("/", get(handle_ws_info))
            .route("/metrics", get(handle_metrics))
            .route("/metrics/json", get(handle_metrics_json))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                rate_limit_middleware_fn,
            ))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                request_logging_middleware,
            ));

        let app = Router::new()
            .merge(protected_routes)
            .merge(public_routes)
            .with_state(state)
            .layer(cors);

        let addr: SocketAddr = self
            .bind_addr
            .parse()
            .map_err(|e| McpError::InternalError(format!("Invalid bind address: {}", e)))?;

        // Log startup info
        let auth_enabled = self.api_token.is_some();
        let rate_limit_enabled = self.rate_limit_config.as_ref().is_some_and(|c| c.enabled);
        let logging_enabled = self.log_config.as_ref().is_some_and(|c| c.enabled);
        let metrics_enabled = self.metrics_config.as_ref().is_some_and(|c| c.enabled);

        // Build features string for startup message
        let mut features = Vec::new();
        if auth_enabled {
            features.push("auth");
        }
        if rate_limit_enabled {
            features.push("rate-limit");
        }
        if logging_enabled {
            features.push("logging");
        }
        if metrics_enabled {
            features.push("metrics");
        }

        if features.is_empty() {
            info!("Starting MCP WebSocket server on ws://{}/ws", addr);
        } else {
            info!(
                "Starting MCP WebSocket server on ws://{}/ws ({})",
                addr,
                features.join(", ")
            );
        }

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| McpError::InternalError(format!("Failed to bind: {}", e)))?;

        axum::serve(listener, app)
            .await
            .map_err(|e| McpError::InternalError(format!("Server error: {}", e)))?;

        Ok(())
    }
}

impl Default for WebSocketTransport {
    fn default() -> Self {
        Self::new("127.0.0.1:3002")
    }
}

/// Info endpoint for WebSocket server
pub async fn handle_ws_info() -> impl IntoResponse {
    Json(serde_json::json!({
        "name": "dashprove-mcp",
        "version": env!("CARGO_PKG_VERSION"),
        "protocol": "MCP",
        "transport": "WebSocket",
        "websocket_endpoint": "ws://host:port/ws",
        "features": {
            "multiplexing": "Multiple independent sessions per WebSocket connection",
            "subscriptions": "Real-time verification event streaming",
            "json_rpc": "Full JSON-RPC 2.0 support",
            "session_timeout": "Automatic session expiration after idle period",
            "metadata_updates": "Runtime session metadata updates"
        },
        "message_types": {
            "client": {
                "json_rpc": "{ \"type\": \"json_rpc\", \"jsonrpc\": \"2.0\", \"method\": \"...\", ... }",
                "subscribe": "{ \"type\": \"subscribe\", \"session_id\": \"...\" }",
                "unsubscribe": "{ \"type\": \"unsubscribe\", \"session_id\": \"...\" }",
                "ping": "{ \"type\": \"ping\" }",
                "create_session": "{ \"type\": \"create_session\", \"session_id\": \"...\", \"metadata\": {...} }",
                "destroy_session": "{ \"type\": \"destroy_session\", \"session_id\": \"...\" }",
                "list_sessions": "{ \"type\": \"list_sessions\" }",
                "get_session_info": "{ \"type\": \"get_session_info\", \"session_id\": \"...\" }",
                "update_session_metadata": "{ \"type\": \"update_session_metadata\", \"session_id\": \"...\", \"metadata\": {...} }",
                "touch_session": "{ \"type\": \"touch_session\", \"session_id\": \"...\" }"
            },
            "server": {
                "json_rpc": "{ \"type\": \"json_rpc\", \"jsonrpc\": \"2.0\", ... }",
                "event": "{ \"type\": \"event\", \"session_id\": \"...\", \"event\": {...} }",
                "subscribed": "{ \"type\": \"subscribed\", \"session_id\": \"...\" }",
                "unsubscribed": "{ \"type\": \"unsubscribed\", \"session_id\": \"...\" }",
                "subscription_error": "{ \"type\": \"subscription_error\", \"session_id\": \"...\", \"message\": \"...\" }",
                "pong": "{ \"type\": \"pong\" }",
                "error": "{ \"type\": \"error\", \"message\": \"...\" }",
                "session_created": "{ \"type\": \"session_created\", \"session_id\": \"...\", \"metadata\": {...} }",
                "session_destroyed": "{ \"type\": \"session_destroyed\", \"session_id\": \"...\" }",
                "session_destroy_error": "{ \"type\": \"session_destroy_error\", \"session_id\": \"...\", \"message\": \"...\" }",
                "session_list": "{ \"type\": \"session_list\", \"sessions\": [...] }",
                "session_info": "{ \"type\": \"session_info\", \"session_id\": \"...\", ... }",
                "session_info_error": "{ \"type\": \"session_info_error\", \"session_id\": \"...\", \"message\": \"...\" }",
                "session_metadata_updated": "{ \"type\": \"session_metadata_updated\", \"session_id\": \"...\", \"metadata\": {...} }",
                "session_metadata_update_error": "{ \"type\": \"session_metadata_update_error\", \"session_id\": \"...\", \"message\": \"...\" }",
                "session_touched": "{ \"type\": \"session_touched\", \"session_id\": \"...\", \"last_activity\": <unix_ms> }",
                "session_touch_error": "{ \"type\": \"session_touch_error\", \"session_id\": \"...\", \"message\": \"...\" }",
                "session_expired": "{ \"type\": \"session_expired\", \"session_id\": \"...\", \"idle_duration_secs\": <secs> }"
            }
        },
        "multiplexing": {
            "description": "Create multiple independent sessions within a single WebSocket connection",
            "use_cases": [
                "Isolate different verification workflows",
                "Track metrics per logical session",
                "Manage subscriptions per session"
            ],
            "session_info_fields": {
                "session_id": "Unique session identifier",
                "metadata": "Client-provided metadata (optional)",
                "active_subscriptions": "Number of active event subscriptions",
                "created_at": "Unix timestamp (ms) when session was created",
                "request_count": "Number of requests processed in this session",
                "last_activity": "Unix timestamp (ms) of last activity",
                "is_expired": "Whether session has exceeded idle timeout"
            },
            "session_timeout": {
                "description": "Sessions automatically expire after period of inactivity",
                "default_idle_timeout": "30 minutes",
                "keep_alive": "Use touch_session or any request to keep session alive"
            }
        },
        "notes": [
            "All messages are JSON text frames",
            "Raw JSON-RPC requests (without type wrapper) are also accepted for backward compatibility",
            "Subscribe to verification sessions to receive real-time verification events",
            "Use ping/pong for connection keep-alive",
            "Multiplexed sessions are optional - connection works without explicit session creation",
            "A default session is auto-created when needed for backward compatibility",
            "Sessions expire after 30 minutes of inactivity by default",
            "Use touch_session to explicitly keep a session alive"
        ]
    }))
}
